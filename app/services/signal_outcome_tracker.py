import os
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class _PendingSignalsDict(dict):
    """
    Backward-compatibility proxy dict for pending_signals.
    When pop() or clear() is called, it synchronizes deletions with open_signals and persists to disk.
    """
    def __init__(self, tracker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracker = tracker

    def pop(self, k, default=None):
        val = super().pop(k, default)
        if isinstance(val, dict):
            sym = val.get("symbol", "").upper()
            if sym in self._tracker.open_signals:
                self._tracker.open_signals.pop(sym, None)
                self._tracker._save_to_disk()
        else:
            for s, sig in list(self._tracker.open_signals.items()):
                if sig.get("signal_id") == k or s == str(k).upper():
                    self._tracker.open_signals.pop(s, None)
                    self._tracker._save_to_disk()
        return val

    def clear(self):
        super().clear()
        self._tracker.open_signals.clear()
        self._tracker._save_to_disk()

class SignalOutcomeTracker:
    """
    Durable Signal Outcome Ledger & Open Signals Manager (Briefs 14, 17, 18).
    Maintains the rule: A symbol may have at most one unresolved published signal at a time.
    
    Structure:
      open_signals[symbol] = {
          "signal_id": signal_id,
          "entry": entry,
          "stop_loss": stop_loss,
          "take_profit_1": take_profit_1,
          "published_at": published_at,
          "bars_elapsed": bars_elapsed
      }

    Resolution conditions (ONLY these 3 remove an entry):
      - bar low touches stop -> SL
      - bar high touches target -> TP1
      - 96 M15 bars pass -> TIMEOUT
      - Both inside same bar resolves as SL (pessimistic).

    Nothing else removes an entry. Not a restart, not a redeploy, not a bridge reconnect.
    """

    def __init__(self, data_dir: Optional[str] = None, open_signals_file: Optional[str] = None, outcomes_file: Optional[str] = None, manual_outcomes_file: Optional[str] = None):
        self.data_dir = data_dir or self._find_data_dir()
        os.makedirs(self.data_dir, exist_ok=True)
        self.outcomes_file = outcomes_file or os.path.join(self.data_dir, "signal_outcomes.jsonl")
        self.manual_outcomes_file = manual_outcomes_file or os.path.join(self.data_dir, "manual_outcomes.jsonl")
        self.open_signals_file = open_signals_file or os.path.join(self.data_dir, "open_signals.json")
        self.legacy_pending_file = os.path.join(self.data_dir, "pending_signal_evaluations.json")
        
        self.open_signals: Dict[str, Dict[str, Any]] = {}
        self._last_loaded_mtime: float = 0.0
        self._load_from_disk()
        try:
            self.evaluate_pending_timeouts()
        except Exception as e:
            logger.debug(f"Initial timeout sweep error: {e}")

    def _find_data_dir(self) -> str:
        candidates = [
            "/app/data",
            os.path.join(os.path.dirname(__file__), "..", "..", "data"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "vps_backend", "data")
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[1]

    def _load_from_disk(self):
        """Loads open signals from disk. Survives restarts and redeploys without wiping."""
        target_file = None
        if os.path.exists(self.open_signals_file):
            target_file = self.open_signals_file
        elif os.path.exists(self.legacy_pending_file):
            target_file = self.legacy_pending_file

        if not target_file:
            self.open_signals = {}
            self._last_loaded_mtime = 0.0
            return

        try:
            mtime = os.path.getmtime(target_file)
            with open(target_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            parsed: Dict[str, Dict[str, Any]] = {}
            if isinstance(raw_data, dict):
                for k, v in raw_data.items():
                    if not isinstance(v, dict):
                        continue
                    sym = v.get("symbol") or (k if len(k) <= 8 and not k.startswith("snap_") else "")
                    sym = sym.upper()
                    if not sym:
                        continue

                    entry_val = float(v.get("entry") or v.get("entry_price") or 0.0)
                    sl_val = float(v.get("stop_loss") or v.get("sl_price") or 0.0)
                    tp1_val = float(v.get("take_profit_1") or v.get("tp1_price") or 0.0)
                    tp2_val = float(v.get("take_profit_2") or v.get("tp2_price") or 0.0)
                    tp3_val = float(v.get("take_profit_3") or v.get("tp3_price") or 0.0)
                    bars_val = int(v.get("bars_elapsed") if v.get("bars_elapsed") is not None else v.get("bars_monitored", 0))

                    if entry_val <= 0 or sl_val <= 0 or tp1_val <= 0:
                        continue

                    sig_obj = {
                        "signal_id": v.get("signal_id") or k,
                        "symbol": sym,
                        "direction": str(v.get("direction") or "BUY").upper(),
                        "entry": entry_val,
                        "entry_price": entry_val,
                        "stop_loss": sl_val,
                        "sl_price": sl_val,
                        "take_profit_1": tp1_val,
                        "tp1_price": tp1_val,
                        "take_profit_2": tp2_val if tp2_val > 0 else None,
                        "tp2_price": tp2_val if tp2_val > 0 else None,
                        "take_profit_3": tp3_val if tp3_val > 0 else None,
                        "tp3_price": tp3_val if tp3_val > 0 else None,
                        "atr": float(v.get("atr", 0.0)),
                        "published_at": str(v.get("published_at") or v.get("created_at_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
                        "published_timestamp": float(v.get("published_timestamp") or v.get("created_timestamp") or v.get("timestamp", time.time())),
                        "bars_elapsed": bars_val,
                        "bars_monitored": bars_val,
                        "last_evaluated_bar_time": int(v.get("last_evaluated_bar_time", 0)),
                        "status": "ACTIVE"
                    }
                    parsed[sym] = sig_obj

            self.open_signals = parsed
            self._last_loaded_mtime = mtime
            logger.info(f"💾 [OUTCOME_TRACKER] Loaded {len(self.open_signals)} active open signal(s) from disk ({list(self.open_signals.keys())}).")
        except Exception as e:
            logger.error(f"Failed to load open signals from {target_file}: {e}")

    def _sync_from_disk(self):
        """Re-reads disk state if the file has been modified externally."""
        target_file = self.open_signals_file if os.path.exists(self.open_signals_file) else self.legacy_pending_file
        if os.path.exists(target_file):
            try:
                mtime = os.path.getmtime(target_file)
                if mtime > self._last_loaded_mtime:
                    self._load_from_disk()
            except Exception as e:
                logger.debug(f"Sync check failed: {e}")

    def _save_to_disk(self):
        """Persists open signals to disk atomically."""
        try:
            temp_file = f"{self.open_signals_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.open_signals, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.open_signals_file):
                os.replace(temp_file, self.open_signals_file)
            else:
                os.rename(temp_file, self.open_signals_file)
            self._last_loaded_mtime = os.path.getmtime(self.open_signals_file)

            legacy_dict = {}
            for sym, sig in self.open_signals.items():
                legacy_dict[sig["signal_id"]] = sig
            leg_temp = f"{self.legacy_pending_file}.tmp"
            with open(leg_temp, "w", encoding="utf-8") as f:
                json.dump(legacy_dict, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.legacy_pending_file):
                os.replace(leg_temp, self.legacy_pending_file)
            else:
                os.rename(leg_temp, self.legacy_pending_file)
        except Exception as e:
            logger.error(f"Failed to save open signals to disk: {e}")

    @property
    def pending_signals(self) -> Dict[str, Dict[str, Any]]:
        self._sync_from_disk()
        res = _PendingSignalsDict(self)
        for sym, sig in self.open_signals.items():
            res[sig["signal_id"]] = sig
        return res

    def _save_pending(self):
        self._save_to_disk()

    # -------------------------------------------------------------------------
    # Signal Registration (Brief 18: write before anything can clear it)
    # -------------------------------------------------------------------------
    def register_open_signal(
        self,
        symbol: str,
        signal_id: str,
        entry: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: Optional[float] = None,
        take_profit_3: Optional[float] = None,
        direction: str = "BUY",
        atr: float = 0.0,
        published_at: Optional[str] = None,
        published_timestamp: Optional[float] = None
    ) -> bool:
        """
        Registers a published signal directly under open_signals[symbol].
        Survives restarts, redeploys, and bridge reconnects.
        """
        sym = symbol.upper()
        if not signal_id or entry <= 0 or stop_loss <= 0 or take_profit_1 <= 0:
            logger.warning(f"Cannot register open signal for {sym}: invalid levels (entry={entry}, sl={stop_loss}, tp1={take_profit_1})")
            return False

        now_ts = published_timestamp or time.time()
        pub_at = published_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        self._sync_from_disk()

        sig_obj = {
            "signal_id": signal_id,
            "symbol": sym,
            "direction": str(direction).upper(),
            "entry": float(entry),
            "entry_price": float(entry),
            "stop_loss": float(stop_loss),
            "sl_price": float(stop_loss),
            "take_profit_1": float(take_profit_1),
            "tp1_price": float(take_profit_1),
            "take_profit_2": float(take_profit_2) if take_profit_2 else None,
            "tp2_price": float(take_profit_2) if take_profit_2 else None,
            "take_profit_3": float(take_profit_3) if take_profit_3 else None,
            "tp3_price": float(take_profit_3) if take_profit_3 else None,
            "atr": float(atr),
            "published_at": pub_at,
            "created_at_utc": pub_at,
            "published_timestamp": now_ts,
            "created_timestamp": now_ts,
            "bars_elapsed": 0,
            "bars_monitored": 0,
            "last_evaluated_bar_time": 0,
            "status": "ACTIVE"
        }

        self.open_signals[sym] = sig_obj
        self._save_to_disk()
        logger.info(f"📌 [OPEN_SIGNAL_REGISTERED] open_signals['{sym}'] = {signal_id} (Entry={entry}, SL={stop_loss}, TP1={take_profit_1}). Persisted to disk.")
        return True

    def register_signal(self, card: Dict[str, Any]) -> bool:
        """Adapter for legacy register_signal calls taking a dictionary card."""
        snap_id = card.get("snapshot_id")
        if not snap_id:
            return False

        sym = card.get("symbol", "").upper()
        direction = card.get("decision") or card.get("direction") or "BUY"
        entry = float(card.get("entry_price") or card.get("entry") or card.get("levels", {}).get("entry") or 0.0)
        sl = float(card.get("sl_price") or card.get("sl") or card.get("stop_loss") or card.get("levels", {}).get("stop_loss") or 0.0)
        tp1 = float(card.get("tp1_price") or card.get("tp1") or card.get("take_profit_1") or card.get("levels", {}).get("take_profit_1") or 0.0)
        tp2 = float(card.get("tp2_price") or card.get("tp2") or card.get("take_profit_2") or card.get("levels", {}).get("take_profit_2") or 0.0)
        tp3 = float(card.get("tp3_price") or card.get("tp3") or card.get("take_profit_3") or card.get("levels", {}).get("take_profit_3") or 0.0)
        atr = float(card.get("atr") or 0.0)
        pub_at = card.get("created_at_utc") or card.get("time_utc")
        pub_ts = card.get("timestamp") or time.time()

        return self.register_open_signal(
            symbol=sym,
            signal_id=snap_id,
            entry=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2 if tp2 > 0 else None,
            take_profit_3=tp3 if tp3 > 0 else None,
            direction=direction,
            atr=atr,
            published_at=pub_at,
            published_timestamp=pub_ts
        )

    # -------------------------------------------------------------------------
    # Retrieval and Status Queries
    # -------------------------------------------------------------------------
    def get_open_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Returns active open signal for symbol, reloading from disk if updated."""
        self._sync_from_disk()
        return self.open_signals.get(symbol.upper())

    def has_open_signal(self, symbol: str) -> bool:
        return self.get_open_signal(symbol) is not None

    def evaluate_pending_timeouts(self, now_epoch: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Brief 24 Section 4 & Brief 26 Section 5: Wall-clock timeout defense.
        Sweeps open_signals for any signals that have exceeded 96 M15 bars (24 hours / 86400s)
        even if the market closed over a weekend or incoming candle feed paused.
        Resolves them, frees the symbol, appends outcome, and dispatches to Telegram publisher.
        Fix for 43-hour bug: accurately calculates bars_monitored from elapsed wall-clock time
        so that weekend-frozen bar counts (e.g. 6 or 1) do not corrupt the duration.
        """
        self._sync_from_disk()
        if not self.open_signals:
            return []

        curr_time = now_epoch if now_epoch is not None else time.time()
        resolved: List[Dict[str, Any]] = []

        for sym, sig in list(self.open_signals.items()):
            pub_ts = float(sig.get("published_timestamp") or sig.get("created_timestamp") or 0.0)
            bars_elapsed = int(sig.get("bars_elapsed", 0))
            elapsed_sec = (curr_time - pub_ts) if pub_ts > 0 else 0.0

            if bars_elapsed >= 96 or elapsed_sec >= 86400.0:
                direction = sig.get("direction", "BUY").upper()
                entry = float(sig.get("entry") or sig.get("entry_price") or 0.0)
                sl = float(sig.get("stop_loss") or sig.get("sl_price") or 0.0)
                tp1 = float(sig.get("take_profit_1") or sig.get("tp1_price") or 0.0)

                # Use last known price if available, otherwise entry
                exit_price = float(sig.get("last_known_price") or sig.get("close") or entry)
                if exit_price <= 0 and entry > 0:
                    exit_price = entry

                risk = abs(entry - sl) if abs(entry - sl) > 0 else (0.001 if "JPY" not in sym else 0.1)
                gain = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
                r_mult = round(gain / risk, 2)
                snap_id = sig.get("signal_id", f"snap_{sym}_timeout")

                now_dt = datetime.fromtimestamp(curr_time, tz=timezone.utc)
                resolved_time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

                # Accurate bar calculation for timeout: at least 96 bars for 24h+ trades
                calc_bars = int(round(elapsed_sec / 900.0))
                bars_held = max(bars_elapsed, min(96, calc_bars if calc_bars > 0 else 96))

                record = {
                    "signal_id": snap_id,
                    "symbol": sym,
                    "direction": direction,
                    "entry": entry,
                    "entry_price": entry,
                    "stop_loss": sl,
                    "sl_price": sl,
                    "take_profit_1": tp1,
                    "tp1_price": tp1,
                    "take_profit_2": sig.get("take_profit_2"),
                    "tp2_price": sig.get("tp2_price"),
                    "take_profit_3": sig.get("take_profit_3"),
                    "tp3_price": sig.get("tp3_price"),
                    "exit_price": exit_price,
                    "barrier_hit": "TIMEOUT",
                    "r_multiple": r_mult,
                    "signal_time_utc": sig.get("published_at", ""),
                    "published_utc": sig.get("published_at", ""),
                    "resolved_time_utc": resolved_time_str,
                    "resolved_utc": resolved_time_str,
                    "bars_monitored": bars_held,
                    "bars_held": bars_held,
                    "recorded_at_epoch": curr_time
                }

                self._append_outcome(record)
                self.open_signals.pop(sym, None)
                self._save_to_disk()
                resolved.append(record)

                logger.info(
                    f"⌛ [SIGNAL_TIMEOUT_WALLCLOCK] Signal {snap_id} ({sym} {direction}) exceeded 24-hour limit "
                    f"({elapsed_sec:.0f}s elapsed, {bars_held} bars). "
                    f"Closed at {exit_price} -> R={r_mult:+.2f}. Symbol freed."
                )

        return resolved

    def get_open_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Brief 18 Section 1 & Acceptance:
        Returns all currently active open signals formatted exactly per contract:
        open_signals[symbol] = {signal_id, entry, stop_loss, take_profit_1, published_at, bars_elapsed}
        """
        self._sync_from_disk()
        self.evaluate_pending_timeouts()
        res = {}
        now_ts = time.time()
        for sym, sig in self.open_signals.items():
            pub_ts = float(sig.get("published_timestamp") or now_ts)
            dur_s = max(0, int(now_ts - pub_ts))
            h = dur_s // 3600
            m = (dur_s % 3600) // 60
            dur_str = f"{h}h{m:02d}m" if h > 0 else f"{m}m"

            res[sym] = {
                "signal_id": sig.get("signal_id"),
                "symbol": sym,
                "direction": sig.get("direction", "BUY"),
                "entry": sig.get("entry"),
                "stop_loss": sig.get("stop_loss"),
                "take_profit_1": sig.get("take_profit_1"),
                "take_profit_2": sig.get("take_profit_2"),
                "take_profit_3": sig.get("take_profit_3"),
                "published_at": sig.get("published_at"),
                "bars_elapsed": sig.get("bars_elapsed", 0),
                "open_duration": dur_str
            }
        return res

    def get_open_signals_summary(self) -> Dict[str, Any]:
        """Backward compatible alias for status endpoint."""
        return self.get_open_signals()

    # -------------------------------------------------------------------------
    # Resolution Engine (Incoming Bar / Price Evaluation)
    # -------------------------------------------------------------------------
    def check_price_update(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
        is_bar: bool = False,
        bar_time: Optional[int] = None,
        timestamp_utc: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates active open signals against new tick or candle data.
        Returns a list of outcome records resolved during this tick/bar (or empty list).

        Resolution Rules:
          - BUY:
              * if low <= stop_loss -> SL (-1.0 R)
              * if high >= take_profit_1 -> TP1 (+ realized R)
              * if bars_elapsed >= 96 or elapsed_sec >= 86400 (24h) -> TIMEOUT (realized R)
          - SELL:
              * if high >= stop_loss -> SL (-1.0 R)
              * if low <= take_profit_1 -> TP1 (+ realized R)
              * if bars_elapsed >= 96 or elapsed_sec >= 86400 (24h) -> TIMEOUT (realized R)
        """
        sym = symbol.upper()
        self._sync_from_disk()

        sig = self.open_signals.get(sym)
        if not sig:
            return []

        resolved: List[Dict[str, Any]] = []
        now_dt = datetime.now(timezone.utc)
        resolved_time_str = timestamp_utc or now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        if is_bar:
            if bar_time is not None and bar_time > 0:
                last_b = sig.get("last_evaluated_bar_time", 0)
                if last_b != bar_time:
                    sig["bars_elapsed"] = sig.get("bars_elapsed", 0) + 1
                    sig["bars_monitored"] = sig["bars_elapsed"]
                    sig["last_evaluated_bar_time"] = bar_time
            else:
                sig["bars_elapsed"] = sig.get("bars_elapsed", 0) + 1
                sig["bars_monitored"] = sig["bars_elapsed"]

        # Keep track of last known close price
        if close > 0:
            sig["last_known_price"] = close

        direction = sig.get("direction", "BUY").upper()
        entry = float(sig["entry"])
        sl = float(sig["stop_loss"])
        tp1 = float(sig["take_profit_1"])
        tp2 = float(sig.get("take_profit_2") or sig.get("tp2_price") or 0.0)
        tp3 = float(sig.get("take_profit_3") or sig.get("tp3_price") or 0.0)
        risk = abs(entry - sl) if abs(entry - sl) > 0 else 0.001

        barrier_hit = None
        exit_price = None
        r_mult = 0.0

        pub_ts = float(sig.get("published_timestamp") or sig.get("created_timestamp") or 0.0)
        elapsed_sec = (time.time() - pub_ts) if pub_ts > 0 else 0.0
        # 96 M15 bars = 24 hours = 86400 seconds wall-clock
        is_timed_out = (sig.get("bars_elapsed", 0) >= 96) or (elapsed_sec >= 86400.0)

        if direction == "BUY":
            if low <= sl:
                barrier_hit = "SL"
                exit_price = sl
                r_mult = -1.0
            elif tp3 > 0 and high >= tp3:
                barrier_hit = "TP3"
                exit_price = tp3
                r_mult = round((tp3 - entry) / risk, 2)
            elif tp2 > 0 and high >= tp2:
                barrier_hit = "TP2"
                exit_price = tp2
                r_mult = round((tp2 - entry) / risk, 2)
            elif high >= tp1:
                barrier_hit = "TP1"
                exit_price = tp1
                r_mult = round((tp1 - entry) / risk, 2) if abs(tp1 - entry) > 0 else 2.0
            elif is_timed_out:
                barrier_hit = "TIMEOUT"
                exit_price = close if (close is not None and close > 0) else entry
                gain = exit_price - entry
                r_mult = round(gain / risk, 2)

        elif direction == "SELL":
            if high >= sl:
                barrier_hit = "SL"
                exit_price = sl
                r_mult = -1.0
            elif tp3 > 0 and low <= tp3:
                barrier_hit = "TP3"
                exit_price = tp3
                r_mult = round((entry - tp3) / risk, 2)
            elif tp2 > 0 and low <= tp2:
                barrier_hit = "TP2"
                exit_price = tp2
                r_mult = round((entry - tp2) / risk, 2)
            elif low <= tp1:
                barrier_hit = "TP1"
                exit_price = tp1
                r_mult = round((entry - tp1) / risk, 2) if abs(entry - tp1) > 0 else 2.0
            elif is_timed_out:
                barrier_hit = "TIMEOUT"
                exit_price = close if (close is not None and close > 0) else entry
                gain = entry - exit_price
                r_mult = round(gain / risk, 2)

        if barrier_hit and exit_price is not None:
            snap_id = sig.get("signal_id", "unknown")
            calc_bars = int(round(elapsed_sec / 900.0))
            bars_held = max(sig.get("bars_elapsed", 1), min(96, calc_bars if calc_bars > 0 else 96)) if barrier_hit == "TIMEOUT" else sig.get("bars_elapsed", 1)

            record = {
                "signal_id": snap_id,
                "symbol": sym,
                "direction": direction,
                "entry": entry,
                "entry_price": entry,
                "stop_loss": sl,
                "sl_price": sl,
                "take_profit_1": tp1,
                "tp1_price": tp1,
                "take_profit_2": sig.get("take_profit_2"),
                "tp2_price": sig.get("tp2_price"),
                "take_profit_3": sig.get("take_profit_3"),
                "tp3_price": sig.get("tp3_price"),
                "exit_price": exit_price,
                "barrier_hit": barrier_hit,
                "r_multiple": r_mult,
                "signal_time_utc": sig.get("published_at", ""),
                "published_utc": sig.get("published_at", ""),
                "resolved_time_utc": resolved_time_str,
                "resolved_utc": resolved_time_str,
                "bars_monitored": bars_held,
                "bars_held": bars_held,
                "recorded_at_epoch": time.time()
            }
            self._append_outcome(record)
            self.open_signals.pop(sym, None)
            self._save_to_disk()
            resolved.append(record)

            if barrier_hit == "TIMEOUT":
                logger.info(
                    f"⌛ [SIGNAL_TIMEOUT] Signal {snap_id} ({sym} {direction}) reached 96 M15 bars timeout. "
                    f"Closed at {exit_price} -> R={r_mult:+.2f}. Symbol freed."
                )
            else:
                logger.info(
                    f"🎯 [SIGNAL_OUTCOME_RECORDED] Signal {snap_id} ({sym} {direction}): "
                    f"Hit {barrier_hit} at {exit_price} -> R={r_mult:+.1f} (Bars held: {record['bars_monitored']}). Symbol freed."
                )
        elif is_bar:
            self._save_to_disk()

        return resolved

    def _notify_outcome(self, record: Dict[str, Any]):
        """Brief 23: Dispatches signal outcome to Telegram result publisher"""
        try:
            from app.services.telegram_bot import telegram_bot
            telegram_bot.dispatch_signal_result(record)
        except Exception as e:
            logger.error(f"Error notifying telegram_bot of outcome: {e}")

    def _append_outcome(self, record: Dict[str, Any], is_manual: bool = False):
        """
        Appends a resolved outcome record to durable JSONL.
        Agent Brief 26 Standing Rule & Isolation:
        Manual outcomes go ONLY to manual_outcomes.jsonl and NEVER dispatch to Telegram.
        """
        line = json.dumps(record, ensure_ascii=False) + "\n"
        if is_manual or record.get("manual", False):
            target_files = [self.manual_outcomes_file]
            reports_dir = Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports"
            if reports_dir.exists():
                target_files.append(str(reports_dir / "manual_outcomes.jsonl"))

            for fpath in target_files:
                try:
                    os.makedirs(os.path.dirname(fpath), exist_ok=True)
                    with open(fpath, "a", encoding="utf-8") as f:
                        f.write(line)
                except Exception as e:
                    logger.error(f"Failed to append manual outcome to {fpath}: {e}")
            # CRITICAL: Manual outcomes NEVER dispatch to Telegram!
            return

        target_files = [self.outcomes_file]
        reports_dir = Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports"
        if reports_dir.exists():
            target_files.append(str(reports_dir / "signal_outcomes.jsonl"))

        for fpath in target_files:
            try:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                logger.error(f"Failed to append outcome to {fpath}: {e}")

        # Hook: Publish result when a model signal closes (Brief 23)
        self._notify_outcome(record)

    def record_manual_outcome(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        barrier_hit: str,
        r_multiple: float,
        signal_time_utc: str,
        resolved_time_utc: Optional[str] = None
    ) -> Dict[str, Any]:
        """Manually records an outcome for a signal and frees the symbol into manual_outcomes.jsonl."""
        sym = symbol.upper()
        res_time = resolved_time_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        record = {
            "signal_id": signal_id,
            "symbol": sym,
            "direction": direction.upper(),
            "entry": float(entry_price),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "barrier_hit": barrier_hit.upper(),
            "r_multiple": round(float(r_multiple), 2),
            "signal_time_utc": signal_time_utc,
            "published_utc": signal_time_utc,
            "resolved_time_utc": res_time,
            "resolved_utc": res_time,
            "bars_monitored": 1,
            "recorded_at_epoch": time.time(),
            "manual": True
        }
        self._append_outcome(record, is_manual=True)
        self.open_signals.pop(sym, None)
        self._save_to_disk()
        return record

    def get_manual_outcomes(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Reads resolved manual outcomes from manual_outcomes.jsonl, optionally filtered by symbol."""
        outcomes = []
        if os.path.exists(self.manual_outcomes_file):
            try:
                with open(self.manual_outcomes_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            if symbol:
                                if rec.get("symbol", "").upper() == symbol.upper():
                                    outcomes.append(rec)
                            else:
                                outcomes.append(rec)
            except Exception as e:
                logger.error(f"Error reading manual outcomes file: {e}")
        return outcomes[-limit:]

    def get_outcomes(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Reads resolved outcomes from disk, optionally filtered by symbol."""
        outcomes = []
        if os.path.exists(self.outcomes_file):
            try:
                with open(self.outcomes_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            if symbol:
                                if rec.get("symbol", "").upper() == symbol.upper():
                                    outcomes.append(rec)
                            else:
                                outcomes.append(rec)
            except Exception as e:
                logger.error(f"Error reading outcomes file: {e}")
        return outcomes[-limit:]

    def get_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        all_outcomes = self.get_outcomes(symbol=symbol, limit=100000)
        total = len(all_outcomes)
        if total == 0:
            return {
                "symbol": symbol.upper() if symbol else "ALL",
                "completed_trades": 0,
                "wins": 0,
                "losses": 0,
                "breakeven": 0,
                "win_rate": 0.0,
                "total_r": 0.0,
                "expectancy_r": 0.0
            }

        wins = sum(1 for o in all_outcomes if float(o.get("r_multiple", 0.0)) > 0)
        losses = sum(1 for o in all_outcomes if float(o.get("r_multiple", 0.0)) < 0)
        be = sum(1 for o in all_outcomes if float(o.get("r_multiple", 0.0)) == 0)
        total_r = sum(float(o.get("r_multiple", 0.0)) for o in all_outcomes)
        win_rate = round(wins / total, 3)
        exp_r = round(total_r / total, 4)

        return {
            "symbol": symbol.upper() if symbol else "ALL",
            "completed_trades": total,
            "wins": wins,
            "losses": losses,
            "breakeven": be,
            "win_rate": win_rate,
            "total_r": round(total_r, 2),
            "expectancy_r": exp_r
        }

    def resolve_signal(
        self,
        signal_id: str,
        barrier_hit: str,
        exit_price: float,
        reason: str = "Manual resolution"
    ) -> Optional[Dict[str, Any]]:
        self._sync_from_disk()
        target_sym = None
        for sym, sig in self.open_signals.items():
            if sig.get("signal_id") == signal_id:
                target_sym = sym
                break
        
        if not target_sym:
            return None

        sig = self.open_signals[target_sym]
        entry = float(sig["entry"])
        sl = float(sig["stop_loss"])
        direction = sig.get("direction", "BUY").upper()
        risk = abs(entry - sl)
        if risk <= 0:
            risk = 0.001

        if barrier_hit == "SL":
            r_mult = -1.0
        elif barrier_hit.startswith("TP"):
            r_mult = round(abs(exit_price - entry) / risk, 2)
        else:
            gain = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
            r_mult = round(gain / risk, 2)

        return self.record_manual_outcome(
            signal_id=signal_id,
            symbol=target_sym,
            direction=direction,
            entry_price=entry,
            exit_price=exit_price,
            barrier_hit=barrier_hit,
            r_multiple=r_mult,
            signal_time_utc=sig.get("published_at", ""),
            resolved_time_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        )

    def get_pending_signals(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        self._sync_from_disk()
        sigs = list(self.open_signals.values())
        if symbol:
            return [s for s in sigs if s.get("symbol", "").upper() == symbol.upper()]
        return sigs


# Singleton instance
signal_outcome_tracker = SignalOutcomeTracker()
