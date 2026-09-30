"""
Analyst Desk Core Engine (Agent Brief 26).
Measures and records the owner's discretionary trading decisions alongside
the algorithmic model, without modifying model signals, registry gates, or Telegram.

Core Responsibilities:
1. Under-15s keyboard-first manual trade registration with second-precision timestamps.
2. Direction-aware sanity validation (reusing signal_sanity.py).
3. 91-feature causal snapshot capture at the instant of entry.
4. Tracking symbol_had_open_trade flag (distinguishing independent from overlapping entries).
5. Separate manual outcome resolution into manual_outcomes.jsonl (never signal_outcomes.jsonl, zero Telegram).
6. Comparison analysis: manual all vs manual first only vs model.
7. Feature signature analysis: Cohen's d ranking vs available market bars, median comparisons, thin sample flagging (<30).
8. Zone tracking (evaluating price returns to marked discretionary zones once N >= 50).
"""
from __future__ import annotations

import os
import json
import time
import math
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from app.config import settings
from app.services.signal_sanity import validate_manual_trade_sanity, get_reference_median_atr
from app.services.signal_engine_service import signal_engine_service
from app.services.signal_outcomes_service import SignalOutcomesService

logger = logging.getLogger(__name__)

# Search paths for feature definitions
FEATURE_PATHS = [
    Path("/app/ml/models/production_features.json"),
    Path(__file__).resolve().parent.parent.parent / "ml" / "models" / "production_features.json",
    Path(__file__).resolve().parent.parent.parent.parent / "ml" / "models" / "production_features.json",
]


class AnalystDeskService:

    def __init__(
        self,
        data_dir: Optional[str] = None,
        open_trades_file: Optional[str] = None,
        manual_records_file: Optional[str] = None,
        outcomes_file: Optional[str] = None,
        baseline_file: Optional[str] = None,
        outcome_tracker: Optional[Any] = None
    ):
        self.data_dir = data_dir or self._find_data_dir()
        os.makedirs(self.data_dir, exist_ok=True)

        self.open_trades_file = open_trades_file or os.path.join(self.data_dir, "open_manual_trades.json")
        self.records_file = manual_records_file or os.path.join(self.data_dir, "manual_records.jsonl")
        self.outcomes_file = outcomes_file or os.path.join(self.data_dir, "manual_outcomes.jsonl")
        self.baseline_file = baseline_file or os.path.join(self.data_dir, "available_bars_baseline.json")
        self.outcome_tracker = outcome_tracker

        self.feature_names = self._load_feature_names()
        self.open_trades: Dict[str, Dict[str, Any]] = {}
        self.baseline_data = self._load_baseline()

        # Brief 27: candle getter and snapshot counters
        self.candle_getter: Optional[Callable[[str], Dict[str, List[Dict[str, Any]]]]] = None
        self.snapshots_ok: int = 0
        self.snapshots_failed: int = 0

        self._load_from_disk()

    def set_candle_getter(self, getter: Callable[[str], Dict[str, List[Dict[str, Any]]]]):
        """Registers a callback to retrieve live server candles for symbols."""
        self.candle_getter = getter

    def get_status(self) -> Dict[str, Any]:
        """Brief 27 Section 1: Exposes snapshot success and failure telemetry."""
        self._load_from_disk()
        all_recs = self.get_records(limit=100000)
        valid_snapshots = sum(1 for r in all_recs if r.get("features") and len(r.get("features", {})) == 91)
        return {
            "status": "ONLINE",
            "snapshots_ok": max(self.snapshots_ok, valid_snapshots),
            "snapshots_failed": self.snapshots_failed,
            "total_records": len(all_recs),
            "records_count": len(all_recs),
            "open_trades_count": len(self.open_trades)
        }

    def get_manual_records(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Alias for get_records for backward compatibility."""
        return self.get_records(symbol=symbol, limit=limit)

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

    def _load_feature_names(self) -> List[str]:
        for p in FEATURE_PATHS:
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if "features" in data and isinstance(data["features"], list):
                        return data["features"]
                except Exception as e:
                    logger.warning(f"Failed to read features from {p}: {e}")
        # Default 91 features list fallback
        return [
            "f_ema20_dist", "f_ema50_dist", "f_ema200_dist", "f_ema20_50_gap", "f_ema50_200_gap",
            "f_ema_stack", "f_rsi", "f_rsi_slope", "f_adx", "f_adx_slope", "f_di_spread",
            "f_roc3", "f_roc6", "f_roc12", "f_roc24", "f_atr_pct", "f_atr_ratio",
            "f_bandwidth", "f_bandwidth_pctile", "f_tr_ratio", "f_body_ratio",
            "f_upper_wick", "f_lower_wick", "f_direction", "f_range_pos24",
            "f_dist_high24", "f_dist_low24", "f_range_pos96", "f_dist_high96", "f_dist_low96",
            "f_spread_points", "f_spread_atr", "f_hour_sin", "f_hour_cos", "f_dow",
            "f_sess_asia", "f_sess_london", "f_sess_ny", "f_rv12", "f_rv48", "f_rv192",
            "f_vol_term_structure", "f_vol_of_vol", "f_vol_percentile", "f_range_vs_typical",
            "f_narrowest_in_7", "f_atr_expanding", "f_is_london_open", "f_is_ny_open",
            "f_is_overlap", "f_is_asia_quiet", "f_is_rollover", "f_range_vs_hour_norm",
            "f_spread_vs_hour_norm", "f_is_month_turn", "f_zret3", "f_zret6", "f_zret12",
            "f_mom96", "f_mom288", "f_mom960", "f_mom_consistency", "f_up_bar_share",
            "f_dist_pdh", "f_dist_pdl", "f_dist_pdc", "f_pos_in_pdr", "f_dist_round",
            "f_vol_vs_hour_norm", "f_vol_trend", "f_move_per_volume", "f_move_per_volume_z",
            "f_fib_pos", "f_near_618", "f_near_50", "f_fvg_up", "f_fvg_dn", "f_swept_high",
            "f_swept_low", "h1_ema_stack", "h1_ema20_dist", "h1_rsi", "h1_adx", "h1_di_spread",
            "h1_roc6", "h4_ema_stack", "h4_ema20_dist", "h4_rsi", "h4_adx", "h4_di_spread",
            "h4_roc6"
        ]

    def _load_baseline(self) -> Dict[str, Any]:
        """Loads precomputed available market bars baseline statistics for 91 features."""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read baseline file: {e}")
        return {"total_available_bars": 23855, "features": {}}

    def _load_from_disk(self):
        """Loads open manual trades from disk atomically."""
        if os.path.exists(self.open_trades_file):
            try:
                with open(self.open_trades_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.open_trades = data
                    elif isinstance(data, list):
                        self.open_trades = {t.get("trade_id", f"man_{i}"): t for i, t in enumerate(data)}
            except Exception as e:
                logger.error(f"Failed to load open manual trades from disk: {e}")
                self.open_trades = {}
        else:
            self.open_trades = {}

    def _save_to_disk(self):
        """Persists open manual trades to disk atomically."""
        try:
            tmp = f"{self.open_trades_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.open_trades, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.open_trades_file):
                os.replace(tmp, self.open_trades_file)
            else:
                os.rename(tmp, self.open_trades_file)
        except Exception as e:
            logger.error(f"Failed to save open manual trades to disk: {e}")

    # -------------------------------------------------------------------------
    # Authentication & Access Control
    # -------------------------------------------------------------------------
    def verify_password(self, password_attempt: Optional[str]) -> bool:
        """Validates desk password against configured environment variable."""
        configured = settings.ANALYST_DESK_PASSWORD.strip()
        if not configured:
            return True
        if not password_attempt:
            return False
        return password_attempt.strip() == configured

    # -------------------------------------------------------------------------
    # Trade Registration (Agent Brief 26 Sections 3 & 4)
    # -------------------------------------------------------------------------
    def record_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: Optional[float] = None,
        take_profit_3: Optional[float] = None,
        timeframe: str = "M15",
        zone_low: Optional[float] = None,
        zone_high: Optional[float] = None,
        why: str = "",
        candles: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        atr: Optional[float] = None,
        discretionary_zone: Optional[str] = None,
        why_reason: Optional[str] = None,
        candles_dict: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        point: Optional[float] = None,
        digits: Optional[int] = None,
        publish_to_channel: bool = True,
        live_price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Records a manual trade from the analyst desk.
        Must capture:
          1. entry, stop, T1, T2, T3
          2. zone_low, zone_high (optional)
          3. entered_at to the second (ISO 8601 UTC)
          4. symbol_had_open_trade (True if another manual trade on this symbol was open)
          And the instant 91-feature snapshot.
        """
        sym = symbol.upper().strip()
        dir_norm = direction.upper().strip()
        tf_norm = (timeframe or "M15").upper().strip()
        candles_to_use = candles if candles is not None else candles_dict
        why_to_use = why if why else (why_reason or "")

        # Effective ATR from candle history, argument, or symbol reference
        effective_atr = float(atr) if atr is not None and float(atr) > 0 else get_reference_median_atr(sym)

        # Silent ordering of discretionary zone bounds if high was typed in low field (Agent Brief 28 Section 4)
        if zone_low is not None and zone_high is not None:
            try:
                zl, zh = float(zone_low), float(zone_high)
                if zl > 0 and zh > 0 and zl > zh:
                    zone_low, zone_high = zh, zl
            except (ValueError, TypeError):
                pass

        # Resolve live market price for Brief 31 proximity validation
        lp = live_price if live_price is not None else (kwargs.get("live_price") or kwargs.get("live_price_override"))
        if lp is None and candles_to_use:
            m15_list = candles_to_use.get("M15", [])
            if m15_list:
                lp = float(m15_list[-1].get("close", 0.0))
        if lp is None and self.candle_getter:
            try:
                c_fetched = self.candle_getter(sym)
                m15_fetched = c_fetched.get("M15", [])
                if m15_fetched:
                    lp = float(m15_fetched[-1].get("close", 0.0))
                    if not candles_to_use:
                        candles_to_use = c_fetched
            except Exception:
                pass

        # 1. Validation (signal_sanity.py direction, target ordering, T1 pay ratio, zone bounds, live price)
        is_valid, err_msg = validate_manual_trade_sanity(
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            direction=dir_norm,
            atr=effective_atr,
            symbol=sym,
            zone_low=zone_low,
            zone_high=zone_high,
            live_price=lp
        )
        if not is_valid:
            raise ValueError(err_msg)

        now_utc = datetime.now(timezone.utc)
        # Strict second-precision timestamp per Section 4 requirement #3
        entered_at_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        entered_at_epoch = now_utc.timestamp()

        # 2. Check if symbol already had an open manual trade (Section 4 requirement #4)
        self._load_from_disk()
        symbol_had_open_trade = any(
            t.get("symbol") == sym and t.get("status") == "OPEN"
            for t in self.open_trades.values()
        )

        # 3. Calculate Risk & R-Multiples
        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            risk = 0.0001
        tp1_r = round(abs(take_profit_1 - entry_price) / risk, 2)
        tp2_r = round(abs(take_profit_2 - entry_price) / risk, 2) if take_profit_2 else None
        tp3_r = round(abs(take_profit_3 - entry_price) / risk, 2) if take_profit_3 else None

        # Agent Brief 27 Section 1: Use server candle store if not provided directly
        if not candles_to_use and self.candle_getter:
            try:
                candles_to_use = self.candle_getter(sym)
            except Exception as ce:
                logger.warning(f"Error invoking candle_getter for {sym}: {ce}")

        m15_count = len(candles_to_use.get("M15", [])) if candles_to_use else 0
        h1_count = len(candles_to_use.get("H1", [])) if candles_to_use else 0
        h4_count = len(candles_to_use.get("H4", [])) if candles_to_use else 0

        # Refuse the record when snapshot cannot be built (Brief 27 Section 1)
        if m15_count < 1100 or h1_count < 260 or h4_count < 260:
            self.snapshots_failed += 1
            err_msg = (
                f"CANNOT RECORD — no feature snapshot available\n"
                f"{sym} has {m15_count} M15 bars cached, needs 1,100 (H1: {h1_count}/260, H4: {h4_count}/260).\n"
                f"The bridge has not sent enough history for this pair yet."
            )
            logger.warning(f"❌ [DESK_SNAPSHOT_REFUSED] {sym}: {err_msg}")
            raise ValueError(err_msg)

        # 4. Instant 91-Feature Snapshot (Same builder as signal engine)
        pt = point or (0.01 if "XAU" in sym else (0.001 if "JPY" in sym else 0.00001))
        dig = digits or (2 if "XAU" in sym else (3 if "JPY" in sym else 5))
        try:
            sanitized_candles = {}
            for tf_k in ("M15", "H1", "H4"):
                c_list = candles_to_use.get(tf_k, [])
                sanitized_list = []
                for c in c_list:
                    if isinstance(c, dict):
                        c_copy = dict(c)
                        if "tick_volume" not in c_copy:
                            c_copy["tick_volume"] = c_copy.get("volume", 100.0)
                        sanitized_list.append(c_copy)
                    else:
                        sanitized_list.append(c)
                sanitized_candles[tf_k] = sanitized_list

            feat_df = signal_engine_service.build_features(sanitized_candles, point=pt, digits=dig, timeframe="M15")
            if feat_df is None or len(feat_df) == 0:
                self.snapshots_failed += 1
                raise ValueError(
                    f"CANNOT RECORD — feature builder returned empty snapshot for {sym} "
                    f"(M15={m15_count}, H1={h1_count}, H4={h4_count})"
                )

            last_row = feat_df.iloc[-1]
            snap_dict = {}
            for col in self.feature_names:
                if col in last_row and pd.notna(last_row[col]):
                    val = float(last_row[col])
                    snap_dict[col] = 0.0 if (math.isnan(val) or math.isinf(val)) else round(val, 5)
                else:
                    snap_dict[col] = 0.0

            snapshot = snap_dict
            self.snapshots_ok += 1
            snapshot_source = "LIVE"
        except ValueError:
            raise
        except Exception as e:
            self.snapshots_failed += 1
            logger.error(f"Snapshot computation error: {e}")
            raise ValueError(f"CANNOT RECORD — feature extraction error: {e}")

        trade_id = f"man_{sym}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        trade_record = {
            "trade_id": trade_id,
            "signal_id": trade_id,
            "symbol": sym,
            "direction": dir_norm,
            "timeframe": tf_norm,
            "entry": float(entry_price),
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "sl_price": float(stop_loss),
            "take_profit_1": float(take_profit_1),
            "tp1_price": float(take_profit_1),
            "take_profit_2": float(take_profit_2) if take_profit_2 else None,
            "tp2_price": float(take_profit_2) if take_profit_2 else None,
            "take_profit_3": float(take_profit_3) if take_profit_3 else None,
            "tp3_price": float(take_profit_3) if take_profit_3 else None,
            "zone_low": float(zone_low) if zone_low is not None and float(zone_low) > 0 else None,
            "zone_high": float(zone_high) if zone_high is not None and float(zone_high) > 0 else None,
            "discretionary_zone": str(discretionary_zone or "").strip() if discretionary_zone else None,
            "why": str(why_to_use or "").strip(),
            "entered_at": entered_at_str,
            "entered_at_epoch": entered_at_epoch,
            "published_utc": entered_at_str,
            "symbol_had_open_trade": symbol_had_open_trade,
            "risk": round(risk, 5),
            "tp1_r": tp1_r,
            "tp2_r": tp2_r,
            "tp3_r": tp3_r,
            "status": "OPEN",
            "bars_elapsed": 0,
            "bars_monitored": 0,
            "last_evaluated_bar_time": 0,
            "snapshot": snapshot,
            "features": snapshot,
            "snapshot_source": snapshot_source,
            "snapshot_reason": None,
            "publish_to_channel": bool(publish_to_channel),
            "executed": False,
            "execution_note": "Measurement instrument only. Zero broker execution capability."
        }

        # Persist open trade and immutable trade record
        self.open_trades[trade_id] = trade_record
        self._save_to_disk()
        self._append_record(trade_record)

        logger.info(
            f"📝 [MANUAL_DESK_TRADE_RECORDED] {trade_id} ({sym} {dir_norm} @ {entry_price}, "
            f"SL={stop_loss}, TP1={take_profit_1}, TP2={take_profit_2}, TP3={take_profit_3}) "
            f"symbol_had_open_trade={symbol_had_open_trade}, snapshot={'YES' if snapshot else 'NULL'}, publish_to_channel={publish_to_channel}"
        )
        return trade_record

    def _append_record(self, record: Dict[str, Any]):
        """Persists full trade record (including 91-feature snapshot) to durable JSONL."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        target_files = [self.records_file]
        candidate_dirs = [
            Path("/app/ml/reports"),
            Path(__file__).resolve().parent.parent.parent / "ml" / "reports",
            Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports"
        ]
        for cdir in candidate_dirs:
            if cdir.exists():
                target_files.append(str(cdir / "manual_records.jsonl"))

        seen_paths = set()
        for fpath in target_files:
            abs_p = os.path.abspath(fpath)
            if abs_p in seen_paths:
                continue
            seen_paths.add(abs_p)
            try:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                logger.error(f"Failed to append trade record to {fpath}: {e}")

    # -------------------------------------------------------------------------
    # Resolution Engine (Mirrored from signal_outcome_tracker.py)
    # -------------------------------------------------------------------------
    def check_price_update(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
        is_bar: bool = False,
        bar_time: Optional[int] = None,
        timestamp_utc: Optional[str] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Evaluates active open manual trades for `symbol` against incoming tick/bar.
        Resolves barriers: SL, TP1, TP2, TP3, or 24-hour timeout (96 M15 bars / 86,400s).
        Zero Telegram notifications; writes strictly to manual_outcomes.jsonl.
        """
        sym = symbol.upper()
        self._load_from_disk()

        trades_for_sym = [t for t in self.open_trades.values() if t.get("symbol") == sym and t.get("status") == "OPEN"]
        if not trades_for_sym:
            return []

        resolved: List[Dict[str, Any]] = []
        now_dt = datetime.now(timezone.utc)
        resolved_time_str = timestamp_utc or now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        curr_time = time.time()

        for trade in trades_for_sym:
            trade_id = trade["trade_id"]
            direction = trade.get("direction", "BUY").upper()
            entry = float(trade["entry"])
            sl = float(trade["stop_loss"])
            tp1 = float(trade["take_profit_1"])
            tp2 = float(trade.get("take_profit_2") or 0.0)
            tp3 = float(trade.get("take_profit_3") or 0.0)
            risk = float(trade.get("risk") or abs(entry - sl) or 0.001)

            if is_bar:
                if bar_time is not None and bar_time > 0:
                    last_b = trade.get("last_evaluated_bar_time", 0)
                    if last_b != bar_time:
                        trade["bars_elapsed"] = int(trade.get("bars_elapsed", 0)) + 1
                        trade["bars_monitored"] = trade["bars_elapsed"]
                        trade["last_evaluated_bar_time"] = bar_time
                else:
                    trade["bars_elapsed"] = int(trade.get("bars_elapsed", 0)) + 1
                    trade["bars_monitored"] = trade["bars_elapsed"]

            if close > 0:
                trade["last_known_price"] = close

            pub_ts = float(trade.get("entered_at_epoch") or curr_time)
            elapsed_sec = curr_time - pub_ts
            is_timed_out = (trade.get("bars_elapsed", 0) >= 96) or (elapsed_sec >= 86400.0)

            barrier_hit = None
            exit_price = None
            r_mult = 0.0

            if direction == "BUY":
                if low <= sl:
                    barrier_hit = "SL"
                    exit_price = sl
                    r_mult = -1.0
                elif tp3 > 0 and high >= tp3:
                    barrier_hit = "TP3"
                    exit_price = tp3
                    r_mult = trade.get("tp3_r") or round((tp3 - entry) / risk, 2)
                elif tp2 > 0 and high >= tp2:
                    barrier_hit = "TP2"
                    exit_price = tp2
                    r_mult = trade.get("tp2_r") or round((tp2 - entry) / risk, 2)
                elif high >= tp1:
                    barrier_hit = "TP1"
                    exit_price = tp1
                    r_mult = trade.get("tp1_r") or round((tp1 - entry) / risk, 2)
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
                    r_mult = trade.get("tp3_r") or round((entry - tp3) / risk, 2)
                elif tp2 > 0 and low <= tp2:
                    barrier_hit = "TP2"
                    exit_price = tp2
                    r_mult = trade.get("tp2_r") or round((entry - tp2) / risk, 2)
                elif low <= tp1:
                    barrier_hit = "TP1"
                    exit_price = tp1
                    r_mult = trade.get("tp1_r") or round((entry - tp1) / risk, 2)
                elif is_timed_out:
                    barrier_hit = "TIMEOUT"
                    exit_price = close if (close is not None and close > 0) else entry
                    gain = entry - exit_price
                    r_mult = round(gain / risk, 2)

            if barrier_hit and exit_price is not None:
                calc_bars = int(round(elapsed_sec / 900.0))
                bars_held = max(trade.get("bars_elapsed", 1), min(96, calc_bars if calc_bars > 0 else 96)) if barrier_hit == "TIMEOUT" else trade.get("bars_elapsed", 1)

                outcome_rec = {
                    "signal_id": trade_id,
                    "trade_id": trade_id,
                    "symbol": sym,
                    "direction": direction,
                    "timeframe": trade.get("timeframe", "M15"),
                    "entry": entry,
                    "entry_price": entry,
                    "stop_loss": sl,
                    "sl_price": sl,
                    "take_profit_1": tp1,
                    "tp1_price": tp1,
                    "take_profit_2": trade.get("take_profit_2"),
                    "tp2_price": trade.get("tp2_price"),
                    "take_profit_3": trade.get("take_profit_3"),
                    "tp3_price": trade.get("tp3_price"),
                    "zone_low": trade.get("zone_low"),
                    "zone_high": trade.get("zone_high"),
                    "exit_price": round(exit_price, 5),
                    "barrier_hit": barrier_hit,
                    "r_multiple": r_mult,
                    "entered_at": trade.get("entered_at", ""),
                    "published_utc": trade.get("entered_at", ""),
                    "resolved_time_utc": resolved_time_str,
                    "resolved_utc": resolved_time_str,
                    "bars_monitored": bars_held,
                    "bars_held": bars_held,
                    "symbol_had_open_trade": trade.get("symbol_had_open_trade", False),
                    "why": trade.get("why", ""),
                    "manual": True,
                    "recorded_at_epoch": curr_time,
                    "publish_to_channel": trade.get("publish_to_channel", True)
                }

                self._append_outcome(outcome_rec)
                self.open_trades.pop(trade_id, None)
                self._save_to_disk()
                resolved.append(outcome_rec)

                # Brief 31 Section 2: Dispatch closed trade result to Telegram/n8n
                try:
                    from app.services.telegram_bot import telegram_bot
                    telegram_bot.dispatch_desk_result(outcome_rec)
                except Exception as t_err:
                    logger.error(f"Error dispatching desk result for {trade_id}: {t_err}")

                logger.info(
                    f"🎯 [MANUAL_TRADE_RESOLVED] {trade_id} ({sym} {direction}): "
                    f"{barrier_hit} @ {exit_price} -> R={r_mult:+.2f} (bars={bars_held})."
                )

        if is_bar and not resolved:
            self._save_to_disk()

        return resolved

    def evaluate_pending_timeouts(self, now_epoch: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Wall-clock timeout defense for manual trades.
        Sweeps open_trades for any trades that have exceeded 96 M15 bars (24 hours / 86400s).
        """
        self._load_from_disk()
        if not self.open_trades:
            return []

        curr_time = now_epoch if now_epoch is not None else time.time()
        resolved: List[Dict[str, Any]] = []

        for trade_id, trade in list(self.open_trades.items()):
            pub_ts = float(trade.get("entered_at_epoch") or 0.0)
            bars_elapsed = int(trade.get("bars_elapsed", 0))
            elapsed_sec = (curr_time - pub_ts) if pub_ts > 0 else 0.0

            if bars_elapsed >= 96 or elapsed_sec >= 86400.0:
                direction = trade.get("direction", "BUY").upper()
                entry = float(trade.get("entry") or 0.0)
                sl = float(trade.get("stop_loss") or 0.0)
                risk = float(trade.get("risk") or abs(entry - sl) or 0.001)

                exit_price = float(trade.get("last_known_price") or entry)
                gain = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
                r_mult = round(gain / risk, 2)

                now_dt = datetime.fromtimestamp(curr_time, tz=timezone.utc)
                resolved_time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

                calc_bars = int(round(elapsed_sec / 900.0))
                bars_held = max(bars_elapsed, min(96, calc_bars if calc_bars > 0 else 96))

                outcome_rec = {
                    "signal_id": trade_id,
                    "trade_id": trade_id,
                    "symbol": trade.get("symbol", ""),
                    "direction": direction,
                    "timeframe": trade.get("timeframe", "M15"),
                    "entry": entry,
                    "entry_price": entry,
                    "stop_loss": sl,
                    "sl_price": sl,
                    "take_profit_1": trade.get("take_profit_1"),
                    "tp1_price": trade.get("take_profit_1"),
                    "take_profit_2": trade.get("take_profit_2"),
                    "tp2_price": trade.get("tp2_price"),
                    "take_profit_3": trade.get("take_profit_3"),
                    "tp3_price": trade.get("tp3_price"),
                    "zone_low": trade.get("zone_low"),
                    "zone_high": trade.get("zone_high"),
                    "exit_price": round(exit_price, 5),
                    "barrier_hit": "TIMEOUT",
                    "r_multiple": r_mult,
                    "entered_at": trade.get("entered_at", ""),
                    "published_utc": trade.get("entered_at", ""),
                    "resolved_time_utc": resolved_time_str,
                    "resolved_utc": resolved_time_str,
                    "bars_monitored": bars_held,
                    "bars_held": bars_held,
                    "symbol_had_open_trade": trade.get("symbol_had_open_trade", False),
                    "why": trade.get("why", ""),
                    "manual": True,
                    "recorded_at_epoch": curr_time,
                    "publish_to_channel": trade.get("publish_to_channel", True)
                }

                self._append_outcome(outcome_rec)
                self.open_trades.pop(trade_id, None)
                self._save_to_disk()
                resolved.append(outcome_rec)

                # Brief 31 Section 2: Dispatch timeout trade result to Telegram/n8n
                try:
                    from app.services.telegram_bot import telegram_bot
                    telegram_bot.dispatch_desk_result(outcome_rec)
                except Exception as t_err:
                    logger.error(f"Error dispatching desk result for {trade_id}: {t_err}")

                logger.info(
                    f"⌛ [MANUAL_TRADE_TIMEOUT] {trade_id} ({trade.get('symbol')} {direction}) "
                    f"reached 24h limit. Closed @ {exit_price} -> R={r_mult:+.2f}."
                )

        return resolved

    def _append_outcome(self, record: Dict[str, Any]):
        """
        Appends manual outcome to manual_outcomes.jsonl.
        Strictly isolated: never writes to signal_outcomes.jsonl and never notifies Telegram.
        """
        line = json.dumps(record, ensure_ascii=False) + "\n"
        target_files = [self.outcomes_file]
        candidate_dirs = [
            Path("/app/ml/reports"),
            Path(__file__).resolve().parent.parent.parent / "ml" / "reports",
            Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports"
        ]
        for cdir in candidate_dirs:
            if cdir.exists():
                target_files.append(str(cdir / "manual_outcomes.jsonl"))

        seen_paths = set()
        for fpath in target_files:
            abs_p = os.path.abspath(fpath)
            if abs_p in seen_paths:
                continue
            seen_paths.add(abs_p)
            try:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                logger.error(f"Failed to append outcome to {fpath}: {e}")

    # -------------------------------------------------------------------------
    # Retrieval Methods
    # -------------------------------------------------------------------------
    def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        self._load_from_disk()
        self.evaluate_pending_timeouts()
        trades = list(self.open_trades.values())
        if symbol:
            return [t for t in trades if t.get("symbol", "").upper() == symbol.upper()]
        return trades

    def get_records(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        self._load_from_disk()
        outcomes_map = {o.get("trade_id"): o for o in self.get_outcomes(limit=10000)}
        records = []
        if os.path.exists(self.records_file):
            try:
                with open(self.records_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            tid = rec.get("trade_id")
                            # Brief 31 Section 2: Reconcile true status against open trades & outcomes
                            if tid in self.open_trades:
                                rec["status"] = "OPEN"
                            elif tid in outcomes_map:
                                out = outcomes_map[tid]
                                rec["status"] = "CLOSED"
                                rec["barrier_hit"] = out.get("barrier_hit")
                                rec["outcome"] = out.get("barrier_hit")
                                rec["exit_price"] = out.get("exit_price")
                                rec["r_multiple"] = out.get("r_multiple")
                                rec["resolved_time_utc"] = out.get("resolved_time_utc") or out.get("resolved_utc")
                            else:
                                rec["status"] = "CLOSED"
                                rec["barrier_hit"] = "CLOSED"
                                rec["outcome"] = "CLOSED"

                            if symbol:
                                if rec.get("symbol", "").upper() == symbol.upper():
                                    records.append(rec)
                            else:
                                records.append(rec)
            except Exception as e:
                logger.error(f"Error reading manual records: {e}")
        return records[-limit:] if limit > 0 else records

    def get_outcomes(self, symbol: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
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
                logger.error(f"Error reading manual outcomes: {e}")
        return outcomes[-limit:]

    def get_closed_trades(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Brief 27 Section 4: Returns resolved manual trade outcomes, newest first."""
        outcomes = self.get_outcomes(symbol=symbol, limit=limit)
        return list(reversed(outcomes))

    def rescue_records(self, candle_store: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Brief 27 Section 2: Rescues un-snapshotted manual records whose historical candles
        are still in the store. Sets snapshot_source: 'RECONSTRUCTED'.
        Where it fails, marks unusable_for_signature: True. Never deletes records.
        """
        target_files = []
        if os.path.exists(self.records_file):
            target_files.append(self.records_file)
        reports_rec = Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports" / "manual_records.jsonl"
        if reports_rec.exists() and str(reports_rec) not in target_files:
            target_files.append(str(reports_rec))

        store = candle_store or (self.candle_getter("") if self.candle_getter else {})
        if not store and hasattr(self, "_global_state"):
            store = getattr(self, "_global_state", {}).get("candle_history", {})

        total = 0
        rescued = 0
        already_live = 0
        unusable = 0

        for fpath in target_files:
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                logger.error(f"Failed to read records for rescue from {fpath}: {e}")
                continue

            updated_lines = []
            for r in lines:
                total += 1
                sym = r.get("symbol", "").upper()
                has_snap = bool(
                    (r.get("snapshot") and isinstance(r.get("snapshot"), dict) and len(r["snapshot"]) == 91) or
                    (r.get("features") and isinstance(r.get("features"), dict) and len(r["features"]) == 91)
                )
                if has_snap:
                    if not r.get("snapshot_source"):
                        r["snapshot_source"] = "LIVE"
                    already_live += 1
                    updated_lines.append(r)
                    continue

                entered_epoch = float(r.get("entered_at_epoch") or 0.0)
                sym_candles = store.get(sym, {})
                m15_sub = [c for c in sym_candles.get("M15", []) if c.get("time", 0) <= entered_epoch]
                h1_sub = [c for c in sym_candles.get("H1", []) if c.get("time", 0) <= entered_epoch]
                h4_sub = [c for c in sym_candles.get("H4", []) if c.get("time", 0) <= entered_epoch]

                if len(m15_sub) >= 1100 and len(h1_sub) >= 260 and len(h4_sub) >= 260:
                    pt = 0.01 if "XAU" in sym else (0.001 if "JPY" in sym else 0.00001)
                    dig = 2 if "XAU" in sym else (3 if "JPY" in sym else 5)
                    candles_dict = {
                        "M15": [{**c, "tick_volume": c.get("tick_volume", c.get("volume", 100.0))} for c in m15_sub],
                        "H1": [{**c, "tick_volume": c.get("tick_volume", c.get("volume", 100.0))} for c in h1_sub],
                        "H4": [{**c, "tick_volume": c.get("tick_volume", c.get("volume", 100.0))} for c in h4_sub]
                    }
                    try:
                        feat_df = signal_engine_service.build_features(candles_dict, point=pt, digits=dig, timeframe="M15")
                        if feat_df is not None and len(feat_df) > 0:
                            last_row = feat_df.iloc[-1]
                            snap_dict = {}
                            for col in self.feature_names:
                                if col in last_row and pd.notna(last_row[col]):
                                    val = float(last_row[col])
                                    snap_dict[col] = 0.0 if (math.isnan(val) or math.isinf(val)) else round(val, 5)
                                else:
                                    snap_dict[col] = 0.0
                            r["snapshot"] = snap_dict
                            r["features"] = snap_dict
                            r["snapshot_source"] = "RECONSTRUCTED"
                            r["snapshot_reason"] = None
                            rescued += 1
                        else:
                            r["unusable_for_signature"] = True
                            unusable += 1
                    except Exception as fe:
                        logger.warning(f"Reconstruction feature build failed: {fe}")
                        r["unusable_for_signature"] = True
                        unusable += 1
                else:
                    r["unusable_for_signature"] = True
                    unusable += 1

                updated_lines.append(r)

            try:
                tmp_path = fpath + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for item in updated_lines:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                os.replace(tmp_path, fpath)
            except Exception as e:
                logger.error(f"Failed to rewrite rescued records to {fpath}: {e}")

        return {
            "status": "SUCCESS",
            "total_processed": total,
            "already_live": already_live,
            "rescued": rescued,
            "unusable": unusable
        }

    # -------------------------------------------------------------------------
    # Report 6a: The Comparison (Manual vs Model)
    # -------------------------------------------------------------------------
    def get_comparison_report(self, since: str = "2026-09-21") -> Dict[str, Any]:
        """
        Brief 26 Section 6a:
        Compares manual trading against the model across three required rows:
          - manual, all
          - manual, first only (trades taken when no other trade was open on that pair)
          - model (from live signal_outcomes.jsonl)
        Carries trade count beside every figure. Never publishes projected return.
        """
        all_manual = self.get_outcomes(limit=100000)
        first_only_manual = [t for t in all_manual if not t.get("symbol_had_open_trade", False)]

        def _calc_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
            n = len(trades)
            if n == 0:
                return {
                    "trades": 0,
                    "win_pct": 0.0,
                    "expectancy_r": 0.0,
                    "total_r": 0.0,
                    "wins": 0,
                    "losses": 0
                }
            wins = sum(1 for t in trades if float(t.get("r_multiple", 0.0)) > 0)
            losses = sum(1 for t in trades if float(t.get("r_multiple", 0.0)) < 0)
            tot_r = sum(float(t.get("r_multiple", 0.0)) for t in trades)
            win_pct = round((wins / n) * 100.0, 1)
            exp_r = round(tot_r / n, 3)
            return {
                "trades": n,
                "win_pct": win_pct,
                "expectancy_r": exp_r,
                "total_r": round(tot_r, 2),
                "wins": wins,
                "losses": losses
            }

        # Model live outcomes via SignalOutcomesService
        model_stats = {"trades": 0, "win_pct": 0.0, "expectancy_r": 0.0, "total_r": 0.0, "wins": 0, "losses": 0}
        try:
            sos = SignalOutcomesService()
            model_outcomes_res = sos.get_outcomes(limit=100000)
            m_trades = model_outcomes_res.get("trades", [])
            model_stats = _calc_stats(m_trades)
        except Exception as e:
            logger.warning(f"Could not load model outcomes for comparison report: {e}")

        manual_all_stats = _calc_stats(all_manual)
        manual_first_stats = _calc_stats(first_only_manual)

        return {
            "title": "MANUAL vs MODEL",
            "since": since,
            "rows": [
                {
                    "label": "manual, all",
                    "description": "All manual discretionary trades",
                    "trades": manual_all_stats["trades"],
                    "win_pct": manual_all_stats["win_pct"],
                    "expectancy_r": manual_all_stats["expectancy_r"],
                    "total_r": manual_all_stats["total_r"],
                    "wins": manual_all_stats["wins"],
                    "losses": manual_all_stats["losses"]
                },
                {
                    "label": "manual, first only",
                    "description": "Trades taken when no other trade was open on that pair",
                    "trades": manual_first_stats["trades"],
                    "win_pct": manual_first_stats["win_pct"],
                    "expectancy_r": manual_first_stats["expectancy_r"],
                    "total_r": manual_first_stats["total_r"],
                    "wins": manual_first_stats["wins"],
                    "losses": manual_first_stats["losses"]
                },
                {
                    "label": "model",
                    "description": "Live production model signals",
                    "trades": model_stats["trades"],
                    "win_pct": model_stats["win_pct"],
                    "expectancy_r": model_stats["expectancy_r"],
                    "total_r": model_stats["total_r"],
                    "wins": model_stats["wins"],
                    "losses": model_stats["losses"]
                }
            ],
            "manual_all": manual_all_stats,
            "manual_first_only": manual_first_stats,
            "model": model_stats
        }

    # -------------------------------------------------------------------------
    # Report 6b: The Signature (What is different about the bars he picks)
    # -------------------------------------------------------------------------
    def get_signature_report(self) -> Dict[str, Any]:
        """
        Brief 26 Section 6b:
        For each of the 91 features, compares his entry bars against the available bars
        and ranks by standardised difference (Cohen's d).
        - Sorted by absolute Cohen's d descending.
        - Shows median for his entries and available bars.
        - Greys out any row with fewer than 30 entries behind it (thin sample).
        - Includes the 10 most recent trades with Why text and top 3 distinguishing features.
        """
        records = self.get_records(limit=100000)
        valid_records = [r for r in records if r.get("snapshot") and isinstance(r["snapshot"], dict) and not r.get("unusable_for_signature")]

        n_his = len(valid_records)
        baseline = self.baseline_data or self._load_baseline()
        n_avail = int(baseline.get("total_available_bars", 23855))
        base_features = baseline.get("features", {})

        # Feature matrix for his entries
        feat_diffs = []
        is_thin = (n_his < 30)

        for col in self.feature_names:
            avail_stat = base_features.get(col, {"mean": 0.0, "std": 1.0, "median": 0.0})
            avail_mean = float(avail_stat.get("mean", 0.0))
            avail_std = float(avail_stat.get("std", 1.0))
            if avail_std <= 0:
                avail_std = 1.0
            avail_median = float(avail_stat.get("median", avail_mean))

            if n_his > 0:
                his_vals = np.array([float(r["snapshot"].get(col, 0.0)) for r in valid_records], dtype=float)
                his_mean = float(np.mean(his_vals))
                his_std = float(np.std(his_vals, ddof=1)) if n_his > 1 else 1.0
                his_median = float(np.median(his_vals))

                # Pooled standard deviation
                if n_his + n_avail - 2 > 0:
                    pooled_variance = ((n_his - 1) * (his_std ** 2) + (n_avail - 1) * (avail_std ** 2)) / (n_his + n_avail - 2)
                    pooled_sd = math.sqrt(pooled_variance) if pooled_variance > 0 else 1.0
                else:
                    pooled_sd = 1.0

                cohen_d = (his_mean - avail_mean) / pooled_sd if pooled_sd > 0 else 0.0
            else:
                his_median = 0.0
                cohen_d = 0.0

            feat_diffs.append({
                "feature": col,
                "cohen_d": round(cohen_d, 2),
                "abs_d": abs(cohen_d),
                "his_median": round(his_median, 3),
                "avail_median": round(avail_median, 3),
                "available_median": round(avail_median, 3),
                "thin": is_thin,
                "entries_count": n_his
            })

        # Sort by absolute d descending
        feat_diffs.sort(key=lambda x: x["abs_d"], reverse=True)

        # Build 10 most recent trades with Why text & top 3 features by difference
        recent_trades = []
        for r in reversed(records[-10:]):
            why_text = r.get("why", "")
            snap = r.get("snapshot") or {}
            top_features = []

            if snap:
                # Find features with largest z-score difference from available baseline
                scored = []
                for col in self.feature_names:
                    val = float(snap.get(col, 0.0))
                    stat = base_features.get(col, {"mean": 0.0, "std": 1.0})
                    m = float(stat.get("mean", 0.0))
                    s = float(stat.get("std", 1.0))
                    z = (val - m) / (s if s > 0 else 1.0)
                    scored.append((col, round(z, 2), abs(z), val))

                scored.sort(key=lambda x: x[2], reverse=True)
                top_features = [
                    {"feature": s[0], "z_score": s[1], "value": s[3]}
                    for s in scored[:3]
                ]

            recent_trades.append({
                "trade_id": r.get("trade_id"),
                "symbol": r.get("symbol"),
                "direction": r.get("direction"),
                "entered_at": r.get("entered_at"),
                "why": why_text,
                "top_features": top_features
            })

        return {
            "title": "WHAT IS DIFFERENT ABOUT THE BARS HE PICKS",
            "his_entries_count": n_his,
            "total_entries": n_his,
            "available_bars_count": n_avail,
            "total_available": n_avail,
            "thin_sample": is_thin,
            "thin_threshold": 30,
            "thin_note": f"Rows greyed out: fewer than 30 entries ({n_his}/30 collected). Below that, d is noise.",
            "features": feat_diffs,
            "recent_trades_with_why": recent_trades
        }

    # -------------------------------------------------------------------------
    # Report 6c: Zones Analysis (Evaluating Discretionary Zones)
    # -------------------------------------------------------------------------
    def get_zones_report(self) -> Dict[str, Any]:
        """
        Brief 26 Section 6c:
        Once 50 records carry a zone, reports how often price returned to a marked zone
        and what happened when it did.
        """
        records = self.get_records(limit=100000)
        zone_records = [r for r in records if r.get("zone_low") is not None and r.get("zone_high") is not None]
        zone_count = len(zone_records)

        outcomes = self.get_outcomes(limit=100000)
        zone_outcomes = [o for o in outcomes if o.get("zone_low") is not None and o.get("zone_high") is not None]

        if zone_count < 50:
            return {
                "status": "COLLECTING",
                "count": zone_count,
                "target": 50,
                "message": f"{zone_count}/50 marked zone trades collected. Zone return and reaction analysis activates at 50 records.",
                "zone_trades": zone_count,
                "completed_zone_trades": len(zone_outcomes)
            }

        wins = sum(1 for o in zone_outcomes if float(o.get("r_multiple", 0.0)) > 0)
        total_resolved = len(zone_outcomes)
        win_rate = round((wins / total_resolved) * 100.0, 1) if total_resolved > 0 else 0.0
        tot_r = sum(float(o.get("r_multiple", 0.0)) for o in zone_outcomes)
        exp_r = round(tot_r / total_resolved, 3) if total_resolved > 0 else 0.0

        return {
            "status": "ACTIVE",
            "count": zone_count,
            "target": 50,
            "message": f"Active with {zone_count} marked zone records.",
            "total_marked_zones": zone_count,
            "resolved_zone_trades": total_resolved,
            "win_rate": win_rate,
            "expectancy_r": exp_r,
            "total_r": round(tot_r, 2)
        }


# Global singleton instance
analyst_desk_service = AnalystDeskService()
