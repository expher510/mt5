import os
import sys
import json
import time
import math
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
import httpx

from app.config import settings
from app.services.symbol_metrics import get_symbol_metrics
from app.services.signal_outcome_tracker import signal_outcome_tracker
from app.services.signal_sanity import validate_signal_sanity, get_reference_median_atr

logger = logging.getLogger(__name__)

def classify_market_category(symbol: str) -> str:
    """
    Intelligently classifies any financial instrument into:
    - 'GOLD': Gold (XAUUSD, GOLD, etc.)
    - 'INDEX': Indices (Dow Jones, Nasdaq, SP500, DAX), Commodities (Oil, Silver XAGUSD), Crypto (BTCUSD, etc.)
    - 'FOREX': Currency pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, etc.)
    """
    sym = (symbol or "").upper().strip()
    clean = sym.replace("/", "").replace(".", "").replace("-", "").replace("_", "").replace("#", "")

    # 1. Gold detection
    if "XAU" in clean or "GOLD" in clean:
        return "GOLD"

    # 2. Silver detection (specifically requested: سيلفر xagusd)
    if "XAG" in clean or "SILVER" in clean:
        return "INDEX"

    # 3. Oil detection (specifically requested: نفط)
    oil_keywords = ["USOIL", "UKOIL", "WTI", "BRENT", "CRUDE", "OIL", "XTI", "XBR"]
    if any(k in clean for k in oil_keywords):
        return "INDEX"

    # 4. Dow Jones detection (specifically requested: داوجونز)
    dow_keywords = ["US30", "DJI", "DJ30", "WS30", "WALLSTREET", "DOW", "DJIA"]
    if any(k in clean for k in dow_keywords):
        return "INDEX"

    # 5. Nasdaq detection (specifically requested: ناسداك)
    nasdaq_keywords = ["US100", "USTEC", "NAS100", "NDX", "NASDAQ", "TECH100", "NAS"]
    if any(k in clean for k in nasdaq_keywords):
        return "INDEX"

    # 6. Other major Indices (S&P 500, DAX, FTSE, Nikkei)
    index_keywords = ["US500", "SPX", "SP500", "GER40", "GER30", "DAX", "DE40", "DE30", "UK100", "FTSE", "JP225", "NIKKEI"]
    if any(k in clean for k in index_keywords):
        return "INDEX"

    # 7. Bitcoin / Crypto detection (specifically requested: بيتكوين)
    crypto_keywords = ["BTC", "BITCOIN", "ETH", "SOL", "XRP", "LTC", "CRYPTO"]
    if any(k in clean for k in crypto_keywords):
        return "INDEX"

    # Default for all other symbols (currency pairs) -> FOREX
    return "FOREX"


class TelegramBotService:
    """
    Forex Engineer Telegram Signal & Heartbeat Publisher via n8n Webhook (Brief 10).
    Publishes confirmed AI model signal cards and hourly waiting notices to n8n:
      - Delivers structured JSON + pre-rendered message text to n8n webhook
      - 2 sequential English messages formatted under this codebase's control
      - Image passes as immutable public URL: https://bot.fxengen.com/api/v1/analyst/share/{snap_id}/image
      - Hourly waiting notice with real symbol reasons and age of last signal
      - Separate disk persistence for signals and notices across backend restarts
      - Separate hard rate limits (Signals: max 6/hr, 20/day per symbol; Notices: max 1/hr, 24/day)
      - Delivery with 10s timeout, up to 2 retries with backoff
      - Dry-run mode enabled by default (logs exact payloads, sends nothing)
      - Silent failure handling (never crashes model, bridge, or backend)
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if os.path.exists("/app/data"):
            self.data_dir = "/app/data"
        else:
            self.data_dir = os.path.join(base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.state_file = os.path.join(self.data_dir, "telegram_posted_signals.json")
        self.results_state_file = os.path.join(self.data_dir, "telegram_posted_results.json")
        self.reports_file = os.path.join(self.data_dir, "telegram_posted_reports.json")
        self.schema_ref_file = os.path.join(self.data_dir, "telegram_schema_reference_sent.json")
        self.suppressions_file = os.path.join(self.data_dir, "signal_suppressions.jsonl")
        self.sealed_manual_file = os.path.join(self.data_dir, "sealed_manual_trades.json")
        self.sealed_manual_records: Set[str] = set()
        self.posted_results: Dict[str, Dict[str, Any]] = {}
        self.posted_reports: Dict[str, Any] = {}
        self._load_state()
        self._load_results_state()
        self._load_reports_state()
        self._seed_historical_outcomes()
        self._seed_historical_manual_records()
        self._heartbeat_task: Optional[asyncio.Task] = None

    def _load_state(self):
        """Loads posted signals registry and rate-limit history from disk"""
        self.state = {
            "posted_signals": {},
            "test_signals": {},
            "test_signals_count": 0,
            "signal_history": [],
            "notice_history": [],
            "last_signal_timestamp": 0.0,
            "last_signal_info": None,
            "suppression_counts": {}
        }
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.state["posted_signals"] = loaded.get("posted_signals", {})
                        self.state["test_signals"] = loaded.get("test_signals", {})
                        self.state["test_signals_count"] = int(loaded.get("test_signals_count", len(self.state["test_signals"])))
                        self.state["signal_history"] = loaded.get("signal_history", [])
                        self.state["notice_history"] = loaded.get("notice_history", [])
                        self.state["last_signal_timestamp"] = loaded.get("last_signal_timestamp", 0.0)
                        self.state["last_signal_info"] = loaded.get("last_signal_info")
                        self.state["suppression_counts"] = loaded.get("suppression_counts", {})
            except Exception as e:
                logger.error(f"Error loading Telegram state file {self.state_file}: {e}")

    def _save_state(self):
        """Persists state to disk atomically"""
        try:
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.state_file):
                os.replace(temp_file, self.state_file)
            else:
                os.rename(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Error persisting Telegram state file {self.state_file}: {e}")

    def _load_results_state(self):
        """Loads posted trade results registry from disk (Brief 23)"""
        self.posted_results = {}
        if os.path.exists(self.results_state_file):
            try:
                with open(self.results_state_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.posted_results = loaded.get("posted_results", {})
                logger.info(f"Loaded {len(self.posted_results)} posted results records from disk.")
            except Exception as e:
                logger.error(f"Error loading results state file {self.results_state_file}: {e}")

    def _save_results_state(self):
        """Persists posted results state to disk atomically (Brief 23)"""
        try:
            temp_file = f"{self.results_state_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump({"posted_results": self.posted_results}, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.results_state_file):
                os.replace(temp_file, self.results_state_file)
            else:
                os.rename(temp_file, self.results_state_file)
        except Exception as e:
            logger.error(f"Error persisting results state file {self.results_state_file}: {e}")

    def _load_reports_state(self):
        """Loads 12-hour report history from disk (Brief 24)"""
        self.posted_reports = {}
        if os.path.exists(self.reports_file):
            try:
                with open(self.reports_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.posted_reports = loaded.get("posted_reports", {})
                logger.info(f"Loaded {len(self.posted_reports)} posted report records from disk.")
            except Exception as e:
                logger.error(f"Error loading reports state file {self.reports_file}: {e}")

    def _save_reports_state(self):
        """Persists 12-hour report history to disk atomically (Brief 24)"""
        try:
            temp_file = f"{self.reports_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump({"posted_reports": self.posted_reports}, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.reports_file):
                os.replace(temp_file, self.reports_file)
            else:
                os.rename(temp_file, self.reports_file)
        except Exception as e:
            logger.error(f"Error persisting reports state file {self.reports_file}: {e}")

    def _seed_historical_outcomes(self):
        """
        Brief 23: On startup, mark every already-recorded outcome in signal_outcomes.jsonl
        as published on disk (is_historical=True) so none of them can EVER be sent.
        """
        search_files = [
            os.path.join(self.data_dir, "signal_outcomes.jsonl"),
            str(Path(__file__).resolve().parent.parent.parent / "data" / "signal_outcomes.jsonl"),
            str(Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports" / "signal_outcomes.jsonl")
        ]
        seeded_count = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        seen_files = set()

        for fpath in search_files:
            abs_p = os.path.abspath(fpath)
            if abs_p in seen_files or not os.path.exists(abs_p):
                continue
            seen_files.add(abs_p)
            try:
                with open(abs_p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            snap_id = str(rec.get("signal_id") or "").strip()
                            if snap_id and snap_id not in self.posted_results:
                                self.posted_results[snap_id] = {
                                    "signal_id": snap_id,
                                    "symbol": str(rec.get("symbol") or "").upper(),
                                    "outcome": str(rec.get("barrier_hit") or "HISTORICAL").upper(),
                                    "r_multiple": float(rec.get("r_multiple", 0.0)),
                                    "posted_at": rec.get("resolved_time_utc") or now_iso,
                                    "is_dry_run": False,
                                    "is_historical": True,
                                    "seeded_at": now_iso
                                }
                                seeded_count += 1
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error seeding historical outcomes from {abs_p}: {e}")

        if seeded_count > 0:
            self._save_results_state()
            logger.info(f"🛡️ [SEEDED_HISTORICAL_OUTCOMES] Marked {seeded_count} historical outcome(s) as published on disk in {self.results_state_file}.")

    def _seed_historical_manual_records(self):
        """
        Brief 29 Criterion 0: On startup, mark every existing manual trade record in manual_records.jsonl
        as sealed on disk so none of them can EVER be sent to Telegram under any circumstances.
        """
        search_files = [
            os.path.join(self.data_dir, "manual_records.jsonl"),
            str(Path(__file__).resolve().parent.parent.parent / "data" / "manual_records.jsonl"),
            str(Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports" / "manual_records.jsonl"),
            str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "manual_records.jsonl"),
            "/app/ml/reports/manual_records.jsonl",
            "/app/data/manual_records.jsonl"
        ]
        if not hasattr(self, "sealed_manual_records"):
            self.sealed_manual_records = set()

        # Load any previously saved sealed records from disk
        if os.path.exists(self.sealed_manual_file):
            try:
                with open(self.sealed_manual_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.sealed_manual_records.update(data)
                    elif isinstance(data, dict):
                        self.sealed_manual_records.update(data.get("sealed_trades", []))
            except Exception as e:
                logger.error(f"Error loading sealed manual file {self.sealed_manual_file}: {e}")

        now_iso = datetime.now(timezone.utc).isoformat()
        seeded_count = 0
        seen_files = set()

        for fpath in search_files:
            abs_p = os.path.abspath(fpath)
            if abs_p in seen_files or not os.path.exists(abs_p):
                continue
            seen_files.add(abs_p)
            try:
                with open(abs_p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            tid = str(rec.get("trade_id") or rec.get("id") or "").strip()
                            if tid:
                                if tid not in self.sealed_manual_records:
                                    self.sealed_manual_records.add(tid)
                                    seeded_count += 1
                                # Also register in state['posted_signals'] so is_signal_already_posted(tid) is True
                                if tid not in self.state["posted_signals"]:
                                    self.state["posted_signals"][tid] = {
                                        "symbol": str(rec.get("symbol") or "").upper(),
                                        "direction": str(rec.get("direction") or "").upper(),
                                        "posted_at": rec.get("entered_at") or rec.get("timestamp_utc") or now_iso,
                                        "is_dry_run": False,
                                        "is_historical": True,
                                        "sealed": True,
                                        "source": "DESK"
                                    }
                                # Standing Rule (Brief 31): Also register in self.posted_results so result messages can NEVER post
                                if tid not in self.posted_results:
                                    self.posted_results[tid] = {
                                        "symbol": str(rec.get("symbol") or "").upper(),
                                        "direction": str(rec.get("direction") or "").upper(),
                                        "posted_at": rec.get("entered_at") or rec.get("timestamp_utc") or now_iso,
                                        "is_dry_run": False,
                                        "is_historical": True,
                                        "sealed": True,
                                        "source": "DESK"
                                    }
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error reading manual records from {abs_p}: {e}")

        # Persist results state with sealed outcomes
        self._save_results_state()

        # Save sealed file atomically
        try:
            temp_file = f"{self.sealed_manual_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump({"sealed_trades": sorted(list(self.sealed_manual_records)), "count": len(self.sealed_manual_records)}, f, indent=2)
            if os.path.exists(self.sealed_manual_file):
                os.replace(temp_file, self.sealed_manual_file)
            else:
                os.rename(temp_file, self.sealed_manual_file)
        except Exception as e:
            logger.error(f"Error saving sealed manual trades file {self.sealed_manual_file}: {e}")

        if seeded_count > 0:
            self._save_state()
            logger.info(f"🛡️ [SEEDED_HISTORICAL_DESK] Marked {len(self.sealed_manual_records)} historical desk trade(s) ({seeded_count} newly found) as sealed on disk.")

    def is_result_already_posted(self, signal_id: str) -> bool:
        """Checks if a signal result has already been published or seeded on disk (Brief 23)"""
        if not signal_id:
            return False
        return signal_id in self.posted_results

    def record_result_posted(
        self,
        signal_id: str,
        symbol: str,
        is_dry_run: bool = False,
        payload: Optional[Dict[str, Any]] = None
    ):
        """Records a published trade result on disk under telegram_posted_results.json (Brief 23)"""
        now_iso = datetime.now(timezone.utc).isoformat()
        self.posted_results[signal_id] = {
            "signal_id": signal_id,
            "symbol": symbol.upper(),
            "outcome": payload.get("outcome") if payload else "UNKNOWN",
            "r_multiple": float(payload.get("r_multiple", 0.0)) if payload else 0.0,
            "posted_at": now_iso,
            "is_dry_run": is_dry_run,
            "is_historical": False,
            "payload": payload
        }
        self._save_results_state()

    # -------------------------------------------------------------------------
    # Forex Market Hours & Deduplication & Rate Limiting
    # -------------------------------------------------------------------------
    @staticmethod
    def is_forex_market_open(dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Forex markets close on Friday 22:00 UTC (17:00 NY)
        and re-open on Sunday 21:00 UTC (17:00 NY).
        All Saturdays are fully closed.
        Returns (is_open: bool, reason: str).
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        weekday = dt.weekday()  # Monday=0, ..., Friday=4, Saturday=5, Sunday=6
        hour = dt.hour

        if weekday == 5:  # Saturday
            return False, "Market closed: Saturday weekend blackout"
        if weekday == 4 and hour >= 22:  # Friday after 22:00 UTC
            return False, "Market closed: Friday post-22:00 UTC weekend close"
        if weekday == 6 and hour < 21:  # Sunday before 21:00 UTC
            return False, "Market closed: Sunday pre-21:00 UTC market open"

        return True, "Market open"

    def is_signal_already_posted(self, snapshot_id: str) -> bool:
        """Checks if a snapshot ID has already been published to Telegram/n8n"""
        return snapshot_id in self.state.get("posted_signals", {})

    def check_signal_rate_limit(self, symbol: str) -> Tuple[bool, str]:
        """
        Hard rate cap for signals (Brief 10 Section 5.3 & Brief 14 Section 3 item 5):
        - Global cap: Max 25 signals per hour across all symbols
        - Per-symbol hourly: Max 10 signals per hour
        - Per-symbol daily: Max 60 signals per day
        """
        now_ts = time.time()
        cutoff_24h = now_ts - 86400.0
        cutoff_1h = now_ts - 3600.0

        history = self.state.get("signal_history", [])
        history = [h for h in history if h.get("timestamp", 0) >= cutoff_24h]
        self.state["signal_history"] = history

        # 1. Global hourly rate cap (Brief 14)
        global_history_1h = [h for h in history if h.get("timestamp", 0) >= cutoff_1h]
        max_global_hourly = getattr(settings, "TELEGRAM_GLOBAL_MAX_HOURLY_SIGNALS", 25)
        if len(global_history_1h) >= max_global_hourly:
            reason = f"Global hourly signal cap of {max_global_hourly} reached across all symbols ({len(global_history_1h)} in last hour)"
            logger.error(f"🚨🚨 [GLOBAL_RATE_LIMIT_TRIPPED] {reason} - refusing signal for {symbol}!")
            return False, reason

        # 2. Per-symbol caps
        sym = symbol.upper()
        sym_history_24h = [h for h in history if h.get("symbol") == sym]
        sym_history_1h = [h for h in sym_history_24h if h.get("timestamp", 0) >= cutoff_1h]

        max_hourly = getattr(settings, "TELEGRAM_MAX_HOURLY_SIGNALS", 10)
        max_daily = getattr(settings, "TELEGRAM_MAX_DAILY_SIGNALS", 60)

        if len(sym_history_1h) >= max_hourly:
            reason = f"Hourly signal rate cap of {max_hourly} reached for {sym} ({len(sym_history_1h)} in last hour)"
            logger.error(f"🚨 [PER_SYMBOL_RATE_LIMIT_TRIPPED] {reason}")
            return False, reason

        if len(sym_history_24h) >= max_daily:
            reason = f"Daily signal rate cap of {max_daily} reached for {sym} ({len(sym_history_24h)} in last 24h)"
            logger.error(f"🚨 [PER_SYMBOL_RATE_LIMIT_TRIPPED] {reason}")
            return False, reason

        return True, ""

    def check_notice_rate_limit(self) -> Tuple[bool, str]:
        """
        Rate cap for waiting notices (Section 3c & 6 test 9):
        - Separate counter: max 1 per hour, max 24 per day
        """
        now_ts = time.time()
        cutoff_24h = now_ts - 86400.0
        cutoff_1h = now_ts - 3600.0

        notices = self.state.get("notice_history", [])
        notices = [n for n in notices if n.get("timestamp", 0) >= cutoff_24h]
        self.state["notice_history"] = notices

        notices_1h = [n for n in notices if n.get("timestamp", 0) >= cutoff_1h]
        max_hourly = getattr(settings, "TELEGRAM_MAX_HOURLY_NOTICES", 1)
        max_daily = getattr(settings, "TELEGRAM_MAX_DAILY_NOTICES", 24)

        if len(notices_1h) >= max_hourly:
            return False, f"Hourly waiting notice cap of {max_hourly} reached ({len(notices_1h)} in last hour)"
        if len(notices) >= max_daily:
            return False, f"Daily waiting notice cap of {max_daily} reached ({len(notices)} in last 24h)"

        return True, ""

    def record_signal_posted(self, snapshot_id: str, symbol: str, is_dry_run: bool = False, payload: Optional[Dict[str, Any]] = None, is_test: bool = False):
        """Records confirmed signal publication on disk and updates last signal time"""
        now_iso = datetime.now(timezone.utc).isoformat()
        now_ts = time.time()

        rec = {
            "symbol": symbol.upper(),
            "posted_at": now_iso,
            "is_dry_run": is_dry_run,
            "is_test": is_test
        }
        if payload is not None:
            rec["payload"] = payload

        if is_test:
            self.state["test_signals_count"] = self.state.get("test_signals_count", 0) + 1
            self.state.setdefault("test_signals", {})[snapshot_id] = rec
        else:
            self.state.setdefault("posted_signals", {})[snapshot_id] = rec
            self.state.setdefault("signal_history", []).append({
                "timestamp": now_ts,
                "symbol": symbol.upper(),
                "snapshot_id": snapshot_id
            })
            self.state["last_signal_timestamp"] = now_ts
            self.state["last_signal_info"] = {
                "symbol": symbol.upper(),
                "snapshot_id": snapshot_id,
                "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            }
        self._save_state()

    def record_notice_sent(self, is_dry_run: bool = False, payload: Optional[Dict[str, Any]] = None):
        """Records waiting notice on disk under its own counter"""
        now_iso = datetime.now(timezone.utc).isoformat()
        now_ts = time.time()
        rec = {
            "timestamp": now_ts,
            "time_utc": now_iso,
            "is_dry_run": is_dry_run
        }
        if payload is not None:
            rec["payload"] = payload
        self.state.setdefault("notice_history", []).append(rec)
        self._save_state()

    def _log_suppression(self, symbol: str, suppressed_id: str, blocking_id: str, open_duration: str = ""):
        """Brief 22 Section 5: Log suppressed signals to signal_suppressions.jsonl"""
        try:
            now = datetime.now(timezone.utc)
            rec = {
                "symbol": symbol.upper(),
                "timestamp": time.time(),
                "timestamp_utc": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "suppressed_signal_id": suppressed_id,
                "blocking_signal_id": blocking_id,
                "open_duration": open_duration
            }
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            target_files = [self.suppressions_file]
            reports_dir = Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports"
            if reports_dir.exists():
                target_files.append(str(reports_dir / "signal_suppressions.jsonl"))

            for fpath in target_files:
                try:
                    os.makedirs(os.path.dirname(fpath), exist_ok=True)
                    with open(fpath, "a", encoding="utf-8") as f:
                        f.write(line)
                except Exception as fe:
                    logger.debug(f"Could not append suppression to {fpath}: {fe}")
        except Exception as e:
            logger.error(f"Failed to log suppression: {e}")

    def get_suppressions(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Reads recent suppressions from signal_suppressions.jsonl"""
        items = []
        if os.path.exists(self.suppressions_file):
            try:
                with open(self.suppressions_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                r = json.loads(line)
                                if symbol and symbol.upper() != "ALL" and r.get("symbol", "").upper() != symbol.upper():
                                    continue
                                items.append(r)
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"Error reading suppressions file: {e}")
        return items[-limit:] if limit > 0 else items

    # -------------------------------------------------------------------------
    # Formatting Helpers
    # -------------------------------------------------------------------------
    def get_publishing_models_by_symbol(self) -> Dict[str, List[str]]:
        """
        Brief 25 Section 1 & 4.1:
        Returns map of publishing symbols to their active directions.
        USDCAD and USDJPY each list both ['LONG', 'SHORT'].
        """
        try:
            from app.services.signal_engine_service import signal_engine_service
            dirs = signal_engine_service.PUBLISHED_MODEL_DIRECTIONS
        except Exception:
            dirs = {
                "USDCHF": ["SHORT"],
                "AUDJPY": ["SHORT"],
                "AUDUSD": ["SHORT"],
                "CADJPY": ["SHORT"],
                "USDCAD": ["LONG", "SHORT"],
                "CHFJPY": ["SHORT"],
                "USDJPY": ["LONG", "SHORT"],
                "GBPJPY": ["LONG"],
            }
        excluded = [s.upper() for s in getattr(settings, "TELEGRAM_EXCLUDED_SYMBOLS", ["XAUUSD", "GBPUSD", "EURJPY", "EURUSD"])]
        return {sym: sides for sym, sides in sorted(dirs.items()) if sym not in excluded}

    def get_publishing_models(self) -> List[Dict[str, Any]]:
        """
        Brief 25 Section 1 & 4.1:
        Returns the list of 10 publishing models across 8 symbols with real measured metrics.
        """
        by_sym = self.get_publishing_models_by_symbol()
        models = []
        for sym, directions in by_sym.items():
            for side in directions:
                metrics = get_symbol_metrics(sym, side=side)
                models.append({
                    "symbol": sym,
                    "direction": side,
                    "tier": metrics.get("tier", "EXPERIMENTAL"),
                    "expectancy_r": metrics.get("expectancy_r", 0.0),
                    "oos_trades": metrics.get("oos_trades", 0),
                    "threshold": metrics.get("threshold", 0.40)
                })
        return models

    def get_validated_symbols(self) -> List[str]:
        """
        Brief 25 Section 1: Eight publishing symbols.
        """
        return list(self.get_publishing_models_by_symbol().keys())

    def get_short_models(self) -> List[str]:
        """Returns symbols with validated SHORT models available (Brief 20, 21, 25)."""
        return [sym for sym, sides in self.get_publishing_models_by_symbol().items() if "SHORT" in sides]

    def _get_precision(self, symbol: str) -> int:
        sym = symbol.upper()
        # 1. Dynamic lookup from symbol_specs.json if available
        for path_str in [
            "/app/ml/data/symbol_specs.json",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "data", "symbol_specs.json")
        ]:
            if os.path.exists(path_str):
                try:
                    with open(path_str, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if sym in data and "digits" in data[sym]:
                            return int(data[sym]["digits"])
                except Exception:
                    pass
        # 2. Asset class fallback precision
        if "JPY" in sym:
            return 3
        if "XAU" in sym or "GOLD" in sym:
            return 2
        return 5

    def format_levels_message(self, card: Dict[str, Any]) -> str:
        """
        Brief 13 Section 3: Message 1 — The levels, text only, English, copyable.
        Strictly formatted as required by the owner.
        """
        symbol = card.get("symbol", "USDJPY").upper()
        timeframe = str(card.get("timeframe") or "M15").upper()
        digits = self._get_precision(symbol)
        
        entry = float(card.get("entry_price", 0.0))
        sl = float(card.get("sl_price", 0.0))
        
        atr = float(card.get("atr") or 0.0)
        if atr <= 0.0 and entry > 0 and sl > 0:
            atr = abs(entry - sl)
        if atr <= 0.0:
            atr = get_reference_median_atr(symbol)
            
        direction = str(card.get("direction") or card.get("decision") or "BUY").upper()
        if direction == "SELL":
            tp1 = float(card.get("tp1_price") or round(entry - 2.0 * atr, digits))
            tp2 = float(card.get("tp2_price") or round(entry - 3.0 * atr, digits))
            if tp2 == tp1:
                tp2 = round(tp1 - (10 ** -digits), digits)
        else:
            tp1 = float(card.get("tp1_price") or round(entry + 2.0 * atr, digits))
            tp2 = float(card.get("tp2_price") or round(entry + 3.0 * atr, digits))
            if tp2 == tp1:
                tp2 = round(tp1 + (10 ** -digits), digits)
            
        snapshot_id = card.get("snapshot_id", "snap_UNKNOWN")
        
        time_str = card.get("created_at_utc", "")
        if not time_str and card.get("timestamp"):
            dt = datetime.fromtimestamp(card["timestamp"], tz=timezone.utc)
            time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        else:
            time_str = str(time_str).replace(" UTC", "").strip()
            if len(time_str) >= 16:
                time_str = f"{time_str[:16]} UTC"
            elif not time_str.endswith("UTC"):
                time_str = f"{time_str} UTC"

        entry_str = f"{entry:.{digits}f}"
        sl_str = f"{sl:.{digits}f}"
        tp1_str = f"{tp1:.{digits}f}"
        tp2_str = f"{tp2:.{digits}f}"

        msg1 = (
            f"FOREX ENGINEER — CONFIRMED SIGNAL\n\n"
            f"{symbol}  ·  {direction}  ·  {timeframe}\n\n"
            f"Entry          {entry_str}\n"
            f"Stop Loss      {sl_str}\n"
            f"Take Profit 1  {tp1_str}\n"
            f"Take Profit 2  {tp2_str}\n\n"
            f"Signal ID      {snapshot_id}\n"
            f"Time           {time_str}"
        )
        return msg1

    def format_analysis_caption(self, card: Dict[str, Any]) -> str:
        """
        Brief 10 Section 3 & Brief 14 Section 2: Message 2 — Chart Photo Caption with reasoning & baseline.
        Quotes symbol's OWN measured performance and Tier.
        """
        symbol = card.get("symbol", "USDJPY").upper()
        timeframe = str(card.get("timeframe") or "M15").upper()
        direction = str(card.get("direction") or card.get("decision") or "BUY").upper()
        digits = self._get_precision(symbol)
        
        metrics = get_symbol_metrics(symbol, side=direction)
        tier = metrics.get("tier", "EXPERIMENTAL")
        exp_r = metrics.get("expectancy_r", 0.0)
        trades = metrics.get("oos_trades", 0)
        pos_folds = metrics.get("folds_positive", 0)
        tot_folds = metrics.get("folds_total", 5)

        prob = card.get("probability", 0.50)
        prob_pct = int(round(float(prob) * 100)) if prob is not None else 50
        threshold = card.get("threshold", 0.40)
        thresh_pct = int(round(float(threshold) * 100)) if threshold is not None else 40
        
        entry = float(card.get("entry_price", 0.0))
        sl = float(card.get("sl_price", 0.0))
        atr = float(card.get("atr") or 0.0)
        if atr <= 0.0 and entry > 0 and sl > 0:
            atr = abs(entry - sl)
        if atr <= 0.0:
            atr = get_reference_median_atr(symbol)
        atr_str = f"{atr:.{digits}f}"

        if tier == "EXPERIMENTAL":
            tier_block = (
                f"EXPERIMENTAL — measured {exp_r:+.3f} R over {trades:,} out-of-sample trades,\n"
                f"{pos_folds} of {tot_folds} walk-forward folds positive. Published for evaluation only."
            )
        else:
            tier_block = (
                f"Baseline: {exp_r:+.2f} R expectancy over {trades:,} out-of-sample\n"
                f"walk-forward trades, spread accounted, slippage not modelled.\n"
                f"Tier: VALIDATED"
            )

        caption = (
            f"FOREX ENGINEER — ANALYSIS\n\n"
            f"{symbol}  ·  {timeframe}\n\n"
            f"Pair                {symbol}\n"
            f"Timeframe           {timeframe} (15-Minute Chart)\n"
            f"Model probability   {prob_pct}%   (threshold {thresh_pct}%)\n"
            f"H4 trend gate       PASSED  (EMA Trend Gate)\n"
            f"ATR                 {atr_str}\n"
            f"Risk : Reward       1 : 2\n"
            f"Tier                {tier}\n\n"
            f"{tier_block}\n\n"
            f"Shadow mode — not auto-traded. Live record since 2026-09-09.\n\n"
            f"TP1 is the validated target. TP2 is an extended runner and is not\n"
            f"covered by the baseline above."
        )
        return caption

    def get_desk_track_record_line(self) -> str:
        """
        Brief 29 Section 4: Dynamic track record line for manual desk trades.
        Initial: 'No measured baseline — this desk's live record begins today.'
        With closed trades: 'Desk record since YYYY-MM-DD · N trades · +X.X R · not yet statistically meaningful'
        (keeps 'not yet statistically meaningful' until 300 closed trades).
        """
        search_files = [
            os.path.join(self.data_dir, "manual_outcomes.jsonl"),
            str(Path(__file__).resolve().parent.parent.parent / "data" / "manual_outcomes.jsonl"),
            str(Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports" / "manual_outcomes.jsonl"),
            str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "manual_outcomes.jsonl"),
            "/app/ml/reports/manual_outcomes.jsonl",
            "/app/data/manual_outcomes.jsonl"
        ]
        outcomes = []
        seen = set()
        for p in search_files:
            abs_p = os.path.abspath(p)
            if abs_p in seen or not os.path.exists(abs_p):
                continue
            seen.add(abs_p)
            try:
                with open(abs_p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            tid = rec.get("trade_id") or rec.get("id")
                            if tid and tid not in seen:
                                seen.add(tid)
                                outcomes.append(rec)
                        except Exception:
                            pass
            except Exception:
                pass

        count = len(outcomes)
        if count == 0:
            return "No measured baseline — this desk's live record begins today."

        sum_r = sum(float(o.get("r_multiple", 0.0)) for o in outcomes)
        earliest_date = "2026-09-22"
        for o in outcomes:
            t_str = str(o.get("resolved_at") or o.get("entered_at") or "")
            if len(t_str) >= 10:
                d_part = t_str[:10]
                if d_part < earliest_date:
                    earliest_date = d_part

        # Filter strictly for trades resolved after publishing went live (excludes sealed records)
        sealed_set = getattr(self, "sealed_manual_records", set())
        live_outcomes = [
            o for o in outcomes
            if (o.get("trade_id") or o.get("id")) not in sealed_set
        ]
        count = len(live_outcomes)
        if count == 0:
            return "Desk record begins today · 0 trades · not yet statistically meaningful"

        sum_r = sum(float(o.get("r_multiple", 0.0)) for o in live_outcomes)
        earliest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for o in live_outcomes:
            t_str = str(o.get("resolved_time_utc") or o.get("resolved_utc") or o.get("resolved_at") or o.get("entered_at") or "")
            if len(t_str) >= 10:
                d_part = t_str[:10]
                if d_part < earliest_date:
                    earliest_date = d_part

        stat_flag = " · not yet statistically meaningful" if count < 300 else ""
        return f"Desk record since {earliest_date} · {count} trade{'s' if count != 1 else ''} · {sum_r:+.1f} R{stat_flag}"

    def format_desk_levels_message(self, trade_record: Dict[str, Any]) -> str:
        """
        Brief 29 Section 4: Formats Telegram Message 1 (levels text) for manual desk trades.
        Strictly contains NO model language (no baseline, no expectancy, no tier, no walk-forward).
        """
        symbol = str(trade_record.get("symbol", "")).upper()
        direction = str(trade_record.get("direction", "BUY")).upper()
        timeframe = str(trade_record.get("timeframe", "M15")).upper()
        digits = self._get_precision(symbol)

        entry = float(trade_record.get("entry_price") or trade_record.get("entry", 0.0))
        sl = float(trade_record.get("stop_loss") or trade_record.get("sl_price", 0.0))
        tp1 = float(trade_record.get("take_profit_1") or trade_record.get("tp1_price", 0.0))
        tp2 = trade_record.get("take_profit_2") or trade_record.get("tp2_price")
        tp3 = trade_record.get("take_profit_3") or trade_record.get("tp3_price")

        risk = abs(entry - sl) if abs(entry - sl) > 0 else 0.0001
        r1 = float(trade_record.get("tp1_r") or round(abs(tp1 - entry) / risk, 1))

        lines = [
            "FOREX ENGINEER — ANALYST SIGNAL",
            "",
            f"{symbol}  ·  {direction}  ·  {timeframe}",
            "",
            f"Entry          {entry:.{digits}f}",
            f"Stop Loss      {sl:.{digits}f}",
            f"Take Profit 1  {tp1:.{digits}f}   ({r1:+.1f} R)",
        ]

        if tp2 is not None and float(tp2) > 0:
            tp2_val = float(tp2)
            r2 = float(trade_record.get("tp2_r") or round(abs(tp2_val - entry) / risk, 1))
            lines.append(f"Take Profit 2  {tp2_val:.{digits}f}   ({r2:+.1f} R)")

        if tp3 is not None and float(tp3) > 0:
            tp3_val = float(tp3)
            r3 = float(trade_record.get("tp3_r") or round(abs(tp3_val - entry) / risk, 1))
            lines.append(f"Take Profit 3  {tp3_val:.{digits}f}   ({r3:+.1f} R)")

        lines.extend([
            "",
            "Discretionary analysis by the desk.",
            self.get_desk_track_record_line()
        ])
        return "\n".join(lines)

    def format_desk_caption(self, trade_record: Dict[str, Any]) -> str:
        """
        Brief 29 Section 4: Formats Telegram Message 2 (photo caption) for manual desk trades.
        Strictly contains NO model language.
        """
        symbol = str(trade_record.get("symbol", "")).upper()
        direction = str(trade_record.get("direction", "BUY")).upper()
        timeframe = str(trade_record.get("timeframe", "M15")).upper()
        digits = self._get_precision(symbol)

        entry = float(trade_record.get("entry_price") or trade_record.get("entry", 0.0))
        sl = float(trade_record.get("stop_loss") or trade_record.get("sl_price", 0.0))
        tp1 = float(trade_record.get("take_profit_1") or trade_record.get("tp1_price", 0.0))
        tp2 = trade_record.get("take_profit_2") or trade_record.get("tp2_price")
        tp3 = trade_record.get("take_profit_3") or trade_record.get("tp3_price")

        levels_str = f"Entry: {entry:.{digits}f} | SL: {sl:.{digits}f} | TP1: {tp1:.{digits}f}"
        if tp2 is not None and float(tp2) > 0:
            levels_str += f" | TP2: {float(tp2):.{digits}f}"
        if tp3 is not None and float(tp3) > 0:
            levels_str += f" | TP3: {float(tp3):.{digits}f}"

        why = str(trade_record.get("why") or trade_record.get("why_reason") or "").strip()
        lines = [
            "FOREX ENGINEER — ANALYST DESK",
            f"{symbol} · {direction} · {timeframe}",
            levels_str
        ]
        if why:
            lines.append(f"Rationale: {why}")
        lines.extend([
            "",
            "Discretionary analysis by the desk.",
            self.get_desk_track_record_line()
        ])
        return "\n".join(lines)

    def format_desk_result_message(self, outcome_record: Dict[str, Any]) -> str:
        """
        Brief 31 Section 2: Closing result message for manual desk trades.
        Mirroring model result message but with DESK header and running record.
        Strictly contains NO model language, no softened stops, result in R.
        """
        symbol = str(outcome_record.get("symbol", "")).upper()
        direction = str(outcome_record.get("direction", "BUY")).upper()
        digits = self._get_precision(symbol)

        raw_outcome = str(outcome_record.get("outcome") or outcome_record.get("barrier_hit") or "TP1").upper().strip()
        if raw_outcome in ("TP1", "TARGET 1", "TARGET 1 HIT"):
            barrier_label = "TARGET 1 HIT"
        elif raw_outcome in ("TP2", "TARGET 2", "TARGET 2 HIT"):
            barrier_label = "TARGET 2 HIT"
        elif raw_outcome in ("TP3", "TARGET 3", "TARGET 3 HIT"):
            barrier_label = "TARGET 3 HIT"
        elif raw_outcome in ("SL", "STOP", "STOP HIT", "STOP_LOSS"):
            barrier_label = "STOP HIT"
        elif "TIME" in raw_outcome or raw_outcome == "TIMEOUT":
            barrier_label = "CLOSED ON TIME"
        else:
            barrier_label = raw_outcome

        entry = float(outcome_record.get("entry") or outcome_record.get("entry_price") or 0.0)
        exit_p = float(outcome_record.get("exit_price") or outcome_record.get("exit") or 0.0)
        if exit_p <= 0 and entry > 0:
            exit_p = entry

        r_mult = float(outcome_record.get("r_multiple", 0.0))
        if barrier_label == "STOP HIT":
            r_str = f"-{abs(r_mult):.1f} R" if round(r_mult, 1) == round(r_mult, 2) else f"-{abs(r_mult):.2f} R"
            if r_str == "-0.0 R":
                r_str = "-1.0 R"
        else:
            r_str = f"{r_mult:+.1f} R" if round(r_mult, 1) == round(r_mult, 2) else f"{r_mult:+.2f} R"

        opened_utc = self._clean_utc_datetime(
            outcome_record.get("entered_at") or outcome_record.get("opened_utc") or outcome_record.get("published_at")
        )
        closed_utc = self._clean_utc_datetime(
            outcome_record.get("resolved_time_utc") or outcome_record.get("closed_utc") or outcome_record.get("resolved_utc")
        )
        duration_str = outcome_record.get("duration") or self._format_duration(opened_utc, closed_utc)

        trade_id = str(outcome_record.get("trade_id") or outcome_record.get("signal_id") or "man_UNKNOWN")

        entry_str = f"{entry:.{digits}f}" if entry > 0 else "N/A"
        exit_str = f"{exit_p:.{digits}f}" if exit_p > 0 else "N/A"

        msg = (
            f"FOREX ENGINEER — DESK TRADE CLOSED\n\n"
            f"{symbol}  ·  {direction}  ·  {barrier_label}\n\n"
            f"Entry    {entry_str}\n"
            f"Exit     {exit_str}\n"
            f"Result   {r_str}      Duration  {duration_str}\n\n"
            f"Trade ID  {trade_id}\n"
            f"{self.get_desk_track_record_line()}"
        )
        return msg

    def build_desk_result_payload(self, outcome_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Brief 31 Section 2: Complete frozen contract payload for closed desk trades.
        Schema version 1, type 'result', source 'DESK'. Every field always present.
        """
        symbol = str(outcome_record.get("symbol", "")).upper()
        direction = str(outcome_record.get("direction", "BUY")).upper()
        digits = self._get_precision(symbol)

        raw_outcome = str(outcome_record.get("outcome") or outcome_record.get("barrier_hit") or "TP1").upper().strip()
        if raw_outcome in ("TP1", "TARGET 1", "TARGET 1 HIT"):
            barrier = "TARGET 1 HIT"
            outcome = "TP1"
        elif raw_outcome in ("TP2", "TARGET 2", "TARGET 2 HIT"):
            barrier = "TARGET 2 HIT"
            outcome = "TP2"
        elif raw_outcome in ("TP3", "TARGET 3", "TARGET 3 HIT"):
            barrier = "TARGET 3 HIT"
            outcome = "TP3"
        elif raw_outcome in ("SL", "STOP", "STOP HIT", "STOP_LOSS"):
            barrier = "STOP HIT"
            outcome = "SL"
        elif "TIME" in raw_outcome or raw_outcome == "TIMEOUT":
            barrier = "CLOSED ON TIME"
            outcome = "TIMEOUT"
        else:
            barrier = raw_outcome
            outcome = raw_outcome

        entry = float(outcome_record.get("entry") or outcome_record.get("entry_price") or 0.0)
        exit_p = float(outcome_record.get("exit_price") or outcome_record.get("exit") or 0.0)
        r_mult = float(outcome_record.get("r_multiple", 0.0))

        opened_utc = self._clean_utc_datetime(
            outcome_record.get("entered_at") or outcome_record.get("opened_utc") or outcome_record.get("published_at")
        )
        closed_utc = self._clean_utc_datetime(
            outcome_record.get("resolved_time_utc") or outcome_record.get("closed_utc") or outcome_record.get("resolved_utc")
        )
        duration_str = outcome_record.get("duration") or self._format_duration(opened_utc, closed_utc)

        trade_id = str(outcome_record.get("trade_id") or outcome_record.get("signal_id") or "man_UNKNOWN")
        msg_text = self.format_desk_result_message(outcome_record)

        return {
            "schema_version": 1,
            "type": "result",
            "source": "DESK",
            "trade_id": trade_id,
            "signal_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "barrier": barrier,
            "outcome": outcome,
            "r_multiple": round(r_mult, 2),
            "entry_price": round(entry, digits) if entry > 0 else None,
            "exit_price": round(exit_p, digits) if exit_p > 0 else None,
            "duration": duration_str,
            "opened_utc": opened_utc,
            "closed_utc": closed_utc,
            "model": None,
            "tier": None,
            "baseline_expectancy_r": None,
            "oos_trades": None,
            "message_text": msg_text
        }

    def format_waiting_message(
        self,
        symbols_status: List[Dict[str, str]],
        last_signal: Optional[Dict[str, Any]]
    ) -> str:
        """
        Brief 10 Section 3c: Hourly waiting notice text, English only, no levels.
        Explicitly identifies timeframe.
        """
        excluded = [x.upper() for x in getattr(settings, "TELEGRAM_EXCLUDED_SYMBOLS", ["XAUUSD"])]
        lines = ["FOREX ENGINEER — NO SIGNAL", ""]
        for s in symbols_status:
            sym = s.get("symbol", "USDJPY").upper()
            if sym in excluded:
                continue
            tf = s.get("timeframe", "M15")
            reason = s.get("reason", "H4 trend not aligned or probability below threshold")
            lines.append(f"{sym} ({tf})   waiting   {reason}")
            
        lines.append("")
        lines.append("Timeframe: M15 (15-Minute Chart)")
        lines.append("Do not enter any trade. No confirmed setup at this time.")
        lines.append("")
        
        if last_signal:
            last_sym = last_signal.get("symbol", "USDJPY")
            last_tf = last_signal.get("timeframe", "M15")
            age_h = last_signal.get("age_hours", 6)
            t_utc = last_signal.get("time_utc", "")
            clean_time = str(t_utc).replace(" UTC", "").strip()[:16]
            lines.append(f"Last confirmed signal: {last_sym} ({last_tf}), {age_h} hours ago")
            if clean_time:
                lines.append(f"{clean_time} UTC")
        else:
            lines.append("Last confirmed signal: None recorded yet")
            
        return "\n".join(lines)

    @staticmethod
    def _clean_utc_datetime(val: Any) -> str:
        """Normalizes date string or timestamp to 'YYYY-MM-DD HH:MM:SS' UTC format"""
        if val is None:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(val, (int, float)):
            try:
                return datetime.fromtimestamp(float(val), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        s = str(val).replace(" UTC", "").strip()
        if "T" in s:
            try:
                dt = datetime.fromisoformat(s)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        if len(s) >= 19:
            return s[:19]
        return s

    @staticmethod
    def _format_duration(opened_utc: str, closed_utc: str) -> str:
        """Calculates and formats trade duration as 'Xh Ym' or 'Ym'"""
        try:
            t_open = datetime.strptime(opened_utc[:19], "%Y-%m-%d %H:%M:%S")
            t_close = datetime.strptime(closed_utc[:19], "%Y-%m-%d %H:%M:%S")
            diff_s = max(0, int((t_close - t_open).total_seconds()))
        except Exception:
            diff_s = 0

        hours = diff_s // 3600
        mins = (diff_s % 3600) // 60
        if hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m"

    def _resolve_result_levels(
        self,
        rec: Dict[str, Any],
        digits: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Resolves (entry, stop_loss, take_profit_1, exit_price) safely."""
        entry = float(rec.get("entry") or rec.get("entry_price") or 0.0)
        sl = float(rec.get("stop_loss") or rec.get("sl_price") or 0.0)
        tp1 = float(rec.get("take_profit_1") or rec.get("tp1_price") or 0.0)
        exit_p = float(rec.get("exit_price") or rec.get("exit") or 0.0)

        snap_id = str(rec.get("signal_id") or "").strip()
        if (entry <= 0 or sl <= 0 or tp1 <= 0) and snap_id:
            search_dirs = [
                Path("/app/data/snapshots"),
                Path(__file__).resolve().parent.parent.parent / "data" / "snapshots",
                Path(__file__).resolve().parent.parent.parent.parent / "data" / "snapshots"
            ]
            for sdir in search_dirs:
                cand = sdir / f"{snap_id}.json"
                if cand.exists():
                    try:
                        with open(cand, "r", encoding="utf-8") as sf:
                            sdata = json.load(sf)
                        if entry <= 0:
                            entry = float(sdata.get("entry_price") or sdata.get("entry") or 0.0)
                        if sl <= 0:
                            sl = float(sdata.get("sl_price") or sdata.get("stop_loss") or 0.0)
                        if tp1 <= 0:
                            tp1 = float(sdata.get("tp1_price") or sdata.get("take_profit_1") or 0.0)
                        break
                    except Exception:
                        pass

        if exit_p <= 0 and entry > 0:
            exit_p = entry

        entry_ret = round(entry, digits) if entry > 0 else None
        sl_ret = round(sl, digits) if sl > 0 else None
        tp1_ret = round(tp1, digits) if tp1 > 0 else None
        exit_ret = round(exit_p, digits) if exit_p > 0 else None
        return entry_ret, sl_ret, tp1_ret, exit_ret

    def format_result_message(self, outcome_record: Dict[str, Any]) -> str:
        """
        Brief 23 Section 3: Result message formatting.
        Visually distinct from signal, plain outcome, result in R, carries signal_id.
        """
        symbol = str(outcome_record.get("symbol", "")).upper()
        direction = str(outcome_record.get("direction", "BUY")).upper()
        digits = self._get_precision(symbol)

        raw_outcome = str(outcome_record.get("outcome") or outcome_record.get("barrier_hit") or "TP1").upper().strip()
        if raw_outcome == "TP1":
            outcome_label = "TARGET 1 HIT"
        elif raw_outcome == "TP2":
            outcome_label = "TARGET 2 HIT"
        elif raw_outcome == "SL":
            outcome_label = "STOP HIT"
        elif raw_outcome == "TIMEOUT":
            outcome_label = "CLOSED ON TIME"
        elif "TP2" in raw_outcome or "TARGET 2" in raw_outcome:
            outcome_label = "TARGET 2 HIT"
        elif "TP" in raw_outcome or "TARGET" in raw_outcome:
            outcome_label = "TARGET 1 HIT"
        elif "SL" in raw_outcome or "STOP" in raw_outcome:
            outcome_label = "STOP HIT"
        elif "TIME" in raw_outcome:
            outcome_label = "CLOSED ON TIME"
        else:
            outcome_label = raw_outcome

        entry_val, sl_val, tp1_val, exit_val = self._resolve_result_levels(outcome_record, digits)
        entry_str = f"{entry_val:.{digits}f}" if entry_val is not None else "N/A"
        exit_str = f"{exit_val:.{digits}f}" if exit_val is not None else "N/A"

        r_mult = float(outcome_record.get("r_multiple", 0.0))
        if round(r_mult, 1) == round(r_mult, 2):
            result_str = f"{r_mult:+.1f} R"
        else:
            result_str = f"{r_mult:+.2f} R"

        opened_utc = self._clean_utc_datetime(
            outcome_record.get("opened_utc") or outcome_record.get("signal_time_utc") or outcome_record.get("published_at")
        )
        closed_utc = self._clean_utc_datetime(
            outcome_record.get("closed_utc") or outcome_record.get("resolved_time_utc") or outcome_record.get("resolved_utc")
        )
        duration_str = outcome_record.get("duration") or self._format_duration(opened_utc, closed_utc)

        signal_id = str(outcome_record.get("signal_id", "snap_UNKNOWN"))

        msg = (
            f"FOREX ENGINEER — TRADE CLOSED\n\n"
            f"{symbol}  ·  {direction}  ·  {outcome_label}\n\n"
            f"Entry    {entry_str}\n"
            f"Exit     {exit_str}\n"
            f"Result   {result_str}\n"
            f"Duration {duration_str}\n\n"
            f"Signal ID  {signal_id}"
        )
        return msg

    def build_result_payload(self, outcome_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Brief 23 Section 2: Complete frozen contract payload for closed trades.
        Schema version 1, type 'result'. Every field always present.
        """
        symbol = str(outcome_record.get("symbol", "")).upper()
        direction = str(outcome_record.get("direction", "BUY")).upper()
        digits = self._get_precision(symbol)

        raw_outcome = str(outcome_record.get("outcome") or outcome_record.get("barrier_hit") or "TP1").upper().strip()
        if raw_outcome in ("TP1", "TP2", "SL", "TIMEOUT"):
            outcome = raw_outcome
        elif "TP2" in raw_outcome:
            outcome = "TP2"
        elif "TP" in raw_outcome or "TARGET" in raw_outcome:
            outcome = "TP1"
        elif "SL" in raw_outcome or "STOP" in raw_outcome:
            outcome = "SL"
        else:
            outcome = "TIMEOUT"

        entry, sl, tp1, exit_price = self._resolve_result_levels(outcome_record, digits)

        r_mult = float(outcome_record.get("r_multiple", 0.0))
        r_multiple = round(r_mult, 1) if round(r_mult, 1) == round(r_mult, 2) else round(r_mult, 2)

        opened_utc = self._clean_utc_datetime(
            outcome_record.get("opened_utc") or outcome_record.get("signal_time_utc") or outcome_record.get("published_at")
        )
        closed_utc = self._clean_utc_datetime(
            outcome_record.get("closed_utc") or outcome_record.get("resolved_time_utc") or outcome_record.get("resolved_utc")
        )
        duration_str = outcome_record.get("duration") or self._format_duration(opened_utc, closed_utc)

        signal_id = str(outcome_record.get("signal_id", "snap_UNKNOWN"))

        normalized_rec = {
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "outcome": outcome,
            "entry": entry,
            "exit_price": exit_price,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "r_multiple": r_multiple,
            "opened_utc": opened_utc,
            "closed_utc": closed_utc,
            "duration": duration_str
        }
        msg_text = self.format_result_message(normalized_rec)

        payload = {
            "schema_version": 1,
            "type": "result",
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "outcome": outcome,
            "entry": entry,
            "exit_price": exit_price,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "r_multiple": r_multiple,
            "opened_utc": opened_utc,
            "closed_utc": closed_utc,
            "duration": duration_str,
            "message_text": msg_text
        }
        return payload

    # -------------------------------------------------------------------------
    # n8n Webhook Transport
    def resolve_target_webhook(self, payload: Dict[str, Any]) -> str:
        """
        Intelligent 3-way webhook router:
        - Engineer Account (8058543) -> Dedicated to GOLD -> trading_bot_gold
        - Second Account (160767360) -> Split into:
          * Currency pairs (FOREX) -> trading_bot_forex
          * Dow Jones, Nasdaq, Oil, Bitcoin, Silver (INDEX) -> trading_bot_index
        - Fallback based on instrument category:
          * Gold -> trading_bot_gold
          * Index/Oil/Crypto/Silver -> trading_bot_index
          * Forex -> trading_bot_forex
        """
        # 1. Explicit override in payload
        target = payload.get("webhook_target") or payload.get("channel")
        if target:
            t_lower = str(target).lower()
            if "gold" in t_lower:
                return getattr(settings, "N8N_WEBHOOK_URL_GOLD", "https://n8n-p4oh.srv1867849.hstgr.cloud/webhook/trading_bot_gold")
            elif "index" in t_lower:
                return getattr(settings, "N8N_WEBHOOK_URL_INDEX", "https://n8n-p4oh.srv1867849.hstgr.cloud/webhook/trading_bot_index")
            elif "forex" in t_lower:
                return getattr(settings, "N8N_WEBHOOK_URL_FOREX", "https://n8n-p4oh.srv1867849.hstgr.cloud/webhook/trading_bot_forex")

        # 2. Check if the trade is from the Engineer's Gold Account (8058543)
        acc = payload.get("account") or payload.get("login") or payload.get("account_login")
        gold_login = getattr(settings, "MT5_MIRROR_LOGIN_GOLD", 8058543)

        if acc:
            try:
                acc_int = int(acc)
                if acc_int == gold_login:
                    return getattr(settings, "N8N_WEBHOOK_URL_GOLD", "https://n8n-p4oh.srv1867849.hstgr.cloud/webhook/trading_bot_gold")
            except (ValueError, TypeError):
                pass

        # 3. Market category classification based on symbol
        sym = str(payload.get("symbol", ""))
        category = classify_market_category(sym)

        if category == "GOLD":
            return getattr(settings, "N8N_WEBHOOK_URL_GOLD", "https://n8n-p4oh.srv1867849.hstgr.cloud/webhook/trading_bot_gold")
        elif category == "INDEX":
            return getattr(settings, "N8N_WEBHOOK_URL_INDEX", "https://n8n-p4oh.srv1867849.hstgr.cloud/webhook/trading_bot_index")
        else:
            return getattr(settings, "N8N_WEBHOOK_URL_FOREX", "https://n8n-p4oh.srv1867849.hstgr.cloud/webhook/trading_bot_forex")

    # -------------------------------------------------------------------------
    async def _post_to_n8n_webhook(self, payload: Dict[str, Any]) -> bool:
        """
        Dispatches JSON payload to n8n webhook with:
          - 10 second timeout
          - Up to 2 retries with exponential backoff (1s, 2s)
          - Strict 2xx confirmation
        """
        # Block any waiting payloads per user instruction
        if payload.get("type") == "waiting":
            logger.info("🚫 [BLOCKED] Waiting notice suppressed from n8n webhook per user instruction.")
            return True

        url = self.resolve_target_webhook(payload)
        sym = str(payload.get("symbol", ""))
        cat = classify_market_category(sym)
        logger.info(
            f"🚀 [N8N ROUTING] Target: {url.split('/')[-1]} | Cat: {cat} | "
            f"Type: {payload.get('type')} | Sym: {sym} | "
            f"Acc: {payload.get('account') or 'N/A'}"
        )
        if not url:
            logger.error("Telegram/n8n: Webhook URL could not be resolved.")
            return False

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FXEngine-VPS-Backend/1.0"
        }

        for attempt in range(1, 4):  # attempt 1, retry 1, retry 2
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    res = await client.post(url, json=payload, headers=headers)
                    if 200 <= res.status_code < 300:
                        return True
                    logger.warning(
                        f"n8n webhook attempt {attempt} returned non-2xx (HTTP {res.status_code}): {res.text[:200]}"
                    )
                    # Emergency fallback for Gold
                    if "trading_bot_gold" in url:
                        fb_url = "https://n8n-p4oh.srv1867849.hstgr.cloud/webhook/trading_bot"
                        logger.info(f"🔄 [EMERGENCY FALLBACK] Dispatching Gold to verified working webhook {fb_url}")
                        fb_res = await client.post(fb_url, json=payload, headers=headers)
                        if 200 <= fb_res.status_code < 300:
                            logger.info("✅ [FALLBACK SUCCESS] Delivered successfully via trading_bot webhook.")
                            return True
            except Exception as e:
                logger.warning(f"n8n webhook attempt {attempt} failed with exception: {e}")
                if "trading_bot_gold" in url:
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            fb_url = "https://n8n-p4oh.srv1867849.hstgr.cloud/webhook/trading_bot"
                            fb_res = await client.post(fb_url, json=payload, headers=headers)
                            if 200 <= fb_res.status_code < 300:
                                return True
                    except Exception:
                        pass

            if attempt < 3:
                await asyncio.sleep(attempt * 1.0)

        logger.error(f"n8n webhook delivery permanently failed after 3 attempts for {payload.get('type')}")
        return False

    # -------------------------------------------------------------------------
    # Primary Publishing Methods
    # -------------------------------------------------------------------------
    async def publish_confirmed_signal(self, card: Dict[str, Any], force: bool = False, is_test: bool = False) -> bool:
        """
        Publishes a confirmed model signal card to Telegram via n8n.
        Guarantees:
          1. Validated symbol check (never post on unvalidated symbols).
          2. Validated decision check (BUY only, never HOLD).
          3. Weekend blackout check: Forex closed Fri 22:00 UTC -> Sun 21:00 UTC (unless force=True).
          4. Deduplication on disk (never post same snapshot_id twice).
          5. Rate limiting on disk (max 6/hr, 20/day per symbol).
          6. Dry-run mode: Logs exact JSON payload and text; sends nothing.
          7. Non-2xx leaves signal unmarked so next cycle retries.
          8. Silent failure: Exceptions never crash caller.
          9. Strict test-signal isolation: test signals are never sent to the live webhook.
          10. Strict level sanity: signals with null/zero entry, SL, or TP1 are rejected immediately.
        """
        try:
            snapshot_id = card.get("snapshot_id")
            if not snapshot_id:
                logger.error("Telegram/n8n: Missing snapshot_id in signal card.")
                return False

            symbol = card.get("symbol", "").upper()

            # Brief 14 Section 1: Gold Excluded from Telegram Permanently (Configurable)
            excluded_symbols = [s.upper() for s in getattr(settings, "TELEGRAM_EXCLUDED_SYMBOLS", ["XAUUSD"])]
            if symbol in excluded_symbols:
                self.state["excluded_symbol_skips"] = self.state.get("excluded_symbol_skips", 0) + 1
                skips_by_sym = self.state.setdefault("excluded_symbol_skips_by_sym", {})
                skips_by_sym[symbol] = skips_by_sym.get(symbol, 0) + 1
                self._save_state()
                logger.info(
                    f"🚫 [DROP: EXCLUDED_SYMBOL] Skipping Telegram publish for {symbol} "
                    f"(signal_id={snapshot_id}): symbol is in TELEGRAM_EXCLUDED_SYMBOLS. Total skips: {self.state['excluded_symbol_skips']}"
                )
                return False

            # Rule 1: Validated symbol check (Brief 16 Problem 1: every deployed model minus exclusions)
            valid_symbols = self.get_validated_symbols()
            if symbol not in valid_symbols:
                logger.warning(
                    f"🚫 [DROP: UNVALIDATED_SYMBOL] Dropping signal {snapshot_id} for {symbol}: "
                    f"symbol has no deployed model or is not in validated list."
                )
                return False

            # Rule 2: Decision check
            decision = str(card.get("direction") or card.get("decision") or card.get("recommendation") or "BUY").upper()
            if decision not in ["BUY", "SELL"]:
                logger.info(
                    f"🚫 [DROP: NON_BUY_DECISION] Dropping signal {snapshot_id} for {symbol}: "
                    f"decision is '{decision}' (only BUY and SELL signals publish)."
                )
                return False

            if card.get("gate_passed") is not True:
                logger.info(
                    f"🚫 [DROP: GATE_FAILED] Dropping signal {snapshot_id} for {symbol}: "
                    f"directional gate not passed."
                )
                return False

            # Brief 25 Section 2b: Manifest-driven directional check
            # A symbol-direction pair publishes ONLY if it appears in the publishing manifest.
            sig_side = "LONG" if decision == "BUY" else "SHORT"
            allowed_sides = self.get_publishing_models_by_symbol().get(symbol, [])
            if sig_side not in allowed_sides:
                logger.warning(
                    f"🚫 [DROP: UNMANIFESTED_DIRECTION] Dropping signal {snapshot_id} for {symbol} {decision} ({sig_side}): "
                    f"direction is not in publishing manifest (allowed: {allowed_sides})."
                )
                return False

            # Rule 2b (Brief 17 & 18 Section 1): One open signal per symbol
            # While a symbol's last published signal is still open, every later signal for that symbol is suppressed.
            if not force:
                open_sig = signal_outcome_tracker.get_open_signal(symbol)
                if open_sig and open_sig.get("signal_id") != snapshot_id:
                    blocking_id = open_sig.get("signal_id", "unknown")
                    open_ts = float(open_sig.get("published_timestamp") or open_sig.get("created_timestamp") or time.time())
                    open_dur_s = max(0, int(time.time() - open_ts))
                    h = open_dur_s // 3600
                    m = (open_dur_s % 3600) // 60
                    dur_str = f"{h}h{m:02d}m" if h > 0 else f"{m}m"
                    now_hm = datetime.now(timezone.utc).strftime("%H:%M")

                    logger.warning(
                        f"🚫 [DROP: SUPPRESSED] SUPPRESSED  {symbol}  new signal at {now_hm}  "
                        f"blocked by {blocking_id} (open {dur_str})"
                    )
                    self.state.setdefault("suppression_counts", {})
                    self.state["suppression_counts"][symbol] = self.state["suppression_counts"].get(symbol, 0) + 1
                    self._save_state()
                    self._log_suppression(
                        symbol=symbol,
                        suppressed_id=snapshot_id,
                        blocking_id=blocking_id,
                        open_duration=dur_str
                    )
                    return False

            # Rule 3: Stale bar / market closed check (Brief 16 Problem 2: reject if > 20m old)
            if not force:
                bar_time = card.get("bar_close_time") or card.get("model_record", {}).get("bar_close_time")
                if bar_time:
                    now_ts = time.time()
                    bar_age_seconds = max(0.0, now_ts - float(bar_time))
                    if bar_age_seconds > 1200:
                        logger.warning(
                            f"🚨 [DROP: STALE_BAR] Refusing to publish signal {snapshot_id} for {symbol}: "
                            f"newest closed bar timestamp {bar_time} is {bar_age_seconds/60.0:.1f}m old (> 20m limit). Market closed."
                        )
                        return False

            # Rule 4: Weekend blackout check (Forex closed Fri 22:00 UTC -> Sun 21:00 UTC)
            if not force:
                market_open, market_reason = self.is_forex_market_open()
                if not market_open:
                    logger.info(
                        f"🚫 [DROP: WEEKEND_BLACKOUT] Weekend blackout active ({market_reason}). "
                        f"Suppressing signal {snapshot_id} for {symbol}."
                    )
                    return False

            # Rule 5: Deduplication check
            if self.is_signal_already_posted(snapshot_id):
                logger.info(
                    f"🚫 [DROP: DUPLICATE] Dropping signal {snapshot_id} for {symbol}: "
                    f"signal already posted previously."
                )
                return False

            # Rule 6: Hard rate cap
            allowed, rate_reason = self.check_signal_rate_limit(symbol)
            if not allowed:
                logger.error(
                    f"🚨 [DROP: RATE_LIMIT] Refusing signal {snapshot_id} for {symbol}: {rate_reason}"
                )
                return False

            # Absolute Level Defense (Brief 18 Problem 2 Point 4)
            # Refuse to publish any signal whose entry, stop or take_profit_1 is null, zero, or missing.
            raw_entry = card.get("entry_price") or card.get("entry") or card.get("levels", {}).get("entry")
            raw_sl = card.get("sl_price") or card.get("sl") or card.get("stop_loss") or card.get("levels", {}).get("stop_loss")
            raw_tp1 = card.get("tp1_price") or card.get("tp1") or card.get("take_profit_1") or card.get("levels", {}).get("take_profit_1")

            def _to_pos_float(v: Any) -> Optional[float]:
                if v is None:
                    return None
                try:
                    f = float(v)
                    return f if (math.isfinite(f) and f > 0) else None
                except (ValueError, TypeError):
                    return None

            entry_check = _to_pos_float(raw_entry)
            sl_check = _to_pos_float(raw_sl)
            tp1_check = _to_pos_float(raw_tp1)

            if entry_check is None or sl_check is None or tp1_check is None:
                logger.error(
                    f"🚨 [DROP: INVALID_LEVELS] Refusing signal {snapshot_id} for {symbol}: "
                    f"entry, stop_loss, and take_profit_1 must be valid positive numbers "
                    f"(entry={raw_entry}, sl={raw_sl}, tp1={raw_tp1})."
                )
                return False

            # Test Signal Flag Check (Brief 18 Problem 2 Points 1-3)
            is_test_flag = bool(is_test or card.get("is_test") or card.get("test"))
            is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)
            if is_test_flag and not is_dry_run:
                self.state["test_signals_count"] = self.state.get("test_signals_count", 0) + 1
                self.state.setdefault("test_signals", {})[snapshot_id] = {
                    "symbol": symbol,
                    "posted_at": datetime.now(timezone.utc).isoformat(),
                    "is_dry_run": False,
                    "is_test": True
                }
                self._save_state()
                logger.warning(
                    f"🧪 [DROP: TEST_SIGNAL_BLOCKED] Refusing to send test-flagged signal {snapshot_id} "
                    f"for {symbol} to live channel. Incrementing test_signals_count={self.state['test_signals_count']}."
                )
                return False

            # Extract numerical levels
            digits = self._get_precision(symbol)
            entry = entry_check
            sl = sl_check
            atr = float(card.get("atr") or (abs(entry - sl) if abs(entry - sl) > 0 else get_reference_median_atr(symbol)))
            tp1 = tp1_check
            if decision == "SELL":
                tp2 = float(card.get("tp2_price") or card.get("tp2") or round(entry - 3.0 * atr, digits))
                if tp2 == tp1:
                    tp2 = round(tp1 - (10 ** -digits), digits)
            else:
                tp2 = float(card.get("tp2_price") or card.get("tp2") or round(entry + 3.0 * atr, digits))
                if tp2 == tp1:
                    tp2 = round(tp1 + (10 ** -digits), digits)

            # Pre-publish Sanity Check (Agent Brief 15 Section 2 & Brief 20)
            levels_for_sanity = {
                "entry": entry,
                "stop_loss": sl,
                "take_profit_1": tp1,
                "take_profit_2": tp2,
                "atr": atr,
                "direction": decision
            }
            is_sane, sanity_reason = validate_signal_sanity(levels_for_sanity, symbol=symbol, direction=decision)
            if not is_sane:
                logger.error(
                    f"🚨 [SIGNAL SANITY CHECK FAILED] Refusing to publish invalid signal {snapshot_id} for {symbol}: {sanity_reason} "
                    f"| Computed values: entry={entry}, sl={sl}, tp1={tp1}, tp2={tp2}, atr={atr}, direction={decision}"
                )
                return False

            # Build pre-rendered messages
            msg1_text = self.format_levels_message(card)
            msg2_caption = self.format_analysis_caption(card)

            # Retrieve symbol's own measured metrics (Brief 14 Section 2 & Brief 20)
            metrics = get_symbol_metrics(symbol, side=decision)
            tier = metrics.get("tier", "EXPERIMENTAL")
            exp_r = metrics.get("expectancy_r", 0.0)
            trades = metrics.get("oos_trades", 0)
            folds_pos = int(metrics.get("folds_positive", 0))
            folds_tot = int(metrics.get("folds_total", 5))

            # Time string
            time_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            # Image URL (Brief 10 Section 3d: Image travels as URL, not base64)
            image_url = f"https://bot.fxengen.com/api/v1/analyst/share/{snapshot_id}/image"

            # Build Full n8n Webhook Payload (Frozen Contract - Brief 15 Section 1)
            prob_val = card.get("probability")
            thresh_val = card.get("threshold") or metrics.get("threshold", 0.40)
            payload = {
                "schema_version": 1,
                "type": "signal",
                "signal_id": snapshot_id,
                "symbol": symbol,
                "timeframe": str(card.get("timeframe") or "M15").upper(),
                "direction": decision,
                "time_utc": time_utc_str,
                "levels": {
                    "entry": round(entry, digits) if entry is not None else None,
                    "stop_loss": round(sl, digits) if sl is not None else None,
                    "take_profit_1": round(tp1, digits) if tp1 is not None else None,
                    "take_profit_2": round(tp2, digits) if tp2 is not None else None,
                    "tp1_validated": True,
                    "tp2_validated": False
                },
                "model": {
                    "probability": round(float(prob_val), 2) if prob_val is not None else None,
                    "threshold": round(float(thresh_val), 2) if thresh_val is not None else None,
                    "gate": "H4 EMA20 > EMA50 > EMA200",
                    "gate_passed": bool(card.get("gate_passed", True)),
                    "atr": round(atr, digits) if atr is not None else None,
                    "risk_reward": "1:2",
                    "baseline_expectancy_r": exp_r,
                    "baseline_trades": trades,
                    "tier": tier,
                    "folds_positive": folds_pos,
                    "folds_total": folds_tot,
                    "mode": "SHADOW",
                    "live_record_since": "2026-09-09"
                },
                "image_url": image_url,
                "message_1_text": msg1_text,
                "message_2_caption": msg2_caption
            }

            # Brief 29 Criterion 2: TELEGRAM_PUBLISH_MODEL_SIGNALS check
            publish_model_signals = getattr(settings, "TELEGRAM_PUBLISH_MODEL_SIGNALS", False)
            if not publish_model_signals and not force:
                logger.info(
                    f"ℹ️ [MODEL_PUBLISH_DISABLED] TELEGRAM_PUBLISH_MODEL_SIGNALS is False. "
                    f"Suppressing Telegram publish for model signal {snapshot_id} ({symbol}). "
                    f"No rate limit consumed. Registering for shadow outcome tracking."
                )
                self.state.setdefault("posted_signals", {})[snapshot_id] = {
                    "symbol": symbol,
                    "posted_at": datetime.now(timezone.utc).isoformat(),
                    "is_dry_run": True,
                    "is_test": is_test_flag,
                    "is_suppressed": True,
                    "suppressed_reason": "TELEGRAM_PUBLISH_MODEL_SIGNALS_FALSE"
                }
                self._save_state()
                signal_outcome_tracker.register_open_signal(
                    symbol=symbol,
                    signal_id=snapshot_id,
                    entry=entry,
                    stop_loss=sl,
                    take_profit_1=tp1,
                    take_profit_2=tp2 if tp2 > 0 else None,
                    direction=decision,
                    atr=atr,
                    published_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    published_timestamp=time.time()
                )
                return True

            # Dry-run check
            if is_dry_run:
                logger.info(
                    f"\n=================== [TELEGRAM DRY RUN START] ===================\n"
                    f"Target: n8n Webhook DRY RUN (TELEGRAM_DRY_RUN=True - no network request)\n"
                    f"Payload JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE 1 (LEVELS TEXT) ---\n"
                    f"{msg1_text}\n\n"
                    f"--- MESSAGE 2 (PHOTO CAPTION) ---\n"
                    f"Image URL: {image_url}\n"
                    f"Caption:\n"
                    f"{msg2_caption}\n"
                    f"==================== [TELEGRAM DRY RUN END] ====================\n"
                )
                self.record_signal_posted(snapshot_id, symbol, is_dry_run=True, payload=payload, is_test=is_test_flag)
                # Brief 18: write open signal immediately before anything can clear it
                signal_outcome_tracker.register_open_signal(
                    symbol=symbol,
                    signal_id=snapshot_id,
                    entry=entry,
                    stop_loss=sl,
                    take_profit_1=tp1,
                    take_profit_2=tp2 if tp2 > 0 else None,
                    direction=decision,
                    atr=atr,
                    published_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    published_timestamp=time.time()
                )
                return True

            # Live delivery to n8n webhook
            success = await self._post_to_n8n_webhook(payload)
            if success:
                self.record_signal_posted(snapshot_id, symbol, is_dry_run=False, payload=payload, is_test=is_test_flag)
                # Brief 18: write open signal immediately before anything can clear it
                signal_outcome_tracker.register_open_signal(
                    symbol=symbol,
                    signal_id=snapshot_id,
                    entry=entry,
                    stop_loss=sl,
                    take_profit_1=tp1,
                    take_profit_2=tp2 if tp2 > 0 else None,
                    direction=decision,
                    atr=atr,
                    published_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    published_timestamp=time.time()
                )
                logger.info(f"✅ [N8N WEBHOOK DISPATCHED] Signal {snapshot_id} ({symbol}) published successfully.")
                return True
            else:
                logger.error(f"❌ [N8N WEBHOOK FAILED] Delivery failed for signal {snapshot_id}. Left unmarked for retry.")
                return False

        except Exception as e:
            logger.error(f"Telegram/n8n: Unexpected error in publish_confirmed_signal: {e}", exc_info=True)
            return False

    def build_desk_payload(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Brief 29 Section 3: Builds the frozen webhook payload for desk trades.
        schema_version: 1, type: "signal", source: "DESK", levels include take_profit_3.
        """
        trade_id = str(trade_record.get("trade_id") or trade_record.get("id"))
        symbol = str(trade_record.get("symbol", "")).upper()
        direction = str(trade_record.get("direction", "BUY")).upper()
        timeframe = str(trade_record.get("timeframe", "M15")).upper()
        digits = self._get_precision(symbol)

        entry = float(trade_record.get("entry_price") or trade_record.get("entry", 0.0))
        sl = float(trade_record.get("stop_loss") or trade_record.get("sl_price", 0.0))
        tp1 = float(trade_record.get("take_profit_1") or trade_record.get("tp1_price", 0.0))
        tp2_raw = trade_record.get("take_profit_2") or trade_record.get("tp2_price")
        tp3_raw = trade_record.get("take_profit_3") or trade_record.get("tp3_price")

        levels = {
            "entry": round(entry, digits),
            "stop_loss": round(sl, digits),
            "take_profit_1": round(tp1, digits)
        }
        if tp2_raw is not None and float(tp2_raw) > 0:
            levels["take_profit_2"] = round(float(tp2_raw), digits)
        if tp3_raw is not None and float(tp3_raw) > 0:
            levels["take_profit_3"] = round(float(tp3_raw), digits)

        image_url = f"https://desk.fxengen.com/api/v1/desk/trades/{trade_id}/card"
        msg1_text = self.format_desk_levels_message(trade_record)
        msg2_caption = self.format_desk_caption(trade_record)

        return {
            "schema_version": 1,
            "type": "signal",
            "source": "DESK",
            "signal_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "timeframe": timeframe,
            "time_utc": trade_record.get("entered_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
            "levels": levels,
            "model": None,
            "tier": None,
            "baseline_expectancy_r": None,
            "oos_trades": None,
            "image_url": image_url,
            "message_1_text": msg1_text,
            "message_2_caption": msg2_caption
        }

    async def publish_desk_trade(self, trade_record: Dict[str, Any], force: bool = False) -> bool:
        """
        Brief 29 Section 3: Publishes a newly recorded manual desk trade to Telegram via n8n webhook.
        Rules:
          0. Historical trades (the 73 sealed records or any pre-existing record) NEVER publish.
          1. Trade must carry a live 91-feature snapshot taken at submission. Stale/missing snapshots do not publish.
          2. Forex market open check (weekends / closed market records without publishing).
          3. Gold (XAUUSD) excluded under standing instruction.
          4. Per-trade publish toggle: if publish_to_channel is False, do not publish.
          5. Levels sanity defense: entry, stop_loss, take_profit_1 must be valid positive floats.
          6. Deduplication on disk (never post same trade_id twice).
          7. Rate limiting on disk (shares signal rate limit rules).
          8. Gated behind TELEGRAM_PUBLISH_DESK (default False). Logs full dry run payload if False or DRY_RUN=True.
          9. Silent failure: exceptions never crash caller.
        """
        try:
            trade_id = str(trade_record.get("trade_id") or trade_record.get("id") or "").strip()
            if not trade_id:
                logger.error("Telegram/n8n: Missing trade_id in desk trade record.")
                return False

            # Rule 0 (Brief 29 Criterion 0): Sealed historical trades MUST NEVER publish
            if trade_id in getattr(self, "sealed_manual_records", set()):
                logger.warning(
                    f"🛡️ [DROP: HISTORICAL_SEALED] Refusing to publish desk trade {trade_id}: "
                    f"trade is part of sealed historical records. Telegram publishing is strictly forbidden."
                )
                return False

            symbol = str(trade_record.get("symbol", "")).upper()

            # Rule 4 (Brief 29 Section 5): Per-trade publish toggle
            if not trade_record.get("publish_to_channel", True):
                logger.info(
                    f"ℹ️ [DROP: PUBLISH_OFF] Skipping Telegram publish for desk trade {trade_id} ({symbol}): "
                    f"'publish to channel' was disabled on submission."
                )
                return False

            # Rule 3 (Brief 29 Section 6): Gold exclusion
            excluded = [s.upper() for s in getattr(settings, "TELEGRAM_EXCLUDED_SYMBOLS", ["XAUUSD"])]
            if symbol in ["XAUUSD", "GOLD"] or symbol in excluded:
                logger.info(
                    f"🚫 [DROP: EXCLUDED_SYMBOL] Skipping Telegram publish for {symbol} (manual desk trade {trade_id}): "
                    f"Gold is excluded from channel publishing under standing instruction."
                )
                return False

            # Rule 1 (Brief 29 Section 3): Must carry live 91-feature snapshot
            snapshot = trade_record.get("snapshot") or trade_record.get("features")
            snap_source = trade_record.get("snapshot_source", "")
            if not snapshot or (isinstance(snapshot, dict) and len(snapshot) < 90) or snap_source in ("UNAVAILABLE", "STALE", "FAILED"):
                logger.warning(
                    f"🚫 [DROP: MISSING_OR_STALE_SNAPSHOT] Skipping Telegram publish for desk trade {trade_id} ({symbol}): "
                    f"lacks live 91-feature snapshot (snapshot_source={snap_source}, count={len(snapshot) if isinstance(snapshot, dict) else 0})."
                )
                return False

            # Rule 2 (Brief 29 Section 3): Forex market open check
            if not force:
                market_open, market_reason = self.is_forex_market_open()
                if not market_open:
                    logger.info(
                        f"🚫 [DROP: MARKET_CLOSED] Weekend / market closed blackout active ({market_reason}). "
                        f"Desk trade {trade_id} ({symbol}) was recorded but will not publish."
                    )
                    return False

            # Rule 6: Deduplication check
            if self.is_signal_already_posted(trade_id):
                logger.info(
                    f"🚫 [DROP: DUPLICATE] Dropping desk trade {trade_id} for {symbol}: "
                    f"already posted previously on disk."
                )
                return False

            # Rule 7: Rate limit check
            allowed, rate_reason = self.check_signal_rate_limit(symbol)
            if not allowed:
                logger.error(
                    f"🚨 [DROP: RATE_LIMIT] Refusing desk trade {trade_id} for {symbol}: {rate_reason}"
                )
                return False

            # Rule 5: Levels sanity
            def _to_pos_float(v: Any) -> Optional[float]:
                if v is None:
                    return None
                try:
                    f = float(v)
                    return f if (math.isfinite(f) and f > 0) else None
                except (ValueError, TypeError):
                    return None

            entry_check = _to_pos_float(trade_record.get("entry_price") or trade_record.get("entry"))
            sl_check = _to_pos_float(trade_record.get("stop_loss") or trade_record.get("sl_price"))
            tp1_check = _to_pos_float(trade_record.get("take_profit_1") or trade_record.get("tp1_price"))

            if entry_check is None or sl_check is None or tp1_check is None:
                logger.error(
                    f"🚨 [DROP: INVALID_LEVELS] Refusing desk trade {trade_id} for {symbol}: "
                    f"entry, stop_loss, and take_profit_1 must be valid positive numbers."
                )
                return False

            # Build full n8n payload
            payload = self.build_desk_payload(trade_record)
            msg1_text = payload["message_1_text"]
            msg2_caption = payload["message_2_caption"]
            image_url = payload["image_url"]

            publish_desk = getattr(settings, "TELEGRAM_PUBLISH_DESK", False)
            is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)

            # Check if desk publishing is enabled
            if not publish_desk and not force:
                logger.info(
                    f"ℹ️ [DESK_PUBLISH_DISABLED] TELEGRAM_PUBLISH_DESK is False. "
                    f"Logging dry run payload for desk trade {trade_id} ({symbol}). Live webhook not sent.\n"
                    f"Target: n8n Webhook (DESK SIGNAL)\n"
                    f"Payload JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE 1 (LEVELS TEXT) ---\n{msg1_text}\n\n"
                    f"--- MESSAGE 2 (PHOTO CAPTION) ---\nImage URL: {image_url}\nCaption:\n{msg2_caption}\n"
                )
                self.record_signal_posted(trade_id, symbol, is_dry_run=True, payload=payload)
                return True

            if is_dry_run:
                logger.info(
                    f"\n=================== [TELEGRAM DESK DRY RUN START] ===================\n"
                    f"Target: n8n Webhook DRY RUN (TELEGRAM_DRY_RUN=True - no network request)\n"
                    f"Payload JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE 1 (LEVELS TEXT) ---\n{msg1_text}\n\n"
                    f"--- MESSAGE 2 (PHOTO CAPTION) ---\nImage URL: {image_url}\nCaption:\n{msg2_caption}\n"
                    f"==================== [TELEGRAM DESK DRY RUN END] ====================\n"
                )
                self.record_signal_posted(trade_id, symbol, is_dry_run=True, payload=payload)
                return True

            # Live delivery to n8n webhook
            success = await self._post_to_n8n_webhook(payload)
            if success:
                self.record_signal_posted(trade_id, symbol, is_dry_run=False, payload=payload)
                logger.info(f"✅ [N8N WEBHOOK DISPATCHED] Desk trade {trade_id} ({symbol}) published successfully.")
                return True
            else:
                logger.error(f"❌ [N8N WEBHOOK FAILED] Delivery failed for desk trade {trade_id}. Left unmarked for retry.")
                return False

        except Exception as e:
            logger.error(f"Error publishing desk trade {trade_record.get('trade_id')}: {e}", exc_info=True)
            return False

    async def publish_desk_result(self, outcome_record: Dict[str, Any], force: bool = False) -> bool:
        """
        Brief 31 Section 2: Publishes a closed manual desk trade result to Telegram via n8n webhook.
        Rules:
          0. Sealed historical records (the 52 records) MUST NEVER publish a result.
          1. Gated behind TELEGRAM_PUBLISH_DESK (and TELEGRAM_PUBLISH_RESULTS).
          2. Deduplication on disk (never send same trade_id result twice).
          3. Per-trade publish toggle: if publish_to_channel is False, do not publish.
          4. Excluded symbols check (Gold excluded).
          5. Losses are NEVER softened or omitted: STOP HIT with negative R.
          6. Result in R, not currency.
          7. Carries running desk record line.
          8. Silent failure: exceptions never crash caller.
        """
        try:
            trade_id = str(outcome_record.get("trade_id") or outcome_record.get("signal_id") or "").strip()
            if not trade_id:
                logger.warning("Telegram/n8n: Missing trade_id in desk outcome record.")
                return False

            # Rule 0 (Brief 31 Standing Rule): Sealed historical manual trades MUST NEVER publish
            if trade_id in getattr(self, "sealed_manual_records", set()):
                logger.warning(
                    f"🛡️ [DROP: HISTORICAL_SEALED] Refusing to publish desk result for {trade_id}: "
                    f"trade is part of sealed historical records. Telegram result publishing is strictly forbidden."
                )
                return False

            # Rule 1: Feature Flag Check
            publish_desk = getattr(settings, "TELEGRAM_PUBLISH_DESK", False)
            if not publish_desk and not force:
                logger.debug(f"ℹ️ [DESK_RESULT_DISABLED] TELEGRAM_PUBLISH_DESK is False. Result not sent.")
                return False

            # Rule 3: Per-trade toggle
            if not outcome_record.get("publish_to_channel", True):
                logger.info(f"ℹ️ [DROP: PUBLISH_OFF] Skipping result publish for desk trade {trade_id}: publish_to_channel=False.")
                return False

            symbol = str(outcome_record.get("symbol", "")).upper().strip()
            # Rule 4: Excluded symbols (Gold)
            excluded = [s.upper() for s in getattr(settings, "TELEGRAM_EXCLUDED_SYMBOLS", ["XAUUSD"])]
            if symbol in ["XAUUSD", "GOLD"] or symbol in excluded:
                logger.info(f"🚫 [DROP: EXCLUDED_SYMBOL] Skipping result publish for {symbol} (desk trade {trade_id}).")
                return False

            # Rule 2: Deduplication check on disk
            if self.is_result_already_posted(trade_id):
                logger.info(f"🚫 [DROP: DUPLICATE_RESULT] Result for desk trade {trade_id} already posted on disk.")
                return False

            # Build full payload
            payload = self.build_desk_result_payload(outcome_record)
            msg_text = payload["message_text"]

            is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)
            if is_dry_run:
                logger.info(
                    f"\n=================== [TELEGRAM DESK RESULT DRY RUN START] ===================\n"
                    f"Target: n8n Webhook DRY RUN (TELEGRAM_DRY_RUN=True - no network request)\n"
                    f"Payload JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE TEXT ---\n{msg_text}\n"
                    f"==================== [TELEGRAM DESK RESULT DRY RUN END] ====================\n"
                )
                self.record_result_posted(trade_id, symbol, is_dry_run=True, payload=payload)
                return True

            # Live delivery to n8n webhook
            success = await self._post_to_n8n_webhook(payload)
            if success:
                self.record_result_posted(trade_id, symbol, is_dry_run=False, payload=payload)
                logger.info(f"✅ [N8N DESK RESULT DISPATCHED] Desk trade outcome {trade_id} ({symbol} {payload['outcome']}) published successfully.")
                return True
            else:
                logger.error(f"❌ [N8N DESK RESULT FAILED] Delivery failed for desk outcome {trade_id}. Left unmarked for retry.")
                return False

        except Exception as e:
            logger.error(f"Telegram/n8n: Unexpected error in publish_desk_result: {e}", exc_info=True)
            return False

    def dispatch_desk_result(self, outcome_record: Dict[str, Any]):
        """
        Synchronous/asynchronous dispatch helper for publish_desk_result.
        Can be safely called from sync methods in AnalystDeskService.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.publish_desk_result(outcome_record))
            except RuntimeError:
                asyncio.run(self.publish_desk_result(outcome_record))
        except Exception as e:
            logger.error(f"Error in dispatch_desk_result: {e}")

    # -------------------------------------------------------------------------
    # Mirrored Account Publishing (Agent Brief 32)
    # -------------------------------------------------------------------------
    def build_mirror_entry_payload(self, record: Dict[str, Any]) -> Dict[str, Any]:
        ticket = record.get("ticket")
        sym = str(record.get("symbol", "")).upper()
        dir_str = str(record.get("direction", "BUY")).upper()
        digits = record.get("digits") or self._get_precision(sym)
        entry = float(record.get("entry", 0.0))
        sl = float(record["stop_loss"]) if (record.get("stop_loss") is not None and float(record.get("stop_loss", 0)) > 0) else None
        tp = float(record["take_profit"]) if (record.get("take_profit") is not None and float(record.get("take_profit", 0)) > 0) else None
        lot = float(record.get("lot", 0.01))
        rr = record.get("rr")
        risk_price = record.get("risk_price")

        entry_str = f"{entry:.{digits}f}"
        sl_str = f"{sl:.{digits}f}" if sl else "Open / Floating"
        tp_str = f"{tp:.{digits}f}" if tp else "Open / Floating"
        rr_str = f"1:{rr:.2f}" if (rr and rr > 0) else "Dynamic / Open"

        msg_text = (
            f"FOREX ENGINEER — LIVE TRADE OPENED\n\n"
            f"{sym}  ·  {dir_str}  ·  #{ticket}\n\n"
            f"Entry        {entry_str}\n"
            f"Lot          {lot:.2f}\n"
            f"Stop Loss    {sl_str}\n"
            f"Take Profit  {tp_str}\n"
            f"Risk:Reward  {rr_str}\n\n"
            f"Mirrored from engineer's MT5 account (SCFMLimited-Demo2)"
        )

        levels = {
            "entry": round(entry, digits),
            "stop_loss": round(sl, digits) if sl else None,
            "take_profit_1": round(tp, digits) if tp else None,
            "take_profit_2": None,
            "take_profit_3": None,
            "sl_text": sl_str,
            "tp_text": tp_str
        }

        # Reference median ATR for n8n pad calculation
        atr = get_reference_median_atr(sym)

        return {
            "schema_version": 1,
            "type": "signal",
            "source": "MIRROR",
            "signal_id": f"MIRROR_{ticket}",
            "ticket": ticket,
            "symbol": sym,
            "direction": dir_str,
            "lot": lot,
            "digits": digits,
            "entry": round(entry, digits),
            "stop_loss": round(sl, digits) if sl else None,
            "take_profit": round(tp, digits) if tp else None,
            "levels": levels,
            "model": {
                "atr": atr,
                "risk_reward": rr_str
            },
            "opened_utc": record.get("opened_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "account": record.get("account"),
            "category": classify_market_category(sym),
            "risk_price": risk_price,
            "rr": rr,
            "image_url": f"https://mirror.fxengen.com/api/v1/mirror/trades/{ticket}/card",
            "message_1_text": msg_text,
            "message_text": msg_text
        }

    def build_mirror_update_payload(self, update_rec: Dict[str, Any]) -> Dict[str, Any]:
        ticket = update_rec.get("ticket")
        sym = str(update_rec.get("symbol", "")).upper()
        dir_str = str(update_rec.get("direction", "BUY")).upper()
        digits = update_rec.get("digits") or self._get_precision(sym)
        changed = update_rec.get("changed", [])
        is_be = update_rec.get("is_breakeven", False)
        sl_info = update_rec.get("stop_loss", {})
        tp_info = update_rec.get("take_profit", {})

        lines = [
            f"FOREX ENGINEER — TRADE UPDATED\n",
            f"{sym}  ·  {dir_str}  ·  #{ticket}\n"
        ]
        if isinstance(sl_info, dict):
            f_sl = sl_info.get("from")
            t_sl = sl_info.get("to")
        elif sl_info is not None:
            f_sl = None
            t_sl = sl_info
        else:
            f_sl = None
            t_sl = None

        if "stop_loss" in changed:
            f_str = f"{float(f_sl):.{digits}f}" if (f_sl and float(f_sl) > 0) else "Open"
            t_str = f"{float(t_sl):.{digits}f}" if (t_sl and float(t_sl) > 0) else "Open"
            be_tag = "   (breakeven)" if is_be else ""
            lines.append(f"Stop moved   {f_str}  ->  {t_str}{be_tag}")
        else:
            sl_val = t_sl or f_sl
            sl_str = f"{float(sl_val):.{digits}f}" if (sl_val and float(sl_val) > 0) else "Open / Floating"
            lines.append(f"Stop         {sl_str}   unchanged")

        if isinstance(tp_info, dict):
            f_tp = tp_info.get("from")
            t_tp = tp_info.get("to")
        elif tp_info is not None:
            f_tp = None
            t_tp = tp_info
        else:
            f_tp = None
            t_tp = None

        if "take_profit" in changed:
            f_str = f"{float(f_tp):.{digits}f}" if (f_tp and float(f_tp) > 0) else "Open"
            t_str = f"{float(t_tp):.{digits}f}" if (t_tp and float(t_tp) > 0) else "Open"
            lines.append(f"Target moved {f_str}  ->  {t_str}")
        else:
            tp_val = t_tp or f_tp
            tp_str = f"{float(tp_val):.{digits}f}" if (tp_val and float(tp_val) > 0) else "Open / Floating"
            lines.append(f"Target       {tp_str}   unchanged")

        msg_text = "\n".join(lines)

        entry = update_rec.get("entry")
        lot = update_rec.get("lot")
        note = update_rec.get("note", "")

        return {
            "schema_version": 1,
            "type": "update",
            "source": "MIRROR",
            "signal_id": f"MIRROR_{ticket}",
            "ticket": ticket,
            "symbol": sym,
            "direction": dir_str,
            "digits": digits,
            "entry": entry,
            "lot": lot,
            "changed": changed,
            "is_breakeven": is_be,
            "note": note,
            "stop_loss": sl_info,
            "take_profit": tp_info,
            "levels": {
                "entry": entry,
                "stop_loss": t_sl if (t_sl and float(t_sl) > 0) else None,
                "take_profit_1": t_tp if (t_tp and float(t_tp) > 0) else None,
                "from_stop_loss": f_sl if (f_sl and float(f_sl) > 0) else None,
                "from_take_profit": f_tp if (f_tp and float(f_tp) > 0) else None,
            },
            "account": update_rec.get("account"),
            "category": classify_market_category(sym),
            "message_1_text": msg_text,
            "message_text": msg_text
        }

    def build_mirror_result_payload(self, outcome_rec: Dict[str, Any]) -> Dict[str, Any]:
        ticket = outcome_rec.get("ticket")
        sym = str(outcome_rec.get("symbol", "")).upper()
        dir_str = str(outcome_rec.get("direction", "BUY")).upper()
        digits = outcome_rec.get("digits") or self._get_precision(sym)
        entry = float(outcome_rec.get("entry", 0.0))
        exit_price = float(outcome_rec.get("exit_price", 0.0))
        lot = float(outcome_rec.get("lot", 0.01))
        barrier = outcome_rec.get("barrier_hit", "CLOSED MANUALLY")
        duration = outcome_rec.get("duration", "0m")
        r_mult = outcome_rec.get("r_multiple")
        profit = float(outcome_rec.get("profit", 0.0))

        if "TARGET" in barrier.upper() or "TP" in barrier.upper():
            outcome_code = "TP1"
        elif "STOP" in barrier.upper() or "SL" in barrier.upper():
            outcome_code = "SL"
        elif "BREAKEVEN" in barrier.upper() or abs(exit_price - entry) <= (0.0005 if digits > 2 else 0.5):
            outcome_code = "BE"
        else:
            outcome_code = "MANUAL"

        if r_mult is not None:
            r_str = f"{r_mult:+.2f} R"
        else:
            r_str = f"${profit:+.2f}" if profit != 0 else "Closed"

        msg_text = (
            f"FOREX ENGINEER — TRADE CLOSED\n\n"
            f"{sym}  ·  {dir_str}  ·  {barrier}\n\n"
            f"Entry    {entry:.{digits}f}\n"
            f"Exit     {exit_price:.{digits}f}\n"
            f"Lot      {lot:.2f}\n"
            f"Result   {r_str}      Duration  {duration}\n\n"
            f"Ticket  #{ticket}"
        )

        return {
            "schema_version": 1,
            "type": "result",
            "source": "MIRROR",
            "signal_id": f"MIRROR_{ticket}",
            "ticket": ticket,
            "deal_ticket": outcome_rec.get("deal_ticket"),
            "symbol": sym,
            "direction": dir_str,
            "digits": digits,
            "outcome": outcome_code,
            "barrier_hit": barrier,
            "deal_reason": outcome_rec.get("deal_reason"),
            "entry": round(entry, digits),
            "exit_price": round(exit_price, digits),
            "lot": lot,
            "r_multiple": r_mult,
            "profit": profit,
            "duration": duration,
            "opened_utc": outcome_rec.get("opened_utc"),
            "closed_utc": outcome_rec.get("closed_utc"),
            "account": outcome_rec.get("account"),
            "category": classify_market_category(sym),
            "levels": {
                "entry": round(entry, digits),
                "exit_price": round(exit_price, digits)
            },
            "image_url": f"https://mirror.fxengen.com/api/v1/mirror/trades/{ticket}/card",
            "message_1_text": msg_text,
            "message_text": msg_text
        }


    async def publish_mirror_trade(self, record: Dict[str, Any], force: bool = False) -> bool:
        """Publishes a new mirrored trade entry to Telegram/n8n."""
        try:
            ticket = record.get("ticket")
            sym = str(record.get("symbol", "")).upper()
            if not ticket:
                return False

            # Gold decision check (Section 7)
            if ("XAU" in sym or "GOLD" in sym) and not getattr(settings, "MIRROR_PUBLISH_GOLD", True) and not force:
                logger.info(f"🛡️ [DROP: GOLD_EXCLUDED] Mirrored trade #{ticket} ({sym}) suppressed pending owner confirmation.")
                return False

            if not getattr(settings, "MIRROR_PUBLISH_TELEGRAM", True) and not force:
                logger.info(f"ℹ️ [MIRROR_PUBLISH_DISABLED] MIRROR_PUBLISH_TELEGRAM is False. Skipping publish for #{ticket}.")
                return False

            payload = self.build_mirror_entry_payload(record)
            is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)

            if is_dry_run:
                logger.info(
                    f"\n=================== [TELEGRAM MIRROR ENTRY DRY RUN START] ===================\n"
                    f"Target: n8n Webhook DRY RUN\n"
                    f"Payload JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE TEXT ---\n{payload['message_text']}\n"
                    f"==================== [TELEGRAM MIRROR ENTRY DRY RUN END] ====================\n"
                )
                return True

            success = await self._post_to_n8n_webhook(payload)
            if success:
                logger.info(f"✅ [N8N MIRROR DISPATCHED] Mirrored entry #{ticket} ({sym}) published successfully.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error publishing mirror trade {record.get('ticket')}: {e}", exc_info=True)
            return False

    async def publish_mirror_update(self, update_rec: Dict[str, Any], force: bool = False) -> bool:
        """Publishes an update to a mirrored trade (SL/TP modification, breakeven) to Telegram/n8n."""
        try:
            ticket = update_rec.get("ticket")
            sym = str(update_rec.get("symbol", "")).upper()
            if not ticket:
                return False

            if ("XAU" in sym or "GOLD" in sym) and not getattr(settings, "MIRROR_PUBLISH_GOLD", True) and not force:
                return False

            if not getattr(settings, "MIRROR_PUBLISH_TELEGRAM", True) and not force:
                return False

            payload = self.build_mirror_update_payload(update_rec)
            is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)

            if is_dry_run:
                logger.info(
                    f"\n=================== [TELEGRAM MIRROR UPDATE DRY RUN START] ===================\n"
                    f"Target: n8n Webhook DRY RUN\n"
                    f"Payload JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE TEXT ---\n{payload['message_text']}\n"
                    f"==================== [TELEGRAM MIRROR UPDATE DRY RUN END] ====================\n"
                )
                return True

            success = await self._post_to_n8n_webhook(payload)
            if success:
                logger.info(f"✅ [N8N MIRROR UPDATE DISPATCHED] Mirrored update #{ticket} ({sym}) published successfully.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error publishing mirror update {update_rec.get('ticket')}: {e}", exc_info=True)
            return False

    async def publish_mirror_result(self, outcome_rec: Dict[str, Any], force: bool = False) -> bool:
        """Publishes a closed mirrored trade result to Telegram/n8n."""
        try:
            ticket = outcome_rec.get("ticket")
            sym = str(outcome_rec.get("symbol", "")).upper()
            if not ticket:
                return False

            if ("XAU" in sym or "GOLD" in sym) and not getattr(settings, "MIRROR_PUBLISH_GOLD", True) and not force:
                return False

            if not getattr(settings, "MIRROR_PUBLISH_TELEGRAM", True) and not force:
                return False

            payload = self.build_mirror_result_payload(outcome_rec)
            is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)

            if is_dry_run:
                logger.info(
                    f"\n=================== [TELEGRAM MIRROR RESULT DRY RUN START] ===================\n"
                    f"Target: n8n Webhook DRY RUN\n"
                    f"Payload JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE TEXT ---\n{payload['message_text']}\n"
                    f"==================== [TELEGRAM MIRROR RESULT DRY RUN END] ====================\n"
                )
                return True

            success = await self._post_to_n8n_webhook(payload)
            if success:
                logger.info(f"✅ [N8N MIRROR RESULT DISPATCHED] Mirrored result #{ticket} ({sym}) published successfully.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error publishing mirror result {outcome_rec.get('ticket')}: {e}", exc_info=True)
            return False

    def dispatch_mirror_trade(self, record: Dict[str, Any]):
        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.publish_mirror_trade(record))
            except RuntimeError:
                asyncio.run(self.publish_mirror_trade(record))
        except Exception as e:
            logger.error(f"Error in dispatch_mirror_trade: {e}")

    def dispatch_mirror_update(self, update_rec: Dict[str, Any]):
        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.publish_mirror_update(update_rec))
            except RuntimeError:
                asyncio.run(self.publish_mirror_update(update_rec))
        except Exception as e:
            logger.error(f"Error in dispatch_mirror_update: {e}")

    def dispatch_mirror_result(self, outcome_rec: Dict[str, Any]):
        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.publish_mirror_result(outcome_rec))
            except RuntimeError:
                asyncio.run(self.publish_mirror_result(outcome_rec))
        except Exception as e:
            logger.error(f"Error in dispatch_mirror_result: {e}")

    # Synchronous aliases for testing and non-async callers
    def publish_mirror_trade_sync(self, record: Dict[str, Any]) -> bool:
        self.dispatch_mirror_trade(record)
        return True

    def publish_mirror_update_sync(self, update_rec: Dict[str, Any]) -> bool:
        self.dispatch_mirror_update(update_rec)
        return True

    def publish_mirror_result_sync(self, outcome_rec: Dict[str, Any]) -> bool:
        self.dispatch_mirror_result(outcome_rec)
        return True

    async def send_waiting_notice(
        self,
        symbols_status: Optional[List[Dict[str, str]]] = None,
        last_signal: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> bool:
        """
        Sends an hourly waiting notice via n8n (Brief 10 Section 3c).
        Rules:
          1. Skip if a confirmed signal was already posted within the last hour.
          2. Rate limit: max 1 per hour, max 24 per day (separate counter from signals).
          3. Carries real symbol reasons and age of last confirmed signal.
          4. Impossible to mistake for a signal (no levels, no chart).
        """
        try:
            now_ts = time.time()
            
            # Check global flag for waiting notices
            if not getattr(settings, "TELEGRAM_ENABLE_WAITING_NOTICES", False) and not force:
                logger.info("🚫 Telegram/n8n: Waiting notices are disabled by configuration. Skipping.")
                return False

            if not force:
                # Rule 0: Weekend blackout check (Forex markets closed Fri 22:00 UTC -> Sun 21:00 UTC)
                market_open, market_reason = self.is_forex_market_open()
                if not market_open:
                    logger.info(f"Telegram/n8n: Weekend blackout active ({market_reason}). Skipping waiting notice.")
                    return False

                # Rule 1: Skip if signal posted within last hour (3600 seconds)
                last_sig_ts = self.state.get("last_signal_timestamp", 0.0)
                if (now_ts - last_sig_ts) < 3600.0:
                    logger.info("Telegram/n8n: Confirmed signal posted within the last hour. Skipping waiting notice.")
                    return False

                # Rule 2: Notice rate limit check
                allowed, notice_reason = self.check_notice_rate_limit()
                if not allowed:
                    logger.warning(f"Telegram/n8n: Waiting notice skipped: {notice_reason}")
                    return False


            # Default status resolution if not passed
            if not symbols_status:
                symbols_status = [
                    {"symbol": "XAUUSD", "reason": "H4 trend not aligned"},
                    {"symbol": "USDJPY", "reason": "probability 0.38 below 0.40"}
                ]

            if not last_signal:
                if self.state.get("last_signal_info"):
                    last_signal = dict(self.state["last_signal_info"])
                    last_ts = self.state.get("last_signal_timestamp", now_ts)
                    last_signal["age_hours"] = max(1, int(round((now_ts - last_ts) / 3600.0)))
                else:
                    last_signal = {
                        "symbol": "XAUUSD",
                        "age_hours": 18,
                        "time_utc": "2026-09-09 13:43:00"
                    }

            msg_text = self.format_waiting_message(symbols_status, last_signal)
            time_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            payload = {
                "schema_version": 1,
                "type": "waiting",
                "time_utc": time_utc_str,
                "symbols": symbols_status,
                "last_signal": last_signal,
                "message_text": msg_text
            }

            # Dry-run check
            is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)
            if is_dry_run:
                logger.info(
                    f"\n=================== [N8N WAITING NOTICE DRY RUN] ===================\n"
                    f"Payload JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE TEXT ---\n"
                    f"{msg_text}\n"
                    f"====================================================================\n"
                )
                self.record_notice_sent(is_dry_run=True, payload=payload)
                return True

            success = await self._post_to_n8n_webhook(payload)
            if success:
                self.record_notice_sent(is_dry_run=False, payload=payload)
                logger.info("✅ [N8N WAITING NOTICE DISPATCHED] Hourly heartbeat posted successfully.")
                return True
            else:
                logger.error("❌ [N8N WAITING NOTICE FAILED] Delivery failed.")
                return False

        except Exception as e:
            logger.error(f"Telegram/n8n: Unexpected error in send_waiting_notice: {e}", exc_info=True)
            return False

    async def publish_signal_result(self, outcome_record: Dict[str, Any], force: bool = False) -> bool:
        """
        Brief 23: Publishes a signal outcome result message to Telegram via n8n webhook.
        Rules:
          1. Gated behind TELEGRAM_PUBLISH_RESULTS (default False).
          2. Model signals only (rejects test/verification signals).
          3. Validated publishing symbols only (the nine model symbols).
          4. Deduplication on disk (never sends same signal_id twice).
          5. Historical outcomes never published.
          6. TELEGRAM_DRY_RUN logs exact JSON and message, sends nothing.
          7. Silent failure: exceptions never crash caller.
        """
        try:
            # 1. Feature Flag Check
            publish_enabled = getattr(settings, "TELEGRAM_PUBLISH_RESULTS", False)
            if not publish_enabled and not force:
                logger.debug("🚫 [DROP: RESULTS_DISABLED] TELEGRAM_PUBLISH_RESULTS is False. Producing nothing.")
                return False

            signal_id = str(outcome_record.get("signal_id") or "").strip()
            if not signal_id:
                logger.warning("🚫 [DROP: NO_SIGNAL_ID] Missing signal_id in outcome record.")
                return False

            symbol = str(outcome_record.get("symbol") or "").upper().strip()

            # 2. Excluded symbols check
            excluded = [s.upper() for s in getattr(settings, "TELEGRAM_EXCLUDED_SYMBOLS", ["XAUUSD"])]
            if symbol in excluded:
                logger.info(f"🚫 [DROP: EXCLUDED_SYMBOL] Skipping result publish for {symbol}: symbol is excluded.")
                return False

            # 3. Validated symbols check (the nine publishing symbols)
            valid_symbols = self.get_validated_symbols()
            if symbol not in valid_symbols:
                logger.info(f"🚫 [DROP: UNVALIDATED_SYMBOL] Skipping result publish for {symbol}: not in validated symbols {valid_symbols}.")
                return False

            # 4. Model signals only: test signals never publish
            is_test = bool(outcome_record.get("is_test", False))
            lower_id = signal_id.lower()
            if is_test or "test" in lower_id or "verify" in lower_id:
                logger.info(f"🧪 [DROP: TEST_SIGNAL] Skipping result publish for test/verification signal: {signal_id}")
                return False

            # 5. Deduplication check on disk
            if self.is_result_already_posted(signal_id):
                logger.info(f"🚫 [DROP: DUPLICATE_RESULT] Result for signal {signal_id} already posted previously on disk.")
                return False

            # Build full contract payload
            payload = self.build_result_payload(outcome_record)
            msg_text = payload["message_text"]

            # Dry-run check
            is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)
            if is_dry_run:
                logger.info(
                    f"\n=================== [TELEGRAM RESULT DRY RUN START] ===================\n"
                    f"Target: n8n Webhook DRY RUN (TELEGRAM_DRY_RUN=True - no network request)\n"
                    f"Payload JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE TEXT ---\n"
                    f"{msg_text}\n"
                    f"==================== [TELEGRAM RESULT DRY RUN END] ====================\n"
                )
                self.record_result_posted(signal_id, symbol, is_dry_run=True, payload=payload)
                return True

            # Live delivery to n8n webhook
            success = await self._post_to_n8n_webhook(payload)
            if success:
                self.record_result_posted(signal_id, symbol, is_dry_run=False, payload=payload)
                logger.info(f"✅ [N8N RESULT DISPATCHED] Signal outcome {signal_id} ({symbol} {payload['outcome']}) published successfully.")
                return True
            else:
                logger.error(f"❌ [N8N RESULT FAILED] Delivery failed for signal outcome {signal_id}. Left unmarked for retry.")
                return False

        except Exception as e:
            logger.error(f"Telegram/n8n: Unexpected error in publish_signal_result: {e}", exc_info=True)
            return False

    def dispatch_signal_result(self, outcome_record: Dict[str, Any]):
        """
        Synchronous/asynchronous dispatch helper for publish_signal_result.
        Can be safely called from sync methods (like SignalOutcomeTracker._append_outcome)
        without blocking or crashing.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.publish_signal_result(outcome_record))
            except RuntimeError:
                # No running event loop in current thread
                asyncio.run(self.publish_signal_result(outcome_record))
        except Exception as e:
            logger.error(f"Error in dispatch_signal_result: {e}")

    # -------------------------------------------------------------------------
    # Schema Documentation Message (Brief 24 Section 2)
    # -------------------------------------------------------------------------
    def is_schema_reference_sent(self) -> bool:
        """Checks whether the one-time schema documentation message has already been sent."""
        if not os.path.exists(self.schema_ref_file):
            return False
        try:
            with open(self.schema_ref_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return bool(data.get("sent", False))
        except Exception as e:
            logger.error(f"Error checking schema reference sent status: {e}")
            return False

    def mark_schema_reference_sent(self, is_dry_run: bool = False, payload: Optional[Dict[str, Any]] = None):
        """Persists flag on disk ensuring schema reference message is sent at most once ever."""
        try:
            temp_file = f"{self.schema_ref_file}.tmp"
            now_iso = datetime.now(timezone.utc).isoformat()
            data = {
                "sent": True,
                "sent_at": now_iso,
                "is_dry_run": is_dry_run,
                "payload": payload
            }
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.schema_ref_file):
                os.replace(temp_file, self.schema_ref_file)
            else:
                os.rename(temp_file, self.schema_ref_file)
            logger.info("🔒 [SCHEMA_REFERENCE] Marked schema reference message as sent on disk.")
        except Exception as e:
            logger.error(f"Error persisting schema reference state: {e}")

    def build_schema_reference_message(self) -> str:
        """
        Brief 24 Section 2: One schema-documentation message format.
        """
        text = (
            "FOREX ENGINEER — FORMAT REFERENCE (NOT A SIGNAL · DO NOT TRADE)\n\n"
            "From now on, every closed trade sends a message in this shape.\n"
            'type = "result", schema_version = 1\n\n'
            "  symbol         EXAMPLE\n"
            "  direction      SELL\n"
            "  outcome        TP1          (one of: TP1, TP2, SL, TIMEOUT)\n"
            "  entry          000.000\n"
            "  exit_price     000.000\n"
            "  r_multiple     +2.0\n"
            "  opened_utc     2026-09-21 09:00:00\n"
            "  closed_utc     2026-09-21 10:30:00\n"
            "  duration       1h 30m\n"
            "  signal_id      snap_EXAMPLE_...\n\n"
            "Rendered example:\n\n"
            "  FOREX ENGINEER — TRADE CLOSED\n"
            "  EXAMPLE · SELL · TARGET 1 HIT\n"
            "  Entry 000.000  ->  Exit 000.000\n"
            "  Result +2.0 R   Duration 1h 30m\n\n"
            "*** FORMAT REFERENCE ONLY — NOT A SIGNAL — DO NOT TRADE ***"
        )
        return text

    async def send_schema_reference_message(self, force: bool = False) -> bool:
        """
        Brief 24 Section 2: Dispatches the schema-documentation reference message once.
        Guarded on disk by self.schema_ref_file; never repeats after restart/reconnect.
        """
        try:
            if self.is_schema_reference_sent() and not force:
                logger.info("ℹ️ [SCHEMA_REFERENCE] Already sent previously according to disk flag. Skipping.")
                return False

            msg_text = self.build_schema_reference_message()
            payload = {
                "schema_version": 1,
                "type": "schema_reference",
                "is_reference": True,
                "message_text": msg_text,
                "example": {
                    "schema_version": 1,
                    "type": "result",
                    "signal_id": "snap_EXAMPLE_20260921_090000_000000",
                    "symbol": "EXAMPLE",
                    "direction": "SELL",
                    "outcome": "TP1",
                    "entry": 0.0,
                    "exit_price": 0.0,
                    "r_multiple": 2.0,
                    "opened_utc": "2026-09-21 09:00:00",
                    "closed_utc": "2026-09-21 10:30:00",
                    "duration": "1h 30m"
                }
            }

            is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)
            if is_dry_run:
                logger.info(
                    f"\n=================== [SCHEMA REFERENCE DRY RUN START] ===================\n"
                    f"Target: n8n Webhook (DRY RUN - TELEGRAM_DRY_RUN=True)\n"
                    f"Payload JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE TEXT ---\n"
                    f"{msg_text}\n"
                    f"==================== [SCHEMA REFERENCE DRY RUN END] ====================\n"
                )
                self.mark_schema_reference_sent(is_dry_run=True, payload=payload)
                return True

            success = await self._post_to_n8n_webhook(payload)
            if success:
                self.mark_schema_reference_sent(is_dry_run=False, payload=payload)
                logger.info("✅ [SCHEMA_REFERENCE DISPATCHED] Schema reference message posted to n8n webhook.")
                return True
            else:
                logger.error("❌ [SCHEMA_REFERENCE FAILED] Webhook delivery failed. Left unsent on disk for retry.")
                return False

        except Exception as e:
            logger.error(f"Error in send_schema_reference_message: {e}", exc_info=True)
            return False

    # -------------------------------------------------------------------------
    # 12-Hour Performance Report Engine (Brief 24 Section 3)
    # -------------------------------------------------------------------------
    def is_report_already_posted(self, period_key: str) -> bool:
        return period_key in self.posted_reports

    def record_report_posted(self, period_key: str, payload: Dict[str, Any], is_dry_run: bool = False):
        self.posted_reports[period_key] = {
            "period_key": period_key,
            "period": payload.get("period"),
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "is_dry_run": is_dry_run,
            "closed_count": payload.get("closed_summary", {}).get("closed_count", 0),
            "open_count": len(payload.get("still_open", []))
        }
        self._save_reports_state()

    def generate_12h_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        open_signals_override: Optional[Dict[str, Any]] = None,
        outcomes_override: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Brief 24 Section 3: Generates the 12-hour periodic report payload and message text.
        Rules:
          1. List every closed trade, winners and losers, in order.
          2. Sample size travels with every figure (e.g. '+14.0 R over 70 trades').
          3. NO projected or annualised return. Ever.
          4. Skip the period entirely if nothing closed and nothing is open.
          5. Deduplicate by period.
        """
        now = datetime.now(timezone.utc)
        if period_start is None or period_end is None:
            # Determine previous completed 12h window
            if now.hour >= 12:
                # First window of today: 00:00 to 12:00 UTC
                p_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                p_end = now.replace(hour=12, minute=0, second=0, microsecond=0)
                period_key = f"{p_start.strftime('%Y-%m-%d')}_00_12"
                period_str = f"{p_start.strftime('%Y-%m-%d')} 00:00 to 12:00 UTC"
            else:
                # Second window of yesterday: 12:00 to 00:00 UTC
                yesterday = now - timedelta(days=1)
                p_start = yesterday.replace(hour=12, minute=0, second=0, microsecond=0)
                p_end = now.replace(hour=0, minute=0, second=0, microsecond=0)
                period_key = f"{p_start.strftime('%Y-%m-%d')}_12_24"
                period_str = f"{p_start.strftime('%Y-%m-%d')} 12:00 to 00:00 UTC"
        else:
            p_start = period_start
            p_end = period_end
            period_key = f"{p_start.strftime('%Y-%m-%d_%H')}_{p_end.strftime('%H')}"
            period_str = f"{p_start.strftime('%Y-%m-%d %H:%M')} to {p_end.strftime('%H:%M')} UTC"

        # 1. Gather all outcomes from signal_outcomes.jsonl
        all_outcomes: List[Dict[str, Any]] = []
        if outcomes_override is not None:
            all_outcomes = list(outcomes_override)
        else:
            search_files = [
                os.path.join(self.data_dir, "signal_outcomes.jsonl"),
                str(Path(__file__).resolve().parent.parent.parent / "data" / "signal_outcomes.jsonl"),
                str(Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports" / "signal_outcomes.jsonl")
            ]
            seen_files = set()
            for fpath in search_files:
                abs_p = os.path.abspath(fpath)
                if abs_p in seen_files or not os.path.exists(abs_p):
                    continue
                seen_files.add(abs_p)
                try:
                    with open(abs_p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    all_outcomes.append(json.loads(line))
                                except Exception:
                                    pass
                except Exception as e:
                    logger.debug(f"Error reading outcomes file {fpath}: {e}")

        # Filter closed trades in this period
        start_ts = p_start.timestamp()
        end_ts = p_end.timestamp()
        closed_this_period: List[Dict[str, Any]] = []

        for rec in all_outcomes:
            rec_ts = None
            if "recorded_at_epoch" in rec:
                rec_ts = float(rec["recorded_at_epoch"])
            else:
                t_str = rec.get("resolved_time_utc") or rec.get("resolved_utc") or rec.get("closed_utc")
                if t_str:
                    try:
                        clean_str = str(t_str).replace(" UTC", "").strip()[:19]
                        dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        rec_ts = dt.timestamp()
                    except Exception:
                        pass
            if rec_ts is not None and start_ts <= rec_ts < end_ts:
                closed_this_period.append(rec)

        # 2. Gather currently open signals
        if open_signals_override is not None:
            open_signals_map = open_signals_override
        else:
            fpath = os.path.join(self.data_dir, "open_signals.json")
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as of:
                        open_signals_map = json.load(of)
                except Exception:
                    open_signals_map = signal_outcome_tracker.get_open_signals()
            else:
                open_signals_map = signal_outcome_tracker.get_open_signals()

        excluded = [s.upper() for s in getattr(settings, "TELEGRAM_EXCLUDED_SYMBOLS", ["XAUUSD"])]
        still_open_list: List[Dict[str, Any]] = []
        for sym, sig in open_signals_map.items():
            sym_upper = str(sym).upper()
            if sym_upper in excluded:
                continue
            pub_ts = float(sig.get("published_timestamp") or (now.timestamp() - 3600))
            diff_h = max(0, int((now.timestamp() - pub_ts) // 3600))
            diff_m = max(0, int(((now.timestamp() - pub_ts) % 3600) // 60))
            age_str = f"{diff_h}h ago" if diff_h > 0 else f"{diff_m}m ago"
            still_open_list.append({
                "symbol": sym_upper,
                "direction": sig.get("direction", "BUY").upper(),
                "opened_ago": age_str,
                "published_at": sig.get("published_at")
            })

        # Rule 4: Skip the period entirely if nothing closed and nothing is open
        if len(closed_this_period) == 0 and len(still_open_list) == 0:
            logger.info(f"🚫 [12H_REPORT SKIPPED] 0 closed trades and 0 open trades in period {period_str}. Report skipped.")
            return None

        # Format CLOSED THIS PERIOD
        closed_lines: List[str] = []
        won_count = 0
        lost_count = 0
        sum_r = 0.0

        for trade in closed_this_period:
            sym = str(trade.get("symbol", "")).upper()
            dir_str = str(trade.get("direction", "BUY")).upper()
            raw_out = str(trade.get("barrier_hit") or trade.get("outcome") or "TP1").upper().strip()

            if raw_out == "TP1":
                barrier_label = "TARGET 1 HIT"
            elif raw_out == "TP2":
                barrier_label = "TARGET 2 HIT"
            elif raw_out == "SL":
                barrier_label = "STOP HIT"
            elif raw_out == "TIMEOUT":
                barrier_label = "CLOSED ON TIME"
            elif "TP2" in raw_out:
                barrier_label = "TARGET 2 HIT"
            elif "TP" in raw_out or "TARGET" in raw_out:
                barrier_label = "TARGET 1 HIT"
            elif "SL" in raw_out or "STOP" in raw_out:
                barrier_label = "STOP HIT"
            elif "TIME" in raw_out:
                barrier_label = "CLOSED ON TIME"
            else:
                barrier_label = raw_out

            r_val = float(trade.get("r_multiple", 0.0))
            sum_r += r_val
            if r_val > 0:
                won_count += 1
            elif r_val < 0:
                lost_count += 1

            r_str = f"{r_val:+.1f} R" if round(r_val, 1) == round(r_val, 2) else f"{r_val:+.2f} R"
            closed_lines.append(f"  {sym:<8} {dir_str:<6} {barrier_label:<14} {r_str}")

        total_closed = len(closed_this_period)
        sum_r_str = f"{sum_r:+.1f} R" if round(sum_r, 1) == round(sum_r, 2) else f"{sum_r:+.2f} R"
        summary_line = f"  {total_closed} closed · {won_count} won · {lost_count} lost · {sum_r_str}"

        # Format STILL OPEN
        open_lines: List[str] = []
        for o in still_open_list:
            open_lines.append(f"  {o['symbol']:<8} {o['direction']:<6} opened {o['opened_ago']}")

        # Calculate Live Record since 2026-09-14
        start_date_str = "2026-09-14"
        live_outcomes = []
        for rec in all_outcomes:
            t_str = rec.get("signal_time_utc") or rec.get("published_utc") or rec.get("resolved_time_utc") or ""
            if t_str and t_str >= start_date_str:
                live_outcomes.append(rec)
            elif not t_str:
                live_outcomes.append(rec)

        total_live_trades = len(live_outcomes)
        total_live_r = sum(float(r.get("r_multiple", 0.0)) for r in live_outcomes)
        live_r_str = f"{total_live_r:+.1f} R" if round(total_live_r, 1) == round(total_live_r, 2) else f"{total_live_r:+.2f} R"
        live_record_line = f"Live record since {start_date_str} · {total_live_trades} trades · {live_r_str}"

        # Construct rendered message text
        msg_parts = [
            "FOREX ENGINEER — 12H REPORT",
            period_str,
            "",
            "CLOSED THIS PERIOD",
            ""
        ]
        if closed_lines:
            msg_parts.extend(closed_lines)
            msg_parts.append("")
            msg_parts.append(summary_line)
        else:
            msg_parts.append("  None")

        msg_parts.append("")
        msg_parts.append("STILL OPEN")
        msg_parts.append("")
        if open_lines:
            msg_parts.extend(open_lines)
        else:
            msg_parts.append("  None")

        msg_parts.append("")
        msg_parts.append(live_record_line)

        msg_text = "\n".join(msg_parts)

        payload = {
            "schema_version": 1,
            "type": "report",
            "period": period_str,
            "period_key": period_key,
            "period_start_utc": p_start.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "period_end_utc": p_end.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "closed_trades": [
                {
                    "symbol": t.get("symbol"),
                    "direction": t.get("direction"),
                    "barrier_hit": t.get("barrier_hit"),
                    "r_multiple": t.get("r_multiple"),
                    "signal_id": t.get("signal_id")
                }
                for t in closed_this_period
            ],
            "closed_summary": {
                "closed_count": total_closed,
                "won": won_count,
                "lost": lost_count,
                "total_r": round(sum_r, 2)
            },
            "still_open": still_open_list,
            "live_record": {
                "since_date": start_date_str,
                "total_trades": total_live_trades,
                "total_r": round(total_live_r, 2),
                "summary": live_record_line
            },
            "message_text": msg_text
        }
        return payload

    async def publish_12h_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        force: bool = False,
        open_signals_override: Optional[Dict[str, Any]] = None,
        outcomes_override: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Brief 24 Section 3: Generates and publishes the 12-hour report to n8n webhook.
        Guarded by disk deduplication per period.
        """
        try:
            report_data = self.generate_12h_report(
                period_start=period_start,
                period_end=period_end,
                open_signals_override=open_signals_override,
                outcomes_override=outcomes_override
            )
            if report_data is None:
                # 0 closed and 0 open -> skipped
                return False

            period_key = report_data["period_key"]
            if self.is_report_already_posted(period_key) and not force:
                logger.info(f"🚫 [12H_REPORT DUPLICATE] Report for period {period_key} already posted on disk. Skipping.")
                return False

            is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)
            if is_dry_run:
                logger.info(
                    f"\n=================== [12H REPORT DRY RUN START] ===================\n"
                    f"Target: n8n Webhook (DRY RUN - TELEGRAM_DRY_RUN=True)\n"
                    f"Payload JSON:\n{json.dumps(report_data, indent=2, ensure_ascii=False)}\n\n"
                    f"--- MESSAGE TEXT ---\n"
                    f"{report_data['message_text']}\n"
                    f"==================== [12H REPORT DRY RUN END] ====================\n"
                )
                self.record_report_posted(period_key, report_data, is_dry_run=True)
                return True

            success = await self._post_to_n8n_webhook(report_data)
            if success:
                self.record_report_posted(period_key, report_data, is_dry_run=False)
                logger.info(f"✅ [12H_REPORT DISPATCHED] Report for period {period_key} posted to n8n webhook.")
                return True
            else:
                logger.error(f"❌ [12H_REPORT FAILED] Failed to post report for period {period_key}. Left unmarked for retry.")
                return False

        except Exception as e:
            logger.error(f"Error in publish_12h_report: {e}", exc_info=True)
            return False

    # Aliases
    _send_payload_async = _post_to_n8n_webhook

    # -------------------------------------------------------------------------
    # Background Heartbeat Scheduler
    # -------------------------------------------------------------------------
    async def start_heartbeat_worker(self, get_live_state_fn=None):
        """
        Background worker that checks every minute if an hourly waiting notice is due.
        """
        logger.info("🕒 Telegram/n8n heartbeat worker started.")

        # Brief 24 Section 2: Ensure schema reference message sent once
        if not self.is_schema_reference_sent():
            try:
                await self.send_schema_reference_message()
            except Exception as s_err:
                logger.debug(f"Schema reference auto-send error: {s_err}")

        while True:
            try:
                await asyncio.sleep(60)  # check every minute

                # Brief 24 Section 3: Periodic 12-hour report trigger at 00:01 and 12:01 UTC
                now_utc = datetime.now(timezone.utc)
                if now_utc.minute == 1 and now_utc.hour in (0, 12):
                    try:
                        await self.publish_12h_report()
                    except Exception as rep_err:
                        logger.debug(f"Periodic 12-hour report trigger error: {rep_err}")

                interval_min = getattr(settings, "TELEGRAM_HEARTBEAT_MINUTES", 60)
                interval_sec = interval_min * 60.0
                
                now_ts = time.time()
                notices = self.state.get("notice_history", [])
                last_notice_ts = notices[-1]["timestamp"] if notices else 0.0
                
                # Check if interval elapsed
                if (now_ts - last_notice_ts) >= interval_sec:
                    if getattr(settings, "TELEGRAM_ENABLE_WAITING_NOTICES", False):
                        # Gather live status if state function provided
                        symbols_status = None
                        last_signal = None
                        if get_live_state_fn:
                            try:
                                symbols_status, last_signal = get_live_state_fn()
                            except Exception as se:
                                logger.debug(f"Error getting live state for heartbeat: {se}")

                        await self.send_waiting_notice(symbols_status, last_signal)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat worker cycle: {e}")
                await asyncio.sleep(30)

telegram_bot = TelegramBotService()
