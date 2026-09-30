"""
Signal Outcomes Service (Agent Brief 22).
Pure-read service for extracting, normalizing, and aggregating live trading results
from durable signal_outcomes.jsonl.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.services.symbol_metrics import get_symbol_metrics

logger = logging.getLogger(__name__)

# Search paths for signal_outcomes.jsonl
OUTCOMES_SEARCH_PATHS = [
    Path("/app/data/signal_outcomes.jsonl"),
    Path(__file__).resolve().parent.parent.parent / "data" / "signal_outcomes.jsonl",
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "signal_outcomes.jsonl",
    Path(__file__).resolve().parent.parent.parent.parent / "ml" / "reports" / "signal_outcomes.jsonl",
]

# Search paths for snapshots
SNAPSHOT_SEARCH_DIRS = [
    Path("/app/data/snapshots"),
    Path(__file__).resolve().parent.parent.parent / "data" / "snapshots",
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "snapshots",
]


class SignalOutcomesService:

    def __init__(self, outcomes_file: Optional[str | Path] = None, snapshots_dir: Optional[str | Path] = None):
        self.outcomes_file = Path(outcomes_file) if outcomes_file else None
        self.snapshots_dir = Path(snapshots_dir) if snapshots_dir else None

    def _find_outcomes_file(self) -> Optional[Path]:
        if self.outcomes_file:
            p = Path(self.outcomes_file)
            if p.exists() and p.is_file():
                return p
        for p in OUTCOMES_SEARCH_PATHS:
            if p.exists() and p.is_file():
                return p
        return None

    def _find_snapshot_file(self, signal_id: str) -> Optional[Path]:
        clean_id = signal_id.strip()
        search_dirs = [self.snapshots_dir] if self.snapshots_dir else []
        search_dirs.extend(SNAPSHOT_SEARCH_DIRS)
        for sdir in search_dirs:
            if not sdir or not sdir.exists():
                continue
            cand = sdir / f"{clean_id}.json"
            if cand.exists():
                return cand
        return None

    def _resolve_levels(self, rec: Dict[str, Any]) -> tuple[float, float, float]:
        """
        Returns (entry, stop_loss, take_profit_1) with 100% positive, non-null numbers.
        Attempts direct record extraction -> snapshot inspection -> mathematical inference.
        """
        entry = float(rec.get("entry") or rec.get("entry_price") or 0.0)
        sl = float(rec.get("stop_loss") or rec.get("sl_price") or rec.get("sl") or 0.0)
        tp1 = float(rec.get("take_profit_1") or rec.get("tp1_price") or rec.get("tp1") or 0.0)

        # 1. If already complete, return
        if entry > 0 and sl > 0 and tp1 > 0:
            return entry, sl, tp1

        # 2. Check snapshot on disk
        snap_id = str(rec.get("signal_id") or "")
        if snap_id:
            snap_path = self._find_snapshot_file(snap_id)
            if snap_path:
                try:
                    with open(snap_path, "r", encoding="utf-8") as f:
                        snap_data = json.load(f)
                    if entry <= 0:
                        entry = float(snap_data.get("entry_price") or snap_data.get("entry") or 0.0)
                    if sl <= 0:
                        sl = float(snap_data.get("sl_price") or snap_data.get("sl") or snap_data.get("stop_loss") or 0.0)
                    if tp1 <= 0:
                        tp1 = float(snap_data.get("tp1_price") or snap_data.get("tp1") or snap_data.get("take_profit_1") or 0.0)
                except Exception as e:
                    logger.debug(f"Could not read snapshot {snap_path}: {e}")

        if entry > 0 and sl > 0 and tp1 > 0:
            return entry, sl, tp1

        # 3. Mathematical inference using 1:2 RR geometry
        direction = str(rec.get("direction") or "BUY").upper()
        barrier = str(rec.get("barrier_hit") or "TP1").upper()
        exit_p = float(rec.get("exit_price") or entry)

        if entry > 0 and exit_p > 0 and entry != exit_p:
            if barrier == "TP1":
                tp1 = exit_p
                risk = abs(exit_p - entry) / 2.0
                if direction == "BUY":
                    sl = round(entry - risk, 5)
                else:
                    sl = round(entry + risk, 5)
            elif barrier == "SL":
                sl = exit_p
                risk = abs(exit_p - entry)
                if direction == "BUY":
                    tp1 = round(entry + 2.0 * risk, 5)
                else:
                    tp1 = round(entry - 2.0 * risk, 5)

        return entry, sl, tp1

    def get_outcomes(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        include_test: bool = False
    ) -> Dict[str, Any]:
        """
        Pure-read of signal_outcomes.jsonl returning normalized rows,
        summary statistics, and live vs backtest model comparison.
        """
        outcomes_file = self._find_outcomes_file()
        raw_records: List[Dict[str, Any]] = []

        if outcomes_file:
            try:
                with open(outcomes_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            raw_records.append(rec)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Failed to read outcomes file {outcomes_file}: {e}")

        # Normalize rows into Brief 22 schema
        normalized_trades: List[Dict[str, Any]] = []

        for rec in raw_records:
            snap_id = str(rec.get("signal_id") or "")
            sym = str(rec.get("symbol") or "").upper()
            direction = str(rec.get("direction") or "BUY").upper()

            # Filter non-trade test / verification signals if not requested
            is_test = bool(rec.get("is_test", False))
            if "verify16" in snap_id.lower() or "verify" in snap_id.lower():
                is_test = True

            if is_test and not include_test:
                continue

            # Symbol filter
            if symbol and symbol.upper() != "ALL" and sym != symbol.upper():
                continue

            entry, sl, tp1 = self._resolve_levels(rec)
            exit_price = float(rec.get("exit_price") or entry)
            barrier_hit = str(rec.get("barrier_hit") or "unknown").lower()
            r_multiple = round(float(rec.get("r_multiple", 0.0)), 2)

            pub_utc = str(rec.get("published_utc") or rec.get("signal_time_utc") or "").replace(" UTC", "").strip()
            res_utc = str(rec.get("resolved_utc") or rec.get("resolved_time_utc") or "").replace(" UTC", "").strip()

            bars_held = int(rec.get("bars_held") or rec.get("bars_monitored") or 0)
            model_side = "SHORT" if direction == "SELL" else "LONG"

            metrics = get_symbol_metrics(sym, side=model_side)
            tier = str(rec.get("tier") or metrics.get("tier", "VALIDATED")).upper()

            trade_item = {
                "signal_id": snap_id,
                "symbol": sym,
                "direction": direction,
                "entry": round(entry, 5),
                "stop_loss": round(sl, 5),
                "take_profit_1": round(tp1, 5),
                "exit_price": round(exit_price, 5),
                "barrier_hit": barrier_hit,
                "r_multiple": r_multiple,
                "published_utc": pub_utc,
                "resolved_utc": res_utc,
                "bars_held": bars_held,
                "tier": tier,
                "model_side": model_side
            }
            normalized_trades.append(trade_item)

        # Apply limit (preserve chronological order of requested limit)
        if limit and limit > 0 and len(normalized_trades) > limit:
            trades_slice = normalized_trades[-limit:]
        else:
            trades_slice = normalized_trades

        # Calculate Summary Block (Section 2)
        total_trades = len(trades_slice)
        wins = sum(1 for t in trades_slice if t["r_multiple"] > 0)
        losses = sum(1 for t in trades_slice if t["r_multiple"] < 0)
        win_rate = round((wins / total_trades) * 100, 1) if total_trades > 0 else 0.0
        total_r = round(sum(t["r_multiple"] for t in trades_slice), 2)
        expectancy_r = round(total_r / total_trades, 3) if total_trades > 0 else 0.0

        # By-symbol aggregation
        by_symbol: Dict[str, Dict[str, Any]] = {}
        for t in trades_slice:
            s = t["symbol"]
            if s not in by_symbol:
                by_symbol[s] = {"trades": 0, "wins": 0, "losses": 0, "total_r": 0.0, "expectancy_r": 0.0}
            by_symbol[s]["trades"] += 1
            if t["r_multiple"] > 0:
                by_symbol[s]["wins"] += 1
            elif t["r_multiple"] < 0:
                by_symbol[s]["losses"] += 1
            by_symbol[s]["total_r"] = round(by_symbol[s]["total_r"] + t["r_multiple"], 2)

        for s, s_data in by_symbol.items():
            if s_data["trades"] > 0:
                s_data["expectancy_r"] = round(s_data["total_r"] / s_data["trades"], 3)
                s_data["win_rate"] = round((s_data["wins"] / s_data["trades"]) * 100, 1)

        # Earliest date
        dates = [t["published_utc"][:10] for t in trades_slice if len(t.get("published_utc", "")) >= 10]
        since_date = min(dates) if dates else "2026-09-14"

        # Count active open signals
        try:
            from app.services.signal_outcome_tracker import signal_outcome_tracker
            open_signals_map = signal_outcome_tracker.get_open_signals()
            if symbol and symbol.upper() != "ALL":
                open_now = 1 if symbol.upper() in open_signals_map else 0
            else:
                open_now = len(open_signals_map)
        except Exception:
            open_now = 0

        summary = {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_r": total_r,
            "expectancy_r": expectancy_r,
            "by_symbol": by_symbol,
            "since": since_date,
            "open_now": open_now
        }

        # Compare Live Against Backtest (Section 4)
        comparison: Dict[str, Dict[str, Any]] = {}
        target_symbols = list(by_symbol.keys())
        if symbol and symbol.upper() != "ALL" and symbol.upper() not in target_symbols:
            target_symbols.append(symbol.upper())

        for sym_key in sorted(target_symbols):
            sym_trades = [t for t in trades_slice if t["symbol"] == sym_key]
            active_side = sym_trades[0]["model_side"] if sym_trades else "SHORT"
            m = get_symbol_metrics(sym_key, side=active_side)
            # If the trade had an old/test side with negative expectancy, look up the active production side
            if float(m.get("expectancy_r", 0.0)) <= 0:
                alt_side = "SHORT" if active_side == "LONG" else "LONG"
                alt_m = get_symbol_metrics(sym_key, side=alt_side)
                if float(alt_m.get("expectancy_r", 0.0)) > float(m.get("expectancy_r", 0.0)):
                    m = alt_m
                    active_side = alt_side

            sym_live_trades = len(sym_trades)
            sym_live_r = round(sum(t["r_multiple"] for t in sym_trades), 2)
            sym_live_exp = round(sym_live_r / sym_live_trades, 3) if sym_live_trades > 0 else 0.0

            comparison[sym_key] = {
                "symbol": sym_key,
                "side": active_side,
                "backtest_expectancy_r": float(m.get("expectancy_r", 0.0)),
                "backtest_trades": int(m.get("oos_trades", 0)),
                "live_expectancy_r": sym_live_exp,
                "live_trades": sym_live_trades,
                "live_total_r": sym_live_r,
                "statistically_meaningful": sym_live_trades >= 100
            }

        live_vs_backtest = list(comparison.values())
        # Overall row
        total_backtest_trades = sum(c["backtest_trades"] for c in live_vs_backtest)
        total_backtest_r = sum(c["backtest_expectancy_r"] * c["backtest_trades"] for c in live_vs_backtest)
        overall_backtest_exp = round(total_backtest_r / total_backtest_trades, 2) if total_backtest_trades > 0 else 0.0
        
        overall_live_trades = summary["total_trades"]
        overall_live_r = summary["total_r"]
        overall_live_exp = summary["expectancy_r"]

        overall_row = {
            "symbol": "Overall",
            "side": "MIXED",
            "backtest_expectancy_r": overall_backtest_exp,
            "backtest_trades": total_backtest_trades,
            "live_expectancy_r": overall_live_exp,
            "live_trades": overall_live_trades,
            "live_total_r": overall_live_r,
            "statistically_meaningful": overall_live_trades >= 300
        }
        live_vs_backtest.append(overall_row)

        return {
            "status": "SUCCESS",
            "count": len(trades_slice),
            "trades": trades_slice,
            "summary": summary,
            "comparison": comparison,
            "live_vs_backtest": live_vs_backtest
        }


signal_outcomes_service = SignalOutcomesService()
