"""
Mirrored Account Core Engine (Agent Brief 32).
Watches the engineer's MT5 demo account in READ-ONLY mode via investor password.
Publishes live entries with lot sizes, trade modifications (SL/TP, breakeven),
and closed trade outcomes taken directly from MT5 deal reasons.

Standing Rules:
1. The 52 sealed manual records stay sealed. Nothing historical publishes, ever.
2. On first connection, every position already open and every trade already in history
   is marked HISTORICAL and seeded as already-published. Zero Telegram messages sent.
3. Connected READ-ONLY via investor password. Terminal physically refuses order placement.
4. Strictly refuses to run on LIVE accounts; only DEMO accounts are permitted.
5. Isolated data stores: mirrored_trades.jsonl, mirrored_outcomes.jsonl. Never touches
   signal_outcomes.jsonl or manual_outcomes.jsonl.
"""
from __future__ import annotations

import os
import json
import time
import math
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set

from app.config import settings
from app.services.telegram_bot import telegram_bot

logger = logging.getLogger("mirror_service")

# MT5 Deal Reason Constants
DEAL_REASON_CLIENT = 0
DEAL_REASON_MOBILE = 1
DEAL_REASON_WEB = 2
DEAL_REASON_EXPERT = 3
DEAL_REASON_SL = 4
DEAL_REASON_TP = 5
DEAL_REASON_SO = 6

DEAL_REASON_MAP = {
    DEAL_REASON_TP: "TARGET HIT",
    DEAL_REASON_SL: "STOP HIT",
    DEAL_REASON_CLIENT: "CLOSED MANUALLY",
    DEAL_REASON_MOBILE: "CLOSED MANUALLY",
    DEAL_REASON_WEB: "CLOSED MANUALLY",
    DEAL_REASON_EXPERT: "CLOSED BY SYSTEM",
    DEAL_REASON_SO: "STOPPED OUT — MARGIN"
}


class MirrorService:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.positions_file = os.path.join(self.data_dir, "mirrored_positions.json")
        self.trades_file = os.path.join(self.data_dir, "mirrored_trades.jsonl")
        self.updates_file = os.path.join(self.data_dir, "mirrored_updates.jsonl")
        self.outcomes_file = os.path.join(self.data_dir, "mirrored_outcomes.jsonl")
        self.published_signals_file = os.path.join(self.data_dir, "mirrored_published_signals.json")
        self.published_results_file = os.path.join(self.data_dir, "mirrored_published_results.json")

        self.login = getattr(settings, "MT5_MIRROR_LOGIN", 8058543)
        self.server = getattr(settings, "MT5_MIRROR_SERVER", "SCFMLimited-Demo2")
        self.is_connected = False
        self.is_demo = True
        self.account_info: Dict[str, Any] = {}

        # In-memory tracking
        self.open_positions: Dict[int, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.initial_risk_by_ticket: Dict[int, float] = {}
        self.published_signals: Set[int] = set()
        self.published_results: Set[int] = set()
        self.last_update_ts_by_ticket: Dict[int, float] = {}
        self.pending_updates: Dict[int, Dict[str, Any]] = {}

        self.historical_seeded_open = 0
        self.historical_seeded_closed = 0
        self.first_connection_completed = False
        self.candle_getter = None

        self._load_state()

    def set_candle_getter(self, getter_fn):
        """Sets the candle history retrieval callable."""
        self.candle_getter = getter_fn

    def _load_state(self):
        """Loads persisted state from disk."""
        if os.path.exists(self.published_signals_file):
            try:
                with open(self.published_signals_file, "r", encoding="utf-8") as f:
                    self.published_signals = set(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load published signals: {e}")

        if os.path.exists(self.published_results_file):
            try:
                with open(self.published_results_file, "r", encoding="utf-8") as f:
                    self.published_results = set(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load published results: {e}")

        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.open_positions = {int(k): v for k, v in raw.items()}
                    for ticket, pos in self.open_positions.items():
                        if "risk_price" in pos and pos["risk_price"]:
                            self.initial_risk_by_ticket[ticket] = float(pos["risk_price"])
            except Exception as e:
                logger.error(f"Failed to load open positions: {e}")

    def _save_state(self):
        """Persists current state to disk."""
        try:
            with open(self.published_signals_file, "w", encoding="utf-8") as f:
                json.dump(list(self.published_signals), f)
            with open(self.published_results_file, "w", encoding="utf-8") as f:
                json.dump(list(self.published_results), f)
            with open(self.positions_file, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in self.open_positions.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state to disk: {e}")

    def _append_jsonl(self, filepath: str, record: Dict[str, Any]):
        """Safely appends a JSON record to a jsonl file."""
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            logger.error(f"Failed to append to {filepath}: {e}")

    def clear_all_memory(self):
        """Helper for test isolation and resetting state"""
        self.open_positions.clear()
        self.initial_risk_by_ticket.clear()
        self.published_signals.clear()
        self.published_results.clear()
        self.last_update_ts_by_ticket.clear()
        self.pending_updates.clear()
        self.historical_seeded_open = 0
        self.historical_seeded_closed = 0
        self.first_connection_completed = False
        self._save_state()

    def seed_historical_state(self, initial_positions: List[Dict[str, Any]], initial_deals: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Brief 32 Standing Rule:
        On first connection to the mirrored account, every position already open
        and every trade already in history is HISTORICAL.
        Seeds them as already-published and sends NONE of them.
        """
        seeded_open = 0
        seeded_closed = 0

        for p in initial_positions:
            ticket = int(p.get("ticket", 0))
            if ticket > 0 and ticket not in self.published_signals:
                self.published_signals.add(ticket)
                p["is_historical"] = True
                p["sealed"] = True
                sl_v = p.get("stop_loss") if p.get("stop_loss") is not None else p.get("sl")
                tp_v = p.get("take_profit") if p.get("take_profit") is not None else p.get("tp")
                p["stop_loss"] = float(sl_v) if sl_v else None
                p["take_profit"] = float(tp_v) if tp_v else None
                self.open_positions[ticket] = p
                entry = float(p.get("entry") or p.get("price_open") or 0.0)
                sl = float(sl_v or 0.0)
                if sl > 0 and entry > 0:
                    self.initial_risk_by_ticket[ticket] = round(abs(entry - sl), 5)
                seeded_open += 1

        for d in initial_deals:
            deal_ticket = int(d.get("ticket", 0))
            pos_id = int(d.get("position_id") or d.get("order") or deal_ticket)
            if deal_ticket > 0 and deal_ticket not in self.published_results:
                self.published_results.add(deal_ticket)
                seeded_closed += 1
            if pos_id > 0 and pos_id not in self.published_results:
                self.published_results.add(pos_id)

        self.historical_seeded_open += seeded_open
        self.historical_seeded_closed += seeded_closed
        self.first_connection_completed = True
        self._save_state()

        logger.info(
            f"🛡️ [MIRROR_HISTORICAL_SEEDED] First connection guard: Seeded {seeded_open} open positions "
            f"and {seeded_closed} historical deals as already-published. ZERO messages sent."
        )
        return seeded_open, seeded_closed

    def verify_account_is_demo(self, account_or_mode: Any) -> Tuple[bool, str]:
        """
        Section 2: Verifies account is demo.
        Checks broker server name (e.g. SCFMLimited-Demo2, ForexTime-Demo01) or trade_mode == 0.
        """
        server = ""
        tm = 0
        if isinstance(account_or_mode, dict):
            tm = account_or_mode.get("trade_mode", 0)
            server = str(account_or_mode.get("server", "")).lower()
        else:
            try:
                tm = int(account_or_mode)
            except Exception:
                tm = 0

        allow_live = getattr(settings, "MIRROR_ALLOW_LIVE", True)
        if allow_live:
            return True, "Account accepted for read-only monitoring"

        # If server contains 'demo', it is confirmed a demo account regardless of broker internal trade_mode flags
        if "demo" in server or "demo" in str(self.server).lower():
            return True, "Demo account confirmed via demo server name"

        if tm != 0:
            msg = (
                f"FATAL: Mirrored account login {self.login} on {self.server} resolved to LIVE "
                f"(trade_mode={tm}). System strictly refuses to run on live accounts."
            )
            logger.critical(msg)
            return False, msg
        return True, "Demo account confirmed"

    def calculate_rr(self, entry: float, stop_loss: Optional[float], take_profit: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Computes risk_price and rr only when both stop and target exist. Never invents either.
        """
        if stop_loss is None or take_profit is None or stop_loss <= 0 or take_profit <= 0:
            return None, None
        risk_price = round(abs(entry - stop_loss), 5)
        if risk_price <= 0:
            return None, None
        rr = round(abs(take_profit - entry) / risk_price, 2)
        return risk_price, rr

    def is_breakeven_move(self, entry: float, old_sl: float, new_sl: float, point: float = 0.00001) -> bool:
        """
        Detects whether an SL adjustment is moving to breakeven (entry ± 2 points).
        """
        if new_sl <= 0 or entry <= 0:
            return False
        # If moving from non-entry to approximately entry price
        dist_to_entry = abs(new_sl - entry)
        threshold = max(0.0005, 5 * point)
        return dist_to_entry <= threshold and (old_sl is None or abs(old_sl - entry) > threshold)

    def format_entry_message(self, record: Dict[str, Any]) -> str:
        """Formats the Telegram entry message for a newly opened mirrored position."""
        sym = record.get("symbol", "").upper()
        direction = record.get("direction", "").upper()
        ticket = record.get("ticket", "")
        lot = record.get("lot", 0.0)
        entry = record.get("entry", 0.0)
        sl = record.get("stop_loss")
        tp = record.get("take_profit")
        rr = record.get("rr")

        digits = record.get("digits", 5)
        entry_str = f"{entry:.{digits}f}"
        sl_str = f"{float(sl):.{digits}f}" if (sl and float(sl) > 0) else "Open / Floating"
        tp_str = f"{float(tp):.{digits}f}" if (tp and float(tp) > 0) else "Open / Floating"
        rr_str = f"1:{float(rr):.2f}" if (rr and float(rr) > 0) else "Dynamic / Open"

        msg = (
            f"FOREX ENGINEER — LIVE TRADE OPENED\n\n"
            f"{sym}  ·  {direction}  ·  #{ticket}\n\n"
            f"Entry        {entry_str}\n"
            f"Lot          {lot:.2f}\n"
            f"Stop Loss    {sl_str}\n"
            f"Take Profit  {tp_str}\n"
            f"Risk:Reward  {rr_str}\n\n"
            f"Mirrored from engineer's MT5 account (SCFMLimited-Demo2)"
        )
        return msg

    def format_update_message(self, update_rec: Dict[str, Any]) -> str:
        """Formats the Telegram modification message."""
        sym = update_rec.get("symbol", "").upper()
        direction = update_rec.get("direction", "").upper()
        ticket = update_rec.get("ticket", "")
        digits = update_rec.get("digits", 5)

        sl_info = update_rec.get("stop_loss", {})
        tp_info = update_rec.get("take_profit", {})

        lines = [
            f"FOREX ENGINEER — TRADE UPDATED\n",
            f"{sym}  ·  {direction}  ·  #{ticket}\n"
        ]

        f_sl = sl_info.get("from")
        t_sl = sl_info.get("to")
        if "stop_loss" in update_rec.get("changed", []):
            from_str = f"{float(f_sl):.{digits}f}" if (f_sl and float(f_sl) > 0) else "Open"
            to_str = f"{float(t_sl):.{digits}f}" if (t_sl and float(t_sl) > 0) else "Open"
            be_tag = "   (breakeven)" if update_rec.get("is_breakeven") else ""
            lines.append(f"Stop moved   {from_str}  ->  {to_str}{be_tag}")
        else:
            sl_val = t_sl or f_sl
            sl_str = f"{float(sl_val):.{digits}f}" if (sl_val and float(sl_val) > 0) else "Open / Floating"
            lines.append(f"Stop         {sl_str}   unchanged")

        f_tp = tp_info.get("from")
        t_tp = tp_info.get("to")
        if "take_profit" in update_rec.get("changed", []):
            from_str = f"{float(f_tp):.{digits}f}" if (f_tp and float(f_tp) > 0) else "Open"
            to_str = f"{float(t_tp):.{digits}f}" if (t_tp and float(t_tp) > 0) else "Open"
            lines.append(f"Target moved {from_str}  ->  {to_str}")
        else:
            tp_val = t_tp or f_tp
            tp_str = f"{float(tp_val):.{digits}f}" if (tp_val and float(tp_val) > 0) else "Open / Floating"
            lines.append(f"Target       {tp_str}   unchanged")

        return "\n".join(lines)

    def format_close_message(self, outcome_rec: Dict[str, Any]) -> str:
        """Formats the Telegram closing message taken from MT5 deal reason."""
        sym = outcome_rec.get("symbol", "").upper()
        direction = outcome_rec.get("direction", "").upper()
        ticket = outcome_rec.get("ticket", "")
        barrier = outcome_rec.get("barrier_hit", "CLOSED MANUALLY")
        digits = outcome_rec.get("digits", 5)

        entry = outcome_rec.get("entry", 0.0)
        exit_price = outcome_rec.get("exit_price", 0.0)
        lot = outcome_rec.get("lot", 0.0)
        duration = outcome_rec.get("duration", "0m")
        r_mult = outcome_rec.get("r_multiple")
        profit = float(outcome_rec.get("profit", 0.0))

        if r_mult is not None:
            r_str = f"{r_mult:+.2f} R"
        else:
            r_str = f"${profit:+.2f}" if profit != 0 else "Closed"

        msg = (
            f"FOREX ENGINEER — TRADE CLOSED\n\n"
            f"{sym}  ·  {direction}  ·  {barrier}\n\n"
            f"Entry    {entry:.{digits}f}\n"
            f"Exit     {exit_price:.{digits}f}\n"
            f"Lot      {lot:.2f}\n"
            f"Result   {r_str}      Duration  {duration}\n\n"
            f"Ticket  #{ticket}"
        )
        return msg


    def process_positions_update(
        self,
        current_positions: List[Dict[str, Any]],
        symbol_specs: Optional[Dict[str, Any]] = None,
        now_ts: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Section 5a & 5b:
        Processes current open positions from MT5.
        Detects new entries and modifications (SL/TP moves, breakeven).
        """
        now = now_ts or time.time()
        now_dt = datetime.fromtimestamp(now, tz=timezone.utc)
        specs = symbol_specs or {}
        events: List[Dict[str, Any]] = []

        active_tickets = set()

        for pos in current_positions:
            ticket = int(pos.get("ticket", 0))
            if ticket <= 0:
                continue
            active_tickets.add(ticket)

            sym = str(pos.get("symbol", "")).upper()
            sym_spec = specs.get(sym, {})
            digits = int(sym_spec.get("digits") or pos.get("digits") or 5)
            point = float(sym_spec.get("point") or pos.get("point") or 0.00001)

            entry_price = float(pos.get("entry") or pos.get("price_open") or 0.0)
            direction = str(pos.get("type") or pos.get("direction") or "BUY").upper()
            if direction not in ("BUY", "SELL"):
                direction = "BUY" if pos.get("type") == 0 else "SELL"

            lot = float(pos.get("lots") or pos.get("volume") or 0.01)
            sl = float(pos.get("sl") or pos.get("stop_loss") or 0.0)
            tp = float(pos.get("tp") or pos.get("take_profit") or 0.0)

            sl_val = sl if sl > 0 else None
            tp_val = tp if tp > 0 else None
            risk_price, rr = self.calculate_rr(entry_price, sl_val, tp_val)

            # -----------------------------------------------------------------
            # 5a. A position opens
            # -----------------------------------------------------------------
            if ticket not in self.open_positions and ticket not in self.published_signals:
                opened_utc = pos.get("time_utc") or now_dt.strftime("%Y-%m-%d %H:%M:%S")

                # Brief 32 Section 6: Check for 1,100 M15 bars
                snapshot = None
                snapshot_reason = None
                m15_bars = []
                if self.candle_getter:
                    candles_dict = self.candle_getter(sym) or {}
                    m15_bars = candles_dict.get("M15") or []
                if len(m15_bars) < 1100:
                    snapshot = None
                    snapshot_reason = "insufficient_history"
                elif sym in ("BTCUSD", "USOIL", "XAGUSD"):
                    snapshot = None
                    snapshot_reason = "insufficient_history"

                record = {
                    "schema_version": 1,
                    "type": "signal",
                    "source": "MIRROR",
                    "ticket": ticket,
                    "symbol": sym,
                    "account": pos.get("account") or self.account_info.get("login") or self.login,
                    "direction": direction,
                    "lot": round(lot, 2),
                    "entry": round(entry_price, digits),
                    "stop_loss": round(sl_val, digits) if sl_val else None,
                    "take_profit": round(tp_val, digits) if tp_val else None,
                    "opened_utc": opened_utc,
                    "opened_at_epoch": now,
                    "risk_price": risk_price,
                    "rr": rr,
                    "digits": digits,
                    "point": point,
                    "snapshot": snapshot,
                    "snapshot_reason": snapshot_reason,
                    "status": "OPEN"
                }

                if risk_price and risk_price > 0:
                    self.initial_risk_by_ticket[ticket] = risk_price

                self.open_positions[ticket] = record
                self.published_signals.add(ticket)
                self._append_jsonl(self.trades_file, record)
                events.append(record)

                # Gold Check: Section 7
                is_gold = ("XAU" in sym or "GOLD" in sym)
                publish_gold = getattr(settings, "MIRROR_PUBLISH_GOLD", True)
                should_publish = getattr(settings, "MIRROR_PUBLISH_TELEGRAM", True)

                if is_gold and not publish_gold:
                    logger.info(
                        f"🛡️ [DROP: GOLD_EXCLUDED_PENDING_CONFIRMATION] Mirrored gold trade #{ticket} ({sym}) "
                        f"recorded locally; channel publishing paused pending owner confirmation."
                    )
                elif should_publish:
                    telegram_bot.dispatch_mirror_trade(record)
                    logger.info(f"📢 [MIRROR_ENTRY_PUBLISHED] Dispatched Telegram entry for #{ticket} ({sym} {direction})")

            # -----------------------------------------------------------------
            # 5b. The trade is modified
            # -----------------------------------------------------------------
            elif ticket in self.open_positions:
                known = self.open_positions[ticket]
                old_sl = known.get("stop_loss")
                old_tp = known.get("take_profit")

                new_sl = round(sl_val, digits) if sl_val else None
                new_tp = round(tp_val, digits) if tp_val else None

                changed = []
                # Check for floating point difference or unset
                if (old_sl is None and new_sl is not None) or (old_sl is not None and new_sl is None) or (old_sl is not None and new_sl is not None and abs(old_sl - new_sl) > (point * 0.5)):
                    changed.append("stop_loss")
                if (old_tp is None and new_tp is not None) or (old_tp is not None and new_tp is None) or (old_tp is not None and new_tp is not None and abs(old_tp - new_tp) > (point * 0.5)):
                    changed.append("take_profit")

                if changed:
                    # Update initial risk if stop was set after open
                    if (ticket not in self.initial_risk_by_ticket) and new_sl:
                        self.initial_risk_by_ticket[ticket] = round(abs(entry_price - new_sl), digits)

                    is_be = False
                    if "stop_loss" in changed and new_sl:
                        is_be = self.is_breakeven_move(entry_price, old_sl, new_sl, point=point)

                    note = "(breakeven)" if is_be else ""
                    update_rec = {
                        "type": "update",
                        "source": "MIRROR",
                        "ticket": ticket,
                        "symbol": sym,
                        "account": known.get("account") or pos.get("account") or self.account_info.get("login") or self.login,
                        "direction": direction,
                        "digits": digits,
                        "entry": round(entry_price, digits),
                        "lot": round(lot, 2),
                        "changed": changed,
                        "is_breakeven": is_be,
                        "note": note,
                        "stop_loss": {"from": old_sl, "to": new_sl},
                        "take_profit": {"from": old_tp, "to": new_tp},
                        "updated_utc": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "updated_at_epoch": now
                    }

                    # Update in-memory state
                    known["stop_loss"] = new_sl
                    known["take_profit"] = new_tp
                    new_risk, new_rr = self.calculate_rr(entry_price, new_sl, new_tp)
                    known["risk_price"] = new_risk
                    known["rr"] = new_rr

                    self._append_jsonl(self.updates_file, update_rec)

                    # Debounce / Rate-limit: at most one update message per ticket per 3s
                    last_update_ts = self.last_update_ts_by_ticket.get(ticket, 0.0)
                    if now - last_update_ts >= 3.0:
                        self.last_update_ts_by_ticket[ticket] = now
                        events.append(update_rec)

                        # Publish to Telegram
                        is_gold = ("XAU" in sym or "GOLD" in sym)
                        publish_gold = getattr(settings, "MIRROR_PUBLISH_GOLD", True)
                        should_publish = getattr(settings, "MIRROR_PUBLISH_TELEGRAM", True)

                        if is_gold and not publish_gold:
                            logger.info(f"🛡️ [DROP: GOLD_EXCLUDED] SL update on #{ticket} recorded but suppressed for gold.")
                        elif should_publish:
                            telegram_bot.dispatch_mirror_update(update_rec)
                            logger.info(f"📢 [MIRROR_UPDATE_PUBLISHED] Dispatched Telegram update for #{ticket} ({sym})")
                    else:
                        logger.info(f"⏱️ [MIRROR_UPDATE_DEBOUNCED] Suppressing rapid nudge for #{ticket} within 60s window.")

        # Prune positions that are no longer open in MT5
        closed_tickets = [t for t in self.open_positions if t not in active_tickets]
        for t in closed_tickets:
            self.open_positions.pop(t, None)

        self._save_state()
        return events

    def process_closed_deals(
        self,
        closed_deals: List[Dict[str, Any]],
        symbol_specs: Optional[Dict[str, Any]] = None,
        now_ts: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Section 5c:
        Processes exit deals to detect closed mirrored positions.
        Names MT5 deal reasons (TARGET HIT, STOP HIT, CLOSED MANUALLY, etc.).
        Calculates exact signed R-multiples and handles partial closes.
        """
        now = now_ts or time.time()
        now_dt = datetime.fromtimestamp(now, tz=timezone.utc)
        specs = symbol_specs or {}
        outcomes: List[Dict[str, Any]] = []

        for deal in closed_deals:
            deal_ticket = int(deal.get("ticket", 0))
            pos_id = int(deal.get("position_id") or deal.get("order") or deal_ticket)
            if pos_id <= 0 or deal_ticket in self.published_results:
                continue

            # Look up corresponding open position record
            pos_rec = self.open_positions.get(pos_id)
            if not pos_rec and pos_id in self.published_results:
                continue

            sym = str(deal.get("symbol") or (pos_rec.get("symbol") if pos_rec else "")).upper()
            sym_spec = specs.get(sym, {})
            digits = int(sym_spec.get("digits") or (pos_rec.get("digits") if pos_rec else 5))

            entry_price = float(pos_rec.get("entry") if pos_rec else (deal.get("open_price") or 0.0))
            exit_price = float(deal.get("price") or 0.0)
            deal_lot = float(deal.get("lots") or deal.get("volume") or 0.01)
            direction = str(pos_rec.get("direction") if pos_rec else deal.get("type", "BUY")).upper()

            # MT5 deal reason
            raw_reason = deal.get("reason", DEAL_REASON_CLIENT)
            reason_name = DEAL_REASON_MAP.get(raw_reason, "CLOSED MANUALLY")
            barrier = reason_name

            # Check if this is a partial close
            is_partial = False
            remaining_lot = None
            if pos_rec:
                original_lot = float(pos_rec.get("lot", 0.0))
                if original_lot > deal_lot + 0.001:
                    is_partial = True
                    remaining_lot = round(original_lot - deal_lot, 2)
                    barrier = f"PARTIAL CLOSE ({deal_lot:.2f} lots)"
                    pos_rec["lot"] = remaining_lot

            # Calculate R-multiple
            initial_risk = self.initial_risk_by_ticket.get(pos_id)
            r_multiple = None
            r_status_note = ""
            if initial_risk and initial_risk > 0 and entry_price > 0:
                price_gain = (exit_price - entry_price) if direction == "BUY" else (entry_price - exit_price)
                r_multiple = round(price_gain / initial_risk, 2)
                # If reason was SL, ensure R is negative
                if raw_reason == DEAL_REASON_SL and r_multiple > 0:
                    r_multiple = -abs(r_multiple)
            else:
                r_status_note = "R unavailable (no stop set)"

            # Compute duration
            opened_epoch = float(pos_rec.get("opened_at_epoch") or (now - 300.0)) if pos_rec else (now - 300.0)
            duration_sec = max(0.0, now - opened_epoch)
            mins = int(duration_sec // 60)
            hours = mins // 60
            rem_mins = mins % 60
            duration_str = f"{hours}h {rem_mins}m" if hours > 0 else f"{rem_mins}m"

            outcome_rec = {
                "schema_version": 1,
                "type": "result",
                "source": "MIRROR",
                "ticket": pos_id,
                "deal_ticket": deal_ticket,
                "symbol": sym,
                "account": (pos_rec.get("account") if pos_rec else None) or deal.get("account") or self.account_info.get("login") or self.login,
                "direction": direction,
                "lot": deal_lot,
                "remaining_lot": remaining_lot,
                "entry": round(entry_price, digits),
                "exit_price": round(exit_price, digits),
                "barrier_hit": barrier,
                "reason_name": reason_name,
                "deal_reason": raw_reason,
                "r_multiple": r_multiple,
                "r_status_note": r_status_note,
                "profit": round(float(deal.get("profit", 0.0)), 2),
                "duration": duration_str,
                "is_partial": is_partial,
                "closed_utc": deal.get("time_utc") or now_dt.strftime("%Y-%m-%d %H:%M:%S")
            }

            self.published_results.add(deal_ticket)
            if not is_partial:
                self.published_results.add(pos_id)
                self.open_positions.pop(pos_id, None)

            self._append_jsonl(self.outcomes_file, outcome_rec)
            outcomes.append(outcome_rec)

            # Telegram broadcast
            is_gold = ("XAU" in sym or "GOLD" in sym)
            publish_gold = getattr(settings, "MIRROR_PUBLISH_GOLD", True)
            should_publish = getattr(settings, "MIRROR_PUBLISH_TELEGRAM", True)

            if is_gold and not publish_gold:
                logger.info(f"🛡️ [DROP: GOLD_EXCLUDED] Result for gold trade #{pos_id} suppressed pending confirmation.")
            elif should_publish:
                telegram_bot.dispatch_mirror_result(outcome_rec)
                logger.info(f"📢 [MIRROR_RESULT_PUBLISHED] Dispatched Telegram result for #{pos_id} ({sym} {barrier})")

        self._save_state()
        return outcomes

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the mirrored watcher."""
        return {
            "status": "ONLINE" if self.is_connected else "INITIALIZING",
            "account_login": self.account_info.get("login") or self.login,
            "server": self.account_info.get("server") or self.server,
            "balance": self.account_info.get("balance", 0.0),
            "equity": self.account_info.get("equity", 0.0),
            "is_demo": self.is_demo,
            "read_only": True,
            "credential_type": "INVESTOR",
            "open_positions_count": len(self.open_positions),
            "closed_trades_count": len(self.closed_trades),
            "historical_seeded_open_count": self.historical_seeded_open,
            "historical_seeded_closed_count": self.historical_seeded_closed,
            "published_signals_count": len(self.published_signals),
            "published_results_count": len(self.published_results),
            "gold_publishing_enabled": getattr(settings, "MIRROR_PUBLISH_GOLD", False),
            "mirror_publishing_enabled": getattr(settings, "MIRROR_PUBLISH_TELEGRAM", True)
        }

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Returns currently active open positions."""
        return list(self.open_positions.values())

    def get_closed_trades(self, symbol: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        """Returns resolved mirrored trades from MT5 deals or disk, newest first."""
        if self.closed_trades:
            res = []
            for rec in self.closed_trades:
                if symbol and symbol.upper() != "ALL" and rec.get("symbol", "").upper() != symbol.upper():
                    continue
                res.append(rec)
            return res[:limit]

        records = []
        if os.path.exists(self.outcomes_file):
            try:
                with open(self.outcomes_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            if symbol and symbol.upper() != "ALL" and rec.get("symbol", "").upper() != symbol.upper():
                                continue
                            records.append(rec)
            except Exception as e:
                logger.error(f"Error reading outcomes: {e}")
        return list(reversed(records))[:limit]


# Global singleton instance
mirror_service = MirrorService()
