import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone

from app.config import settings
from app.services.telegram_bot import telegram_bot

logger = logging.getLogger(__name__)

class TradeEventService:
    """
    Agent Brief 19: Publishes executed MT5 trades (trade_opened, trade_closed)
    to Telegram via n8n webhook.
    
    Guards & Invariants:
    1. Deduplication by ticket: exactly one trade_opened and one trade_closed per ticket, ever.
    2. Startup seeding: existing open positions at startup are marked seen, never published.
    3. Rate cap: at most 10 trades per hour and 60 per day (separate counter).
    4. Excluded symbols: XAUUSD/Gold trades are never published.
    5. Feature flag: gated behind TELEGRAM_PUBLISH_TRADES (default False).
    6. Silent failure: webhook errors never interrupt bridge execution.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            self.data_dir = os.getenv("DATA_DIR", "/app/data")
            if not os.path.exists(self.data_dir):
                # Fallback to local data dir
                local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
                os.makedirs(local_dir, exist_ok=True)
                self.data_dir = local_dir
        else:
            self.data_dir = data_dir

        self.state_file = os.path.join(self.data_dir, "telegram_posted_trades.json")
        self.state: Dict[str, Any] = {
            "opened_tickets": {},
            "closed_tickets": {},
            "active_positions": {},
            "trade_history": []
        }
        self.startup_seeded: bool = False
        self._load_state()

    def _load_state(self):
        """Loads persisted trade state from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.state["opened_tickets"] = loaded.get("opened_tickets", {})
                        self.state["closed_tickets"] = loaded.get("closed_tickets", {})
                        self.state["active_positions"] = loaded.get("active_positions", {})
                        self.state["trade_history"] = loaded.get("trade_history", [])
                logger.info(
                    f"Loaded trade event state: {len(self.state['opened_tickets'])} opened, "
                    f"{len(self.state['closed_tickets'])} closed"
                )
            except Exception as e:
                logger.error(f"Error loading {self.state_file}: {e}")

    def _save_state(self):
        """Atomically saves state to disk."""
        try:
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.state_file):
                os.replace(temp_file, self.state_file)
            else:
                os.rename(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Error saving {self.state_file}: {e}")

    def seed_initial_positions(self, open_trades: List[Dict[str, Any]]):
        """
        Brief 19 Guard 2:
        On startup, do not publish existing positions. Record them as already seen.
        Only genuinely new tickets publish.
        """
        if self.startup_seeded:
            return

        newly_seeded = 0
        for t in open_trades:
            ticket = t.get("ticket")
            if not ticket:
                continue
            ticket_str = str(ticket)
            if ticket_str not in self.state["opened_tickets"]:
                self.state["opened_tickets"][ticket_str] = {
                    "ticket": int(ticket),
                    "symbol": str(t.get("symbol", "")).upper(),
                    "direction": str(t.get("type", "BUY")).upper(),
                    "entry": float(t.get("entry", t.get("open_price", 0.0)) or 0.0),
                    "sl": float(t.get("sl", 0.0) or 0.0) if t.get("sl") else None,
                    "tp": float(t.get("tp", 0.0) or 0.0) if t.get("tp") else None,
                    "volume": float(t.get("lots", t.get("volume", 0.01)) or 0.01),
                    "seeded_at_startup": True,
                    "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                }
                newly_seeded += 1

            # Also cache in active_positions for future close tracking
            if ticket_str not in self.state["active_positions"]:
                self.state["active_positions"][ticket_str] = {
                    "ticket": int(ticket),
                    "symbol": str(t.get("symbol", "")).upper(),
                    "direction": str(t.get("type", "BUY")).upper(),
                    "entry": float(t.get("entry", t.get("open_price", 0.0)) or 0.0),
                    "sl": float(t.get("sl", 0.0) or 0.0) if t.get("sl") else None,
                    "tp": float(t.get("tp", 0.0) or 0.0) if t.get("tp") else None,
                    "volume": float(t.get("lots", t.get("volume", 0.01)) or 0.01),
                    "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                }

        self.startup_seeded = True
        if newly_seeded > 0:
            self._save_state()
            logger.info(f"🛡️ [TRADE_EVENTS] Seeded {newly_seeded} pre-existing open positions at startup as seen.")

    def _check_rate_limit(self) -> Tuple[bool, str]:
        """
        Brief 19 Guard 3:
        Trades have their own counter, separate from signals and notices:
        at most 10 per hour and 60 per day.
        """
        now_ts = time.time()
        cutoff_24h = now_ts - 86400.0
        cutoff_1h = now_ts - 3600.0

        history = self.state.get("trade_history", [])
        history = [h for h in history if h.get("timestamp", 0) >= cutoff_24h]
        self.state["trade_history"] = history

        trades_1h = [h for h in history if h.get("timestamp", 0) >= cutoff_1h]
        max_hourly = getattr(settings, "TELEGRAM_MAX_HOURLY_TRADES", 10)
        max_daily = getattr(settings, "TELEGRAM_MAX_DAILY_TRADES", 60)

        if len(trades_1h) >= max_hourly:
            reason = f"Hourly trade rate cap of {max_hourly} reached ({len(trades_1h)} in last hour)"
            logger.error(f"🚨 [TRADE_RATE_LIMIT_TRIPPED] {reason}")
            return False, reason

        if len(history) >= max_daily:
            reason = f"Daily trade rate cap of {max_daily} reached ({len(history)} in last 24h)"
            logger.error(f"🚨 [TRADE_RATE_LIMIT_TRIPPED] {reason}")
            return False, reason

        return True, ""

    def _record_trade_rate(self, ticket: int, event_type: str):
        """Records timestamp for trade event rate limiting."""
        self.state.setdefault("trade_history", []).append({
            "timestamp": time.time(),
            "ticket": ticket,
            "event": event_type
        })
        self._save_state()

    def _get_precision(self, symbol: str) -> int:
        sym = symbol.upper()
        if "JPY" in sym:
            return 3
        if "XAU" in sym or "GOLD" in sym:
            return 2
        return 5

    def _is_symbol_eligible(self, symbol: str) -> bool:
        """Verifies symbol is in the 11 validated forex symbols and not Gold."""
        sym = symbol.upper()
        if sym in getattr(settings, "TELEGRAM_EXCLUDED_SYMBOLS", ["XAUUSD"]) or "XAU" in sym or "GOLD" in sym:
            return False
        valid_symbols = telegram_bot.get_validated_symbols()
        return sym in valid_symbols

    # -------------------------------------------------------------------------
    # Formatting Helpers
    # -------------------------------------------------------------------------
    def format_trade_opened_message(
        self,
        symbol: str,
        direction: str,
        volume: float,
        entry: float,
        sl: Optional[float],
        tp: Optional[float],
        ticket: int,
        time_utc_str: str,
        source_label: str = "rule engine (not the ML model)"
    ) -> str:
        """
        Brief 19 Section 4: Message wording for trade_opened.
        """
        digits = self._get_precision(symbol)
        entry_str = f"{entry:.{digits}f}"
        sl_str = f"{sl:.{digits}f}" if (sl and sl > 0) else "None"
        tp_str = f"{tp:.{digits}f}" if (tp and tp > 0) else "None"

        # Format UTC time string: e.g. 2026-09-14 09:40 UTC
        clean_time = time_utc_str.replace("T", " ")
        if len(clean_time) >= 16:
            clean_time = clean_time[:16]
        if not clean_time.endswith("UTC"):
            clean_time = f"{clean_time} UTC"

        msg = (
            f"FOREX ENGINEER — TRADE OPENED\n\n"
            f"{symbol}  ·  {direction}  ·  {volume:.2f} lot\n\n"
            f"Entry        {entry_str}\n"
            f"Stop Loss    {sl_str}\n"
            f"Take Profit  {tp_str}\n\n"
            f"Opened by: {source_label}\n"
            f"Ticket     {ticket}\n"
            f"Time       {clean_time}"
        )
        return msg

    def format_trade_closed_message(
        self,
        symbol: str,
        direction: str,
        outcome_label: str,
        entry: float,
        exit_price: float,
        r_multiple: Optional[float],
        profit_currency: float,
        ticket: int,
        source_label: str = "rule engine (not the ML model)"
    ) -> str:
        """
        Brief 19 Section 4: Message wording for trade_closed.
        """
        digits = self._get_precision(symbol)
        entry_str = f"{entry:.{digits}f}"
        exit_str = f"{exit_price:.{digits}f}"

        if r_multiple is not None:
            r_str = f"({r_multiple:+.1f} R)"
        else:
            r_str = f"(${profit_currency:+.2f})"

        msg = (
            f"FOREX ENGINEER — TRADE CLOSED\n\n"
            f"{symbol}  ·  {direction}  ·  {outcome_label}\n"
            f"Entry {entry_str}  ->  Exit {exit_str}   {r_str}\n\n"
            f"Opened by: {source_label}\n"
            f"Ticket     {ticket}"
        )
        return msg

    # -------------------------------------------------------------------------
    # Ingestion Handlers
    # -------------------------------------------------------------------------
    async def on_positions_update(self, raw_trades: List[Dict[str, Any]]):
        """
        Called when ACCOUNT_UPDATE arrives with active positions.
        """
        if not self.startup_seeded:
            self.seed_initial_positions(raw_trades)
            return

        for t in raw_trades:
            ticket = t.get("ticket")
            if not ticket:
                continue
            ticket_str = str(ticket)

            # Check deduplication
            if ticket_str in self.state["opened_tickets"]:
                continue

            symbol = str(t.get("symbol", "")).upper()
            if not self._is_symbol_eligible(symbol):
                continue

            await self.publish_trade_opened(t)

    async def on_closed_trades_update(self, incoming_closed: List[Dict[str, Any]]):
        """
        Called when ACCOUNT_UPDATE arrives with closed trades.
        """
        for ct in incoming_closed:
            ticket = ct.get("ticket")
            if not ticket:
                continue
            ticket_str = str(ticket)

            # Check deduplication
            if ticket_str in self.state["closed_tickets"]:
                continue

            symbol = str(ct.get("symbol", "")).upper()
            if not self._is_symbol_eligible(symbol):
                continue

            await self.publish_trade_closed(ct)

    # -------------------------------------------------------------------------
    # Publisher Methods
    # -------------------------------------------------------------------------
    async def publish_trade_opened(self, trade_dict: Dict[str, Any], force: bool = False) -> bool:
        """
        Publishes a trade_opened event to n8n webhook.
        """
        ticket = trade_dict.get("ticket")
        if not ticket:
            return False
        ticket_str = str(ticket)

        # 1. Deduplication Guard
        if not force and ticket_str in self.state["opened_tickets"]:
            logger.info(f"🚫 [TRADE_DROP: DUPLICATE_OPEN] Ticket #{ticket} already published as opened.")
            return False

        symbol = str(trade_dict.get("symbol", "")).upper()
        # 2. Exclusion Guard
        if not force and not self._is_symbol_eligible(symbol):
            logger.info(f"🚫 [TRADE_DROP: EXCLUDED_SYMBOL] Symbol {symbol} not eligible for trade publishing.")
            return False

        # 3. Rate Limit Guard
        if not force:
            allowed, reason = self._check_rate_limit()
            if not allowed:
                logger.error(f"🚨 [TRADE_DROP: RATE_LIMIT] Refusing trade_opened for #{ticket}: {reason}")
                return False

        # 4. Feature Flag Guard
        if not force and not getattr(settings, "TELEGRAM_PUBLISH_TRADES", False):
            logger.info(
                f"🚫 [TRADE_DROP: DISABLED] TELEGRAM_PUBLISH_TRADES is False. "
                f"Skipping trade_opened for Ticket #{ticket} ({symbol})."
            )
            # Record ticket in state so it is not re-evaluated continuously
            self.state["opened_tickets"][ticket_str] = {
                "ticket": int(ticket),
                "symbol": symbol,
                "posted_at": None,
                "status": "SUPPRESSED_BY_FEATURE_FLAG"
            }
            self._save_state()
            return False

        direction = str(trade_dict.get("type", "BUY")).upper()
        volume = float(trade_dict.get("lots", trade_dict.get("volume", 0.01)) or 0.01)
        entry = float(trade_dict.get("entry", trade_dict.get("open_price", 0.0)) or 0.0)
        sl = float(trade_dict.get("sl", 0.0) or 0.0) if trade_dict.get("sl") else None
        tp = float(trade_dict.get("tp", 0.0) or 0.0) if trade_dict.get("tp") else None
        digits = self._get_precision(symbol)

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        time_utc_str = trade_dict.get("time_utc", now_utc)

        msg_text = self.format_trade_opened_message(
            symbol=symbol,
            direction=direction,
            volume=volume,
            entry=entry,
            sl=sl,
            tp=tp,
            ticket=int(ticket),
            time_utc_str=time_utc_str
        )

        payload = {
            "schema_version": 1,
            "type": "trade_opened",
            "ticket": int(ticket),
            "symbol": symbol,
            "direction": direction,
            "time_utc": time_utc_str,
            "volume": round(volume, 2),
            "levels": {
                "entry": round(entry, digits),
                "stop_loss": round(sl, digits) if sl else None,
                "take_profit_1": round(tp, digits) if tp else None,
                "take_profit_2": None,
                "tp1_validated": False,
                "tp2_validated": False
            },
            "source": "RULE_ENGINE",
            "source_note": "Executed by the rule engine, not the ML model.",
            "model": None,
            "message_text": msg_text
        }

        # Cache active position details for close calculations
        self.state.setdefault("active_positions", {})[ticket_str] = {
            "ticket": int(ticket),
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "volume": volume,
            "time_utc": time_utc_str
        }

        # Dispatch via telegram_bot._send_payload_async
        is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)
        try:
            if is_dry_run:
                logger.info(f"🧪 [DRY RUN] Would send trade_opened for #{ticket} ({symbol}):\n{msg_text}")
                success = True
            else:
                success = await telegram_bot._send_payload_async(payload)
        except Exception as e:
            logger.error(f"Silent error dispatching trade_opened: {e}")
            success = False

        if success:
            self.state["opened_tickets"][ticket_str] = {
                "ticket": int(ticket),
                "symbol": symbol,
                "posted_at": datetime.now(timezone.utc).isoformat(),
                "payload": payload
            }
            self._record_trade_rate(int(ticket), "trade_opened")
            logger.info(f"✅ [TRADE_EVENT: OPENED] Published trade_opened for #{ticket} ({symbol} {direction})")

        return success

    async def publish_trade_closed(self, closed_dict: Dict[str, Any], force: bool = False) -> bool:
        """
        Publishes a trade_closed event to n8n webhook.
        """
        ticket = closed_dict.get("ticket")
        if not ticket:
            return False
        ticket_str = str(ticket)

        # 1. Deduplication Guard
        if not force and ticket_str in self.state["closed_tickets"]:
            logger.info(f"🚫 [TRADE_DROP: DUPLICATE_CLOSE] Ticket #{ticket} already published as closed.")
            return False

        symbol = str(closed_dict.get("symbol", "")).upper()
        # 2. Exclusion Guard
        if not force and not self._is_symbol_eligible(symbol):
            logger.info(f"🚫 [TRADE_DROP: EXCLUDED_SYMBOL] Symbol {symbol} not eligible for trade publishing.")
            return False

        # 3. Rate Limit Guard
        if not force:
            allowed, reason = self._check_rate_limit()
            if not allowed:
                logger.error(f"🚨 [TRADE_DROP: RATE_LIMIT] Refusing trade_closed for #{ticket}: {reason}")
                return False

        # 4. Feature Flag Guard
        if not force and not getattr(settings, "TELEGRAM_PUBLISH_TRADES", False):
            logger.info(
                f"🚫 [TRADE_DROP: DISABLED] TELEGRAM_PUBLISH_TRADES is False. "
                f"Skipping trade_closed for Ticket #{ticket} ({symbol})."
            )
            self.state["closed_tickets"][ticket_str] = {
                "ticket": int(ticket),
                "symbol": symbol,
                "posted_at": None,
                "status": "SUPPRESSED_BY_FEATURE_FLAG"
            }
            self._save_state()
            return False

        direction = str(closed_dict.get("type", "BUY")).upper()
        profit = float(closed_dict.get("profit", 0.0) or 0.0)
        exit_price = float(closed_dict.get("price", closed_dict.get("close_price", 0.0)) or 0.0)
        
        # Retrieve entry details from cached active position if available
        active_rec = self.state.get("active_positions", {}).get(ticket_str, {})
        entry = float(closed_dict.get("open_price", active_rec.get("entry", 0.0)) or 0.0)
        sl = float(active_rec.get("sl", 0.0) or 0.0) if active_rec.get("sl") else None
        tp = float(active_rec.get("tp", 0.0) or 0.0) if active_rec.get("tp") else None
        opened_utc = active_rec.get("time_utc", closed_dict.get("time_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")))
        closed_utc = closed_dict.get("time_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))

        digits = self._get_precision(symbol)

        # Calculate R-Multiple
        r_multiple: Optional[float] = None
        outcome = "MANUAL"
        outcome_label = "Closed"

        if entry > 0 and sl and sl > 0:
            risk = abs(entry - sl)
            if risk > 0:
                if direction == "BUY":
                    r_raw = (exit_price - entry) / risk
                else:
                    r_raw = (entry - exit_price) / risk
                r_multiple = round(r_raw, 1)

        # Determine outcome and label
        if r_multiple is not None:
            if r_multiple >= 1.8 or (tp and exit_price >= tp and direction == "BUY") or (tp and exit_price <= tp and direction == "SELL"):
                outcome = "TP1"
                outcome_label = "TP1 hit"
            elif r_multiple <= -0.8 or (sl and exit_price <= sl and direction == "BUY") or (sl and exit_price >= sl and direction == "SELL"):
                outcome = "SL"
                outcome_label = "SL hit"
            elif profit > 0:
                outcome = "PROFIT"
                outcome_label = "Closed in profit"
            elif profit < 0:
                outcome = "LOSS"
                outcome_label = "Closed in loss"
            else:
                outcome = "BE"
                outcome_label = "Break-even"
        else:
            if profit > 0:
                outcome = "PROFIT"
                outcome_label = "Closed in profit"
            elif profit < 0:
                outcome = "LOSS"
                outcome_label = "Closed in loss"
            else:
                outcome = "BE"
                outcome_label = "Break-even"

        msg_text = self.format_trade_closed_message(
            symbol=symbol,
            direction=direction,
            outcome_label=outcome_label,
            entry=entry,
            exit_price=exit_price,
            r_multiple=r_multiple,
            profit_currency=profit,
            ticket=int(ticket)
        )

        payload = {
            "schema_version": 1,
            "type": "trade_closed",
            "ticket": int(ticket),
            "symbol": symbol,
            "direction": direction,
            "opened_utc": opened_utc,
            "closed_utc": closed_utc,
            "entry": round(entry, digits),
            "exit": round(exit_price, digits),
            "outcome": outcome,
            "profit_currency": round(profit, 2),
            "r_multiple": r_multiple,
            "source": "RULE_ENGINE",
            "message_text": msg_text
        }

        # Dispatch via telegram_bot
        is_dry_run = getattr(settings, "TELEGRAM_DRY_RUN", True)
        try:
            if is_dry_run:
                logger.info(f"🧪 [DRY RUN] Would send trade_closed for #{ticket} ({symbol}):\n{msg_text}")
                success = True
            else:
                success = await telegram_bot._send_payload_async(payload)
        except Exception as e:
            logger.error(f"Silent error dispatching trade_closed: {e}")
            success = False

        if success:
            self.state["closed_tickets"][ticket_str] = {
                "ticket": int(ticket),
                "symbol": symbol,
                "posted_at": datetime.now(timezone.utc).isoformat(),
                "payload": payload
            }
            # Clean up active_positions
            self.state.get("active_positions", {}).pop(ticket_str, None)
            self._record_trade_rate(int(ticket), "trade_closed")
            logger.info(f"✅ [TRADE_EVENT: CLOSED] Published trade_closed for #{ticket} ({symbol} {outcome})")

        return success

    def get_opened_count(self) -> int:
        """Returns count of published trade_opened events."""
        return len([k for k, v in self.state.get("opened_tickets", {}).items() if v.get("posted_at") is not None])

    def get_closed_count(self) -> int:
        """Returns count of published trade_closed events."""
        return len([k for k, v in self.state.get("closed_tickets", {}).items() if v.get("posted_at") is not None])


trade_event_service = TradeEventService()
