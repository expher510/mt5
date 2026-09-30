import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from app.config import settings, LotMode
from app.services.brainstorm_service import brainstorm_service

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Capital Preservation, Dynamic Position Sizing & Drawdown Guard (Enterprise).
    Features:
    1. Precise Lot Sizing based on real contract sizes (Gold 100oz, FX 100k units).
    2. Exact currency conversion for Pip Values (USD-base, USD-quote, JPY crosses, Gold).
    3. Daily Drawdown Circuit Breaker with UTC midnight date tracking & persistent state.
    4. Max Spread & Slippage validation.
    """

    def __init__(self):
        self.last_reset_date: Optional[str] = None
        # Populated by calculate_lot_size so callers can surface the TRUE risk taken
        self.last_risk_note: Dict[str, Any] = {}
        self.max_allowed_spread_pips = {
            "XAUUSD": 4.5,   # $0.45 spread
            "GOLD": 4.5,
            "EURUSD": 2.0,
            "GBPUSD": 2.5,
            "USDJPY": 2.5,
            "EURJPY": 2.8,
            "GBPJPY": 3.5,
            "USDCHF": 3.0,
            "AUDUSD": 2.5,
            "USDCAD": 3.0,
            "AUDJPY": 3.0,
            "CADJPY": 3.0
        }

    def get_pip_multiplier(self, symbol: str) -> float:
        sym = symbol.upper()
        if "XAU" in sym or "GOLD" in sym:
            return 10.0  # $1.00 move = 10 pips / 100 points
        elif "JPY" in sym:
            return 100.0 # 0.01 move = 1 pip
        else:
            return 10000.0 # 0.0001 move = 1 pip

    def calculate_pip_value_usd(self, symbol: str, current_price: float, usdjpy_price: float = 150.0) -> float:
        """
        Calculates exact USD value of 1.0 pip for a 1.0 Standard Lot (100,000 units / 100 oz)
        """
        sym = symbol.upper()
        
        # 1. Gold XAUUSD: 1.0 Lot = 100 oz. $1 move = $100 PnL -> 1 pip ($0.10) = $10.00
        if "XAU" in sym or "GOLD" in sym:
            return 10.0

        # 2. JPY Crosses (USDJPY, EURJPY, GBPJPY, AUDJPY, CADJPY):
        # 1 pip = 0.01 JPY * 100,000 = 1,000 JPY. Converted to USD = 1000 / USDJPY_RATE
        if "JPY" in sym:
            rate = usdjpy_price if usdjpy_price > 50 else (current_price if "USDJPY" in sym else 150.0)
            return 1000.0 / rate if rate > 0 else 6.67

        # 3. USD Base Pairs (USDCAD, USDCHF):
        # 1 pip = 0.0001 Quote Currency * 100,000 = 10 Quote Currency. Converted to USD = 10 / Current_Price
        if sym.startswith("USD"):
            return 10.0 / current_price if current_price > 0 else 7.40

        # 4. USD Quote Pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD):
        # 1 pip = 0.0001 USD * 100,000 = $10.00 USD
        return 10.0

    def calculate_lot_size(
        self,
        account_balance: float,
        entry_price: float,
        sl_price: float,
        symbol: str,
        risk_percent: float = None,
        lot_mode: LotMode = LotMode.DYNAMIC_PERCENT,
        fixed_lot: float = 0.05,
        live_prices: dict = None,
        free_margin: float = None
    ) -> float:
        """
        Calculates exact lot size according to risk parameters with full Margin Protection:
        - FIXED_LOT: returns fixed lot size clamped between 0.01 and 10.0 (safe against margin)
        - DYNAMIC_PERCENT:
          Risk Amount = Effective Capital * (Risk% / 100)
          Lot Size = Risk Amount / (SL distance in pips * Pip Value per 1.0 lot)
        """
        if account_balance <= 0 or entry_price <= 0 or sl_price <= 0:
            return 0.01

        # Use effective available capital to prevent "No Money" when margin is used
        effective_cap = account_balance
        if free_margin is not None and free_margin > 0:
            effective_cap = min(account_balance, free_margin)

        # Strict Micro-Account Ceiling (Protects small accounts < $500 from dangerous oversize lots)
        if effective_cap < 250.0:
            max_account_lot = 0.02
        elif effective_cap < 500.0:
            max_account_lot = 0.03
        elif effective_cap < 1000.0:
            max_account_lot = 0.05
        else:
            max_account_lot = 5.0

        if lot_mode == LotMode.FIXED_LOT or (hasattr(lot_mode, "value") and lot_mode.value == "FIXED_LOT"):
            desired_lot = max(0.01, min(round(float(fixed_lot), 2), max_account_lot))

            # A fixed lot is a sizing choice, not a licence to ignore the account.
            # This branch used to return immediately, so FIXED_LOT bypassed every
            # risk check: a 0.2 lot with a $20 gold stop is $400 at risk, which on a
            # $141 balance is a margin call on the first trade. The hard ceiling is
            # enforced here too, and the lot is scaled down to fit rather than
            # refused outright.
            sl_dist = abs(entry_price - sl_price)
            if sl_dist > 0:
                pip_mult = self.get_pip_multiplier(symbol)
                usdjpy_rate = 150.0
                if live_prices and "USDJPY" in live_prices:
                    usdjpy_rate = live_prices["USDJPY"].get("bid", 150.0) or 150.0
                pip_value = self.calculate_pip_value_usd(symbol, entry_price, usdjpy_rate)
                per_lot_risk = sl_dist * pip_mult * pip_value
                ceiling = getattr(settings, "MAX_ABSOLUTE_RISK_PERCENT", 6.0)
                max_risk_usd = effective_cap * (ceiling / 100.0)

                if per_lot_risk > 0 and (desired_lot * per_lot_risk) > max_risk_usd:
                    safe_lot = round(max_risk_usd / per_lot_risk, 2)
                    if safe_lot < 0.01:
                        self.last_risk_note = {
                            "lot": 0.0,
                            "risk_usd": round(0.01 * per_lot_risk, 2),
                            "risk_percent": round(0.01 * per_lot_risk / effective_cap * 100.0, 2),
                            "target_percent": ceiling,
                            "min_lot_floor_applied": False,
                            "reason": (f"FIXED_LOT {desired_lot} would risk "
                                       f"${desired_lot * per_lot_risk:.2f}; even 0.01 lot exceeds "
                                       f"the {ceiling}% ceiling on ${effective_cap:.2f}.")
                        }
                        logger.warning(f"FIXED_LOT rejected on {symbol}: {self.last_risk_note['reason']}")
                        return 0.0

                    logger.warning(
                        f"FIXED_LOT capped on {symbol}: {desired_lot} -> {safe_lot} lots "
                        f"(${desired_lot * per_lot_risk:.2f} exceeded the {ceiling}% ceiling "
                        f"of ${max_risk_usd:.2f} on ${effective_cap:.2f})"
                    )
                    desired_lot = min(safe_lot, max_account_lot)

                self.last_risk_note = {
                    "lot": desired_lot,
                    "risk_usd": round(desired_lot * per_lot_risk, 2),
                    "risk_percent": round(desired_lot * per_lot_risk / effective_cap * 100.0, 2)
                    if effective_cap > 0 else 0.0,
                    "target_percent": getattr(settings, "MAX_ABSOLUTE_RISK_PERCENT", 6.0),
                    "min_lot_floor_applied": False,
                    "reason": "Fixed lot, capped by the absolute risk ceiling."
                }

            logger.debug(f"Direct User Configured Lot applied: {desired_lot} Lots (Mode: FIXED_LOT) for {symbol}")
            return desired_lot

        if risk_percent is None:
            risk_percent = settings.MAX_RISK_PER_TRADE_PERCENT

        sl_distance_price = abs(entry_price - sl_price)
        if sl_distance_price <= 0:
            return max(0.01, round(float(fixed_lot or 0.01), 2))

        risk_amount = effective_cap * (risk_percent / 100.0)
        pip_mult = self.get_pip_multiplier(symbol)
        sl_pips = sl_distance_price * pip_mult

        usdjpy_rate = 150.0
        if live_prices and "USDJPY" in live_prices:
            usdjpy_rate = live_prices["USDJPY"].get("bid", 150.0) or 150.0

        pip_value_usd = self.calculate_pip_value_usd(symbol, entry_price, usdjpy_rate)
        
        # Exact Lot formula: Risk Amount / (Pips in SL * Pip Value per 1.0 lot)
        raw_lot = risk_amount / (max(sl_pips, 1.0) * pip_value_usd)
        lot_size = min(round(raw_lot, 2), max_account_lot)

        # Small-account handling.
        #
        # On a small balance the broker's minimum 0.01 lot can risk more than the
        # target percentage. Silently rounding up to 0.01 (the original behaviour)
        # hid that: a $100 account "at 1% risk" was really risking 3.5% per trade.
        # Refusing outright is equally wrong - it locks small accounts out of the
        # only instrument they trade.
        #
        # So: allow the minimum lot, but only while its TRUE risk stays inside a
        # declared hard ceiling, and report that true risk to the caller so it is
        # visible on the dashboard instead of hidden.
        if lot_size < 0.01:
            min_lot_risk = 0.01 * max(sl_pips, 1.0) * pip_value_usd
            min_lot_pct = (min_lot_risk / effective_cap * 100.0) if effective_cap > 0 else 999.0
            hard_ceiling = getattr(settings, "MAX_ABSOLUTE_RISK_PERCENT", 6.0)

            if min_lot_pct <= hard_ceiling:
                self.last_risk_note = {
                    "lot": 0.01,
                    "risk_usd": round(min_lot_risk, 2),
                    "risk_percent": round(min_lot_pct, 2),
                    "target_percent": risk_percent,
                    "min_lot_floor_applied": True,
                    "reason": (f"Broker minimum 0.01 lot risks ${min_lot_risk:.2f} "
                               f"({min_lot_pct:.1f}% of ${effective_cap:.2f}) - above the "
                               f"{risk_percent}% target but inside the {hard_ceiling}% hard ceiling.")
                }
                logger.info(
                    f"Minimum-lot floor applied on {symbol}: 0.01 lot = ${min_lot_risk:.2f} "
                    f"({min_lot_pct:.1f}% of ${effective_cap:.2f}, target {risk_percent}%, ceiling {hard_ceiling}%)"
                )
                return 0.01

            self.last_risk_note = {
                "lot": 0.0,
                "risk_usd": round(min_lot_risk, 2),
                "risk_percent": round(min_lot_pct, 2),
                "target_percent": risk_percent,
                "min_lot_floor_applied": False,
                "reason": (f"Even the minimum 0.01 lot would risk ${min_lot_risk:.2f} "
                           f"({min_lot_pct:.1f}%), above the {hard_ceiling}% hard ceiling.")
            }
            logger.warning(f"Lot rejected for {symbol}: {self.last_risk_note['reason']}")
            return 0.0

        final_lot = min(lot_size, 5.0)  # Max safety ceiling per trade
        actual_risk = final_lot * max(sl_pips, 1.0) * pip_value_usd
        self.last_risk_note = {
            "lot": final_lot,
            "risk_usd": round(actual_risk, 2),
            "risk_percent": round(actual_risk / effective_cap * 100.0, 2) if effective_cap > 0 else 0.0,
            "target_percent": risk_percent,
            "min_lot_floor_applied": False,
            "reason": "Sized to the configured risk percentage."
        }

        logger.debug(f"Calculated Safe Lot: {final_lot} Lots for {symbol}")
        return final_lot

    def check_spread_allowed(self, symbol: str, bid: float, ask: float) -> Dict[str, Any]:
        """Validates whether current market spread is within safe threshold"""
        if bid <= 0 or ask <= 0:
            return {"allowed": True, "spread_pips": 0.0}

        spread_raw = ask - bid
        pip_mult = self.get_pip_multiplier(symbol)
        spread_pips = round(spread_raw * pip_mult, 1)

        sym_key = symbol.upper()
        max_allowed = self.max_allowed_spread_pips.get(sym_key, 3.5)

        if spread_pips > max_allowed:
            return {
                "allowed": False,
                "spread_pips": spread_pips,
                "max_allowed": max_allowed,
                "reason": f"High Spread Alert on {symbol}: Current {spread_pips} pips > Max {max_allowed} pips"
            }

        return {
            "allowed": True,
            "spread_pips": spread_pips,
            "max_allowed": max_allowed
        }

    def check_daily_drawdown_limit(
        self,
        starting_equity: float,
        current_equity: float
    ) -> Dict[str, Any]:
        """Circuit breaker check for daily drawdown (Protected for Small & Active Accounts)"""
        if starting_equity <= 0:
            return {"drawdown_breached": False, "drawdown_percent": 0.0}

        dollar_drawdown = starting_equity - current_equity
        drawdown_pct = (dollar_drawdown / starting_equity) * 100.0 if starting_equity > 0 else 0.0

        # Either limit trips the breaker. Requiring both (AND) meant a large account
        # could bleed thousands of dollars without ever reaching the percentage, and
        # a small one could lose most of its percentage without reaching the dollars.
        max_allowed_pct = getattr(settings, "MAX_DAILY_DRAWDOWN_PERCENT", 4.0)
        max_allowed_dollars = getattr(settings, "MAX_DAILY_DRAWDOWN_DOLLARS", 0.0)

        is_breached = drawdown_pct >= max_allowed_pct
        if max_allowed_dollars > 0 and dollar_drawdown >= max_allowed_dollars:
            is_breached = True

        if is_breached and not getattr(self, "_last_breach_alerted", False):
            self._last_breach_alerted = True
            logger.error(
                f"🚨 Daily Drawdown Circuit Breaker Triggered: -${dollar_drawdown:.2f} ({drawdown_pct:.1f}%)"
            )
            brainstorm_service.add_log(
                level="RISK",
                category="RISK_GATE",
                symbol="ACCOUNT",
                message=f"🚨 CIRCUIT BREAKER: Daily Drawdown breached (-${dollar_drawdown:.2f} | {drawdown_pct:.1f}% >= {max_allowed_pct}%)! Halting auto trading."
            )
        elif not is_breached:
            self._last_breach_alerted = False

        return {
            "drawdown_breached": is_breached,
            "drawdown_percent": round(max(0.0, drawdown_pct), 2),
            "dollar_drawdown": round(max(0.0, dollar_drawdown), 2),
            "max_allowed_percent": max_allowed_pct
        }

    def reset_daily_circuit_breaker(self, current_equity: float, state: dict) -> Dict[str, Any]:
        """Manually resets circuit breaker and unlocks trading immediately"""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.last_reset_date = today_str
        self._last_breach_alerted = False
        state["last_baseline_date"] = today_str
        state["starting_daily_equity"] = current_equity
        
        brainstorm_service.add_log(
            level="SYSTEM",
            category="CONFIG",
            symbol="ACCOUNT",
            message=f"🔄 Circuit Breaker Manually RESET! Daily Equity Baseline updated to: ${current_equity:.2f} -> Auto Trading UNLOCKED."
        )
        return {"status": "SUCCESS", "new_baseline": current_equity}

    def check_and_reset_daily_baseline(self, current_equity: float, state: dict) -> bool:
        """Resets starting daily equity at 00:00 UTC each new trading day, preserving state across reboots"""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Check if state already has a stored baseline date
        stored_date = state.get("last_baseline_date")
        if self.last_reset_date is None and stored_date == today_str and state.get("starting_daily_equity", 0) > 0:
            self.last_reset_date = today_str
            return False

        if self.last_reset_date != today_str:
            self.last_reset_date = today_str
            self._last_breach_alerted = False
            state["last_baseline_date"] = today_str
            state["starting_daily_equity"] = current_equity
            brainstorm_service.add_log(
                level="SYSTEM",
                category="RISK_GATE",
                symbol="ACCOUNT",
                message=f"🌅 New Trading Day ({today_str} UTC). Daily Equity Baseline set to: ${current_equity:.2f}"
            )
            return True
        return False

risk_manager = RiskManager()
