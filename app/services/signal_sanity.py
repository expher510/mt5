import logging
import math
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Reference median M15 ATR values per symbol (Agent Brief 15 Section 2)
# Computed from deep broker history and validated across all liquid instruments
REFERENCE_MEDIAN_M15_ATR: Dict[str, float] = {
    "XAUUSD": 9.43,     # Reference median M15
    "GOLD": 9.43,
    "USDJPY": 0.094,    # Reference median M15
    "EURUSD": 0.00059,
    "GBPUSD": 0.00076,
    "EURJPY": 0.1088,
    "GBPJPY": 0.1384,
    "AUDJPY": 0.0856,
    "CADJPY": 0.0793,
    "CHFJPY": 0.1193,
    "AUDUSD": 0.00052,
    "USDCAD": 0.00058,
    "USDCHF": 0.00053,
    "AUDCHF": 0.00045,
    "NZDUSD": 0.00045,
    "EURGBP": 0.00040,
    "GBPAUD": 0.00110,
    "GBPCAD": 0.00095,
    "AUDCAD": 0.00048,
    "AUDNZD": 0.00045,
    "CADCHF": 0.00045,
    "EURAUD": 0.00090,
    "EURCAD": 0.00075,
    "EURCHF": 0.00045,
    "NZDJPY": 0.0850,
    "GBPNZD": 0.00120,
    "EURNZD": 0.00105,
    "NZDCAD": 0.00050,
    "NZDCHF": 0.00045,
    "US30": 45.0,
    "NAS100": 30.0,
    "XAGUSD": 0.150,
    "BTCUSD": 350.0,
    "USOIL": 0.350,
}


def get_reference_median_atr(symbol: str) -> float:
    """Returns the reference median M15 ATR for a given symbol."""
    sym = symbol.upper()
    if sym in REFERENCE_MEDIAN_M15_ATR:
        return REFERENCE_MEDIAN_M15_ATR[sym]
    if "JPY" in sym:
        return 0.100
    if "XAU" in sym or "GOLD" in sym:
        return 9.43
    return 0.00060

def validate_signal_sanity(
    card_or_levels: Dict[str, Any],
    symbol: Optional[str] = None,
    direction: str = "BUY"
) -> Tuple[bool, str]:
    """
    Strict pre-publish sanity check for trading signal levels (Agent Brief 15 Section 2).
    Rejects and logs, preventing dispatch, when:
      1. Entry, stop, or either target is null, zero, or on the wrong side of entry.
      2. ATR is more than 10x or less than 0.1x the symbol's median M15 ATR.
      3. The stop distance exceeds 3 ATR or is below 0.2 ATR for that symbol.

    Returns (is_valid: bool, error_reason: str).
    """
    sym = (symbol or card_or_levels.get("symbol") or "UNKNOWN").upper()
    dir_str = str(card_or_levels.get("direction") or direction).upper()

    # Extract price levels safely
    def _to_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
            return f if (math.isfinite(f) and f > 0) else None
        except (ValueError, TypeError):
            return None

    entry = _to_float(card_or_levels.get("entry") or card_or_levels.get("entry_price"))
    sl = _to_float(card_or_levels.get("stop_loss") or card_or_levels.get("sl") or card_or_levels.get("sl_price"))
    tp1 = _to_float(card_or_levels.get("take_profit_1") or card_or_levels.get("tp1") or card_or_levels.get("tp") or card_or_levels.get("tp1_price"))
    tp2 = _to_float(card_or_levels.get("take_profit_2") or card_or_levels.get("tp2") or card_or_levels.get("tp2_price"))
    tp3 = _to_float(card_or_levels.get("take_profit_3") or card_or_levels.get("tp3") or card_or_levels.get("tp3_price"))
    atr = _to_float(card_or_levels.get("atr"))

    # Check 1: Null, zero, or non-positive values
    if entry is None:
        return False, f"Entry price is null, zero, or non-positive (entry={entry})"
    if sl is None:
        return False, f"Stop Loss price is null, zero, or non-positive (sl={sl})"
    if tp1 is None:
        return False, f"Take Profit 1 price is null, zero, or non-positive (tp1={tp1})"
    if atr is None:
        return False, f"ATR is null, zero, or non-positive (atr={atr})"

    # Check 2: Directional side check
    if dir_str == "BUY":
        if sl >= entry:
            return False, f"BUY signal SL ({sl}) must be strictly below Entry ({entry})"
        if tp1 <= entry:
            return False, f"BUY signal TP1 ({tp1}) must be strictly above Entry ({entry})"
        if tp2 is not None and tp2 <= tp1:
            return False, f"BUY signal TP2 ({tp2}) must be strictly above TP1 ({tp1})"
        if tp3 is not None and tp2 is not None and tp3 <= tp2:
            return False, f"BUY signal TP3 ({tp3}) must be strictly above TP2 ({tp2})"
    elif dir_str == "SELL":
        if sl <= entry:
            return False, f"SELL signal SL ({sl}) must be strictly above Entry ({entry})"
        if tp1 >= entry:
            return False, f"SELL signal TP1 ({tp1}) must be strictly below Entry ({entry})"
        if tp2 is not None and tp2 >= tp1:
            return False, f"SELL signal TP2 ({tp2}) must be strictly below TP1 ({tp1})"
        if tp3 is not None and tp2 is not None and tp3 >= tp2:
            return False, f"SELL signal TP3 ({tp3}) must be strictly below TP2 ({tp2})"

    # Check 3: ATR magnitude vs Symbol's Median M15 ATR
    median_atr = get_reference_median_atr(sym)
    if atr > 10.0 * median_atr:
        return False, f"ATR {atr:.5f} exceeds 10x median M15 ATR ({median_atr:.5f}) for {sym} ({atr / median_atr:.1f}x median)"
    if atr < 0.10 * median_atr:
        return False, f"ATR {atr:.5f} is below 0.1x median M15 ATR ({median_atr:.5f}) for {sym} ({atr / median_atr:.2f}x median)"

    # Check 4: Stop distance in terms of ATR (must be within [0.2 ATR, 3.0 ATR])
    stop_dist = abs(entry - sl)
    if stop_dist > 3.0 * atr:
        return False, f"Stop distance {stop_dist:.5f} exceeds 3.0 ATR limit ({3.0 * atr:.5f}) for {sym} ({stop_dist / atr:.2f} ATR)"
    if stop_dist < 0.2 * atr:
        return False, f"Stop distance {stop_dist:.5f} is below 0.2 ATR limit ({0.2 * atr:.5f}) for {sym} ({stop_dist / atr:.2f} ATR)"

    return True, "OK"


def validate_manual_trade_sanity(
    entry: float,
    stop_loss: float,
    take_profit_1: Optional[float] = None,
    take_profit_2: Optional[float] = None,
    take_profit_3: Optional[float] = None,
    direction: str = "BUY",
    atr: Optional[float] = None,
    symbol: Optional[str] = None,
    tp1: Optional[float] = None,
    tp2: Optional[float] = None,
    tp3: Optional[float] = None,
    zone_low: Optional[float] = None,
    zone_high: Optional[float] = None,
    live_price: Optional[float] = None,
    **kwargs
) -> Tuple[bool, str]:
    """
    Direction-aware sanity validation for manual analyst desk entries (Agent Brief 26 & 28 & 31).
    Rules:
      1. Stop on correct side of entry for direction (BUY: sl < entry; SELL: sl > entry).
      2. Target 1 beyond entry in trade direction.
      3. Targets strictly ordered away from entry (T1 -> T2 -> T3).
      4. T1 must pay: T1 >= 1.0 R. Refuses if T1 < 1.0 R with exact breakeven win rate.
      5. Zone must be near price: |zone - entry| <= 20 * risk. Refuses with distance & suggested fix.
      6. Risk distance not absurd: reject below 0.1 ATR or above 10.0 ATR.
      7. High in low field: silently swapped.
      8. Reject and preserve what he typed; never silent price adjustments.
      9. Live price check (Brief 31): reject if |entry - live_price| > 20 * risk.
    """
    sym = (symbol or "UNKNOWN").upper()
    dir_str = str(direction).upper()

    t1 = take_profit_1 if take_profit_1 is not None else tp1
    t2 = take_profit_2 if take_profit_2 is not None else tp2
    t3 = take_profit_3 if take_profit_3 is not None else tp3

    if t1 is None:
        return False, "Target 1 (take_profit_1) is required."

    try:
        e = float(entry)
        sl = float(stop_loss)
        tp1 = float(t1)
        tp2 = float(t2) if t2 is not None and float(t2) > 0 else None
        tp3 = float(t3) if t3 is not None and float(t3) > 0 else None
    except (ValueError, TypeError):
        return False, "All price values (Entry, Stop Loss, Target 1) must be valid positive numbers."

    if e <= 0:
        return False, f"Entry price must be positive (got {e})"
    if sl <= 0:
        return False, f"Stop Loss must be positive (got {sl})"
    if tp1 <= 0:
        return False, f"Target 1 must be positive (got {tp1})"

    # Effective ATR from bridge or reference median
    effective_atr = float(atr) if atr is not None and float(atr) > 0 else get_reference_median_atr(sym)

    # 1. Directional Stop check
    if dir_str == "BUY":
        if sl >= e:
            return False, f"Stop Loss ({sl}) must be strictly below Entry ({e}) for BUY"
    elif dir_str == "SELL":
        if sl <= e:
            return False, f"Stop Loss ({sl}) must be strictly above Entry ({e}) for SELL"
    else:
        return False, f"Invalid direction '{direction}' (must be BUY or SELL)"

    # 2. Risk distance check: reject below 0.1 ATR or above 10.0 ATR
    risk_dist = abs(e - sl)
    if risk_dist <= 0:
        return False, "CANNOT RECORD — Risk distance between Entry and Stop Loss cannot be zero."

    # 2b. Live market price proximity check (Agent Brief 31 Section 1)
    # reject if |entry - live_price| > 20 * risk
    lp = live_price if live_price is not None else kwargs.get("live_price")
    if lp is not None:
        try:
            lp_val = float(lp)
            if lp_val > 0:
                gap = abs(e - lp_val)
                gap_r = gap / risk_dist
                if gap_r > 20.0:
                    digits = 2 if ("XAU" in sym or "GOLD" in sym) else (3 if "JPY" in sym else 5)
                    return False, (
                        f"CANNOT RECORD — entry {e:.{digits}f} is {gap:,.{digits}f} away from the market at {lp_val:.{digits}f} "
                        f"({gap_r:.0f} R). Check the price before recording."
                    )
        except (ValueError, TypeError):
            pass

    min_risk = 0.10 * effective_atr
    max_risk = 10.0 * effective_atr

    if risk_dist < min_risk:
        return False, (
            f"Risk distance ({risk_dist:.5f}) is absurdly small: below 0.1 ATR "
            f"({min_risk:.5f}, ATR={effective_atr:.5f})"
        )
    if risk_dist > max_risk:
        return False, (
            f"Risk distance ({risk_dist:.5f}) is absurdly large: exceeds 10.0 ATR "
            f"({max_risk:.5f}, ATR={effective_atr:.5f})"
        )

    # 3. Targets ordering and direction checks (Agent Brief 28 Section 2)
    if dir_str == "BUY":
        if tp1 <= e:
            return False, f"CANNOT RECORD — Target 1 ({tp1}) must be beyond entry ({e}) in the trade direction."
        if tp2 is not None and tp2 <= tp1:
            return False, f"CANNOT RECORD — Target 2 ({tp2}) must be beyond Target 1 ({tp1}) in the trade direction."
        if tp3 is not None:
            prev_tp = tp2 if tp2 is not None else tp1
            if tp3 <= prev_tp:
                return False, f"CANNOT RECORD — Target 3 ({tp3}) must be beyond Target {2 if tp2 is not None else 1} ({prev_tp}) in the trade direction."
    elif dir_str == "SELL":
        if tp1 >= e:
            return False, f"CANNOT RECORD — Target 1 ({tp1}) must be beyond entry ({e}) in the trade direction."
        if tp2 is not None and tp2 >= tp1:
            return False, f"CANNOT RECORD — Target 2 ({tp2}) must be beyond Target 1 ({tp1}) in the trade direction."
        if tp3 is not None:
            prev_tp = tp2 if tp2 is not None else tp1
            if tp3 >= prev_tp:
                return False, f"CANNOT RECORD — Target 3 ({tp3}) must be beyond Target {2 if tp2 is not None else 1} ({prev_tp}) in the trade direction."

    # 4. Target 1 must pay: T1 >= 1.0 R (Agent Brief 28 Section 2)
    tp1_r = abs(tp1 - e) / risk_dist
    if round(tp1_r, 2) < 1.00:
        tp1_r_disp = round(tp1_r, 1)
        be_pct = int(round(100.0 / (1.0 + (tp1_r_disp if tp1_r_disp > 0 else 0.01))))
        return False, (
            f"CANNOT RECORD — T1 at {tp1_r_disp:.1f} R needs an {be_pct}% win rate to break even.\n"
            f"Targets must be at least 1.0 R beyond entry."
        )

    # 5. Zone must be near price (Agent Brief 28 Section 2)
    # Order them silently if swapped
    z_low = float(zone_low) if (zone_low is not None and float(zone_low) > 0) else None
    z_high = float(zone_high) if (zone_high is not None and float(zone_high) > 0) else None
    if z_low is not None and z_high is not None and z_low > z_high:
        z_low, z_high = z_high, z_low

    for z_cand in (z_low, z_high):
        if z_cand is not None:
            dist_pts = abs(z_cand - e)
            dist_r = dist_pts / risk_dist
            if dist_r > 20.0:
                # Suggest decimal shift if plausible
                if 0.1 * e <= z_cand / 10.0 <= 10.0 * e:
                    suggested = f"{z_cand / 10.0:.4f}"
                elif 0.1 * e <= z_cand / 100.0 <= 10.0 * e:
                    suggested = f"{z_cand / 100.0:.4f}"
                elif 0.1 * e <= z_cand * 10.0 <= 10.0 * e:
                    suggested = f"{z_cand * 10.0:.4f}"
                else:
                    suggested = f"{e:.4f}"
                return False, (
                    f"CANNOT RECORD — zone {z_cand:.5f} is {round(dist_r):,d} R from your entry of {e:.5f}.\n"
                    f"Did you mean {suggested}, or leave the zone empty?"
                )

    return True, "OK"

