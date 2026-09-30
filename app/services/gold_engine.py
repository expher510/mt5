import logging
import time
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone, timedelta

from app.services.learning_engine import trade_learning_engine
from app.services.experience_memory import episodic_memory

logger = logging.getLogger("GoldEngine")

# =============================================================================
# Engine-wide tuning constants (single source of truth - no hidden magic numbers)
# =============================================================================
MIN_BASE_SCORE = 65          # Minimum confluence score (allows 2 confirmed institutional models e.g. MSS+Pullback=75)
A_PLUS_MARGIN = 20           # Score surplus above MIN_BASE_SCORE required for A+ (85+)
MIN_MODELS_REQUIRED = 2      # Independent models that must agree
SCALP_SL_ATR_MULT = 1.50     # Stop distance = 1.5 x ATR (volatility driven)
SCALP_SL_MIN = 2.50          # Absolute floor ($) - broker/spread sanity only
SCALP_SL_MAX = 12.00         # Absolute ceiling ($) - catastrophic-volatility guard
SWING_SL_ATR_MULT = 1.80
SWING_SL_MIN = 4.00
SWING_SL_MAX = 20.00
# Target R-multiples.
#
# Chosen from this account's own measured record rather than by convention. At the
# 36.9% win rate in trade_learning_memory.json, a 1.8R target nets -0.08R per trade
# once a $0.35 gold spread on a ~$3 stop is charged - i.e. a losing system even with
# every winner running to target. 2.5R nets +0.17R at the same win rate.
#
# Raising the target is expected to lower the win rate somewhat; the learning engine
# now measures expectancy in R per model, so /api/v1/learning/performance is the
# thing to check before changing these again.
RR_TP1 = 2.5                 # TP1 = 2.5R
RR_TP2 = 4.0                 # TP2 = 4.0R
RR_TP3 = 6.0                 # TP3 = 6.0R
RSI_BUY_EXHAUSTION = 78.0    # Block fresh longs above this
RSI_SELL_EXHAUSTION = 22.0   # Block fresh shorts below this
MAX_EXTENSION_ATR = 2.20     # Block entries more than 2.2 x ATR away from EMA20
STRUCTURE_ROOM_FACTOR = 0.75 # Swing TP1 must fit within 75% of the distance to the wall
BANDWIDTH_SQUEEZE_RATIO = 0.40  # Veto when bandwidth < 40% of its own recent median


class GoldTraderEngine:
    """
    ==============================================================================
    FXENGIN INSTITUTIONAL GOLD TRADER CORE (XAUUSD SPECIALIST v4.0)
    ==============================================================================
    Signal generation runs strictly on CLOSED candles; only the entry anchor uses
    the live price. Every risk parameter is derived from measured volatility (ATR)
    and every target is expressed as a multiple of the actual stop distance, so the
    published risk:reward is the risk:reward that gets executed.

    1. Closed-bar discipline
       - The forming candle is dropped before any indicator is computed.
       - At most one actionable signal is emitted per closed M5 bar.
    2. Regime & chop protection
       - ADX / bandwidth / ATR each independently veto execution (OR, not AND).
       - The learning engine can genuinely tighten the ADX floor over time.
    3. Trend lock + exhaustion guard
       - No counter-trend entries, and no entries into an exhausted extension.
    4. Honest scoring
       - Session boost can upgrade a grade but can never manufacture a signal.
    ==============================================================================
    """

    def __init__(self):
        self.symbol = "XAUUSD"
        self.digits = 2
        self.pip_multiplier = 10.0  # $1.00 move = 10 pips / 100 points
        self.contract_size = 100.0  # 1 standard lot = 100 oz of Gold
        self.active_signal_latch: Optional[Dict[str, Any]] = None
        self.active_scalp_latch: Optional[Dict[str, Any]] = None

    # =========================================================================
    # Data preparation
    # =========================================================================
    @staticmethod
    def _prepare_df(candles: Optional[List[Dict[str, Any]]], drop_forming: bool = True) -> Optional[pd.DataFrame]:
        """
        Builds a numeric OHLC frame and removes the still-forming candle.

        MT5's copy_rates_from_pos(..., 0, n) always returns the live bar at the end.
        Every indicator downstream reads .iloc[-1], so leaving it in makes the whole
        analysis repaint on every tick. It is dropped here, once, for all callers.
        """
        if not candles or len(candles) < 2:
            return None

        df = pd.DataFrame(candles)
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                return None
            df[col] = df[col].astype(float)

        if drop_forming:
            df = df.iloc[:-1]

        return df.reset_index(drop=True)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate complete technical indicator suite for Gold including ADX & Directional Movement"""
        if len(df) < 15:
            return df

        # EMAs (20, 50, 200). EMA200 is only published when the history genuinely
        # supports it - a 200-span EMA over 120 bars is a different indicator, and
        # silently substituting one corrupts the macro trend definition.
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        if len(df) >= 200:
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        else:
            df['ema200'] = np.nan

        # Bollinger Bands (20, 2.0 std & 2.2 std)
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2.0)
        df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2.0)
        df['bb_upper_22'] = df['bb_mid'] + (df['bb_std'] * 2.2)
        df['bb_lower_22'] = df['bb_mid'] - (df['bb_std'] * 2.2)
        df['bb_bandwidth'] = ((df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-9)) * 100.0

        # Wilder's Smoothed RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        df['rsi'] = df['rsi'].fillna(50.0)

        # True Range & ATR (14)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['tr'] = true_range
        df['atr'] = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean().fillna(true_range.rolling(14, min_periods=1).mean())

        # Average Directional Index (ADX 14) + Directional Movement (+DI / -DI)
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        df['plus_dm'] = pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        df['minus_dm'] = pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        smoothed_tr = df['tr'].ewm(alpha=1/14, min_periods=14, adjust=False).mean() + 1e-9
        df['plus_di'] = 100.0 * (df['plus_dm'] / smoothed_tr)
        df['minus_di'] = 100.0 * (df['minus_dm'] / smoothed_tr)

        di_sum = df['plus_di'] + df['minus_di'] + 1e-9
        di_diff = (df['plus_di'] - df['minus_di']).abs()
        df['dx'] = 100.0 * (di_diff / di_sum)
        df['adx'] = df['dx'].ewm(alpha=1/14, min_periods=14, adjust=False).mean().fillna(20.0)

        return df

    @staticmethod
    def _classify_trend(ema20: float, ema50: float, ema200: float,
                        curr_price: float, plus_di: float, minus_di: float) -> str:
        """
        Trend definition. EMA200 participates only when it is a real EMA200
        (NaN otherwise), so the rule is identical across every timeframe.
        """
        has_ema200 = not (ema200 is None or (isinstance(ema200, float) and math.isnan(ema200)))
        above_slow = (curr_price > ema200) if has_ema200 else True
        below_slow = (curr_price < ema200) if has_ema200 else True

        if ema20 > ema50 and plus_di > minus_di and above_slow:
            return "BULLISH"
        if ema20 < ema50 and minus_di > plus_di and below_slow:
            return "BEARISH"
        if ema20 > ema50:
            return "BULLISH_PULLBACK"
        if ema20 < ema50:
            return "BEARISH_PULLBACK"
        return "NEUTRAL"

    # =========================================================================
    # Structure primitives
    # =========================================================================
    def _find_swings(self, df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """
        Identifies prominent Swing Highs and Swing Lows.

        Uses numpy windows rather than nested .iloc access - this runs on every
        timeframe on every incoming bundle, and scalar pandas indexing here was
        stalling the asyncio event loop that also dispatches orders.
        """
        highs, lows = [], []
        n = len(df)
        if n < (left + right + 1):
            return highs, lows

        h = df['high'].to_numpy(dtype=float)
        l = df['low'].to_numpy(dtype=float)
        times = df['time'].to_numpy() if 'time' in df.columns else np.zeros(n)

        for i in range(left, n - right):
            window_h = h[i - left:i + right + 1]
            window_l = l[i - left:i + right + 1]

            # Strict extremum: the centre bar must be the unique max/min of its window
            if h[i] == window_h.max() and (window_h == h[i]).sum() == 1:
                highs.append({"index": i, "price": float(h[i]), "time": int(times[i])})
            if l[i] == window_l.min() and (window_l == l[i]).sum() == 1:
                lows.append({"index": i, "price": float(l[i]), "time": int(times[i])})

        return highs, lows

    def _detect_fvg(self, df: pd.DataFrame) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Detects 3-candle Fair Value Gaps (FVG) on Gold"""
        bullish_fvg, bearish_fvg = None, None
        if len(df) < 5:
            return None, None

        for i in range(len(df) - 1, max(1, len(df) - 15), -1):
            c1_high = df['high'].iloc[i - 2]
            c3_low = df['low'].iloc[i]
            if c3_low > (c1_high + 0.35) and not bullish_fvg:
                bullish_fvg = {
                    "top": round(float(c3_low), 2),
                    "bottom": round(float(c1_high), 2),
                    "mid": round((float(c3_low) + float(c1_high)) / 2.0, 2),
                    "gap_size": round(float(c3_low - c1_high), 2),
                    "index": i - 1
                }

            c1_low = df['low'].iloc[i - 2]
            c3_high = df['high'].iloc[i]
            if c3_high < (c1_low - 0.35) and not bearish_fvg:
                bearish_fvg = {
                    "top": round(float(c1_low), 2),
                    "bottom": round(float(c3_high), 2),
                    "mid": round((float(c1_low) + float(c3_high)) / 2.0, 2),
                    "gap_size": round(float(c1_low - c3_high), 2),
                    "index": i - 1
                }

            if bullish_fvg and bearish_fvg:
                break

        return bullish_fvg, bearish_fvg

    def _detect_divergence(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Detects RSI Regular Bullish/Bearish Divergence (used as an entry veto)"""
        bull_div, bear_div = False, False
        if len(df) < 25:
            return {"bullish_div": False, "bearish_div": False}

        highs, lows = self._find_swings(df, left=2, right=2)
        if len(lows) >= 2:
            l1, l2 = lows[-2], lows[-1]
            p1, p2 = l1['price'], l2['price']
            r1, r2 = df['rsi'].iloc[l1['index']], df['rsi'].iloc[l2['index']]
            if p2 < p1 and r2 > r1:
                bull_div = True

        if len(highs) >= 2:
            h1, h2 = highs[-2], highs[-1]
            p1, p2 = h1['price'], h2['price']
            r1, r2 = df['rsi'].iloc[h1['index']], df['rsi'].iloc[h2['index']]
            if p2 > p1 and r2 < r1:
                bear_div = True

        return {"bullish_div": bull_div, "bearish_div": bear_div}

    def _detect_mss(self, df: pd.DataFrame, left: int = 2, right: int = 2) -> Dict[str, Any]:
        """
        Detects a fresh Market Structure Shift on closed bars only:
        - Bullish MSS: the last closed bar closed above a prior swing high that the
          bar before it was still below.
        - Bearish MSS: mirror image.
        """
        if len(df) < 4:
            return {"bullish_mss": False, "bearish_mss": False, "broken_level": 0.0}

        highs, lows = self._find_swings(df, left=left, right=right)
        last_close = float(df['close'].iloc[-1])
        prior_close = float(df['close'].iloc[-2])

        bullish_mss = False
        bearish_mss = False
        broken_level = 0.0

        if highs:
            recent_sh = highs[-1]['price']
            if last_close > recent_sh and prior_close <= recent_sh:
                bullish_mss = True
                broken_level = recent_sh

        if lows:
            recent_sl = lows[-1]['price']
            if last_close < recent_sl and prior_close >= recent_sl:
                bearish_mss = True
                broken_level = recent_sl

        return {
            "bullish_mss": bullish_mss,
            "bearish_mss": bearish_mss,
            "broken_level": broken_level
        }

    def _get_session_killzone(self) -> Dict[str, Any]:
        """Determines current Institutional Liquidity Session & Killzone"""
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        minute = now_utc.minute
        time_decimal = hour + (minute / 60.0)

        is_london_kz = (7.0 <= time_decimal < 12.0)
        is_ny_kz = (12.0 <= time_decimal < 19.0)
        is_asia = (0.0 <= time_decimal < 7.0)

        if is_london_kz:
            session_name = "LONDON_OPEN_KILLZONE"
            session_name_ar = "جلسة لندن (انفجار السيولة الأوروبية)"
            is_active = True
            boost = 15
        elif is_ny_kz:
            session_name = "NY_OPEN_KILLZONE"
            session_name_ar = "جلسة نيويورك الأمريكية (ذروة السيولة المؤسسية)"
            is_active = True
            boost = 15
        elif is_asia:
            session_name = "ASIA_RANGE_ACCUMULATION"
            session_name_ar = "فترة آسيا (تجميع نطاق السيولة)"
            is_active = False
            boost = 0
        else:
            session_name = "POST_NY_THIN_LIQUIDITY"
            session_name_ar = "ما بعد إغلاق نيويورك (سيولة ضعيفة)"
            is_active = False
            boost = 0

        return {
            "is_killzone_active": is_active,
            "session_name": session_name,
            "session_name_ar": session_name_ar,
            "session_boost": boost,
            "utc_time": now_utc.strftime("%H:%M UTC")
        }

    def _detect_session_liquidity_sweeps(self, df_m15: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Identifies genuine Asia High/Low and Previous Day High/Low sweeps.

        Levels are derived from the bars' actual UTC timestamps: Asia is 00:00-07:00
        UTC of the current trading day, PDH/PDL is the full previous UTC day. Bar
        counts alone ("the last 36 bars") produce levels that drift with the clock
        and have no relationship to the sessions they are named after.
        """
        empty = {"ssl_sweep": False, "bsl_sweep": False, "asia_high": 0.0,
                 "asia_low": 0.0, "pdh": 0.0, "pdl": 0.0, "levels_valid": False}

        if df_m15 is None or len(df_m15) < 10 or 'time' not in df_m15.columns:
            return empty

        df = df_m15.copy()
        ts = pd.to_datetime(df['time'].astype('int64'), unit='s', utc=True)
        df['_date'] = ts.dt.date
        df['_hour'] = ts.dt.hour

        ref_date = df['_date'].iloc[-1]
        prev_date = ref_date - timedelta(days=1)

        asia_bars = df[(df['_date'] == ref_date) & (df['_hour'] < 7)]
        prev_bars = df[df['_date'] == prev_date]

        if asia_bars.empty or prev_bars.empty:
            return empty

        asia_high = float(asia_bars['high'].max())
        asia_low = float(asia_bars['low'].min())
        pdh = float(prev_bars['high'].max())
        pdl = float(prev_bars['low'].min())

        prev_bar = df.iloc[-1]  # last CLOSED bar
        p_low = float(prev_bar['low'])
        p_high = float(prev_bar['high'])
        p_close = float(prev_bar['close'])
        p_open = float(prev_bar['open'])
        p_range = max(p_high - p_low, 0.40)

        ssl_sweep = False
        if (p_low < asia_low and p_close > asia_low) or (p_low < pdl and p_close > pdl):
            lower_wick = min(p_open, p_close) - p_low
            if (lower_wick / p_range) >= 0.30:
                ssl_sweep = True

        bsl_sweep = False
        if (p_high > asia_high and p_close < asia_high) or (p_high > pdh and p_close < pdh):
            upper_wick = p_high - max(p_open, p_close)
            if (upper_wick / p_range) >= 0.30:
                bsl_sweep = True

        return {
            "ssl_sweep": ssl_sweep,
            "bsl_sweep": bsl_sweep,
            "asia_high": round(asia_high, 2),
            "asia_low": round(asia_low, 2),
            "pdh": round(pdh, 2),
            "pdl": round(pdl, 2),
            "levels_valid": True
        }

    # =========================================================================
    # Risk geometry - one place, R-multiple based
    # =========================================================================
    @staticmethod
    def _build_targets(entry: float, direction: str, sl_dist: float) -> Dict[str, float]:
        """
        Derives SL/TP1/TP2/TP3 from a single measured stop distance.

        Targets are R-multiples of the stop that is actually sent to the broker, so
        the advertised risk:reward is arithmetically the executed risk:reward. There
        is no independent clamp that can silently break the ratio.
        """
        tp1_dist = sl_dist * RR_TP1
        tp2_dist = sl_dist * RR_TP2
        tp3_dist = sl_dist * RR_TP3
        sign = 1.0 if direction == "BUY" else -1.0

        return {
            "sl": round(entry - sign * sl_dist, 2),
            "tp1": round(entry + sign * tp1_dist, 2),
            "tp2": round(entry + sign * tp2_dist, 2),
            "tp3": round(entry + sign * tp3_dist, 2),
            "sl_dist": round(sl_dist, 2),
            "tp1_dist": round(tp1_dist, 2),
            "tp2_dist": round(tp2_dist, 2),
            "tp3_dist": round(tp3_dist, 2),
        }

    @staticmethod
    def _grade_and_confidence(base_score: int, session_boost: int) -> Tuple[str, float, int]:
        """
        Converts a confluence score into a grade and a confidence percentage.

        The session boost is applied AFTER the base threshold has been cleared, so
        an active session can upgrade A to A+ but can never create a signal that the
        setup did not earn on its own merits.
        """
        if base_score < MIN_BASE_SCORE:
            return "NO_TRADE", round(min(65.0, 40.0 + base_score * 0.3), 1), base_score

        boosted = base_score + session_boost
        grade = "A+" if boosted >= (MIN_BASE_SCORE + A_PLUS_MARGIN) else "A"
        # Confidence is a bounded, monotonic function of the surplus over threshold
        confidence = min(95.0, 72.0 + (boosted - MIN_BASE_SCORE) * 0.55)
        return grade, round(confidence, 1), boosted

    # =========================================================================
    # 1. PRECISION INSTITUTIONAL MICRO-SCALPER (M1/M5)
    # =========================================================================
    def analyze_gold_scalp_micro(
        self,
        candles_m1: Optional[List[Dict[str, Any]]],
        candles_m5: Optional[List[Dict[str, Any]]],
        h1_trend: str = "NEUTRAL",
        dxy_context: Optional[Dict[str, Any]] = None,
        live_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        High-Precision Institutional Micro-Scalper for Gold (M1/M5).

        Every signal decision is taken on closed M5 bars. `live_price` is used only
        to anchor the entry, SL and TP levels, so the order that reaches the broker
        is priced against the market rather than against a stale bar close.
        """
        def _idle(reason_ar: str, reason_en: str, price: float = 0.0,
                  conf: float = 50.0, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            payload = {
                "scalp_recommendation": "HOLD",
                "scalp_confidence": conf,
                "scalp_grade": "NO_TRADE",
                "scalp_sl": round(price - SCALP_SL_MIN, 2) if price else 0.0,
                "scalp_tp1": round(price + SCALP_SL_MIN * RR_TP1, 2) if price else 0.0,
                "scalp_tp2": round(price + SCALP_SL_MIN * RR_TP2, 2) if price else 0.0,
                "scalp_tp3": round(price + SCALP_SL_MIN * RR_TP3, 2) if price else 0.0,
                "scalp_pips_target": round(SCALP_SL_MIN * RR_TP1 * 10.0, 1),
                "scalp_pips_stop": round(SCALP_SL_MIN * 10.0, 1),
                "breakeven_trigger_pips": 50.0,
                "breakeven_trigger_price": round(price + SCALP_SL_MIN, 2) if price else 0.0,
                "scalp_models_triggered": [],
                "scalp_no_entry_reason_ar": reason_ar,
                "scalp_no_entry_reason": reason_en
            }
            if extra:
                payload.update(extra)
            return payload

        df_m5 = self._prepare_df(candles_m5, drop_forming=True)
        if df_m5 is None or len(df_m5) < 30:
            return _idle(
                "جاري تجميع بيانات شموع الـ 5 دقائق المغلقة للسكالبينج",
                "Collecting closed M5 candle data for scalper"
            )

        df_m5 = self._calculate_indicators(df_m5)

        last_close = float(df_m5['close'].iloc[-1])
        entry_ref = float(live_price) if (live_price and live_price > 0) else last_close
        last_bar_time = int(df_m5['time'].iloc[-1]) if 'time' in df_m5.columns else 0

        m5_atr = float(df_m5['atr'].iloc[-1])
        if not np.isfinite(m5_atr) or m5_atr <= 0:
            return _idle("تعذر حساب تقلب الذهب (ATR) على الشموع المغلقة",
                         "Unable to derive ATR from closed M5 bars", entry_ref)

        m5_rsi = float(df_m5['rsi'].iloc[-1])
        m5_adx = float(df_m5['adx'].iloc[-1])
        m5_plus_di = float(df_m5['plus_di'].iloc[-1])
        m5_minus_di = float(df_m5['minus_di'].iloc[-1])
        m5_bb_u = float(df_m5['bb_upper'].iloc[-1])
        m5_bb_l = float(df_m5['bb_lower'].iloc[-1])
        m5_bandwidth = float(df_m5['bb_bandwidth'].iloc[-1])
        m5_ema20 = float(df_m5['ema20'].iloc[-1])
        m5_ema50 = float(df_m5['ema50'].iloc[-1])

        # Last two CLOSED bars
        prev_bar = df_m5.iloc[-1]
        prev_open = float(prev_bar['open'])
        prev_high = float(prev_bar['high'])
        prev_low = float(prev_bar['low'])
        prev_close = float(prev_bar['close'])
        bar_range = max(prev_high - prev_low, 0.50)
        lower_wick_ratio = (min(prev_open, prev_close) - prev_low) / bar_range
        upper_wick_ratio = (prev_high - max(prev_open, prev_close)) / bar_range

        # Learning Engine dynamic parameters. `max` (not `min`) so that the ADX floor
        # the engine raises after chop losses is actually the floor that gets applied.
        min_adx_required = max(18.0, trade_learning_engine.get_adaptive_adx_threshold())
        sl_adaptive_mult = trade_learning_engine.get_adaptive_sl_multiplier()
        boll_penalty = trade_learning_engine.get_model_penalty("M5_BOLLINGER_REVERSION")

        # Volatility-driven stop, then R-multiple targets off it.
        sl_dist = round(
            min(SCALP_SL_MAX, max(SCALP_SL_MIN, SCALP_SL_ATR_MULT * m5_atr * sl_adaptive_mult)),
            2
        )
        neutral_levels = self._build_targets(entry_ref, "BUY", sl_dist)

        def _hold(reason_ar: str, reason_en: str, conf: float = 45.0) -> Dict[str, Any]:
            return {
                "scalp_recommendation": "HOLD",
                "scalp_confidence": conf,
                "scalp_grade": "NO_TRADE",
                "scalp_sl": neutral_levels["sl"],
                "scalp_tp1": neutral_levels["tp1"],
                "scalp_tp2": neutral_levels["tp2"],
                "scalp_tp3": neutral_levels["tp3"],
                "scalp_pips_target": round(neutral_levels["tp1_dist"] * 10.0, 1),
                "scalp_pips_stop": round(sl_dist * 10.0, 1),
                "breakeven_trigger_pips": round(sl_dist * 10.0, 1),
                "breakeven_trigger_price": round(entry_ref + sl_dist, 2),
                "scalp_models_triggered": [],
                "scalp_atr": round(m5_atr, 2),
                "scalp_adx": round(m5_adx, 1),
                "scalp_no_entry_reason_ar": reason_ar,
                "scalp_no_entry_reason": reason_en
            }

        # ----------------------------------------------------
        # 1. CHOP & FLAT RANGE FILTER (Institutional Dual-Regime)
        # ----------------------------------------------------
        # Truly stagnant dead chop: very low ADX (<14.0) with low ATR (<1.80)
        if m5_adx < 14.0 and m5_atr < 1.80:
            return _hold(
                f"السوق في حالة ركود تام وضعف شديد في الحركة (ADX: {m5_adx:.1f} | ATR: ${m5_atr:.2f}) - وضع الأمان مفعل",
                f"Dead market stagnation (ADX {m5_adx:.1f} | ATR ${m5_atr:.2f}) - execution paused"
            )

        bw_history = df_m5['bb_bandwidth'].tail(60).dropna()
        bw_median = float(bw_history.median()) if len(bw_history) >= 20 else 0.0
        if bw_median > 0 and m5_bandwidth < (BANDWIDTH_SQUEEZE_RATIO * bw_median):
            return _hold(
                f"نطاق بولنجر منكمش بشدة ({m5_bandwidth:.3f}% مقابل وسيط {bw_median:.3f}%) - السوق في مرحلة ضغط قبل الانفجار",
                f"Bollinger bandwidth compressed ({m5_bandwidth:.3f}% vs {bw_median:.3f}% median) - coiling, no entry"
            )

        # ----------------------------------------------------
        # 2. STRICT HIGHER-TIMEFRAME TREND LOCK
        # ----------------------------------------------------
        is_macro_bull = ("BULLISH" in str(h1_trend).upper())
        is_macro_bear = ("BEARISH" in str(h1_trend).upper())

        if not (is_macro_bull or is_macro_bear):
            return _hold(
                f"الاتجاه على الفريم الأكبر غير محدد ({h1_trend}) - لا يوجد انحياز مؤسسي واضح",
                f"No decisive higher-timeframe bias ({h1_trend})"
            )

        is_bullish_momentum = is_macro_bull and (m5_ema20 >= m5_ema50 or m5_plus_di > m5_minus_di)
        is_bearish_momentum = is_macro_bear and (m5_ema20 <= m5_ema50 or m5_minus_di > m5_plus_di)

        # ----------------------------------------------------
        # 3. EXHAUSTION GUARD - do not buy a vertical move or sell a capitulation
        # ----------------------------------------------------
        extension = abs(entry_ref - m5_ema20)
        divs = self._detect_divergence(df_m5)

        if is_macro_bull:
            if m5_rsi >= RSI_BUY_EXHAUSTION:
                return _hold(
                    f"الذهب في منطقة تشبع شرائي حاد (RSI: {m5_rsi:.1f}) - الشراء هنا مطاردة للقمة",
                    f"Overbought exhaustion (RSI {m5_rsi:.1f}) - refusing to chase"
                )
            if extension > (MAX_EXTENSION_ATR * m5_atr):
                return _hold(
                    f"السعر ممتد ${extension:.2f} عن متوسط EMA20 (أكثر من {MAX_EXTENSION_ATR}x ATR) - بانتظار التصحيح",
                    f"Price extended ${extension:.2f} beyond EMA20 (>{MAX_EXTENSION_ATR}x ATR) - awaiting pullback"
                )
            if divs["bearish_div"]:
                return _hold(
                    "دايفرجنس هبوطي على مؤشر RSI يحذر من ضعف الزخم الصاعد - تم إلغاء إشارة الشراء",
                    "Bearish RSI divergence vetoes the long setup"
                )
        else:
            if m5_rsi <= RSI_SELL_EXHAUSTION:
                return _hold(
                    f"الذهب في منطقة تشبع بيعي حاد (RSI: {m5_rsi:.1f}) - البيع هنا مطاردة للقاع",
                    f"Oversold exhaustion (RSI {m5_rsi:.1f}) - refusing to chase"
                )
            if extension > (MAX_EXTENSION_ATR * m5_atr):
                return _hold(
                    f"السعر ممتد ${extension:.2f} عن متوسط EMA20 (أكثر من {MAX_EXTENSION_ATR}x ATR) - بانتظار الارتداد",
                    f"Price extended ${extension:.2f} beyond EMA20 (>{MAX_EXTENSION_ATR}x ATR) - awaiting retrace"
                )
            if divs["bullish_div"]:
                return _hold(
                    "دايفرجنس صعودي على مؤشر RSI يحذر من ضعف الزخم الهابط - تم إلغاء إشارة البيع",
                    "Bullish RSI divergence vetoes the short setup"
                )

        # ----------------------------------------------------
        # 4. CONFLUENCE MODELS (closed-bar evidence only)
        # ----------------------------------------------------
        scalp_buy_score = 0
        scalp_sell_score = 0
        models_triggered = []

        # Models the learning engine has retired on measured evidence contribute
        # nothing. This is what closes the loop: measure expectancy in R -> judge the
        # model over a real sample -> stop taking its signals.
        mss_on = trade_learning_engine.is_model_enabled("MSS_BREAKOUT")
        pullback_on = trade_learning_engine.is_model_enabled("EMA_TREND_PULLBACK")
        fvg_on = trade_learning_engine.is_model_enabled("FVG_IMBALANCE_FILL")
        boll_on = trade_learning_engine.is_model_enabled("BOLLINGER_REVERSION")
        impulse_on = trade_learning_engine.is_model_enabled("TREND_MOMENTUM_EXPANSION")

        # Model 1: Fresh Market Structure Shift
        mss = self._detect_mss(df_m5) if mss_on else {"bullish_mss": False, "bearish_mss": False, "broken_level": 0.0}
        if mss["bullish_mss"] and is_macro_bull:
            scalp_buy_score += 40
            models_triggered.append(f"M5 Bullish MSS Breakout (${mss['broken_level']:.2f})")
        elif mss["bearish_mss"] and is_macro_bear:
            scalp_sell_score += 40
            models_triggered.append(f"M5 Bearish MSS Breakdown (${mss['broken_level']:.2f})")

        # Model 2: Trend-Pullback to EMA 20/50 institutional retest
        if pullback_on and is_bullish_momentum and entry_ref >= (m5_ema50 - 0.80) and entry_ref <= (m5_ema20 + 1.20):
            if m5_rsi >= 42 and (prev_close >= prev_open or lower_wick_ratio >= 0.20):
                scalp_buy_score += 35
                models_triggered.append(f"M5 Bullish EMA20/50 Trend Pullback (${m5_ema20:.2f})")
        elif pullback_on and is_bearish_momentum and entry_ref <= (m5_ema50 + 0.80) and entry_ref >= (m5_ema20 - 1.20):
            if m5_rsi <= 58 and (prev_close <= prev_open or upper_wick_ratio >= 0.20):
                scalp_sell_score += 35
                models_triggered.append(f"M5 Bearish EMA20/50 Trend Pullback (${m5_ema20:.2f})")

        # Model 3: Trend-aligned FVG retest
        fvg_bull, fvg_bear = self._detect_fvg(df_m5) if fvg_on else (None, None)
        if fvg_bull and fvg_bull['bottom'] <= entry_ref <= (fvg_bull['top'] + 0.60) and is_macro_bull:
            scalp_buy_score += 25
            models_triggered.append(f"M5 Bullish FVG Retest (${fvg_bull['mid']:.2f})")
        if fvg_bear and (fvg_bear['bottom'] - 0.60) <= entry_ref <= fvg_bear['top'] and is_macro_bear:
            scalp_sell_score += 25
            models_triggered.append(f"M5 Bearish FVG Retest (${fvg_bear['mid']:.2f})")

        # Model 4: Extreme band reaction, strictly trend aligned
        if boll_on and prev_low <= m5_bb_l and lower_wick_ratio >= 0.30 and is_macro_bull:
            score_contrib = max(0, int(30 - boll_penalty))
            if score_contrib > 0:
                scalp_buy_score += score_contrib
                models_triggered.append(f"Bollinger Lower Band Reaction (${m5_bb_l:.2f})")
        elif boll_on and prev_high >= m5_bb_u and upper_wick_ratio >= 0.30 and is_macro_bear:
            score_contrib = max(0, int(30 - boll_penalty))
            if score_contrib > 0:
                scalp_sell_score += score_contrib
                models_triggered.append(f"Bollinger Upper Band Rejection (${m5_bb_u:.2f})")

        # Model 5: Trend-continuation expansion.
        # Requires a genuine impulse bar (body > 0.5 ATR) rather than merely "price
        # is above EMA20 and the last candle was green", which fired almost always.
        prev_body = abs(prev_close - prev_open)
        is_impulse = prev_body >= (0.50 * m5_atr)
        if impulse_on and is_macro_bull and entry_ref >= m5_ema20 and prev_close > prev_open and is_impulse:
            scalp_buy_score += 30
            models_triggered.append(f"M5 Bullish Impulse Continuation (body ${prev_body:.2f} / ADX {m5_adx:.1f})")
        elif impulse_on and is_macro_bear and entry_ref <= m5_ema20 and prev_close < prev_open and is_impulse:
            scalp_sell_score += 30
            models_triggered.append(f"M5 Bearish Impulse Continuation (body ${prev_body:.2f} / ADX {m5_adx:.1f})")

        # Ironclad higher-timeframe lock
        if is_macro_bear:
            scalp_buy_score = 0
        if is_macro_bull:
            scalp_sell_score = 0

        # ADX high-momentum accelerator
        if m5_adx >= 28.0:
            if m5_plus_di > m5_minus_di and scalp_buy_score > 0:
                scalp_buy_score += 10
            elif m5_minus_di > m5_plus_di and scalp_sell_score > 0:
                scalp_sell_score += 10

        # DXY confluence
        dxy_bias = dxy_context.get("dxy_bias", "NEUTRAL_DXY") if dxy_context else "NEUTRAL_DXY"
        if dxy_bias == "BEARISH_DXY" and scalp_buy_score > 0:
            scalp_buy_score += 10
        elif dxy_bias == "BULLISH_DXY" and scalp_sell_score > 0:
            scalp_sell_score += 10

        # ----------------------------------------------------
        # 5. DECISION
        # ----------------------------------------------------
        kz = self._get_session_killzone()
        session_boost = kz.get("session_boost", 0)

        scalp_rec = "HOLD"
        base_score = max(scalp_buy_score, scalp_sell_score)

        if (scalp_buy_score >= MIN_BASE_SCORE and scalp_buy_score > (scalp_sell_score + 25)
                and len(models_triggered) >= MIN_MODELS_REQUIRED):
            scalp_rec = "BUY"
            base_score = scalp_buy_score
        elif (scalp_sell_score >= MIN_BASE_SCORE and scalp_sell_score > (scalp_buy_score + 25)
              and len(models_triggered) >= MIN_MODELS_REQUIRED):
            scalp_rec = "SELL"
            base_score = scalp_sell_score

        scalp_grade, scalp_conf, boosted_score = self._grade_and_confidence(base_score, session_boost)

        # One actionable signal per closed M5 bar. Without this the same setup is
        # re-emitted every 1.5s for the whole life of the bar.
        if scalp_rec in ("BUY", "SELL"):
            latch = self.active_scalp_latch
            if latch and latch.get("bar_time") == last_bar_time and latch.get("direction") == scalp_rec:
                return _hold(
                    f"تم إصدار إشارة {scalp_rec} بالفعل على شمعة الخمس دقائق الحالية - بانتظار شمعة جديدة",
                    f"{scalp_rec} signal already issued on this closed M5 bar - awaiting a new bar",
                    conf=scalp_conf
                )
            self.active_scalp_latch = {
                "bar_time": last_bar_time,
                "direction": scalp_rec,
                "issued_at": time.time()
            }

        if scalp_rec == "HOLD":
            return _hold(
                f"بانتظار اكتمال نموذج عالي التوافق (قوة التوافق {base_score} | المطلوب {MIN_BASE_SCORE} ونموذجين على الأقل)",
                f"Awaiting high-confluence setup (score {base_score} < {MIN_BASE_SCORE} min)",
                conf=scalp_conf
            )

        levels = self._build_targets(entry_ref, scalp_rec, sl_dist)

        # ----------------------------------------------------
        # Active Autopsy Shield for Gold
        # ----------------------------------------------------
        autopsy_report = episodic_memory.evaluate_setup_against_learned_rules(
            symbol="XAUUSD",
            direction=scalp_rec,
            current_price=entry_ref,
            sl_price=levels["sl"],
            tp_price=levels["tp1"],
            indicators={
                "adx": m5_adx,
                "atr": m5_atr,
                "rsi": m5_rsi,
                "macro_trend": h1_trend,
                "upper_wick_ratio": upper_wick_ratio,
                "lower_wick_ratio": lower_wick_ratio,
                "has_mss": (mss["bullish_mss"] or mss["bearish_mss"]),
                "is_gold": True
            }
        )

        # Expand SL if autopsy identified tight stop volatility noise
        if autopsy_report.get("adjusted_sl_dist", 0.0) > 0:
            sl_dist = max(sl_dist, float(autopsy_report["adjusted_sl_dist"]))
            levels = self._build_targets(entry_ref, scalp_rec, sl_dist)

        sl_pips = round(sl_dist * 10.0, 1)
        tp1_pips = round(levels["tp1_dist"] * 10.0, 1)
        be_dist = sl_dist  # Break-even is armed at exactly +1R

        # Apply penalty or downgrade if traps detected
        if not autopsy_report.get("is_safe", True) or autopsy_report.get("penalty_score", 0.0) >= 20.0:
            scalp_grade = "B" if scalp_grade in ["A+", "A"] else scalp_grade
            scalp_conf = max(45.0, round(scalp_conf - autopsy_report.get("penalty_score", 0.0), 1))

        reason_ar = (f"⚡ إشارة ذهب عالية الجودة مع الاتجاه ({scalp_rec} Grade {scalp_grade} - ثقة {scalp_conf:.0f}%) "
                     f"- هدف أول +{tp1_pips} نقطة (R:R {RR_TP1}:1)")
        if autopsy_report.get("penalty_score", 0.0) > 0:
            reason_ar += f" | 🛡️ حماية التشريح: {autopsy_report.get('advice_ar')}"
        reason_en = (f"⚡ Confirmed High-Quality Gold Setup ({scalp_rec} Grade {scalp_grade} | {scalp_conf:.0f}% Conf "
                     f"| TP1 +{tp1_pips} pips | R:R {RR_TP1}:1)")

        return {
            "scalp_recommendation": scalp_rec,
            "scalp_confidence": scalp_conf,
            "scalp_grade": scalp_grade,
            "scalp_sl": levels["sl"],
            "scalp_tp1": levels["tp1"],
            "scalp_tp2": levels["tp2"],
            "scalp_tp3": levels["tp3"],
            "scalp_entry_ref": round(entry_ref, 2),
            "scalp_pips_target": tp1_pips,
            "scalp_pips_stop": sl_pips,
            "scalp_atr": round(m5_atr, 2),
            "scalp_adx": round(m5_adx, 1),
            "scalp_base_score": base_score,
            "scalp_boosted_score": boosted_score,
            "breakeven_trigger_pips": round(be_dist * 10.0, 1),
            "breakeven_trigger_price": round(entry_ref + (be_dist if scalp_rec == "BUY" else -be_dist), 2),
            "scalp_models_triggered": models_triggered,
            "autopsy_shield": autopsy_report,
            "scalp_no_entry_reason_ar": reason_ar,
            "scalp_no_entry_reason": reason_en
        }

    # =========================================================================
    # 2. MULTI-TIMEFRAME TRI-SCHOOL & HYBRID COORDINATOR
    # =========================================================================
    def analyze_gold_multi_timeframe(
        self,
        timeframe_data: Dict[str, List[Dict[str, Any]]],
        dxy_context: Optional[Dict[str, Any]] = None,
        strategy_profile: str = "HYBRID",
        live_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes both Macro Swing Analysis and Micro Scalp Analysis on closed bars,
        and coordinates execution according to the selected strategy profile.
        """
        tf_results = {}
        closed_frames: Dict[str, pd.DataFrame] = {}

        for tf in ["H4", "H1", "M15", "M5", "M1"]:
            df = self._prepare_df(timeframe_data.get(tf, []), drop_forming=True)
            if df is None or len(df) < 30:
                continue

            df_calc = self._calculate_indicators(df)
            closed_frames[tf] = df_calc

            curr_p = float(df_calc['close'].iloc[-1])
            ema20 = float(df_calc['ema20'].iloc[-1])
            ema50 = float(df_calc['ema50'].iloc[-1])
            ema200_raw = df_calc['ema200'].iloc[-1]
            ema200 = float(ema200_raw) if pd.notna(ema200_raw) else float('nan')
            atr_val = float(df_calc['atr'].iloc[-1])
            rsi_val = float(df_calc['rsi'].iloc[-1])
            adx_val = float(df_calc['adx'].iloc[-1])
            plus_di = float(df_calc['plus_di'].iloc[-1])
            minus_di = float(df_calc['minus_di'].iloc[-1])
            bb_u = float(df_calc['bb_upper'].iloc[-1])
            bb_m = float(df_calc['bb_mid'].iloc[-1])
            bb_l = float(df_calc['bb_lower'].iloc[-1])
            bb_width = float(df_calc['bb_bandwidth'].iloc[-1])

            tf_trend = self._classify_trend(ema20, ema50, ema200, curr_p, plus_di, minus_di)

            bb_status = "NORMAL"
            bw_hist = df_calc['bb_bandwidth'].tail(60).dropna()
            bw_med = float(bw_hist.median()) if len(bw_hist) >= 20 else 0.0
            if bw_med > 0 and bb_width < (BANDWIDTH_SQUEEZE_RATIO * bw_med):
                bb_status = "SQUEEZE_COILING"
            elif curr_p <= (bb_l + 0.35 * atr_val):
                bb_status = "LOWER_BAND_BOUNCE"
            elif curr_p >= (bb_u - 0.35 * atr_val):
                bb_status = "UPPER_BAND_REJECTION"

            sw_h, sw_l = self._find_swings(df_calc, left=2, right=2)
            fvg_bull, fvg_bear = self._detect_fvg(df_calc)
            divs = self._detect_divergence(df_calc)

            sweeps = {"bsl_sweep": False, "ssl_sweep": False}
            if len(sw_h) >= 2 and curr_p < sw_h[-1]['price'] and float(df_calc['high'].iloc[-1]) > sw_h[-1]['price']:
                sweeps["bsl_sweep"] = True
            if len(sw_l) >= 2 and curr_p > sw_l[-1]['price'] and float(df_calc['low'].iloc[-1]) < sw_l[-1]['price']:
                sweeps["ssl_sweep"] = True

            # Institutional SMC Order Block detection over closed bars
            bull_ob, bear_ob = None, None
            o_arr = df_calc['open'].to_numpy(dtype=float)
            c_arr = df_calc['close'].to_numpy(dtype=float)
            h_arr = df_calc['high'].to_numpy(dtype=float)
            l_arr = df_calc['low'].to_numpy(dtype=float)
            t_arr = df_calc['time'].to_numpy() if 'time' in df_calc.columns else np.zeros(len(df_calc))
            n_bars = len(df_calc)

            for k in range(n_bars - 4, max(0, n_bars - 25), -1):
                if c_arr[k] < o_arr[k] and bull_ob is None:
                    subsequent_max_high = h_arr[k + 1:min(n_bars, k + 4)].max()
                    if (subsequent_max_high - h_arr[k]) >= (1.1 * atr_val):
                        bull_ob = {"low": round(float(l_arr[k]), 2),
                                   "high": round(float(h_arr[k]), 2),
                                   "time": int(t_arr[k])}
                elif c_arr[k] > o_arr[k] and bear_ob is None:
                    subsequent_min_low = l_arr[k + 1:min(n_bars, k + 4)].min()
                    if (l_arr[k] - subsequent_min_low) >= (1.1 * atr_val):
                        bear_ob = {"low": round(float(l_arr[k]), 2),
                                   "high": round(float(h_arr[k]), 2),
                                   "time": int(t_arr[k])}
                if bull_ob and bear_ob:
                    break

            # Fibonacci OTE Golden Zone
            fibo_bias = "HOLD"
            fibo_gz_low = curr_p - 5.0
            fibo_gz_high = curr_p + 5.0
            fibo_tp = curr_p + 15.0
            fibo_sl = curr_p - 10.0
            if sw_h and sw_l:
                last_sh = sw_h[-1]['price']
                last_sl = sw_l[-1]['price']
                diff = abs(last_sh - last_sl)
                if diff < (1.5 * atr_val):
                    diff = 3.0 * atr_val

                if sw_l[-1]['index'] < sw_h[-1]['index']:
                    fibo_gz_low = round(last_sh - 0.705 * diff, 2)
                    fibo_gz_high = round(last_sh - 0.500 * diff, 2)
                    fibo_tp = round(last_sh + 0.618 * diff, 2)
                    fibo_sl = round(last_sl - 0.8 * atr_val, 2)
                    if (fibo_gz_low - 0.6 * atr_val) <= curr_p <= (fibo_gz_high + 0.6 * atr_val):
                        fibo_bias = "BUY"
                else:
                    fibo_gz_high = round(last_sl + 0.705 * diff, 2)
                    fibo_gz_low = round(last_sl + 0.500 * diff, 2)
                    fibo_tp = round(last_sl - 0.618 * diff, 2)
                    fibo_sl = round(last_sh + 0.8 * atr_val, 2)
                    if (fibo_gz_low - 0.6 * atr_val) <= curr_p <= (fibo_gz_high + 0.6 * atr_val):
                        fibo_bias = "SELL"

            valid_sups = [s['price'] for s in sw_l if s['price'] < curr_p]
            valid_res = [s['price'] for s in sw_h if s['price'] > curr_p]
            sup_p = max(valid_sups) if valid_sups else float(df_calc['low'].iloc[-20:].min())
            res_p = min(valid_res) if valid_res else float(df_calc['high'].iloc[-20:].max())

            tf_results[tf] = {
                "current_price": curr_p,
                "atr": atr_val,
                "rsi": rsi_val,
                "adx": adx_val,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "trend": tf_trend,
                "ema20": round(ema20, 2),
                "ema50": round(ema50, 2),
                "ema200": round(ema200, 2) if pd.notna(ema200) else None,
                "bb_upper": round(bb_u, 2),
                "bb_mid": round(bb_m, 2),
                "bb_lower": round(bb_l, 2),
                "bb_bandwidth": round(bb_width, 2),
                "bb_status": bb_status,
                "bullish_ob": bull_ob,
                "bearish_ob": bear_ob,
                "bullish_fvg": fvg_bull,
                "bearish_fvg": fvg_bear,
                "liquidity_sweeps": sweeps,
                "divergence": divs,
                "fibo_bias": fibo_bias,
                "fibo_gz_low": fibo_gz_low,
                "fibo_gz_high": fibo_gz_high,
                "fibo_tp": fibo_tp,
                "fibo_sl": fibo_sl,
                "support": round(sup_p, 2),
                "resistance": round(res_p, 2)
            }

        if not tf_results:
            return {"symbol": "XAUUSD", "recommendation": "HOLD", "setup_quality": "NO_DATA",
                    "current_price": round(float(live_price or 0.0), 2),
                    "no_entry_reason": "Insufficient closed-candle history",
                    "no_entry_reason_ar": "لا توجد شموع مغلقة كافية لبدء التحليل"}

        h4_data = tf_results.get("H4")
        h1_data = tf_results.get("H1", h4_data)
        m15_data = tf_results.get("M15", h1_data if h1_data else list(tf_results.values())[0])

        if not m15_data:
            return {"symbol": "XAUUSD", "recommendation": "HOLD", "setup_quality": "NO_DATA",
                    "current_price": round(float(live_price or 0.0), 2),
                    "no_entry_reason": "M15 structure unavailable",
                    "no_entry_reason_ar": "بيانات فريم 15 دقيقة غير متاحة"}

        macro_trend = h1_data["trend"] if h1_data else (h4_data["trend"] if h4_data else "NEUTRAL")
        m15_trend = m15_data.get("trend", "NEUTRAL")
        last_m15_close = m15_data.get("current_price", 0.0)
        curr_price = float(live_price) if (live_price and live_price > 0) else last_m15_close
        gold_atr = m15_data.get("atr", 0.0)
        m15_adx = m15_data.get("adx", 25.0)
        m15_bandwidth = m15_data.get("bb_bandwidth", 0.35)

        if not gold_atr or not np.isfinite(gold_atr) or gold_atr <= 0:
            return {"symbol": "XAUUSD", "recommendation": "HOLD", "setup_quality": "NO_DATA",
                    "current_price": round(curr_price, 2),
                    "no_entry_reason": "ATR unavailable on M15",
                    "no_entry_reason_ar": "تعذر حساب تقلب الذهب على فريم 15 دقيقة"}

        # Baseline volatility from this symbol's own recent history, so the
        # "high volatility expansion" branch of the regime classifier can actually
        # fire. Passing the current ATR as its own average made it unreachable.
        m15_frame = closed_frames.get("M15")
        if m15_frame is not None and 'atr' in m15_frame.columns:
            avg_atr = float(m15_frame['atr'].tail(100).mean())
        else:
            avg_atr = gold_atr
        if not np.isfinite(avg_atr) or avg_atr <= 0:
            avg_atr = gold_atr

        # AI Market Regime Classification (now an enforced gate, see below)
        market_regime = trade_learning_engine.classify_market_regime(
            adx=m15_adx,
            bandwidth=m15_bandwidth,
            atr=gold_atr,
            avg_atr=avg_atr,
            trend_alignment=macro_trend
        )

        # Run Micro Scalp Engine on closed bars, anchored to the live price
        scalp_data = self.analyze_gold_scalp_micro(
            candles_m1=timeframe_data.get("M1"),
            candles_m5=timeframe_data.get("M5"),
            h1_trend=macro_trend,
            dxy_context=dxy_context,
            live_price=curr_price
        )

        # ----------------------------------------------------
        # SWING ENGINE - Killzone & confluence scoring
        # ----------------------------------------------------
        kz_info = self._get_session_killzone()
        session_sweeps = self._detect_session_liquidity_sweeps(closed_frames.get("M15"))

        buy_score = 0
        sell_score = 0
        confluence_notes = []

        is_macro_bull = ("BULLISH" in str(macro_trend).upper())
        is_macro_bear = ("BEARISH" in str(macro_trend).upper())

        if kz_info.get("is_killzone_active"):
            confluence_notes.append(f"⚡ {kz_info.get('session_name_ar')} نشطة (تعزيز الدرجة +{kz_info.get('session_boost')})")

        # Real session sweeps (only when the levels were derived from real timestamps)
        if session_sweeps.get("levels_valid"):
            if session_sweeps.get("ssl_sweep") and not is_macro_bear:
                buy_score += 35
                confluence_notes.append(f"👑 سحب سيولة قاع آسيا/اليوم السابق (${session_sweeps.get('asia_low')}) (+35)")
            if session_sweeps.get("bsl_sweep") and not is_macro_bull:
                sell_score += 35
                confluence_notes.append(f"👑 سحب سيولة قمة آسيا/اليوم السابق (${session_sweeps.get('asia_high')}) (+35)")

        # Macro & M15 trend alignment
        if is_macro_bull and m15_trend in ["BULLISH", "BULLISH_PULLBACK"]:
            buy_score += 30
            confluence_notes.append("Strong Macro (H1/H4) & M15 Bullish Alignment (+30)")
            sell_score = 0
        elif is_macro_bear and m15_trend in ["BEARISH", "BEARISH_PULLBACK"]:
            sell_score += 30
            confluence_notes.append("Strong Macro (H1/H4) & M15 Bearish Alignment (+30)")
            buy_score = 0

        m15_ob_bull = m15_data.get("bullish_ob")
        m15_ob_bear = m15_data.get("bearish_ob")
        m15_fvg_bull = m15_data.get("bullish_fvg")
        m15_fvg_bear = m15_data.get("bearish_fvg")

        if m15_ob_bull and (m15_ob_bull['low'] - 0.8 * gold_atr) <= curr_price <= (m15_ob_bull['high'] + 0.8 * gold_atr) and not is_macro_bear:
            buy_score += 25
            confluence_notes.append(f"Institutional Demand OB Retest (${m15_ob_bull['low']} - ${m15_ob_bull['high']}) (+25)")
        if m15_fvg_bull and m15_fvg_bull['bottom'] <= curr_price <= (m15_fvg_bull['top'] + 0.50) and not is_macro_bear:
            buy_score += 15
            confluence_notes.append(f"Bullish FVG Tap at ${m15_fvg_bull['mid']} (+15)")

        if m15_ob_bear and (m15_ob_bear['low'] - 0.8 * gold_atr) <= curr_price <= (m15_ob_bear['high'] + 0.8 * gold_atr) and not is_macro_bull:
            sell_score += 25
            confluence_notes.append(f"Institutional Supply OB Rejection (${m15_ob_bear['low']} - ${m15_ob_bear['high']}) (+25)")
        if m15_fvg_bear and (m15_fvg_bear['bottom'] - 0.50) <= curr_price <= m15_fvg_bear['top'] and not is_macro_bull:
            sell_score += 15
            confluence_notes.append(f"Bearish FVG Tap at ${m15_fvg_bear['mid']} (+15)")

        if m15_data.get("fibo_bias") == "BUY" and not is_macro_bear:
            buy_score += 20
            confluence_notes.append(f"Fibonacci 0.618-0.786 OTE Golden Zone (${m15_data['fibo_gz_low']} - ${m15_data['fibo_gz_high']}) (+20)")
        elif m15_data.get("fibo_bias") == "SELL" and not is_macro_bull:
            sell_score += 20
            confluence_notes.append(f"Fibonacci 0.618-0.786 OTE Bearish Golden Zone (${m15_data['fibo_gz_low']} - ${m15_data['fibo_gz_high']}) (+20)")

        # Ironclad higher-timeframe lock
        if is_macro_bear:
            buy_score = 0
        if is_macro_bull:
            sell_score = 0

        # Swing exhaustion guard (mirrors the scalp guard)
        m15_rsi = m15_data.get("rsi", 50.0)
        m15_ema20 = m15_data.get("ema20", curr_price)
        m15_extension = abs(curr_price - m15_ema20)
        m15_divs = m15_data.get("divergence", {})

        if buy_score > 0 and (m15_rsi >= RSI_BUY_EXHAUSTION
                              or m15_extension > (MAX_EXTENSION_ATR * gold_atr)
                              or m15_divs.get("bearish_div")):
            buy_score = 0
            confluence_notes.append("🚫 تم إلغاء انحياز الشراء: تشبع شرائي / امتداد مفرط / دايفرجنس هبوطي")
        if sell_score > 0 and (m15_rsi <= RSI_SELL_EXHAUSTION
                               or m15_extension > (MAX_EXTENSION_ATR * gold_atr)
                               or m15_divs.get("bullish_div")):
            sell_score = 0
            confluence_notes.append("🚫 تم إلغاء انحياز البيع: تشبع بيعي / امتداد مفرط / دايفرجنس صعودي")

        # DXY Correlation
        dxy_bias = "NEUTRAL_DXY"
        dxy_desc = "DXY Stable"
        if dxy_context:
            dxy_bias = dxy_context.get("dxy_bias", "NEUTRAL_DXY")
            dxy_chg = dxy_context.get("dxy_change_pct", 0.0)
            if dxy_bias == "BEARISH_DXY" and buy_score > 0:
                buy_score += 10
                confluence_notes.append(f"Dollar Index (DXY) Breakdown ({dxy_chg}%) -> Bullish Gold (+10)")
                dxy_desc = f"DXY Bearish Breakdown ({dxy_chg}%)"
            elif dxy_bias == "BULLISH_DXY" and sell_score > 0:
                sell_score += 10
                confluence_notes.append(f"Dollar Index (DXY) Expansion ({dxy_chg}%) -> Bearish Gold (+10)")
                dxy_desc = f"DXY Bullish Expansion ({dxy_chg}%)"

        # Decision on the UNBOOSTED score - the session can only upgrade the grade
        swing_rec = "HOLD"
        swing_base = max(buy_score, sell_score)
        if buy_score >= MIN_BASE_SCORE and buy_score > (sell_score + 25):
            swing_rec = "BUY"
            swing_base = buy_score
        elif sell_score >= MIN_BASE_SCORE and sell_score > (buy_score + 25):
            swing_rec = "SELL"
            swing_base = sell_score

        swing_grade, swing_conf, swing_boosted = self._grade_and_confidence(
            swing_base, kz_info.get("session_boost", 0)
        )

        # Swing risk geometry: structure first, bounded by volatility rather than by
        # an arbitrary dollar cap that would leave the stop floating in mid-air.
        sl_adaptive_mult = trade_learning_engine.get_adaptive_sl_multiplier()
        atr_sl = SWING_SL_ATR_MULT * gold_atr * sl_adaptive_mult
        # The ceiling must never fall below the floor, otherwise a very quiet market
        # produces a stop tighter than SWING_SL_MIN via the volatility cap.
        max_swing_sl = max(SWING_SL_MIN, min(SWING_SL_MAX, 3.5 * gold_atr))

        if swing_rec == "BUY":
            ob_l = m15_ob_bull['low'] if m15_ob_bull else 0.0
            candidates = [p for p in (m15_data["support"], ob_l) if p and p > 0 and p < curr_price]
            struct_dist = (curr_price - min(candidates) + 1.50) if candidates else atr_sl
            swing_sl_dist = min(max_swing_sl, max(SWING_SL_MIN, atr_sl, struct_dist))
        elif swing_rec == "SELL":
            ob_h = m15_ob_bear['high'] if m15_ob_bear else 0.0
            candidates = [p for p in (m15_data["resistance"], ob_h) if p and p > 0 and p > curr_price]
            struct_dist = (max(candidates) - curr_price + 1.50) if candidates else atr_sl
            swing_sl_dist = min(max_swing_sl, max(SWING_SL_MIN, atr_sl, struct_dist))
        else:
            swing_sl_dist = min(max_swing_sl, max(SWING_SL_MIN, atr_sl))

        swing_levels = self._build_targets(curr_price, swing_rec if swing_rec != "HOLD" else "BUY", swing_sl_dist)
        swing_sl = swing_levels["sl"]
        swing_tp1 = swing_levels["tp1"]
        swing_tp2 = swing_levels["tp2"]
        swing_tp3 = swing_levels["tp3"]
        swing_sl_pips = round(swing_sl_dist * 10.0, 1)
        swing_tp_pips = round(swing_levels["tp1_dist"] * 10.0, 1)

        # ----------------------------------------------------
        # Strategy Profile Selection (HYBRID / SCALPING / SWING)
        # ----------------------------------------------------
        scalp_actionable = (scalp_data["scalp_grade"] in ["A+", "A"]
                            and scalp_data["scalp_recommendation"] in ["BUY", "SELL"])
        swing_actionable = (swing_grade in ["A+", "A"] and swing_rec in ["BUY", "SELL"])

        if strategy_profile == "SCALPING":
            use_scalp, use_swing = scalp_actionable, False
        elif strategy_profile == "SWING":
            use_scalp, use_swing = False, swing_actionable
        else:  # HYBRID - scalp has priority, swing is the fallback
            use_scalp = scalp_actionable
            use_swing = (not scalp_actionable) and swing_actionable

        if use_scalp:
            primary_rec = scalp_data["scalp_recommendation"]
            primary_conf = scalp_data["scalp_confidence"]
            primary_grade = scalp_data["scalp_grade"]
            primary_sl = scalp_data["scalp_sl"]
            primary_tp1 = scalp_data["scalp_tp1"]
            primary_tp2 = scalp_data["scalp_tp2"]
            primary_tp3 = scalp_data["scalp_tp3"]
            primary_tp_pips = scalp_data["scalp_pips_target"]
            primary_sl_pips = scalp_data["scalp_pips_stop"]
            primary_reason_ar = scalp_data["scalp_no_entry_reason_ar"]
            primary_reason_en = scalp_data["scalp_no_entry_reason"]
            active_trade_type = "SCALP"
        elif use_swing:
            primary_rec = swing_rec
            primary_conf = swing_conf
            primary_grade = swing_grade
            primary_sl = swing_sl
            primary_tp1 = swing_tp1
            primary_tp2 = swing_tp2
            primary_tp3 = swing_tp3
            primary_tp_pips = swing_tp_pips
            primary_sl_pips = swing_sl_pips
            primary_reason_ar = f"⚡ إشارة مؤسسية كبرى مؤكدة ({swing_rec} Grade {swing_grade} - ثقة {swing_conf:.0f}%)"
            primary_reason_en = f"⚡ Confirmed Institutional Wave ({swing_rec} Grade {swing_grade})"
            active_trade_type = "SWING"
        else:
            primary_rec = "HOLD"
            primary_conf = max(swing_conf, scalp_data["scalp_confidence"])
            primary_grade = "NO_TRADE"
            primary_sl = swing_sl
            primary_tp1 = swing_tp1
            primary_tp2 = swing_tp2
            primary_tp3 = swing_tp3
            primary_tp_pips = swing_tp_pips
            primary_sl_pips = swing_sl_pips
            primary_reason_ar = scalp_data["scalp_no_entry_reason_ar"]
            primary_reason_en = scalp_data["scalp_no_entry_reason"]
            active_trade_type = "HYBRID_SCAN"

        # ----------------------------------------------------
        # HARD VETOES on the selected primary signal
        # ----------------------------------------------------
        veto_ar = None
        veto_en = None

        # 1. Market regime veto - previously computed and then ignored
        if primary_rec in ("BUY", "SELL") and not market_regime.get("is_tradeable", True):
            veto_ar = f"نظام السوق الحالي غير صالح للتداول: {market_regime.get('regime_ar')}"
            veto_en = f"Market regime not tradeable: {market_regime.get('regime')}"

        # 2. Structural room veto - refuse to target straight through a wall.
        #    Applied to SWING entries only. A swing target is several ATR wide, so a
        #    real M15 level in front of it genuinely caps the move. A scalp target is
        #    ~2.7 x M5 ATR, which in a trending market routinely and legitimately
        #    passes through minor M15 pivots; judging scalps against those pivots
        #    vetoed essentially every valid scalp.
        apply_room_veto = (active_trade_type == "SWING")

        if apply_room_veto and primary_rec == "BUY" and veto_ar is None:
            wall = m15_data.get("resistance", 0.0)
            if wall and wall > curr_price:
                room = wall - curr_price
                needed = abs(primary_tp1 - curr_price)
                if room < (needed * STRUCTURE_ROOM_FACTOR):
                    veto_ar = (f"لا توجد مساحة كافية للهدف: أقرب مقاومة عند ${wall:.2f} "
                               f"(${room:.2f}) بينما الهدف يحتاج ${needed:.2f}")
                    veto_en = (f"Insufficient room to target: resistance at ${wall:.2f} "
                               f"(${room:.2f} away) vs ${needed:.2f} needed")
        elif apply_room_veto and primary_rec == "SELL" and veto_ar is None:
            wall = m15_data.get("support", 0.0)
            if wall and wall < curr_price:
                room = curr_price - wall
                needed = abs(curr_price - primary_tp1)
                if room < (needed * STRUCTURE_ROOM_FACTOR):
                    veto_ar = (f"لا توجد مساحة كافية للهدف: أقرب دعم عند ${wall:.2f} "
                               f"(${room:.2f}) بينما الهدف يحتاج ${needed:.2f}")
                    veto_en = (f"Insufficient room to target: support at ${wall:.2f} "
                               f"(${room:.2f} away) vs ${needed:.2f} needed")

        # 3. Sanity veto - the stop must sit on the correct side of the entry
        if primary_rec == "BUY" and veto_ar is None and not (primary_sl < curr_price < primary_tp1):
            veto_ar = "مستويات الوقف/الهدف غير منطقية بالنسبة للسعر الحالي - تم إلغاء الإشارة"
            veto_en = "SL/TP levels inconsistent with current price - signal voided"
        elif primary_rec == "SELL" and veto_ar is None and not (primary_tp1 < curr_price < primary_sl):
            veto_ar = "مستويات الوقف/الهدف غير منطقية بالنسبة للسعر الحالي - تم إلغاء الإشارة"
            veto_en = "SL/TP levels inconsistent with current price - signal voided"

        if veto_ar:
            primary_rec = "HOLD"
            primary_grade = "NO_TRADE"
            primary_reason_ar = veto_ar
            primary_reason_en = veto_en
            active_trade_type = "HYBRID_SCAN"

        calc_rr = round(primary_tp_pips / (primary_sl_pips + 1e-9), 2)

        return {
            "symbol": "XAUUSD",
            "name": "Gold / US Dollar",
            "current_price": round(curr_price, 2),
            "last_closed_price": round(last_m15_close, 2),
            "strategy_profile": strategy_profile,
            "active_trade_type": active_trade_type,
            "recommendation": primary_rec,
            "confidence_percent": primary_conf,
            "setup_quality": primary_grade,
            "suggested_sl": primary_sl,
            "suggested_tp": primary_tp1,
            "suggested_tp1": primary_tp1,
            "suggested_tp2": primary_tp2,
            "suggested_tp3": primary_tp3,
            "risk_reward_ratio": calc_rr,
            "rr_ratio": calc_rr,
            "pips_target": primary_tp_pips,
            "pips_stop": primary_sl_pips,
            "dollar_target": round(abs(primary_tp1 - curr_price), 2),
            "dollar_stop": round(abs(curr_price - primary_sl), 2),
            "atr": round(gold_atr, 2),
            "rsi": round(m15_data["rsi"], 1),
            "adx": round(m15_adx, 1),
            "macro_trend": macro_trend,
            "m15_trend": m15_trend,
            "session_name": kz_info.get("session_name", "UNKNOWN"),
            "bollinger": {
                "upper": m15_data["bb_upper"],
                "mid": m15_data["bb_mid"],
                "lower": m15_data["bb_lower"],
                "bandwidth": m15_data["bb_bandwidth"],
                "status": m15_data["bb_status"]
            },
            "ema_cloud": {
                "ema20": m15_data["ema20"],
                "ema50": m15_data["ema50"],
                "ema200": m15_data["ema200"]
            },
            "ict_smc": {
                "bullish_ob": m15_ob_bull,
                "bearish_ob": m15_ob_bear,
                "bullish_fvg": m15_fvg_bull,
                "bearish_fvg": m15_fvg_bear,
                "liquidity_sweeps": m15_data.get("liquidity_sweeps", {}),
                "session_levels": session_sweeps
            },
            "fibonacci": {
                "golden_low": m15_data["fibo_gz_low"],
                "golden_high": m15_data["fibo_gz_high"],
                "bias": m15_data["fibo_bias"],
                "tp_target": m15_data["fibo_tp"]
            },
            "classical": {
                "support": m15_data["support"],
                "resistance": m15_data["resistance"]
            },
            "dxy_correlation": {
                "bias": dxy_bias,
                "desc": dxy_desc
            },
            "market_regime": market_regime,
            "split_targets": {
                "tp1": primary_tp1,
                "tp2": primary_tp2,
                "tp3": primary_tp3,
                "sl": primary_sl,
                "tp1_pips": primary_tp_pips,
                "tp2_pips": round(abs(primary_tp2 - curr_price) * 10.0, 1),
                "tp3_pips": round(abs(primary_tp3 - curr_price) * 10.0, 1),
                "sl_pips": primary_sl_pips
            },
            "multi_target_enabled": trade_learning_engine.is_multi_target_enabled(),
            "confluence_score": {
                "buy_score": buy_score,
                "sell_score": sell_score,
                "min_required": MIN_BASE_SCORE,
                "reasons": confluence_notes
            },
            "learning_insights": trade_learning_engine.get_learning_summary(),
            "scalp_analysis": scalp_data,
            "swing_analysis": {
                "recommendation": swing_rec,
                "confidence": swing_conf,
                "grade": swing_grade,
                "sl": swing_sl,
                "tp1": swing_tp1,
                "tp2": swing_tp2,
                "tp3": swing_tp3,
                "pips_target": swing_tp_pips,
                "pips_stop": swing_sl_pips,
                "base_score": swing_base,
                "boosted_score": swing_boosted
            },
            "no_entry_reason": primary_reason_en,
            "no_entry_reason_ar": primary_reason_ar,
            "planned_exit_reason": (f"R-Multiple Targets: TP1 ${primary_tp1:.2f} (1.8R) | "
                                    f"TP2 ${primary_tp2:.2f} (3R) | TP3 ${primary_tp3:.2f} (5R) | "
                                    f"SL ${primary_sl:.2f}")
        }


gold_engine = GoldTraderEngine()
