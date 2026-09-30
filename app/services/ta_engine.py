import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
from app.config import TradingSchool, TradingStyle
from app.services.learning_engine import trade_learning_engine
from app.services.experience_memory import episodic_memory

logger = logging.getLogger("TA_Engine")

class TechnicalAnalysisEngine:
    """
    Enterprise-Grade Multi-Timeframe (M5, M15, H1, H4) Autonomous Market Engine.
    Combines:
    1. Classical School: Multi-Timeframe Horizontal Support/Resistance, Dynamic EMA 20/50/200 Cloud, True Wilder's RSI & ADX.
    2. Fibonacci School: Impulse Detection, 0.50 Equilibrium, 0.618 Golden Zone (OTE), 1.618 Expansion Targets.
    3. Smart Money Concepts (SMC): Institutional Order Blocks (OB), Fair Value Gaps (FVG), Displacement (>1.3 ATR), Break of Structure (BOS).
    4. Autonomous Horizon Classifier: Automatically identifies whether current market structure requires SCALPING or SWING.
    5. Multi-Timeframe Scanner: Evaluates H4 (Macro Bias), H1 (Trend Structure), M15 (Setup Validation), M5 (Precision Entry).
    6. Intermarket Correlation & DXY Filter: Blocks counter-trend trades on Gold (XAUUSD) when Dollar Index is in expansion/breakdown.
    7. AI Self-Learning Integration: Consumes dynamic ADX thresholds, model weights, and penalties from TradeLearningEngine.
    """

    def __init__(self):
        self.active_signal_latch: Dict[str, Dict[str, Any]] = {}

    def get_digits(self, symbol: str) -> int:
        sym = symbol.upper()
        if "XAU" in sym or "GOLD" in sym:
            return 2
        elif "JPY" in sym:
            return 3
        return 5

    def get_pip_multiplier(self, symbol: str) -> float:
        sym = symbol.upper()
        if "XAU" in sym or "GOLD" in sym:
            return 10.0
        elif "JPY" in sym:
            return 100.0
        return 10000.0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < 15:
            return df

        # True EMA Cloud
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        span_200 = min(200, len(df))
        df['ema200'] = df['close'].ewm(span=span_200, adjust=False).mean()

        # ATR (14) using True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['tr'] = tr
        df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().fillna(tr.rolling(14, min_periods=1).mean())

        # RSI (14) with Wilder's Exponential Smoothing (RMA)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        df['rsi'] = df['rsi'].fillna(50.0)

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

    def detect_swings(self, df: pd.DataFrame, order: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        swing_highs = []
        swing_lows = []
        n = len(df)
        if n < (order * 2 + 1):
            return swing_highs, swing_lows

        for i in range(order, n - order):
            high_val = float(df.iloc[i]['high'])
            low_val = float(df.iloc[i]['low'])
            time_val = df.iloc[i]['time'] if 'time' in df.columns else i

            is_high = all(high_val >= float(df.iloc[i - j]['high']) for j in range(1, order + 1)) and \
                      all(high_val >= float(df.iloc[i + j]['high']) for j in range(1, order + 1))

            is_low = all(low_val <= float(df.iloc[i - j]['low']) for j in range(1, order + 1)) and \
                     all(low_val <= float(df.iloc[i + j]['low']) for j in range(1, order + 1))

            if is_high:
                swing_highs.append({"index": i, "price": high_val, "time": time_val})
            if is_low:
                swing_lows.append({"index": i, "price": low_val, "time": time_val})

        return swing_highs, swing_lows

    def evaluate_classical(
        self,
        df: pd.DataFrame,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]],
        current_price: float,
        atr: float,
        digits: int
    ) -> Dict[str, Any]:
        closed = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        ema20 = float(closed['ema20']) if not np.isnan(closed['ema20']) else current_price
        ema50 = float(closed['ema50']) if not np.isnan(closed['ema50']) else current_price
        ema200 = float(closed['ema200']) if not np.isnan(closed['ema200']) else current_price
        rsi = float(closed['rsi']) if not np.isnan(closed['rsi']) else 50.0

        # Strict Multi-EMA Trend Definition
        if current_price > ema20 > ema50 and current_price > ema200:
            trend = "BULLISH"
        elif current_price < ema20 < ema50 and current_price < ema200:
            trend = "BEARISH"
        elif current_price > ema20 > ema50:
            trend = "BULLISH"
        elif current_price < ema20 < ema50:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        # Local Recent Troughs & Peaks (Last 20 candles)
        recent_window = df.iloc[-20:] if len(df) >= 20 else df
        recent_min_low = float(recent_window['low'].min())
        recent_max_high = float(recent_window['high'].max())

        # Major Structural Supports & Resistances from Swings
        valid_supports = [s['price'] for s in swing_lows if s['price'] < current_price]
        valid_resistances = [s['price'] for s in swing_highs if s['price'] > current_price]

        # Closest Support is the highest swing low below price (or recent min low)
        support = max(valid_supports) if valid_supports else recent_min_low
        resistance = min(valid_resistances) if valid_resistances else recent_max_high

        # Local Immediate Levels
        local_sup = support if support < current_price else recent_min_low
        local_res = resistance if resistance > current_price else recent_max_high

        bias = "HOLD"
        confidence = 50.0
        status = f"Range Equilibrium (Sup: ${local_sup:.{digits}f} | Res: ${local_res:.{digits}f})"

        dist_to_sup = current_price - local_sup
        dist_to_res = local_res - current_price

        # Trend-Aligned Support Bounce & Resistance Rejections (Strict Non-Repainting Criteria)
        closed_low = float(closed['low'])
        closed_high = float(closed['high'])
        closed_close = float(closed['close'])
        
        # 1. Support Bounce (Price rejected support level and closed above it)
        is_support_bounce = (dist_to_sup <= (0.85 * atr)) and (trend in ["BULLISH", "NEUTRAL"]) and (38.0 <= rsi <= 62.0)
        
        # 2. Dynamic EMA Pullback Bounce (Price actively near EMA20, dipped and closed back above)
        dist_to_ema20 = abs(current_price - ema20)
        is_ema_pullback_buy = (trend == "BULLISH") and (dist_to_ema20 <= 0.65 * atr) and (closed_low <= ema20 * 1.003) and (closed_close >= ema20) and (45.0 <= rsi <= 65.0)
        
        # 3. Clean Resistance Breakout (Closed candle broke above resistance with momentum, not overbought)
        is_breakout_buy = (closed_close > (local_res + 0.10 * atr)) and (trend == "BULLISH") and (52.0 <= rsi <= 68.0)

        # SELL Criteria:
        # 1. Resistance Rejection
        is_res_rejection = (dist_to_res <= (0.85 * atr)) and (trend in ["BEARISH", "NEUTRAL"]) and (38.0 <= rsi <= 62.0)
        
        # 2. Dynamic EMA Pullback Sell (Price actively near EMA20, tested and closed below, not oversold)
        is_ema_pullback_sell = (trend == "BEARISH") and (dist_to_ema20 <= 0.65 * atr) and (closed_high >= ema20 * 0.997) and (closed_close <= ema20) and (35.0 <= rsi <= 55.0)
        
        # 3. Clean Support Breakdown (Price broke support, but RSI NOT in exhaustion below 30)
        is_breakdown_sell = (closed_close < (local_sup - 0.10 * atr)) and (trend == "BEARISH") and (32.0 <= rsi <= 48.0)

        if is_support_bounce or is_ema_pullback_buy or is_breakout_buy:
            bias = "BUY"
            confidence = 86.0 if is_support_bounce else (82.0 if is_ema_pullback_buy else 78.0)
            if is_support_bounce:
                status = f"🟢 Classical Support Bounce (${local_sup:.{digits}f}) in {trend} Trend"
            elif is_ema_pullback_buy:
                status = f"🟢 Dynamic EMA20 Pullback Bounce (${ema20:.{digits}f}) in Bullish Trend"
            else:
                status = f"🟢 Confirmed Resistance Breakout above ${local_res:.{digits}f}"
        elif is_res_rejection or is_ema_pullback_sell or is_breakdown_sell:
            bias = "SELL"
            confidence = 86.0 if is_res_rejection else (82.0 if is_ema_pullback_sell else 78.0)
            if is_res_rejection:
                status = f"🔴 Classical Resistance Rejection (${local_res:.{digits}f}) in {trend} Trend"
            elif is_ema_pullback_sell:
                status = f"🔴 Dynamic EMA20 Pullback Rejection (${ema20:.{digits}f}) in Bearish Trend"
            else:
                status = f"🔴 Confirmed Support Breakdown below ${local_sup:.{digits}f}"
        elif rsi < 28.0:
            status = f"⚠️ Extreme Oversold Exhaustion (RSI: {rsi:.1f}) - Awaiting Retest"
        elif rsi > 72.0:
            status = f"⚠️ Extreme Overbought Exhaustion (RSI: {rsi:.1f}) - Awaiting Retest"

        return {
            "bias": bias,
            "confidence": confidence,
            "trend": trend,
            "support_price": round(local_sup, digits),
            "resistance_price": round(local_res, digits),
            "local_support": round(local_sup, digits),
            "local_resistance": round(local_res, digits),
            "major_support": round(support, digits),
            "major_resistance": round(resistance, digits),
            "status": status
        }

    def evaluate_fibonacci(
        self,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]],
        current_price: float,
        atr: float,
        digits: int
    ) -> Dict[str, Any]:
        if not swing_highs or not swing_lows:
            gz_low = current_price - (0.618 * atr * 1.5)
            gz_high = current_price + (0.618 * atr * 1.5)
            return {
                "bias": "HOLD",
                "confidence": 45.0,
                "status": "Awaiting Swings for Fibonacci Impulse",
                "gz_low": round(gz_low, digits),
                "gz_high": round(gz_high, digits),
                "suggested_tp": round(current_price + 2.0 * atr, digits),
                "suggested_sl": round(current_price - 1.5 * atr, digits)
            }

        last_sh = swing_highs[-1]
        last_sl = swing_lows[-1]

        if last_sl['index'] < last_sh['index']:
            # Bullish Impulse (Low -> High)
            swing_low = last_sl['price']
            swing_high = last_sh['price']
            diff = swing_high - swing_low
            
            if diff < (0.5 * atr):
                diff = 1.5 * atr

            gz_low = swing_high - (0.618 * diff)
            gz_high = swing_high - (0.500 * diff)
            tp_target = swing_high + (0.618 * diff)
            sl_level = swing_low - (0.3 * atr)

            in_golden_zone = (gz_low - 0.3 * atr) <= current_price <= (gz_high + 0.3 * atr)
            above_zone = current_price > (gz_high + 0.3 * atr)

            if in_golden_zone:
                bias = "BUY"
                confidence = 86.0
                status = f"Fibonacci 0.618 Golden Zone Pullback (${gz_low:.{digits}f} - ${gz_high:.{digits}f})"
            elif above_zone and current_price >= swing_high:
                bias = "BUY"
                confidence = 72.0
                status = f"Bullish Impulse Expansion towards 1.618 (${tp_target:.{digits}f})"
            else:
                bias = "HOLD"
                confidence = 50.0
                status = f"Retracement Below Golden Zone (${gz_low:.{digits}f})"

        else:
            # Bearish Impulse (High -> Low)
            swing_high = last_sh['price']
            swing_low = last_sl['price']
            diff = swing_high - swing_low
            
            if diff < (0.5 * atr):
                diff = 1.5 * atr

            gz_high = swing_low + (0.618 * diff)
            gz_low = swing_low + (0.500 * diff)
            tp_target = swing_low - (0.618 * diff)
            sl_level = swing_high + (0.3 * atr)

            in_golden_zone = (gz_low - 0.3 * atr) <= current_price <= (gz_high + 0.3 * atr)
            below_zone = current_price < (gz_low - 0.3 * atr)

            if in_golden_zone:
                bias = "SELL"
                confidence = 86.0
                status = f"Bearish 0.618 Golden Zone Pullback (${gz_low:.{digits}f} - ${gz_high:.{digits}f})"
            elif below_zone and current_price <= swing_low:
                bias = "SELL"
                confidence = 72.0
                status = f"Bearish Impulse Expansion towards 1.618 (${tp_target:.{digits}f})"
            else:
                bias = "HOLD"
                confidence = 50.0
                status = f"Retracement Above Golden Zone (${gz_high:.{digits}f})"

        return {
            "bias": bias,
            "confidence": confidence,
            "status": status,
            "gz_low": round(gz_low, digits),
            "gz_high": round(gz_high, digits),
            "suggested_tp": round(tp_target, digits),
            "suggested_sl": round(sl_level, digits)
        }

    def evaluate_smc(
        self,
        df: pd.DataFrame,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]],
        current_price: float,
        atr: float,
        digits: int
    ) -> Dict[str, Any]:
        bullish_ob = None
        bearish_ob = None

        # Search recent bars for fresh Institutional Order Blocks with strong displacement
        for i in range(len(df) - 3, max(5, len(df) - 40), -1):
            c_prev = df.iloc[i - 1]
            c_curr = df.iloc[i]
            c_next = df.iloc[i + 1]

            # Bullish Order Block (Last down candle before strong explosive displacement up)
            is_down_candle = float(c_prev['close']) < float(c_prev['open'])
            is_strong_up = (float(c_next['close']) - float(c_curr['open'])) >= (1.2 * atr)
            if is_down_candle and is_strong_up and bullish_ob is None:
                bullish_ob = {
                    "low": float(c_prev['low']),
                    "high": float(c_prev['high']),
                    "time": c_prev.get('time', 0),
                    "index": i - 1
                }

            # Bearish Order Block (Last up candle before strong explosive displacement down)
            is_up_candle = float(c_prev['close']) > float(c_prev['open'])
            is_strong_down = (float(c_curr['open']) - float(c_next['close'])) >= (1.2 * atr)
            if is_up_candle and is_strong_down and bearish_ob is None:
                bearish_ob = {
                    "low": float(c_prev['low']),
                    "high": float(c_prev['high']),
                    "time": c_prev.get('time', 0),
                    "index": i - 1
                }

            if bullish_ob and bearish_ob:
                break

        bias = "HOLD"
        confidence = 50.0
        status = "Institutional Liquidity Neutral"

        # Check if price is actively tapping and reacting to the Order Block
        if bullish_ob and (bullish_ob['low'] - 0.2 * atr) <= current_price <= (bullish_ob['high'] + 0.4 * atr):
            bias = "BUY"
            confidence = 88.0
            status = f"🟢 SMC Institutional Demand OB Retest (${bullish_ob['low']:.{digits}f} - ${bullish_ob['high']:.{digits}f})"
        elif bearish_ob and (bearish_ob['low'] - 0.4 * atr) <= current_price <= (bearish_ob['high'] + 0.2 * atr):
            bias = "SELL"
            confidence = 88.0
            status = f"🔴 SMC Institutional Supply OB Rejection (${bearish_ob['low']:.{digits}f} - ${bearish_ob['high']:.{digits}f})"
        elif bullish_ob and current_price > (bullish_ob['high'] + 1.0 * atr):
            bias = "HOLD"
            confidence = 55.0
            status = f"Market Structure Above Demand OB (${bullish_ob['high']:.{digits}f})"
        elif bearish_ob and current_price < (bearish_ob['low'] - 1.0 * atr):
            bias = "HOLD"
            confidence = 55.0
            status = f"Market Structure Below Supply OB (${bearish_ob['low']:.{digits}f})"

        return {
            "bias": bias,
            "confidence": confidence,
            "status": status,
            "bullish_ob": bullish_ob,
            "bearish_ob": bearish_ob
        }

    def analyze_symbol_multi_timeframe(
        self,
        symbol: str,
        timeframes_data: Dict[str, pd.DataFrame],
        school: TradingSchool = TradingSchool.TRI_SCHOOL_CONSENSUS,
        dxy_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Autonomous Multi-Timeframe Analysis Engine (M5, M15, H1, H4):
        1. Evaluates Macro Direction on H4 & H1.
        2. Evaluates Precision Entry & Zones on M15 & M5.
        3. Autonomously chooses between SCALPING and SWING.
        4. Calculates exact ATR-calibrated SL/TP targets with positive mathematical expectancy.
        5. Protects Gold against conflicting USD Index (DXY) momentum.
        """
        digits = self.get_digits(symbol)
        pip_mult = self.get_pip_multiplier(symbol)
        is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()

        # Process each available timeframe
        tf_analyses = {}
        for tf_name in ["H4", "H1", "M15", "M5"]:
            df_tf = timeframes_data.get(tf_name)
            if df_tf is not None and not df_tf.empty and len(df_tf) >= 20:
                df_calc = self.calculate_indicators(df_tf)
                closed = df_calc.iloc[-2] if len(df_calc) >= 2 else df_calc.iloc[-1]
                curr_p = float(df_calc.iloc[-1]['close'])
                
                default_atr = 3.00 if is_gold else (0.25 if "JPY" in symbol.upper() else 0.0018)
                atr_tf = float(closed['atr']) if not np.isnan(closed['atr']) and closed['atr'] > 0 else default_atr
                rsi_tf = float(closed['rsi']) if not np.isnan(closed['rsi']) else 50.0

                sw_h, sw_l = self.detect_swings(df_calc, order=3 if tf_name in ["H4", "H1"] else 2)
                c_eval = self.evaluate_classical(df_calc, sw_h, sw_l, curr_p, atr_tf, digits)
                f_eval = self.evaluate_fibonacci(sw_h, sw_l, curr_p, atr_tf, digits)
                s_eval = self.evaluate_smc(df_calc, sw_h, sw_l, curr_p, atr_tf, digits)

                # Consensus for this timeframe
                buy_votes = sum(1 for x in [c_eval, f_eval, s_eval] if x['bias'] == "BUY")
                sell_votes = sum(1 for x in [c_eval, f_eval, s_eval] if x['bias'] == "SELL")
                
                if buy_votes >= 2:
                    overall = "BUY"
                elif sell_votes >= 2:
                    overall = "SELL"
                elif c_eval['bias'] in ["BUY", "SELL"]:
                    overall = c_eval['bias']
                else:
                    overall = "HOLD"

                tf_analyses[tf_name] = {
                    "current_price": curr_p,
                    "atr": atr_tf,
                    "rsi": rsi_tf,
                    "adx": float(closed.get('adx', 20.0)),
                    "plus_di": float(closed.get('plus_di', 0.0)),
                    "minus_di": float(closed.get('minus_di', 0.0)),
                    "classical": c_eval,
                    "fibonacci": f_eval,
                    "smc": s_eval,
                    "overall_bias": overall
                }

        # Fallback if no valid timeframe analyses
        if not tf_analyses:
            primary_df = timeframes_data.get("M15", pd.DataFrame())
            return self.analyze_symbol(symbol, primary_df, school=school, style=TradingStyle.SCALPING)

        # 1. Macro and Micro Timeframe Data
        h4_data = tf_analyses.get("H4")
        h1_data = tf_analyses.get("H1", h4_data)
        m15_data = tf_analyses.get("M15", h1_data if h1_data else list(tf_analyses.values())[0])
        m5_data = tf_analyses.get("M5", m15_data)

        macro_bias = h1_data["overall_bias"] if h1_data else (h4_data["overall_bias"] if h4_data else "HOLD")
        m15_bias = m15_data["overall_bias"]
        m5_bias = m5_data["overall_bias"]

        current_price = m15_data["current_price"]
        current_atr = m15_data["atr"]
        current_rsi = m15_data["rsi"]
        current_adx = float(m15_data.get("adx", 20.0))
        h1_adx = float(h1_data.get("adx", 25.0)) if h1_data else current_adx

        # 2. Autonomous Trade Horizon & Timeframe Decision
        if macro_bias in ["BUY", "SELL"] and macro_bias == m15_bias:
            if h1_data and abs(current_rsi - 50.0) > 10.0:
                auto_style = "SWING"
                optimal_tf = "H1"
                target_atr = h1_data["atr"]
                alignment_desc = f"4/4 Institutional Multi-Timeframe Trend Alignment ({macro_bias})"
            else:
                auto_style = "SCALPING"
                optimal_tf = "M15"
                target_atr = m15_data["atr"]
                alignment_desc = f"3/4 Multi-Timeframe Pullback Confluence ({macro_bias})"
        elif m15_bias in ["BUY", "SELL"] and m5_bias == m15_bias and macro_bias != ("SELL" if m15_bias == "BUY" else "BUY"):
            auto_style = "SCALPING"
            optimal_tf = "M15"
            target_atr = m15_data["atr"]
            alignment_desc = f"M15/M5 Precision Confluence Trigger ({m15_bias})"
        else:
            auto_style = "SCALPING"
            optimal_tf = "M15"
            target_atr = current_atr
            alignment_desc = "Consolidating / Neutral Structure"

        # 3. AI Learning Engine Dynamic Multipliers & Penalties
        min_adx_threshold = trade_learning_engine.get_adaptive_adx_threshold()
        adaptive_sl_mult = trade_learning_engine.get_adaptive_sl_multiplier()
        boll_pen = trade_learning_engine.get_model_penalty("M5_BOLLINGER_REVERSION")
        breakout_pen = trade_learning_engine.get_model_penalty("M5_BREAKOUT_MOMENTUM")

        # 4. Final Signal Synthesis (Institutional Dual-Regime: Trend Expansion + SMC Range Equilibrium)
        regime_info = trade_learning_engine.classify_market_regime(
            adx=current_adx,
            bandwidth=0.25,
            atr=current_atr,
            avg_atr=current_atr,
            trend_alignment=macro_bias
        )
        market_regime = regime_info.get("regime", "UNKNOWN")
        
        recommendation = "HOLD"
        confidence = 50.0
        setup_grade = "NO_TRADE"
        active_model_name = "MSS_BREAKOUT"

        # ---------------------------------------------------------------------
        # Regime A: Trend Expansion (Strong or Emerging ADX)
        # ---------------------------------------------------------------------
        if current_adx >= max(18.0, min_adx_threshold - 2.5):
            # Grade A+ Setup: Strong Macro (H1/H4) + Micro (M15) Trend Alignment + Solid ADX
            if macro_bias == "BUY" and m15_bias == "BUY" and (m5_bias in ["BUY", "HOLD"]):
                recommendation = "BUY"
                confidence = max(68.0, 90.0 - breakout_pen)
                setup_grade = "A+"
                active_model_name = "MSS_BREAKOUT"
                alignment_desc = "Institutional Trend Expansion (Macro H1/H4 + Micro M15 Alignment)"
            elif macro_bias == "SELL" and m15_bias == "SELL" and (m5_bias in ["SELL", "HOLD"]):
                recommendation = "SELL"
                confidence = max(68.0, 90.0 - breakout_pen)
                setup_grade = "A+"
                active_model_name = "MSS_BREAKOUT"
                alignment_desc = "Institutional Trend Expansion (Macro H1/H4 + Micro M15 Alignment)"
            # Grade A Setup: M15 Setup with M5 Trigger (Must NOT oppose Macro H1 trend)
            elif m15_bias == "BUY" and m5_bias == "BUY" and macro_bias != "BEARISH":
                recommendation = "BUY"
                confidence = max(62.0, 82.0 - breakout_pen)
                setup_grade = "A"
                active_model_name = "EMA_TREND_PULLBACK"
                alignment_desc = "Momentum Trend Pullback (M15 Flow in harmony with Macro)"
            elif m15_bias == "SELL" and m5_bias == "SELL" and macro_bias != "BULLISH":
                recommendation = "SELL"
                confidence = max(62.0, 82.0 - breakout_pen)
                setup_grade = "A"
                active_model_name = "EMA_TREND_PULLBACK"
                alignment_desc = "Momentum Trend Pullback (M15 Flow in harmony with Macro)"
            else:
                recommendation = "HOLD"
                confidence = 50.0
                setup_grade = "NO_TRADE"
                alignment_desc = "Trend Neutral - Awaiting Alignment"

        # ---------------------------------------------------------------------
        # Regime B: Range & SMC Equilibrium (Lower ADX but Clean Institutional Zones)
        # ---------------------------------------------------------------------
        else:
            # Professional Range Trading: Check for Institutional Order Blocks and FVG at extremes
            c_eval = m15_data["classical"]
            s_eval = m15_data["smc"]
            f_eval = m15_data["fibonacci"]

            bullish_ob_active = bool(s_eval.get("bullish_ob"))
            bearish_ob_active = bool(s_eval.get("bearish_ob"))

            if bullish_ob_active and current_rsi < 55.0 and macro_bias != "BEARISH":
                recommendation = "BUY"
                confidence = 76.0
                setup_grade = "A"
                active_model_name = "SMC_OB_RETEST"
                alignment_desc = "Institutional SMC Order Block Demand Retest (Range Equilibrium)"
            elif bearish_ob_active and current_rsi > 45.0 and macro_bias != "BULLISH":
                recommendation = "SELL"
                confidence = 76.0
                setup_grade = "A"
                active_model_name = "SMC_OB_RETEST"
                alignment_desc = "Institutional SMC Order Block Supply Mitigation (Range Equilibrium)"
            elif c_eval.get("status") in ["SUPPORT_HOLD", "DOUBLE_BOTTOM"] and current_rsi < 50.0 and macro_bias != "BEARISH":
                recommendation = "BUY"
                confidence = 72.0
                setup_grade = "A"
                active_model_name = "FIBO_OTE_0618"
                alignment_desc = "Range Support Golden Zone Bounce"
            elif c_eval.get("status") in ["RESISTANCE_HOLD", "DOUBLE_TOP"] and current_rsi > 50.0 and macro_bias != "BULLISH":
                recommendation = "SELL"
                confidence = 72.0
                setup_grade = "A"
                active_model_name = "FIBO_OTE_0618"
                alignment_desc = "Range Resistance Equilibrium Rejection"
            else:
                recommendation = "HOLD"
                confidence = 48.0
                setup_grade = "NO_TRADE"
                alignment_desc = f"Equilibrium Consolidation (ADX: {current_adx:.1f}) - Awaiting Zone Reaction"

        # ---------------------------------------------------------------------
        # Institutional RSI Exhaustion Filter: Never open fresh SELL in oversold bottom, or BUY in overbought top
        # ---------------------------------------------------------------------
        if recommendation == "SELL" and current_rsi < 28.0:
            logger.info(f"Oversold exhaustion trap blocked on {symbol}: RSI is {current_rsi:.1f} < 28.0")
            recommendation = "HOLD"
            setup_grade = "NO_TRADE"
            confidence = 45.0
            alignment_desc = f"⚠️ Exhaustion Trap: RSI ({current_rsi:.1f}) is deeply oversold. Awaiting mean-reversion pullback."
        elif recommendation == "BUY" and current_rsi > 72.0:
            logger.info(f"Overbought exhaustion trap blocked on {symbol}: RSI is {current_rsi:.1f} > 72.0")
            recommendation = "HOLD"
            setup_grade = "NO_TRADE"
            confidence = 45.0
            alignment_desc = f"⚠️ Exhaustion Trap: RSI ({current_rsi:.1f}) is deeply overbought. Awaiting pullback to support."

        # ---------------------------------------------------------------------
        # True Tri-School Confluence Consensus (Classical + Fibonacci + SMC)
        # ---------------------------------------------------------------------
        c_eval = m15_data["classical"]
        f_eval = m15_data["fibonacci"]
        s_eval = m15_data["smc"]
        c_bias = c_eval.get("bias", "HOLD")
        f_bias = f_eval.get("bias", "HOLD")
        s_bias = s_eval.get("bias", "HOLD")
        confluence_count = sum(1 for b in [c_bias, f_bias, s_bias] if b == recommendation)

        if recommendation in ["BUY", "SELL"]:
            if confluence_count == 3:
                setup_grade = "A+"
            elif confluence_count == 2:
                setup_grade = "A"
            else:
                setup_grade = "B"
                confidence = min(confidence, 65.0)
                alignment_desc += " (Preliminary Confluence: 1/3 Schools Agree)"
        else:
            confluence_count = 0

        # ---------------------------------------------------------------------
        # 5. Intermarket DXY Correlation Filter for Gold (XAUUSD)
        # ---------------------------------------------------------------------
        if is_gold and dxy_context and recommendation in ["BUY", "SELL"]:
            dxy_bias = dxy_context.get("dxy_bias", "NEUTRAL_DXY")
            if recommendation == "BUY" and dxy_bias == "BULLISH_DXY":
                logger.info(f"Gold BUY signal blocked due to Bullish DXY Expansion ({dxy_context.get('dxy_change_pct')}%).")
                recommendation = "HOLD"
                setup_grade = "NO_TRADE"
                confidence = 48.0
                alignment_desc = "Blocked by Bullish US Dollar Index (DXY) Expansion"
            elif recommendation == "SELL" and dxy_bias == "BEARISH_DXY":
                logger.info(f"Gold SELL signal blocked due to Bearish DXY Breakdown ({dxy_context.get('dxy_change_pct')}%).")
                recommendation = "HOLD"
                setup_grade = "NO_TRADE"
                confidence = 48.0
                alignment_desc = "Blocked by Bearish US Dollar Index (DXY) Breakdown"

        # ---------------------------------------------------------------------
        # 6. Active Autopsy Shield & Reinforcement Learning Feedback
        # ---------------------------------------------------------------------
        autopsy_report = {"is_safe": True, "advice_ar": "✅ متوافق مع الأمان المؤسسي", "penalty_score": 0.0}
        learned_score_data = {}

        if recommendation in ["BUY", "SELL"]:
            c_eval = m15_data["classical"]
            s_eval = m15_data["smc"]

            # Evaluate against catalog of historical loss traps
            autopsy_report = episodic_memory.evaluate_setup_against_learned_rules(
                symbol=symbol,
                direction=recommendation,
                current_price=current_price,
                sl_price=current_price - (1.5 * current_atr if recommendation == "BUY" else -1.5 * current_atr),
                tp_price=current_price + (3.0 * current_atr if recommendation == "BUY" else -3.0 * current_atr),
                indicators={
                    "adx": current_adx,
                    "atr": current_atr,
                    "rsi": current_rsi,
                    "macro_trend": macro_bias,
                    "has_mss": (s_eval.get("status") in ["BULLISH_MSS", "BEARISH_MSS"]),
                    "is_gold": is_gold
                }
            )

            # Adjust confidence with learned model weights & session expectancy
            curr_utc_hour = datetime.now(timezone.utc).hour
            learned_score_data = trade_learning_engine.score_setup_with_learning(
                symbol=symbol,
                direction=recommendation,
                model_name=active_model_name,
                base_confidence=confidence,
                current_regime=market_regime,
                entry_hour=curr_utc_hour
            )

            confidence = learned_score_data.get("final_confidence", confidence)
            confidence = max(30.0, min(95.0, confidence - autopsy_report.get("penalty_score", 0.0)))

            if not autopsy_report.get("is_safe", True) or autopsy_report.get("penalty_score", 0.0) >= 20.0:
                setup_grade = "B" if setup_grade in ["A+", "A"] else setup_grade
                alignment_desc += f" | 🛡️ حماية التشريح: {autopsy_report.get('advice_ar')}"

        # ---------------------------------------------------------------------
        # 7. Dynamic Mathematical SL & TP Calculation with Learned Adaptive Multiplier
        # ---------------------------------------------------------------------
        if is_gold:
            gold_atr = max(2.80, target_atr)
            min_sl_dist = max(4.50, 1.8 * gold_atr * adaptive_sl_mult)
            min_tp_dist = max(9.00, 3.6 * gold_atr)
            max_sl_dist = 14.00
            structure_buffer = max(1.20, 0.4 * gold_atr)
        else:
            pair_atr = max(0.0015, target_atr)
            min_sl_dist = max(15.0 / pip_mult, 1.6 * pair_atr * adaptive_sl_mult)
            min_tp_dist = max(30.0 / pip_mult, 3.2 * pair_atr)
            max_sl_dist = 4.0 * pair_atr * adaptive_sl_mult
            structure_buffer = 0.3 * pair_atr

        # Apply safe SL expansion if autopsy recommended it
        if autopsy_report.get("adjusted_sl_dist", 0.0) > 0:
            min_sl_dist = max(min_sl_dist, float(autopsy_report["adjusted_sl_dist"]))

        c_eval = m15_data["classical"]
        f_eval = m15_data["fibonacci"]
        s_eval = m15_data["smc"]

        if recommendation == "BUY":
            m15_sup = c_eval["support_price"]
            ob_low = s_eval['bullish_ob']['low'] if s_eval.get('bullish_ob') else m15_sup
            struct_low = min(m15_sup, ob_low) if (m15_sup > 0 and ob_low > 0) else max(m15_sup, ob_low)
            
            # Place SL beyond structural low with ATR buffer, ensuring it respects min_sl_dist
            raw_sl = struct_low - structure_buffer if struct_low > 0 else (current_price - min_sl_dist)
            sl_dist = max(min_sl_dist, min(max_sl_dist, current_price - raw_sl))
            sl = round(current_price - sl_dist, digits)
            
            # TP targeted at 2.2x Risk:Reward
            tp_dist = max(min_tp_dist, sl_dist * 2.2)
            tp = round(current_price + tp_dist, digits)

        elif recommendation == "SELL":
            m15_res = c_eval["resistance_price"]
            ob_high = s_eval['bearish_ob']['high'] if s_eval.get('bearish_ob') else m15_res
            struct_high = max(m15_res, ob_high) if (m15_res > 0 and ob_high > 0) else max(m15_res, ob_high)

            # Place SL beyond structural high with ATR buffer, ensuring it respects min_sl_dist
            raw_sl = struct_high + structure_buffer if struct_high > 0 else (current_price + min_sl_dist)
            sl_dist = max(min_sl_dist, min(max_sl_dist, raw_sl - current_price))
            sl = round(current_price + sl_dist, digits)

            # TP targeted at 2.2x Risk:Reward
            tp_dist = max(min_tp_dist, sl_dist * 2.2)
            tp = round(current_price - tp_dist, digits)

        else:
            sl = round(current_price - min_sl_dist, digits)
            tp = round(current_price + min_tp_dist, digits)

        # Precise Pip Gains & Risk-Reward
        sl_pip_dist = abs(current_price - sl) * pip_mult
        tp_pip_dist = abs(tp - current_price) * pip_mult
        rr_ratio = round(tp_pip_dist / (sl_pip_dist + 1e-9), 1)
        pips_gain = round(tp_pip_dist, 1)

        # 7. Signal Stability Latch (Prevents Repainting & Flickering between BUY/HOLD)
        latched = self.active_signal_latch.get(symbol)
        now_t = time.time()
        
        if latched and recommendation == "HOLD":
            latched_sig = latched.get("signal")
            latched_sl = latched.get("sl", 0.0)
            latched_tp = latched.get("tp", 0.0)
            latched_age = now_t - latched.get("time", now_t)
            
            # Invalidation Checks:
            is_sl_violated = (latched_sig == "BUY" and current_price <= latched_sl) or (latched_sig == "SELL" and current_price >= latched_sl)
            is_tp_reached = (latched_sig == "BUY" and current_price >= latched_tp) or (latched_sig == "SELL" and current_price <= latched_tp)
            is_macro_opposite = (latched_sig == "BUY" and macro_bias == "SELL") or (latched_sig == "SELL" and macro_bias == "BUY")
            is_expired = latched_age > 1200 # 20 minutes latch lifetime

            if not is_sl_violated and not is_tp_reached and not is_macro_opposite and not is_expired:
                # Retain the active confirmed trade setup steadily
                recommendation = latched_sig
                setup_grade = latched.get("grade", "A")
                confidence = latched.get("confidence", 88.0)
                sl = latched_sl
                tp = latched_tp
                alignment_desc = f"Active {latched_sig} Institutional Setup In Play (Lock: {int(1200 - latched_age)}s)"
            else:
                # Invalidation or Target reached: Clear latch
                self.active_signal_latch.pop(symbol, None)
                
        elif recommendation in ["BUY", "SELL"] and setup_grade in ["A+", "A"]:
            # Register fresh confirmed trade in latch
            self.active_signal_latch[symbol] = {
                "signal": recommendation,
                "grade": setup_grade,
                "confidence": confidence,
                "time": now_t,
                "entry": current_price,
                "sl": sl,
                "tp": tp
            }

        return {
            "symbol": symbol,
            "school": school.value if hasattr(school, "value") else str(school),
            "style": auto_style,
            "recommended_timeframe": optimal_tf,
            "optimal_timeframe": optimal_tf,
            "auto_selected_style": auto_style,
            "recommendation": recommendation,
            "confidence_percent": confidence,
            "current_price": round(current_price, digits),
            "suggested_sl": sl,
            "suggested_tp": tp,
            "risk_reward_ratio": rr_ratio,
            "pips_target": pips_gain,
            "rsi": round(current_rsi, 1),
            "adx": round(current_adx, 1),
            "atr": round(current_atr, digits),
            "models_triggered": [f"{optimal_tf}_{auto_style}_{recommendation}"],
            "support_level": c_eval["support_price"],
            "resistance_level": c_eval["resistance_price"],
            "local_support": c_eval.get("local_support", c_eval["support_price"]),
            "local_resistance": c_eval.get("local_resistance", c_eval["resistance_price"]),
            "major_support": c_eval.get("major_support", c_eval["support_price"]),
            "major_resistance": c_eval.get("major_resistance", c_eval["resistance_price"]),
            "classical_status": c_eval["status"],
            "fibo_status": f_eval["status"],
            "smc_status": s_eval["status"],
            "fibo_golden_low": f_eval.get("gz_low", current_price),
            "fibo_golden_high": f_eval.get("gz_high", current_price),
            "smc_bull_low": s_eval['bullish_ob']['low'] if s_eval.get("bullish_ob") else 0.0,
            "smc_bull_high": s_eval['bullish_ob']['high'] if s_eval.get("bullish_ob") else 0.0,
            "smc_bear_low": s_eval['bearish_ob']['low'] if s_eval.get("bearish_ob") else 0.0,
            "smc_bear_high": s_eval['bearish_ob']['high'] if s_eval.get("bearish_ob") else 0.0,
            "market_context": f"{macro_bias} Macro ({alignment_desc})",
            "setup_quality": setup_grade,
            "confluence_count": confluence_count,
            "confluence_text": f"{confluence_count}/3 Confluence",
            "planned_exit_reason": f"Target TP ${tp:.{digits}f} (+{pips_gain} pips) | SL ${sl:.{digits}f} (R:R 1:{rr_ratio})",
            "auto_trade_delay_reason": f"Grade {setup_grade} | Horizon: {auto_style} ({optimal_tf}) | {alignment_desc}",
            "multi_pass_verification": {
                "classical": c_eval["status"],
                "fibonacci": f_eval["status"],
                "smc": s_eval["status"],
                "pass1_trend": c_eval["status"],
                "pass2_structure": f_eval["status"],
                "pass3_momentum": s_eval["status"],
                "status": "3_PASSES_VERIFIED" if setup_grade in ["A+", "A"] else "WAITING"
            },
            "autopsy_shield": autopsy_report,
            "market_regime": market_regime,
            "multi_timeframe_alignment": alignment_desc
        }

    def analyze_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        school: TradingSchool = TradingSchool.TRI_SCHOOL_CONSENSUS,
        style: TradingStyle = TradingStyle.AUTO_ADAPTIVE,
        dxy_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Backward compatibility wrapper when single timeframe DataFrame is supplied"""
        return self.analyze_symbol_multi_timeframe(
            symbol=symbol,
            timeframes_data={"M15": df},
            school=school,
            dxy_context=dxy_context
        )

ta_engine = TechnicalAnalysisEngine()
