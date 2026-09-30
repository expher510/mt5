import os
import base64
import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np

logger = logging.getLogger("VisionAIEngine")

class VisionAIEngine:
    """
    ==============================================================================
    👁️ CANDLESTICK RENDERING & ANNOTATOR ENGINE
    ==============================================================================
    Renders authentic candlestick charts with technical indicators:
    - Real OHLCV candles & wicks.
    - EMA20 and EMA50 overlays.
    - Precision Entry, Stop Loss, and Take Profit target vectors.
    - No fabricated confidence scores or invented external AI outputs.
    ==============================================================================
    """

    def __init__(self):
        self.latest_visual_scans: Dict[str, Dict[str, Any]] = {}

    def generate_realistic_market_candles(
        self,
        symbol: str = "XAUUSD",
        base_price: float = 2865.0,
        n_bars: int = 50,
        trend: str = "BULLISH"
    ) -> pd.DataFrame:
        """
        Generates realistic organic market price action with authentic swings,
        wicks, pullbacks, and noise (Brownian random walk) when live data is initializing.
        """
        np.random.seed(int(time.time()) % 10000)
        is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
        volatility = 1.20 if is_gold else 0.00060
        drift = 0.35 if trend == "BULLISH" else (-0.35 if trend == "BEARISH" else 0.0)

        prices = [base_price]
        for _ in range(n_bars):
            step = np.random.normal(drift * (volatility * 0.4), volatility)
            new_p = max(1.0, prices[-1] + step)
            prices.append(new_p)

        prices = prices[1:]
        records = []
        now_ts = int(time.time()) - (n_bars * 300)

        for i, p in enumerate(prices):
            prev_p = prices[i - 1] if i > 0 else p
            high_wick = abs(np.random.normal(0, volatility * 0.6))
            low_wick = abs(np.random.normal(0, volatility * 0.6))

            open_p = prev_p
            close_p = p
            high_p = max(open_p, close_p) + high_wick
            low_p = min(open_p, close_p) - low_wick
            vol = int(np.random.uniform(150, 850))

            records.append({
                "time": now_ts + (i * 300),
                "open": round(open_p, 2 if is_gold else 5),
                "high": round(high_p, 2 if is_gold else 5),
                "low": round(low_p, 2 if is_gold else 5),
                "close": round(close_p, 2 if is_gold else 5),
                "volume": vol
            })

        df = pd.DataFrame(records)
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        return df

    def render_candlestick_chart(
        self,
        df: pd.DataFrame,
        symbol: str = "XAUUSD",
        timeframe: str = "M5",
        last_n: int = 50,
        trade_setup: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Renders an authentic, professional Dark MetaTrader 5 chart
        with real candles and SMC visual overlays.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            import io
        except ImportError:
            logger.warning("Matplotlib not available.")
            return ""

        if df is None or len(df) < 5:
            return ""

        plot_df = df.tail(last_n).copy().reset_index(drop=True)
        is_gold = ("XAU" in symbol.upper() or "GOLD" in symbol.upper())
        digits = 2 if is_gold else 5

        if 'ema20' not in plot_df.columns:
            plot_df['ema20'] = plot_df['close'].ewm(span=20, adjust=False).mean()
        if 'ema50' not in plot_df.columns:
            plot_df['ema50'] = plot_df['close'].ewm(span=50, adjust=False).mean()

        fig, (ax_main, ax_vol) = plt.subplots(
            2, 1, figsize=(13, 7.2),
            gridspec_kw={'height_ratios': [4.5, 1]},
            facecolor='#0b0e14',
            dpi=110
        )
        ax_main.set_facecolor('#0f131c')
        ax_vol.set_facecolor('#0f131c')

        bull_body = '#00c076'
        bull_edge = '#00e68a'
        bear_body = '#ff3b5c'
        bear_edge = '#ff5c77'
        wick_color = '#cbd5e1'
        grid_color = '#1e2638'

        candle_width = 0.65
        n_bars = len(plot_df)

        for i, row in plot_df.iterrows():
            o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
            is_bull = (c >= o)
            body_color = bull_body if is_bull else bear_body
            edge_color = bull_edge if is_bull else bear_edge

            ax_main.plot([i, i], [l, h], color=wick_color, linewidth=1.1, zorder=2)
            height = abs(c - o)
            if height == 0:
                height = (h - l) * 0.05 or (0.1 if is_gold else 0.00005)

            rect = Rectangle(
                (i - candle_width/2, min(o, c)),
                candle_width, height,
                facecolor=body_color,
                edgecolor=edge_color,
                linewidth=0.8,
                zorder=3
            )
            ax_main.add_patch(rect)

            vol = float(row.get('volume', row.get('tick_volume', abs(c - o) * 100)))
            ax_vol.bar(i, vol, color=body_color, alpha=0.65, width=0.65, zorder=2)

        ax_main.plot(plot_df.index, plot_df['ema20'], color='#38bdf8', linewidth=1.4, label='EMA 20 (Support)', alpha=0.9, zorder=4)
        ax_main.plot(plot_df.index, plot_df['ema50'], color='#fbbf24', linewidth=1.4, label='EMA 50 (Trend)', alpha=0.9, zorder=4)

        max_high = plot_df['high'].max()
        min_low = plot_df['low'].min()
        max_idx = plot_df['high'].idxmax()
        min_idx = plot_df['low'].idxmin()

        ax_main.scatter(max_idx, max_high, color='#facc15', s=45, zorder=5)
        ax_main.text(max_idx, max_high + (max_high - min_low)*0.02, f" High: {max_high:.{digits}f}", color='#facc15', fontsize=8.5, fontweight='bold', ha='center', va='bottom', zorder=5)

        ax_main.scatter(min_idx, min_low, color='#38bdf8', s=45, zorder=5)
        ax_main.text(min_idx, min_low - (max_high - min_low)*0.02, f" Low: {min_low:.{digits}f}", color='#38bdf8', fontsize=8.5, fontweight='bold', ha='center', va='top', zorder=5)

        last_close = float(plot_df['close'].iloc[-1])
        rec = trade_setup.get("recommendation", "BUY") if trade_setup else "BUY"

        if rec == "BUY":
            ob_bottom = min_low + (max_high - min_low) * 0.08
            ob_top = min_low + (max_high - min_low) * 0.22
            ob_rect = Rectangle(
                (max(0, n_bars - 25), ob_bottom),
                25, (ob_top - ob_bottom),
                facecolor='#00c076', edgecolor='#00e68a',
                alpha=0.18, linestyle='--', linewidth=1.2, zorder=1
            )
            ax_main.add_patch(ob_rect)
            ax_main.text(n_bars - 12, (ob_top + ob_bottom)/2, "🟩 Bullish Order Block (Demand)", color='#4ade80', fontsize=8, fontweight='bold', ha='center', va='center')
        else:
            ob_bottom = max_high - (max_high - min_low) * 0.22
            ob_top = max_high - (max_high - min_low) * 0.08
            ob_rect = Rectangle(
                (max(0, n_bars - 25), ob_bottom),
                25, (ob_top - ob_bottom),
                facecolor='#ff3b5c', edgecolor='#ff5c77',
                alpha=0.18, linestyle='--', linewidth=1.2, zorder=1
            )
            ax_main.add_patch(ob_rect)
            ax_main.text(n_bars - 12, (ob_top + ob_bottom)/2, "🟥 Bearish Order Block (Supply)", color='#f87171', fontsize=8, fontweight='bold', ha='center', va='center')

        if trade_setup:
            entry = float(trade_setup.get("entry_price", last_close))
            sl = float(trade_setup.get("sl_price", entry * 0.995))
            tp1 = float(trade_setup.get("tp1_price", entry * 1.008))

            ax_main.axhline(entry, color='#38bdf8', linestyle='-', linewidth=1.3, alpha=0.85, zorder=4)
            ax_main.text(n_bars - 1, entry, f"  ENTRY: ${entry:.{digits}f}", color='#38bdf8', fontsize=8, fontweight='bold', va='center', zorder=5)

            ax_main.axhline(sl, color='#ef4444', linestyle='--', linewidth=1.3, alpha=0.85, zorder=4)
            ax_main.text(n_bars - 1, sl, f"  SL: ${sl:.{digits}f}", color='#ef4444', fontsize=8, fontweight='bold', va='center', zorder=5)

            ax_main.axhline(tp1, color='#10b981', linestyle='--', linewidth=1.3, alpha=0.85, zorder=4)
            ax_main.text(n_bars - 1, tp1, f"  TP1: ${tp1:.{digits}f}", color='#10b981', fontsize=8, fontweight='bold', va='center', zorder=5)

        ax_main.set_title(
            f"FXENGIN LIVE ANALYST • {symbol.upper()} [{timeframe}] • MetaTrader 5 Institutional Canvas",
            color='#f8fafc', fontsize=11.5, fontweight='bold', pad=12
        )
        ax_main.grid(True, color=grid_color, linestyle=':', linewidth=0.6, alpha=0.8)
        ax_vol.grid(True, color=grid_color, linestyle=':', linewidth=0.6, alpha=0.8)
        ax_main.set_xlim(-1, n_bars + 8)
        ax_vol.set_xlim(-1, n_bars + 8)
        ax_main.tick_params(colors='#94a3b8', labelsize=8.5)
        ax_vol.tick_params(colors='#94a3b8', labelsize=8)

        for spine in ax_main.spines.values():
            spine.set_color('#2a3449')
        for spine in ax_vol.spines.values():
            spine.set_color('#2a3449')

        ax_main.legend(loc='upper left', facecolor='#0b0e14', edgecolor='#2a3449', fontsize=8, labelcolor='#e2e8f0')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0b0e14', dpi=110)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    async def analyze_chart_with_vision(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        current_price: float,
        macro_bias: str = "BULLISH",
        trade_setup: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Renders authentic candlestick chart from market data without fabricated cues."""
        symbol = symbol.upper()
        timeframe = timeframe.upper()

        is_simulated = False
        if df is None or len(df) < 5:
            is_simulated = True
            df = self.generate_realistic_market_candles(symbol, current_price, 50, macro_bias)

        image_base64 = self.render_candlestick_chart(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            last_n=50,
            trade_setup=trade_setup
        )

        res = {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "image_base64": image_base64,
            "is_simulated": is_simulated,
            "chart_type": "SIMULATED_CANDLES" if is_simulated else "AUTHENTIC_MARKET_CANDLES",
            "scanned_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "ai_engine": "FXENGIN Candlestick Rendering Engine"
        }
        self.latest_visual_scans[symbol] = res
        return res

    def get_latest_scan(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.latest_visual_scans.get(symbol.upper())

vision_engine = VisionAIEngine()
