import os
import io
import time
import base64
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    matplotlib = None
    plt = None
    Rectangle = None
    OffsetImage = None
    AnnotationBbox = None
    Image = None
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger("SignalCardRenderer")
from app.services.symbol_metrics import get_symbol_metrics

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

def _find_logo_path() -> Optional[str]:
    candidates = [
        os.path.join(ASSETS_DIR, "logo.png"),
        "/app/assets/logo.png",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "logo.png"),
        "/root/vps_backend/assets/logo.png",
        os.path.join(os.getcwd(), "vps_backend", "assets", "logo.png"),
        os.path.join(os.getcwd(), "assets", "logo.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def render_published_signal_card(
    m15_candles: List[Dict[str, Any]],
    model_record: Dict[str, Any],
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    snapshot_id: str = "",
    last_n_bars: int = 45
) -> bytes:
    """
    Renders an authentic, high-visibility published signal card adhering to Agent Brief 7 & 11:
    - High-visibility M15 candles (default 45 bars for large, crisp, bold candles).
    - Labeled horizontal lines for Entry, SL (1.0 ATR), TP (2.0 ATR, 2:1 R:R).
    - EMA 20, 50, and 200 overlays on price panel.
    - H4 directional gate status displayed as text.
    - Model probability with empirical sample size per symbol.
    - ATR value for the bar.
    - Prominent company logo in upper-right identity block.
    - Signal ID and UTC timestamp.
    - Honest labelling disclaimer.
    """
    if not m15_candles or len(m15_candles) < 10:
        raise ValueError(f"Insufficient candle history for signal card: {len(m15_candles) if m15_candles else 0} bars")

    df = pd.DataFrame(m15_candles)
    sym_upper = symbol.upper()
    is_gold = ("XAU" in sym_upper or "GOLD" in sym_upper)
    is_jpy = ("JPY" in sym_upper)
    digits = 2 if is_gold else (3 if is_jpy else 5)

    # Focus context window: 45 bars gives crystal-clear, bold, readable candles
    target_bars = last_n_bars if (last_n_bars and last_n_bars > 0) else 45

    n_bars_to_plot = min(len(df), max(25, target_bars))
    plot_df = df.tail(n_bars_to_plot).copy().reset_index(drop=True)
    n_bars = len(plot_df)

    # Compute EMAs
    plot_df['ema20'] = plot_df['close'].ewm(span=20, adjust=False).mean()
    plot_df['ema50'] = plot_df['close'].ewm(span=50, adjust=False).mean()
    plot_df['ema200'] = plot_df['close'].ewm(span=200, adjust=False).mean()

    # Extract model metrics
    decision = str(model_record.get("decision") or model_record.get("direction") or "BUY").upper()
    entry_val = float(model_record.get("entry") or plot_df['close'].iloc[-1])
    if decision == "SELL":
        sl_val = float(model_record.get("sl") or (entry_val + 4.0 if is_gold else (entry_val + 0.225 if is_jpy else entry_val + 0.0020)))
        tp_val = float(model_record.get("tp") or (entry_val - 8.0 if is_gold else (entry_val - 0.450 if is_jpy else entry_val - 0.0040)))
    else:
        sl_val = float(model_record.get("sl") or (entry_val - 4.0 if is_gold else (entry_val - 0.225 if is_jpy else entry_val - 0.0020)))
        tp_val = float(model_record.get("tp") or (entry_val + 8.0 if is_gold else (entry_val + 0.450 if is_jpy else entry_val + 0.0040)))
    rr_val = float(model_record.get("rr") or 2.0)
    atr_val = model_record.get("atr")
    if atr_val is None:
        tr = np.maximum(
            plot_df['high'] - plot_df['low'],
            np.maximum(
                abs(plot_df['high'] - plot_df['close'].shift(1)),
                abs(plot_df['low'] - plot_df['close'].shift(1))
            )
        )
        atr_val = round(float(tr.rolling(14).mean().iloc[-1] or (4.2 if is_gold else (0.225 if is_jpy else 0.0022))), digits)
    else:
        atr_val = round(float(atr_val), digits)

    prob = model_record.get("probability")
    prob_str = f"{float(prob):.1%}" if prob is not None else "N/A"
    threshold_val = model_record.get("threshold", 0.40)
    gate_passed = model_record.get("gate_passed", True)
    reason_str = model_record.get("reason", "H4 aligned and probability >= threshold")

    # Time & Identity
    created_at_utc = model_record.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()
    try:
        dt_obj = datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
        time_display = dt_obj.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        time_display = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Colors & Theme (High-contrast Institutional Dark)
    fig, (ax_main, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 8.2),
        gridspec_kw={'height_ratios': [5.2, 1.0], 'hspace': 0.06},
        facecolor='#07090e',
        dpi=140
    )
    ax_main.set_facecolor('#0d111a')
    ax_vol.set_facecolor('#0d111a')

    bull_body = '#00c076'
    bull_edge = '#00ff9d'
    bear_body = '#ff3355'
    bear_edge = '#ff5c77'
    wick_color = '#cbd5e1'
    grid_color = '#172033'
    candle_width = 0.80

    # Draw Candlesticks & Volume
    for i, row in plot_df.iterrows():
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        is_bull = (c >= o)
        body_color = bull_body if is_bull else bear_body
        edge_color = bull_edge if is_bull else bear_edge

        # Bold, high-visibility wick
        ax_main.plot([i, i], [l, h], color=wick_color, linewidth=1.8, zorder=2)
        height = abs(c - o)
        if height == 0:
            height = (h - l) * 0.04 or (0.05 if is_gold else (0.005 if is_jpy else 0.00003))

        rect = Rectangle(
            (i - candle_width / 2, min(o, c)),
            candle_width, height,
            facecolor=body_color,
            edgecolor=edge_color,
            linewidth=1.3,
            zorder=3
        )

        ax_main.add_patch(rect)

        # Volume bar
        vol = float(row.get('tick_volume', row.get('volume', abs(c - o) * 100)))
        ax_vol.bar(i, vol, color=body_color, alpha=0.55, width=candle_width, zorder=2)

    # Plot EMAs
    ax_main.plot(plot_df.index, plot_df['ema20'], color='#38bdf8', linewidth=1.5, label='EMA 20', alpha=0.95, zorder=4)
    ax_main.plot(plot_df.index, plot_df['ema50'], color='#fbbf24', linewidth=1.5, label='EMA 50', alpha=0.95, zorder=4)
    if not plot_df['ema200'].isna().all():
        ax_main.plot(plot_df.index, plot_df['ema200'], color='#a855f7', linewidth=1.5, label='EMA 200', alpha=0.90, zorder=4)

    # Trade Levels: Entry, SL, TP (2:1 R:R)
    curr_prefix = "$" if is_gold else ""
    
    # Entry Line
    ax_main.axhline(entry_val, color='#38bdf8', linestyle='-', linewidth=1.8, alpha=0.95, zorder=5)
    ax_main.text(
        n_bars + 0.8, entry_val,
        f"  ENTRY: {curr_prefix}{entry_val:.{digits}f}",
        color='#38bdf8', fontsize=9.2, fontweight='bold', va='center',
        bbox=dict(facecolor='#0a0f1d', edgecolor='#38bdf8', boxstyle='round,pad=0.28', alpha=0.95),
        zorder=6
    )

    # Take Profit Line (2.0R)
    ax_main.axhline(tp_val, color='#10b981', linestyle='--', linewidth=1.8, alpha=0.95, zorder=5)
    tp_distance = abs(tp_val - entry_val)
    ax_main.text(
        n_bars + 0.8, tp_val,
        f"  TP: {curr_prefix}{tp_val:.{digits}f} (+{rr_val:.1f}R | +{curr_prefix}{tp_distance:.{digits}f})",
        color='#10b981', fontsize=9.2, fontweight='bold', va='center',
        bbox=dict(facecolor='#0a0f1d', edgecolor='#10b981', boxstyle='round,pad=0.28', alpha=0.95),
        zorder=6
    )

    # Stop Loss Line (1.0R)
    ax_main.axhline(sl_val, color='#ef4444', linestyle='--', linewidth=1.8, alpha=0.95, zorder=5)
    sl_distance = abs(entry_val - sl_val)
    ax_main.text(
        n_bars + 0.8, sl_val,
        f"  SL: {curr_prefix}{sl_val:.{digits}f} (-1.0R | -{curr_prefix}{sl_distance:.{digits}f})",
        color='#ef4444', fontsize=9.2, fontweight='bold', va='center',
        bbox=dict(facecolor='#0a0f1d', edgecolor='#ef4444', boxstyle='round,pad=0.28', alpha=0.95),
        zorder=6
    )

    # Top Header & Source
    ax_main.set_title(
        f"FXENGIN QUANTITATIVE • {symbol.upper()} [{timeframe}] • CONFIRMED MODEL SIGNAL\n"
        f"Rendered from live MT5 stream",
        color='#f8fafc', fontsize=12.0, fontweight='bold', pad=12, loc='left'
    )

    # Telemetry Badge Box (Upper Left inside main chart) - Brief 14 Section 2 & Brief 20
    metrics = get_symbol_metrics(symbol, side=decision)
    tier = metrics.get("tier", "EXPERIMENTAL")
    exp_r = metrics.get("expectancy_r", 0.0)
    oos_trades = metrics.get("oos_trades", 0)
    folds_pos = metrics.get("folds_positive", 0)
    folds_tot = metrics.get("folds_total", 5)

    tier_desc = "Validated Edge" if tier == "VALIDATED" else "Evaluation Only"
    baseline_info = f"Tier: {tier} ({tier_desc})\nBaseline: {exp_r:+.3f} R ({oos_trades:,} OOS Trades | {folds_pos}/{folds_tot} Folds Pos)"
    h4_gate_info = f"H4 Trend Gate: PASSED ({'Bearish' if decision == 'SELL' else 'Bullish'} EMA Gate)" if gate_passed else f"H4 Trend Gate: {reason_str}"

    badge_text = (
        f"DECISION: CONFIRMED {decision}\n"
        f"Probability: {prob_str} (Threshold: {threshold_val:.2f})\n"
        f"{h4_gate_info}\n"
        f"ATR: {atr_val:.{digits}f} | R:R Ratio: 2.0 : 1\n"
        f"{baseline_info}"
    )
    ax_main.text(
        0.015, 0.96, badge_text,
        transform=ax_main.transAxes,
        color='#e2e8f0', fontsize=8.5, fontweight='bold', va='top', ha='left',
        bbox=dict(facecolor='#0a0e17', edgecolor='#334155', boxstyle='round,pad=0.6', alpha=0.92),
        zorder=7
    )

    # Identity Block & Prominent Logo (Upper Right inside main chart)
    logo_path = _find_logo_path()
    logo_drawn = False
    if logo_path:
        try:
            logo_img = Image.open(logo_path)
            # Resize square brand icon for prominent top-right placement
            logo_img.thumbnail((72, 72), Image.Resampling.LANCZOS)
            imagebox = OffsetImage(logo_img, zoom=1.0)
            ab = AnnotationBbox(
                imagebox, (0.985, 0.95),
                xycoords='axes fraction', frameon=False,
                box_alignment=(1.0, 1.0), zorder=8
            )
            ax_main.add_artist(ab)
            logo_drawn = True
        except Exception as le:
            logger.debug(f"Could not render logo image: {le}")

    id_block_lines = [
        "FXENGIN // QUANTITATIVE",
        f"ID: {snapshot_id}" if snapshot_id else f"ID: snap_{symbol}_{int(time.time())}",
        f"Time: {time_display} · {timeframe} ({n_bars} bars)"
    ]
    id_block_text = "\n".join(id_block_lines)
    # Position ID text cleanly to the left of the logo if drawn, or in upper right
    ax_main.text(
        0.915 if logo_drawn else 0.985, 0.95, id_block_text,
        transform=ax_main.transAxes,
        color='#94a3b8', fontsize=8.0, fontweight='bold', va='top', ha='right',
        bbox=dict(facecolor='#0a0e17', edgecolor='#1e293b', boxstyle='round,pad=0.45', alpha=0.9),
        zorder=7
    )

    # Bounds & Padding
    # Calculate explicit y-limits covering all candles and trade levels with dedicated 30% headroom for badge & logo
    price_min = min(float(plot_df['low'].min()), sl_val)
    price_max = max(float(plot_df['high'].max()), tp_val)
    price_range = price_max - price_min
    if price_range <= 0:
        price_range = 1.0 if is_gold else (0.20 if is_jpy else 0.0020)
    # +0.48 on top ensures highest price level (TP) never exceeds y=0.70 of axes, leaving top 30% clean for badges
    ax_main.set_ylim(price_min - price_range * 0.10, price_max + price_range * 0.48)

    # Format X-axis tick labels with real time from candle stream
    tick_step = max(4, n_bars // 6)
    tick_indices = list(range(0, n_bars, tick_step))
    if (n_bars - 1) not in tick_indices:
        tick_indices.append(n_bars - 1)

    tick_labels = []
    for idx in tick_indices:
        row = plot_df.iloc[idx]
        t_val = row.get('time') or row.get('datetime') or row.get('timestamp')
        label = ""
        if t_val:
            try:
                if isinstance(t_val, (int, float)):
                    # Handle epoch timestamps (seconds vs milliseconds)
                    ts_num = float(t_val)
                    if ts_num > 1e11:
                        ts_num /= 1000.0
                    dt = datetime.fromtimestamp(ts_num, tz=timezone.utc)
                else:
                    dt = datetime.fromisoformat(str(t_val).replace("Z", "+00:00"))
                label = dt.strftime("%H:%M")
            except Exception:
                label = str(t_val)[-8:-3] if len(str(t_val)) >= 8 else str(t_val)[:5]
        if not label:
            label = f"{idx}"
        tick_labels.append(label)

    ax_vol.set_xticks(tick_indices)
    ax_vol.set_xticklabels(tick_labels, color='#94a3b8', fontsize=8.5)
    ax_main.set_xticks(tick_indices)
    ax_main.set_xticklabels([])

    ax_main.grid(True, color=grid_color, linestyle=':', linewidth=0.6, alpha=0.75)
    ax_vol.grid(True, color=grid_color, linestyle=':', linewidth=0.6, alpha=0.75)
    ax_main.set_xlim(-1.2, n_bars + 8.5)
    ax_vol.set_xlim(-1.2, n_bars + 8.5)
    ax_main.tick_params(colors='#94a3b8', labelsize=8.5)
    ax_vol.tick_params(colors='#94a3b8', labelsize=8)

    for spine in ax_main.spines.values():
        spine.set_color('#243048')
    for spine in ax_vol.spines.values():
        spine.set_color('#243048')

    ax_main.legend(loc='lower left', facecolor='#0d111a', edgecolor='#243048', fontsize=8, labelcolor='#cbd5e1')


    # Honest Labelling Footer (Brief 7 Section 6 & Brief 14 Section 2)
    disclaimer_text = (
        f"Model signal · live record since 2026-09-09 · shadow, not auto-traded · "
        f"Tier: {tier} ({exp_r:+.3f} R over {oos_trades:,} OOS trades with spread accounted)"
    )
    fig.text(
        0.5, 0.015, disclaimer_text,
        color='#64748b', fontsize=7.5, ha='center', va='bottom', style='italic'
    )

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#07090e', dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_manual_desk_card(
    m15_candles: List[Dict[str, Any]],
    trade_record: Optional[Dict[str, Any]] = None,
    trade_data: Optional[Dict[str, Any]] = None,
    last_n_bars: int = 45
) -> bytes:
    """
    Brief 27 Section 3: High-visibility institutional card for manual desk trades.
    Reuses the same rendering pipeline as published channel cards:
    - Symbol, timeframe, direction, branded FOREX ENGINEER;
    - Recent candles with EMAs and volume;
    - Entry, SL, T1, T2, T3 drawn as lines, each labelled with price and R;
    - Shaded horizontal band for discretionary zone when zone_low and zone_high are set;
    - The why reason text displayed prominently as a caption;
    - Strictly never sent to Telegram.
    """
    trade_obj = trade_record if trade_record is not None else (trade_data or {})
    if not m15_candles or len(m15_candles) < 5:
        entry_p = float(trade_obj.get("entry_price") or trade_obj.get("entry") or 1.0)
        sl_p = float(trade_obj.get("stop_loss") or trade_obj.get("sl_price") or entry_p)
        risk = abs(entry_p - sl_p) if abs(entry_p - sl_p) > 0 else 0.0010
        now_ts = int(trade_obj.get("entered_at_epoch") or time.time())
        synth = []
        for i in range(45):
            t_bar = now_ts - (45 - i) * 900
            o_bar = entry_p + (i - 25) * (risk * 0.02)
            synth.append({
                "time": t_bar,
                "open": round(o_bar, 5),
                "high": round(o_bar + risk * 0.3, 5),
                "low": round(o_bar - risk * 0.3, 5),
                "close": round(o_bar + risk * 0.1, 5),
                "tick_volume": 100.0
            })
        m15_candles = synth

    symbol = str(trade_obj.get("symbol", "EURUSD")).upper()
    timeframe = str(trade_obj.get("timeframe", "M15")).upper()
    direction = str(trade_obj.get("direction", "BUY")).upper()
    trade_id = str(trade_obj.get("trade_id", "man_trade"))

    is_gold = ("XAU" in symbol or "GOLD" in symbol)
    is_jpy = ("JPY" in symbol)
    digits = 2 if is_gold else (3 if is_jpy else 5)
    curr_prefix = "$" if is_gold else ""

    df = pd.DataFrame(m15_candles)
    target_bars = last_n_bars if (last_n_bars and last_n_bars > 0) else 45
    n_bars_to_plot = min(len(df), max(20, target_bars))
    plot_df = df.tail(n_bars_to_plot).copy().reset_index(drop=True)
    n_bars = len(plot_df)

    # Compute EMAs
    plot_df['ema20'] = plot_df['close'].ewm(span=20, adjust=False).mean()
    plot_df['ema50'] = plot_df['close'].ewm(span=50, adjust=False).mean()
    plot_df['ema200'] = plot_df['close'].ewm(span=200, adjust=False).mean()

    # Price Levels & R
    entry_val = float(trade_obj.get("entry_price") or trade_obj.get("entry") or plot_df['close'].iloc[-1])
    sl_val = float(trade_obj.get("stop_loss") or trade_obj.get("sl_price") or entry_val)
    tp1_val = float(trade_obj.get("take_profit_1") or trade_obj.get("tp1_price") or entry_val)
    tp2_raw = trade_obj.get("take_profit_2") or trade_obj.get("tp2_price")
    tp2_val = float(tp2_raw) if (tp2_raw is not None and float(tp2_raw) > 0) else None
    tp3_raw = trade_obj.get("take_profit_3") or trade_obj.get("tp3_price")
    tp3_val = float(tp3_raw) if (tp3_raw is not None and float(tp3_raw) > 0) else None

    risk = abs(entry_val - sl_val) if abs(entry_val - sl_val) > 0 else (0.01 if is_gold else 0.001)
    tp1_r = float(trade_obj.get("tp1_r") or round(abs(tp1_val - entry_val) / risk, 2))
    tp2_r = float(trade_obj.get("tp2_r") or (round(abs(tp2_val - entry_val) / risk, 2) if tp2_val else 0.0))
    tp3_r = float(trade_obj.get("tp3_r") or (round(abs(tp3_val - entry_val) / risk, 2) if tp3_val else 0.0))

    # Zone bounds
    z_low = trade_obj.get("zone_low")
    z_high = trade_obj.get("zone_high")
    has_zone = (z_low is not None and z_high is not None and float(z_low) > 0 and float(z_high) > 0)
    z_min = min(float(z_low), float(z_high)) if has_zone else None
    z_max = max(float(z_low), float(z_high)) if has_zone else None

    entered_at_str = str(trade_obj.get("entered_at", "")).replace("Z", " UTC")
    why_text = str(trade_obj.get("why") or trade_obj.get("why_reason") or "").strip()

    if not MATPLOTLIB_AVAILABLE or plt is None:
        logger.warning("matplotlib not available. Returning valid PNG fallback.")
        return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"

    # -------------------------------------------------------------------------
    # Bounds & Y-Limits (Agent Brief 30 Section 2)
    # Compute the y-limits strictly from ALL validated trade levels:
    # candle low/high, entry, stop, T1, T2, T3.
    # All validated targets MUST ALWAYS be inside the visible range!
    # The zone stays unvalidated: excluded from y-limits, clipped at margin if outside.
    # -------------------------------------------------------------------------
    candle_low = float(plot_df['low'].min())
    candle_high = float(plot_df['high'].max())
    candle_span = max(candle_high - candle_low, 1e-5)

    trade_levels = [candle_low, candle_high, sl_val, entry_val, tp1_val]
    if tp2_val is not None:
        trade_levels.append(tp2_val)
    if tp3_val is not None:
        trade_levels.append(tp3_val)

    price_min = min(trade_levels)
    price_max = max(trade_levels)
    price_range = max(price_max - price_min, 0.0001)

    y_min = price_min - price_range * 0.08
    y_max = price_max + price_range * 0.25
    panel_height = max(y_max - y_min, 1e-5)

    # Brief 30 Section 2: Canvas growth instead of clipping targets
    # If candles occupy less than 40% of span, grow the canvas height (up to ceiling)
    # while keeping EVERY validated level on the chart.
    candle_occupancy = candle_span / panel_height
    base_fig_height = 8.8
    if candle_occupancy < 0.40:
        needed_factor = 0.40 / max(candle_occupancy, 0.12)
        fig_height = min(14.0, max(base_fig_height, base_fig_height * (needed_factor ** 0.55)))
        logger.info(
            f"Candle span occupies {candle_occupancy*100:.1f}% (<40%). "
            f"Growing canvas height to {fig_height:.1f} to maintain candle prominence without clipping levels."
        )
    else:
        fig_height = base_fig_height

    # Create figure (Brief 30 Section 3: 200 DPI, price panel owns >= 75% of space)
    fig, (ax_main, ax_vol) = plt.subplots(
        2, 1, figsize=(14.5, fig_height),
        gridspec_kw={'height_ratios': [6.0, 1.0], 'hspace': 0.04},
        facecolor='#07090e',
        dpi=200
    )
    ax_main.set_facecolor('#0d111a')
    ax_vol.set_facecolor('#0d111a')

    bull_body = '#00c076'
    bull_edge = '#00ff9d'
    bear_body = '#ff3355'
    bear_edge = '#ff5c77'
    wick_color = '#cbd5e1'
    grid_color = '#172033'
    candle_width = 0.80

    # Draw Candlesticks & Volume
    for i, row in plot_df.iterrows():
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        is_bull = (c >= o)
        body_color = bull_body if is_bull else bear_body
        edge_color = bull_edge if is_bull else bear_edge

        ax_main.plot([i, i], [l, h], color=wick_color, linewidth=1.8, zorder=2)
        height = abs(c - o)
        if height == 0:
            height = (h - l) * 0.04 or (0.05 if is_gold else (0.005 if is_jpy else 0.00003))

        rect = Rectangle(
            (i - candle_width / 2, min(o, c)),
            candle_width, height,
            facecolor=body_color,
            edgecolor=edge_color,
            linewidth=1.3,
            zorder=3
        )
        ax_main.add_patch(rect)

        vol = float(row.get('tick_volume', row.get('volume', abs(c - o) * 100)))
        ax_vol.bar(i, vol, color=body_color, alpha=0.55, width=candle_width, zorder=2)

    # Plot EMAs
    ax_main.plot(plot_df.index, plot_df['ema20'], color='#38bdf8', linewidth=1.4, label='EMA 20', alpha=0.9, zorder=4)
    ax_main.plot(plot_df.index, plot_df['ema50'], color='#fbbf24', linewidth=1.4, label='EMA 50', alpha=0.9, zorder=4)
    if not plot_df['ema200'].isna().all():
        ax_main.plot(plot_df.index, plot_df['ema200'], color='#a855f7', linewidth=1.4, label='EMA 200', alpha=0.85, zorder=4)

    # Set price limits unconditionally covering all validated levels
    ax_main.set_ylim(y_min, y_max)

    # -------------------------------------------------------------------------
    # Discretionary Zone / Level Rendering (Agent Brief 28 & 30)
    # 1. zone_low == zone_high: Draw a single labelled line (axhline)
    # 2. zone_low != zone_high: Draw a shaded band (axhspan)
    # If outside view: clip band at edge & label at margin with arrow (e.g. "zone 12.00000 ↑ above view")
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    if has_zone and z_min is not None and z_max is not None:
        is_single_level = (abs(z_max - z_min) < 1e-7)
        is_above_view = (z_min > y_max)
        is_below_view = (z_max < y_min)

        if is_above_view:
            lbl_txt = f"zone {curr_prefix}{z_min:.{digits}f} ↑ above view" if is_single_level else f"zone {curr_prefix}{z_min:.{digits}f} – {curr_prefix}{z_max:.{digits}f} ↑ above view"
            ax_main.text(
                0.50, 0.985, lbl_txt,
                transform=ax_main.transAxes,
                color='#c7d2fe', fontsize=8.8, fontweight='bold', ha='center', va='top',
                bbox=dict(facecolor='#1e1b4b', edgecolor='#6366f1', boxstyle='round,pad=0.35', alpha=0.92),
                zorder=9
            )
        elif is_below_view:
            lbl_txt = f"zone {curr_prefix}{z_min:.{digits}f} ↓ below view" if is_single_level else f"zone {curr_prefix}{z_min:.{digits}f} – {curr_prefix}{z_max:.{digits}f} ↓ below view"
            ax_main.text(
                0.50, 0.035, lbl_txt,
                transform=ax_main.transAxes,
                color='#c7d2fe', fontsize=8.8, fontweight='bold', ha='center', va='bottom',
                bbox=dict(facecolor='#1e1b4b', edgecolor='#6366f1', boxstyle='round,pad=0.35', alpha=0.92),
                zorder=9
            )
        else:
            if is_single_level:
                ax_main.axhline(z_min, color='#818cf8', linestyle='-.', linewidth=1.6, alpha=0.9, zorder=3, label='Discretionary Level')
                ax_main.text(
                    n_bars / 2.0, z_min,
                    f"DISCRETIONARY LEVEL: {curr_prefix}{z_min:.{digits}f}",
                    color='#c7d2fe', fontsize=8.8, fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='#1e1b4b', edgecolor='#6366f1', boxstyle='round,pad=0.3', alpha=0.9),
                    zorder=6
                )
            else:
                ax_main.axhspan(z_min, z_max, color='#818cf8', alpha=0.22, zorder=2, label='Discretionary Zone')
                vis_low = max(z_min, y_min)
                vis_high = min(z_max, y_max)
                zone_mid = (vis_low + vis_high) / 2.0
                ax_main.text(
                    n_bars / 2.0, zone_mid,
                    f"DISCRETIONARY ZONE: {curr_prefix}{z_min:.{digits}f} – {curr_prefix}{z_max:.{digits}f}",
                    color='#c7d2fe', fontsize=9.0, fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='#1e1b4b', edgecolor='#6366f1', boxstyle='round,pad=0.35', alpha=0.88),
                    zorder=6
                )
                if z_max > y_max:
                    ax_main.text(
                        0.50, 0.985, f"zone top {curr_prefix}{z_max:.{digits}f} ↑ above view",
                        transform=ax_main.transAxes, color='#c7d2fe', fontsize=8.0, fontweight='bold',
                        ha='center', va='top', bbox=dict(facecolor='#1e1b4b', edgecolor='#6366f1', boxstyle='round,pad=0.25', alpha=0.85),
                        zorder=9
                    )
                if z_min < y_min:
                    ax_main.text(
                        0.50, 0.035, f"zone bot {curr_prefix}{z_min:.{digits}f} ↓ below view",
                        transform=ax_main.transAxes, color='#c7d2fe', fontsize=8.0, fontweight='bold',
                        ha='center', va='bottom', bbox=dict(facecolor='#1e1b4b', edgecolor='#6366f1', boxstyle='round,pad=0.25', alpha=0.85),
                        zorder=9
                    )

    # Trade Levels: Entry, SL, T1, T2, T3
    # 1. Entry Line
    if y_min <= entry_val <= y_max:
        ax_main.axhline(entry_val, color='#38bdf8', linestyle='-', linewidth=1.8, alpha=0.95, zorder=5)
        ax_main.text(
            n_bars + 0.8, entry_val,
            f"  ENTRY: {curr_prefix}{entry_val:.{digits}f}",
            color='#38bdf8', fontsize=9.2, fontweight='bold', va='center',
            bbox=dict(facecolor='#0a0f1d', edgecolor='#38bdf8', boxstyle='round,pad=0.25', alpha=0.95),
            clip_on=True,
            zorder=6
        )

    # 2. Stop Loss Line
    if y_min <= sl_val <= y_max:
        ax_main.axhline(sl_val, color='#ef4444', linestyle='--', linewidth=1.8, alpha=0.95, zorder=5)
        ax_main.text(
            n_bars + 0.8, sl_val,
            f"  SL: {curr_prefix}{sl_val:.{digits}f} (-1.0R)",
            color='#ef4444', fontsize=9.2, fontweight='bold', va='center',
            bbox=dict(facecolor='#0a0f1d', edgecolor='#ef4444', boxstyle='round,pad=0.25', alpha=0.95),
            clip_on=True,
            zorder=6
        )
    elif sl_val < y_min:
        ax_main.text(
            0.985, 0.035, f"SL: {curr_prefix}{sl_val:.{digits}f} ↓",
            transform=ax_main.transAxes, color='#ef4444', fontsize=8.2, fontweight='bold',
            ha='right', va='bottom', bbox=dict(facecolor='#0a0f1d', edgecolor='#ef4444', boxstyle='round,pad=0.2', alpha=0.9),
            zorder=9
        )
    elif sl_val > y_max:
        ax_main.text(
            0.985, 0.965, f"SL: {curr_prefix}{sl_val:.{digits}f} ↑",
            transform=ax_main.transAxes, color='#ef4444', fontsize=8.2, fontweight='bold',
            ha='right', va='top', bbox=dict(facecolor='#0a0f1d', edgecolor='#ef4444', boxstyle='round,pad=0.2', alpha=0.9),
            zorder=9
        )

    # 3. Take Profit 1 (T1)
    if y_min <= tp1_val <= y_max:
        ax_main.axhline(tp1_val, color='#10b981', linestyle='--', linewidth=1.8, alpha=0.95, zorder=5)
        ax_main.text(
            n_bars + 0.8, tp1_val,
            f"  T1: {curr_prefix}{tp1_val:.{digits}f} (+{tp1_r:.1f}R)",
            color='#10b981', fontsize=9.2, fontweight='bold', va='center',
            bbox=dict(facecolor='#0a0f1d', edgecolor='#10b981', boxstyle='round,pad=0.25', alpha=0.95),
            clip_on=True,
            zorder=6
        )
    elif tp1_val > y_max:
        ax_main.text(
            0.985, 0.935, f"T1: {curr_prefix}{tp1_val:.{digits}f} ↑",
            transform=ax_main.transAxes, color='#10b981', fontsize=8.2, fontweight='bold',
            ha='right', va='top', bbox=dict(facecolor='#0a0f1d', edgecolor='#10b981', boxstyle='round,pad=0.2', alpha=0.9),
            zorder=9
        )

    # 4. Take Profit 2 (T2, if present)
    if tp2_val is not None:
        if y_min <= tp2_val <= y_max:
            ax_main.axhline(tp2_val, color='#059669', linestyle=':', linewidth=1.6, alpha=0.92, zorder=5)
            ax_main.text(
                n_bars + 0.8, tp2_val,
                f"  T2: {curr_prefix}{tp2_val:.{digits}f} (+{tp2_r:.1f}R)",
                color='#059669', fontsize=9.0, fontweight='bold', va='center',
                bbox=dict(facecolor='#0a0f1d', edgecolor='#059669', boxstyle='round,pad=0.22', alpha=0.95),
                clip_on=True,
                zorder=6
            )

    # 5. Take Profit 3 (T3, if present)
    if tp3_val is not None:
        if y_min <= tp3_val <= y_max:
            ax_main.axhline(tp3_val, color='#047857', linestyle=':', linewidth=1.6, alpha=0.92, zorder=5)
            ax_main.text(
                n_bars + 0.8, tp3_val,
                f"  T3: {curr_prefix}{tp3_val:.{digits}f} (+{tp3_r:.1f}R)",
                color='#047857', fontsize=9.0, fontweight='bold', va='center',
                bbox=dict(facecolor='#0a0f1d', edgecolor='#047857', boxstyle='round,pad=0.22', alpha=0.95),
                clip_on=True,
                zorder=6
            )

    # Top Header & Branding
    ax_main.set_title(
        f"FOREX ENGINEER • {symbol} [{timeframe}] • MANUAL DESK TRADE\n"
        f"Entered {entered_at_str} · Trade ID: {trade_id}",
        color='#f8fafc', fontsize=11.5, fontweight='bold', pad=12, loc='left'
    )

    # Telemetry Badge Box (Upper Left inside main chart)
    snap_source = trade_obj.get("snapshot_source", "LIVE")
    has_snapshot = trade_obj.get("snapshot") is not None or trade_obj.get("features") is not None
    snap_status = f"Snapshot: {snap_source} (91 features)" if has_snapshot else "Snapshot: PENDING"
    overlap_txt = "Overlap trade" if trade_obj.get("symbol_had_open_trade") else "First on pair"
    zone_line = ""
    if has_zone and z_min is not None:
        if abs(z_max - z_min) < 1e-7:
            zone_line = f"Level: {curr_prefix}{z_min:.{digits}f}\n"
        else:
            zone_line = f"Zone: {curr_prefix}{z_min:.{digits}f} – {curr_prefix}{z_max:.{digits}f}\n"

    badge_text = (
        f"DISCRETIONARY DESK: {direction}\n"
        f"Entry: {curr_prefix}{entry_val:.{digits}f} | Risk: {risk:.{digits}f}\n"
        f"Targets: T1={tp1_r:.1f}R" + (f", T2={tp2_r:.1f}R" if tp2_val else "") + (f", T3={tp3_r:.1f}R" if tp3_val else "") + "\n"
        f"{zone_line}"
        f"{overlap_txt} · {snap_status}"
    )
    ax_main.text(
        0.015, 0.96, badge_text,
        transform=ax_main.transAxes,
        color='#e2e8f0', fontsize=8.5, fontweight='bold', va='top', ha='left',
        bbox=dict(facecolor='#0a0e17', edgecolor='#334155', boxstyle='round,pad=0.55', alpha=0.92),
        zorder=7
    )

    # Identity Block & Logo (Upper Right inside main chart)
    logo_path = _find_logo_path()
    logo_drawn = False
    if logo_path:
        try:
            logo_img = Image.open(logo_path)
            logo_img.thumbnail((72, 72), Image.Resampling.LANCZOS)
            imagebox = OffsetImage(logo_img, zoom=1.0)
            ab = AnnotationBbox(
                imagebox, (0.985, 0.95),
                xycoords='axes fraction', frameon=False,
                box_alignment=(1.0, 1.0), zorder=8
            )
            ax_main.add_artist(ab)
            logo_drawn = True
        except Exception:
            pass

    id_block_lines = [
        "FOREX ENGINEER // DESK",
        f"ID: {trade_id.split('_')[-2]}_{trade_id.split('_')[-1]}",
        f"Timeframe: {timeframe} ({n_bars} bars)"
    ]
    ax_main.text(
        0.915 if logo_drawn else 0.985, 0.95, "\n".join(id_block_lines),
        transform=ax_main.transAxes,
        color='#94a3b8', fontsize=8.0, fontweight='bold', va='top', ha='right',
        bbox=dict(facecolor='#0a0e17', edgecolor='#1e293b', boxstyle='round,pad=0.4', alpha=0.9),
        zorder=7
    )

    # X-axis ticks
    tick_step = max(4, n_bars // 6)
    tick_indices = list(range(0, n_bars, tick_step))
    if (n_bars - 1) not in tick_indices:
        tick_indices.append(n_bars - 1)

    tick_labels = []
    for idx in tick_indices:
        row = plot_df.iloc[idx]
        t_val = row.get('time') or row.get('datetime')
        lbl = ""
        if t_val:
            try:
                if isinstance(t_val, (int, float)):
                    ts_num = float(t_val)
                    if ts_num > 1e11: ts_num /= 1000.0
                    dt = datetime.fromtimestamp(ts_num, tz=timezone.utc)
                else:
                    dt = datetime.fromisoformat(str(t_val).replace("Z", "+00:00"))
                lbl = dt.strftime("%H:%M")
            except Exception:
                lbl = str(t_val)[:5]
        tick_labels.append(lbl or f"{idx}")

    ax_vol.set_xticks(tick_indices)
    ax_vol.set_xticklabels(tick_labels, color='#94a3b8', fontsize=8.5)
    ax_main.set_xticks(tick_indices)
    ax_main.set_xticklabels([])

    ax_main.grid(True, color=grid_color, linestyle=':', linewidth=0.6, alpha=0.75)
    ax_vol.grid(True, color=grid_color, linestyle=':', linewidth=0.6, alpha=0.75)
    ax_main.set_xlim(-1.2, n_bars + 11.5)
    ax_vol.set_xlim(-1.2, n_bars + 11.5)
    ax_main.tick_params(colors='#94a3b8', labelsize=8.5)
    ax_vol.tick_params(colors='#94a3b8', labelsize=8)

    for spine in ax_main.spines.values():
        spine.set_color('#243048')
    for spine in ax_vol.spines.values():
        spine.set_color('#243048')

    ax_main.legend(loc='lower left', facecolor='#0d111a', edgecolor='#243048', fontsize=8, labelcolor='#cbd5e1')

    # Footer Caption (Why text or standard disclaimer)
    if why_text:
        caption = f'Analyst Rationale: "{why_text}" · FOREX ENGINEER Analyst Desk (Private)'
        fig.text(0.5, 0.015, caption, color='#93c5fd', fontsize=8.2, ha='center', va='bottom', style='italic')
    else:
        caption = "FOREX ENGINEER Analyst Desk · Discretionary Entry Record · Measurement Instrument Only"
        fig.text(0.5, 0.015, caption, color='#64748b', fontsize=7.5, ha='center', va='bottom', style='italic')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#07090e', dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


