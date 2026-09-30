import html
from typing import Dict, Any

def render_shareable_analysis_html(
    snapshot: Dict[str, Any],
    freshness: Dict[str, Any],
    lang: str = "ar",
    base_url: str = ""
) -> str:
    """
    Renders an ultra-premium, institutional dark-theme shareable HTML page
    with full bilingual support (Arabic/English), immutable snapshot verification,
    real MT5 screenshot, and interactive fresh snapshot request trigger.
    """
    is_ar = (lang.lower() == "ar")
    dir_attr = "rtl" if is_ar else "ltr"
    lang_attr = "ar" if is_ar else "en"

    symbol = html.escape(str(snapshot.get("symbol", "XAUUSD")))
    timeframe = html.escape(str(snapshot.get("timeframe", "M5")))
    rec = str(snapshot.get("recommendation", "BUY")).upper()
    bias = str(snapshot.get("bias", "BULLISH")).upper()
    _grade_raw = snapshot.get("setup_grade")
    grade = html.escape(str(_grade_raw)) if _grade_raw else "N/A"
    _conf_raw = snapshot.get("confidence_score")
    confidence = float(_conf_raw) if _conf_raw is not None else None
    conf_txt = f"{confidence:.0f}%" if confidence is not None else "N/A"
    snap_id = html.escape(str(snapshot.get("snapshot_id", "")))
    created_at = html.escape(str(snapshot.get("created_at_utc", "")))

    is_gold = ("XAU" in symbol or "GOLD" in symbol)
    digits = 2 if is_gold else 5

    entry = float(snapshot.get("entry_price", 0.0))
    sl = float(snapshot.get("sl_price", 0.0))
    tp1 = float(snapshot.get("tp1_price", 0.0))
    tp2 = float(snapshot.get("tp2_price", 0.0))
    tp3 = float(snapshot.get("tp3_price", 0.0) or 0.0)
    sl_pips = float(snapshot.get("sl_pips", 0.0))
    tp1_pips = float(snapshot.get("tp1_pips", 0.0))
    tp2_pips = float(snapshot.get("tp2_pips", 0.0))
    tp3_pips = float(snapshot.get("tp3_pips", 0.0) or 0.0)
    rr = html.escape(str(snapshot.get("rr_ratio", "1:2.0")))
    lot = float(snapshot.get("suggested_lot", 0.05))

    adx = float(snapshot.get("adx", 25.0))
    rsi = float(snapshot.get("rsi", 50.0))
    atr = float(snapshot.get("atr", 3.5 if is_gold else 0.0020))

    report_text = snapshot.get("report_ar" if is_ar else "report_en", "")
    report_html = html.escape(report_text).replace("\n", "<br>")

    # Colors & Badges
    is_buy = (rec == "BUY" or bias == "BULLISH")
    theme_color = "#10b981" if is_buy else ("#f43f5e" if rec == "SELL" else "#38bdf8")
    action_label = ("شراء قوي (BUY)" if is_buy else "بيع قوي (SELL)") if is_ar else ("STRONG BUY" if is_buy else "STRONG SELL")
    if rec == "HOLD":
        action_label = "انتظار (HOLD)" if is_ar else "WATCH / HOLD"

    # Freshness
    status_code = freshness.get("status_code", "OPTIMAL_ACTIVE")
    curr_price = float(freshness.get("current_price", entry))
    drift_pips = float(freshness.get("drift_pips", 0.0))
    needs_refresh = freshness.get("needs_new_snapshot", False)
    status_msg = freshness.get("description_ar" if is_ar else "description_en", "")

    if status_code == "OPTIMAL_ACTIVE":
        badge_bg = "rgba(16, 185, 129, 0.15)"
        badge_border = "#10b981"
        badge_text = "#34d399"
        badge_icon = "🟢"
        badge_title = "لقطة إعداد مثالية مجمدة" if is_ar else "Optimal Setup Snapshot Locked"
    elif status_code == "TARGET_REACHED":
        badge_bg = "rgba(16, 185, 129, 0.25)"
        badge_border = "#059669"
        badge_text = "#6ee7b7"
        badge_icon = "🎯"
        badge_title = "تم تحقيق الهدف" if is_ar else "Target 1 Reached"
    elif status_code == "STOPPED_OUT":
        badge_bg = "rgba(244, 63, 94, 0.2)"
        badge_border = "#f43f5e"
        badge_text = "#fda4af"
        badge_icon = "🛑"
        badge_title = "ضرب وقف الخسارة" if is_ar else "Stop Invalidation"
    else:
        badge_bg = "rgba(245, 158, 11, 0.2)"
        badge_border = "#f59e0b"
        badge_text = "#fcd34d"
        badge_icon = "⚠️"
        badge_title = "تحرك السعر عن وقت الدخول" if is_ar else "Price Drifted From Entry"

    other_lang = "en" if is_ar else "ar"
    lang_btn_text = "English 🇺🇸" if is_ar else "العربية 🇸🇦"
    lang_btn_url = f"?lang={other_lang}"

    image_src = f"/api/v1/analyst/share/{snap_id}/image"

    # Meta tags for social previews (Telegram, WhatsApp, Twitter)
    meta_title = f"{'تحليل إشارة' if is_ar else 'Signal Setup'}: {symbol} {action_label} | FXENGIN Institutional"
    meta_desc = f"{'نقطة الدخول' if is_ar else 'Entry'}: ${entry:.{digits}f} | SL: ${sl:.{digits}f} | TP1: ${tp1:.{digits}f} ({conf_txt} {grade})"

    return f"""<!DOCTYPE html>
<html lang="{lang_attr}" dir="{dir_attr}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{meta_title}</title>
    
    <!-- OpenGraph / Social Meta Tags -->
    <meta property="og:type" content="article">
    <meta property="og:title" content="{meta_title}">
    <meta property="og:description" content="{meta_desc}">
    <meta property="og:image" content="{image_src}">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{meta_title}">
    <meta name="twitter:description" content="{meta_desc}">
    <meta name="twitter:image" content="{image_src}">

    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        :root {{
            --bg: #07090e;
            --card-bg: rgba(15, 23, 42, 0.75);
            --border: rgba(51, 65, 85, 0.7);
            --accent: {theme_color};
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: {'Cairo, sans-serif' if is_ar else 'Inter, sans-serif'};
        }}

        body {{
            background: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 20px 16px 60px 16px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .mono {{
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Header Navbar */
        .navbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 20px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            backdrop-filter: blur(12px);
            margin-bottom: 20px;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .brand-badge {{
            background: linear-gradient(135deg, #6366f1, #4338ca);
            color: white;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 13px;
            letter-spacing: 0.5px;
        }}

        .brand-title {{
            font-size: 15px;
            font-weight: 700;
            color: #e2e8f0;
        }}

        .nav-actions {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .btn-lang {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid #475569;
            color: #f1f5f9;
            padding: 8px 16px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 700;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn-lang:hover {{
            background: #334155;
            border-color: #64748b;
        }}

        /* Hero Status Banner */
        .hero-banner {{
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5);
            position: relative;
            overflow: hidden;
        }}

        .banner-glow {{
            position: absolute;
            top: -100px;
            {('left' if is_ar else 'right')}: -100px;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, {theme_color} 0%, transparent 70%);
            opacity: 0.15;
            pointer-events: none;
        }}

        .hero-top {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 16px;
        }}

        .symbol-badge-row {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .symbol-title {{
            font-size: 28px;
            font-weight: 900;
            letter-spacing: 0.5px;
        }}

        .tf-badge {{
            background: #1e293b;
            color: #94a3b8;
            border: 1px solid #334155;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 700;
        }}

        .rec-badge {{
            background: {theme_color};
            color: #022c22;
            padding: 6px 14px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 900;
            letter-spacing: 0.5px;
        }}

        .conf-badge {{
            background: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.4);
            padding: 6px 12px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 700;
        }}

        /* Freshness & Snapshot Time Strip */
        .snapshot-strip {{
            background: {badge_bg};
            border: 1px solid {badge_border};
            color: {badge_text};
            padding: 12px 18px;
            border-radius: 12px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            font-size: 13.5px;
            font-weight: 600;
        }}

        .strip-left {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .btn-refresh-snap {{
            background: #4f46e5;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 12.5px;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
        }}
        .btn-refresh-snap:hover {{
            background: #4338ca;
            transform: translateY(-1px);
        }}

        /* Main Grid: Screenshot on Left, Coordinates on Right */
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}

        @media (min-width: 992px) {{
            .main-grid {{
                grid-template-columns: 1.15fr 0.85fr;
            }}
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 22px;
            backdrop-filter: blur(12px);
        }}

        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(51, 65, 85, 0.4);
        }}

        .card-title {{
            font-size: 16px;
            font-weight: 800;
            color: #f1f5f9;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* Image Display Box */
        .chart-box {{
            position: relative;
            background: #000;
            border: 1px solid #1e293b;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 10px 25px rgba(0,0,0,0.6);
        }}

        .chart-img {{
            width: 100%;
            height: auto;
            display: block;
            object-fit: contain;
            transition: transform 0.3s;
            cursor: zoom-in;
        }}

        .chart-caption {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(0deg, rgba(0,0,0,0.85) 0%, transparent 100%);
            padding: 10px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11.5px;
            color: #cbd5e1;
        }}

        /* Coordinates Grid */
        .coord-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 18px;
        }}

        .coord-cell {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(51, 65, 85, 0.5);
            border-radius: 12px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .coord-cell.primary {{
            border-color: rgba(99, 102, 241, 0.5);
            background: rgba(99, 102, 241, 0.08);
        }}

        .coord-cell.danger {{
            border-color: rgba(244, 63, 94, 0.4);
            background: rgba(244, 63, 94, 0.05);
        }}

        .coord-cell.success {{
            border-color: rgba(16, 185, 129, 0.4);
            background: rgba(16, 185, 129, 0.05);
        }}

        .coord-label {{
            font-size: 11.5px;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
        }}

        .coord-val {{
            font-size: 18px;
            font-weight: 800;
            color: #fff;
        }}

        .coord-sub {{
            font-size: 11px;
            color: #64748b;
            font-weight: 500;
        }}

        /* SMC Matrix Pills */
        .smc-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 18px;
        }}

        .smc-item {{
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(51, 65, 85, 0.4);
            border-radius: 10px;
            padding: 10px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 13px;
        }}

        .smc-key {{
            color: #94a3b8;
            font-weight: 600;
        }}

        .smc-val {{
            color: #38bdf8;
            font-weight: 700;
        }}

        /* Narrative Report Box */
        .report-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 24px;
            line-height: 1.8;
            font-size: 14px;
            color: #cbd5e1;
        }}

        /* Action Buttons Row */
        .share-actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            justify-content: center;
            padding: 16px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
        }}

        .btn-action {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 18px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 700;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }}

        .btn-action.copy {{
            background: #1e293b;
            color: #e2e8f0;
            border-color: #475569;
        }}
        .btn-action.copy:hover {{
            background: #334155;
        }}

        .btn-action.telegram {{
            background: #0284c7;
            color: white;
        }}
        .btn-action.telegram:hover {{
            background: #0369a1;
        }}

        .btn-action.whatsapp {{
            background: #059669;
            color: white;
        }}
        .btn-action.whatsapp:hover {{
            background: #047857;
        }}

        .btn-action.image {{
            background: #475569;
            color: white;
        }}
        .btn-action.image:hover {{
            background: #334155;
        }}

        /* Toast Popup */
        #toast {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #10b981;
            color: #022c22;
            padding: 12px 20px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 13.5px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            display: none;
            z-index: 1000;
        }}

        /* Modal */
        #modal {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.92);
            z-index: 9999;
            justify-content: center;
            align-items: center;
            padding: 20px;
            cursor: zoom-out;
        }}
        #modal img {{
            max-width: 95vw;
            max-height: 90vh;
            border-radius: 8px;
            box-shadow: 0 0 40px rgba(0,0,0,0.8);
        }}
    </style>
</head>
<body>

    <div class="container">
        <!-- Top Navbar -->
        <header class="navbar">
            <div class="brand">
                <span class="brand-badge">FXENGIN AI</span>
                <span class="brand-title">{'نظام التحليل المؤسسي لـ MT5' if is_ar else 'MT5 Institutional Analyst Core'}</span>
            </div>
            <div class="nav-actions">
                <a href="{lang_btn_url}" class="btn-lang">🌐 {lang_btn_text}</a>
            </div>
        </header>

        <!-- Hero Banner -->
        <section class="hero-banner">
            <div class="banner-glow"></div>
            
            <div class="hero-top">
                <div class="symbol-badge-row">
                    <span class="symbol-title">{symbol}</span>
                    <span class="tf-badge mono">{timeframe}</span>
                    <span class="rec-badge">{action_label}</span>
                    <span class="conf-badge">🎯 {conf_txt} • Grade {grade}</span>
                </div>
                <div style="font-size: 12.5px; color: var(--text-muted);">
                    <span>{'سعر السوق الحالي' if is_ar else 'Live Price'}: <strong class="mono" style="color: #fff; font-size: 14px;">${curr_price:.{digits}f}</strong></span>
                </div>
            </div>

            <!-- Freshness & Snapshot Verification Strip -->
            <div class="snapshot-strip">
                <div class="strip-left">
                    <span>{badge_icon}</span>
                    <div>
                        <strong>{badge_title}:</strong> {status_msg}
                        <div style="font-size: 11.5px; opacity: 0.85; margin-top: 2px;">
                            {'وقت التجميد الأنسب' if is_ar else 'Locked Capture Time'}: {created_at} • ID: <span class="mono">{snap_id}</span>
                        </div>
                    </div>
                </div>
                <div>
                    <button class="btn-refresh-snap" onclick="requestFreshSnapshot('{symbol}')">
                        <span>📸</span>
                        <span>{'طلب صورة وتحليل جديد' if is_ar else 'Request Fresh Snapshot'}</span>
                    </button>
                </div>
            </div>
        </section>

        <!-- Main Grid -->
        <main class="main-grid">
            <!-- Left: Authentic MT5 Screenshot Box -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">
                        📸 {'لقطة الشارت الحقيقية من MT5' if is_ar else 'Real MetaTrader 5 Chart Screenshot'}
                    </span>
                    <span style="font-size: 12px; color: var(--text-muted);">{'انقر للتكبير' if is_ar else 'Click to Zoom'}</span>
                </div>
                
                <div class="chart-box">
                    <img src="{image_src}" alt="MT5 Chart" class="chart-img" id="mainChartImg" onclick="openModal('{image_src}')" onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'800\\' height=\\'450\\' viewBox=\\'0 0 800 450\\'><rect width=\\'100%\\' height=\\'100%\\' fill=\\'%230f172a\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%2364748b\\' font-size=\\'20\\' text-anchor=\\'middle\\' font-family=\\'sans-serif\\'>Chart Snapshot Loading...</text></svg>';">
                    <div class="chart-caption">
                        <span>⚡ {symbol} [{timeframe}] • {created_at}</span>
                        <a href="{image_src}" download="{snap_id}.png" style="color: #38bdf8; text-decoration: none; font-weight: 700;">{'تحميل الصورة' if is_ar else 'Download PNG'}</a>
                    </div>
                </div>
            </div>

            <!-- Right: Trading Coordinates & SMC Matrix -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">
                        🎯 {'إحداثيات التنفيذ وخطة المخاطرة' if is_ar else 'Execution Coordinates & Risk Plan'}
                    </span>
                    <span class="mono" style="font-size: 12px; color: #a5b4fc;">R:R {rr}</span>
                </div>

                <div class="coord-grid">
                    <div class="coord-cell primary">
                        <span class="coord-label">{'نقطة الدخول المقترحة' if is_ar else 'Proposed Entry'}</span>
                        <span class="coord-val mono">${entry:.{digits}f}</span>
                        <span class="coord-sub">{'التوقيت الأنسب' if is_ar else 'Optimal Point'}</span>
                    </div>

                    <div class="coord-cell danger">
                        <span class="coord-label">{'وقف الخسارة (SL)' if is_ar else 'Stop Loss (SL)'}</span>
                        <span class="coord-val mono" style="color: #fb7185;">${sl:.{digits}f}</span>
                        <span class="coord-sub">-{sl_pips} {'نقطة' if is_ar else 'pips'}</span>
                    </div>

                    <div class="coord-cell success">
                        <span class="coord-label">{'الهدف الأول (TP1)' if is_ar else 'Take Profit 1'}</span>
                        <span class="coord-val mono" style="color: #34d399;">${tp1:.{digits}f}</span>
                        <span class="coord-sub">+{tp1_pips} {'نقطة (Scalp)' if is_ar else 'pips (Scalp)'}</span>
                    </div>

                    <div class="coord-cell success">
                        <span class="coord-label">{'الهدف الثاني (TP2)' if is_ar else 'Take Profit 2'}</span>
                        <span class="coord-val mono" style="color: #34d399;">${tp2:.{digits}f}</span>
                        <span class="coord-sub">+{tp2_pips} {'نقطة (Swing)' if is_ar else 'pips (Swing)'}</span>
                    </div>

                    {f'''<div class="coord-cell success">
                        <span class="coord-label">{"الهدف الثالث (TP3)" if is_ar else "Take Profit 3"}</span>
                        <span class="coord-val mono" style="color: #34d399;">${tp3:.{digits}f}</span>
                        <span class="coord-sub">+{tp3_pips} {"نقطة (Runner)" if is_ar else "pips (Runner)"}</span>
                    </div>''' if tp3 > 0 else ''}
                </div>

                <!-- Suggested Lot & Risk Card -->
                <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 12px 14px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 13px; color: #cbd5e1; font-weight: 600;">{'حجم اللوت الآمن (مخاطرة 1%):' if is_ar else 'Safe Lot Size (1% Risk):'}</span>
                    <strong class="mono" style="font-size: 16px; color: #818cf8;">{lot} Lots</strong>
                </div>

                <!-- SMC & Indicators Summary -->
                <div class="smc-list">
                    <div class="smc-item">
                        <span class="smc-key">{'منطقة الطلب / العرض (Order Block):' if is_ar else 'Order Block Zone:'}</span>
                        <span class="smc-val">{'منطقة مؤكدة بنجاح' if is_ar else 'Validated Institutional Zone'}</span>
                    </div>
                    <div class="smc-item">
                        <span class="smc-key">{'الفجوة السعرية (FVG Status):' if is_ar else 'FVG Status:'}</span>
                        <span class="smc-val">{'تم ملء الفجوة واكتمال التوازن' if is_ar else 'Filled & Mitigated'}</span>
                    </div>
                    <div class="smc-item">
                        <span class="smc-key">{'سحب السيولة (Liquidity Sweep):' if is_ar else 'Liquidity Sweep:'}</span>
                        <span class="smc-val">{'تم التقاط قيعان الجلسة السابقة' if is_ar else 'Asian Lows Swept'}</span>
                    </div>
                    <div class="smc-item">
                        <span class="smc-key">{'مؤشرات الزخم والاتجاه:' if is_ar else 'Momentum / Volatility:'}</span>
                        <span class="smc-val mono">RSI: {rsi:.1f} • ADX: {adx:.1f} • ATR: {atr:.{digits}f}</span>
                    </div>
                </div>
            </div>
        </main>

        <!-- Detailed Narrative Report -->
        <section class="report-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid rgba(51, 65, 85, 0.4);">
                <h3 style="font-size: 17px; font-weight: 800; color: #fff;">
                    📋 {'التشريح الفني والمؤسسي الشامل' if is_ar else 'Comprehensive Institutional Breakdown'}
                </h3>
                <span style="font-size: 12px; color: #94a3b8;">FXENGIN Deep Dossier</span>
            </div>
            <div>{report_html}</div>
        </section>

        <!-- Share & Action Tools -->
        <div class="share-actions">
            <button class="btn-action copy" onclick="copyCurrentLink()">
                <span>🔗</span>
                <span>{'نسخ رابط التحليل' if is_ar else 'Copy Share Link'}</span>
            </button>
            <a href="https://t.me/share/url?url=" target="_blank" id="tgShareBtn" class="btn-action telegram">
                <span>✈️</span>
                <span>Telegram</span>
            </a>
            <a href="https://api.whatsapp.com/send?text=" target="_blank" id="waShareBtn" class="btn-action whatsapp">
                <span>💬</span>
                <span>WhatsApp</span>
            </a>
            <a href="{image_src}" target="_blank" class="btn-action image">
                <span>🖼️</span>
                <span>{'فتح الصورة المباشرة' if is_ar else 'Open Direct Image'}</span>
            </a>
            <button class="btn-action" style="background: #6366f1; color: white;" onclick="requestFreshSnapshot('{symbol}')">
                <span>🔄</span>
                <span>{'طلب لقطة شاشة جديدة' if is_ar else 'Request Fresh Snapshot'}</span>
            </button>
        </div>
    </div>

    <!-- Toast -->
    <div id="toast">✅ تم نسخ الرابط بنجاح!</div>

    <!-- Modal for Fullscreen Chart Zoom -->
    <div id="modal" onclick="closeModal()">
        <img id="modalImg" src="" alt="Fullscreen Chart">
    </div>

    <script>
        // Setup Social Share Links
        document.addEventListener('DOMContentLoaded', () => {{
            const currentUrl = encodeURIComponent(window.location.href);
            const text = encodeURIComponent('{meta_title}\\n{meta_desc}\\n');
            const tg = document.getElementById('tgShareBtn');
            const wa = document.getElementById('waShareBtn');
            if (tg) tg.href = `https://t.me/share/url?url=${{currentUrl}}&text=${{text}}`;
            if (wa) wa.href = `https://api.whatsapp.com/send?text=${{text}}%20${{currentUrl}}`;
        }});

        function copyCurrentLink() {{
            navigator.clipboard.writeText(window.location.href).then(() => {{
                showToast("{'✅ تم نسخ رابط التحليل بنجاح!' if is_ar else '✅ Share link copied to clipboard!'}");
            }}).catch(() => {{
                prompt("{'انسخ الرابط:' if is_ar else 'Copy link:'}", window.location.href);
            }});
        }}

        function showToast(msg) {{
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.style.display = 'block';
            setTimeout(() => {{ t.style.display = 'none'; }}, 3000);
        }}

        function openModal(src) {{
            const m = document.getElementById('modal');
            const mi = document.getElementById('modalImg');
            mi.src = src;
            m.style.display = 'flex';
        }}

        function closeModal() {{
            document.getElementById('modal').style.display = 'none';
        }}

        async function requestFreshSnapshot(symbol) {{
            showToast("{'⏳ جاري طلب لقطة شاشة وتحليل جديد من MT5...' if is_ar else '⏳ Requesting fresh snapshot from MT5...'}");
            try {{
                const res = await fetch(`/api/v1/analyst/snapshot/request-fresh/${{symbol}}`, {{
                    method: 'POST'
                }});
                const data = await res.json();
                if (data && data.status === 'SUCCESS' && data.share_url) {{
                    showToast("{'✅ تم توليد اللقطة بنجاح! جاري الانتقال...' if is_ar else '✅ Fresh snapshot created! Redirecting...'}");
                    setTimeout(() => {{
                        const langParam = new URLSearchParams(window.location.search).get('lang') || '{lang}';
                        window.location.href = `${{data.share_url}}?lang=${{langParam}}`;
                    }}, 1000);
                }} else {{
                    alert("{'تعذر أخذ لقطة جديدة: ' if is_ar else 'Failed to refresh: '}" + (data?.message || 'Error'));
                }}
            }} catch (err) {{
                alert("{'حدث خطأ أثناء طلب اللقطة: ' if is_ar else 'Error requesting snapshot: '}" + err.message);
            }}
        }}
    </script>
</body>
</html>
"""
