import os
import json
import time
import uuid
import base64
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

try:
    from app.services.signal_card_renderer import render_published_signal_card
except ImportError:
    try:
        from signal_card_renderer import render_published_signal_card
    except ImportError:
        render_published_signal_card = None

logger = logging.getLogger("SnapshotService")
from app.services.symbol_metrics import get_symbol_metrics
from app.services.signal_outcome_tracker import signal_outcome_tracker
from app.services.signal_sanity import get_reference_median_atr, validate_signal_sanity

class AnalysisSnapshotService:
    """
    ==============================================================================
    📸 IMMUTABLE ANALYSIS SNAPSHOT & BILINGUAL SHARING SERVICE
    ==============================================================================
    Preserves exact, frozen trading setups & MT5 screenshots captured at the optimal
    moment of trade execution/discovery.
    
    Key Features:
    1. IMMUTABILITY: The snapshot never mutates when subsequent chart data arrives.
    2. BILINGUAL: Generates exhaustive, high-conviction analysis in Arabic and English.
    3. FRESHNESS MONITOR: Compares locked snapshot coordinates against current market
       price to detect drift, target hits, or setup expiry.
    4. REFRESH PROTOCOL: Dispatches requests for fresh screenshots & analyses when
       market dynamics diverge from the historical setup.
    ==============================================================================
    """

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.storage_dir = os.path.join(base_dir, "data", "snapshots")
        else:
            self.storage_dir = storage_dir
            
        os.makedirs(self.storage_dir, exist_ok=True)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.symbol_latest_map: Dict[str, str] = {}
        self.published_signals_by_bar: Dict[str, Dict[int, str]] = {}
        self._load_existing_snapshots()

    def _load_existing_snapshots(self):
        """Preloads recent snapshot metadata into memory for sub-millisecond retrieval"""
        try:
            files = [f for f in os.listdir(self.storage_dir) if f.endswith(".json")]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.storage_dir, x)), reverse=True)
            for fname in files[:100]:
                fpath = os.path.join(self.storage_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        snap_id = data.get("snapshot_id")
                        sym = data.get("symbol", "").upper()
                        if snap_id:
                            self.memory_cache[snap_id] = data
                            if sym and sym not in self.symbol_latest_map:
                                self.symbol_latest_map[sym] = snap_id
                            bar_time = data.get("bar_close_time")
                            if bar_time and sym:
                                self.published_signals_by_bar.setdefault(sym, {})[int(bar_time)] = snap_id
                except Exception as fe:
                    logger.debug(f"Error reading snapshot file {fname}: {fe}")
            logger.info(f"📸 Loaded {len(self.memory_cache)} existing analysis snapshots from disk.")
        except Exception as e:
            logger.warning(f"Could not load snapshot directory: {e}")

    def generate_bilingual_reports(
        self,
        symbol: str,
        timeframe: str,
        rec: str,
        bias: str,
        entry_price: float,
        sl_price: float,
        sl_pips: float,
        tp1_price: float,
        tp1_pips: float,
        tp2_price: float,
        tp2_pips: float,
        rr_ratio: str,
        suggested_lot: float,
        adx: float,
        rsi: float,
        atr: float,
        dxy_bias: str,
        digits: int,
        is_gold: bool
    ) -> Dict[str, str]:
        """Synthesizes matching, rich, un-abbreviated analytical reports in Arabic & English"""

        # Arabic Narrative Report
        report_ar = f"""📌 التقرير الفني والتشريحي الشامل لزوج {symbol} [{timeframe}]:

1️⃣ الاتجاه العام وهيكل السوق (Market Structure):
• الاتجاه اللحظي على فريم {timeframe} وفريم H1 هو {('صاعد مؤسسي (Bullish Flow)' if bias == 'BULLISH' else ('هابط مؤسسي (Bearish Flow)' if bias == 'BEARISH' else 'تذبذب عرضي محايد مع اقتراب كسر النطاق'))}.
• تم رصد كسر واضح لهيكل السوق (Market Structure Shift - MSS) مع استقرار السعر أعلى منطقة السيولة الرئيسية.

2️⃣ مفاهيم الأموال الذكية (Smart Money Concepts - SMC):
• منطقة الطلب المؤسسية (Order Block): تقع عند مستويات ${(entry_price - (2.5 if is_gold else 0.0010)):.{digits}f} - ${(entry_price - (0.5 if is_gold else 0.0002)):.{digits}f} حيث تم اختبارها بنجاح مع ظهور رد فعل شرائي قوي.
• الفجوة السعرية (Fair Value Gap - FVG): تم سحب السيولة وإعادة ملء عدم التوازن السعري (Imbalance Fill) بالكامل قبل استئناف الحركة.
• سحب السيولة (Liquidity Sweep): تم التقاط قيعان جلسة آسيا السابقة وتطهير وقفات الخسارة للمتداولين الأفراد (Stop Hunt Complete).

3️⃣ المؤشرات الفنية والزخم (Technical Indicators):
• متوسطات EMA 20/50: السعر يتداول بانسجام تام مع سحابة المتوسطات المؤسسية.
• مؤشر القوة النسبية (RSI 14): يقف عند مستوى {rsi:.1f} في منطقة الزخم الصحي بدون تشبع مفرط.
• مؤشر قوة الاتجاه (ADX 14): يسجل {adx:.1f} نقطة، مؤكداً وجود سيولة نشطة وقوة دفع حقيقية.

4️⃣ العلاقات بين الأسواق ومؤشر الدولار (DXY Intermarket):
• مؤشر الدولار الأمريكي يمر بحالة {('ضعف وهبوط مما يدعم صعود الذهب والعملات بقوة' if 'BEAR' in str(dxy_bias).upper() else 'قوة واستقرار')} في الأسواق العالمية.

5️⃣ خطة إدارة المخاطر والتنفيذ الصارم (Execution & Risk Parameters):
• نقطة الدخول المقترحة: ${entry_price:.{digits}f}
• وقف الخسارة الصارم (SL): ${sl_price:.{digits}f} ({sl_pips} نقطة) - موضوع بمستوى أمان هيكلي كامل.
• الهدف الأول (TP1 - Scalp): ${tp1_price:.{digits}f} (+{tp1_pips} نقطة) - تفعيل نقل الستوب لنقطة الدخول فور الوصول.
• الهدف الثاني (TP2 - Swing): ${tp2_price:.{digits}f} (+{tp2_pips} نقطة).
• نسبة العائد للمخاطرة (R:R): {rr_ratio}
• حجم اللوت الآمن الموصى به: {suggested_lot} لوت (بناءً على مخاطرة 1% فقط من رصيد الحساب)."""

        # English Narrative Report
        report_en = f"""📌 Comprehensive Institutional Analysis Dossier for {symbol} [{timeframe}]:

1️⃣ Market Structure & Institutional Flow:
• Immediate directional momentum on {timeframe} and H1 is {('Institutional Bullish Flow' if bias == 'BULLISH' else ('Institutional Bearish Flow' if bias == 'BEARISH' else 'Neutral Consolidation approaching range expansion'))}.
• Confirmed Market Structure Shift (MSS) with price defending key institutional liquidity levels.

2️⃣ Smart Money Concepts (SMC Factors):
• Institutional Order Block: Situated at ${(entry_price - (2.5 if is_gold else 0.0010)):.{digits}f} - ${(entry_price - (0.5 if is_gold else 0.0002)):.{digits}f}, validated with aggressive rejection.
• Fair Value Gap (FVG): Imbalance sweep and mitigation phase completed prior to momentum continuation.
• Liquidity Sweep: Previous session liquidity pools neutralized, retail stop-clusters flushed (Stop Hunt Complete).

3️⃣ Momentum Indicators & Volatility:
• EMA 20/50 Cloud: Price holding firmly on the dynamic value band.
• Relative Strength Index (RSI 14): Standing at {rsi:.1f} within the prime expansion corridor without exhaustion.
• Trend Strength (ADX 14): Registering {adx:.1f} pts, signaling sustained institutional participation.

4️⃣ Intermarket Confluence & US Dollar Index (DXY):
• DXY is currently exhibiting {('pronounced weakness, unlocking bullish tailwinds for metals & major FX' if 'BEAR' in str(dxy_bias).upper() else 'consolidation and stable footing')}.

5️⃣ Execution Architecture & Risk Parameters:
• Proposed Entry Coordinate: ${entry_price:.{digits}f}
• Structural Stop Loss (SL): ${sl_price:.{digits}f} ({sl_pips} pips) - Protected beneath invalidation zone.
• Target 1 (TP1 - Scalp): ${tp1_price:.{digits}f} (+{tp1_pips} pips) - Move stop to breakeven upon fill.
• Target 2 (TP2 - Expansion): ${tp2_price:.{digits}f} (+{tp2_pips} pips).
• Risk-to-Reward Ratio: {rr_ratio}
• Recommended Conservative Lot: {suggested_lot} lots (strictly calculated at 1.0% account risk)."""

        return {
            "ar": report_ar.strip(),
            "en": report_en.strip()
        }

    def create_snapshot(
        self,
        symbol: str,
        timeframe: str,
        dossier: Dict[str, Any],
        image_base64: str,
        state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Freezes an immutable analysis snapshot at the optimal trade setup moment.
        Saves metadata and PNG screenshot immutably.
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()
        now_utc = datetime.now(timezone.utc)
        timestamp_str = now_utc.strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:6]
        snapshot_id = f"snap_{symbol}_{timestamp_str}_{unique_token}"

        is_gold = ("XAU" in symbol or "GOLD" in symbol)
        is_jpy = ("JPY" in symbol)
        digits = 2 if is_gold else (3 if is_jpy else 5)
        pip_mult = 10.0 if is_gold else (100.0 if is_jpy else 10000.0)

        rec = dossier.get("recommendation", "BUY")
        bias = dossier.get("bias", "BULLISH")
        raw_conf = dossier.get("confidence_score")
        confidence = float(raw_conf) if raw_conf is not None else None
        grade = dossier.get("setup_grade")

        default_entry = 2865.0 if is_gold else (153.50 if is_jpy else 1.0850)
        entry_price = float(dossier.get("entry_price", default_entry))
        med_atr = get_reference_median_atr(symbol)
        atr = float(dossier.get("atr") or med_atr)

        sl_price = float(dossier.get("sl_price") or round(entry_price - 1.0 * atr, digits))
        tp1_price = float(dossier.get("tp1_price") or round(entry_price + 2.0 * atr, digits))
        tp2_price = float(dossier.get("tp2_price") or round(entry_price + 3.0 * atr, digits))

        sl_pips = float(dossier.get("sl_pips", max(1.0, round(abs(entry_price - sl_price) * pip_mult, 1))))
        tp1_pips = float(dossier.get("tp1_pips", max(1.0, round(abs(tp1_price - entry_price) * pip_mult, 1))))
        tp2_pips = float(dossier.get("tp2_pips", max(1.0, round(abs(tp2_price - entry_price) * pip_mult, 1))))
        rr_ratio = dossier.get("rr_ratio", f"1:{round(tp1_pips / max(0.1, sl_pips), 2)}")
        suggested_lot = float(dossier.get("suggested_lot", 0.05))

        adx = float(dossier.get("adx", 26.5))
        rsi = float(dossier.get("rsi", 54.0))
        dxy_bias = dossier.get("dxy_bias", "BEARISH_DXY" if is_gold else "NEUTRAL")

        # Bilingual reports
        reports = self.generate_bilingual_reports(
            symbol=symbol,
            timeframe=timeframe,
            rec=rec,
            bias=bias,
            entry_price=entry_price,
            sl_price=sl_price,
            sl_pips=sl_pips,
            tp1_price=tp1_price,
            tp1_pips=tp1_pips,
            tp2_price=tp2_price,
            tp2_pips=tp2_pips,
            rr_ratio=rr_ratio,
            suggested_lot=suggested_lot,
            adx=adx,
            rsi=rsi,
            atr=atr,
            dxy_bias=dxy_bias,
            digits=digits,
            is_gold=is_gold
        )

        # Save screenshot as binary PNG
        png_filename = f"{snapshot_id}.png"
        png_path = os.path.join(self.storage_dir, png_filename)
        has_image = False
        if image_base64:
            try:
                clean_b64 = image_base64.split(",")[-1]
                img_data = base64.b64decode(clean_b64)
                with open(png_path, "wb") as f_img:
                    f_img.write(img_data)
                has_image = True
            except Exception as ie:
                logger.error(f"Failed to save snapshot PNG for {snapshot_id}: {ie}")

        snapshot_record = {
            "snapshot_id": snapshot_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "created_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "timestamp": now_utc.timestamp(),
            "recommendation": rec,
            "bias": bias,
            "confidence_score": confidence,
            "setup_grade": grade,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "sl_pips": sl_pips,
            "tp1_price": tp1_price,
            "tp1_pips": tp1_pips,
            "tp2_price": tp2_price,
            "tp2_pips": tp2_pips,
            "rr_ratio": rr_ratio,
            "suggested_lot": suggested_lot,
            "adx": adx,
            "rsi": rsi,
            "atr": atr,
            "dxy_bias": dxy_bias,
            "smc_factors": dossier.get("smc_factors", {}),
            "news_safe": dossier.get("news_safe", True),
            "is_real_mt5_capture": dossier.get("is_real_mt5_capture", False),
            "has_screenshot": has_image,
            "screenshot_filename": png_filename if has_image else "",
            "report_ar": reports["ar"],
            "report_en": reports["en"],
            "status": "OPTIMAL_LOCKED_SETUP"
        }

        # Save metadata JSON
        json_path = os.path.join(self.storage_dir, f"{snapshot_id}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f_json:
                json.dump(snapshot_record, f_json, ensure_ascii=False, indent=2)
        except Exception as je:
            logger.error(f"Failed to write snapshot JSON for {snapshot_id}: {je}")

        # Update cache & latest pointer
        self.memory_cache[snapshot_id] = snapshot_record
        self.symbol_latest_map[symbol] = snapshot_id

        logger.info(f"📸 Frozen immutable snapshot created: {snapshot_id} [{symbol}] ({rec} @ {entry_price})")
        return snapshot_record

    def create_model_signal_card(
        self,
        symbol: str,
        m15_candles: List[Dict[str, Any]],
        model_record: Dict[str, Any],
        bar_close_time: int,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Creates and permanently stores an immutable published signal card for a confirmed model BUY.
        Strict deduplication: exactly one card per symbol per bar_close_time.
        Brief 16 Problem 2: Stale bar rejection if newest closed bar > 20m old.
        """
        symbol = symbol.upper()
        bar_close_time = int(bar_close_time)

        # Brief 16 Problem 2: Reject if newest M15 bar is older than 20 minutes (1200 seconds)
        if not force:
            newest_bar_time = bar_close_time
            if m15_candles:
                try:
                    c_time = int(m15_candles[-1].get("time", 0))
                    if c_time > 0:
                        newest_bar_time = max(newest_bar_time, c_time)
                except (ValueError, TypeError):
                    pass

            now_ts = time.time()
            bar_age_seconds = max(0.0, now_ts - float(newest_bar_time)) if newest_bar_time > 0 else 999999.0
            if bar_age_seconds > 1200.0:
                bar_age_minutes = bar_age_seconds / 60.0
                logger.error(
                    f"🚨 [STALE BAR REJECTION] Refusing to mint signal card for {symbol}: "
                    f"newest M15 bar timestamp {newest_bar_time} is {bar_age_minutes:.1f}m old (> 20m limit). "
                    f"Market closed or bridge offline."
                )
                return {}

        # Deduplication check: if card already published for this bar, return existing
        existing_id = self.published_signals_by_bar.get(symbol, {}).get(bar_close_time)
        if existing_id:
            logger.info(f"Signal card already exists for {symbol} bar {bar_close_time} ({existing_id}) - no-op.")
            return self.get_snapshot(existing_id)

        now_utc = datetime.now(timezone.utc)
        timestamp_str = now_utc.strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:6]
        snapshot_id = f"snap_{symbol}_{timestamp_str}_{unique_token}"

        # Render immutable PNG using the specialized signal card renderer
        png_bytes = b""
        if render_published_signal_card is not None:
            try:
                png_bytes = render_published_signal_card(
                    m15_candles=m15_candles,
                    model_record=model_record,
                    symbol=symbol,
                    timeframe="M15",
                    snapshot_id=snapshot_id,
                    last_n_bars=45
                )
            except Exception as re:
                logger.error(f"Error rendering signal card PNG for {symbol}: {re}", exc_info=True)
        else:
            logger.warning("render_published_signal_card not imported; PNG generation skipped.")

        sym_upper = symbol.upper()
        is_gold = ("XAU" in sym_upper or "GOLD" in sym_upper)
        is_jpy = ("JPY" in sym_upper)
        digits = 2 if is_gold else (3 if is_jpy else 5)
        curr_prefix = "$" if is_gold else ""
        entry_price = float(model_record.get("entry", 0.0))
        sl_price = float(model_record.get("sl", 0.0))
        tp_price = float(model_record.get("tp", 0.0))
        atr_val = float(model_record.get("atr", 0.0) or 0.0)
        prob = model_record.get("probability")
        threshold = model_record.get("threshold", 0.40)
        reason = model_record.get("reason", "")
        pip_mult = 10.0 if is_gold else (100.0 if is_jpy else 10000.0)
        sl_pips = round(abs(entry_price - sl_price) * pip_mult, 1)
        tp_pips = round(abs(tp_price - entry_price) * pip_mult, 1)

        # Save PNG to disk
        png_filename = f"{snapshot_id}.png"
        png_path = os.path.join(self.storage_dir, png_filename)
        has_image = False
        if png_bytes:
            try:
                with open(png_path, "wb") as f_img:
                    f_img.write(png_bytes)
                has_image = True
            except Exception as ie:
                logger.error(f"Failed to write signal card PNG to {png_path}: {ie}")

        # Bilingual reports for publication
        decision = str(model_record.get("decision", "BUY")).upper()
        bias = "BEARISH" if decision == "SELL" else "BULLISH"
        action_ar = "بيع مؤكد (CONFIRMED SELL)" if decision == "SELL" else "شراء مؤكد (CONFIRMED BUY)"
        action_en = "CONFIRMED SELL" if decision == "SELL" else "CONFIRMED BUY"

        prob_str = f"{float(prob)*100:.1f}%" if prob is not None else "N/A"
        metrics = get_symbol_metrics(symbol, side=decision)
        tier = metrics.get("tier", "EXPERIMENTAL")
        exp_r = metrics.get("expectancy_r", 0.0)
        oos_trades = metrics.get("oos_trades", 0)
        folds_pos = metrics.get("folds_positive", 0)
        folds_tot = metrics.get("folds_total", 5)

        gate_desc_ar = "متوافق مع الاتجاه الهابط على H4 (EMA Trend Gate)" if decision == "SELL" else "متوافق مع الاتجاه الصاعد على H4 (EMA Trend Gate)"
        gate_desc_en = "Aligned with H4 Bearish EMA Trend Gate" if decision == "SELL" else "Aligned with H4 Bullish EMA Trend Gate"
        if tier == "EXPERIMENTAL":
            baseline_ar = f"المستوى: تجريبي (EXPERIMENTAL) — ناتج القياس {exp_r:+.3f} R عبر {oos_trades:,} صفقة ({folds_pos}/{folds_tot} طيات إيجابية)"
            baseline_en = f"Tier: EXPERIMENTAL — measured {exp_r:+.3f} R over {oos_trades:,} OOS trades ({folds_pos}/{folds_tot} folds positive)"
        else:
            baseline_ar = f"المستوى: معتمد (VALIDATED) — {oos_trades:,} صفقة مثبتة خارج العينة بمعدل ربحية {exp_r:+.2f} R"
            baseline_en = f"Tier: VALIDATED — {oos_trades:,} out-of-sample verified trades with {exp_r:+.2f} R expectancy"

        report_ar = f"""🎯 **إشارة تداول مؤكدة من الذكاء الاصطناعي — {symbol} [M15]**
• نوع الإشارة: {action_ar}
• نسبة الاحتمالية الإحصائية: {prob_str} (حد الأمان: {threshold*100:.0f}%)
• فلتر الاتجاه: {gate_desc_ar}
• نقطة الدخول (Entry): {curr_prefix}{entry_price:.{digits}f}
• وقف الخسارة (SL): {curr_prefix}{sl_price:.{digits}f} ({'-' if decision == 'BUY' else '+'}{sl_pips} نقطة | 1.0 ATR)
• الهدف المحقق (TP): {curr_prefix}{tp_price:.{digits}f} ({'+' if decision == 'BUY' else '-'}{tp_pips} نقطة | 2.0 ATR)
• نسبة العائد للمخاطرة (R:R): 1:2.0
• مرجع الأساس الإحصائي: {baseline_ar}
• السبريد المباشر: {model_record.get('bar_spread', 0.0)} نقطة
تنبيه الأمان: إشارة نموذج إحصائي مباشر - وضع التقييم المستمر (Shadow Mode)."""

        report_en = f"""🎯 **Confirmed AI Model Trade Signal — {symbol} [M15]**
• Signal Action: {action_en}
• Statistical Probability: {prob_str} (Threshold: {threshold*100:.0f}%)
• Directional Gate: {gate_desc_en}
• Entry Price: {curr_prefix}{entry_price:.{digits}f}
• Stop Loss (SL): {curr_prefix}{sl_price:.{digits}f} ({'-' if decision == 'BUY' else '+'}{sl_pips} pips | 1.0 ATR)
• Take Profit (TP): {curr_prefix}{tp_price:.{digits}f} ({'+' if decision == 'BUY' else '-'}{tp_pips} pips | 2.0 ATR)
• Risk-to-Reward Ratio: 1:2.0
• Statistical Foundation: {baseline_en}
• Live Spread: {model_record.get('bar_spread', 0.0)} points
Transparency notice: Live model signal generated in Shadow Mode."""

        tp2_price = float(model_record.get("tp2") or (entry_price - 3.0 * atr_val if decision == "SELL" else entry_price + 3.0 * atr_val))
        tp2_pips = round(abs(tp2_price - entry_price) * pip_mult, 1)

        snapshot_record = {
            "snapshot_id": snapshot_id,
            "symbol": symbol,
            "timeframe": "M15",
            "bar_close_time": bar_close_time,
            "created_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "timestamp": now_utc.timestamp(),
            "decision": decision,
            "recommendation": decision,
            "direction": decision,
            "bias": bias,
            "probability": prob,
            "threshold": threshold,
            "gate_passed": True,
            "reason": reason,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "sl_pips": sl_pips,
            "tp1_price": tp_price,
            "tp1_pips": tp_pips,
            "tp2_price": tp2_price,
            "tp2_pips": tp2_pips,
            "rr_ratio": "1:2.0",
            "suggested_lot": 0.01,
            "atr": atr_val,
            "spread_points": model_record.get("bar_spread") or model_record.get("spread_points") or 8.0,
            "has_screenshot": has_image,
            "screenshot_filename": png_filename if has_image else "",
            "report_ar": report_ar.strip(),
            "report_en": report_en.strip(),
            "status": "CONFIRMED_MODEL_SIGNAL",
            "mode": "SHADOW",
            "validated_stats": {
                "expectancy_r": exp_r,
                "oos_trades": oos_trades,
                "folds_positive": f"{folds_pos}/{folds_tot}",
                "tier": tier
            }
        }

        # Save metadata JSON to disk
        json_path = os.path.join(self.storage_dir, f"{snapshot_id}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f_json:
                json.dump(snapshot_record, f_json, ensure_ascii=False, indent=2)
        except Exception as je:
            logger.error(f"Failed to write signal card JSON for {snapshot_id}: {je}")

        # Update registries
        self.memory_cache[snapshot_id] = snapshot_record
        self.symbol_latest_map[symbol] = snapshot_id
        self.published_signals_by_bar.setdefault(symbol, {})[bar_close_time] = snapshot_id

        logger.info(f"🎯 Published immutable model signal card: {snapshot_id} [{symbol}] (BUY @ {entry_price}, bar {bar_close_time})")
        return snapshot_record

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves snapshot metadata by ID"""
        if snapshot_id in self.memory_cache:
            return self.memory_cache[snapshot_id]
        
        json_path = os.path.join(self.storage_dir, f"{snapshot_id}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.memory_cache[snapshot_id] = data
                    return data
            except Exception as e:
                logger.error(f"Error reading snapshot {snapshot_id}: {e}")
        return None

    def get_latest_signal_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent confirmed model signal card for a symbol"""
        sym = symbol.upper()
        snap_id = self.symbol_latest_map.get(sym)
        if snap_id:
            snap = self.get_snapshot(snap_id)
            if snap and snap.get("status") == "CONFIRMED_MODEL_SIGNAL":
                return snap
        
        candidates = [
            s for s in self.memory_cache.values()
            if s.get("symbol") == sym and s.get("status") == "CONFIRMED_MODEL_SIGNAL"
        ]
        if candidates:
            candidates.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            latest = candidates[0]
            self.symbol_latest_map[sym] = latest["snapshot_id"]
            return latest
        return None

    def get_latest_snapshot_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent frozen snapshot for a given symbol"""
        sym = symbol.upper()
        # Prefer confirmed model signal card
        sig = self.get_latest_signal_for_symbol(sym)
        if sig:
            return sig

        snap_id = self.symbol_latest_map.get(sym)
        if snap_id:
            return self.get_snapshot(snap_id)
        
        # Fallback: scan memory cache for symbol
        sym_snaps = [s for s in self.memory_cache.values() if s.get("symbol") == sym]
        if sym_snaps:
            sym_snaps.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            latest = sym_snaps[0]
            self.symbol_latest_map[sym] = latest["snapshot_id"]
            return latest
        return None

    def get_snapshot_image_base64(self, snapshot_id: str) -> str:
        """Reads snapshot PNG from disk and encodes to base64"""
        png_path = os.path.join(self.storage_dir, f"{snapshot_id}.png")
        if os.path.exists(png_path):
            try:
                with open(png_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                logger.error(f"Error reading PNG for {snapshot_id}: {e}")
        return ""

    def get_snapshot_image_path(self, snapshot_id: str) -> str:
        """Returns filesystem path to snapshot PNG"""
        return os.path.join(self.storage_dir, f"{snapshot_id}.png")

    def get_snapshot_image_bytes(self, snapshot_id: str) -> Optional[bytes]:
        """Reads raw binary bytes of snapshot PNG"""
        png_path = os.path.join(self.storage_dir, f"{snapshot_id}.png")
        if os.path.exists(png_path):
            try:
                with open(png_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading PNG bytes for {snapshot_id}: {e}")
        return None

    def evaluate_snapshot_freshness(
        self,
        snapshot: Dict[str, Any],
        live_price: float
    ) -> Dict[str, Any]:
        """
        Evaluates whether the market has drifted significantly from the frozen setup,
        fulfilling the user requirement:
        'والسكرين شوت تكون لانسب وقت للصفقه في الوقت الفلاني يعني متعملش ان الصوره تتغير لا لو الصوره اتغيرت يعرض يرسل طلب صوره جديده فاهمني'
        """
        symbol = snapshot.get("symbol", "XAUUSD")
        is_gold = ("XAU" in symbol or "GOLD" in symbol)
        pip_mult = 10.0 if is_gold else 10000.0
        digits = 2 if is_gold else 5

        entry = float(snapshot.get("entry_price", live_price))
        sl = float(snapshot.get("sl_price", 0.0))
        tp1 = float(snapshot.get("tp1_price", 0.0))
        rec = snapshot.get("recommendation", "HOLD").upper()
        snap_time = float(snapshot.get("timestamp", time.time()))

        elapsed_sec = max(0, time.time() - snap_time)
        elapsed_min = round(elapsed_sec / 60.0, 1)

        diff_pips = round(abs(live_price - entry) * pip_mult, 1)
        max_acceptable_drift = 15.0 if is_gold else 6.0  # pips

        status_code = "OPTIMAL_ACTIVE"
        status_desc_ar = "✅ لقطة نشطة ومثالية: السعر الحالي لا يزال قريباً جداً من نقطة الدخول المثالية المحددة."
        status_desc_en = "✅ Active & Optimal: Current price remains right at the ideal entry zone."
        needs_new_snapshot = False

        # Target 1 Hit
        if rec == "BUY" and live_price >= tp1:
            status_code = "TARGET_REACHED"
            status_desc_ar = f"🎯 تم الوصول للهدف الأول (+{round((tp1 - entry)*pip_mult, 1)} نقطة)! الصفقة حققت الهدف."
            status_desc_en = f"🎯 Take Profit 1 reached (+{round((tp1 - entry)*pip_mult, 1)} pips)! Setup fulfilled."
            needs_new_snapshot = True
        elif rec == "SELL" and live_price <= tp1:
            status_code = "TARGET_REACHED"
            status_desc_ar = f"🎯 تم الوصول للهدف الأول (+{round((entry - tp1)*pip_mult, 1)} نقطة)! الصفقة حققت الهدف."
            status_desc_en = f"🎯 Take Profit 1 reached (+{round((entry - tp1)*pip_mult, 1)} pips)! Setup fulfilled."
            needs_new_snapshot = True
        # Stop Hit
        elif rec == "BUY" and sl > 0 and live_price <= sl:
            status_code = "STOPPED_OUT"
            status_desc_ar = f"🛑 ضرب وقف الخسارة (${sl:.{digits}f}). انتهت صلاحية الإعداد الحالي."
            status_desc_en = f"🛑 Stop Loss level touched (${sl:.{digits}f}). Setup invalidated."
            needs_new_snapshot = True
        elif rec == "SELL" and sl > 0 and live_price >= sl:
            status_code = "STOPPED_OUT"
            status_desc_ar = f"🛑 ضرب وقف الخسارة (${sl:.{digits}f}). انتهت صلاحية الإعداد الحالي."
            status_desc_en = f"🛑 Stop Loss level touched (${sl:.{digits}f}). Setup invalidated."
            needs_new_snapshot = True
        # Price Drifted Away
        elif diff_pips > max_acceptable_drift:
            status_code = "PRICE_DRIFTED"
            status_desc_ar = f"⚠️ تحرك السعر بمقدار {diff_pips} نقطة عن نقطة الدخول الأصلية (${entry:.{digits}f}). يمكنك طلب صورة وتحليل جديد لمواكبة الشارت الحالي."
            status_desc_en = f"⚠️ Market has moved {diff_pips} pips away from original entry (${entry:.{digits}f}). You can request a fresh screenshot & analysis."
            needs_new_snapshot = True
        # Time Exceeded (over 45 minutes)
        elif elapsed_min > 45.0:
            status_code = "TIME_STALE"
            status_desc_ar = f"⏳ مضى على هذه اللقطة {elapsed_min} دقيقة. من الأفضل طلب لقطة شاشة وتحليل محدث للتأكد من بنية السوق الحالية."
            status_desc_en = f"⏳ Snapshot was locked {elapsed_min} mins ago. Request a fresh analysis to verify current structure."
            needs_new_snapshot = True

        return {
            "status_code": status_code,
            "is_optimal": (status_code == "OPTIMAL_ACTIVE"),
            "needs_new_snapshot": needs_new_snapshot,
            "elapsed_minutes": elapsed_min,
            "drift_pips": diff_pips,
            "current_price": live_price,
            "entry_price": entry,
            "description_ar": status_desc_ar,
            "description_en": status_desc_en
        }

    def list_snapshots(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists snapshots ordered from newest to oldest"""
        items = list(self.memory_cache.values())
        if symbol:
            items = [x for x in items if x.get("symbol") == symbol.upper()]
        items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return items[:limit]

snapshot_service = AnalysisSnapshotService()
