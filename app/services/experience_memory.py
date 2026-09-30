import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.config import settings
from app.services.brainstorm_service import brainstorm_service

logger = logging.getLogger("ExperienceMemory")

class EpisodicExperienceMemory:
    """
    ==============================================================================
    🧠 EPISODIC EXPERIENCE MEMORY & POST-MORTEM AUTOPSY ENGINE
    ==============================================================================
    Transforms trade outcomes into persistent visual and contextual knowledge.
    Core Capabilities:
    1. 📸 Trade Snapshot Vault: Stores Entry Chart, Exit Chart, Context, and Rationale.
    2. 🔬 Visual Post-Mortem Autopsy: Diagnoses root causes of loss vs win.
    3. 📜 Rule & Lesson Formulation: Extracts actionable trading rules (e.g. RULE_GOLD_M5_WICK_TRAP).
    4. 🔁 In-Context Experience Replay: Feeds relevant past mistakes to the Vision AI
       before new setups are executed to strictly prevent repeating known traps.
    ==============================================================================
    """

    def __init__(self):
        self.memory_file = settings.EXPERIENCE_MEMORY_FILE
        self.active_snapshots: Dict[str, Dict[str, Any]] = {}
        self.memory_state = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        default_state = {
            "version": "2.0.0",
            "total_autopsies_conducted": 0,
            "total_lessons_formulated": 0,
            "total_prevented_traps": 0,
            "experiences": [],
            "learned_rules": [
                {
                    "rule_id": "RULE_GEN_01_CHOP_TRAP",
                    "symbol": "ALL",
                    "trigger_condition": "ADX < 22 and tight Bollinger Band squeeze",
                    "lesson_ar": "تجنب تماماً الدخول في صفقات كسر وهمي أثناء ضيق مؤشر بولينجر وضعف ADX",
                    "lesson_en": "Never enter breakout trades during low ADX chop (<22) inside tight Bollinger squeeze.",
                    "severity": "CRITICAL",
                    "times_enforced": 5,
                    "created_at": "2026-08-15 10:00:00 UTC"
                },
                {
                    "rule_id": "RULE_GOLD_M5_WICK_SWEEP",
                    "symbol": "XAUUSD",
                    "trigger_condition": "M5 Long wick rejection at Asia High during London Open",
                    "lesson_ar": "على الذهب: لا تشتري فوراً عند كسر قمة آسيا، انتظر إغلاق شمعة M5 صريحة لتفادي سحب السيولة",
                    "lesson_en": "On Gold: Do not market buy upon Asia High break; wait for a confirmed M5 candle close to prevent liquidity sweep fakeouts.",
                    "severity": "HIGH",
                    "times_enforced": 8,
                    "created_at": "2026-08-20 08:30:00 UTC"
                }
            ],
            "trap_catalogs": {
                "LIQUIDITY_SWEEP_FAKEOUT": 0,
                "COUNTER_TREND_TRAP": 0,
                "HIGH_IMPACT_NEWS_SLIPPAGE": 0,
                "PREMATURE_BREAKOUT": 0,
                "SPREAD_EXPANSION_STOP_HUNT": 0
            }
        }
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    default_state.update(saved)
                    logger.info(f"Loaded {len(default_state.get('experiences', []))} visual experiences and {len(default_state.get('learned_rules', []))} learned rules from disk.")
        except Exception as e:
            logger.warning(f"Could not load experience memory: {e}")
        return default_state

    def save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory_state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving experience memory: {e}")

    def capture_entry_experience(
        self,
        symbol: str,
        ticket: Optional[int],
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        lot_size: float,
        chart_image_base64: Optional[str] = None,
        vision_analysis: Optional[Dict[str, Any]] = None,
        strategy_name: str = "HYBRID"
    ):
        """
        Saves a rich visual and structural snapshot at the precise moment of order entry.
        """
        snapshot_id = str(ticket) if ticket and ticket > 0 else f"{symbol}_{int(time.time())}"
        
        snapshot = {
            "snapshot_id": snapshot_id,
            "ticket": ticket,
            "symbol": symbol.upper(),
            "direction": direction.upper(),
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "lot_size": lot_size,
            "entry_time": time.time(),
            "entry_time_str": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "strategy_name": strategy_name,
            "entry_chart_image": chart_image_base64 or "",
            "vision_analysis": vision_analysis or {},
            "context_bias": vision_analysis.get("bias", "NEUTRAL") if vision_analysis else "NEUTRAL",
            "confidence_at_entry": vision_analysis.get("confidence", 75) if vision_analysis else 75
        }

        self.active_snapshots[snapshot_id] = snapshot
        self.active_snapshots[f"LATEST_{symbol.upper()}"] = snapshot
        logger.info(f"📸 Captured Entry Experience Snapshot #{snapshot_id} for {symbol} {direction}")

    def conduct_post_mortem_autopsy(
        self,
        closed_trade: Dict[str, Any],
        exit_chart_image_base64: Optional[str] = None,
        ai_autopsy_rationale: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Conducts an automated visual & quantitative post-mortem autopsy when a trade closes.
        Formulates a permanent Learned Rule if the trade resulted in a loss.
        """
        ticket = closed_trade.get("ticket")
        order_id = closed_trade.get("order")
        symbol = closed_trade.get("symbol", "XAUUSD").upper()
        profit = float(closed_trade.get("profit", 0.0))
        is_loss = (profit < -0.05)

        # Retrieve entry snapshot
        snapshot = None
        if ticket and str(ticket) in self.active_snapshots:
            snapshot = self.active_snapshots.pop(str(ticket))
        elif order_id and str(order_id) in self.active_snapshots:
            snapshot = self.active_snapshots.pop(str(order_id))
        elif f"LATEST_{symbol}" in self.active_snapshots:
            snapshot = self.active_snapshots.get(f"LATEST_{symbol}")

        self.memory_state["total_autopsies_conducted"] += 1

        entry_chart = snapshot.get("entry_chart_image", "") if snapshot else ""
        direction = snapshot.get("direction", closed_trade.get("type", "BUY")) if snapshot else closed_trade.get("type", "BUY")
        entry_price = snapshot.get("entry_price", closed_trade.get("price", 0.0)) if snapshot else closed_trade.get("price", 0.0)

        # Diagnose root cause
        trap_type = "PREMATURE_BREAKOUT"
        lesson_ar = ""
        lesson_en = ""

        if is_loss:
            if ai_autopsy_rationale and "trap_type" in ai_autopsy_rationale:
                trap_type = ai_autopsy_rationale["trap_type"]
                lesson_ar = ai_autopsy_rationale.get("lesson_ar", "")
                lesson_en = ai_autopsy_rationale.get("lesson_en", "")
            else:
                # Heuristic Autopsy Diagnosis
                if "XAU" in symbol or "GOLD" in symbol:
                    trap_type = "LIQUIDITY_SWEEP_FAKEOUT"
                    lesson_ar = f"تشريح خسارة صفقة الذهب #{ticket}: حدث سحب سيولة خفي عند المستويات السعرية قبل متابعة الاتجاه. يجب زيادة مسافة الأمان تحت ذيل شمعة السيولة."
                    lesson_en = f"Gold Trade #{ticket} Loss Autopsy: Hidden liquidity sweep invalidated entry. Expand SL buffer below liquidity wick."
                else:
                    trap_type = "COUNTER_TREND_TRAP"
                    lesson_ar = f"تشريح خسارة صفقة {symbol} #{ticket}: محاولة الدخول بعكس اتجاه الفريم الأكبر، يجب تأكيد توافق فريم H1 قبل دخول صفقات M5."
                    lesson_en = f"{symbol} #{ticket} Loss Autopsy: Counter-trend entry caught in higher timeframe flow. Enforce H1 alignment before M5 triggers."

            # Update trap catalog
            self.memory_state["trap_catalogs"][trap_type] = self.memory_state["trap_catalogs"].get(trap_type, 0) + 1

            # Generate new permanent Learned Rule
            rule_id = f"RULE_{symbol}_{trap_type[:8]}_{int(time.time()) % 10000}"
            new_rule = {
                "rule_id": rule_id,
                "symbol": symbol,
                "trigger_condition": f"Pattern resembling trade #{ticket} ({direction} on {symbol})",
                "lesson_ar": lesson_ar,
                "lesson_en": lesson_en,
                "severity": "CRITICAL" if abs(profit) > 30 else "HIGH",
                "times_enforced": 1,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            }
            self.memory_state["learned_rules"].insert(0, new_rule)
        else:
            lesson_ar = f"صفقة رابحة على {symbol} (+${profit:.2f}): النمط الفني واستهداف السيولة حقق الهدف بنجاح تام وفق هيكل السوق."
            lesson_en = f"Winning trade on {symbol} (+${profit:.2f}): Structure alignment and liquidity targets executed successfully."

        # Determine SL hit vs TP hit vs BE
        lot = float(closed_trade.get("volume", closed_trade.get("lots", snapshot.get("lot_size", 0.05) if snapshot else 0.05)))
        comment = str(closed_trade.get("comment", "")).lower()

        if profit <= -0.15 or "sl" in comment or "stop" in comment:
            hit_sl = True
            hit_sl_label = "نعم (ضرب الستوب)"
            exit_reason = "HIT_STOP_LOSS"
        elif profit >= 0.15 or "tp" in comment or "take" in comment:
            hit_sl = False
            hit_sl_label = "لا (حقق الهدف بربح)"
            exit_reason = "HIT_TAKE_PROFIT"
        else:
            hit_sl = False
            hit_sl_label = "لا (تأمين دخول / تعادل)"
            exit_reason = "BREAK_EVEN"

        # Ensure chart screenshot is attached
        trade_chart = exit_chart_image_base64 or entry_chart
        if not trade_chart:
            try:
                from app.services.analyst_service import analyst_service
                trade_chart = analyst_service.cached_mt5_screenshots.get(symbol, {}).get("image_base64", "")
            except Exception:
                pass

        autopsy_record = {
            "autopsy_id": f"AUTOPSY_{ticket or int(time.time())}",
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "lot_size": lot,
            "entry_price": entry_price,
            "exit_price": closed_trade.get("price", 0.0),
            "profit": profit,
            "outcome": "WIN" if not is_loss else "LOSS",
            "hit_sl": hit_sl,
            "hit_sl_label": hit_sl_label,
            "exit_reason": exit_reason,
            "trap_type": trap_type if is_loss else "SUCCESSFUL_EXECUTION",
            "lesson_ar": lesson_ar,
            "lesson_en": lesson_en,
            "chart_screenshot_base64": trade_chart,
            "closed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "closed_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "post_mortem_details": ai_autopsy_rationale or {}
        }

        # Keep last 150 experiences in memory
        self.memory_state["experiences"].insert(0, autopsy_record)
        self.memory_state["experiences"] = self.memory_state["experiences"][:150]
        
        # Keep maximum 30 active learned rules
        self.memory_state["learned_rules"] = self.memory_state["learned_rules"][:30]

        self.save_memory()

        log_level = "RISK" if is_loss else "EXECUTION"
        brainstorm_service.add_log(
            level=log_level,
            category="AI_LEARNING",
            symbol=symbol,
            message=f"🔬 [AI Autopsy] Trade #{ticket} on {symbol} closed ({'WIN: +$' if not is_loss else 'LOSS: -$'}{abs(profit):.2f}) | Hit SL: {hit_sl_label} -> {lesson_ar}"
        )

        logger.info(f"🔬 Autopsy Completed for #{ticket} ({symbol}): Outcome={autopsy_record['outcome']} | Hit SL={hit_sl_label}")
        return autopsy_record

    def get_detailed_trade_screen_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns exhaustive list of completed trades with screenshots, SL status, and detailed analysis"""
        return self.memory_state.get("experiences", [])[:limit]

    def get_in_context_experience_prompt(self, symbol: str, current_bias: str = "BULLISH") -> str:
        """
        Retrieves relevant past lessons and failure traps for the specified symbol
        to inject directly into the Vision AI prompt for in-context learning.
        """
        sym = symbol.upper()
        relevant_rules = [
            r for r in self.memory_state.get("learned_rules", [])
            if r.get("symbol") in [sym, "ALL"]
        ][:4]

        if not relevant_rules:
            return "No historical visual trap recorded for this symbol. Proceed with standard institutional confirmation."

        prompt_block = "⚠️ CRITICAL LEARNED LESSONS FROM YOUR PAST MISTAKES ON THIS SYMBOL (DO NOT REPEAT):\n"
        for idx, rule in enumerate(relevant_rules, 1):
            prompt_block += f"{idx}. [{rule.get('rule_id')}]: {rule.get('lesson_en')}\n"

        prompt_block += "Strictly verify that the current chart does NOT exhibit any of the trap signatures listed above before confirming high confidence.\n"
        return prompt_block

    def evaluate_setup_against_learned_rules(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        sl_price: float,
        tp_price: float,
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        🛡️ ACTIVE AUTOPSY SHIELD:
        Cross-examines an incoming trade setup against the catalog of historical loss traps
        and learned rules to actively protect capital and adjust parameters BEFORE execution.
        """
        sym = symbol.upper()
        dir_clean = direction.upper()
        is_gold = ("XAU" in sym or "GOLD" in sym)
        
        adx = float(indicators.get("adx", 25.0))
        atr = float(indicators.get("atr", 3.5 if is_gold else 0.0020))
        rsi = float(indicators.get("rsi", 50.0))
        bandwidth = float(indicators.get("bandwidth", 0.30))
        macro_trend = str(indicators.get("macro_trend", "NEUTRAL")).upper()
        wick_ratio = float(indicators.get("upper_wick_ratio" if dir_clean == "BUY" else "lower_wick_ratio", 0.0))
        has_mss = bool(indicators.get("has_mss", False))
        
        traps_flagged = []
        rules_enforced = []
        penalty_score = 0.0
        adjusted_sl_dist = 0.0
        advice_ar = []
        advice_en = []
        
        # 1. Check for Counter-Trend Trap (from historical loss autopsies)
        is_counter_trend = (dir_clean == "BUY" and "BEAR" in macro_trend) or (dir_clean == "SELL" and "BULL" in macro_trend)
        if is_counter_trend and not has_mss:
            traps_flagged.append("COUNTER_TREND_TRAP")
            penalty_score += 15.0
            rules_enforced.append("RULE_COUNTER_TREND_PROTECTION")
            advice_ar.append(f"⛔ الصفقة تعاكس الاتجاه العام ({macro_trend}) دون تأكيد كسر هيكل واضح (MSS) - تم خفض الثقة وتصنيفها كصفقة مبدئية عالية الخطورة.")
            advice_en.append(f"Counter-trend setup opposing {macro_trend} macro flow without confirmed MSS - confidence penalised.")

        # 2. Check for Wick / Liquidity Sweep Trap (especially on Gold)
        if is_gold and wick_ratio > 0.40:
            traps_flagged.append("LIQUIDITY_SWEEP_FAKEOUT")
            penalty_score += 12.0
            rules_enforced.append("RULE_GOLD_M5_WICK_SWEEP")
            advice_ar.append("⚠️ ذيل شمعة طويل يعكس رفضاً سعرياً (احتمال سحب سيولة خادع) - يجب انتظار إغلاق شمعة M5 صريحة أو تأمين وقف الخسارة خلف الذيل.")
            advice_en.append("Long rejection wick detected - high risk of liquidity sweep fakeout; confirm structural close.")

        # 3. Check for Low ADX Stagnant Chop Trap
        if adx < 20.0 and bandwidth < 0.15:
            traps_flagged.append("CHOP_FALSE_BREAKOUT")
            penalty_score += 15.0
            rules_enforced.append("RULE_GEN_01_CHOP_TRAP")
            advice_ar.append(f"⚠️ ضعف الزخم الشديد (ADX: {adx:.1f}) مع انكماش البولنجر - خطر مصيدة كسر كاذب.")
            advice_en.append(f"Low ADX ({adx:.1f}) compression detected - false breakout trap risk.")

        # 4. Check for Stop Loss Too Tight for Volatility (Noise Hit Trap)
        actual_sl_dist = abs(current_price - sl_price) if (current_price > 0 and sl_price > 0) else 0.0
        min_safe_sl = 1.35 * atr
        if atr > 0 and actual_sl_dist > 0 and actual_sl_dist < min_safe_sl:
            traps_flagged.append("TIGHT_SL_VOLATILITY_NOISE")
            adjusted_sl_dist = round(min_safe_sl, 2 if is_gold else 5)
            advice_ar.append(f"🛡️ الوقف المقترح كان ضيقاً جداً مقارنة بتقلبات السوق - تم تعديل مسافة الأمان إلى {adjusted_sl_dist:.2f} لتفادي ضرب الستوب بالضوضاء.")
            advice_en.append(f"Stop loss was too tight for current ATR - safe distance adjusted to {adjusted_sl_dist}.")

        # 5. Increment enforcement count in memory state
        for r_id in rules_enforced:
            for rule in self.memory_state.get("learned_rules", []):
                if rule.get("rule_id") == r_id or r_id in rule.get("rule_id", ""):
                    rule["times_enforced"] = rule.get("times_enforced", 0) + 1
                    self.memory_state["total_prevented_traps"] = self.memory_state.get("total_prevented_traps", 0) + 1

        is_safe = (len(traps_flagged) == 0 or penalty_score < 20.0)
        
        return {
            "is_safe": is_safe,
            "penalty_score": penalty_score,
            "traps_flagged": traps_flagged,
            "rules_enforced": rules_enforced,
            "adjusted_sl_dist": adjusted_sl_dist,
            "advice_ar": " | ".join(advice_ar) if advice_ar else "✅ اجتاز الفحص بنجاح: لا توجد مصائد سابقة مسجلة على هذا الإعداد.",
            "advice_en": " | ".join(advice_en) if advice_en else "Clean setup: No historical failure traps detected."
        }

    def get_memory_summary(self) -> Dict[str, Any]:
        """Returns full episodic experience memory data for Dashboard visualization"""
        return {
            "total_autopsies": self.memory_state.get("total_autopsies_conducted", 0),
            "total_lessons": self.memory_state.get("total_lessons_formulated", 0),
            "learned_rules": self.memory_state.get("learned_rules", []),
            "trap_catalogs": self.memory_state.get("trap_catalogs", {}),
            "recent_experiences": self.memory_state.get("experiences", [])[:15]
        }

    def clear_memory(self):
        """Resets the experience memory (for testing or fresh start)"""
        self.memory_state["experiences"] = []
        self.memory_state["learned_rules"] = []
        self.memory_state["total_autopsies_conducted"] = 0
        self.memory_state["total_lessons_formulated"] = 0
        self.save_memory()

episodic_memory = EpisodicExperienceMemory()
