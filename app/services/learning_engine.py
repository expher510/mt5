import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from app.services.brainstorm_service import brainstorm_service

logger = logging.getLogger("LearningEngine")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
LEARNING_MEMORY_FILE = os.path.join(DATA_DIR, "trade_learning_memory.json")
PENDING_SNAPSHOTS_FILE = os.path.join(DATA_DIR, "pending_snapshots.json")
DECISIONS_DIR = os.path.join(DATA_DIR, "decisions")


def log_model_decision(record: Dict[str, Any]):
    """
    Appends a model decision (including HOLD) to daily JSON Lines log.
    Flushes immediately to guarantee durability against crashes.
    """
    try:
        os.makedirs(DECISIONS_DIR, exist_ok=True)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = os.path.join(DECISIONS_DIR, f"decisions_{today_str}.jsonl")
        line = json.dumps(record, default=str) + "\n"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    except Exception as e:
        logger.error(f"Failed to log model decision to disk: {e}")


class TradeLearningEngine:
    """
    ==============================================================================
    🧠 FXENGIN AUTONOMOUS ADAPTIVE REINFORCEMENT LEARNING ENGINE (PRO VERSION)
    ==============================================================================
    Core Pillars:
    1. 🌐 Real-Time Market Regime Classifier:
       - TRENDING_EXPANSION: Explosive direction (Multi-Target & Pyramiding Active).
       - RANGE_BOUND: Defined equilibrium (SMC OB / FVG / Fibo Golden Zones Active).
       - VOLATILE_CHOP: Stagnant low-momentum chop (Safety Hold Active).
       - HIGH_VOLATILITY_EXPANSION: News / High-impact spikes (Dynamic Buffer Active).
    2. ⚖️ Model Weight & Reinforcement Matrix (Reward / Penalty):
       - MSS_BREAKOUT (Market Structure Shift)
       - EMA_TREND_PULLBACK (Trend Continuation Retest)
       - SMC_OB_RETEST (Order Block Demand/Supply)
       - FVG_IMBALANCE_FILL (Fair Value Gap Equilibrium)
       - FIBO_OTE_0618 (0.618 - 0.786 Golden Zone)
       - LIQUIDITY_SWEEP (Asia/London Session Sweeps)
    3. 🎯 Multi-Target 3-in-1 Coordinated Execution Splitter:
       - Computes TP1, TP2, TP3 lots and coordinates Break-Even once TP1 is captured.
    4. 🔬 Post-Mortem Loss Forensic & Self-Tuning Parameters:
       - Automatically tunes ADX thresholds, SL multipliers, and model weights.
    ==============================================================================
    """

    MAX_PENDING_PER_SYMBOL = 6

    # A model must have at least this many closed trades before its measured
    # expectancy is allowed to change its weight. Reacting to one or two outcomes is
    # noise-chasing, not learning - it was what made the old weights oscillate.
    MIN_SAMPLES_FOR_JUDGEMENT = 20
    # Below this measured expectancy (in R) a model with enough samples is retired.
    DISABLE_EXPECTANCY_R = -0.15
    MAX_LEDGER = 500

    def __init__(self):
        self.trade_snapshots: Dict[str, Dict[str, Any]] = {}
        # Snapshots awaiting a ticket, oldest first, per symbol
        self.pending_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        self._load_pending_snapshots()
        self.learning_state: Dict[str, Any] = self._load_learning_memory()

    def _load_pending_snapshots(self):
        """Loads pending snapshots from disk so container restarts do not orphan open trades."""
        try:
            if os.path.exists(PENDING_SNAPSHOTS_FILE):
                with open(PENDING_SNAPSHOTS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.trade_snapshots = data.get("trade_snapshots", {})
                    self.pending_by_symbol = data.get("pending_by_symbol", {})
                    logger.info(f"Loaded {len(self.trade_snapshots)} trade snapshots from disk.")
        except Exception as e:
            logger.warning(f"Could not load pending snapshots: {e}")

    def _save_pending_snapshots(self):
        """Persists pending snapshots to disk so container restarts do not orphan open trades."""
        try:
            os.makedirs(os.path.dirname(PENDING_SNAPSHOTS_FILE), exist_ok=True)
            with open(PENDING_SNAPSHOTS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "trade_snapshots": self.trade_snapshots,
                    "pending_by_symbol": self.pending_by_symbol
                }, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error saving pending snapshots to disk: {e}")

    def _load_learning_memory(self) -> Dict[str, Any]:
        default_state = {
            "total_trades_analyzed": 0,
            "total_losses_diagnosed": 0,
            "total_wins_recorded": 0,
            "win_rate_percent": 0.0,
            "root_cause_counts": {
                "CHOP_FALSE_BREAKOUT": 0,
                "COUNTER_TREND_EXHAUSTION_FAIL": 0,
                "TIGHT_SL_VOLATILITY_NOISE": 0,
                "SPREAD_SLIPPAGE_DRAG": 0,
                "PREMATURE_REVERSAL": 0,
                "OTHER": 0
            },
            # Model Weights (Base: 50.0, Max: 100.0, Min: 10.0)
            "model_weights": {
                "MSS_BREAKOUT": 70.0,
                "EMA_TREND_PULLBACK": 65.0,
                "SMC_OB_RETEST": 65.0,
                "FVG_IMBALANCE_FILL": 60.0,
                "FIBO_OTE_0618": 60.0,
                "LIQUIDITY_SWEEP": 75.0,
                "TREND_MOMENTUM_EXPANSION": 70.0,
                "BOLLINGER_REVERSION": 35.0
            },
            "model_penalties": {
                "M5_BOLLINGER_REVERSION": 0.0,
                "M5_EMA_PULLBACK": 0.0,
                "M5_BREAKOUT_MOMENTUM": 0.0,
                "M5_TURTLE_SOUP_SWEEP": 0.0,
                "M5_FVG_TAP": 0.0,
                "SWING_SMC_ORDERBLOCK": 0.0,
                "SWING_FIBO_OTE": 0.0
            },
            "adaptive_adx_min": 22.0,
            "adaptive_sl_multiplier": 1.0,
            "adaptive_tp1_mult": 1.0,
            "adaptive_tp2_mult": 1.0,
            "adaptive_tp3_mult": 1.0,
            "multi_target_split_enabled": True,
            "recent_post_mortems": [],
            "recent_reinforcements": [],

            # ---- Measured performance ledger (the part that actually learns) ----
            # Every closed trade is recorded in R-multiples (profit / money risked),
            # which is the only outcome measure comparable across lot sizes, stop
            # widths and account balances. Win/loss counts alone cannot tell you
            # whether a strategy makes money - a 37% win rate is excellent at 3R and
            # ruinous at 0.5R.
            "trade_ledger": [],          # rolling list of {r, model, session, regime, ...}
            "model_stats": {},           # model -> {n, wins, sum_r, disabled}
            "session_stats": {},         # UTC hour bucket -> {n, wins, sum_r}
            "regime_stats": {},          # market regime -> {n, wins, sum_r}
            "overall_expectancy_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "payoff_ratio": 0.0,
            "required_payoff_ratio": 0.0
        }
        try:
            if os.path.exists(LEARNING_MEMORY_FILE):
                with open(LEARNING_MEMORY_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    default_state.update(saved)
                    logger.info("Loaded AI Trade Learning Memory from disk.")
        except Exception as e:
            logger.warning(f"Could not load learning memory: {e}")
        return default_state

    def save_learning_memory(self):
        try:
            os.makedirs(os.path.dirname(LEARNING_MEMORY_FILE), exist_ok=True)
            with open(LEARNING_MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.learning_state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving learning memory: {e}")

    def classify_market_regime(
        self,
        adx: float,
        bandwidth: float,
        atr: float,
        avg_atr: float = 4.0,
        trend_alignment: str = "NEUTRAL"
    ) -> Dict[str, Any]:
        """
        Classifies the real-time market regime using quantitative multi-factor metrics.
        """
        min_adx = self.get_adaptive_adx_threshold()
        
        # High Volatility Spike
        if atr > (avg_atr * 1.8):
            regime = "HIGH_VOLATILITY_EXPANSION"
            regime_ar = "توسع عالي التقلب (حركة سعرية حادة / أخبار)"
            allowed_styles = ["BREAKOUT_MOMENTUM", "LIQUIDITY_SWEEP"]
            risk_adjustment = 0.8  # Reduce lot slightly for safety
            
        # Explosive Trend
        elif adx >= max(24.0, min_adx) and trend_alignment in ["BULLISH", "BEARISH"]:
            regime = "TRENDING_EXPANSION"
            regime_ar = "اتجاه صاروخي متسارع (تفعيل الركوب الاتجاهي والتجزئة الثلاثية)"
            allowed_styles = ["MSS_BREAKOUT", "EMA_TREND_PULLBACK", "TREND_MOMENTUM_EXPANSION", "SMC_OB_RETEST"]
            risk_adjustment = 1.0
            
        # Range / Equilibrium
        elif 18.0 <= adx < max(24.0, min_adx) or bandwidth >= 0.28:
            regime = "RANGE_BOUND"
            regime_ar = "نطاق عرضي متوازن (تفعيل مناطق الطلب والعرض والفيبوناتشي OTE)"
            allowed_styles = ["SMC_OB_RETEST", "FVG_IMBALANCE_FILL", "FIBO_OTE_0618", "LIQUIDITY_SWEEP"]
            risk_adjustment = 1.0
            
        # Stagnant Chop
        else:
            regime = "VOLATILE_CHOP"
            regime_ar = "تذبذب عرضي ضعيف الزخم (تفعيل وضع الأمان والانتظار HOLD)"
            allowed_styles = []
            risk_adjustment = 0.0

        return {
            "regime": regime,
            "regime_ar": regime_ar,
            "adx": round(adx, 1),
            "bandwidth": round(bandwidth, 3),
            "atr": round(atr, 2),
            "allowed_styles": allowed_styles,
            "risk_adjustment": risk_adjustment,
            "is_tradeable": (regime != "VOLATILE_CHOP")
        }

    def get_pip_multiplier(self, symbol: str) -> float:
        sym = symbol.upper()
        if "XAU" in sym or "GOLD" in sym:
            return 10.0
        elif "JPY" in sym:
            return 100.0
        return 10000.0

    def record_entry_snapshot(
        self,
        symbol: str,
        ticket: Optional[int],
        direction: str,
        lot_size: float,
        entry_price: float,
        sl: float,
        tp: float,
        analysis_context: Dict[str, Any],
        comment: str = ""
    ):
        """Captures a complete market state snapshot at the moment an order is dispatched/filled"""
        # Unique even for three split legs dispatched within the same second
        self._snapshot_seq = getattr(self, "_snapshot_seq", 0) + 1
        snapshot_id = (str(ticket) if ticket and ticket > 0
                       else f"{symbol}_{int(time.time())}_{self._snapshot_seq}")
        
        models_triggered = analysis_context.get("scalp_analysis", {}).get("scalp_models_triggered", [])
        if not models_triggered:
            models_triggered = [analysis_context.get("active_trade_type", analysis_context.get("classical_status", "HYBRID"))]

        pip_mult = self.get_pip_multiplier(symbol)
        raw_sl_dist = abs(entry_price - sl) if (sl > 0 and entry_price > 0) else 0.0
        digits = 2 if ("XAU" in symbol.upper() or "GOLD" in symbol.upper()) else (3 if "JPY" in symbol.upper() else 5)
        
        snapshot = {
            "snapshot_id": snapshot_id,
            "symbol": symbol,
            "ticket": ticket,
            "direction": direction.upper(),
            "lot_size": lot_size,
            "entry_price": round(entry_price, digits),
            "sl": round(sl, digits),
            "tp": round(tp, digits),
            "tp1": analysis_context.get("suggested_tp1", tp),
            "tp2": analysis_context.get("suggested_tp2", tp),
            "tp3": analysis_context.get("suggested_tp3", tp),
            "sl_distance": round(raw_sl_dist, digits),
            "sl_pips": round(raw_sl_dist * pip_mult, 1),
            "entry_time": time.time(),
            "entry_time_str": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "comment": comment,
            "adx": float(analysis_context.get("adx", 25.0)),
            "rsi": float(analysis_context.get("rsi", 50.0)),
            "atr": float(analysis_context.get("atr", 4.0 if "XAU" in symbol.upper() else 0.0020)),
            "macro_trend": analysis_context.get("macro_trend", analysis_context.get("market_context", "NEUTRAL")),
            "m15_trend": analysis_context.get("m15_trend", "NEUTRAL"),
            "market_regime": analysis_context.get("market_regime", {}).get("regime", "UNKNOWN"),
            "strategy_profile": analysis_context.get("strategy_profile", "HYBRID"),
            "models_triggered": models_triggered,
            "session_name": analysis_context.get("session_name", "UNKNOWN"),
            "dxy_bias": analysis_context.get("dxy_correlation", {}).get("bias", "NEUTRAL_DXY")
        }
        
        self.trade_snapshots[snapshot_id] = snapshot

        # Pending queue per symbol. A split order records three snapshots before any
        # ticket is known; keeping only "LATEST_<symbol>" meant all three closes were
        # diagnosed against the last one. The queue is bounded so unmatched snapshots
        # (rejected orders) cannot grow without limit.
        queue = self.pending_by_symbol.setdefault(symbol.upper(), [])
        queue.append(snapshot)
        del queue[:-self.MAX_PENDING_PER_SYMBOL]

        # Drop keyed snapshots older than 24h - previously nothing ever removed them.
        cutoff = time.time() - 86400
        for key in [k for k, v in self.trade_snapshots.items()
                    if isinstance(v, dict) and v.get("entry_time", 0) < cutoff]:
            self.trade_snapshots.pop(key, None)

        self._save_pending_snapshots()
        logger.info(f"📸 Black-Box Snapshot recorded for {symbol} {direction} (ID: {snapshot_id}) | SL: {snapshot['sl_pips']} pips | Models: {models_triggered}")

    def analyze_closed_trade(self, closed_trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Performs AI Forensic Analysis & Reinforcement Learning Feedback on a closed trade.
        """
        ticket = closed_trade.get("ticket")
        order_id = closed_trade.get("order")
        symbol = closed_trade.get("symbol", "XAUUSD")
        profit = float(closed_trade.get("profit", 0.0))
        is_loss = (profit < -0.05)
        pip_mult = self.get_pip_multiplier(symbol)
        is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
        
        # Look up the entry snapshot: exact ticket/order first, then the oldest
        # unmatched snapshot for this symbol (FIFO), so the three legs of a split
        # order are each diagnosed against their own entry context.
        snapshot = None
        if ticket and str(ticket) in self.trade_snapshots:
            snapshot = self.trade_snapshots.pop(str(ticket))
        elif order_id and str(order_id) in self.trade_snapshots:
            snapshot = self.trade_snapshots.pop(str(order_id))
        else:
            queue = self.pending_by_symbol.get(symbol.upper(), [])
            if queue:
                snapshot = queue.pop(0)

        if snapshot:
            self.trade_snapshots.pop(snapshot.get("snapshot_id", ""), None)
            for q in self.pending_by_symbol.values():
                if snapshot in q:
                    q.remove(snapshot)
            self._save_pending_snapshots()

        # File the outcome in R BEFORE the narrative diagnosis below. This is the
        # measurement the system actually learns from; the prose post-mortem is for
        # the dashboard.
        r_multiple = self.record_outcome(snapshot, closed_trade)

        self.learning_state["total_trades_analyzed"] += 1
        models = snapshot.get("models_triggered", []) if snapshot else []

        # ----------------------------------------------------
        # 🟢 WIN REINFORCEMENT (REWARD LOOP)
        # ----------------------------------------------------
        if not is_loss:
            self.learning_state["total_wins_recorded"] += 1
            tot = self.learning_state["total_trades_analyzed"]
            wins = self.learning_state["total_wins_recorded"]
            self.learning_state["win_rate_percent"] = round((wins / max(1, tot)) * 100.0, 1)

            # Decay existing penalties and gently normalize adaptive ADX
            for k in self.learning_state["model_penalties"]:
                self.learning_state["model_penalties"][k] = max(0.0, self.learning_state["model_penalties"][k] - 3.0)
            self.learning_state["adaptive_adx_min"] = max(20.0, round(self.learning_state.get("adaptive_adx_min", 22.0) - 0.5, 1))

            # Record Reinforcement Event
            reinforce_item = {
                "ticket": ticket,
                "type": "REWARD",
                "profit": profit,
                "symbol": symbol,
                "models": models,
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "note_ar": f"صفقة رابحة على {symbol} (+${profit:.2f}) -> مكافأة النماذج وتعزيز أوزانها في الذاكرة"
            }
            self.learning_state.setdefault("recent_reinforcements", []).insert(0, reinforce_item)
            self.learning_state["recent_reinforcements"] = self.learning_state["recent_reinforcements"][:10]

            self.save_learning_memory()
            brainstorm_service.add_log(
                level="EXECUTION",
                category="AI_LEARNING",
                symbol=symbol,
                message=f"🌟 [AI Reward] Deal #{ticket} Closed in PROFIT (+${profit:.2f}) on {symbol} -> Reinforcing Models ({models})"
            )
            return {
                "outcome": "WIN",
                "profit": profit,
                "symbol": symbol,
                "message": f"Winning trade on {symbol} (+${profit:.2f}) verified. Learning models reinforced."
            }

        # ----------------------------------------------------
        # 🔴 LOSS DIAGNOSIS & PENALTY (PUNISHMENT LOOP)
        # ----------------------------------------------------
        self.learning_state["total_losses_diagnosed"] += 1
        tot = self.learning_state["total_trades_analyzed"]
        wins = self.learning_state["total_wins_recorded"]
        self.learning_state["win_rate_percent"] = round((wins / max(1, tot)) * 100.0, 1)

        adx_at_entry = snapshot.get("adx", 25.0) if snapshot else 20.0
        atr_at_entry = snapshot.get("atr", 4.0 if is_gold else 0.0020) if snapshot else (4.0 if is_gold else 0.0020)
        sl_dist = snapshot.get("sl_distance", 2.50 if is_gold else 0.0025) if snapshot else (2.50 if is_gold else 0.0025)
        sl_pips = snapshot.get("sl_pips", sl_dist * pip_mult) if snapshot else (sl_dist * pip_mult)
        macro_trend = snapshot.get("macro_trend", "NEUTRAL") if snapshot else "NEUTRAL"
        direction = snapshot.get("direction", "BUY") if snapshot else "BUY"

        root_cause = "OTHER"
        root_cause_ar = "عوامل تقلب عامة"
        adaptation_action = ""
        adaptation_action_ar = ""

        # Case 1: Entered during Chop / Flat Range
        if adx_at_entry < 24.0:
            root_cause = "CHOP_FALSE_BREAKOUT"
            root_cause_ar = f"دخول في نطاق تذبذب أفقي ضعيف الزخم (ADX: {adx_at_entry:.1f} < 24.0) على {symbol}"
            self.learning_state["adaptive_adx_min"] = min(28.5, max(23.0, self.learning_state.get("adaptive_adx_min", 22.0) + 1.5))
            adaptation_action = f"Elevated minimum ADX filter to {self.learning_state['adaptive_adx_min']:.1f} to block flat chop."
            adaptation_action_ar = f"تم رفع حد فلتر الاتجاه (ADX) تلقائياً إلى {self.learning_state['adaptive_adx_min']:.1f} لمنع التداول في التذبذب."

        # Case 2: Counter-Trend Trade
        elif (direction == "SELL" and "BULL" in str(macro_trend).upper()) or (direction == "BUY" and "BEAR" in str(macro_trend).upper()):
            root_cause = "COUNTER_TREND_EXHAUSTION_FAIL"
            root_cause_ar = f"محاولة التقاط قمة/قاع معاكسة لاتجاه الفريم الأكبر ({macro_trend}) على {symbol}"
            self.learning_state["model_penalties"]["M5_BOLLINGER_REVERSION"] = min(40.0, self.learning_state["model_penalties"].get("M5_BOLLINGER_REVERSION", 0.0) + 15.0)
            adaptation_action = "Heavily penalized Counter-Trend Mean-Reversion models (-15 pts score)."
            adaptation_action_ar = "تم فرض عقوبة صارمة على نماذج الارتداد المعاكس للاتجاه (-15 نقطة) لمنع التداول ضد المسار العام."

        # Case 3: Stop loss too tight for current ATR
        elif atr_at_entry > 0 and sl_dist > 0 and sl_dist < (1.2 * atr_at_entry):
            root_cause = "TIGHT_SL_VOLATILITY_NOISE"
            if is_gold:
                root_cause_ar = f"وقف الخسارة كان ضيقاً جداً (${sl_dist:.2f}) مقارنة بتقلبات الذهب الحالية (ATR: ${atr_at_entry:.2f})"
            else:
                root_cause_ar = f"وقف الخسارة كان ضيقاً ({sl_pips:.1f} نقطة) مقارنة بتقلبات الزوج (ATR: {atr_at_entry * pip_mult:.1f} نقطة)"
            self.learning_state["adaptive_sl_multiplier"] = min(1.6, self.learning_state.get("adaptive_sl_multiplier", 1.0) + 0.15)
            adaptation_action = f"Increased adaptive SL multiplier to {self.learning_state['adaptive_sl_multiplier']:.2f}x ATR."
            adaptation_action_ar = f"تمت زيادة معامل الوقف تلقائياً إلى {self.learning_state['adaptive_sl_multiplier']:.2f}x لتفادي الضوضاء السعرية."

        # Case 4: Premature structural failure
        else:
            root_cause = "PREMATURE_REVERSAL"
            root_cause_ar = f"فشل اختراق الزخم اللحظي وانعكاس السعر قبل استكمال الهيكل على {symbol}"
            self.learning_state["model_penalties"]["M5_BREAKOUT_MOMENTUM"] = min(30.0, self.learning_state["model_penalties"].get("M5_BREAKOUT_MOMENTUM", 0.0) + 10.0)
            adaptation_action = "Applied 10-point penalty to Momentum breakout model."
            adaptation_action_ar = "تم خفض وزن نموذج اختراق الزخم بـ 10 نقاط لتأكيد الدخول بإعادة الاختبار فقط."

        # Update root cause counters
        if root_cause in self.learning_state["root_cause_counts"]:
            self.learning_state["root_cause_counts"][root_cause] += 1
        else:
            self.learning_state["root_cause_counts"]["OTHER"] += 1

        post_mortem_report = {
            "ticket": ticket,
            "symbol": symbol,
            "profit": profit,
            "direction": direction,
            "closed_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "root_cause": root_cause,
            "root_cause_ar": root_cause_ar,
            "adaptation_action": adaptation_action,
            "adaptation_action_ar": adaptation_action_ar,
            "adx_at_entry": round(adx_at_entry, 1),
            "atr_at_entry": round(atr_at_entry, 4),
            "sl_distance": round(sl_dist, 5),
            "sl_pips": round(sl_pips, 1),
            "models": models
        }

        # Keep last 25 post-mortems in memory
        recent = self.learning_state.get("recent_post_mortems", [])
        recent.insert(0, post_mortem_report)
        self.learning_state["recent_post_mortems"] = recent[:25]
        
        self.save_learning_memory()

        brainstorm_service.add_log(
            level="RISK",
            category="AI_LEARNING",
            symbol=symbol,
            message=f"🧠 [AI Loss Diagnosis] Deal #{ticket} on {symbol} (Loss: ${profit:.2f}) -> Cause: {root_cause_ar}. {adaptation_action_ar}"
        )

        logger.info(f"🧠 AI Post-Mortem on #{ticket} ({symbol}): Cause={root_cause} | Action={adaptation_action}")
        return post_mortem_report

    def calculate_multi_target_split(
        self,
        total_lot_size: float,
        sl_price: float,
        tp1_price: float,
        tp2_price: float,
        tp3_price: float,
        direction: str,
        symbol: str = "XAUUSD"
    ) -> List[Dict[str, Any]]:
        """
        Splits a single trade order into 3 coordinated sub-orders:
        Order 1 (35%): Fast TP1 (Locks in quick profit & triggers Coordinated Break-Even)
        Order 2 (35%): Mid-Term Structural TP2 (Captures swing move)
        Order 3 (30%): Runner TP3 (Captures major trend runner with dynamic trailing)
        """
        min_lot = 0.01
        lot_step = 0.01
        
        # If total lot is minimal (0.01 or 0.02), we cannot split into 3 without violating minimum lot
        if total_lot_size < 0.03:
            return [{
                "order_index": 1,
                "type_name": "PRIMARY_FULL",
                "lot_size": round(total_lot_size, 2),
                "sl": sl_price,
                "tp": tp1_price,
                "comment": "FXENGIN_SINGLE_AI"
            }]

        lot1 = round(max(min_lot, total_lot_size * 0.35), 2)
        lot2 = round(max(min_lot, total_lot_size * 0.35), 2)
        lot3 = round(max(min_lot, total_lot_size - lot1 - lot2), 2)

        # Re-adjust rounding difference
        current_sum = round(lot1 + lot2 + lot3, 2)
        if current_sum > total_lot_size:
            lot1 = round(lot1 - (current_sum - total_lot_size), 2)
        elif current_sum < total_lot_size:
            lot3 = round(lot3 + (total_lot_size - current_sum), 2)

        return [
            {
                "order_index": 1,
                "type_name": "SCALP_TP1",
                "lot_size": round(max(min_lot, lot1), 2),
                "sl": sl_price,
                "tp": tp1_price,
                "target_pips": round(abs(tp1_price - sl_price) * 10.0, 1),
                "comment": "FXENGIN_TP1_SCALP"
            },
            {
                "order_index": 2,
                "type_name": "SWING_TP2",
                "lot_size": round(max(min_lot, lot2), 2),
                "sl": sl_price,
                "tp": tp2_price,
                "target_pips": round(abs(tp2_price - sl_price) * 10.0, 1),
                "comment": "FXENGIN_TP2_SWING"
            },
            {
                "order_index": 3,
                "type_name": "RUNNER_TP3",
                "lot_size": round(max(min_lot, lot3), 2),
                "sl": sl_price,
                "tp": tp3_price,
                "target_pips": round(abs(tp3_price - sl_price) * 10.0, 1),
                "comment": "FXENGIN_TP3_RUNNER"
            }
        ]

    # =========================================================================
    # MEASURED PERFORMANCE LEDGER  (the part that actually learns)
    # =========================================================================
    @staticmethod
    def _normalise_model(raw: str) -> str:
        """Collapses a decorated model label into a stable key."""
        t = str(raw).upper()
        for key, token in (
            ("MSS_BREAKOUT", "MSS"),
            ("EMA_TREND_PULLBACK", "PULLBACK"),
            ("SMC_OB_RETEST", "OB"),
            ("FVG_IMBALANCE_FILL", "FVG"),
            ("FIBO_OTE_0618", "FIBO"),
            ("LIQUIDITY_SWEEP", "SWEEP"),
            ("TREND_MOMENTUM_EXPANSION", "IMPULSE"),
            ("BOLLINGER_REVERSION", "BOLLINGER"),
        ):
            if token in t:
                return key
        return "OTHER"

    def record_outcome(self, snapshot: Optional[Dict[str, Any]], closed_trade: Dict[str, Any]) -> Optional[float]:
        """
        Converts a closed trade into an R-multiple and files it in the ledger.

        R = realised profit / money that was at risk when the trade was opened.

        Expressing outcomes in R is what makes "is this model any good?" answerable:
        R is invariant to lot size, stop width and account balance, so trades taken
        weeks apart under different settings stay directly comparable. Win/loss
        counts cannot do this - a 37% win rate is excellent at 3R and ruinous at 0.5R,
        and counting wins is exactly how this bot concluded it was doing fine while
        losing money.
        """
        profit = float(closed_trade.get("profit", 0.0))
        fee = float(closed_trade.get("fee", 0.0) or 0.0)
        net = profit + fee

        if not snapshot:
            return None

        entry = float(snapshot.get("entry_price", 0.0) or 0.0)
        sl = float(snapshot.get("sl", 0.0) or 0.0)
        lot = float(snapshot.get("lot_size", 0.0) or 0.0)
        symbol = snapshot.get("symbol", closed_trade.get("symbol", "XAUUSD"))

        if entry <= 0 or sl <= 0 or lot <= 0:
            return None

        pip_mult = self.get_pip_multiplier(symbol)
        pip_value = 10.0
        risk_usd = abs(entry - sl) * pip_mult * pip_value * lot
        if risk_usd <= 0:
            return None

        r = round(net / risk_usd, 3)

        entry_hour = None
        try:
            entry_hour = datetime.fromtimestamp(
                float(snapshot.get("entry_time", 0)), tz=timezone.utc
            ).hour
        except Exception:
            pass

        models = sorted({self._normalise_model(m) for m in (snapshot.get("models_triggered") or ["OTHER"])})

        record = {
            "ticket": closed_trade.get("ticket"),
            "symbol": symbol,
            "direction": snapshot.get("direction", ""),
            "r": r,
            "net_usd": round(net, 2),
            "risk_usd": round(risk_usd, 2),
            "models": models,
            "regime": snapshot.get("market_regime", "UNKNOWN"),
            "hour_utc": entry_hour,
            "adx": snapshot.get("adx"),
            "closed_at": datetime.now(timezone.utc).isoformat()
        }

        ledger = self.learning_state.setdefault("trade_ledger", [])
        ledger.append(record)
        del ledger[:-self.MAX_LEDGER]

        is_win = r > 0
        model_bucket = self.learning_state.setdefault("model_stats", {})
        for k in models:
            st = model_bucket.setdefault(k, {"n": 0, "wins": 0, "sum_r": 0.0, "disabled": False})
            st["n"] += 1
            st["wins"] += 1 if is_win else 0
            st["sum_r"] = round(st["sum_r"] + r, 3)

        for bucket_name, k in (("regime_stats", record["regime"]),
                               ("session_stats", str(entry_hour))):
            if k in (None, "None", ""):
                continue
            st = self.learning_state.setdefault(bucket_name, {}).setdefault(
                k, {"n": 0, "wins": 0, "sum_r": 0.0})
            st["n"] += 1
            st["wins"] += 1 if is_win else 0
            st["sum_r"] = round(st["sum_r"] + r, 3)

        self._recompute_aggregates()
        self._retune_from_evidence()
        return r

    def _recompute_aggregates(self):
        """Recomputes the headline statistics that decide whether this is viable."""
        ledger = self.learning_state.get("trade_ledger", [])
        if not ledger:
            return
        rs = [t["r"] for t in ledger]
        wins = [r for r in rs if r > 0]
        losses = [abs(r) for r in rs if r <= 0]

        self.learning_state["overall_expectancy_r"] = round(sum(rs) / len(rs), 3)
        avg_win = round(sum(wins) / len(wins), 3) if wins else 0.0
        avg_loss = round(sum(losses) / len(losses), 3) if losses else 0.0
        self.learning_state["avg_win_r"] = avg_win
        self.learning_state["avg_loss_r"] = avg_loss

        wr = len(wins) / len(rs)
        self.learning_state["payoff_ratio"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0
        self.learning_state["required_payoff_ratio"] = round((1 - wr) / wr, 2) if wr > 0 else 999.0

    def _retune_from_evidence(self):
        """
        Adjusts model weights from MEASURED expectancy, not from the last outcome.

        A model is judged only once it has MIN_SAMPLES_FOR_JUDGEMENT closed trades,
        and its weight is then set from its average R rather than nudged by a fixed
        step. Reacting to single outcomes (the previous behaviour: -8 on a loss, +5
        on a win) is noise-chasing: it moves weights fastest exactly when the sample
        is smallest and least trustworthy.
        """
        stats = self.learning_state.get("model_stats", {})
        weights = self.learning_state.setdefault("model_weights", {})

        for model, st in stats.items():
            if model == "OTHER" or st["n"] < self.MIN_SAMPLES_FOR_JUDGEMENT:
                continue
            exp_r = st["sum_r"] / st["n"]

            # Expectancy in R mapped onto the 10-98 weight scale used by the scorers:
            # -0.5R -> 10, 0R -> 50, +0.5R -> 90, clamped at both ends.
            weights[model] = round(max(10.0, min(98.0, 50.0 + exp_r * 80.0)), 1)

            should_disable = exp_r <= self.DISABLE_EXPECTANCY_R
            if should_disable and not st.get("disabled"):
                st["disabled"] = True
                brainstorm_service.add_log(
                    level="RISK", category="AI_LEARNING", symbol="SYSTEM",
                    message=(f"Model {model} RETIRED: measured expectancy {exp_r:+.2f}R over "
                             f"{st['n']} trades (win rate {st['wins'] / st['n'] * 100:.0f}%).")
                )
            elif not should_disable and st.get("disabled"):
                st["disabled"] = False
                brainstorm_service.add_log(
                    level="INFO", category="AI_LEARNING", symbol="SYSTEM",
                    message=f"Model {model} reinstated: expectancy recovered to {exp_r:+.2f}R over {st['n']} trades."
                )

    def is_model_enabled(self, model_name: str) -> bool:
        """False once a model has proven, over a real sample, that it loses money."""
        key = self._normalise_model(model_name)
        st = self.learning_state.get("model_stats", {}).get(key)
        return not (st and st.get("disabled"))

    def get_performance_report(self) -> Dict[str, Any]:
        """Everything needed to judge whether the system is actually working."""
        ledger = self.learning_state.get("trade_ledger", [])
        exp = self.learning_state.get("overall_expectancy_r", 0.0)
        models = {}
        for m, st in self.learning_state.get("model_stats", {}).items():
            if st["n"] <= 0:
                continue
            e = st["sum_r"] / st["n"]
            models[m] = {
                "trades": st["n"],
                "win_rate": round(st["wins"] / st["n"] * 100, 1),
                "expectancy_r": round(e, 3),
                "total_r": round(st["sum_r"], 2),
                "disabled": st.get("disabled", False),
                "verdict": ("not enough data" if st["n"] < self.MIN_SAMPLES_FOR_JUDGEMENT
                            else "profitable" if e > 0.05 else "losing" if e < 0 else "flat")
            }
        return {
            "trades_recorded": len(ledger),
            "expectancy_r": exp,
            "expectancy_verdict": ("profitable" if exp > 0.05 else "losing" if exp < 0 else "flat"),
            "avg_win_r": self.learning_state.get("avg_win_r", 0.0),
            "avg_loss_r": self.learning_state.get("avg_loss_r", 0.0),
            "payoff_ratio": self.learning_state.get("payoff_ratio", 0.0),
            "required_payoff_ratio": self.learning_state.get("required_payoff_ratio", 0.0),
            "models": models,
            "by_session_hour": self.learning_state.get("session_stats", {}),
            "by_regime": self.learning_state.get("regime_stats", {}),
            "min_samples_before_judgement": self.MIN_SAMPLES_FOR_JUDGEMENT
        }

    def get_model_weight(self, model_name: str) -> float:
        """Returns the dynamic learned score weight for an entry style"""
        return self.learning_state.get("model_weights", {}).get(model_name, 50.0)

    def get_model_penalty(self, model_name: str) -> float:
        """Returns the dynamic learned penalty for a specific strategy model"""
        return self.learning_state.get("model_penalties", {}).get(model_name, 0.0)

    def get_adaptive_adx_threshold(self) -> float:
        """Returns the dynamically learned ADX threshold to prevent chop entries"""
        return self.learning_state.get("adaptive_adx_min", 22.0)

    def get_adaptive_sl_multiplier(self) -> float:
        """Returns the dynamically learned SL distance multiplier"""
        return self.learning_state.get("adaptive_sl_multiplier", 1.0)

    def is_multi_target_enabled(self) -> bool:
        return self.learning_state.get("multi_target_split_enabled", True)

    def set_multi_target_enabled(self, enabled: bool):
        self.learning_state["multi_target_split_enabled"] = bool(enabled)
        self.save_learning_memory()

    def get_learning_summary(self) -> Dict[str, Any]:
        """Provides full AI learning analytics for Dashboard"""
        return {
            "total_analyzed": self.learning_state.get("total_trades_analyzed", 0),
            "total_wins": self.learning_state.get("total_wins_recorded", 0),
            "total_losses": self.learning_state.get("total_losses_diagnosed", 0),
            "win_rate": self.learning_state.get("win_rate_percent", 0.0),
            "adaptive_adx": round(self.learning_state.get("adaptive_adx_min", 22.0), 1),
            "adaptive_sl_mult": round(self.learning_state.get("adaptive_sl_multiplier", 1.0), 2),
            "model_weights": self.learning_state.get("model_weights", {}),
            "root_causes": self.learning_state.get("root_cause_counts", {}),
            "multi_target_enabled": self.learning_state.get("multi_target_split_enabled", True),
            "recent_diagnoses": self.learning_state.get("recent_post_mortems", [])[:5],
            "recent_reinforcements": self.learning_state.get("recent_reinforcements", [])[:5],
            "performance": self.get_performance_report()
        }

    def score_setup_with_learning(
        self,
        symbol: str,
        direction: str,
        model_name: str,
        base_confidence: float,
        current_regime: str = "TRENDING_EXPANSION",
        entry_hour: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Dynamically adjusts setup confidence and validation using learned model weights,
        hourly session statistics, and market regime performance from the measured trade ledger.
        """
        norm_model = self._normalise_model(model_name)
        model_wt = self.get_model_weight(norm_model) # 10.0 to 98.0
        model_pen = self.get_model_penalty(model_name)
        is_enabled = self.is_model_enabled(norm_model)

        # Baseline weight adjustment: 50 is neutral, >50 boosts confidence, <50 reduces confidence
        weight_delta = (model_wt - 50.0) * 0.20

        # Session Expectancy Adjustment
        session_delta = 0.0
        if entry_hour is not None:
            session_stat = self.learning_state.get("session_stats", {}).get(str(entry_hour), {})
            session_n = session_stat.get("n", 0)
            if session_n >= 5:
                session_exp = session_stat.get("sum_r", 0.0) / session_n
                session_delta = max(-6.0, min(6.0, session_exp * 8.0))

        # Regime Adjustment
        regime_stat = self.learning_state.get("regime_stats", {}).get(current_regime, {})
        regime_n = regime_stat.get("n", 0)
        regime_delta = 0.0
        if regime_n >= 5:
            regime_exp = regime_stat.get("sum_r", 0.0) / regime_n
            regime_delta = max(-6.0, min(6.0, regime_exp * 8.0))

        final_confidence = round(max(25.0, min(95.0, base_confidence + weight_delta - model_pen + session_delta + regime_delta)), 1)

        return {
            "model_enabled": is_enabled,
            "final_confidence": final_confidence,
            "model_weight": model_wt,
            "penalty_applied": model_pen,
            "weight_delta": round(weight_delta, 1),
            "session_delta": round(session_delta, 1),
            "regime_delta": round(regime_delta, 1)
        }

trade_learning_engine = TradeLearningEngine()
