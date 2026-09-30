"""
Dual-Side Production Signal Engine Service (Agent Brief 20).
Manages both LONG and SHORT model registries, 3-state H4 gate selection,
and mirrored level generation.
"""
from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from app.config import settings

logger = logging.getLogger(__name__)

SL_ATR = 1.0
TP_ATR = 2.0
MAX_SPREAD_ATR = 0.15
MIN_BASE_BARS = 1_100
MIN_HTF_BARS = 260

# Search directories for model artifacts
MODEL_SEARCH_DIRS = [
    Path("/app/ml/models"),
    Path(__file__).resolve().parent.parent.parent / "ml" / "models",
    Path(__file__).resolve().parent.parent.parent.parent / "ml" / "models",
    Path("/root/ml/models")
]

ML_CODE_DIRS = [
    Path("/app/ml"),
    Path(__file__).resolve().parent.parent.parent / "ml",
    Path(__file__).resolve().parent.parent.parent.parent / "ml",
    Path("/root/ml")
]

# Ensure ml directory is on sys.path to load features and features_expert
for p in ML_CODE_DIRS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


class ModelEntry:
    def __init__(
        self,
        symbol: str,
        side: str,
        cfg_path: Path,
        model_path: Path
    ):
        self.symbol = symbol.upper()
        self.side = side.upper()  # "LONG" or "SHORT"
        self.cfg_path = cfg_path
        self.model_path = model_path

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.features: List[str] = cfg["features"]
        self.threshold: float = float(cfg.get("threshold", 0.40))
        self.timeframe: str = cfg.get("timeframe", "M15")

        gate = cfg.get("gate", {})
        self.gate_col: str = gate.get("column", "h4_ema_stack")
        self.gate_val = gate.get("equals", 2 if self.side == "LONG" else 0)

        self.measured: Dict[str, Any] = cfg.get("measured", {})
        self.booster = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))


class SignalEngineService:
    # Brief 25 Section 1: Ten models across eight publishing symbols
    PUBLISHED_MODEL_DIRECTIONS: Dict[str, List[str]] = {
        "USDCHF": ["SHORT"],
        "AUDJPY": ["SHORT"],
        "AUDUSD": ["SHORT"],
        "CADJPY": ["SHORT"],
        "USDCAD": ["LONG", "SHORT"],
        "CHFJPY": ["SHORT"],
        "USDJPY": ["LONG", "SHORT"],
        "GBPJPY": ["LONG"],
    }

    def __init__(self):
        self.long_models: Dict[str, ModelEntry] = {}
        self.short_models: Dict[str, ModelEntry] = {}
        self.eval_long_models: Dict[str, ModelEntry] = {}
        self.eval_short_models: Dict[str, ModelEntry] = {}
        self.published_directions: Dict[str, List[str]] = {}
        self._load_all_models()

    def _find_models_dir(self) -> Optional[Path]:
        for d in MODEL_SEARCH_DIRS:
            if d.exists() and any(d.glob("production_*.txt")):
                return d
        return None

    def _load_all_models(self):
        models_dir = self._find_models_dir()
        if not models_dir:
            logger.warning("SignalEngineService: No model directory with production_*.txt files found.")
            return

        manifest_path = models_dir / "PUBLISH_MANIFEST.json"
        if manifest_path.exists():
            # Brief 25 Section 2b: Build the registry from the manifest, NOT by globbing the directory
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            models_list = manifest_data.get("models", [])
            self.published_directions = {}
            for item in models_list:
                sym = item["symbol"].upper()
                side = item["side"].upper()  # "LONG" or "SHORT"
                model_fname = item["model_file"]
                config_fname = item["config_file"]

                txt_file = models_dir / model_fname
                cfg_file = models_dir / config_fname

                if not txt_file.exists():
                    raise RuntimeError(f"Manifest required model file missing: {txt_file}")
                if not cfg_file.exists():
                    raise RuntimeError(f"Manifest required config file missing: {cfg_file}")

                entry = ModelEntry(sym, side, cfg_file, txt_file)
                if side == "LONG":
                    self.long_models[sym] = entry
                else:
                    self.short_models[sym] = entry
                self.published_directions.setdefault(sym, []).append(side)

            # Shadow evaluation models for excluded symbols (EURUSD, GBPUSD, EURJPY)
            # so they continue evaluating in decision log without publishing to Telegram
            excluded = [s.upper() for s in getattr(settings, "TELEGRAM_EXCLUDED_SYMBOLS", ["XAUUSD", "GBPUSD", "EURJPY", "EURUSD"])]
            for sym in excluded:
                cfg_l = models_dir / f"production_{sym}_features.json"
                txt_l = models_dir / f"production_{sym}_long.txt"
                if cfg_l.exists() and txt_l.exists():
                    try:
                        self.eval_long_models[sym] = ModelEntry(sym, "LONG", cfg_l, txt_l)
                    except Exception as e:
                        logger.warning(f"Could not load shadow LONG model for {sym}: {e}")
                cfg_s = models_dir / f"production_{sym}_short_features.json"
                txt_s = models_dir / f"production_{sym}_short.txt"
                if cfg_s.exists() and txt_s.exists():
                    try:
                        self.eval_short_models[sym] = ModelEntry(sym, "SHORT", cfg_s, txt_s)
                    except Exception as e:
                        logger.warning(f"Could not load shadow SHORT model for {sym}: {e}")

            logger.info(
                f"SignalEngineService (Manifest-driven): Loaded {len(self.long_models)} LONG models "
                f"and {len(self.short_models)} SHORT models ({list(self.short_models.keys())}) across "
                f"{len(self.published_directions)} publishing symbols from {manifest_path.name}."
            )
            return

        # Fallback if manifest is missing (legacy)
        for cfg_file in sorted(models_dir.glob("production_*_features.json")):
            name = cfg_file.name
            if name.endswith("_short_features.json"):
                continue
            sym = name[len("production_"):-len("_features.json")].upper()
            txt_file = models_dir / f"production_{sym}_long.txt"
            if txt_file.exists():
                try:
                    entry = ModelEntry(sym, "LONG", cfg_file, txt_file)
                    self.long_models[sym] = entry
                except Exception as e:
                    logger.error(f"Error loading LONG model for {sym}: {e}")

        for cfg_file in sorted(models_dir.glob("production_*_short_features.json")):
            name = cfg_file.name
            sym = name[len("production_"):-len("_short_features.json")].upper()
            txt_file = models_dir / f"production_{sym}_short.txt"
            if txt_file.exists():
                try:
                    entry = ModelEntry(sym, "SHORT", cfg_file, txt_file)
                    self.short_models[sym] = entry
                except Exception as e:
                    logger.error(f"Error loading SHORT model for {sym}: {e}")

    def reload(self):
        self.long_models.clear()
        self.short_models.clear()
        self.eval_long_models.clear()
        self.eval_short_models.clear()
        self.published_directions.clear()
        self._load_all_models()

    def get_available_symbols(self) -> List[str]:
        """All symbols with either a long or a short model in active manifest."""
        return sorted(list(set(list(self.long_models.keys()) + list(self.short_models.keys()))))

    def get_available_short_symbols(self) -> List[str]:
        """Symbols with a deployed SHORT model in active manifest."""
        return sorted(list(self.short_models.keys()))

    def get_available_long_symbols(self) -> List[str]:
        """Symbols with a deployed LONG model in active manifest."""
        return sorted(list(self.long_models.keys()))

    def is_symbol_supported(self, symbol: str) -> bool:
        sym = symbol.upper()
        return (
            sym in self.long_models or
            sym in self.short_models or
            (sym in settings.TELEGRAM_EXCLUDED_SYMBOLS and (sym in self.eval_long_models or sym in self.eval_short_models))
        )

    def is_direction_supported(self, symbol: str, direction: str) -> bool:
        """Checks if a given direction is active and publishing for the symbol."""
        sym = symbol.upper()
        dir_norm = direction.upper()
        if sym in settings.TELEGRAM_EXCLUDED_SYMBOLS:
            # Allow evaluation in decision log if model exists on disk
            if dir_norm == "LONG":
                return sym in self.long_models or sym in self.eval_long_models
            return sym in self.short_models or sym in self.eval_short_models
        allowed = self.published_directions.get(sym, self.PUBLISHED_MODEL_DIRECTIONS.get(sym, []))
        return dir_norm in allowed

    def get_preferred_side(self, symbol: str) -> Optional[str]:
        """
        Brief 25 Section 2:
        Rule 2c (Brief 21) is removed. Symbols with both sides publish both.
        Returns None to indicate no single-direction restriction.
        """
        return None

    def build_features(
        self,
        candles: Dict[str, List[Dict[str, Any]]],
        point: float,
        digits: int,
        timeframe: str = "M15"
    ) -> Optional[pd.DataFrame]:
        try:
            from features import add_features, htf_context, merge_htf
            from features_expert import add_all_expert_features
        except ImportError as ie:
            logger.error(f"Failed to import feature modules from ml/: {ie}")
            return None

        base = candles.get(timeframe)
        if not base or len(base) < MIN_BASE_BARS:
            return None

        df = pd.DataFrame(base)
        if "datetime" not in df:
            df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if "spread" not in df:
            df["spread"] = 0.0
        # Agent Brief 27 Section 1: Fix the 'tick_volume' KeyError
        if "tick_volume" not in df:
            if "volume" in df:
                df["tick_volume"] = df["volume"]
            else:
                df["tick_volume"] = 100.0
        df["tick_volume"] = df["tick_volume"].fillna(100.0)
        # Ensure non-zero float values to avoid division by zero or NaN in indicators
        df["tick_volume"] = df["tick_volume"].astype(float).replace(0.0, 100.0)

        feat = add_features(df, spread_point=point)
        feat = add_all_expert_features(feat, digits=digits)

        for tf, prefix in (("H1", "h1"), ("H4", "h4")):
            rows = candles.get(tf)
            if not rows or len(rows) < MIN_HTF_BARS:
                return None
            h = pd.DataFrame(rows)
            if "datetime" not in h:
                h["datetime"] = pd.to_datetime(h["time"], unit="s", utc=True)
            if "tick_volume" not in h:
                if "volume" in h:
                    h["tick_volume"] = h["volume"]
                else:
                    h["tick_volume"] = 100.0
            h["tick_volume"] = h["tick_volume"].fillna(100.0)
            h["tick_volume"] = h["tick_volume"].astype(float).replace(0.0, 100.0)
            feat = merge_htf(feat, htf_context(h, prefix))

        return feat

    def decide(
        self,
        symbol: str,
        candles: Dict[str, List[Dict[str, Any]]],
        point: float = 0.01,
        digits: int = 2,
        live_price: Optional[float] = None,
        force_h4_stack: Optional[float] = None,
        force_prob: Optional[float] = None,
        force_atr: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the symbol against incoming candles using 3-state H4 gate:
          h4_ema_stack == 2 and long model exists  -> evaluate long model
          h4_ema_stack == 0 and short model exists -> evaluate short model
          otherwise                                -> HOLD
        """
        sym = symbol.upper()
        has_long = sym in self.long_models or (sym in settings.TELEGRAM_EXCLUDED_SYMBOLS and sym in self.eval_long_models)
        has_short = sym in self.short_models or (sym in settings.TELEGRAM_EXCLUDED_SYMBOLS and sym in self.eval_short_models)

        if not has_long and not has_short:
            return self._hold(f"no validated model for {sym}", gate_passed=False)

        if force_h4_stack is not None:
            stack_num = float(force_h4_stack)
            row = None
        else:
            feat = self.build_features(candles, point, digits, timeframe="M15")
            if feat is None or len(feat) == 0:
                return self._hold(
                    f"insufficient candles for feature building (need {MIN_BASE_BARS} M15, {MIN_HTF_BARS} H1/H4)",
                    gate_passed=False
                )

            row = feat.iloc[[-1]]
            if "h4_ema_stack" not in row.columns:
                return self._hold("missing h4_ema_stack feature", gate_passed=False)

            stack_val = row["h4_ema_stack"].iloc[0]
            if pd.isna(stack_val):
                return self._hold("indicators still warming up", gate_passed=False)

            stack_num = float(stack_val)

        # 3-State Gate Selection (Brief 20 Section 1 & 3)
        target_side: Optional[str] = None
        chosen_model: Optional[ModelEntry] = None

        if stack_num == 2.0:
            if has_long and self.is_direction_supported(sym, "LONG"):
                target_side = "LONG"
                chosen_model = self.long_models.get(sym) or self.eval_long_models.get(sym)
            else:
                return self._hold(
                    f"H4 uptrend (h4_ema_stack=2.0) but no active long model exists for {sym}",
                    gate_passed=False
                )
        elif stack_num == 0.0:
            if has_short and self.is_direction_supported(sym, "SHORT"):
                target_side = "SHORT"
                chosen_model = self.short_models.get(sym) or self.eval_short_models.get(sym)
            else:
                return self._hold(
                    f"H4 downtrend (h4_ema_stack=0.0) but no active short model exists for {sym}",
                    gate_passed=False
                )
        elif stack_num == 1.0:
            return self._hold(
                "H4 trend not aligned (h4_ema_stack=1.0, needs 2 for LONG or 0 for SHORT)",
                gate_passed=False
            )
        else:
            return self._hold(
                f"H4 trend not aligned (h4_ema_stack={stack_num})",
                gate_passed=False
            )

        if force_prob is not None:
            proba = float(force_prob)
            atr = float(force_atr or 0.100)
            spread_price = 1.0 * point
        else:
            # Check model features readiness
            missing = [c for c in chosen_model.features if c not in row.columns]
            if missing:
                return self._hold(f"missing {len(missing)} features, e.g. {missing[:3]}", gate_passed=False)
            if row[chosen_model.features].isna().any(axis=1).iloc[0]:
                return self._hold("indicators still warming up", gate_passed=False)

            atr = float(row["atr"].iloc[0])
            if not np.isfinite(atr) or atr <= 0:
                return self._hold("ATR unavailable", gate_passed=False)

            spread_price = float(row["spread"].iloc[0]) * point
            if spread_price / atr > MAX_SPREAD_ATR:
                return self._hold(
                    f"spread {spread_price / atr:.1%} of ATR exceeds {MAX_SPREAD_ATR:.0%} limit",
                    gate_passed=False
                )

            # Model Inference
            proba = float(chosen_model.booster.predict(row[chosen_model.features].to_numpy())[0])

        threshold = chosen_model.threshold

        if proba < threshold:
            return self._hold(
                f"probability {proba:.3f} below {threshold}",
                gate_passed=True,
                probability=round(proba, 4),
                threshold=threshold,
                side=target_side
            )

        # Construct Confirmed Signal Levels
        if live_price and live_price > 0:
            entry = float(live_price)
        elif row is not None and "close" in row:
            entry = float(row["close"].iloc[0])
        else:
            entry = 100.0
        
        if target_side == "LONG":
            decision = "BUY"
            sl = round(entry - SL_ATR * atr, digits)
            tp = round(entry + TP_ATR * atr, digits)
            tp2 = round(entry + 3.0 * atr, digits)
            risk_price = round(entry - sl, digits)
        else:  # SHORT
            decision = "SELL"
            sl = round(entry + SL_ATR * atr, digits)   # Stop above entry
            tp = round(entry - TP_ATR * atr, digits)   # Target 1 below entry
            tp2 = round(entry - 3.0 * atr, digits)  # Target 2 below entry
            risk_price = round(sl - entry, digits)

        is_retired = bool(sym in settings.TELEGRAM_EXCLUDED_SYMBOLS)
        if is_retired:
            logger.info(
                f"ℹ️ [EXCLUDED_SYMBOL] {sym} {target_side} model evaluated for decision log, but symbol is excluded from Telegram."
            )

        return {
            "decision": decision,
            "side": target_side,
            "preferred_side": None,
            "is_retired": is_retired,
            "probability": round(proba, 4),
            "threshold": threshold,
            "entry": round(entry, digits),
            "sl": sl,
            "tp": tp,
            "tp2": tp2,
            "risk_price": risk_price,
            "rr": round(TP_ATR / SL_ATR, 1),
            "atr": round(atr, digits),
            "spread_cost_r": round(spread_price / risk_price, 4) if risk_price > 0 else 0.0,
            "gate_passed": True,
            "reason": f"H4 aligned and model probability {proba:.3f} >= {threshold}" if not is_retired else f"{sym} evaluated for decision log (Telegram publishing excluded)",
            "measured": chosen_model.measured
        }

    @staticmethod
    def _hold(reason: str, **extra) -> Dict[str, Any]:
        res = {"decision": "HOLD", "reason": reason}
        res.update(extra)
        return res


signal_engine_service = SignalEngineService()
