import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Search paths for model artifacts
MODEL_SEARCH_DIRS = [
    Path("/app/ml/models"),
    Path(__file__).resolve().parent.parent.parent / "ml" / "models",
    Path(__file__).resolve().parent.parent.parent.parent / "ml" / "models",
    Path("/root/ml/models")
]

def get_symbol_metrics(symbol: str, side: str = "LONG") -> Dict[str, Any]:
    """
    Retrieves measured out-of-sample performance and Tier for a given symbol and side.
    Per Agent Brief 14 & 20:
    - If side is SHORT / SELL, reads from production_{symbol}_short_features.json
    - If side is LONG / BUY, reads from production_{symbol}_features.json
    - Falls back to production_{symbol}_report.json if present.
    - If unmeasured/untested, returns EXPERIMENTAL tier with real or neutral stats.
    - NEVER serves one symbol's numbers as another's.
    """
    sym = symbol.upper()
    side_norm = "SHORT" if str(side).upper() in ["SHORT", "SELL"] else "LONG"
    
    # 1. Search for features JSON
    feat_data = None
    for mdir in MODEL_SEARCH_DIRS:
        if not mdir.exists():
            continue
        
        # Check side-specific features file
        if side_norm == "SHORT":
            target = mdir / f"production_{sym}_short_features.json"
        else:
            target = mdir / f"production_{sym}_features.json"

        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    feat_data = json.load(f)
                    break
            except Exception as e:
                logger.warning(f"Failed to load {target}: {e}")
                
        # XAUUSD fallback to legacy name if needed (long only)
        if sym == "XAUUSD" and side_norm == "LONG":
            target_legacy = mdir / "production_features.json"
            if target_legacy.exists():
                try:
                    with open(target_legacy, "r", encoding="utf-8") as f:
                        feat_data = json.load(f)
                        break
                except Exception as e:
                    logger.warning(f"Failed to load {target_legacy}: {e}")

    # If features JSON has the "measured" block, extract directly
    if feat_data and isinstance(feat_data.get("measured"), dict):
        m = feat_data["measured"]
        pub = m.get("published_rule") if isinstance(m.get("published_rule"), dict) else {}

        # Brief 18 Section 2b: read card and message baselines from published_rule block if present
        exp_r = float(pub.get("expectancy_r") if "expectancy_r" in pub else m.get("expectancy_r", 0.0))
        oos_trades = int(pub.get("oos_trades") if "oos_trades" in pub else m.get("oos_trades", 0))
        folds_pos = int(pub.get("folds_positive") if "folds_positive" in pub else m.get("folds_positive", 0))
        folds_tot = int(pub.get("folds_total") if "folds_total" in pub else m.get("folds_total", 5))

        raw_tier = str(m.get("tier", "EXPERIMENTAL")).upper()
        if exp_r < 0:
            tier = "EXPERIMENTAL"
        elif raw_tier == "VALIDATED":
            tier = "VALIDATED"
        else:
            ship = bool(m.get("passed_ship_gate", exp_r > 0.05))
            tier = "VALIDATED" if ship else "EXPERIMENTAL"

        return {
            "symbol": sym,
            "expectancy_r": round(exp_r, 4),
            "baseline_expectancy_r": round(exp_r, 4),
            "expectancy_std": round(float(m.get("expectancy_std", 0.0)), 4),
            "oos_trades": oos_trades,
            "baseline_trades": oos_trades,
            "folds_positive": folds_pos,
            "folds_total": folds_tot,
            "worst_fold_r": round(float(m.get("worst_fold_r", 0.0)), 4),
            "passed_ship_gate": (tier == "VALIDATED"),
            "tier": tier,
            "threshold": float(feat_data.get("threshold", 0.40)),
            "source": "features_json"
        }

    # 2. Search for report JSON
    rep_data = None
    for mdir in MODEL_SEARCH_DIRS:
        if not mdir.exists():
            continue
        if side_norm == "SHORT":
            rep_target = mdir / f"production_{sym}_short_report.json"
        else:
            rep_target = mdir / f"production_{sym}_report.json"
        if rep_target.exists():
            try:
                with open(rep_target, "r", encoding="utf-8") as f:
                    rep_data = json.load(f)
                    break
            except Exception as e:
                logger.warning(f"Failed to load {rep_target}: {e}")
                
        if sym == "XAUUSD":
            rep_legacy = mdir / "production_report.json"
            if rep_legacy.exists():
                try:
                    with open(rep_legacy, "r", encoding="utf-8") as f:
                        rep_data = json.load(f)
                        break
                except Exception as e:
                    logger.warning(f"Failed to load {rep_legacy}: {e}")

    if rep_data:
        exp_r = float(rep_data.get("expectancy_r_mean", 0.0))
        exp_std = float(rep_data.get("expectancy_r_std", 0.0))
        trades = int(rep_data.get("trades_oos_mean", 0))
        folds = rep_data.get("per_fold_expectancy", {})
        pos_folds = sum(1 for v in folds.values() if float(v) > 0)
        tot_folds = len(folds) if folds else 5
        worst = min([float(v) for v in folds.values()]) if folds else 0.0
        ship = exp_r > 0.05 and pos_folds >= 4
        tier = "VALIDATED" if ship else "EXPERIMENTAL"
        return {
            "symbol": sym,
            "expectancy_r": round(exp_r, 4),
            "expectancy_std": round(exp_std, 4),
            "oos_trades": trades,
            "folds_positive": pos_folds,
            "folds_total": tot_folds,
            "worst_fold_r": round(worst, 4),
            "passed_ship_gate": ship,
            "tier": tier,
            "source": "report_json"
        }

    # 3. Default fallback for unmeasured symbols
    return {
        "symbol": sym,
        "expectancy_r": 0.0,
        "baseline_expectancy_r": 0.0,
        "expectancy_std": 0.0,
        "oos_trades": 0,
        "baseline_trades": 0,
        "folds_positive": 0,
        "folds_total": 5,
        "worst_fold_r": 0.0,
        "passed_ship_gate": False,
        "tier": "EXPERIMENTAL",
        "threshold": 0.40,
        "source": "unmeasured_default"
    }


def format_caption_baseline(metrics: Dict[str, Any]) -> str:
    """
    Brief 14 Section 2: Formats the baseline block for Message 2.
    For EXPERIMENTAL:
        EXPERIMENTAL — measured -0.027 R over 2,569 out-of-sample trades,
        2 of 5 walk-forward folds positive. Published for evaluation only.
    For VALIDATED:
        Baseline: +0.07 R expectancy over 6,468 out-of-sample
        walk-forward trades, spread accounted, slippage not modelled.
        Tier: VALIDATED
    """
    tier = metrics.get("tier", "EXPERIMENTAL")
    exp_r = metrics.get("expectancy_r", 0.0)
    trades = metrics.get("oos_trades", 0)
    pos_folds = metrics.get("folds_positive", 0)
    tot_folds = metrics.get("folds_total", 5)

    if tier == "EXPERIMENTAL":
        return (
            f"EXPERIMENTAL — measured {exp_r:+.3f} R over {trades:,} out-of-sample trades,\n"
            f"{pos_folds} of {tot_folds} walk-forward folds positive. Published for evaluation only."
        )
    
    return (
        f"Baseline: {exp_r:+.2f} R expectancy over {trades:,} out-of-sample\n"
        f"walk-forward trades, spread accounted, slippage not modelled.\n"
        f"Tier: VALIDATED"
    )
