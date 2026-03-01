"""
core/simulator.py
Strategy simulation engine — perturb features, predict delta, rank by ROI.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------------------- #
# SIMULATE A SINGLE INTERVENTION
# --------------------------------------------------------------------------- #

def simulate_intervention(
    X: pd.DataFrame,
    pipeline: Pipeline,
    feature_col: str,
    delta_pct: float,
    problem_type: str,
    y_actual: pd.Series,
    revenue_col: Optional[str] = None,
    df_full: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Apply delta_pct change to feature_col, re-predict, measure outcome change.
    """
    X_mod = X.copy()

    if feature_col not in X_mod.columns:
        return {}

    # Apply perturbation
    if pd.api.types.is_numeric_dtype(X_mod[feature_col]):
        X_mod[feature_col] = X_mod[feature_col] * (1 + delta_pct / 100)
    else:
        # Can't meaningfully perturb categorical — skip
        return {}

    # Predictions
    if problem_type == "classification":
        try:
            orig_proba = pipeline.predict_proba(X)[:, 1]
            new_proba = pipeline.predict_proba(X_mod)[:, 1]
        except Exception:
            return {}

        orig_mean = orig_proba.mean()
        new_mean = new_proba.mean()
        outcome_delta = new_mean - orig_mean

        # Revenue impact
        revenue_impact = None
        if revenue_col and df_full is not None and revenue_col in df_full.columns:
            rev = df_full[revenue_col].fillna(0)
            saved_customers = int(abs(outcome_delta) * len(X))
            avg_rev = rev.mean()
            revenue_impact = saved_customers * avg_rev
        else:
            # Proxy: assume $100 per unit change in probability
            saved_customers = int(abs(outcome_delta) * len(X))
            revenue_impact = saved_customers * 100  # placeholder

        return {
            "feature": feature_col,
            "delta_pct": delta_pct,
            "orig_outcome": round(float(orig_mean), 4),
            "new_outcome": round(float(new_mean), 4),
            "outcome_delta": round(float(outcome_delta), 4),
            "outcome_delta_pct": round(outcome_delta / max(abs(orig_mean), 1e-9) * 100, 2),
            "saved_customers": saved_customers,
            "revenue_impact": round(float(revenue_impact), 2),
            "confidence": round(1 - np.std(new_proba - orig_proba), 4),
        }

    else:  # regression
        try:
            orig_pred = pipeline.predict(X)
            new_pred = pipeline.predict(X_mod)
        except Exception:
            return {}

        orig_mean = orig_pred.mean()
        new_mean = new_pred.mean()
        outcome_delta = new_mean - orig_mean

        return {
            "feature": feature_col,
            "delta_pct": delta_pct,
            "orig_outcome": round(float(orig_mean), 4),
            "new_outcome": round(float(new_mean), 4),
            "outcome_delta": round(float(outcome_delta), 4),
            "outcome_delta_pct": round(outcome_delta / max(abs(orig_mean), 1e-9) * 100, 2),
            "saved_customers": None,
            "revenue_impact": round(float(outcome_delta * len(X)), 2),
            "confidence": round(1 - abs(np.std(new_pred - orig_pred) / max(abs(orig_mean), 1e-9)), 4),
        }


# --------------------------------------------------------------------------- #
# GENERATE STRATEGY CANDIDATES
# --------------------------------------------------------------------------- #

def generate_strategies(
    model_result: Dict[str, Any],
    top_n_features: int = 4,
    delta_options: List[float] = None,
    revenue_col: Optional[str] = None,
    df_full: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    For each of the top N features, simulate +10% and -10% interventions.
    Return ranked list of strategies.
    """
    if delta_options is None:
        delta_options = [10.0, -10.0, 20.0, -20.0]

    pipeline = model_result["pipeline"]
    X = model_result["X"]
    y = model_result["y"]
    problem_type = model_result["problem_type"]
    feat_importance = model_result["feature_importance"]

    if feat_importance.empty:
        return []

    top_features = feat_importance["feature"].head(top_n_features).tolist()

    strategy_map = {
        "positive_classification": {
            10: ("Increase {feature} by 10%", "Moderate effort — expected to reduce risk by improving {feature} in the target population."),
            20: ("Increase {feature} by 20%", "Higher effort intervention. Aggressive improvement of {feature} via targeted campaign."),
            -10: ("Reduce {feature} by 10%", "Reduce {feature} friction or volume, lowering risk signals."),
            -20: ("Reduce {feature} by 20%", "Aggressive reduction in {feature} — high effort, high reward if feasible."),
        },
        "positive_regression": {
            10: ("Increase {feature} by 10%", "Incremental improvement to {feature} expected to lift outcome."),
            20: ("Increase {feature} by 20%", "Strong push on {feature} — higher cost, higher upside."),
            -10: ("Decrease {feature} by 10%", "Reduce {feature} to optimize outcomes."),
            -20: ("Decrease {feature} by 20%", "Significant {feature} reduction — evaluate feasibility carefully."),
        }
    }

    all_strategies = []

    for feat in top_features:
        feat_rank = feat_importance[feat_importance["feature"] == feat].index[0]
        importance_score = float(feat_importance.loc[feat_rank, "importance"])

        for delta in delta_options:
            result = simulate_intervention(
                X=X,
                pipeline=pipeline,
                feature_col=feat,
                delta_pct=delta,
                problem_type=problem_type,
                y_actual=y,
                revenue_col=revenue_col,
                df_full=df_full,
            )
            if not result:
                continue

            # Only keep strategies with meaningful impact
            if abs(result.get("outcome_delta_pct", 0)) < 0.5:
                continue

            # Determine if this direction is beneficial
            if problem_type == "classification":
                beneficial = result["outcome_delta"] < 0  # Lower churn probability = good
                impact_abs = abs(result["outcome_delta_pct"])
            else:
                beneficial = result["outcome_delta"] > 0  # Higher value = good
                impact_abs = abs(result["outcome_delta_pct"])

            # Strategy name
            direction = "Increase" if delta > 0 else "Decrease"
            abs_delta = abs(int(delta))
            name = f"{direction} {feat} by {abs_delta}%"
            desc = f"{'Improve' if delta > 0 else 'Reduce'} {feat} by {abs_delta}% through targeted intervention on the most at-risk population segment."

            # Estimated cost proxy (importance × effort)
            cost_proxy = abs_delta * importance_score * 5000

            # ROI
            rev_impact = abs(result.get("revenue_impact", 0))
            roi = ((rev_impact - cost_proxy) / max(cost_proxy, 1)) * 100 if cost_proxy > 0 else 0

            # Composite score
            confidence = max(min(result.get("confidence", 0.7), 1.0), 0.0)
            risk_score = 1.0 - confidence
            time_score = 1.0 - (abs_delta / 40.0)  # lower delta = faster
            composite = (
                0.40 * min(impact_abs / 20, 1.0)  # Financial impact
                + 0.20 * (1 - risk_score)          # Risk
                + 0.20 * time_score                 # Time to implement
                + 0.20 * confidence                 # Confidence
            ) * 100

            all_strategies.append({
                "name": name,
                "description": desc,
                "feature": feat,
                "delta_pct": delta,
                "feature_importance": round(importance_score, 4),
                "outcome_delta_pct": result["outcome_delta_pct"],
                "orig_outcome": result["orig_outcome"],
                "new_outcome": result["new_outcome"],
                "revenue_impact": rev_impact,
                "estimated_cost": round(cost_proxy, 2),
                "roi_pct": round(roi, 1),
                "confidence": round(confidence * 100, 1),
                "risk": "Low" if risk_score < 0.2 else ("Medium" if risk_score < 0.4 else "High"),
                "composite_score": round(composite, 1),
                "beneficial": beneficial,
            })

    # Sort: beneficial first, then by composite score
    all_strategies.sort(key=lambda x: (-x["beneficial"], -x["composite_score"]))
    return all_strategies


# --------------------------------------------------------------------------- #
# CUSTOM SIMULATION (for slider UI)
# --------------------------------------------------------------------------- #

def run_custom_simulation(
    model_result: Dict[str, Any],
    feature_deltas: Dict[str, float],
    revenue_col: Optional[str] = None,
    df_full: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Apply multiple feature deltas simultaneously and return combined prediction delta.
    """
    pipeline = model_result["pipeline"]
    X = model_result["X"].copy()
    problem_type = model_result["problem_type"]

    # Apply all deltas
    for feat, delta_pct in feature_deltas.items():
        if feat in X.columns and pd.api.types.is_numeric_dtype(X[feat]):
            X[feat] = X[feat] * (1 + delta_pct / 100)

    X_orig = model_result["X"]

    try:
        if problem_type == "classification":
            orig_proba = pipeline.predict_proba(X_orig)[:, 1]
            new_proba = pipeline.predict_proba(X)[:, 1]
            orig_mean = orig_proba.mean()
            new_mean = new_proba.mean()
            delta = new_mean - orig_mean
            delta_pct = delta / max(abs(orig_mean), 1e-9) * 100
            customers_affected = int(abs(delta) * len(X))

            if revenue_col and df_full is not None and revenue_col in df_full.columns:
                avg_rev = df_full[revenue_col].fillna(0).mean()
                rev_impact = customers_affected * avg_rev
            else:
                rev_impact = customers_affected * 100

            return {
                "orig_outcome": round(float(orig_mean * 100), 2),
                "new_outcome": round(float(new_mean * 100), 2),
                "outcome_delta": round(float(delta * 100), 2),
                "outcome_delta_pct": round(float(delta_pct), 2),
                "customers_affected": customers_affected,
                "revenue_impact": round(float(rev_impact), 2),
                "confidence": round(float(1 - np.std(new_proba - orig_proba)), 3),
                "label": "Predicted Probability",
            }
        else:
            orig_pred = pipeline.predict(X_orig)
            new_pred = pipeline.predict(X)
            orig_mean = orig_pred.mean()
            new_mean = new_pred.mean()
            delta = new_mean - orig_mean
            delta_pct = delta / max(abs(orig_mean), 1e-9) * 100

            return {
                "orig_outcome": round(float(orig_mean), 4),
                "new_outcome": round(float(new_mean), 4),
                "outcome_delta": round(float(delta), 4),
                "outcome_delta_pct": round(float(delta_pct), 2),
                "customers_affected": len(X),
                "revenue_impact": round(float(delta * len(X)), 2),
                "confidence": round(float(1 - abs(np.std(new_pred) / max(abs(orig_mean), 1e-9))), 3),
                "label": "Predicted Value",
            }
    except Exception as e:
        return {"error": str(e)}
