"""
core/model_builder.py
Automatic ML pipeline: preprocessing → multi-model training → best model selection → SHAP.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# --------------------------------------------------------------------------- #
# PREPROCESSING
# --------------------------------------------------------------------------- #

def build_preprocessor(df: pd.DataFrame, feature_cols: List[str], col_types: Dict[str, str]) -> Tuple[ColumnTransformer, List[str]]:
    num_cols = [c for c in feature_cols if col_types.get(c) in ("numeric", "boolean")]
    cat_cols = [c for c in feature_cols if col_types.get(c) == "categorical"]

    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    feature_names_out = num_cols + cat_cols
    return preprocessor, feature_names_out


def prepare_data(df: pd.DataFrame, target_col: str, col_types: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return X, y, feature_col_list — drop id/datetime/text/target columns."""
    drop_types = {"id", "datetime", "text"}
    feature_cols = [
        c for c in df.columns
        if c != target_col and col_types.get(c) not in drop_types
    ]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode target if classification
    if y.dtype == object or (y.nunique() <= 10 and y.dtype != float):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)

    return X, y, feature_cols


# --------------------------------------------------------------------------- #
# MODEL CANDIDATES
# --------------------------------------------------------------------------- #

def get_candidates(problem_type: str) -> List[Tuple[str, Any]]:
    candidates = []

    if problem_type == "classification":
        candidates.append(("Logistic Regression", LogisticRegression(max_iter=500, random_state=42)))
        candidates.append(("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)))
        candidates.append(("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)))
        if HAS_XGB:
            candidates.append(("XGBoost", XGBClassifier(n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")))
        if HAS_LGB:
            candidates.append(("LightGBM", LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)))
    else:
        candidates.append(("Ridge Regression", Ridge(random_state=42) if hasattr(Ridge, "random_state") else Ridge()))
        candidates.append(("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
        candidates.append(("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)))
        if HAS_XGB:
            candidates.append(("XGBoost", XGBRegressor(n_estimators=100, random_state=42, verbosity=0)))
        if HAS_LGB:
            candidates.append(("LightGBM", LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)))

    return candidates


# --------------------------------------------------------------------------- #
# MAIN BUILD FUNCTION
# --------------------------------------------------------------------------- #

def build_model(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    col_types: Dict[str, str],
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Full AutoML pipeline. Returns dict with best model, metrics, feature importance, SHAP values.
    """
    X, y, feature_cols = prepare_data(df, target_col, col_types)
    preprocessor, feat_names = build_preprocessor(X, feature_cols, col_types)
    candidates = get_candidates(problem_type)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if problem_type == "classification" else KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "roc_auc" if problem_type == "classification" else "r2"

    results = []
    total = len(candidates)

    for i, (name, model) in enumerate(candidates):
        if progress_callback:
            progress_callback(i / total, f"Training {name}...")

        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            results.append({
                "name": name,
                "model": model,
                "score": scores.mean(),
                "score_std": scores.std(),
                "pipe": pipe,
            })
        except Exception as e:
            continue

    if not results:
        raise ValueError("All models failed. Check your data.")

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]

    if progress_callback:
        progress_callback(0.85, f"Fitting best model: {best['name']}...")

    # Fit on full data
    best["pipe"].fit(X, y)

    # Feature importance
    feat_importance = _get_feature_importance(best["pipe"], feat_names, problem_type)

    # SHAP values (sample for speed)
    shap_values = None
    shap_df = None
    if HAS_SHAP:
        try:
            X_transformed = best["pipe"]["preprocessor"].transform(X)
            sample_size = min(300, len(X_transformed))
            X_sample = X_transformed[:sample_size]

            model_obj = best["pipe"]["model"]
            if hasattr(model_obj, "feature_importances_"):
                explainer = shap.TreeExplainer(model_obj)
                shap_vals = explainer.shap_values(X_sample)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                shap_df = pd.DataFrame(
                    np.abs(shap_vals).mean(axis=0).reshape(1, -1),
                    columns=feat_names[:X_sample.shape[1]]
                )
        except Exception:
            shap_df = None

    # All model comparison
    model_comparison = [
        {
            "Model": r["name"],
            "Score": round(r["score"], 4),
            "Std": round(r["score_std"], 4),
        }
        for r in results
    ]

    if progress_callback:
        progress_callback(1.0, "Done!")

    return {
        "best_model_name": best["name"],
        "pipeline": best["pipe"],
        "score": best["score"],
        "score_std": best["score_std"],
        "scoring_metric": scoring,
        "problem_type": problem_type,
        "feature_cols": feature_cols,
        "feat_names": feat_names,
        "feature_importance": feat_importance,
        "shap_df": shap_df,
        "model_comparison": model_comparison,
        "X": X,
        "y": y,
    }


def _get_feature_importance(pipe: Pipeline, feat_names: List[str], problem_type: str) -> pd.DataFrame:
    model = pipe["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    n = min(len(importances), len(feat_names))
    df = pd.DataFrame({
        "feature": feat_names[:n],
        "importance": importances[:n],
    })
    df["importance"] = df["importance"] / df["importance"].sum()
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
