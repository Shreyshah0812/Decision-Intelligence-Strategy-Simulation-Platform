"""
core/profiler.py
Automatic data profiling — schema detection, statistics, target inference, drift detection.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional


# --------------------------------------------------------------------------- #
# SCHEMA DETECTION
# --------------------------------------------------------------------------- #

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify every column as: id | datetime | numeric | categorical | boolean | target_candidate
    """
    col_types = {}
    for col in df.columns:
        series = df[col].dropna()
        n_unique = series.nunique()
        n_total = len(series)

        # Boolean
        if set(series.unique()).issubset({0, 1, True, False, "0", "1"}):
            col_types[col] = "boolean"
            continue

        # Try datetime parse
        if df[col].dtype == object:
            try:
                pd.to_datetime(series.head(100), infer_datetime_format=True)
                col_types[col] = "datetime"
                continue
            except Exception:
                pass

        # ID heuristic: unique ratio > 0.95 and not numeric-continuous
        if n_unique / max(n_total, 1) > 0.95 and df[col].dtype == object:
            col_types[col] = "id"
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            if n_unique <= 2:
                col_types[col] = "boolean"
            elif n_unique <= 20 and n_unique / n_total < 0.05:
                col_types[col] = "categorical"
            else:
                col_types[col] = "numeric"
            continue

        # Categorical
        if n_unique / max(n_total, 1) < 0.10:
            col_types[col] = "categorical"
        else:
            col_types[col] = "text"

    return col_types


# --------------------------------------------------------------------------- #
# TARGET INFERENCE
# --------------------------------------------------------------------------- #

TARGET_KEYWORDS = [
    "churn", "target", "label", "outcome", "converted", "conversion",
    "default", "fraud", "purchased", "clicked", "subscribed", "renewed",
    "revenue", "sales", "profit", "mrr", "arr", "ltv", "clv",
    "price", "amount", "value", "score", "rating", "satisfaction",
    "status", "result", "response", "flag", "indicator"
]


def infer_target(df: pd.DataFrame, col_types: Dict[str, str]) -> Tuple[Optional[str], str, float]:
    """
    Returns (target_column, problem_type, confidence).
    problem_type in: 'classification', 'regression', 'time_series'
    """
    candidates = []

    for col in df.columns:
        ctype = col_types.get(col, "")
        if ctype in ("id", "datetime", "text"):
            continue

        col_lower = col.lower().replace("_", "").replace("-", "").replace(" ", "")
        score = 0.0

        # Keyword match
        for kw in TARGET_KEYWORDS:
            if kw in col_lower:
                score += 0.4
                break

        # Boolean / binary → likely classification target
        if ctype == "boolean":
            score += 0.3
        elif ctype == "numeric":
            score += 0.1

        # Not all-null
        null_pct = df[col].isnull().mean()
        if null_pct < 0.05:
            score += 0.1

        # Reasonable variance
        if ctype == "numeric" and df[col].std() > 0:
            score += 0.1

        candidates.append((col, score, ctype))

    if not candidates:
        return None, "unknown", 0.0

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_col, best_score, best_ctype = candidates[0]
    confidence = min(best_score, 1.0)

    # Determine problem type
    n_unique = df[best_col].nunique()
    if best_ctype == "boolean" or n_unique <= 5:
        problem_type = "classification"
    elif best_ctype == "numeric":
        # Check if there's a datetime column → could be time series
        has_datetime = any(v == "datetime" for v in col_types.values())
        if has_datetime:
            problem_type = "time_series"
        else:
            problem_type = "regression"
    else:
        problem_type = "classification"

    return best_col, problem_type, confidence


# --------------------------------------------------------------------------- #
# STATISTICAL PROFILING
# --------------------------------------------------------------------------- #

def profile_column(series: pd.Series) -> Dict[str, Any]:
    """Full stats for one column."""
    s = series.dropna()
    result = {
        "count": len(series),
        "missing": int(series.isnull().sum()),
        "missing_pct": round(series.isnull().mean() * 100, 2),
        "n_unique": s.nunique(),
        "unique_ratio": round(s.nunique() / max(len(s), 1), 4),
    }

    if pd.api.types.is_numeric_dtype(series):
        result.update({
            "mean": round(float(s.mean()), 4) if len(s) else None,
            "median": round(float(s.median()), 4) if len(s) else None,
            "std": round(float(s.std()), 4) if len(s) else None,
            "min": round(float(s.min()), 4) if len(s) else None,
            "max": round(float(s.max()), 4) if len(s) else None,
            "skewness": round(float(s.skew()), 4) if len(s) > 2 else None,
            "outliers_iqr": _count_outliers_iqr(s),
        })
    else:
        top_val = s.value_counts().index[0] if len(s) > 0 else None
        top_freq = s.value_counts().iloc[0] if len(s) > 0 else None
        result.update({
            "top_value": str(top_val) if top_val is not None else None,
            "top_freq": int(top_freq) if top_freq is not None else None,
            "top_freq_pct": round(top_freq / max(len(s), 1) * 100, 2) if top_freq else None,
        })

    return result


def _count_outliers_iqr(s: pd.Series) -> int:
    if len(s) < 4:
        return 0
    try:
        s = s.astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    except Exception:
        return 0


def full_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Run full profiling on entire dataframe."""
    col_types = detect_column_types(df)
    col_profiles = {col: profile_column(df[col]) for col in df.columns}

    # Correlation matrix (numeric only)
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    corr_matrix = df[num_cols].corr().round(3).to_dict() if len(num_cols) >= 2 else {}

    return {
        "shape": df.shape,
        "col_types": col_types,
        "col_profiles": col_profiles,
        "corr_matrix": corr_matrix,
        "numeric_cols": num_cols,
        "categorical_cols": [c for c, t in col_types.items() if t == "categorical"],
        "datetime_cols": [c for c, t in col_types.items() if t == "datetime"],
        "boolean_cols": [c for c, t in col_types.items() if t == "boolean"],
        "total_missing_pct": round(df.isnull().mean().mean() * 100, 2),
    }


# --------------------------------------------------------------------------- #
# DRIFT / ANOMALY DETECTION
# --------------------------------------------------------------------------- #

def detect_drift(df: pd.DataFrame, col_types: Dict[str, str], datetime_col: Optional[str] = None) -> List[Dict]:
    """
    Detect statistical shifts. If a datetime col is present, split first half vs second half.
    Otherwise compare top quantile vs bottom quantile.
    Returns list of findings.
    """
    findings = []
    numeric_cols = [c for c, t in col_types.items() if t in ("numeric", "boolean") and c != datetime_col]

    if not numeric_cols:
        return findings

    # Split strategy
    if datetime_col:
        try:
            df = df.copy()
            df[datetime_col] = pd.to_datetime(df[datetime_col], infer_datetime_format=True)
            df_sorted = df.sort_values(datetime_col)
            mid = len(df_sorted) // 2
            group_a = df_sorted.iloc[:mid]
            group_b = df_sorted.iloc[mid:]
            split_label = "early period vs recent period"
        except Exception:
            mid = len(df) // 2
            group_a, group_b = df.iloc[:mid], df.iloc[mid:]
            split_label = "first half vs second half"
    else:
        mid = len(df) // 2
        group_a, group_b = df.iloc[:mid], df.iloc[mid:]
        split_label = "first half vs second half"

    for col in numeric_cols[:8]:  # Limit to 8 columns
        a = group_a[col].dropna()
        b = group_b[col].dropna()
        if len(a) < 20 or len(b) < 20:
            continue

        # t-test
        t_stat, p_val = stats.ttest_ind(a, b)
        mean_a, mean_b = a.mean(), b.mean()
        pct_change = (mean_b - mean_a) / max(abs(mean_a), 1e-9) * 100

        if p_val < 0.05:
            severity = "high" if p_val < 0.01 else "medium"
            direction = "increased" if mean_b > mean_a else "decreased"
            findings.append({
                "column": col,
                "finding": f"{col} {direction} by {abs(pct_change):.1f}% ({split_label})",
                "p_value": round(p_val, 4),
                "mean_before": round(mean_a, 4),
                "mean_after": round(mean_b, 4),
                "pct_change": round(pct_change, 2),
                "severity": severity,
            })

    findings.sort(key=lambda x: x["p_value"])
    return findings
