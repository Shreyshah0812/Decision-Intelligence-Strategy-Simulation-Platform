"""
core/profiler.py
Automatic data profiling — schema detection, statistics, target inference, drift detection.
Handles real-world messy datasets: string booleans, mixed types, bad dates, nulls.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #

def _safe_float_series(s: pd.Series) -> Optional[pd.Series]:
    """Convert series to float safely. Returns None if not possible."""
    try:
        return pd.to_numeric(s, errors="coerce").dropna().astype(float)
    except Exception:
        return None


def _is_string_boolean(series: pd.Series) -> bool:
    """Check if a column contains string TRUE/FALSE/Yes/No/0/1 values."""
    try:
        s = series.dropna().astype(str).str.strip().str.upper()
        bool_vals = {"TRUE", "FALSE", "YES", "NO", "0", "1", "T", "F", "Y", "N"}
        unique_vals = set(s.unique())
        return len(unique_vals) >= 1 and unique_vals.issubset(bool_vals)
    except Exception:
        return False


def _is_datetime_col(series: pd.Series) -> bool:
    """Safely check if a column looks like dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype != object:
        return False
    try:
        sample = series.dropna().head(50).astype(str)
        parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
        return parsed.notna().mean() > 0.7
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# SCHEMA DETECTION
# --------------------------------------------------------------------------- #

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify every column as: id | datetime | numeric | categorical | boolean | text
    Handles string TRUE/FALSE, mixed types, messy real-world data.
    """
    col_types = {}

    for col in df.columns:
        series = df[col]
        s_clean = series.dropna()
        n_unique = s_clean.nunique()
        n_total = max(len(s_clean), 1)

        # Empty column
        if len(s_clean) == 0:
            col_types[col] = "categorical"
            continue

        # Native boolean dtype
        if pd.api.types.is_bool_dtype(series):
            col_types[col] = "boolean"
            continue

        # String TRUE/FALSE/Yes/No
        if series.dtype == object and _is_string_boolean(series):
            col_types[col] = "boolean"
            continue

        # Native numeric
        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= 2:
                col_types[col] = "boolean"
            elif n_unique <= 15 and (n_unique / n_total) < 0.05:
                col_types[col] = "categorical"
            else:
                col_types[col] = "numeric"
            continue

        # Object column
        if series.dtype == object:
            if _is_datetime_col(series):
                col_types[col] = "datetime"
                continue

            unique_ratio = n_unique / n_total
            col_lower = col.lower()
            is_id_name = any(kw in col_lower for kw in ["id", "order", "code", "key", "uuid", "sku", "asin", "index"])
            if unique_ratio > 0.90 or (unique_ratio > 0.70 and is_id_name):
                col_types[col] = "id"
                continue

            if n_unique <= 30 or (n_unique / n_total) < 0.05:
                col_types[col] = "categorical"
                continue

            avg_len = s_clean.astype(str).str.len().mean()
            if avg_len > 50:
                col_types[col] = "text"
                continue

            col_types[col] = "categorical"
            continue

        col_types[col] = "categorical"

    return col_types


# --------------------------------------------------------------------------- #
# TARGET INFERENCE
# --------------------------------------------------------------------------- #

TARGET_KEYWORDS = [
    "churn", "target", "label", "outcome", "converted", "conversion",
    "default", "fraud", "purchased", "clicked", "subscribed", "renewed",
    "revenue", "sales", "profit", "mrr", "arr", "ltv", "clv",
    "price", "amount", "value", "score", "rating", "satisfaction",
    "status", "result", "response", "flag", "indicator", "qty", "quantity"
]

ANTI_TARGET_KEYWORDS = [
    "id", "order", "city", "state", "country", "postal", "zip", "code",
    "name", "address", "email", "phone", "sku", "asin", "index",
    "date", "time", "currency", "channel", "promotion", "unnamed"
]


def infer_target(df: pd.DataFrame, col_types: Dict[str, str]) -> Tuple[Optional[str], str, float]:
    """Returns (target_column, problem_type, confidence)."""
    candidates = []

    for col in df.columns:
        ctype = col_types.get(col, "")
        if ctype in ("id", "datetime", "text"):
            continue

        col_lower = col.lower().replace("_", "").replace("-", "").replace(" ", "")
        is_anti = any(kw in col_lower for kw in ANTI_TARGET_KEYWORDS)
        if is_anti:
            continue

        score = 0.0
        for kw in TARGET_KEYWORDS:
            if kw in col_lower:
                score += 0.4
                break

        if ctype == "boolean":
            score += 0.3
        elif ctype == "numeric":
            score += 0.15
        elif ctype == "categorical":
            score += 0.05

        null_pct = df[col].isnull().mean()
        if null_pct < 0.05:
            score += 0.1
        elif null_pct > 0.3:
            score -= 0.2

        if ctype == "numeric":
            try:
                if pd.to_numeric(df[col], errors="coerce").std() > 0:
                    score += 0.1
            except Exception:
                pass

        candidates.append((col, score, ctype))

    if not candidates:
        for col in df.columns:
            ctype = col_types.get(col, "")
            if ctype in ("numeric", "boolean"):
                return col, "classification" if ctype == "boolean" else "regression", 0.3
        return None, "unknown", 0.0

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_col, best_score, best_ctype = candidates[0]
    confidence = min(best_score, 1.0)

    n_unique = df[best_col].nunique()
    if best_ctype == "boolean" or n_unique <= 5:
        problem_type = "classification"
    elif best_ctype == "categorical":
        problem_type = "classification"
    elif best_ctype == "numeric":
        has_datetime = any(v == "datetime" for v in col_types.values())
        problem_type = "time_series" if has_datetime else "regression"
    else:
        problem_type = "classification"

    return best_col, problem_type, confidence


# --------------------------------------------------------------------------- #
# STATISTICAL PROFILING
# --------------------------------------------------------------------------- #

def profile_column(series: pd.Series) -> Dict[str, Any]:
    """Full stats for one column. Fully safe — never raises."""
    try:
        s = series.dropna()
        result = {
            "count": len(series),
            "missing": int(series.isnull().sum()),
            "missing_pct": round(series.isnull().mean() * 100, 2),
            "n_unique": int(s.nunique()),
            "unique_ratio": round(s.nunique() / max(len(s), 1), 4),
        }

        # Numeric stats — only for actual numeric dtype, always cast to float
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            num_s = s.astype(float)
            if len(num_s) > 0:
                result.update({
                    "mean":         _safe_round(num_s.mean()),
                    "median":       _safe_round(num_s.median()),
                    "std":          _safe_round(num_s.std()),
                    "min":          _safe_round(num_s.min()),
                    "max":          _safe_round(num_s.max()),
                    "skewness":     _safe_round(num_s.skew()) if len(num_s) > 2 else None,
                    "outliers_iqr": _count_outliers_iqr(num_s),
                })
        else:
            try:
                vc = s.value_counts()
                top_val  = vc.index[0]  if len(vc) > 0 else None
                top_freq = int(vc.iloc[0]) if len(vc) > 0 else None
                result.update({
                    "top_value":    str(top_val) if top_val is not None else None,
                    "top_freq":     top_freq,
                    "top_freq_pct": round(top_freq / max(len(s), 1) * 100, 2) if top_freq else None,
                })
            except Exception:
                result.update({"top_value": None, "top_freq": None, "top_freq_pct": None})

        return result

    except Exception as e:
        return {
            "count": len(series),
            "missing": int(series.isnull().sum()),
            "missing_pct": round(series.isnull().mean() * 100, 2),
            "n_unique": 0,
            "unique_ratio": 0,
            "error": str(e),
        }


def _safe_round(val, digits=4):
    try:
        return round(float(val), digits)
    except Exception:
        return None


def _count_outliers_iqr(s: pd.Series) -> int:
    """Count outliers using IQR. Always works on float series."""
    try:
        if len(s) < 4:
            return 0
        s = s.astype(float)
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            return 0
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return int(((s < lower) | (s > upper)).sum())
    except Exception:
        return 0


def full_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Run full profiling. Never crashes."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_types = detect_column_types(df)

    col_profiles = {}
    for col in df.columns:
        col_profiles[col] = profile_column(df[col])

    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    try:
        corr_matrix = (
            df[num_cols].apply(pd.to_numeric, errors="coerce").corr().round(3).to_dict()
            if len(num_cols) >= 2 else {}
        )
    except Exception:
        corr_matrix = {}

    return {
        "shape":             df.shape,
        "col_types":         col_types,
        "col_profiles":      col_profiles,
        "corr_matrix":       corr_matrix,
        "numeric_cols":      num_cols,
        "categorical_cols":  [c for c, t in col_types.items() if t == "categorical"],
        "datetime_cols":     [c for c, t in col_types.items() if t == "datetime"],
        "boolean_cols":      [c for c, t in col_types.items() if t == "boolean"],
        "id_cols":           [c for c, t in col_types.items() if t == "id"],
        "text_cols":         [c for c, t in col_types.items() if t == "text"],
        "total_missing_pct": round(df.isnull().mean().mean() * 100, 2),
    }


# --------------------------------------------------------------------------- #
# DRIFT / ANOMALY DETECTION
# --------------------------------------------------------------------------- #

def detect_drift(df: pd.DataFrame, col_types: Dict[str, str], datetime_col: Optional[str] = None) -> List[Dict]:
    """
    Detect statistical shifts between first and second half of data.
    Only runs on strictly numeric columns. Always safe.
    """
    findings = []

    numeric_cols = [
        c for c, t in col_types.items()
        if t == "numeric" and c != datetime_col
    ]

    if not numeric_cols:
        return findings

    df = df.copy()
    if datetime_col and datetime_col in df.columns:
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col], infer_datetime_format=True, errors="coerce")
            df_sorted = df.dropna(subset=[datetime_col]).sort_values(datetime_col)
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

    for col in numeric_cols[:8]:
        try:
            a = pd.to_numeric(group_a[col], errors="coerce").dropna().astype(float)
            b = pd.to_numeric(group_b[col], errors="coerce").dropna().astype(float)

            if len(a) < 20 or len(b) < 20:
                continue

            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            mean_a = float(a.mean())
            mean_b = float(b.mean())
            pct_change = (mean_b - mean_a) / max(abs(mean_a), 1e-9) * 100

            if p_val < 0.05:
                severity = "high" if p_val < 0.01 else "medium"
                direction = "increased" if mean_b > mean_a else "decreased"
                findings.append({
                    "column":      col,
                    "finding":     f"{col} {direction} by {abs(pct_change):.1f}% ({split_label})",
                    "p_value":     round(float(p_val), 4),
                    "mean_before": round(mean_a, 4),
                    "mean_after":  round(mean_b, 4),
                    "pct_change":  round(pct_change, 2),
                    "severity":    severity,
                })
        except Exception:
            continue

    findings.sort(key=lambda x: x["p_value"])
    return findings

