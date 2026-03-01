"""
utils/charts.py
All Plotly chart functions used across the Streamlit pages.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional


PALETTE = {
    "bg": "#0a0b0f",
    "surface": "#13161f",
    "border": "#1e2330",
    "accent": "#00e5a0",
    "purple": "#7b61ff",
    "orange": "#ff6b35",
    "red": "#ff4757",
    "text": "#e8ecf5",
    "text2": "#6b7591",
}

LAYOUT_BASE = dict(
    paper_bgcolor=PALETTE["surface"],
    plot_bgcolor=PALETTE["surface"],
    font=dict(family="monospace", color=PALETTE["text"], size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
    yaxis=dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
)


def missing_heatmap(df: pd.DataFrame) -> go.Figure:
    missing = (df.isnull().mean() * 100).reset_index()
    missing.columns = ["Column", "Missing %"]
    missing = missing.sort_values("Missing %", ascending=True)

    fig = go.Figure(go.Bar(
        x=missing["Missing %"],
        y=missing["Column"],
        orientation="h",
        marker=dict(
            color=missing["Missing %"],
            colorscale=[[0, PALETTE["accent"]], [0.5, PALETTE["purple"]], [1, PALETTE["red"]]],
            showscale=True,
            colorbar=dict(title="% Missing", tickfont=dict(color=PALETTE["text2"])),
        ),
        text=[f"{v:.1f}%" for v in missing["Missing %"]],
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title="Missing Values by Column", height=max(250, len(df.columns) * 22))
    return fig


def correlation_heatmap(df: pd.DataFrame, num_cols: List[str]) -> go.Figure:
    if len(num_cols) < 2:
        return None
    corr = df[num_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, PALETTE["red"]], [0.5, PALETTE["surface"]], [1, PALETTE["accent"]]],
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        showscale=True,
    ))
    fig.update_layout(**LAYOUT_BASE, title="Correlation Matrix", height=max(300, len(num_cols) * 40))
    return fig


def distribution_plot(series: pd.Series, col_name: str) -> go.Figure:
    s = series.dropna()
    if pd.api.types.is_numeric_dtype(series):
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=s, nbinsx=40,
            marker_color=PALETTE["accent"],
            opacity=0.8,
            name=col_name,
        ))
        # KDE overlay
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(s)
            x_range = np.linspace(s.min(), s.max(), 200)
            kde_vals = kde(x_range) * len(s) * (s.max() - s.min()) / 40
            fig.add_trace(go.Scatter(
                x=x_range, y=kde_vals,
                mode="lines", line=dict(color=PALETTE["purple"], width=2),
                name="KDE",
            ))
        except Exception:
            pass
    else:
        vc = s.value_counts().head(15)
        fig = go.Figure(go.Bar(
            x=vc.index.astype(str),
            y=vc.values,
            marker_color=PALETTE["accent"],
            opacity=0.85,
        ))

    fig.update_layout(**LAYOUT_BASE, title=f"Distribution: {col_name}", height=300,
                      showlegend=False)
    return fig


def feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    df = fi_df.head(12).sort_values("importance")
    colors_list = [
        PALETTE["accent"] if i >= len(df) - 3
        else PALETTE["purple"] if i >= len(df) - 6
        else PALETTE["orange"]
        for i in range(len(df))
    ]
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color=colors_list,
        text=[f"{v:.3f}" for v in df["importance"]],
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title="Feature Importance (Normalized)", height=max(300, len(df) * 30 + 80))
    return fig


def model_comparison_chart(model_comparison: List[Dict]) -> go.Figure:
    names = [m["Model"] for m in model_comparison]
    scores = [m["Score"] for m in model_comparison]
    stds = [m["Std"] for m in model_comparison]

    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i])
    names = [names[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    highlight = [PALETTE["accent"] if i == len(names) - 1 else PALETTE["purple"] for i in range(len(names))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores, y=names, orientation="h",
        marker_color=highlight,
        error_x=dict(type="data", array=stds, color=PALETTE["text2"]),
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title="Model Comparison (CV Score)", height=280)
    return fig


def strategy_ranking_chart(strategies: List[Dict]) -> go.Figure:
    if not strategies:
        return None
    df = pd.DataFrame(strategies[:8])
    df = df.sort_values("composite_score")

    colors_list = [PALETTE["accent"] if i == len(df) - 1 else PALETTE["purple"] for i in range(len(df))]

    fig = go.Figure(go.Bar(
        x=df["composite_score"],
        y=df["name"],
        orientation="h",
        marker_color=colors_list,
        text=[f"{v:.0f}" for v in df["composite_score"]],
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title="Strategy Composite Score", height=max(280, len(df) * 35 + 80))
    return fig


def simulation_delta_chart(orig: float, new_val: float, label: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=new_val,
        delta={
            "reference": orig,
            "valueformat": ".4f",
            "increasing": {"color": PALETTE["red"] if "prob" in label.lower() else PALETTE["accent"]},
            "decreasing": {"color": PALETTE["accent"] if "prob" in label.lower() else PALETTE["red"]},
        },
        number={"font": {"color": PALETTE["text"], "size": 40}},
        title={"text": label, "font": {"color": PALETTE["text2"], "size": 12}},
    ))

    fig.update_layout(
        paper_bgcolor=PALETTE["surface"],
        height=160,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def target_distribution(y: pd.Series, target_col: str) -> go.Figure:
    vc = y.value_counts()
    fig = go.Figure(go.Bar(
        x=vc.index.astype(str),
        y=vc.values,
        marker_color=[PALETTE["accent"], PALETTE["red"]] + [PALETTE["purple"]] * max(0, len(vc) - 2),
        text=vc.values,
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title=f"Target Distribution: {target_col}", height=280)
    return fig


def drift_chart(findings: List[Dict]) -> go.Figure:
    if not findings:
        return None
    df = pd.DataFrame(findings).head(8)
    df = df.sort_values("pct_change")

    colors_list = [PALETTE["red"] if v > 0 else PALETTE["accent"] for v in df["pct_change"]]

    fig = go.Figure(go.Bar(
        x=df["pct_change"],
        y=df["column"],
        orientation="h",
        marker_color=colors_list,
        text=[f"{v:+.1f}%" for v in df["pct_change"]],
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title="Detected Shifts (% Change, First Half → Second Half)",
                      height=max(250, len(df) * 32 + 80))
    return fig
