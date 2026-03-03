"""
app.py
DECIS — Domain-Agnostic Decision Intelligence Engine
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import traceback
from typing import Optional

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="DECIS · Decision Intelligence Engine",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    .stApp { background-color: #0a0b0f; }
    section[data-testid="stSidebar"] { background-color: #111318 !important; border-right: 1px solid #1e2330; }
    section[data-testid="stSidebar"] * { color: #9aa3c0 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #13161f;
        border: 1px solid #1e2330;
        border-radius: 10px;
        padding: 16px !important;
        border-left: 3px solid #00e5a0;
    }
    [data-testid="stMetricLabel"] { color: #6b7591 !important; font-size: 10px !important; letter-spacing: 0.1em; }
    [data-testid="stMetricValue"] { color: #e8ecf5 !important; font-size: 24px !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 11px !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #111318; border-bottom: 1px solid #1e2330; gap: 0; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #6b7591 !important;
        border: none;
        font-size: 11px;
        letter-spacing: 0.05em;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0,229,160,0.08) !important;
        color: #00e5a0 !important;
        border-bottom: 2px solid #00e5a0 !important;
    }

    /* Dataframes */
    .stDataFrame { background: #13161f; border: 1px solid #1e2330; border-radius: 8px; }

    /* Buttons */
    .stButton button {
        background-color: #00e5a0 !important;
        color: #0a0b0f !important;
        border: none !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 11px !important;
        letter-spacing: 0.05em !important;
        border-radius: 6px !important;
        transition: all 0.2s !important;
    }
    .stButton button:hover { box-shadow: 0 0 16px rgba(0,229,160,0.35) !important; }

    /* Download buttons */
    .stDownloadButton button {
        background-color: transparent !important;
        color: #9aa3c0 !important;
        border: 1px solid #1e2330 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        border-radius: 6px !important;
    }
    .stDownloadButton button:hover { border-color: #00e5a0 !important; color: #00e5a0 !important; }

    /* Sliders */
    .stSlider [data-baseweb="slider"] { padding: 0 4px; }

    /* Alerts */
    .stAlert { border-radius: 8px !important; font-size: 11px !important; }

    /* Headers */
    h1, h2, h3 { color: #e8ecf5 !important; font-family: 'JetBrains Mono', monospace !important; }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] { background-color: #13161f !important; border-color: #1e2330 !important; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #13161f;
        border: 1.5px dashed #1e2330;
        border-radius: 10px;
        padding: 12px;
    }

    /* Expander */
    .streamlit-expanderHeader { background: #13161f !important; color: #9aa3c0 !important; font-size: 11px !important; }
    .streamlit-expanderContent { background: #111318 !important; }

    /* Progress */
    .stProgress > div > div { background-color: #00e5a0 !important; }

    /* Info/warning/error */
    div[data-baseweb="notification"] { border-radius: 8px !important; }

    /* Card style via markdown */
    .card {
        background: #13161f;
        border: 1px solid #1e2330;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .accent-border { border-left: 3px solid #00e5a0 !important; }
    .page-title { color: #e8ecf5; font-size: 22px; font-weight: 700; margin-bottom: 4px; }
    .page-sub { color: #6b7591; font-size: 11px; letter-spacing: 0.06em; margin-bottom: 20px; }
    .section-label {
        color: #00e5a0;
        font-size: 9px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .separator { border-top: 1px solid #1e2330; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (to avoid startup cost) ─────────────────────────────────────
@st.cache_resource
def _imports():
    from core.profiler import full_profile, detect_column_types, infer_target, detect_drift
    from core.model_builder import build_model
    from core.simulator import generate_strategies, run_custom_simulation
    from core.report_generator import generate_pdf_report
    from utils.charts import (
        missing_heatmap, correlation_heatmap, distribution_plot,
        feature_importance_chart, model_comparison_chart,
        strategy_ranking_chart, simulation_delta_chart,
        target_distribution, drift_chart,
    )
    return {
        "full_profile": full_profile,
        "detect_column_types": detect_column_types,
        "infer_target": infer_target,
        "detect_drift": detect_drift,
        "build_model": build_model,
        "generate_strategies": generate_strategies,
        "run_custom_simulation": run_custom_simulation,
        "generate_pdf_report": generate_pdf_report,
        "missing_heatmap": missing_heatmap,
        "correlation_heatmap": correlation_heatmap,
        "distribution_plot": distribution_plot,
        "feature_importance_chart": feature_importance_chart,
        "model_comparison_chart": model_comparison_chart,
        "strategy_ranking_chart": strategy_ranking_chart,
        "simulation_delta_chart": simulation_delta_chart,
        "target_distribution": target_distribution,
        "drift_chart": drift_chart,
    }


# ── Session state helpers ─────────────────────────────────────────────────────
def ss_get(key, default=None):
    return st.session_state.get(key, default)

def ss_set(key, val):
    st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⬡ DECIS")
        st.markdown("<div style='font-size:9px;color:#3a4060;letter-spacing:0.15em;margin-bottom:20px'>DECISION INTELLIGENCE ENGINE</div>", unsafe_allow_html=True)

        pages = {
            "📤  Upload Data":       "upload",
            "📊  Data Profile":      "profile",
            "🤖  Model Intelligence":"model",
            "🔬  Simulation Lab":    "simulation",
            "🏆  Strategy Ranking":  "ranking",
            "📋  Executive Report":  "report",
        }

        for label, key in pages.items():
            is_locked = (key != "upload") and ss_get("df") is None
            is_model_locked = key in ("simulation", "ranking", "report") and ss_get("model_result") is None

            disabled_marker = " 🔒" if (is_locked or is_model_locked) else ""
            current = ss_get("page", "upload")
            style = "color: #00e5a0 !important; font-weight: 700;" if key == current else ""

            if st.sidebar.button(
                label + disabled_marker,
                key=f"nav_{key}",
                disabled=is_locked or is_model_locked,
                width="stretch",
            ):
                ss_set("page", key)

        st.sidebar.markdown("---")
        st.sidebar.markdown("<div style='font-size:9px;color:#3a4060'>● ENGINE READY · v1.0</div>", unsafe_allow_html=True)

        # Show current dataset info
        if ss_get("df") is not None:
            df = ss_get("df")
            st.sidebar.markdown(f"""
            <div style='font-size:9px;color:#6b7591;margin-top:10px;line-height:1.8'>
            📁 {ss_get('filename','dataset')}<br>
            📐 {df.shape[0]:,} rows · {df.shape[1]} cols<br>
            🎯 {ss_get('target_col','—')}
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
def page_upload(fn):
    st.markdown("<div class='page-title'>Upload <span style='color:#00e5a0'>Any</span> Dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>CSV · XLSX · XLS — zero configuration required</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.markdown("""<div class='card accent-border'>
        <div class='section-label'>Step 1 · Auto</div>
        <b style='color:#e8ecf5'>Data Profiling</b><br>
        <small style='color:#6b7591'>Schema · distributions · drift detection · anomalies</small>
    </div>""", unsafe_allow_html=True)
    col2.markdown("""<div class='card' style='border-left:3px solid #7b61ff'>
        <div class='section-label' style='color:#7b61ff'>Step 2 · Auto</div>
        <b style='color:#e8ecf5'>Model Building</b><br>
        <small style='color:#6b7591'>Target inference · AutoML · SHAP feature importance</small>
    </div>""", unsafe_allow_html=True)
    col3.markdown("""<div class='card' style='border-left:3px solid #ff6b35'>
        <div class='section-label' style='color:#ff6b35'>Step 3 · Auto</div>
        <b style='color:#e8ecf5'>Decision Output</b><br>
        <small style='color:#6b7591'>Strategy simulation · ROI ranking · executive report</small>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your dataset here",
        type=["csv", "xlsx", "xls"],
        help="Any structured dataset. The engine will automatically infer the target variable and problem type.",
    )

    if uploaded:
        with st.spinner("Reading file..."):
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                ss_set("df", df)
                ss_set("filename", uploaded.name)
                st.success(f"✓ Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                return

        df = ss_get("df")
        st.markdown("#### Preview (first 5 rows)")
        st.dataframe(df.head(), width="stretch")

        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
        st.markdown("#### Configure (optional — engine will auto-detect)")

        # Target column selector
        col_a, col_b = st.columns(2)
        with col_a:
            # Auto-infer default
            if ss_get("target_col") is None:
                col_types = fn["detect_column_types"](df)
                t_col, _, _ = fn["infer_target"](df, col_types)
                ss_set("target_col", t_col)

            all_cols = df.columns.tolist()
            current_target = ss_get("target_col")
            default_idx = all_cols.index(current_target) if current_target in all_cols else 0

            target_col = st.selectbox(
                "Target Variable",
                options=all_cols,
                index=default_idx,
                help="The column you want to predict or analyze. Engine auto-detects this.",
            )
            ss_set("target_col", target_col)

        with col_b:
            revenue_cols = ["— None —"] + [c for c in df.columns if any(kw in c.lower() for kw in ["revenue", "sales", "price", "amount", "value", "mrr", "arr"])]
            rev_col = st.selectbox(
                "Revenue Column (for ROI estimation)",
                options=revenue_cols,
                help="If available, used to estimate dollar impact of interventions.",
            )
            ss_set("revenue_col", None if rev_col == "— None —" else rev_col)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀  Run Decision Engine", width="content"):
            _run_full_pipeline(fn)


def _run_full_pipeline(fn):
    df = ss_get("df")
    target_col = ss_get("target_col")

    if df is None or target_col is None:
        st.error("Please upload a dataset first.")
        return

    progress_bar = st.progress(0, text="Profiling data...")

    try:
        # Step 1: Profile
        progress_bar.progress(0.1, "Profiling schema and statistics...")
        col_types = fn["detect_column_types"](df)
        profile = fn["full_profile"](df)
        ss_set("col_types", col_types)
        ss_set("profile", profile)

        # Step 2: Drift
        progress_bar.progress(0.25, "Detecting statistical shifts...")
        dt_cols = profile.get("datetime_cols", [])
        dt_col = dt_cols[0] if dt_cols else None
        drift = fn["detect_drift"](df, col_types, dt_col)
        ss_set("drift", drift)

        # Step 3: Infer problem type
        progress_bar.progress(0.35, "Inferring target variable and problem type...")
        _, problem_type, confidence = fn["infer_target"](df, col_types)
        ss_set("problem_type", problem_type)
        ss_set("target_confidence", confidence)

        # Step 4: Build model
        def model_progress(pct, msg):
            progress_bar.progress(0.35 + pct * 0.50, msg)

        model_result = fn["build_model"](df, target_col, problem_type, col_types, progress_callback=model_progress)
        ss_set("model_result", model_result)

        # Step 5: Generate strategies
        progress_bar.progress(0.90, "Simulating strategies and ranking...")
        rev_col = ss_get("revenue_col")
        strategies = fn["generate_strategies"](
            model_result,
            top_n_features=4,
            revenue_col=rev_col,
            df_full=df,
        )
        ss_set("strategies", strategies)

        progress_bar.progress(1.0, "✓ Complete!")
        st.success("✓ Analysis complete! Navigate through the pages in the sidebar.")
        ss_set("page", "profile")
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        st.error(f"Pipeline error: {e}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: DATA PROFILE
# ─────────────────────────────────────────────────────────────────────────────
def page_profile(fn):
    df = ss_get("df")
    profile = ss_get("profile")
    col_types = ss_get("col_types")
    drift = ss_get("drift", [])

    if df is None or profile is None:
        st.info("Upload and run the engine first."); return

    filename = ss_get("filename", "dataset")
    rows, cols = profile["shape"]

    st.markdown(f"<div class='page-title'>Data <span style='color:#00e5a0'>Profile</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-sub'>{filename} · {rows:,} rows · {cols} columns</div>", unsafe_allow_html=True)

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", str(cols))
    c3.metric("Numeric", str(len(profile["numeric_cols"])))
    c4.metric("Categorical", str(len(profile["categorical_cols"])))
    c5.metric("Missing %", f"{profile['total_missing_pct']}%",
              delta="Clean" if profile["total_missing_pct"] < 5 else "Needs attention",
              delta_color="normal" if profile["total_missing_pct"] < 5 else "inverse")

    # Drift alerts
    if drift:
        st.markdown("---")
        st.markdown("#### 🔍 Statistical Shift Alerts")
        for finding in drift[:5]:
            sev_color = "🔴" if finding["severity"] == "high" else "🟡"
            st.warning(f"{sev_color} **{finding['column']}** — {finding['finding']} (p={finding['p_value']})")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📋  Column Schema", "📉  Missing Values", "📊  Distributions", "🔗  Correlations"])

    with tab1:
        schema_rows = []
        for col in df.columns:
            p = profile["col_profiles"][col]
            ctype = col_types.get(col, "—")
            schema_rows.append({
                "Column": col,
                "Type": ctype,
                "Missing %": f"{p['missing_pct']}%",
                "Unique": p["n_unique"],
                "Mean/Top": str(p.get("mean", p.get("top_value", "—"))),
                "Std/Freq%": str(p.get("std", str(p.get("top_freq_pct", "")) + "%")),
            })
        st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)

    with tab2:
        fig = fn["missing_heatmap"](df)
        st.plotly_chart(fig, width="stretch")

    with tab3:
        col_pick = st.selectbox("Select column to plot", df.columns.tolist(), key="dist_col")
        fig = fn["distribution_plot"](df[col_pick], col_pick)
        st.plotly_chart(fig, width="stretch")

        # Quick stats
        p = profile["col_profiles"][col_pick]
        if "mean" in p:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{p['mean']:.4f}" if p["mean"] else "—")
            c2.metric("Std", f"{p['std']:.4f}" if p["std"] else "—")
            c3.metric("Outliers (IQR)", str(p.get("outliers_iqr", "—")))
            c4.metric("Skewness", f"{p.get('skewness', 0):.3f}")

    with tab4:
        num_cols = profile["numeric_cols"]
        if len(num_cols) >= 2:
            selected_cols = st.multiselect(
                "Select columns for correlation",
                num_cols,
                default=num_cols[:min(8, len(num_cols))],
                key="corr_cols",
            )
            if len(selected_cols) >= 2:
                fig = fn["correlation_heatmap"](df, selected_cols)
                if fig:
                    st.plotly_chart(fig, width="stretch")
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    # Drift chart
    if drift:
        st.markdown("---")
        st.markdown("#### Statistical Drift Summary")
        fig = fn["drift_chart"](drift)
        if fig:
            st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
def page_model(fn):
    model_result = ss_get("model_result")
    if model_result is None:
        st.info("Run the engine first."); return

    target_col = ss_get("target_col")
    problem_type = ss_get("problem_type")

    st.markdown("<div class='page-title'>Model <span style='color:#00e5a0'>Intelligence</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-sub'>Auto-selected: {model_result['best_model_name']} · Problem: {problem_type.upper()}</div>", unsafe_allow_html=True)

    # Metric cards
    metric_name = model_result["scoring_metric"].upper()
    score = model_result["score"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"CV {metric_name}", f"{score:.4f}", delta="Excellent" if score > 0.85 else ("Good" if score > 0.70 else "Moderate"))
    c2.metric("Std Dev", f"±{model_result['score_std']:.4f}")
    c3.metric("Best Model", model_result["best_model_name"])
    c4.metric("Problem Type", problem_type.capitalize())

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊  Feature Importance", "🏆  Model Comparison", "🎯  Target Distribution"])

    with tab1:
        fi = model_result.get("feature_importance")
        if fi is not None and not fi.empty:
            fig = fn["feature_importance_chart"](fi)
            st.plotly_chart(fig, width="stretch")

            st.markdown("#### Top Features Table")
            display_fi = fi.copy()
            display_fi["importance"] = display_fi["importance"].round(4)
            display_fi.index = range(1, len(display_fi) + 1)
            st.dataframe(display_fi.head(15), width="stretch")
        else:
            st.info("Feature importance not available for this model type.")

    with tab2:
        mc = model_result.get("model_comparison", [])
        if mc:
            fig = fn["model_comparison_chart"](mc)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame(mc), width="stretch", hide_index=True)

    with tab3:
        y = model_result["y"]
        fig = fn["target_distribution"](y, target_col or "Target")
        st.plotly_chart(fig, width="stretch")

        vc = y.value_counts()
        c1, c2 = st.columns(2)
        c1.metric("Unique Classes", str(y.nunique()))
        c2.metric("Most Common", f"{vc.index[0]} ({vc.iloc[0]/len(y)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: SIMULATION LAB
# ─────────────────────────────────────────────────────────────────────────────
def page_simulation(fn):
    model_result = ss_get("model_result")
    if model_result is None:
        st.info("Run the engine first."); return

    problem_type = ss_get("problem_type")
    fi = model_result.get("feature_importance")

    st.markdown("<div class='page-title'>Simulation <span style='color:#00e5a0'>Lab</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Adjust top drivers · simulate outcome changes · see impact instantly</div>", unsafe_allow_html=True)

    if fi is None or fi.empty:
        st.warning("Feature importance not available — cannot run simulations."); return

    # Get numeric top features only
    X = model_result["X"]
    top_features = [
        f for f in fi["feature"].head(8).tolist()
        if f in X.columns and pd.api.types.is_numeric_dtype(X[f])
    ][:5]

    if not top_features:
        st.warning("No numeric features available for simulation."); return

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Intervention Sliders")
        st.caption("Adjust feature values by %, then click Simulate.")

        feature_deltas = {}
        for feat in top_features:
            imp_val = fi[fi["feature"] == feat]["importance"].values[0]
            delta = st.slider(
                f"{feat}  (importance: {imp_val:.3f})",
                min_value=-30,
                max_value=30,
                value=0,
                step=5,
                key=f"slider_{feat}",
                format="%d%%",
            )
            feature_deltas[feat] = float(delta)

        run_sim = st.button("⟳  Run Simulation", width="content")

    with col_right:
        st.markdown("#### Simulation Output")

        # Auto-run or on button press
        if run_sim or ss_get("sim_result") is not None:
            if run_sim or ss_get("sim_result") is None:
                with st.spinner("Running simulation..."):
                    result = fn["run_custom_simulation"](
                        model_result,
                        feature_deltas,
                        revenue_col=ss_get("revenue_col"),
                        df_full=ss_get("df"),
                    )
                    ss_set("sim_result", result)
                    ss_set("sim_deltas", feature_deltas)

            result = ss_get("sim_result")
            if "error" in result:
                st.error(f"Simulation error: {result['error']}")
            else:
                # Display results
                delta_pct = result["outcome_delta_pct"]
                rev_impact = result["revenue_impact"]
                customers = result.get("customers_affected", 0)
                conf = result.get("confidence", 0)

                # For classification, lower probability = good (churn reduction)
                if problem_type == "classification":
                    direction = "↓" if delta_pct < 0 else "↑"
                    good = delta_pct < 0
                else:
                    direction = "↑" if delta_pct > 0 else "↓"
                    good = delta_pct > 0

                c1, c2 = st.columns(2)
                c1.metric(
                    result["label"] + " Change",
                    f"{direction} {abs(delta_pct):.2f}%",
                    delta="Beneficial ✓" if good else "Adverse ✗",
                    delta_color="normal" if good else "inverse",
                )
                c2.metric(
                    "Estimated Revenue Impact",
                    f"${abs(rev_impact):,.0f}",
                    delta="↑ Gain" if good else "↓ Loss",
                    delta_color="normal" if good else "inverse",
                )

                c3, c4 = st.columns(2)
                if customers:
                    c3.metric("Customers Affected (est.)", f"{customers:,}")
                c4.metric("Model Confidence", f"{conf*100:.0f}%")

                st.markdown("---")
                st.markdown("**Before vs After**")
                orig = result["orig_outcome"]
                new_val = result["new_outcome"]

                ba_df = pd.DataFrame({
                    "Scenario": ["Before", "After"],
                    "Value": [orig, new_val],
                })
                import plotly.express as px
                fig = px.bar(
                    ba_df, x="Scenario", y="Value",
                    color="Scenario",
                    color_discrete_map={"Before": "#ff4757", "After": "#00e5a0"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    paper_bgcolor="#13161f",
                    plot_bgcolor="#13161f",
                    showlegend=False,
                    height=220,
                    margin=dict(l=20, r=20, t=20, b=30),
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.markdown("""
            <div style='background:#13161f; border:1px solid #1e2330; border-radius:8px; padding:20px; text-align:center; color:#6b7591; font-size:11px;'>
            Adjust sliders and click <b style='color:#00e5a0'>Run Simulation</b> to see results
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: STRATEGY RANKING
# ─────────────────────────────────────────────────────────────────────────────
def page_ranking(fn):
    strategies = ss_get("strategies")
    if not strategies:
        st.info("Run the engine first."); return

    problem_type = ss_get("problem_type", "classification")
    target_col = ss_get("target_col", "target")

    st.markdown("<div class='page-title'>Strategy <span style='color:#00e5a0'>Ranking</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-sub'>{len(strategies)} interventions simulated · ranked by composite score</div>", unsafe_allow_html=True)

    # Top strategy highlight
    top = strategies[0]
    st.markdown("#### 🥇 Top Recommended Strategy")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strategy", top["name"][:25] + ("..." if len(top["name"]) > 25 else ""))
    c2.metric("Outcome Δ", f"{top['outcome_delta_pct']:+.2f}%")
    c3.metric("Revenue Impact", f"${top['revenue_impact']:,.0f}")
    c4.metric("ROI", f"{top['roi_pct']:.0f}%")
    c5.metric("Composite Score", f"{top['composite_score']}/100")

    st.markdown("---")

    # Ranking chart
    fig = fn["strategy_ranking_chart"](strategies)
    if fig:
        st.plotly_chart(fig, width="stretch")

    # Full table
    st.markdown("#### All Strategies")
    table_data = []
    for i, s in enumerate(strategies):
        table_data.append({
            "Rank": i + 1,
            "Strategy": s["name"],
            "Feature": s["feature"],
            "Δ %": f"{s['delta_pct']:+.0f}%",
            "Outcome Δ%": f"{s['outcome_delta_pct']:+.2f}%",
            "Rev. Impact $": f"${s['revenue_impact']:,.0f}",
            "ROI %": f"{s['roi_pct']:.0f}%",
            "Risk": s["risk"],
            "Confidence %": f"{s['confidence']:.0f}%",
            "Score": s["composite_score"],
        })

    st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)

    # Scoring note
    st.caption("Composite score = Financial impact (40%) + Risk (20%) + Time-to-implement (20%) + Model confidence (20%)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: EXECUTIVE REPORT
# ─────────────────────────────────────────────────────────────────────────────
def page_report(fn):
    model_result = ss_get("model_result")
    profile = ss_get("profile")
    strategies = ss_get("strategies", [])
    drift = ss_get("drift", [])
    target_col = ss_get("target_col", "target")
    filename = ss_get("filename", "dataset.csv")

    if model_result is None or profile is None:
        st.info("Run the engine first."); return

    st.markdown("<div class='page-title'>Executive <span style='color:#00e5a0'>Report</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Auto-generated narrative · download as PDF</div>", unsafe_allow_html=True)

    # PDF Download button
    with st.spinner("Building PDF..."):
        try:
            pdf_bytes = fn["generate_pdf_report"](
                filename=filename,
                dataset_name=filename,
                profile_result=profile,
                model_result=model_result,
                strategies=strategies,
                drift_findings=drift,
                target_col=target_col,
            )
            st.download_button(
                label="⬇  Download PDF Report",
                data=pdf_bytes,
                file_name=f"decis_report_{filename.replace('.','_')}.pdf",
                mime="application/pdf",
                width="content",
            )
        except Exception as e:
            st.error(f"PDF generation error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    st.markdown("---")

    # ── Section 1: Dataset Summary ─────────────────────────────────────────
    st.markdown("### 1 · Dataset Summary")
    rows, cols_n = profile["shape"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", str(cols_n))
    c3.metric("Missing Data", f"{profile['total_missing_pct']}%")
    c4.metric("Problem Type", ss_get("problem_type", "—").capitalize())

    # ── Section 2: Drift ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2 · Statistical Shifts")
    if drift:
        for f in drift[:5]:
            badge = "🔴" if f["severity"] == "high" else "🟡"
            st.markdown(f"""
            <div class='card'>
            {badge} <b style='color:#e8ecf5'>{f['column']}</b> — {f['finding']}<br>
            <small style='color:#6b7591'>p-value: {f['p_value']} · Before: {f['mean_before']} → After: {f['mean_after']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'>✅ No significant statistical shifts detected.</div>", unsafe_allow_html=True)

    # ── Section 3: Key Drivers ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3 · Key Drivers")
    fi = model_result.get("feature_importance")
    if fi is not None and not fi.empty:
        best_model = model_result["best_model_name"]
        score = model_result["score"]
        metric = model_result["scoring_metric"].upper()
        top3 = fi.head(3)["feature"].tolist()
        st.markdown(f"""
        <div class='card accent-border'>
        The <b style='color:#00e5a0'>{best_model}</b> model achieved a cross-validation {metric} of <b style='color:#00e5a0'>{score:.4f}</b>.<br>
        The three most influential drivers of <b style='color:#e8ecf5'>{target_col}</b> are:
        <b style='color:#00e5a0'>{top3[0] if len(top3)>0 else '—'}</b>,
        <b style='color:#7b61ff'>{top3[1] if len(top3)>1 else '—'}</b>,
        and <b style='color:#ff6b35'>{top3[2] if len(top3)>2 else '—'}</b>.
        </div>
        """, unsafe_allow_html=True)

    # ── Section 4: Top Strategy ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 4 · Recommended Strategy")
    if strategies:
        top = strategies[0]
        good = top["beneficial"]
        rev = top["revenue_impact"]

        st.markdown(f"""
        <div class='card accent-border'>
        <div class='section-label'>🥇 Highest Ranked</div>
        <b style='color:#e8ecf5; font-size:16px'>{top['name']}</b><br><br>
        <span style='color:#6b7591'>{top['description']}</span><br><br>
        <b style='color:#00e5a0'>Predicted outcome change:</b> {top['outcome_delta_pct']:+.2f}%<br>
        <b style='color:#00e5a0'>Revenue impact:</b> ${rev:,.0f}<br>
        <b style='color:#00e5a0'>ROI:</b> {top['roi_pct']:.0f}%<br>
        <b style='color:#00e5a0'>Risk:</b> {top['risk']}<br>
        <b style='color:#00e5a0'>Model confidence:</b> {top['confidence']:.0f}%<br>
        <b style='color:#00e5a0'>Composite score:</b> {top['composite_score']}/100
        </div>
        """, unsafe_allow_html=True)

        # All strategies summary
        if len(strategies) > 1:
            st.markdown("#### All Strategies — Quick Summary")
            for i, s in enumerate(strategies[:6]):
                icon = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"][i]
                color = "#00e5a0" if i == 0 else "#7b61ff" if i == 1 else "#ff6b35" if i == 2 else "#6b7591"
                st.markdown(f"""
                <div style='padding:8px 0; border-bottom:1px solid #1e2330; font-size:11px;'>
                {icon} <b style='color:{color}'>{s['name']}</b> — 
                Outcome: {s['outcome_delta_pct']:+.2f}% · 
                Revenue: ${s['revenue_impact']:,.0f} · 
                Score: {s['composite_score']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No strategies generated.")

    # ── Section 5: Model Comparison ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 5 · Model Comparison")
    mc = model_result.get("model_comparison", [])
    if mc:
        st.dataframe(pd.DataFrame(mc), width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    fn = _imports()
    render_sidebar()

    page = ss_get("page", "upload")

    if page == "upload":
        page_upload(fn)
    elif page == "profile":
        page_profile(fn)
    elif page == "model":
        page_model(fn)
    elif page == "simulation":
        page_simulation(fn)
    elif page == "ranking":
        page_ranking(fn)
    elif page == "report":
        page_report(fn)
    else:
        page_upload(fn)


if __name__ == "__main__":
    main()
