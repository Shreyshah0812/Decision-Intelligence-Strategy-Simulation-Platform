"""
core/report_generator.py
Generate a professional executive PDF report using reportlab.
"""

import io
import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


# Color palette
C_BG        = colors.HexColor("#0a0b0f")
C_ACCENT    = colors.HexColor("#00e5a0")
C_DARK      = colors.HexColor("#1e2330")
C_TEXT      = colors.HexColor("#2c3e50")
C_LIGHT     = colors.HexColor("#f5f7fa")
C_RED       = colors.HexColor("#e74c3c")
C_YELLOW    = colors.HexColor("#f39c12")
C_GREEN     = colors.HexColor("#27ae60")
C_PURPLE    = colors.HexColor("#7b61ff")
C_HEADER_BG = colors.HexColor("#0d1117")


def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle("title",
            fontName="Helvetica-Bold", fontSize=22, textColor=C_TEXT,
            spaceAfter=6, spaceBefore=0, leading=28),
        "subtitle": ParagraphStyle("subtitle",
            fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#7f8c8d"),
            spaceAfter=20, leading=14),
        "section": ParagraphStyle("section",
            fontName="Helvetica-Bold", fontSize=13, textColor=C_TEXT,
            spaceBefore=18, spaceAfter=6, leading=18,
            borderPad=4),
        "body": ParagraphStyle("body",
            fontName="Helvetica", fontSize=10, textColor=C_TEXT,
            spaceAfter=8, leading=16),
        "caption": ParagraphStyle("caption",
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=colors.HexColor("#95a5a6"),
            spaceAfter=4, leading=12),
        "metric_label": ParagraphStyle("metric_label",
            fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#7f8c8d"),
            alignment=TA_CENTER, leading=12),
        "metric_value": ParagraphStyle("metric_value",
            fontName="Helvetica-Bold", fontSize=18, textColor=C_TEXT,
            alignment=TA_CENTER, leading=22),
        "green_value": ParagraphStyle("green_value",
            fontName="Helvetica-Bold", fontSize=18, textColor=C_GREEN,
            alignment=TA_CENTER, leading=22),
        "red_value": ParagraphStyle("red_value",
            fontName="Helvetica-Bold", fontSize=18, textColor=C_RED,
            alignment=TA_CENTER, leading=22),
        "label": ParagraphStyle("label",
            fontName="Helvetica-Bold", fontSize=8,
            textColor=colors.HexColor("#7b61ff"),
            spaceBefore=0, spaceAfter=3,
            leading=11, leftIndent=0),
    }
    return styles


def generate_pdf_report(
    filename: str,
    dataset_name: str,
    profile_result: Dict[str, Any],
    model_result: Dict[str, Any],
    strategies: List[Dict[str, Any]],
    drift_findings: List[Dict[str, Any]],
    target_col: str,
) -> bytes:
    """
    Build full executive PDF report. Returns bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
        title="Decision Intelligence Report",
        author="DECIS Engine",
    )

    S = build_styles()
    story = []
    W = A4[0] - 4 * cm  # usable width

    # ── HEADER ──────────────────────────────────────────────────────────────
    today = datetime.date.today().strftime("%B %d, %Y")
    story.append(Paragraph("DECIS · Decision Intelligence Report", S["title"]))
    story.append(Paragraph(f"Dataset: {dataset_name}  ·  Generated: {today}  ·  Target: {target_col}", S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT, spaceAfter=16))

    # ── DATASET SUMMARY ──────────────────────────────────────────────────────
    story.append(Paragraph("1 · Dataset Summary", S["section"]))
    rows_n, cols_n = profile_result["shape"]
    missing = profile_result["total_missing_pct"]
    problem = model_result.get("problem_type", "—")
    score = model_result.get("score", 0)
    metric = model_result.get("scoring_metric", "score")
    best_model = model_result.get("best_model_name", "—")

    summary_data = [
        ["Metric", "Value"],
        ["Rows", f"{rows_n:,}"],
        ["Columns", str(cols_n)],
        ["Missing Data", f"{missing}%"],
        ["Problem Type", problem.capitalize()],
        ["Best Model", best_model],
        [f"CV {metric.upper()}", f"{score:.4f}"],
    ]
    story.append(_make_table(summary_data, col_widths=[W * 0.45, W * 0.55]))
    story.append(Spacer(1, 10))

    # ── DRIFT FINDINGS ───────────────────────────────────────────────────────
    story.append(Paragraph("2 · Statistical Shift Analysis", S["section"]))
    if drift_findings:
        story.append(Paragraph(
            f"The engine detected {len(drift_findings)} statistically significant shift(s) in your dataset. "
            "These represent real changes, not random variation (p < 0.05).", S["body"]
        ))
        drift_data = [["Column", "Direction", "Before", "After", "Change %", "p-value", "Severity"]]
        for f in drift_findings[:8]:
            drift_data.append([
                f["column"],
                "▲ Up" if f["pct_change"] > 0 else "▼ Down",
                str(f["mean_before"]),
                str(f["mean_after"]),
                f"{f['pct_change']:+.1f}%",
                str(f["p_value"]),
                f["severity"].upper(),
            ])
        story.append(_make_table(drift_data, header=True, col_widths=[
            W*0.18, W*0.10, W*0.12, W*0.12, W*0.12, W*0.12, W*0.12
        ], highlight_col=6, high_val="HIGH", high_color=C_RED))
    else:
        story.append(Paragraph("No statistically significant shifts detected.", S["body"]))
    story.append(Spacer(1, 10))

    # ── FEATURE IMPORTANCE ───────────────────────────────────────────────────
    story.append(Paragraph("3 · Key Drivers (Feature Importance)", S["section"]))
    story.append(Paragraph(
        f"The {best_model} model identified the following features as the most important predictors of {target_col}.",
        S["body"]
    ))
    fi = model_result.get("feature_importance")
    if fi is not None and not fi.empty:
        fi_data = [["Rank", "Feature", "Importance Score", "Bar"]]
        for i, row in fi.head(8).iterrows():
            bar = "█" * int(row["importance"] * 30) + "░" * (30 - int(row["importance"] * 30))
            fi_data.append([
                str(i + 1),
                row["feature"],
                f"{row['importance']:.4f}",
                bar,
            ])
        story.append(_make_table(fi_data, header=True, col_widths=[W*0.08, W*0.30, W*0.18, W*0.44]))
    story.append(Spacer(1, 10))

    # ── STRATEGY RANKING ─────────────────────────────────────────────────────
    story.append(Paragraph("4 · Strategy Simulation & Ranking", S["section"]))
    story.append(Paragraph(
        "The simulation engine evaluated all feasible interventions by perturbing top drivers and "
        "measuring predicted outcome changes. Strategies are ranked by composite score.",
        S["body"]
    ))

    if strategies:
        top = strategies[0]
        # Highlight box for #1
        story.append(Spacer(1, 6))
        story.append(Paragraph("▶  TOP RECOMMENDED STRATEGY", S["label"]))
        rec_data = [
            [
                Paragraph("Strategy", S["metric_label"]),
                Paragraph("Outcome Δ", S["metric_label"]),
                Paragraph("Revenue Impact", S["metric_label"]),
                Paragraph("ROI", S["metric_label"]),
                Paragraph("Confidence", S["metric_label"]),
            ],
            [
                Paragraph(top["name"], S["metric_value"]),
                Paragraph(f"{top['outcome_delta_pct']:+.2f}%",
                          S["green_value"] if top["beneficial"] else S["red_value"]),
                Paragraph(f"${top['revenue_impact']:,.0f}", S["green_value"]),
                Paragraph(f"{top['roi_pct']:.0f}%", S["metric_value"]),
                Paragraph(f"{top['confidence']:.0f}%", S["metric_value"]),
            ],
        ]
        rec_tbl = Table(rec_data, colWidths=[W / 5] * 5)
        rec_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_LIGHT),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f0fdf8")),
            ("BOX", (0, 0), (-1, -1), 1.5, C_ACCENT),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, C_DARK),
            ("ROWPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(rec_tbl)
        story.append(Spacer(1, 12))

        # All strategies table
        strat_data = [["#", "Strategy", "Δ Outcome %", "Rev. Impact", "ROI %", "Risk", "Score"]]
        for i, s in enumerate(strategies[:10]):
            strat_data.append([
                str(i + 1),
                s["name"],
                f"{s['outcome_delta_pct']:+.2f}%",
                f"${s['revenue_impact']:,.0f}",
                f"{s['roi_pct']:.0f}%",
                s["risk"],
                str(s["composite_score"]),
            ])
        story.append(_make_table(strat_data, header=True, col_widths=[
            W*0.05, W*0.32, W*0.13, W*0.14, W*0.10, W*0.10, W*0.10
        ]))

    # ── MODEL COMPARISON ─────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph("5 · Model Comparison", S["section"]))
    mc = model_result.get("model_comparison", [])
    if mc:
        mc_data = [["Model", f"CV {metric.upper()}", "Std Dev"]]
        for m in mc:
            mc_data.append([m["Model"], str(m["Score"]), str(m["Std"])])
        story.append(_make_table(mc_data, header=True, col_widths=[W*0.45, W*0.28, W*0.27]))

    # ── FOOTER ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=C_DARK, spaceAfter=8))
    story.append(Paragraph(
        "This report was auto-generated by DECIS — Decision Intelligence Engine. "
        "Results are based on statistical analysis and ML predictions. "
        "Validate findings before making business decisions.",
        S["caption"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# --------------------------------------------------------------------------- #
# TABLE HELPER
# --------------------------------------------------------------------------- #

def _make_table(data, header=False, col_widths=None, highlight_col=None, high_val=None, high_color=None):
    tbl = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("BACKGROUND", (0, 0), (-1, 0), C_LIGHT if header else colors.white),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_TEXT),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("GRID", (0, 0), (-1, -1), 0.4, C_DARK),
        ("ROWPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]
    if highlight_col is not None and high_val and high_color:
        for i, row in enumerate(data[1:], start=1):
            if len(row) > highlight_col and str(row[highlight_col]) == high_val:
                style.append(("TEXTCOLOR", (highlight_col, i), (highlight_col, i), high_color))
                style.append(("FONTNAME", (highlight_col, i), (highlight_col, i), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(style))
    return tbl
