# DECIS — Decision Intelligence Engine

Upload **any** structured dataset → automatic profiling, ML modeling, strategy simulation, and executive PDF report.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Generate a sample dataset to test with
```bash
python generate_sample_data.py
```
This creates `sample_customer_data.csv` — 2,000 rows of realistic SaaS customer data.

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## What It Does

| Step | What Happens |
|------|-------------|
| **Upload** | Drop any CSV or Excel file. Engine starts automatically. |
| **Data Profile** | Schema detection, distributions, missing value analysis, statistical drift/anomaly alerts |
| **Model Intelligence** | Trains 4–5 models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM), picks the best via cross-validation, shows SHAP feature importance |
| **Simulation Lab** | Sliders for top drivers — adjust any feature by %, see predicted outcome change + revenue impact |
| **Strategy Ranking** | All interventions ranked by composite score (impact × confidence × risk × speed) |
| **Executive Report** | Full narrative PDF with findings, drivers, recommended strategy, and ROI |

---

## File Structure

```
decis/
├── app.py                    ← Main Streamlit app (run this)
├── requirements.txt          ← All dependencies
├── generate_sample_data.py   ← Test data generator
│
├── core/
│   ├── profiler.py           ← Schema detection, statistics, drift analysis
│   ├── model_builder.py      ← AutoML pipeline, model selection, SHAP
│   ├── simulator.py          ← Strategy simulation, ROI ranking
│   └── report_generator.py   ← PDF report (ReportLab)
│
└── utils/
    └── charts.py             ← All Plotly visualizations
```

---

## Works With Any Dataset

The engine does NOT assume domain. It works with:
- SaaS customer data (churn prediction)
- E-commerce data (conversion, AOV)
- Finance data (default risk, revenue)
- HR data (attrition)
- Any tabular classification or regression problem

---

## Requirements

- Python 3.9+
- ~500MB disk for models
- No GPU needed
- Runs fully local — no API keys, no cloud

---

## Tips

- **Target column**: Auto-detected, but you can override in the Upload page
- **Revenue column**: Select it if present — enables real dollar ROI estimates
- **Large datasets**: For 100k+ rows, the first run may take 2–3 minutes
- **Bad data**: The engine handles missing values and mixed types automatically
