"""
generate_sample_data.py
Generates a realistic SaaS customer dataset for testing DECIS.
Run: python generate_sample_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
N = 2000

# Dates
base_date = datetime(2023, 1, 1)
signup_dates = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(N)]

# Features
plan_types = np.random.choice(["Basic", "Pro", "Mid-tier", "Enterprise"], N, p=[0.35, 0.30, 0.25, 0.10])
tenure_days = np.random.randint(1, 730, N)
usage_score = np.clip(np.random.normal(60, 20, N), 0, 100)
support_tickets = np.random.poisson(2, N)
login_frequency = np.clip(np.random.normal(12, 5, N), 0, 30).astype(int)
revenue_monthly = np.where(
    plan_types == "Basic", np.random.normal(29, 5, N),
    np.where(plan_types == "Mid-tier", np.random.normal(79, 10, N),
    np.where(plan_types == "Pro", np.random.normal(149, 20, N),
    np.random.normal(399, 50, N)))
).clip(10, 1000)

# Churn probability model
churn_prob = (
    0.3
    - 0.003 * usage_score
    + 0.05 * support_tickets
    - 0.0002 * tenure_days
    + 0.08 * (plan_types == "Mid-tier").astype(int)
    - 0.05 * (plan_types == "Enterprise").astype(int)
)
churn_prob = np.clip(churn_prob, 0.02, 0.85)
churn_flag = (np.random.rand(N) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id": [f"CUST_{i:05d}" for i in range(N)],
    "signup_date": signup_dates,
    "plan_type": plan_types,
    "tenure_days": tenure_days,
    "usage_score_30d": usage_score.round(1),
    "support_tickets": support_tickets,
    "login_frequency": login_frequency,
    "revenue_monthly": revenue_monthly.round(2),
    "churn_flag": churn_flag,
})

df.to_csv("sample_customer_data.csv", index=False)
print(f"✓ Generated sample_customer_data.csv — {N} rows × {len(df.columns)} columns")
print(f"  Churn rate: {churn_flag.mean()*100:.1f}%")
print(f"\nNow run: streamlit run app.py")
