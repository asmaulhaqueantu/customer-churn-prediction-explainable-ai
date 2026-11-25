"""Business facts and the value of the retention programme (test set, saved model).

All money figures are in relative units: one retention offer costs 1 unit. They are assumptions,
not company data - replace them in models/model_card.json / below with real numbers.

Usage (from the project root):  python src/business_analysis.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from features import FEATURES, clean

ROOT = Path(__file__).resolve().parents[1]


def programme_cost(y, contact, offer, value, success):
    tp = int((contact & (y == 1)).sum()); fp = int((contact & (y == 0)).sum()); fn = int((~contact & (y == 1)).sum())
    return (tp + fp) * offer + tp * (1 - success) * value + fn * value


def main():
    raw = clean(pd.read_csv(ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"))
    raw["churn"] = raw["Churn"].eq("Yes").astype(int)
    ch, st = raw[raw.churn == 1], raw[raw.churn == 0]
    facts = {
        "customers": len(raw), "churners": int(raw.churn.sum()), "churn_rate_pct": round(raw.churn.mean() * 100, 1),
        "monthly_revenue_total": round(raw.MonthlyCharges.sum(), 0),
        "monthly_revenue_of_churners": round(ch.MonthlyCharges.sum(), 0),
        "revenue_share_of_churners_pct": round(ch.MonthlyCharges.sum() / raw.MonthlyCharges.sum() * 100, 1),
        "avg_monthly_charge_churn": round(ch.MonthlyCharges.mean(), 1), "avg_monthly_charge_stay": round(st.MonthlyCharges.mean(), 1),
        "median_tenure_churn": float(ch.tenure.median()), "median_tenure_stay": float(st.tenure.median()),
    }
    print(json.dumps(facts, indent=2))

    model = joblib.load(ROOT / "models" / "churn_pipeline.joblib")
    card = json.loads((ROOT / "models" / "model_card.json").read_text())
    a = card["cost_assumptions"]; thr = card["decision_threshold"]
    X, y = raw[FEATURES], raw["churn"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    y_test = y_test.to_numpy(); p = model.predict_proba(X_test)[:, 1]
    n = len(y_test); rng = np.random.default_rng(42)
    offer, value, success = a["offer_cost"], a["customer_value"], a["retention_success_rate"]

    strategies = {
        "Do nothing": np.zeros(n, bool),
        "Contact everyone": np.ones(n, bool),
        "Contact a random 30%": rng.random(n) < 0.30,
        f"Model: contact p >= {thr:.2f}": p >= thr,
    }
    rows = []
    base = programme_cost(y_test, strategies["Do nothing"], offer, value, success)
    for name, mask in strategies.items():
        c = programme_cost(y_test, mask, offer, value, success)
        rows.append({"Strategy": name, "Customers contacted": int(mask.sum()),
                     "Churners reached": int((mask & (y_test == 1)).sum()), "Total cost (units)": round(c, 1),
                     "Saving vs do nothing (units)": round(base - c, 1)})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "reports" / "business_scenario.csv", index=False)
    print(table.to_string(index=False))

    sens = []
    for v in (4, 8, 12):
        for s in (0.2, 0.3, 0.4, 0.5):
            mask = p >= offer / (s * v)          # break-even rule: contact when p >= offer / (success x value)
            c = programme_cost(y_test, mask, offer, v, s); b = programme_cost(y_test, np.zeros(n, bool), offer, v, s)
            sens.append({"Customer value (x offer cost)": v, "Offer success rate": s, "Break-even threshold": round(offer / (s * v), 2),
                         "Contacted": int(mask.sum()), "Saving vs do nothing (units)": round(b - c, 1),
                         "Saving as % of do-nothing cost": round((b - c) / b * 100, 1)})
    sens = pd.DataFrame(sens)
    sens.to_csv(ROOT / "reports" / "business_sensitivity.csv", index=False)
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
