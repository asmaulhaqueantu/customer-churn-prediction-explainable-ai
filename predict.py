"""Score customers with the saved churn model.

Output per customer: churn probability, risk tier, contact flag and the top drivers
that push the customer's risk up (reason codes, derived from the linear model).

Usage (from the project root):
    python src/predict.py --input data/raw/Telco-Customer-Churn.csv --output reports/predictions.csv --top 20
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CATEGORICAL, FEATURES, clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
CARD_PATH = ROOT / "models" / "model_card.json"


def _original_feature(encoded_name: str) -> str:
    base = encoded_name.split("__", 1)[1]
    for col in CATEGORICAL:
        if base.startswith(col + "_"):
            return col
    return base


def _reason_codes(pipeline, data: pd.DataFrame, means: dict, top_n: int = 3) -> list:
    """Top features pushing each customer towards churn (coefficient x deviation from the average customer)."""
    encoded = pipeline.named_steps["preprocessor"].transform(data)
    coefs = pd.Series(pipeline.named_steps["classifier"].coef_[0], index=encoded.columns)
    contrib = (encoded - pd.Series(means).reindex(encoded.columns)) * coefs
    grouped = contrib.T.groupby(_original_feature).sum().T
    reasons = []
    for _, row in grouped.iterrows():
        drivers = row[row > 0].sort_values(ascending=False).head(top_n)
        reasons.append("; ".join(drivers.index) if len(drivers) else "none")
    return reasons


def score(df: pd.DataFrame) -> pd.DataFrame:
    pipeline = joblib.load(MODEL_PATH)
    card = json.loads(CARD_PATH.read_text())
    data = clean(df)

    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    data = data.reset_index(drop=True)

    proba = pipeline.predict_proba(data[FEATURES])[:, 1]
    tiers = card.get("risk_tiers", {"high_min": 0.50, "medium_min": 0.30})
    tier = pd.cut(proba, [-0.01, tiers["medium_min"] - 1e-9, tiers["high_min"] - 1e-9, 1.0],
                  labels=["Low", "Medium", "High"])
    out = pd.DataFrame({
        "churn_probability": proba.round(4),
        "risk_tier": tier.astype(str),
        "contact": (proba >= card["decision_threshold"]).astype(int),
    })
    if "encoded_feature_means" in card:
        out["top_risk_drivers"] = _reason_codes(pipeline, data[FEATURES], card["encoded_feature_means"])
    if "customerID" in data.columns:
        out.insert(0, "customerID", data["customerID"].to_numpy())
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn risk.")
    parser.add_argument("--input", required=True, help="CSV with the Telco customer columns")
    parser.add_argument("--output", default=None, help="Where to save the scored CSV")
    parser.add_argument("--top", type=int, default=10, help="How many highest-risk customers to print")
    args = parser.parse_args()

    scored = score(pd.read_csv(args.input))
    threshold = json.loads(CARD_PATH.read_text())["decision_threshold"]
    print(f"Customers scored: {len(scored):,} | flagged for retention contact (p >= {threshold}): {int(scored['contact'].sum()):,}")
    print(scored["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).to_string())
    print(scored.head(args.top).to_string(index=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False)
        print("Saved:", args.output)


if __name__ == "__main__":
    main()
