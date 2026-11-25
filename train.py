"""Reproducible training script (no SHAP needed).

Rebuilds the production model from the raw data with exactly the settings chosen in the notebook,
then writes the model, the model card and the business reports (calibration, risk tiers, lift).

Usage (from the project root):
    python src/train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import BINARY, CATEGORICAL, FEATURES, NUMERIC, TARGET, clean

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
BEST_C = 3                      # chosen by cross-validation in the notebook
OFFER_COST, CUSTOMER_VALUE, SUCCESS_RATE = 1.0, 8.0, 0.40   # assumptions (relative units)
HIGH_RISK, MEDIUM_RISK = 0.50, 0.30                          # tier boundaries (probability)


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False))]),
         CATEGORICAL),
    ]).set_output(transform="pandas")
    return Pipeline([("preprocessor", pre),
                     ("classifier", LogisticRegression(C=BEST_C, max_iter=2000, random_state=RANDOM_STATE))])


def metrics(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred), "F1": f1_score(y, pred),
        "ROC-AUC": roc_auc_score(y, proba), "PR-AUC": average_precision_score(y, proba),
    }


def business_cost(y, proba, threshold):
    contact = proba >= threshold
    tp = int((contact & (y == 1)).sum()); fp = int((contact & (y == 0)).sum()); fn = int((~contact & (y == 1)).sum())
    return (tp + fp) * OFFER_COST + tp * (1 - SUCCESS_RATE) * CUSTOMER_VALUE + fn * CUSTOMER_VALUE


def main():
    df = clean(pd.read_csv(ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"))
    df[TARGET] = df["Churn"].eq("Yes").astype(int)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    # 1) Threshold from out-of-fold predictions on the training set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(build_pipeline(), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    threshold = float(min(grid, key=lambda t: business_cost(y_train.to_numpy(), oof, t)))

    # 2) Calibration check (out-of-fold): are predicted probabilities trustworthy as percentages?
    frac_pos, mean_pred = calibration_curve(y_train, oof, n_bins=10, strategy="quantile")
    calib = pd.DataFrame({"Mean predicted probability": mean_pred, "Observed churn rate": frac_pos}).round(4)
    calib.to_csv(ROOT / "reports" / "calibration_table.csv", index=False)
    brier, brier_ref = brier_score_loss(y_train, oof), brier_score_loss(y_train, np.full(len(y_train), y_train.mean()))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#C44E52", label="Logistic Regression (out-of-fold)")
    ax.set_xlabel("Mean predicted churn probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration curve (training data, 5-fold)"); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "figures" / "15_calibration_curve.png", dpi=150); plt.close(fig)

    # 3) Risk tiers: what share of customers falls in each tier and how many of them really churn?
    tiers = pd.cut(oof, [-0.01, MEDIUM_RISK, HIGH_RISK, 1.0], labels=["Low", "Medium", "High"])
    tier_table = (pd.DataFrame({"tier": tiers, "churn": y_train.to_numpy()})
                  .groupby("tier", observed=True)["churn"].agg(Customers="size", Churn_rate="mean").reset_index())
    tier_table["Share_of_customers"] = tier_table["Customers"] / tier_table["Customers"].sum()
    tier_table["Share_of_all_churners"] = (
        tier_table["Customers"] * tier_table["Churn_rate"] / y_train.sum())
    tier_table.round(4).to_csv(ROOT / "reports" / "risk_tiers.csv", index=False)

    # 4) Final fit and the one-time test evaluation
    model = build_pipeline().fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    test_metrics = metrics(y_test.to_numpy(), proba_test, threshold)

    # 5) Lift on the test set
    ranked = pd.DataFrame({"p": proba_test, "y": y_test.to_numpy()}).sort_values("p", ascending=False).reset_index(drop=True)
    lift_rows = []
    for frac in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        top = ranked.head(int(len(ranked) * frac))
        captured = top["y"].sum() / ranked["y"].sum()
        lift_rows.append({"Contacted share": frac, "Churners captured": round(captured, 4),
                          "Lift vs random": round(captured / frac, 2)})
    pd.DataFrame(lift_rows).to_csv(ROOT / "reports" / "lift_table.csv", index=False)

    # 6) Save model and model card (feature means enable per-customer reason codes in predict.py)
    names = list(model.named_steps["preprocessor"].get_feature_names_out())
    means = model.named_steps["preprocessor"].transform(X_train).mean().to_dict()
    joblib.dump(model, ROOT / "models" / "churn_pipeline.joblib")
    card = {
        "model": "Logistic Regression",
        "best_params": {"C": BEST_C},
        "decision_threshold": threshold,
        "risk_tiers": {"high_min": HIGH_RISK, "medium_min": MEDIUM_RISK},
        "cost_assumptions": {"offer_cost": OFFER_COST, "customer_value": CUSTOMER_VALUE, "retention_success_rate": SUCCESS_RATE},
        "test_metrics_at_threshold": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "calibration": {"brier_score": round(float(brier), 4), "brier_reference_no_skill": round(float(brier_ref), 4)},
        "features": FEATURES,
        "encoded_feature_means": {k: round(float(v), 6) for k, v in means.items()},
        "library_versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__},
    }
    (ROOT / "models" / "model_card.json").write_text(json.dumps(card, indent=2))

    print(f"Threshold (cost-based, training OOF): {threshold:.2f}")
    print("Test metrics:", {k: round(float(v), 4) for k, v in test_metrics.items()})
    print(f"Brier score {brier:.4f} (no-skill {brier_ref:.4f})")
    print(tier_table.round(3).to_string(index=False))
    print(pd.DataFrame(lift_rows).to_string(index=False))
    print(calib.to_string(index=False))


if __name__ == "__main__":
    main()
