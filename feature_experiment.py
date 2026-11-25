"""Does feature engineering improve the model? (5-fold CV on the training set only; test set untouched.)

Usage (from the project root):  python src/feature_experiment.py
"""
import warnings
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import BINARY, CATEGORICAL, FEATURES, NUMERIC, TARGET, clean

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def add_features(df):
    out = df.copy()
    out["n_addons"] = sum(out[c].eq("Yes").astype(int) for c in ADDONS)
    out["charge_per_addon_slot"] = out["MonthlyCharges"] / (1 + out["n_addons"])
    out["is_autopay"] = out["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    out["is_new_customer"] = (out["tenure"] <= 12).astype(int)
    out["has_protection_bundle"] = (out["OnlineSecurity"].eq("Yes") & out["TechSupport"].eq("Yes")).astype(int)
    return out


def pipeline(num, binary, cat, C=3):
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("bin", "passthrough", binary),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=2000, random_state=42))])


def main():
    df = clean(pd.read_csv(ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"))
    y = df["Churn"].eq("Yes").astype(int)
    X = add_features(df[FEATURES])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    new_num, new_bin = ["n_addons", "charge_per_addon_slot"], ["is_autopay", "is_new_customer", "has_protection_bundle"]
    variants = {
        "Current features (production)": (NUMERIC, BINARY, CATEGORICAL),
        "+ service count and price per add-on slot": (NUMERIC + new_num, BINARY, CATEGORICAL),
        "+ all engineered features": (NUMERIC + new_num, BINARY + new_bin, CATEGORICAL),
        "Without TotalCharges (+ engineered)": (["tenure", "MonthlyCharges"] + new_num, BINARY + new_bin, CATEGORICAL),
    }
    rows = []
    for name, (num, binary, cat) in variants.items():
        r = cross_validate(pipeline(num, binary, cat), X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"])
        rows.append({"Variant": name, "CV ROC-AUC": r["test_roc_auc"].mean(), "CV PR-AUC": r["test_average_precision"].mean(),
                     "ROC-AUC std": r["test_roc_auc"].std()})
    table = pd.DataFrame(rows).round(4)
    table.to_csv(ROOT / "reports" / "feature_experiment.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
