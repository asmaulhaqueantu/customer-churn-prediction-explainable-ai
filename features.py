"""Data cleaning and feature definitions shared by training and scoring."""
import pandas as pd

TARGET = "ChurnFlag"
ID_COL = "customerID"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY = ["SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + BINARY + CATEGORICAL


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Blank TotalCharges (customers with tenure 0) -> 0, then cast to numeric."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    return df
