from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "scaled_data_clusters.csv"


def validate_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = [
        "Glucose",
        "Age",
        "BloodPressure",
        "SkinThickness",
        "BMI",
        "Insulin_log",
        "DiabetesPedigreeFunction_log",
        "Cluster",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [
        "Glucose",
        "Age",
        "BloodPressure",
        "SkinThickness",
        "BMI",
        "Insulin_log",
        "DiabetesPedigreeFunction_log",
    ]:
        if df[col].isna().any():
            raise ValueError(f"Null values found in column: {col}")

    if not df["Age"].between(18, 100).all():
        raise ValueError("Age values out of range [18, 100]")

    clusters = set(df["Cluster"].dropna().unique())
    if not clusters.issubset({0, 1}):
        raise ValueError(f"Unexpected Cluster values: {sorted(clusters)}")

    print("Data quality validation successful")


if __name__ == "__main__":
    validate_data()
