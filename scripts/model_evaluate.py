import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

ROOT_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_PATH / "models" / "model.joblib"
DATA_PATH = ROOT_PATH / "data" / "scaled_data_clusters.csv"
FEATURES = [
    "Glucose",
    "Age",
    "BloodPressure",
    "SkinThickness",
    "BMI",
    "Insulin_log",
    "DiabetesPedigreeFunction_log",
]


def validate_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Données introuvables: {}".format(DATA_PATH))

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + ["Cluster"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for evaluation: {missing}")

    X = df[FEATURES]
    y_true = df["Cluster"]

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y_true)
        joblib.dump(model, MODEL_PATH)
    else:
        model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)

    THRESHOLD = 0.80

    print(f"Model Accuracy: {accuracy:.2f}")

    if accuracy < THRESHOLD:
        raise ValueError(f"Performance insuffisante : {accuracy:.2f} < {THRESHOLD}")
    else:
        print("✅ Validation des performances réussie")


if __name__ == "__main__":
    validate_model()
