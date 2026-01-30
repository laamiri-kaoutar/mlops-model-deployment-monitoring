import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score

ROOT_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_PATH / "models" / "model.joblib"
DATA_PATH = ROOT_PATH / "data" / "scaled_data_clusters.csv"

def validate_model():
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        raise FileNotFoundError("Modèle ou données introuvables")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop('Cluster', axis=1)
    y_true = df['Cluster']

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