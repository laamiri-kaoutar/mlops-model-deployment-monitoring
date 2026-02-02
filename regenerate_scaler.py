import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
# Path to your RAW training data (make sure this file exists)
DATA_PATH = "data/scaled_data_clusters.csv"
# Output path for the new scaler
OUTPUT_PATH = "src/app/scaler.pkl"


def generate_scaler():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Could not find data at {DATA_PATH}")
        print("Please check where your 'cleaned_data_clusters.csv' file is located.")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # EXACT feature columns used in your API
    features = [
        "Glucose",
        "Age",
        "BloodPressure",
        "SkinThickness",
        "BMI",
        "Insulin_log",
        "DiabetesPedigreeFunction_log",
    ]

    print("Fitting scaler...")
    scaler = StandardScaler()
    scaler.fit(df[features])

    print(f"Saving new scaler to {OUTPUT_PATH}...")
    # protocol=4 ensures compatibility between Python 3.8 and 3.12+
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(scaler, f, protocol=4)

    print("✅ Success! New scaler.pkl generated.")


if __name__ == "__main__":
    generate_scaler()
