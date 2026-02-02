import pandas as pd
import mlflow.pyfunc
import pickle  # <--- Added to load the scaler file
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.app.schemas import PatientData
from prometheus_fastapi_instrumentator import Instrumentator
import os

# --- CONFIGURATION ---
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "Diabetes_Model_Prod"
STAGE = "Production"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# CRITICAL: The scaler expects columns in this EXACT order. 
# Do not change this list unless you retrain the scaler.
FEATURE_ORDER = [
    'Glucose', 'Age', 'BloodPressure', 'SkinThickness', 
    'BMI', 'Insulin_log', 'DiabetesPedigreeFunction_log'
]

# Global variables
model = None
scaler = None  # <--- Added global variable for scaler

# --- LIFESPAN MANAGER (Loads model & scaler on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # global model, scaler
    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # 1. Load Model
    try:
        model_uri = f"models:/{MODEL_NAME}/{STAGE}"
        print(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you forget to promote a model to 'Production' in MLflow UI?")

    # 2. Load Scaler (New Logic)
    try:
        print(f"Loading scaler from: {SCALER_PATH}")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        print(f"Make sure '{SCALER_PATH}' exists in the container.")

    yield
    # Clean up (if needed)

app = FastAPI(title="Diabetes Prediction API", lifespan=lifespan)

# --- MONITORING (PROMETHEUS) ---
Instrumentator().instrument(app).expose(app)

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is running"}
@app.post("/predict")
def predict(data: PatientData):
    # global model, scaler
    
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded")
    
    try:
        # 1. Convert Input to DataFrame
        df = pd.DataFrame([data.model_dump()])

        # 2. Enforce Column Order
        df = df[FEATURE_ORDER]

        # 3. Scale the Data (Returns a NumPy Array without column names)
        scaled_array = scaler.transform(df)
        
        # --- THE FIX IS HERE ---
        # Convert the "nameless" array back into a DataFrame WITH names
        scaled_df = pd.DataFrame(scaled_array, columns=FEATURE_ORDER)
        # -----------------------

        # 4. Make Prediction (Now passing the DataFrame with names)
        prediction = model.predict(scaled_df)
        
        result = int(prediction[0])
        label = "High Risk" if result == 1 else "Low Risk"
        
        return {"prediction": result, "risk_category": label}
        
    except Exception as e:
        # Print the error to logs for debugging
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))