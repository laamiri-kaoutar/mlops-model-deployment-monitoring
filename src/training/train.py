# src/training/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------
# CONFIGURATION
# ----------------------
MLFLOW_URI = "http://mlflow:5000"  # mlflow server inside docker-compose network
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Diabetes_Risk_Comparison")

DATA_PATH = "data/scaled_data_clusters.csv"


# ----------------------
# UTILITY FUNCTIONS
# ----------------------
def log_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"confusion_matrix_{model_name}.png")
    mlflow.log_artifact(f"confusion_matrix_{model_name}.png", artifact_path="plots")
    plt.close()


def log_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.savefig(f"roc_curve_{model_name}.png")
    mlflow.log_artifact(f"roc_curve_{model_name}.png", artifact_path="plots")
    plt.close()


# ----------------------
# MAIN TRAIN FUNCTION
# ----------------------
def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    features = [
        "Glucose",
        "Age",
        "BloodPressure",
        "SkinThickness",
        "BMI",
        "Insulin_log",
        "DiabetesPedigreeFunction_log",
    ]
    target = "Cluster"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10),
        "SVM": SVC(probability=True),
        "Gradient_Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    }

    # Train each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        with mlflow.start_run(run_name=model_name):
            # Tags for UI
            mlflow.set_tag("model_family", model.__class__.__name__)
            mlflow.set_tag("dataset", "scaled_data_clusters_v1")
            mlflow.set_tag("task", "classification")

            # Log hyperparameters
            mlflow.log_params(model.get_params())

            # Train
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            # Some models like SVM have no predict_proba for multi-class
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred  # fallback for metrics that require probabilities

            # Log metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
            }
            # Only log roc_auc if we have probabilities
            if hasattr(model, "predict_proba"):
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            mlflow.log_metrics(metrics)

            # Artifacts
            log_confusion_matrix(y_test, y_pred, model_name)
            if hasattr(model, "predict_proba"):
                log_roc_curve(y_test, y_proba, model_name)

            # Feature importance
            if hasattr(model, "feature_importances_"):
                fi = pd.DataFrame(
                    {"feature": features, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)
                fi.to_csv(f"feature_importance_{model_name}.csv", index=False)
                mlflow.log_artifact(
                    f"feature_importance_{model_name}.csv",
                    artifact_path="explainability",
                )

            # Log model
            if "XGBoost" in model_name:
                mlflow.xgboost.log_model(
                    model, artifact_path="model", input_example=X_test.iloc[:5]
                )
            else:
                mlflow.sklearn.log_model(
                    model, artifact_path="model", input_example=X_test.iloc[:5]
                )

            print(f"--> {model_name} logged successfully!")

    print("DONE — open MLflow UI and check artifacts, metrics, and models.")


# ----------------------
if __name__ == "__main__":
    main()
