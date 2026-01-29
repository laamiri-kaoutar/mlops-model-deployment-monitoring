FROM python:3.11-slim

WORKDIR /app

# System deps (needed for xgboost, matplotlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + data
COPY src/ src/
COPY data/ data/

# MLflow writes here (shared volume)
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "src/training/train.py"]
    