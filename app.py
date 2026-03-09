from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import joblib
import numpy as np
import os
import json

from database import init_db
from retrain import retrain_model, SYNTHETIC_SEED  # import seed if needed

init_db()

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CONFIDENCE_THRESHOLD = 0.80  

# -------------------------------
# Load latest model
# -------------------------------
def load_latest_model():
    global model

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    models = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")])

    if len(models) == 0:
        initial_model_path = os.path.join(BASE_DIR, "model", "model_v1.pkl")
        model = joblib.load(initial_model_path)
        print("Loaded base model.")
        return

    latest = models[-1]
    model_path = os.path.join(BASE_DIR, "model", "model_v1.pkl")

    model = joblib.load(model_path)
    print("Loaded:", latest)


load_latest_model()

EXPECTED_FEATURES = model.n_features_in_

# -------------------------------
# Input schema
# -------------------------------
class InputData(BaseModel):
    features: list

# -------------------------------
# Save predictions
# -------------------------------
def save_prediction(features, prediction, confidence):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (features, prediction, confidence)
        VALUES (?, ?, ?)
    """, (json.dumps(features), prediction, confidence))

    conn.commit()
    conn.close()

# -------------------------------
# Automatic retraining trigger
# -------------------------------
def check_retraining():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()

    # retrain every 10 predictions automatically
    if count >= 10 and count % 10 == 0:
        print("Automatic retraining triggered...")
        retrain_model(force=False)
        load_latest_model()
        print("Automatic retraining completed.")

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(data: InputData):

    if len(data.features) != EXPECTED_FEATURES:
        return {"error": f"Invalid feature length. Expected {EXPECTED_FEATURES}"}

    input_array = np.array(data.features).reshape(1, -1)

    probabilities = model.predict_proba(input_array)[0]
    prediction = np.argmax(probabilities)
    confidence = float(np.max(probabilities))

    result = "danger" if prediction == 1 else "safe"

    # save to database
    save_prediction(data.features, result, confidence)

    # check if auto retraining needed
    check_retraining()

    return {
        "prediction": result,
        "confidence": confidence
    }

# -------------------------------
# Manual retraining endpoint
# -------------------------------
@app.post("/retrain")
def force_retrain():

    print("Force retraining triggered...")

    # force retrain on entire DB + synthetic seed
    retrain_model(force=True)

    # reload latest model
    load_latest_model()

    return {
        "status": "success",
        "message": "Force retraining completed and model reloaded"
    }
