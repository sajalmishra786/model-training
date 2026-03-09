import sqlite3
import json
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
import random

DB_PATH = "predictions.db"
MODEL_DIR = "models"

RETRAIN_WINDOW = 20
BALANCE_PER_CLASS = 5
CONFIDENCE_THRESHOLD = 0.85

# Tiny synthetic seed to guarantee both classes
SYNTHETIC_SEED = [
    # safe
    {"features": [0.1]*50, "label": 0},
    # danger
    {"features": [1.0]*50, "label": 1}
]


def retrain_model(force=False):
    print("Starting retraining...")

    conn = sqlite3.connect(DB_PATH)

    if force:
        print("Force retrain: using entire database")
        query = "SELECT features, prediction, confidence FROM predictions"
    else:
        query = f"""
            SELECT features, prediction, confidence
            FROM predictions
            ORDER BY id DESC
            LIMIT {RETRAIN_WINDOW}
        """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No data available for retraining.")
        return

    # -------------------------
    # CONFIDENCE FILTER
    # -------------------------
    if not force:
        df = df[df["confidence"] >= CONFIDENCE_THRESHOLD]
        if df.empty:
            print("No high-confidence data. Skipping retrain.")
            return

    # -------------------------
    # PREPARE DATA
    # -------------------------
    df["features"] = df["features"].apply(json.loads)
    df["label"] = df["prediction"].map({"safe": 0, "danger": 1})

    if force:
        # Add synthetic seed to ensure both classes exist
        df = pd.concat([df, pd.DataFrame(SYNTHETIC_SEED)])
    
    safe_df = df[df["label"] == 0]
    danger_df = df[df["label"] == 1]

    if not force:
        if len(safe_df) < 1 or len(danger_df) < 1:
            print("Imbalanced data. Skipping retrain.")
            return

        # Sample a balanced subset
        safe_sample = safe_df.sample(min(BALANCE_PER_CLASS, len(safe_df)), random_state=42)
        danger_sample = danger_df.sample(min(BALANCE_PER_CLASS, len(danger_df)), random_state=42)
        balanced = pd.concat([safe_sample, danger_sample])
    else:
        # Force retrain → use all data + synthetic seed
        balanced = df

    X = pd.DataFrame(balanced["features"].tolist())
    y = balanced["label"]

    # -------------------------
    # TRAIN MODEL
    # -------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    existing_models = [
        f for f in os.listdir(MODEL_DIR)
        if f.startswith("model_v") and f.endswith(".pkl")
    ]

    version = 1 if not existing_models else max([int(f.split("_v")[1].split(".")[0]) for f in existing_models]) + 1
    version_path = os.path.join(MODEL_DIR, f"model_v{version}.pkl")

    joblib.dump(model, version_path)
    joblib.dump(model, os.path.join(MODEL_DIR, "latest_model.pkl"))

    print(f"Model saved as model_v{version}.pkl")

    # -------------------------
    # DELETE OLDEST MODEL IF >3
    # -------------------------
    existing_models = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.startswith("model_v")],
        key=lambda x: int(x.split("_v")[1].split(".")[0])
    )

    while len(existing_models) > 3:
        oldest = existing_models.pop(0)
        os.remove(os.path.join(MODEL_DIR, oldest))
        print(f"Deleted old model: {oldest}")

    print("Retraining completed successfully.")
