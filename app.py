from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "tep_binary_random_forest_model.pkl")
model = joblib.load(model_path)

EXPECTED_FEATURES = model.n_features_in_


class InputData(BaseModel):
    features: list


@app.post("/predict")
def predict(data: InputData):

    if len(data.features) != EXPECTED_FEATURES:
        return {
            "error": f"Model expects {EXPECTED_FEATURES} features, "
                     f"but got {len(data.features)}"
        }

    input_array = np.array(data.features).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    return {
        "prediction": "danger" if prediction == 1 else "safe"
    }