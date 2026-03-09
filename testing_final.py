import os
import random
import time
import warnings

import joblib
import requests

url = "http://44.223.75.58/predict"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model/model_v1.pkl")

REQUEST_INTERVAL_SECONDS = 120
MAX_GENERATION_ATTEMPTS = 60
JITTER_FRACTION = 0.25

# Equal-probability target selection for warnings.
TARGET_CLASSES = ("safe", "danger")

# Most sensitive dimensions to push outside safe bands for danger generation.
DANGER_PUSH_FEATURES = [17, 18, 45, 49, 50, 51]

warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator.*",
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
)

model = joblib.load(MODEL_PATH)
model.n_jobs = 1

# This anchor vector is known to predict SAFE (0) with your current model.
SAFE_ANCHOR = [
    0.257095, 3660.1, 4508.5, 9.34815, 26.872, 42.3255, 2705.95, 75.2165,
    120.4, 0.33999, 80.0315, 50.0825, 2634.95, 25.1605, 50.067, 3102.0,
    22.8315, 65.741, 231.67, 341.16, 94.5865, 77.2865, 32.194, 8.92955,
    26.299, 6.8723, 18.75, 1.6605, 32.9175, 13.854, 23.9855, 1.25406,
    18.6445, 2.2597, 4.8364, 2.30075, 0.016892, 0.84338, 0.0989105, 53.6685,
    43.8735, 63.179, 54.1065, 25.164, 61.202, 22.137, 40.4435, 38.344, 46.69,
    48.1555, 41.258, 18.424
]

# Model-guided safe bands (feature-wise around the safe anchor).
SAFE_RANGES = [
    (0.12245, 0.39174), (3511.8, 3808.4), (4336.9, 4680.1), (8.9727, 9.7236),
    (26.2905, 27.5125), (41.394, 43.257), (2672.8, 2726.7501), (72.919, 77.514),
    (120.345, 120.455), (0.28518, 0.3948), (79.267, 80.8685), (45.874, 54.291),
    (2614.15, 2662.2501), (20.752, 29.569), (45.853, 54.281), (3079.4501, 3122.5502),
    (20.125, 25.538), (65.4575, 66.1925), (227.45, 240.85), (336.955, 346.665),
    (94.237, 95.0685), (76.3695, 78.4045), (30.97, 33.418), (8.4834, 9.3757),
    (24.954, 27.644), (6.4204, 7.3242), (17.527, 19.973), (1.5549, 1.7661),
    (31.367, 34.468), (13.399, 14.309), (22.389, 25.582), (0.82372, 1.6844),
    (17.184, 20.105), (2.1504, 2.369), (4.5311, 5.1417), (2.077, 2.5245),
    (-0.021217, 0.055001), (0.78, 0.90211), (0.060641, 0.13718), (51.564, 55.773),
    (41.768, 45.979), (60.58, 65.778), (52.095, 56.118), (11.977, 36.193),
    (55.961, 65.9695), (21.3925, 23.1515), (33.389, 47.498), (25.959, 50.729),
    (36.937, 56.443), (45.715, 49.2635), (39.4215, 42.7205), (12.086, 20.1355)
]

# Wider process ranges used for generic random generation.
BASE_RANGES = [
    (0.12245, 0.39174), (3511.8, 3808.4), (4336.9, 4680.1), (8.9727, 9.7236),
    (25.951, 27.793), (41.394, 43.257), (2672.8, 2739.1), (72.919, 77.514),
    (120.32, 120.48), (0.28518, 0.3948), (78.98, 81.083), (45.874, 54.291),
    (2600.4, 2669.5), (20.752, 29.569), (45.853, 54.281), (3072.7, 3131.3),
    (20.125, 25.538), (63.952, 67.53), (187.44, 275.9), (334.22, 348.1),
    (93.967, 95.206), (76.141, 78.432), (30.97, 33.418), (8.4834, 9.3757),
    (24.954, 27.644), (6.4204, 7.3242), (17.527, 19.973), (1.5549, 1.7661),
    (31.367, 34.468), (13.399, 14.309), (22.389, 25.582), (0.82372, 1.6844),
    (17.184, 20.105), (2.1504, 2.369), (4.5311, 5.1417), (2.077, 2.5245),
    (-0.021217, 0.055001), (0.75923, 0.92753), (0.060641, 0.13718), (51.564, 55.773),
    (41.768, 45.979), (60.58, 65.778), (52.095, 56.118), (11.977, 38.351),
    (55.961, 66.443), (19.749, 24.525), (33.389, 47.498), (25.959, 50.729),
    (36.937, 56.443), (36.695, 59.616), (38.586, 43.93), (12.086, 24.762)
]

class_counts = {"safe": 0, "danger": 0}


def choose_balanced_target(class_counts):
    safe_count = class_counts["safe"]
    danger_count = class_counts["danger"]

    if safe_count == danger_count:
        return random.choice(TARGET_CLASSES)
    return "safe" if safe_count < danger_count else "danger"


def generate_safe_features():
    # Keep each feature close to a known safe anchor while preserving variability.
    features = []
    for anchor, (low, high) in zip(SAFE_ANCHOR, SAFE_RANGES):
        safe_width = high - low
        window = safe_width * JITTER_FRACTION
        local_low = max(low, anchor - (window / 2))
        local_high = min(high, anchor + (window / 2))
        features.append(random.uniform(local_low, local_high))
    return features


def sample_outside_safe(base_low, base_high, safe_low, safe_high):
    outside_segments = []
    if safe_low > base_low:
        outside_segments.append((base_low, safe_low))
    if safe_high < base_high:
        outside_segments.append((safe_high, base_high))

    if not outside_segments:
        return random.uniform(base_low, base_high)

    low, high = random.choice(outside_segments)
    return random.uniform(low, high)


def generate_danger_features():
    features = [random.uniform(low, high) for (low, high) in BASE_RANGES]

    for idx in DANGER_PUSH_FEATURES:
        base_low, base_high = BASE_RANGES[idx]
        safe_low, safe_high = SAFE_RANGES[idx]
        features[idx] = sample_outside_safe(base_low, base_high, safe_low, safe_high)

    return features


def local_predict_label(features):
    pred = int(model.predict([features])[0])
    return "danger" if pred == 1 else "safe"


def generate_features_for_target(target_label):
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        if target_label == "safe":
            candidate = generate_safe_features()
        else:
            candidate = generate_danger_features()

        if local_predict_label(candidate) == target_label:
            return candidate, attempt

    # deterministic fallback after too many misses
    if target_label == "safe":
        return SAFE_ANCHOR[:], MAX_GENERATION_ATTEMPTS

    # For danger fallback, keep sampling until a local danger is found.
    attempts = MAX_GENERATION_ATTEMPTS
    while True:
        attempts += 1
        candidate = generate_danger_features()
        if local_predict_label(candidate) == "danger":
            return candidate, attempts


while True:
    target_label = choose_balanced_target(class_counts)
    class_counts[target_label] += 1
    features, attempts = generate_features_for_target(target_label)

    try:
        response = requests.post(url, json={"features": features}, timeout=10)
        data = response.json()
        api_prediction = data.get("prediction")
        print(
            {
                "target": target_label,
                "api_prediction": api_prediction,
                "matched_target": api_prediction == target_label,
                "generation_attempts": attempts,
            }
        )

    except Exception as e:
        print("Request failed:", e)

    time.sleep(REQUEST_INTERVAL_SECONDS)

