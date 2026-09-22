import _quiet
import os
import random
import cv2
import joblib
import numpy as np
from feature_extraction import extract_single_hand_features

model = joblib.load("model/mlp_mudra_classifier.joblib")
scaler = joblib.load("model/feature_scaler.joblib")
encoder = joblib.load("model/label_encoder.joblib")

DATASET_DIR = "data/dataset"
N_PER_CLASS = 5

results = []

for label in os.listdir(DATASET_DIR):
    cls_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(cls_path):
        continue

    files = os.listdir(cls_path)
    sample = random.sample(files, min(N_PER_CLASS, len(files)))

    for fname in sample:
        img_path = os.path.join(cls_path, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feats = extract_single_hand_features(image_rgb, static=True)

        if feats is None:
            results.append((label, fname, "NO_HAND_DETECTED", 0.0, False))
            continue

        feats_scaled = scaler.transform([feats])
        probs = model.predict_proba(feats_scaled)[0]
        idx = int(np.argmax(probs))
        pred = encoder.inverse_transform([idx])[0]
        conf = float(probs[idx])
        correct = (pred == label)

        results.append((label, fname, pred, conf, correct))

print(f"\n{'TRUE':<15} {'PREDICTED':<15} {'CONF':<8} {'FILE':<35} RESULT")
print("-" * 85)
for true, fname, pred, conf, correct in results:
    mark = "OK" if correct else "X"
    print(f"{true:<15} {pred:<15} {conf:<8.3f} {fname:<35} {mark}")

n_correct = sum(1 for r in results if r[4])
n_total = len(results)
print(f"\nOverall: {n_correct}/{n_total} correct ({n_correct/n_total:.1%})")