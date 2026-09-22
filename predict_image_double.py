import sys
import cv2
import joblib
import numpy as np
from feature_extraction_double import extract_double_hand_features

model = joblib.load("model_double/mlp_mudra_classifier.joblib")
scaler = joblib.load("model_double/feature_scaler.joblib")
encoder = joblib.load("model_double/label_encoder.joblib")

def predict(image_path, threshold=0.5):
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found", 0.0
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    feats = extract_double_hand_features(image_rgb, static=True)
    if feats is None:
        return "No hand detected", 0.0

    feats_scaled = scaler.transform([feats])
    probs = model.predict_proba(feats_scaled)[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    if top_prob < threshold:
        return "Unsure", top_prob
    return encoder.inverse_transform([top_idx])[0], top_prob

if __name__ == "__main__":
    label, conf = predict(sys.argv[1])
    print(f"{label} ({conf:.3f})")