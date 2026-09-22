import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("keypoints.csv")
feature_cols = [c for c in df.columns if c.startswith("f")]

model = joblib.load("model/mlp_mudra_classifier.joblib")
scaler = joblib.load("model/feature_scaler.joblib")
encoder = joblib.load("model/label_encoder.joblib")

X = df[feature_cols].values
y = encoder.transform(df["label"].values)

# same split logic/seed as training, so this is genuinely held-out data
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_val_scaled = scaler.transform(X_val)
preds = model.predict(X_val_scaled)

print(f"Held-out accuracy: {accuracy_score(y_val, preds):.4f}\n")
print(classification_report(y_val, preds, target_names=encoder.classes_))