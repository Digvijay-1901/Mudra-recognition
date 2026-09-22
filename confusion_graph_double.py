import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("keypoints_double.csv")
feature_cols = [c for c in df.columns if c.startswith("f")]

model = joblib.load("model_double/mlp_mudra_classifier.joblib")
scaler = joblib.load("model_double/feature_scaler.joblib")
encoder = joblib.load("model_double/label_encoder.joblib")

X = df[feature_cols].values
y = encoder.transform(df["label"].values)

_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val_scaled = scaler.transform(X_val)
preds = model.predict(X_val_scaled)

cm = confusion_matrix(y_val, preds, normalize="true")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Double-Hand Held-out Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_double.png")
print("Saved confusion_matrix_double.png")