import os
import datetime
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("keypoints_double.csv")
feature_cols = [c for c in df.columns if c.startswith("f")]

X = df[feature_cols].values
y_raw = df["label"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate="adaptive",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

val_preds = mlp.predict(X_val_scaled)
val_acc = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds, average="macro")

print(f"Held-out validation accuracy: {val_acc:.4f}")
print(f"Held-out validation macro F1: {val_f1:.4f}")

os.makedirs("model_double", exist_ok=True)
joblib.dump(mlp, "model_double/mlp_mudra_classifier.joblib")
joblib.dump(scaler, "model_double/feature_scaler.joblib")
joblib.dump(encoder, "model_double/label_encoder.joblib")

with open("model_double/MODEL_VERSION.txt", "w") as f:
    f.write(f"trained: {datetime.datetime.now()}\n")
    f.write(f"feature_dim: {X.shape[1]}\n")
    f.write(f"n_classes: {len(encoder.classes_)}\n")
    f.write(f"classes: {list(encoder.classes_)}\n")
    f.write(f"held_out_val_accuracy: {val_acc:.4f}\n")
    f.write(f"held_out_val_macro_f1: {val_f1:.4f}\n")

print("Saved model, scaler, encoder, and MODEL_VERSION.txt")