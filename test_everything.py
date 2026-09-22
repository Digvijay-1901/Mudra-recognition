import _quiet
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

def evaluate(csv_path, model_dir, label):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("f")]

    model = joblib.load(f"{model_dir}/mlp_mudra_classifier.joblib")
    scaler = joblib.load(f"{model_dir}/feature_scaler.joblib")
    encoder = joblib.load(f"{model_dir}/label_encoder.joblib")

    X = df[feature_cols].values
    y = encoder.transform(df["label"].values)

    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val_scaled = scaler.transform(X_val)
    preds = model.predict(X_val_scaled)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")

    print(f"\n{'='*60}")
    print(f"{label}  —  {len(encoder.classes_)} classes")
    print(f"{'='*60}")
    print(f"Held-out accuracy: {acc:.4f}")
    print(f"Held-out macro F1: {f1:.4f}\n")
    print(classification_report(y_val, preds, target_names=encoder.classes_))

    report = classification_report(y_val, preds, target_names=encoder.classes_, output_dict=True)
    weak = [(name, s["f1-score"]) for name, s in report.items()
            if name in encoder.classes_ and s["f1-score"] < 0.90]
    if weak:
        print("Classes below 0.90 F1 (worth a closer look):")
        for name, score in sorted(weak, key=lambda x: x[1]):
            print(f"  {name}: {score:.3f}")
    else:
        print("No classes below 0.90 F1 — all healthy.")

    return acc, f1

single_acc, single_f1 = evaluate("keypoints.csv", "model", "SINGLE-HAND")
double_acc, double_f1 = evaluate("keypoints_double.csv", "model_double", "DOUBLE-HAND")

print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")
print(f"Single-hand: {single_acc:.2%} accuracy, {single_f1:.4f} macro F1")
print(f"Double-hand: {double_acc:.2%} accuracy, {double_f1:.4f} macro F1")