"""
Evaluate every dataset image against the currently trained model(s).

Run from the project root:
    python evaluate_dataset.py

This script is deliberately NON-DESTRUCTIVE:
- it never deletes training images
- it records every prediction
- it can repeat the evaluation
- it separates detector failures from classifier mistakes
- it writes CSV reports for later inspection

Important:
The evaluation should NOT be treated as a validation score if these are the
same images used to train the model. A high training-set score can be misleading.
Use a held-out validation/test split for a real generalization measurement.
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import joblib

from feature_extraction import extract_single_hand_features
from feature_extraction_double import extract_double_hand_features


SINGLE_DATASET = Path("data/dataset")
DOUBLE_DATASET = Path("data/dataset_double")

SINGLE_MODEL = Path("model/mlp_mudra_classifier.joblib")
SINGLE_SCALER = Path("model/feature_scaler.joblib")
SINGLE_ENCODER = Path("model/label_encoder.joblib")

DOUBLE_MODEL = Path("model_double/mlp_mudra_classifier.joblib")
DOUBLE_SCALER = Path("model_double/feature_scaler.joblib")
DOUBLE_ENCODER = Path("model_double/label_encoder.joblib")


def load_pipeline(model_path, scaler_path, encoder_path):
    return (
        joblib.load(model_path),
        joblib.load(scaler_path),
        joblib.load(encoder_path),
    )


def image_files(folder):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield p


def evaluate_single(path, model, scaler, encoder):
    image = cv2.imread(str(path))
    if image is None:
        return {
            "status": "image_read_failed",
            "predicted": "",
            "confidence": "",
            "feature_dim": "",
        }

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        features = extract_single_hand_features(rgb, static=True)
    except Exception as exc:
        return {
            "status": f"feature_error:{type(exc).__name__}",
            "predicted": "",
            "confidence": "",
            "feature_dim": "",
        }

    if features is None:
        return {
            "status": "feature_extraction_failed",
            "predicted": "",
            "confidence": "",
            "feature_dim": "",
        }

    feature_dim = len(features)
    if feature_dim != model.n_features_in_:
        return {
            "status": "feature_dimension_mismatch",
            "predicted": "",
            "confidence": "",
            "feature_dim": feature_dim,
        }

    probabilities = model.predict_proba(scaler.transform([features]))[0]
    index = int(np.argmax(probabilities))
    predicted = encoder.inverse_transform([index])[0]

    return {
        "status": "ok",
        "predicted": str(predicted),
        "confidence": float(probabilities[index]),
        "feature_dim": feature_dim,
    }


def evaluate_double(path, model, scaler, encoder):
    image = cv2.imread(str(path))
    if image is None:
        return {
            "status": "image_read_failed",
            "predicted": "",
            "confidence": "",
            "feature_dim": "",
        }

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        features = extract_double_hand_features(rgb, static=True)
    except Exception as exc:
        return {
            "status": f"feature_error:{type(exc).__name__}",
            "predicted": "",
            "confidence": "",
            "feature_dim": "",
        }

    if features is None:
        return {
            "status": "feature_extraction_failed",
            "predicted": "",
            "confidence": "",
            "feature_dim": "",
        }

    feature_dim = len(features)
    if feature_dim != model.n_features_in_:
        return {
            "status": "feature_dimension_mismatch",
            "predicted": "",
            "confidence": "",
            "feature_dim": feature_dim,
        }

    probabilities = model.predict_proba(scaler.transform([features]))[0]
    index = int(np.argmax(probabilities))
    predicted = encoder.inverse_transform([index])[0]

    return {
        "status": "ok",
        "predicted": str(predicted),
        "confidence": float(probabilities[index]),
        "feature_dim": feature_dim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output",
        default="dataset_evaluation.csv",
        help="CSV file for the final run",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    single_model, single_scaler, single_encoder = load_pipeline(
        SINGLE_MODEL, SINGLE_SCALER, SINGLE_ENCODER
    )
    double_model, double_scaler, double_encoder = load_pipeline(
        DOUBLE_MODEL, DOUBLE_SCALER, DOUBLE_ENCODER
    )

    rows = []

    datasets = [
        ("single", SINGLE_DATASET, evaluate_single,
         single_model, single_scaler, single_encoder),
        ("double", DOUBLE_DATASET, evaluate_double,
         double_model, double_scaler, double_encoder),
    ]

    for mode, root, evaluator, model, scaler, encoder in datasets:
        if not root.exists():
            print(f"Skipping missing dataset: {root}")
            continue

        files = list(image_files(root))
        print(f"\n{mode.upper()} DATASET: {len(files)} images")

        for run in range(1, args.runs + 1):
            correct = 0
            usable = 0

            for path in files:
                # Ground-truth label is the immediate parent directory name.
                truth = path.parent.name

                result = evaluator(path, model, scaler, encoder)

                if result["status"] == "ok":
                    usable += 1
                    if result["predicted"] == truth:
                        correct += 1

                # Keep every run for the final CSV. This lets us identify
                # images that are consistently wrong rather than merely noisy.
                rows.append({
                    "run": run,
                    "mode": mode,
                    "path": str(path),
                    "true_label": truth,
                    "predicted_label": result["predicted"],
                    "confidence": result["confidence"],
                    "status": result["status"],
                    "feature_dim": result["feature_dim"],
                    "correct": (
                        result["status"] == "ok"
                        and result["predicted"] == truth
                    ),
                })

            accuracy = correct / usable if usable else 0.0
            print(
                f"run {run}: usable={usable}/{len(files)}, "
                f"accuracy={accuracy:.2%}"
            )

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "mode",
                "path",
                "true_label",
                "predicted_label",
                "confidence",
                "status",
                "feature_dim",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote: {args.output}")

    # Print a concise list of images that failed on EVERY run.
    print("\nCONSISTENTLY WRONG / FAILED IMAGES")
    keys = {}

    for row in rows:
        key = (row["mode"], row["path"])
        keys.setdefault(key, []).append(row)

    count = 0
    for (mode, path), entries in keys.items():
        if all(not e["correct"] for e in entries):
            count += 1
            preds = sorted({
                e["predicted_label"] for e in entries if e["predicted_label"]
            })
            statuses = sorted({e["status"] for e in entries})
            print(
                f"{mode:6s} | {path} | "
                f"predictions={preds} | statuses={statuses}"
            )

    print(f"\nConsistently wrong/failed: {count}")


if __name__ == "__main__":
    main()
