import os
import sys
import cv2
import joblib
import numpy as np
from collections import defaultdict

from feature_extraction import extract_single_hand_features
from feature_extraction_double import extract_double_hand_features

SINGLE_DATASET = "data/dataset"
DOUBLE_DATASET = "data/dataset_double"

SINGLE_MODEL = "model/mlp_mudra_classifier.joblib"
SINGLE_SCALER = "model/feature_scaler.joblib"
SINGLE_ENCODER = "model/label_encoder.joblib"

DOUBLE_MODEL = "model_double/mlp_mudra_classifier.joblib"
DOUBLE_SCALER = "model_double/feature_scaler.joblib"
DOUBLE_ENCODER = "model_double/label_encoder.joblib"


def bar(value, width=25):
    value = max(0.0, min(1.0, value))
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def color(text, code):
    return f"\033[{code}m{text}\033[0m"


def accuracy_color(acc):
    if acc >= 0.90:
        return color(f"{acc * 100:6.2f}%", "92")
    if acc >= 0.75:
        return color(f"{acc * 100:6.2f}%", "93")
    return color(f"{acc * 100:6.2f}%", "91")


def header(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def load_bundle(model_path, scaler_path, encoder_path):
    return (
        joblib.load(model_path),
        joblib.load(scaler_path),
        joblib.load(encoder_path),
    )


def evaluate(dataset_path, model_path, scaler_path, encoder_path,
             extractor, title):

    model, scaler, encoder = load_bundle(
        model_path, scaler_path, encoder_path
    )

    classes = list(encoder.classes_)

    stats = {
        label: {"total": 0, "correct": 0, "wrong": 0, "fail": 0}
        for label in classes
    }

    confusion = defaultdict(lambda: defaultdict(int))
    image_files = []

    for label in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, label)

        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                image_files.append(
                    (label, os.path.join(class_dir, filename))
                )

    total = len(image_files)

    header(title)
    print(f"Dataset images : {total}")
    print(f"Model classes  : {len(classes)}")
    print()

    processed = 0
    total_correct = 0
    total_failed = 0

    for expected, path in image_files:
        image = cv2.imread(path)

        if image is None:
            stats[expected]["fail"] += 1
            total_failed += 1
            processed += 1
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            features = extractor(image_rgb, static=True)
        except Exception:
            features = None

        stats[expected]["total"] += 1

        if features is None or len(features) != model.n_features_in_:
            stats[expected]["fail"] += 1
            total_failed += 1
        else:
            probabilities = model.predict_proba(
                scaler.transform([features])
            )[0]

            prediction_index = int(np.argmax(probabilities))
            predicted = encoder.inverse_transform(
                [prediction_index]
            )[0]

            confusion[expected][predicted] += 1

            if predicted == expected:
                stats[expected]["correct"] += 1
                total_correct += 1
            else:
                stats[expected]["wrong"] += 1

        processed += 1

        if processed == total or processed % 25 == 0:
            percent = processed / max(total, 1)
            sys.stdout.write(
                f"\r  Progress: {processed:5d}/{total:5d} "
                f"[{bar(percent, 30)}] {percent * 100:6.2f}%"
            )
            sys.stdout.flush()

    print()
    print()

    evaluated = total - total_failed
    overall_accuracy = total_correct / evaluated if evaluated else 0.0

    print(
        f"Accuracy        : {accuracy_color(overall_accuracy)} "
        f" {total_correct}/{evaluated}"
    )
    print(f"Feature failures: {total_failed}")
    print()

    print("  PER-CLASS RESULTS")
    print("  " + "-" * 66)
    print(
        f"  {'Mudra':<20} {'Correct':>8} {'Total':>8} "
        f"{'Accuracy':>10}  Graph"
    )
    print("  " + "-" * 66)

    for label in sorted(classes):
        s = stats[label]
        evaluated_class = s["correct"] + s["wrong"]
        acc = s["correct"] / evaluated_class if evaluated_class else 0.0

        print(
            f"  {label:<20} "
            f"{s['correct']:>8} "
            f"{evaluated_class:>8} "
            f"{accuracy_color(acc)}  "
            f"{bar(acc, 20)}"
        )

    print()
    print("  TOP CONFUSIONS")
    print("  " + "-" * 66)

    mistakes = []

    for expected in confusion:
        for predicted, count in confusion[expected].items():
            if expected != predicted:
                mistakes.append((count, expected, predicted))

    mistakes.sort(reverse=True)

    if mistakes:
        for count, expected, predicted in mistakes[:15]:
            print(
                f"  {expected:<20} -> {predicted:<20} "
                f"{count:>5} images"
            )
    else:
        print("  None 🎯")

    return {
        "total": total,
        "evaluated": evaluated,
        "correct": total_correct,
        "failed": total_failed,
        "accuracy": overall_accuracy,
    }


def main():
    print()
    print("=" * 72)
    print("              MUDRA MODEL SELF-TEST V2")
    print("=" * 72)
    print()
    print("Running the dataset through the trained inference pipeline...")
    print("This is an in-sample dataset check, not a held-out test.")
    print()

    single = evaluate(
        SINGLE_DATASET,
        SINGLE_MODEL,
        SINGLE_SCALER,
        SINGLE_ENCODER,
        extract_single_hand_features,
        "SINGLE-HAND MODEL"
    )

    double = evaluate(
        DOUBLE_DATASET,
        DOUBLE_MODEL,
        DOUBLE_SCALER,
        DOUBLE_ENCODER,
        extract_double_hand_features,
        "DOUBLE-HAND MODEL"
    )

    total = single["evaluated"] + double["evaluated"]
    correct = single["correct"] + double["correct"]
    failed = single["failed"] + double["failed"]

    overall = correct / total if total else 0.0

    header("FINAL SUMMARY")

    print(f"  Single-hand accuracy : {single['accuracy'] * 100:6.2f}%")
    print(f"  Double-hand accuracy : {double['accuracy'] * 100:6.2f}%")
    print(f"  Combined accuracy    : {overall * 100:6.2f}%")
    print(f"  Correct              : {correct}")
    print(f"  Evaluated            : {total}")
    print(f"  Feature failures     : {failed}")
    print()

    if failed == 0:
        print(color("  🟢 ALL IMAGES PROCESSED SUCCESSFULLY", "92"))
    else:
        print(
            color(
                f"  🟡 {failed} IMAGE(S) HAD FEATURE-EXTRACTION FAILURES",
                "93"
            )
        )

    print()
    print("=" * 72)
    print("  This is an end-to-end dataset health check.")
    print("  It does NOT replace your held-out validation result.")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
