import os
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import joblib

from detector import detect
from gate import is_double

from feature_extraction import (hand_tilt_degrees, normalized_single_hand_features,
                                single_hand_features_from_landmarks)
from feature_extraction_double import (double_hand_features_from_results,
                                       normalized_double_hand_features)

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIDENCE_THRESHOLD = 0.6
NORMALIZE_AT_DEGREES = 75.0
CLASS_THRESHOLD_FLOOR = 0.30
CLASS_THRESHOLD_POLICY = "model_trials_20261107/capped_thresholds/cutoffs.json"


@lru_cache(maxsize=1)
def _class_thresholds():
    """Frozen inner-group policy; model/scaler/encoder files are unchanged."""
    with open(os.path.join(BASE, CLASS_THRESHOLD_POLICY), encoding="utf8") as handle:
        policy = json.load(handle)["thresholds"]
    for family in ("single", "single_norm", "double"):
        if not policy.get(family) or not all(
            CLASS_THRESHOLD_FLOOR <= float(value) <= CONFIDENCE_THRESHOLD
            for value in policy[family].values()
        ):
            raise ValueError("Invalid validated class threshold policy")
    return policy


def decision_threshold(family, label, floor):
    """The user's score floor can tighten, but never relax, a class minimum."""
    return max(float(floor), float(_class_thresholds()[family].get(label, CONFIDENCE_THRESHOLD)))


@dataclass
class Prediction:
    label: str
    confidence: float
    mode: str              # "Single-hand" or "Double-hand"
    probabilities: np.ndarray
    class_names: list
    n_hands: int
    results: object        # Landmarks from the selected tracker, for drawing
    used_image: np.ndarray # the image the landmarks belong to
    tilt_degrees: float | None = None


@dataclass
class Abstention:
    message: str
    results: object
    used_image: np.ndarray
    tilt_degrees: float | None = None


@lru_cache(maxsize=None)
def _load(folder):
    path = os.path.join(BASE, folder)
    return (
        joblib.load(os.path.join(path, "mlp_mudra_classifier.joblib")),
        joblib.load(os.path.join(path, "feature_scaler.joblib")),
        joblib.load(os.path.join(path, "label_encoder.joblib")),
    )


def _classify(folder, feats):
    model, scaler, encoder = _load(folder)
    if feats is None or len(feats) != model.n_features_in_:
        return None
    probs = model.predict_proba(scaler.transform([feats]))[0]
    idx = int(np.argmax(probs))
    return str(encoder.classes_[idx]), float(probs[idx]), probs, list(encoder.classes_)


def supported_classes():
    """Canonical dataset names in the active single/double label encoders."""
    from hybrid_single import heads
    return {
        "single": tuple(str(name) for name in heads()["single"][2].classes_),
        "double": tuple(str(name) for name in _load("model_double_v3_norm")[2].classes_),
    }


def predict(image_rgb, mode="auto", *, threshold=CLASS_THRESHOLD_FLOOR):
    """
    mode = "auto"   -> router decides single vs double (can be wrong on occluded double-hand poses)
    mode = "single" -> always the single-hand model
    mode = "double" -> always the double-hand model (it was trained on photos where only one
                       hand is visible, so it also works on those)

    Uses MediaPipe routing/double inference and RTMPose single inference.
    Threshold is a score floor combined with validated class minima (0.30--0.60).
    A floor of 0.60 restores the previous uniform classifier cutoff.
    All callers use the production hybrid. None selects the default 0.30 floor.
    """
    if threshold is None:
        threshold = CLASS_THRESHOLD_FLOOR
    if mode not in {"auto", "single", "double"}:
        raise ValueError("mode must be auto, single or double")
    results, n_hands, used_image = detect(image_rgb)
    if n_hands == 0:
        return None

    hand = results.multi_hand_landmarks[0]
    tilt = hand_tilt_degrees(hand, used_image.shape)

    def build(res, label_mode):
        label, conf, probs, names = res
        return Prediction(label, conf, label_mode, probs, names, n_hands, results, used_image, tilt)

    use_normalized_single = tilt is not None and abs(tilt) >= NORMALIZE_AT_DEGREES

    route = mode
    if route == "auto" and n_hands == 1:
        try:
            # Preserve confident raw-gate decisions: natural double-hand
            # configurations can also have a strongly tilted first hand.
            gate_features = single_hand_features_from_landmarks(hand, used_image.shape)
            _, double_score = is_double(gate_features) if gate_features is not None else (False, 0.50)
            if use_normalized_single and .40 < double_score < .60:
                canonical = normalized_single_hand_features(hand, used_image.shape)
                _, double_score = is_double(canonical)
        except FileNotFoundError:
            return Abstention(
                "The routing gate is missing. Restore gate/gate.joblib or choose a hand mode.",
                results, used_image, tilt,
            )
        if 0.40 < double_score < 0.60:
            return Abstention(
                "Only one hand detected. Show both hands for a two-hand mudra, "
                "or choose Single-hand mode if appropriate.",
                results, used_image, tilt,
            )
        route = "double" if double_score >= 0.60 else "single"
    elif route == "auto":
        route = "double"

    if route == "single":
        # Gate above used only MediaPipe landmarks; now switch the selected
        # single path to RTMPose on the ORIGINAL frame, as validated.
        from hybrid_single import classify
        res, results, n_hands, used_image, tilt, family = classify(image_rgb)
        label_mode = "Single-hand"
    else:
        normalized = normalized_double_hand_features(results, used_image.shape)
        res, label_mode = _classify("model_double_v3_norm", normalized), "Double-hand"
        family = "double"
    if res and res[1] >= decision_threshold(family, res[0], threshold):
        return build(res, label_mode)
    return Abstention(
        "Unsure at the selected confidence threshold. Try a clearer photo.",
        results, used_image, tilt,
    )
