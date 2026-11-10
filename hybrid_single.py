"""Production RTMPose single path, identical to the validated staged hybrid.

MediaPipe routing lives in brain.py. This module never supplies gate features.
Missing RTMPose dependencies come from the existing tracker environment; the
production environment keeps precedence. No assets are downloaded at runtime.
"""
import sys
from pathlib import Path
from functools import lru_cache
from threading import RLock
from types import SimpleNamespace as NS

import cv2
import joblib
import numpy as np
from tracker_v2 import geometry

ROOT = Path(__file__).resolve().parent
TRACKER = ROOT / "tracker_v2"
_lock = RLock()


@lru_cache(maxsize=1)
def heads():
    return {family: joblib.load(TRACKER / folder / "bundle.joblib")
            for family, folder in (("single", "model_single/raw"),
                                   ("single_norm", "model_single"))}


@lru_cache(maxsize=1)
def tracker():
    for site in (TRACKER / ".venv/Lib/site-packages",
                 *sorted((TRACKER / ".venv/lib").glob("python*/site-packages"))):
        if site.exists() and str(site) not in sys.path:
            sys.path.append(str(site))
    from tracker_v2.adapter import Tracker
    return Tracker()


def initialize():
    with _lock:
        heads()
        tracker()


def classify(image_rgb):
    # RTMLib's detector has mutable inference state. Serialize across sessions,
    # including its temporary ONNX session factory override during initialization.
    with _lock:
        hands, _, frame = tracker().detect(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    result = NS(multi_hand_landmarks=[NS(landmark=[
        NS(x=float(x), y=float(y), z=float(z)) for x, y, z in hand
    ]) for hand in hands], multi_handedness=[])
    if not hands:
        return None, result, 0, frame, None, "single"
    hand = result.multi_hand_landmarks[0]
    tilt = geometry.hand_tilt_degrees(hand, frame.shape)
    family = "single_norm" if tilt is not None and abs(tilt) >= 75 else "single"
    features = (geometry.normalized_single_hand_features(hand, frame.shape)
                if family == "single_norm"
                else geometry.single_hand_features_from_landmarks(hand, frame.shape))
    model, scaler, encoder = heads()[family]
    if features is None or len(features) != model.n_features_in_:
        return None, result, len(hands), frame, tilt, family
    probabilities = model.predict_proba(scaler.transform([features]))[0]
    index = int(np.argmax(probabilities))
    prediction = (str(encoder.classes_[index]), float(probabilities[index]),
                  probabilities, list(encoder.classes_))
    return prediction, result, len(hands), frame, tilt, family
