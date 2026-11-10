"""Cached MediaPipe-feature routing gate; the hybrid applies its own ambiguity band."""
import os
import joblib
from functools import lru_cache

BASE = os.path.dirname(os.path.abspath(__file__))

# Share the existing gate artifact without changing its fitted parameters.
@lru_cache(maxsize=1)
def _model():
    return joblib.load(os.path.join(BASE, "gate", "gate.joblib"))

# Expose the double score; brain.predict separately handles its ambiguity interval.
def is_double(single_feats, threshold=0.5):
    p = float(_model().predict_proba([single_feats])[0][1])
    return p >= threshold, p