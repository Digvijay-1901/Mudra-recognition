"""Cached inference resources and reference images shared by the UI."""
from pathlib import Path
import streamlit as st

BASE = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Loading class names and classifier heads…")
def load_mlp_models():
    # Vocabulary needs the heads, never the expensive RTMPose tracker.
    from brain import _load
    from hybrid_single import heads
    return {**heads(), "double": _load("model_double_v3_norm")}

@st.cache_resource(show_spinner="Loading routing gate…")
def load_gate_model():
    from gate import _model
    return _model()

@st.cache_resource(show_spinner="Loading MediaPipe hand detector…")
def load_mediapipe_graph():
    from detector import _hands_lock
    from feature_extraction_double import _get_hands
    with _hands_lock:
        return _get_hands(static=True)

def load_class_names():
    models = load_mlp_models()
    return {family: tuple(str(name) for name in models[family][2].classes_)
            for family in ("single", "double")}


@st.cache_data(show_spinner=False, max_entries=34)
def load_reference_image(class_name, group):
    import cv2
    import numpy as np
    folder = BASE / "data" / group / class_name
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            try:
                image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            except (OSError, cv2.error):
                continue
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None
