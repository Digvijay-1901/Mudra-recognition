"""Cached inference resources and reference images shared by the UI."""
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

BASE = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def load_mlp_models():
    # Prewarm the exact production hybrid, including its RTMPose detector.
    from brain import _load
    from hybrid_single import initialize, heads
    initialize()
    return {**heads(), "double": _load("model_double_v3_norm")}

@st.cache_resource(show_spinner=False)
def load_gate_model():
    from gate import _model
    return _model()

@st.cache_resource(show_spinner=False)
def load_mediapipe_graph():
    from detector import _hands_lock
    from feature_extraction_double import _get_hands
    with _hands_lock:
        return _get_hands(static=True)

@st.cache_resource(show_spinner=False)
def initialize_ai_resources():
    with st.spinner("Loading AI models..."):
        return load_mlp_models(), load_gate_model(), load_mediapipe_graph()

@st.cache_data(show_spinner=False, max_entries=34)
def load_reference_image(class_name, group):
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
