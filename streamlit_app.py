"""Page composition; inference, routing and temporal voting remain in their existing modules."""
import _quiet
import hashlib
import logging
import streamlit as st
from app_theme import render_header, render_footer

st.set_page_config(page_title="Bharatanatyam Mudra Recognition", page_icon="🙏", layout="wide")
render_header()
ready = st.empty()
ready.markdown('<span class="status-pill">Models on demand · ready to load</span>', unsafe_allow_html=True)

from app_resources import load_class_names
from app_telemetry import append_diagnostic_row
from app_capture import MODE_KEYS, reset_temporal_history, capture_changed, infer_image, diagnostic_capture
from app_ui import render_result_card, render_diagnostic, render_supported

# Preserve ground truth while its widget is hidden; it never selects a prediction.
st.session_state["expected_mudra"] = st.session_state.get("expected_mudra", "Pathaka")

with st.sidebar:
    st.subheader("Capture settings")
    threshold = st.slider(
        "Minimum score floor", .30, .95, .30, .05, key="confidence_threshold",
        on_change=reset_temporal_history,
        help="Validated class minimums range from 30% to 60%. This floor can make them stricter. Scores are not calibrated probabilities.",
    )
    st.caption("Validated class cutoffs apply. A higher floor can reject more images.")
    hand_mode = st.radio("Hand mode", list(MODE_KEYS), key="hand_mode", on_change=reset_temporal_history)
    save_frames = st.checkbox("Save frames for debugging", value=False, key="save_frames_for_debugging")
    st.caption("Frame saving is optional. Webcam telemetry is logged for each capture.")
mode_key = MODE_KEYS[hand_mode]

try:
    append_diagnostic_row()
except OSError:
    logging.exception("Cannot initialize webcam diagnostic log")
    st.error("Could not open the diagnostic log. Check the reports folder permissions.")

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.subheader("Your gesture")
    upload_tab, webcam_tab = st.tabs(["Upload Image", "Webcam"], key="image_source", on_change="rerun")
with right:
    st.subheader("Recognition")
    result_slot = st.empty()
    with result_slot.container():
        render_result_card()

# Collapsed content performs no model I/O.
supported = st.expander("Supported mudras", key="supported_panel", on_change="rerun")
with supported:
    if supported.open:
        render_supported(load_class_names())

diagnostics = st.expander("Diagnostic Mode", key="diagnostic_panel", on_change="rerun")
with diagnostics:
    if diagnostics.open:
        class_names = load_class_names()
        names = sorted(class_names["single"] + class_names["double"])
        st.selectbox("Expected mudra", names, index=None, key="expected_mudra",
                     on_change=reset_temporal_history, persist_state="session",
                     help="Ground truth for webcam telemetry and replay history. It does not choose the model's answer.")
        st.caption("Webcam captures log raw and displayed results. Only webcam results use recent-capture smoothing.")
    diagnostic_details = st.container()

prediction = None
record = None
original_shape = None
has_input = False
elapsed_ms = None
if upload_tab.open:
    with upload_tab:
        uploaded = st.file_uploader("Upload a mudra photo", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            import cv2
            import numpy as np
            payload = uploaded.getvalue()
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR) if payload else None
            if image is None:
                st.error("Could not read the uploaded image.")
            else:
                st.image(image, channels="BGR", width="stretch")
                identity = (hashlib.sha256(payload).hexdigest(), mode_key, threshold)
                cached = st.session_state.get("upload_prediction")
                if cached is None or cached["identity"] != identity:
                    prediction, elapsed_ms = infer_image(image, mode_key, threshold)
                    cached = dict(identity=identity, prediction=prediction, elapsed_ms=elapsed_ms)
                    st.session_state["upload_prediction"] = cached
                prediction, elapsed_ms = cached["prediction"], cached["elapsed_ms"]
                has_input, original_shape = True, image.shape
else:
    with webcam_tab:
        st.caption("Keep your hand fully in view, then take a photo. Open Diagnostic Mode below to set ground truth.")
        camera = st.camera_input("Capture pose for analysis", key="diagnostic_camera", on_change=capture_changed)
        if camera is not None:
            # Vocabulary supports the unchanged expected-hand warning/logging.
            with st.spinner("Loading recognizer and analyzing capture…"):
                record = diagnostic_capture(camera, st.session_state["expected_mudra"], mode_key, threshold,
                                            save_frames, load_class_names(), render=False)
            prediction, original_shape = record["prediction"], record["image_shape"]
            has_input, elapsed_ms = True, record["elapsed_ms"]

with result_slot.container():
    render_result_card(prediction, original_shape, record=record, has_input=has_input, elapsed_ms=elapsed_ms)
if diagnostics.open and record is not None:
    with diagnostic_details:
        render_diagnostic(record, st.session_state["expected_mudra"], mode_key, threshold, load_class_names())
if has_input and prediction is not None:
    ready.markdown('<span class="status-pill">Models ready · capture processed</span>', unsafe_allow_html=True)

render_footer()
