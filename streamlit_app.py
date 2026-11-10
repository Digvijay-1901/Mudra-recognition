"""Streamlit entry point: page setup, sidebar and page composition."""
import _quiet
import streamlit as st
import cv2
import numpy as np
import hashlib
import logging

from app_theme import render_header, render_footer
st.set_page_config(
    page_title="Bharatanatyam Mudra Recognition",
    page_icon="🙏",
    layout="centered"
)

render_header()

from app_resources import load_class_names, load_reference_image
from app_telemetry import load_diagnostic_lock, append_diagnostic_row
from app_capture import (
    MODE_KEYS, reset_temporal_history, capture_changed, infer_image,
    diagnostic_capture,
)
from app_ui import render_prediction



diagnostic_lock = load_diagnostic_lock()

try:
    append_diagnostic_row()  # Create only the header until a real attempt arrives.
except OSError:
    logging.exception("Cannot initialize webcam diagnostic log")
    st.error("Could not open the diagnostic log. Check the reports folder permissions.")

class_names_by_mode = load_class_names()

mudra_names = sorted(class_names_by_mode["single"] + class_names_by_mode["double"])

threshold = st.sidebar.slider(
    "Minimum score floor",
    0.30,
    0.95,
    0.30,
    0.05,
    key="confidence_threshold",
    on_change=reset_temporal_history,
    help="Validated class minimums range from 30% to 60%. This floor can make them stricter; 60% restores the previous uniform cutoff. Scores are not calibrated probabilities. In Auto, one detected hand may also trigger a retake prompt."
)

st.sidebar.caption("Validated class cutoffs apply. Higher coverage also admits more lower-score errors.")

hand_mode = st.sidebar.radio("Hand mode", list(MODE_KEYS), key="hand_mode", on_change=reset_temporal_history)

mode_key = MODE_KEYS[hand_mode]

save_frames_for_debugging = st.sidebar.checkbox(
    "Save frames for debugging", value=False, key="save_frames_for_debugging"
)

expected_label = st.selectbox(
    "Expected mudra", mudra_names, index=mudra_names.index("Pathaka"), key="expected_mudra",
    on_change=reset_temporal_history,
)

reference_group = "dataset" if expected_label in class_names_by_mode["single"] else "dataset_double"

reference = load_reference_image(expected_label, reference_group)

if reference is not None:
    st.image(reference, caption=expected_label, width=280)
    st.caption("Reference image credit: Jisha Raj dataset")
else:
    st.info(f"No local reference image is available for {expected_label}.")

source = st.radio("Image source", ["Upload Image", "Webcam"], horizontal=True)

if source == "Upload Image":
    st.markdown(
        '<div class="section-title">Choose an image</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Upload a mudra photo",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded is not None:
        payload = uploaded.getvalue()
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR) if payload else None
        if image is None:
            st.error("Could not read the uploaded image.")
        else:
            st.image(image, channels="BGR", width=280)
            identity = (hashlib.sha256(payload).hexdigest(), mode_key, threshold)
            cached = st.session_state.get("upload_prediction")
            if cached is None or cached["identity"] != identity:
                pred, elapsed = infer_image(image, mode_key, threshold)
                cached = dict(identity=identity, prediction=pred, elapsed_ms=elapsed)
                st.session_state["upload_prediction"] = cached
            st.caption(f"Inference: {cached['elapsed_ms']:.0f} ms")
            render_prediction(cached["prediction"], image.shape)
else:
    with st.expander("Diagnostic Mode", expanded=True):
        st.caption("Choose the expected mudra above, then capture your pose. Each new capture is saved for analysis.")
        camera = st.camera_input("Capture pose for analysis", key="diagnostic_camera", on_change=capture_changed)
        if camera is not None:
            diagnostic_capture(
                camera, expected_label, mode_key, threshold,
                save_frames_for_debugging, class_names_by_mode,
            )

render_footer(mudra_names)
