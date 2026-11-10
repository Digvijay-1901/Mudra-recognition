"""Session callbacks, capture deduplication and unchanged inference/smoothing orchestration."""
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from time import perf_counter
import streamlit as st
from diagnostic_temporal import smooth_capture
from app_telemetry import append_diagnostic_row, save_diagnostic_frame

MODE_KEYS = {"Auto": "auto", "Single-hand": "single", "Double-hand": "double"}


def reset_temporal_history():
    st.session_state.pop("diagnostic_temporal", None)

def capture_changed():
    if st.session_state.get("diagnostic_camera") is None:
        st.session_state.pop("diagnostic_attempt", None)
        return
    sequence = st.session_state.get("diagnostic_sequence", 0) + 1
    st.session_state["diagnostic_sequence"] = sequence
    st.session_state["diagnostic_context"] = {
        "sequence": sequence,
        "expected": st.session_state["expected_mudra"],
        "mode": MODE_KEYS[st.session_state["hand_mode"]],
        "threshold": st.session_state["confidence_threshold"],
        "captured_at": perf_counter(),
        "timestamp": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(timespec="seconds"),
    }

def infer_image(image, mode, confidence_threshold):
    import cv2
    import brain
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    started = perf_counter()
    from app_resources import load_mediapipe_graph, load_gate_model
    load_mediapipe_graph()
    if mode == "auto":
        load_gate_model()
    with st.spinner("Loading recognizer and processing frame…"):
        pred = brain.predict(image_rgb, mode=mode, threshold=confidence_threshold)
    return pred, (perf_counter() - started) * 1000

def diagnostic_capture(camera, expected, mode_key, threshold, save_frames_for_debugging, class_names_by_mode, *, render=True):
    import cv2
    import numpy as np
    from brain import Abstention
    payload = camera.getvalue()
    sequence = st.session_state.get("diagnostic_sequence", 0)
    identity = (sequence, getattr(camera, "file_id", None), hashlib.sha256(payload).hexdigest())
    record = st.session_state.get("diagnostic_attempt")
    if record is None or record["identity"] != identity:
        context = st.session_state.get("diagnostic_context", {})
        if context.get("sequence") != sequence:
            context = {
                "expected": expected, "mode": mode_key, "threshold": threshold,
                "timestamp": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(timespec="seconds"),
            }
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR) if payload else None
        record = dict(identity=identity, expected=context["expected"], timestamp=context["timestamp"],
                      mode=context["mode"], threshold=context["threshold"], prediction=None,
                      image_shape=None, elapsed_ms=None, error=None, logged=False,
                      label="INVALID_IMAGE", confidence=None, detected_mode="Unknown", hands=None,
                      tilt_degrees=None, frame_path="")
        if image is None:
            record["error"] = "Could not read the camera image. Take another photo."
        else:
            record["image_shape"] = image.shape
            if save_frames_for_debugging:
                try:
                    record["frame_path"] = save_diagnostic_frame(image, record["timestamp"])
                except OSError:
                    logging.exception("Cannot save raw diagnostic frame")
                    st.error("Could not save the raw frame. Check the reports folder permissions.")
            try:
                pred, elapsed = infer_image(image, context["mode"], context["threshold"])
                tilt = getattr(pred, "tilt_degrees", None)
                record.update(prediction=pred, elapsed_ms=elapsed,
                              tilt_degrees=tilt if tilt is not None and np.isfinite(tilt) else None)
                if pred is None:
                    record.update(label="NO_HAND", hands=0)
                elif isinstance(pred, Abstention):
                    record.update(label="ABSTAIN", hands=len(pred.results.multi_hand_landmarks or []))
                else:
                    record.update(label=pred.label, confidence=pred.confidence,
                                  detected_mode={"Single-hand": "Single", "Double-hand": "Double"}.get(pred.mode, pred.mode),
                                  hands=pred.n_hands)
            except Exception:
                logging.exception("Diagnostic frame inference failed")
                record.update(label="ERROR", error="Could not analyze this capture. Try another photo.")
        history = st.session_state.setdefault("diagnostic_temporal", {})
        record["temporal"] = smooth_capture(
            history, record["expected"], record["mode"], record["threshold"],
            context.get("captured_at", perf_counter()), record["label"],
            record["confidence"], record["detected_mode"],
        )
        st.session_state["diagnostic_attempt"] = record

    if not record["logged"]:
        try:
            displayed_label = record.get("temporal", {}).get("label", record["label"])
            messages = []
            if record["error"]:
                messages.append(record["error"])
            elif isinstance(record["prediction"], Abstention):
                messages.append(record["prediction"].message)
            elif record["prediction"] is None:
                messages.append("No hand detected â€” try a clearer, well-lit image.")
            if (record["expected"] in class_names_by_mode["double"] and record["hands"] == 1
                    and record["detected_mode"] != "Double-hand"):
                messages.append("The expected pose needs both hands, but only one was detected. The two-hand configuration is unverified; show both hands and retake.")
            append_diagnostic_row([
                record["timestamp"], record["expected"], record["label"],
                record["confidence"] if record["confidence"] is not None else "",
                record["detected_mode"], record["hands"] if record["hands"] is not None else "",
                record.get("tilt_degrees") if record.get("tilt_degrees") is not None else "",
                record["frame_path"], displayed_label, " | ".join(messages),
            ])
            record["logged"] = True
        except OSError:
            logging.exception("Cannot append webcam diagnostic attempt")
            st.error("Could not save diagnostic telemetry. Check the reports folder permissions.")

    if render:
        from app_ui import render_diagnostic
        render_diagnostic(record, expected, mode_key, threshold, class_names_by_mode)
    return record
