"""Prediction and diagnostic presentation, including existing confidence/quality notes."""
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from brain import Abstention, NORMALIZE_AT_DEGREES
from overlay import draw

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def draw_landmarks(image, results):
    output = image.copy()

    for hand in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            output,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

    return output

def show_result(label, confidence, mode, probabilities, class_names):
    st.markdown(
        '<div class="result-card">'
        f'<div class="result-label">{label}</div>'
        f'<div class="result-mode">{mode}</div>'
        f'<div class="result-confidence">{confidence:.1%} model score</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Top predictions</div>',
        unsafe_allow_html=True
    )

    top = np.argsort(probabilities)[::-1][:3]

    for i in top:
        name = class_names[i]
        st.progress(
            float(probabilities[i]),
            text=f"{name} · {probabilities[i]:.1%}"
        )

def quality_notes(pred, original_shape):
    notes = []
    if pred.mode == "Double-hand" and pred.n_hands < 2:
        notes.append("Only one hand detected: the two-hand configuration is unverified. Show both hands clearly and retake.")
    img = pred.used_image
    pts = [p for hl in pred.results.multi_hand_landmarks for p in hl.landmark]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    if max(max(xs) - min(xs), max(ys) - min(ys)) < 0.25:
        notes.append("Your hand looks small in the frame. Move closer.")
    if (tuple(original_shape[:2]) != tuple(img.shape[:2])
            or min(xs) < 0.02 or max(xs) > 0.98 or min(ys) < 0.02 or max(ys) > 0.98):
        notes.append("Part of your hand is at the edge of the frame. Bring it fully into view.")
    if cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean() < 60:
        notes.append("The image is dark. Face a light source.")
    if min(original_shape[:2]) < 300:
        notes.append("The image resolution is low. Try a sharper camera or more light.")
    return notes

def render_prediction(pred, original_shape):
    if pred is None:
        st.error("No hand detected — try a clearer, well-lit image.")
        return

    st.markdown(
        '<div class="section-title">Detected landmarks</div>',
        unsafe_allow_html=True
    )

    st.image(
        draw(pred.used_image, pred.results),
        width=340
    )

    if pred.tilt_degrees is not None:
        st.caption(f"First detected hand tilt: {pred.tilt_degrees:+.0f}° from upright.")
        if abs(pred.tilt_degrees) >= NORMALIZE_AT_DEGREES:
            st.warning("Extreme image-plane tilt. If the result looks wrong, retake with the hand upright.")

    if isinstance(pred, Abstention):
        st.warning(pred.message)
        return

    show_result(
        pred.label,
        pred.confidence,
        pred.mode,
        pred.probabilities,
        pred.class_names
    )
    for note in quality_notes(pred, original_shape):
        st.info(note)

def render_diagnostic(record, expected, mode_key, threshold, class_names_by_mode):
    left, right = st.columns(2)
    displayed = record.get("temporal", dict(label=record["label"], confidence=record["confidence"],
                                            mode=record["detected_mode"], changed=False))
    left.metric("Expected", record["expected"])
    right.metric("Predicted", displayed["label"])
    confidence_col, mode_col, hands_col = st.columns(3)
    confidence_col.metric("Confidence", f"{displayed['confidence']:.1%}" if displayed["confidence"] is not None else "—")
    mode_col.metric("Mode", displayed["mode"])
    hands_col.metric("Detected hands", record["hands"] if record["hands"] is not None else "—")
    if displayed["changed"]:
        st.info(f"Recent captures favor {displayed['label']} ({displayed['mode']}). "
                f"Shown score is retained from those captures. Raw frame: {record['label']} "
                f"({record['detected_mode']}, {record['confidence']:.1%}).")
    if (record["expected"] in class_names_by_mode["double"] and record["hands"] == 1
            and record["detected_mode"] != "Double-hand"):
        st.warning("The expected pose needs both hands, but only one was detected. The two-hand configuration is unverified; show both hands and retake.")
    if record["elapsed_ms"] is not None:
        st.caption(f"Inference: {record['elapsed_ms']:.0f} ms · {record['timestamp']} · "
                   f"Input mode: {record['mode']} · Score floor: {record['threshold']:.0%}")
    if (expected != record["expected"] or mode_key != record["mode"] or threshold != record["threshold"]):
        st.info("This result keeps the expected class and settings from capture time. Take another photo to test your new selection.")
    if record["error"]:
        st.error(record["error"])
    else:
        if displayed["changed"]:
            st.caption("Raw frame details and model scores")
        render_prediction(record["prediction"], record["image_shape"])
