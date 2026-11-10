"""One result card, optional raw diagnostics and model-backed supported-name lists."""
from html import escape
import streamlit as st


def render_result_card(pred=None, original_shape=None, *, record=None, has_input=False, elapsed_ms=None):
    with st.container(border=True):
        if not has_input:
            st.markdown('<div class="empty-symbol">✧</div>', unsafe_allow_html=True)
            st.subheader("Your gesture, recognized")
            st.markdown('<p class="muted">Upload a photo or take a webcam capture. The mudra name, score and hand mode will appear here.</p>', unsafe_allow_html=True)
            st.caption("Clear lighting · Whole hand visible · Plain background")
            return
        from brain import Abstention
        if record is not None and record["error"]:
            st.error(record["error"])
            return
        if pred is None:
            st.info("No hand detected — try a clearer, well-lit image.")
            return
        if isinstance(pred, Abstention):
            st.subheader("Take another photo")
            st.warning(pred.message)
            return
        displayed = record["temporal"] if record is not None else dict(label=pred.label, confidence=pred.confidence, mode=pred.mode, changed=False)
        mode = {"Single-hand": "Single", "Double-hand": "Double"}.get(displayed["mode"], displayed["mode"])
        st.markdown(f'<span class="mode-pill">{escape(mode)} hand</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-name">{escape(displayed["label"])}</div>', unsafe_allow_html=True)
        st.progress(float(displayed["confidence"]), text=f'{displayed["confidence"]:.1%} model score')
        st.caption("Model scores are not calibrated probabilities.")
        if displayed["changed"]:
            st.info(f'Recent captures favor {displayed["label"]}. The displayed score is retained from those captures.')
        for note in quality_notes(pred, original_shape):
            st.info(note)
        if pred.tilt_degrees is not None and abs(pred.tilt_degrees) >= 75:
            st.warning("Extreme image-plane tilt. If the result looks wrong, retake with the hand upright.")
        if elapsed_ms is not None:
            st.caption(f"Inference: {elapsed_ms:.0f} ms")


def render_supported(class_names):
    single, double = st.tabs(["Single hand", "Double hand"])
    for tab, family in ((single, "single"), (double, "double")):
        with tab:
            st.markdown('<ul class="name-list">' + ''.join(f'<li>{escape(name)}</li>' for name in sorted(class_names[family])) + '</ul>', unsafe_allow_html=True)


def quality_notes(pred, original_shape):
    import cv2
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
    """Raw landmark evidence stays in diagnostics, separate from the displayed vote."""
    if pred is None:
        st.info("No hand detected — try a clearer, well-lit image.")
        return
    import numpy as np
    from brain import Abstention
    from overlay import draw
    st.image(draw(pred.used_image, pred.results), caption="Raw detected landmarks", width="stretch")
    if pred.tilt_degrees is not None:
        st.caption(f"First detected hand tilt: {pred.tilt_degrees:+.0f}° from upright.")
    if isinstance(pred, Abstention):
        st.warning(pred.message)
        return
    st.caption(f"Raw frame: {pred.label} · {pred.confidence:.1%} · {pred.mode}")
    for index in np.argsort(pred.probabilities)[::-1][:3]:
        st.progress(float(pred.probabilities[index]), text=f"{pred.class_names[index]} · {pred.probabilities[index]:.1%}")


def render_diagnostic(record, expected, mode_key, threshold, class_names_by_mode):
    st.subheader("Capture details")
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
