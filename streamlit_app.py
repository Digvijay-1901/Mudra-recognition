import _quiet
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
import csv
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from threading import Lock
from time import perf_counter

from brain import Abstention, NORMALIZE_AT_DEGREES, predict
from overlay import draw
from diagnostic_temporal import smooth_capture

st.set_page_config(
    page_title="Bharatanatyam Mudra Recognition",
    page_icon="🙏",
    layout="centered"
)

BASE = Path(__file__).resolve().parent
DIAGNOSTICS_CSV = BASE / "reports" / "webcam_diagnostics.csv"
DIAGNOSTIC_FRAMES = BASE / "reports" / "webcam_frames"
DIAGNOSTIC_FIELDS = [
    "Timestamp", "Expected_Class", "Predicted_Class",
    "Confidence", "Mode_Detected", "Hands_Count", "Tilt_Degrees",
    "Frame_Path", "Displayed_Label", "Message",
]
MODE_KEYS = {"Auto": "auto", "Single-hand": "single", "Double-hand": "double"}


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


@st.cache_resource(show_spinner=False)
def load_diagnostic_lock():
    return Lock()


def append_diagnostic_row(row=None):
    """One process lock covers schema rollover, header and complete rows."""
    with diagnostic_lock:
        if row is not None and len(row) != len(DIAGNOSTIC_FIELDS):
            raise ValueError("Diagnostic row does not match the current schema")
        DIAGNOSTICS_CSV.parent.mkdir(parents=True, exist_ok=True)
        if DIAGNOSTICS_CSV.exists() and DIAGNOSTICS_CSV.stat().st_size:
            with DIAGNOSTICS_CSV.open(newline="", encoding="utf-8-sig") as previous:
                reader = csv.reader(previous)
                header = next(reader, None)
                has_rows = header != DIAGNOSTIC_FIELDS and any(any(cell.strip() for cell in item) for item in reader)
            if header != DIAGNOSTIC_FIELDS:
                if header == DIAGNOSTIC_FIELDS[:-1] and not has_rows:
                    # Preserve the existing file when it contains only the legacy header.
                    with DIAGNOSTICS_CSV.open("w", newline="", encoding="utf-8") as output:
                        csv.writer(output).writerow(DIAGNOSTIC_FIELDS)
                else:
                    # Reader is closed before renaming, including on Windows.
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    archive = DIAGNOSTICS_CSV.with_name(f"{DIAGNOSTICS_CSV.stem}.archive-{stamp}.csv")
                    sequence = 0
                    while archive.exists():
                        sequence += 1
                        archive = DIAGNOSTICS_CSV.with_name(f"{DIAGNOSTICS_CSV.stem}.archive-{stamp}-{sequence}.csv")
                    DIAGNOSTICS_CSV.rename(archive)
        with DIAGNOSTICS_CSV.open("a", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            if output.tell() == 0:
                writer.writerow(DIAGNOSTIC_FIELDS)
            if row is not None:
                writer.writerow(row)


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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, header, footer {
    visibility: hidden;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(184,148,255,.12), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(255,173,120,.10), transparent 22%),
        linear-gradient(180deg, #f8f6fc 0%, #efebf5 100%);
}

.block-container {
    max-width: 680px;
    padding-top: 1.7rem;
    padding-bottom: 2.5rem;
}

.hero {
    background: linear-gradient(135deg, #211936 0%, #39275c 100%);
    border-radius: 17px;
    padding: 1.45rem 1.3rem;
    text-align: center;
    color: white;
    box-shadow: 0 9px 25px rgba(43,28,70,.14);
    margin-bottom: 1rem;
}

.hero-icon {
    font-size: 2rem;
    margin-bottom: .2rem;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 600;
    margin: 0;
}

.hero-subtitle {
    color: #d8d0e5;
    font-size: .8rem;
    margin-top: .35rem;
}

.stats {
    display: flex;
    gap: 8px;
    margin-bottom: 1rem;
}

.stat {
    flex: 1;
    background: rgba(255,255,255,.86);
    border: 1px solid #e0d9eb;
    border-radius: 11px;
    padding: .65rem;
    text-align: center;
}

.stat-number {
    font-size: 1.05rem;
    font-weight: 700;
    color: #39275c;
}

.stat-label {
    color: #898296;
    font-size: .61rem;
    text-transform: uppercase;
    letter-spacing: .5px;
}

.section-title {
    color: #3a3150;
    font-size: .65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 1rem 0 .5rem;
}

.result-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f4ff 100%);
    border: 1px solid #ddd4eb;
    border-radius: 15px;
    padding: 1rem 1.2rem;
    margin-top: .8rem;
    box-shadow: 0 5px 18px rgba(53,38,79,.07);
}

.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.65rem;
    font-weight: 600;
    color: #291d40;
    margin: 0;
}

.result-mode {
    display: inline-block;
    background: #e9defb;
    color: #62418e;
    padding: 4px 10px;
    border-radius: 15px;
    font-size: .62rem;
    font-weight: 700;
    margin-top: .35rem;
}

.result-confidence {
    color: #777084;
    font-size: .78rem;
    margin-top: .35rem;
}

.info-card {
    background: linear-gradient(135deg, #fff 0%, #f5f1fa 100%);
    border: 1px solid #e1dbe9;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    color: #686173;
    line-height: 1.55;
    font-size: .78rem;
}

.info-card b {
    color: #332945;
}

.mudra-chip {
    display: inline-block;
    background: rgba(255,255,255,.9);
    border: 1px solid #ddd5e7;
    color: #514861;
    border-radius: 20px;
    padding: 5px 10px;
    margin: 2px;
    font-size: .68rem;
}

[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #ddd6e7;
    box-shadow: 0 5px 16px rgba(40,30,60,.08);
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid #d9d1e4;
    background: white;
    color: #433952;
    font-size: .8rem;
    font-weight: 500;
}

.stButton > button:hover {
    border-color: #7656a6;
    color: #5c4086;
    box-shadow: 0 4px 12px rgba(70,45,100,.08);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,.75);
    border-radius: 9px 9px 0 0;
    padding: 7px 18px;
    color: #766e82;
    font-size: .8rem;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #5c4086 !important;
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,.7);
    border: 1px solid #ded7e8;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #9a94a5;
    font-size: .64rem;
    margin-top: 2rem;
    padding-top: .8rem;
    border-top: 1px solid #dcd6e4;
}

[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #291d40 !important;
}

[data-testid="stProgress"] p {
    color: #291d40 !important;
}

[data-testid="stExpander"] summary p {
    color: #291d40 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<div class="hero-icon">🙏</div>'
    '<div class="hero-title">Bharatanatyam Mudra Recognition</div>'
    '<div class="hero-subtitle">Hand landmarks · geometric features · machine learning</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="stats">'
    '<div class="stat"><div class="stat-number">24</div><div class="stat-label">Single-hand</div></div>'
    '<div class="stat"><div class="stat-number">10</div><div class="stat-label">Double-hand</div></div>'
    '<div class="stat"><div class="stat-number">34</div><div class="stat-label">Mudras</div></div>'
    '</div>',
    unsafe_allow_html=True
)


# MediaPipe drawing utilities are still needed for visualizing landmarks.
models, gate_model, hands_graph = initialize_ai_resources()
diagnostic_lock = load_diagnostic_lock()
try:
    append_diagnostic_row()  # Create only the header until a real attempt arrives.
except OSError:
    logging.exception("Cannot initialize webcam diagnostic log")
    st.error("Could not open the diagnostic log. Check the reports folder permissions.")

class_names_by_mode = {
    "single": tuple(str(name) for name in models["single"][2].classes_),
    "double": tuple(str(name) for name in models["double"][2].classes_),
}
mudra_names = sorted(class_names_by_mode["single"] + class_names_by_mode["double"])

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

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

def infer_image(image, mode, confidence_threshold):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    started = perf_counter()
    with st.spinner("Processing frame..."):
        pred = predict(image_rgb, mode=mode, threshold=confidence_threshold)
    return pred, (perf_counter() - started) * 1000


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


def diagnostic_capture(camera, expected):
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
                # Save the decoded, unannotated camera pixels once per capture,
                # independently of inference success. Exclusive creation avoids
                # overwriting captures that share a timestamp, including sessions.
                try:
                    ok, png = cv2.imencode(".png", image)
                    if not ok:
                        raise OSError("PNG encoding failed")
                    stamp = datetime.fromisoformat(record["timestamp"]).strftime("%Y%m%dT%H%M%S%f%z")
                    with diagnostic_lock:
                        DIAGNOSTIC_FRAMES.mkdir(parents=True, exist_ok=True)
                        suffix = 0
                        while True:
                            path = DIAGNOSTIC_FRAMES / (stamp + (f"-{suffix}" if suffix else "") + ".png")
                            try:
                                with path.open("xb") as output:
                                    output.write(png.tobytes())
                                record["frame_path"] = str(path.resolve())
                                break
                            except FileExistsError:
                                suffix += 1
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
                messages.append("No hand detected — try a clearer, well-lit image.")
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
            diagnostic_capture(camera, expected_label)


st.markdown(
    '<div class="section-title">How it works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-card">'
    '<b>01 · Detect</b><br>'
    'MediaPipe identifies the hand and its 21 landmarks.'
    '<br><br>'
    '<b>02 · Extract</b><br>'
    'Landmarks are converted into geometric features describing hand shape and finger relationships.'
    '<br><br>'
    '<b>03 · Classify</b><br>'
    'An MLP classifier analyzes the features and predicts the mudra.'
    '<br><br>'
    '<b>04 · Respond</b><br>'
    'The application displays the predicted mudra, confidence, and alternative predictions.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Supported mudras</div>',
    unsafe_allow_html=True
)



chips = "".join(
    f'<span class="mudra-chip">{name}</span>'
    for name in mudra_names
)

st.markdown(
    f'<div>{chips}</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer">'
    'Bharatanatyam Mudra Recognition · '
    'MediaPipe · scikit-learn · Streamlit'
    '</div>',
    unsafe_allow_html=True
)
