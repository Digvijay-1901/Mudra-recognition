import _quiet
import streamlit as st
import cv2
import numpy as np
import joblib
import os
import mediapipe as mp

from feature_extraction import extract_single_hand_features
from feature_extraction_double import extract_double_hand_features

st.set_page_config(
    page_title="Bharatanatyam Mudra Recognition",
    page_icon="🙏",
    layout="centered"
)

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
    '<div class="stat"><div class="stat-number">25</div><div class="stat-label">Single-hand</div></div>'
    '<div class="stat"><div class="stat-number">10</div><div class="stat-label">Double-hand</div></div>'
    '<div class="stat"><div class="stat-number">35</div><div class="stat-label">Mudras</div></div>'
    '</div>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_single_model():
    return (
        joblib.load("model/mlp_mudra_classifier.joblib"),
        joblib.load("model/feature_scaler.joblib"),
        joblib.load("model/label_encoder.joblib")
    )

@st.cache_resource
def load_double_model():
    return (
        joblib.load("model_double/mlp_mudra_classifier.joblib"),
        joblib.load("model_double/feature_scaler.joblib"),
        joblib.load("model_double/label_encoder.joblib")
    )

@st.cache_resource
def create_hand_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hand_detector = create_hand_detector()

threshold = st.sidebar.slider(
    "Confidence threshold",
    0.30,
    0.95,
    0.60,
    0.05
)

def detect_hands(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return hand_detector.process(image_rgb)


def detect_hands_robust(image):
    """
    Primary detection followed by a padded-image retry when two hands
    are not found. The number of detected hands still determines which
    classifier is used; classifier confidences are never compared.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(image_rgb)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        return results, image

    h, w = image.shape[:2]
    padded = cv2.copyMakeBorder(
        image,
        h // 3,
        h // 3,
        w // 3,
        w // 3,
        cv2.BORDER_REPLICATE
    )

    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded_results = hand_detector.process(padded_rgb)

    if (
        padded_results.multi_hand_landmarks
        and len(padded_results.multi_hand_landmarks) == 2
    ):
        return padded_results, padded

    # Fall back to whichever pass found more hands.
    original_count = (
        len(results.multi_hand_landmarks)
        if results.multi_hand_landmarks
        else 0
    )
    padded_count = (
        len(padded_results.multi_hand_landmarks)
        if padded_results.multi_hand_landmarks
        else 0
    )

    if padded_count > original_count:
        return padded_results, padded

    return results, image

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

def show_result(label, confidence, mode, probabilities, encoder):
    if confidence >= threshold:
        display_label = label
        confidence_text = f"{confidence:.1%} confidence"
    else:
        display_label = "Unsure"
        confidence_text = f"{confidence:.1%} confidence · below threshold"

    st.markdown(
        '<div class="result-card">'
        f'<div class="result-label">{display_label}</div>'
        f'<div class="result-mode">{mode}</div>'
        f'<div class="result-confidence">{confidence_text}</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Top predictions</div>',
        unsafe_allow_html=True
    )

    top = np.argsort(probabilities)[::-1][:3]

    for i in top:
        name = encoder.inverse_transform([i])[0]
        st.progress(
            float(probabilities[i]),
            text=f"{name} · {probabilities[i]:.1%}"
        )

def process(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    single_model, single_scaler, single_encoder = load_single_model()
    double_model, double_scaler, double_encoder = load_double_model()

    results, _ = detect_hands_robust(image)
    n_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    def show_landmarks():
        if results.multi_hand_landmarks:
            st.markdown('<div class="section-title">Detected landmarks</div>', unsafe_allow_html=True)
            st.image(draw_landmarks(image, results), channels="BGR", width=340)

    if n_hands == 2:
        feats2 = extract_double_hand_features(image_rgb, static=True)
        if feats2 is not None and len(feats2) == double_model.n_features_in_:
            show_landmarks()
            probs = double_model.predict_proba(double_scaler.transform([feats2]))[0]
            idx = int(np.argmax(probs))
            show_result(double_encoder.inverse_transform([idx])[0], float(probs[idx]), "Double-hand", probs, double_encoder)
            return

    feats1 = extract_single_hand_features(image_rgb, static=True)
    single_result = None
    if feats1 is not None and len(feats1) == single_model.n_features_in_:
        probs1 = single_model.predict_proba(single_scaler.transform([feats1]))[0]
        idx1 = int(np.argmax(probs1))
        single_result = (single_encoder.inverse_transform([idx1])[0], float(probs1[idx1]), probs1)

    if single_result and single_result[1] >= 0.6:
        show_landmarks()
        label, conf, probs = single_result
        show_result(label, conf, "Single-hand", probs, single_encoder)
        return

    feats2 = extract_double_hand_features(image_rgb, static=True)
    if feats2 is not None and len(feats2) == double_model.n_features_in_:
        show_landmarks()
        probs2 = double_model.predict_proba(double_scaler.transform([feats2]))[0]
        idx2 = int(np.argmax(probs2))
        show_result(double_encoder.inverse_transform([idx2])[0], float(probs2[idx2]), "Double-hand", probs2, double_encoder)
        return

    if single_result:
        show_landmarks()
        label, conf, probs = single_result
        show_result(label, conf, "Single-hand", probs, single_encoder)
        return

    st.error("No hand detected — try a clearer, well-lit image.")

def pick_example_image(folder, require_two_hands=False, max_tries=None):
    """
    Pick an example that satisfies the detector requirement.

    For double-hand examples, require exactly two detected hands.
    Scan the whole folder by default instead of only the first six files,
    so a valid example is not missed just because early filenames fail.
    """
    if not os.path.isdir(folder):
        return None

    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if max_tries is not None:
        files = files[:max_tries]

    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path)

        if img is None:
            continue

        results, _ = detect_hands_robust(img)

        if not results.multi_hand_landmarks:
            continue

        n_hands = len(results.multi_hand_landmarks)

        if require_two_hands and n_hands != 2:
            continue

        if not require_two_hands and n_hands != 1:
            continue

        print("EXAMPLE SELECTED:", path, "hands:", n_hands)
        return path

    print(
        "NO VALID EXAMPLE:",
        folder,
        "require_two_hands=",
        require_two_hands
    )
    return None

tab1, tab2 = st.tabs(
    ["📁 Upload Image", "📷 Webcam"]
)

with tab1:
    st.markdown(
        '<div class="section-title">Choose an image</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Upload a mudra photo",
        type=["jpg", "jpeg", "png", "webp"]
    )

    st.markdown(
        '<div class="section-title">Or try an example</div>',
        unsafe_allow_html=True
    )

    example_folders = {
        "Katakamukha": ("data/dataset/Katakamukha", False),
        "Mayura": ("data/dataset/Mayura", False),
        "Anjali (2-hand)": ("data/dataset_double/Anjali", True),
        "Karkatta (2-hand)": ("data/dataset_double/Karkatta", True)
    }

    examples = {
        name: pick_example_image(
            folder,
            require_two_hands=require_two_hands
        )
        for name, (folder, require_two_hands) in example_folders.items()
    }

    cols = st.columns(4)
    chosen_path = None

    for col, (name, path) in zip(cols, examples.items()):
        if path and col.button(name, width="stretch"):
            chosen_path = path

    active_image = None

    if uploaded:
        active_image = cv2.imdecode(
            np.frombuffer(uploaded.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
    elif chosen_path:
        active_image = cv2.imread(chosen_path)

    if active_image is not None:
        st.image(active_image, channels="BGR", width=280)
        process(active_image)

with tab2:
    st.markdown(
        '<div class="section-title">Capture a mudra</div>',
        unsafe_allow_html=True
    )

    camera = st.camera_input("Take a picture")

    if camera:
        image = cv2.imdecode(
            np.frombuffer(camera.getvalue(), np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is not None:
            process(image)
        else:
            st.error("Could not read the camera image.")

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
    '<div class="section-title">Model evaluation</div>',
    unsafe_allow_html=True
)

with st.expander("📊 View confusion matrices"):
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", width="stretch")

    if os.path.exists("confusion_matrix_double.png"):
        st.image("confusion_matrix_double.png", width="stretch")

st.markdown(
    '<div class="section-title">Supported mudras</div>',
    unsafe_allow_html=True
)

mudra_names = [
    "Aralam",
    "Ardhachandra",
    "Ardhapathaka",
    "Bramaram",
    "Chandrakala",
    "Chaturam",
    "Kapith",
    "Katakamukha",
    "Kartarimukha",
    "Mayura",
    "Mrigashirsha",
    "Mushti",
    "Pathaka",
    "Shukatundam",
    "Sikharam",
    "Suchi",
    "Tripathaka",
    "Hamsasya",
    "Alapadma",
    "Mukula",
    "Hamsapaksha",
    "Tamrachuda",
    "Simhamukha",
    "Trishula",
    "Kangula",
    "Anjali",
    "Kapotham",
    "Karkatta",
    "Pushpaputa",
    "Shanka",
    "Shivalinga",
    "Garuda",
    "Chakra",
    "Kurma",
    "Nagabandha"
]


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