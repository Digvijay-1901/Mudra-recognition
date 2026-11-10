"""Existing page CSS, header and footer, kept verbatim for visual parity."""
import streamlit as st


STYLE = """
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
"""

def render_header():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="hero-icon">🙏</div><div class="hero-title">Bharatanatyam Mudra Recognition</div><div class="hero-subtitle">Hand landmarks · geometric features · machine learning</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="stats"><div class="stat"><div class="stat-number">24</div><div class="stat-label">Single-hand</div></div><div class="stat"><div class="stat-number">10</div><div class="stat-label">Double-hand</div></div><div class="stat"><div class="stat-number">34</div><div class="stat-label">Mudras</div></div></div>', unsafe_allow_html=True)

def render_footer(mudra_names):
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><b>01 · Detect</b><br>MediaPipe identifies the hand and its 21 landmarks.<br><br><b>02 · Extract</b><br>Landmarks are converted into geometric features describing hand shape and finger relationships.<br><br><b>03 · Classify</b><br>An MLP classifier analyzes the features and predicts the mudra.<br><br><b>04 · Respond</b><br>The application displays the predicted mudra, confidence, and alternative predictions.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Supported mudras</div>', unsafe_allow_html=True)
    chips = ''.join((f'<span class="mudra-chip">{name}</span>' for name in mudra_names))
    st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">Bharatanatyam Mudra Recognition · MediaPipe · scikit-learn · Streamlit</div>', unsafe_allow_html=True)
