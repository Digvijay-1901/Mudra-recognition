"""The page's shared dark palette, typography, header and footer."""
import streamlit as st

STYLE = """
<style>
.stApp { background: #0c1118; }
[data-testid="stHeader"] { background: #0c1118; }
.block-container { max-width: 1120px; padding-top: 2.6rem; padding-bottom: 2rem; }
h1 { font-size: 2.4rem !important; letter-spacing: -.04em; font-weight: 650 !important; }
h2, h3 { letter-spacing: -.02em; }
h3 { font-size: 1.15rem !important; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 14px; }
[data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] {
    background: #141d28; border-radius: 12px;
}
[data-testid="stTabs"] button { font-size: .95rem; }
.eyebrow { color: #63d5c0; letter-spacing: .14em; font-size: .73rem; font-weight: 650; }
.lead { color: #a5b3c4; font-size: 1rem; margin-bottom: 1.5rem; }
.status-pill, .mode-pill {
    display: inline-block; padding: .35rem .7rem; border: 1px solid #304553;
    border-radius: 999px; color: #63d5c0; font-size: .8rem; background: #14262a;
}
.result-name { font-size: 2.2rem; letter-spacing: -.035em; font-weight: 650; margin: .8rem 0; }
.empty-symbol { color: #63d5c0; font-size: 2.5rem; margin: 1rem 0; }
.muted { color: #a5b3c4; font-size: .9rem; line-height: 1.6; }
.name-list { columns: 3 170px; list-style: none; padding: 0; margin: .5rem 0; }
.name-list li { break-inside: avoid; padding: .4rem 0; color: #d7e1eb; }
.page-footer { margin-top: 1.6rem; font-size: .78rem; color: #8393a6; }
@media (max-width: 640px) { h1 { font-size: 1.9rem !important; } }
</style>
"""


def render_header():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">MOVEMENT · FORM · RECOGNITION</div>', unsafe_allow_html=True)
    st.title("Bharatanatyam Mudra Recognition")
    st.markdown('<div class="lead">Explore single- and double-hand gestures with MediaPipe and RTMPose.</div>',
                unsafe_allow_html=True)


def render_footer():
    st.markdown('<div class="page-footer">A pose is more than a prediction. Use clear lighting and keep the whole hand in view.</div>',
                unsafe_allow_html=True)
