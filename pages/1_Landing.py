import streamlit as st
from pathlib import Path

# ---------------------
# Static Assets
# ---------------------
ASSISTANT_AVATAR = "app_ui/static/assistant.png"
CUSTOM_STYLE_PATH = "app_ui/static/styles.css"


def inject_custom_css(css_path: str):
    """
    Injects external CSS for custom styling.
    """
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Custom CSS file not found: {css_path}")


# ---------------------
# Streamlit Page Setup
# ---------------------
st.set_page_config(page_title="Welcome to My Health Chatbot", layout="centered")

inject_custom_css(CUSTOM_STYLE_PATH)

# ---------------------
# Page Content
# ---------------------
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

st.image(ASSISTANT_AVATAR, width=130)

st.markdown("<h1 style='margin-top: 10px;'>My Health Chatbot</h1>", unsafe_allow_html=True)

st.markdown("""
<p style='font-size: 18px; margin-top: 10px;'>
Welcome to your personal AI-powered health assistant, guided by <strong>Dr. Charlotte</strong>.
</p>

<p style='font-size: 16px; color: #555;'>
This tool allows you to upload your medical records and ask health-related questions.<br>
Dr. Charlotte also explores reliable biomedical literature from PubMed to offer you accurate and helpful insights tailored to your needs.
</p>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------
# Continue Button
# ---------------------
st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    if st.button("Continue to Chat", use_container_width=True):
        st.switch_page("pages/2_Chatbot.py")
