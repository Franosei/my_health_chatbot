from datetime import datetime
from pathlib import Path

import streamlit as st


CUSTOM_STYLE_PATH = Path("app_ui/static/styles.css")


def inject_custom_css(css_path: Path = CUSTOM_STYLE_PATH) -> None:
    try:
        with open(css_path, "r", encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Custom CSS file not found: {css_path}")


def format_timestamp(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%d %b %Y, %H:%M")
    except ValueError:
        return value
