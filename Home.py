import streamlit as st
from backend.product_config import PRODUCT_NAME

st.set_page_config(page_title=PRODUCT_NAME, layout="centered", initial_sidebar_state="collapsed")

# Redirect based on session state
target_page = "pages/2_Chatbot.py" if st.session_state.get("current_user") else "pages/1_Landing.py"
st.switch_page(target_page)
