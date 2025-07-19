import streamlit as st
from pathlib import Path
import json
from datetime import datetime

from app_ui.uploader import upload_documents
from backend.rag_system import RAGEngine
from backend.utils import extract_text_from_pdf

# ---------------------
# Chat History Utilities
# ---------------------
HISTORY_PATH = Path("chat_history.json")


def load_history():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


# ---------------------
# Static Assets
# ---------------------
USER_AVATAR = "app_ui/static/user.png"
ASSISTANT_AVATAR = "app_ui/static/assistant.png"
CUSTOM_STYLE_PATH = "app_ui/static/styles.css"


def inject_custom_css(css_path: str):
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Custom CSS file not found: {css_path}")


# ---------------------
# Streamlit Page Setup
# ---------------------
st.set_page_config(page_title="My Health Checks", layout="centered")
inject_custom_css(CUSTOM_STYLE_PATH)

st.title("My Health Checks")
st.caption("Ask your health questions based on uploaded medical records or real-time PubMed research.")

# ---------------------
# Session State Init
# ---------------------
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(embedding_dir="sample_data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()

# ---------------------
# File Upload & Ingestion
# ---------------------
doc_uploaded = upload_documents()

if doc_uploaded:
    with st.spinner("Processing uploaded medical documents..."):
        st.session_state.rag_engine.ingest_documents()
        st.success("Documents processed successfully.")

# ---------------------
# Chat History Display
# ---------------------
for msg in st.session_state.chat_history:
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    is_user = msg["role"] == "user"
    timestamp = msg.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))

    col1, col2 = st.columns([10, 1]) if is_user else st.columns([1, 10])
    with (col2 if is_user else col1):
        st.image(avatar, width=72)
    with (col1 if is_user else col2):
        st.markdown(
            f"""
            <div class='custom-chat-bubble {bubble_class}'>
                {msg['content']}
                <div class='timestamp'>{timestamp}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------
# Chat Input
# ---------------------
user_question = st.chat_input("Ask your health question here...")

if user_question:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Show user's message
    user_col1, user_col2 = st.columns([10, 1])
    with user_col2:
        st.image(USER_AVATAR, width=72)
    with user_col1:
        st.markdown(
            f"""
            <div class='custom-chat-bubble user-bubble'>
                {user_question}
                <div class='timestamp'>{current_time}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question,
        "timestamp": current_time
    })
    save_history(st.session_state.chat_history)

    # Assistant spinner and avatar
    assistant_col1, assistant_col2 = st.columns([1, 10])
    with assistant_col1:
        st.image(ASSISTANT_AVATAR, width=72)
    with assistant_col2:
        bubble_container = st.empty()

        bubble_container.markdown(
            """
            <div class='custom-chat-bubble assistant-bubble'>
                <div style="display: flex; align-items: center;">
                    <div class="dot-flashing"></div>
                    <span style="margin-left: 10px;">Checking for information...</span>
                </div>
                <div class='timestamp'>...</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Stream actual assistant response
    try:
        stream = st.session_state.rag_engine.handle_user_question(
            question=user_question,
            chat_history=st.session_state.chat_history,
            stream=True
        )

        full_reply = ""
        for chunk in stream:
            full_reply += chunk
            bubble_container.markdown(
                f"""
                <div class='custom-chat-bubble assistant-bubble'>
                    {full_reply}
                    <div class='timestamp'>{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_reply,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        save_history(st.session_state.chat_history)

    except Exception as e:
        bubble_container.markdown(
            f"""
            <div class='custom-chat-bubble assistant-bubble'>
                An error occurred while answering: {e}
                <div class='timestamp'>{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"An error occurred while answering the question: {e}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        save_history(st.session_state.chat_history)
