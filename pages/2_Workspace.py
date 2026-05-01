import html

import streamlit as st

from app_ui.theme import inject_custom_css
from backend.product_config import PRODUCT_NAME
from backend.user_store import UserStore


st.set_page_config(
    page_title=f"Workspace - {PRODUCT_NAME}",
    page_icon=":material/dashboard:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_custom_css()


def action_card(title: str, eyebrow: str, body: str, meta: str) -> None:
    st.markdown(
        f"""
        <div class="dashboard-action-card">
            <span>{html.escape(eyebrow)}</span>
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(body)}</p>
            <small>{html.escape(meta)}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


current_user = st.session_state.get("current_user")
if not current_user:
    st.warning("Please sign in to continue.")
    st.session_state.auth_panel = "Sign in"
    st.switch_page("pages/1_Landing.py")

profile = UserStore.get_user_profile(current_user)
display_name = profile.get("display_name", current_user)
symptom_logs = UserStore.get_symptom_logs(current_user, limit=None)
medications = UserStore.get_medications(current_user)
uploads = UserStore.get_uploads(current_user)
vitals = UserStore.get_vitals(current_user, limit=None)
latest_triage = UserStore.get_latest_triage_summary(current_user)

st.markdown(
    f"""
    <div class="workspace-hero dashboard-hero">
        <div class="feature-eyebrow">{PRODUCT_NAME}</div>
        <h1>Welcome back, {html.escape(display_name)}.</h1>
        <p>
            Choose where you want to work today: chat with Dr. Charlotte, review your saved health timeline,
            or search recruiting clinical trials against your saved health context.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4, gap="small")
metric_cols[0].metric("Documents", len(uploads))
metric_cols[1].metric("Symptoms", len(symptom_logs))
metric_cols[2].metric("Medications", len(medications))
metric_cols[3].metric("Vitals/labs", len(vitals))

if latest_triage:
    st.info(
        f"Latest triage: {latest_triage.get('urgency_level', 'Routine')} - "
        f"{latest_triage.get('next_step', 'Self-care')}"
    )

cards = st.columns(3, gap="large")
with cards[0]:
    action_card(
        "Chat",
        "Conversation",
        "Ask a health question, receive an evidence-informed answer, and review triage guidance without clutter.",
        "Best for a new question or follow-up.",
    )
    if st.button("Open chat", use_container_width=True):
        st.switch_page("pages/2_Chatbot.py")

with cards[1]:
    action_card(
        "Health Timeline",
        "Longitudinal view",
        "See your saved conditions, medications, allergies, vitals, major events, and trend cards in one place.",
        "Best for reviewing what changed over time.",
    )
    if st.button("Open timeline", use_container_width=True):
        st.switch_page("pages/3_Health_Timeline.py")

with cards[2]:
    action_card(
        "Find Clinical Trials",
        "Recruiting studies",
        "Use saved health context and your location to rank recruiting ClinicalTrials.gov records.",
        "Best for discovering possible research options.",
    )
    if st.button("Find trials", use_container_width=True):
        st.switch_page("pages/4_Find_Clinical_Trials.py")

