from pathlib import Path

import streamlit as st

from app_ui.theme import inject_custom_css
from backend.user_store import UserStore

ASSISTANT_AVATAR = Path("app_ui/static/assistant.png")
CARE_CONTEXTS = [
    "Personal health guidance",
    "Caregiver support",
    "Clinical / hospital use",
]
ROLES = [
    "Individual",
    "Caregiver",
    "Clinician / care team",
]


def feature_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="surface-card feature-card">
            <div class="feature-eyebrow">Capability</div>
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="My Health Checks",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_custom_css()

current_user = st.session_state.get("current_user")
current_profile = UserStore.get_user_profile(current_user) if current_user else {}

st.markdown(
    """
    <div class="hero-shell login-hero">
        <div class="hero-copy">
            <div class="eyebrow-pill">Clinical-grade health intelligence</div>
            <h1>Client-ready health conversations with evidence, continuity, and traceability.</h1>
            <p>
                Designed for individuals, caregivers, and care teams who need polished answers,
                clickable PubMed citations, and a session that resumes exactly where it left off.
            </p>
            <div class="hero-pill-row">
                <span class="hero-pill">Inline citations</span>
                <span class="hero-pill">Audit-ready traces</span>
                <span class="hero-pill">Persistent user history</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if current_user:
    st.markdown(
        f"""
        <div class="surface-card status-card">
            <div>
                <div class="feature-eyebrow">Workspace Ready</div>
                <h3>Welcome back, {current_profile.get("display_name", current_user)}.</h3>
                <p>Your saved conversation history, uploaded records, and source traces are available.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Resume workspace", type="primary", use_container_width=True):
        st.switch_page("pages/2_Chatbot.py")

hero_left, hero_right = st.columns([1.25, 0.9], gap="large")

with hero_left:
    card_cols = st.columns(3, gap="medium")
    with card_cols[0]:
        feature_card(
            "Evidence You Can Inspect",
            "Responses include clickable references and source drawers so users can review the exact literature behind each recommendation.",
        )
    with card_cols[1]:
        feature_card(
            "Continuity Across Logins",
            "Each account keeps prior conversations, uploaded records, and audit traces so the next session starts with context.",
        )
    with card_cols[2]:
        feature_card(
            "Professional Health UX",
            "The assistant speaks in a calm, clinically literate tone built for both personal use and provider-facing workflows.",
        )

with hero_right:
    st.image(str(ASSISTANT_AVATAR), width=144)
    st.markdown(
        """
        <div class="surface-card trust-card">
            <div class="feature-eyebrow">What Makes This Different</div>
            <h3>Research-backed, polished, and built for trust.</h3>
            <p>
                Your users get structured answers, transparent references, and a durable conversation record
                that supports continuity, traceability, and reviewability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

form_left, form_right = st.columns([1.05, 0.95], gap="large")

with form_left:
    tab_login, tab_signup = st.tabs(["Sign in", "Create account"])

    with tab_login:
        st.markdown("### Return to your workspace")
        with st.form("login_form", clear_on_submit=False):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

        if login_submitted:
            if UserStore.authenticate(login_username, login_password):
                normalized_user = login_username.strip().lower()
                UserStore.update_last_login(normalized_user)
                st.session_state.current_user = normalized_user
                st.session_state.history_user = None
                st.success("Signed in successfully. Opening your workspace...")
                st.switch_page("pages/2_Chatbot.py")
            else:
                st.error("The username or password did not match our records.")

    with tab_signup:
        st.markdown("### Build a persistent, audit-ready account")
        with st.form("signup_form", clear_on_submit=False):
            signup_name = st.text_input("Display name")
            signup_email = st.text_input("Email address (optional)")
            signup_username = st.text_input("Username")
            signup_care_context = st.selectbox("Primary use case", CARE_CONTEXTS)
            signup_role = st.selectbox("Role", ROLES)
            signup_org = st.text_input("Organization (optional)")
            signup_password = st.text_input("Password", type="password")
            signup_confirm = st.text_input("Confirm password", type="password")
            signup_submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)

        if signup_submitted:
            if not signup_username or not signup_password:
                st.error("Please provide both a username and password.")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match.")
            elif len(signup_password) < 8:
                st.error("Please use a password with at least 8 characters.")
            else:
                created = UserStore.create_user(
                    signup_username,
                    signup_password,
                    display_name=signup_name.strip() or signup_username,
                    email=signup_email,
                    care_context=signup_care_context,
                    role=signup_role,
                    organization=signup_org,
                )
                if created:
                    st.success("Account created. You can sign in and continue from any future session.")
                else:
                    st.error("That username is unavailable, or the password does not meet the minimum policy.")

with form_right:
    st.markdown(
        """
        <div class="surface-card checklist-card">
            <div class="feature-eyebrow">Included In Every Session</div>
            <h3>Traceable by design</h3>
            <ul>
                <li>Structured answer sections for easy reading</li>
                <li>Clickable citations that open the source article</li>
                <li>Saved user profile and prior chat continuity</li>
                <li>Audit trail export for review and governance</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="surface-card note-card">
            <div class="feature-eyebrow">Built For</div>
            <p>Personal health research, caregiver handoffs, second-look evidence review, and provider-adjacent education.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
