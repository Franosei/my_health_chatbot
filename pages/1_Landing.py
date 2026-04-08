from datetime import datetime, timezone

import streamlit as st

from app_ui.theme import inject_custom_css
from backend.product_config import (
    FOUNDER_NAME,
    PRIVACY_NOTICE_POINTS,
    PRODUCT_NAME,
    PRODUCT_SUBTITLE,
    PRODUCT_TAGLINE,
    ROLE_OPTIONS,
    SUPPORT_EMAIL,
    TERMS_VERSION,
    default_care_context_for_role,
    get_terms_for_role,
    is_clinician_role,
)
from backend.user_store import UserStore

st.set_page_config(
    page_title=PRODUCT_NAME,
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_custom_css()


def render_feature_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="surface-card feature-card">
            <div class="feature-eyebrow">Overview</div>
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_role_terms(role_label: str) -> None:
    terms = get_terms_for_role(role_label)
    bullets_html = "".join(f"<li>{item}</li>" for item in terms["bullets"])
    st.markdown(
        f"""
        <div class="surface-card terms-card">
            <div class="feature-eyebrow">Terms and Conditions</div>
            <h3>{terms["title"]}</h3>
            <p>{terms["summary"]}</p>
            <ul class="legal-list">
                {bullets_html}
            </ul>
            <p class="terms-version">Terms version: {TERMS_VERSION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_privacy_notice() -> None:
    bullets_html = "".join(f"<li>{item}</li>" for item in PRIVACY_NOTICE_POINTS)
    st.markdown(
        f"""
        <div class="surface-card support-card">
            <div class="feature-eyebrow">Privacy Notice</div>
            <h3>Account, privacy, and support information</h3>
            <ul class="legal-list">
                {bullets_html}
            </ul>
            <p class="support-contact">
                Support contact: <strong>{FOUNDER_NAME}</strong><br />
                <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def is_valid_email(email: str) -> bool:
    email = (email or "").strip()
    return bool(email and "@" in email and "." in email.split("@")[-1])


current_user = st.session_state.get("current_user")
current_profile = UserStore.get_user_profile(current_user) if current_user else {}

st.markdown(
    f"""
    <div class="hero-shell login-hero">
        <div class="hero-copy">
            <div class="eyebrow-pill">Secure Access</div>
            <h1 class="brand-hero-title">{PRODUCT_NAME}</h1>
            <p class="brand-hero-tagline">{PRODUCT_TAGLINE}</p>
            <p>{PRODUCT_SUBTITLE}</p>
            <div class="hero-pill-row">
                <span class="hero-pill">Username or email sign-in</span>
                <span class="hero-pill">Role-aware account setup</span>
                <span class="hero-pill">Saved conversation history</span>
                <span class="hero-pill">Support available</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if current_user:
    status_cols = st.columns([1.4, 1], gap="large")
    with status_cols[0]:
        st.markdown(
            f"""
            <div class="surface-card status-card">
                <div class="feature-eyebrow">Signed In</div>
                <h3>Welcome back, {current_profile.get("display_name", current_user)}.</h3>
                <p>Your account is active and ready to continue.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_cols[1]:
        if st.button("Continue", type="primary", use_container_width=True):
            st.switch_page("pages/2_Chatbot.py")

main_left, main_right = st.columns([1.08, 0.92], gap="large")

with main_left:
    card_cols = st.columns(2, gap="medium")
    with card_cols[0]:
        render_feature_card(
            "How access works",
            "Choose your role, review the matching terms, create your account, and return later with the same sign-in details.",
        )
    with card_cols[1]:
        render_feature_card(
            "What stays with your account",
            "Your saved conversation history, profile details, uploads, and account records remain attached to your sign-in.",
        )

    st.markdown(
        """
        <div class="surface-card checklist-card">
            <div class="feature-eyebrow">Using Dr. Charlotte</div>
            <h3>Account access and continuity</h3>
            <ul class="legal-list">
                <li>Sign in using either a username or an email address</li>
                <li>Select the role that reflects how you will use the service</li>
                <li>Accept the terms and conditions written for that role before account creation</li>
                <li>Keep your account history available when you return</li>
                <li>Contact Francis Osei directly for support, privacy, or account questions</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

#    st.markdown(
#        f"""
#        <div class="surface-card note-card">
#            <div class="feature-eyebrow">Support</div>
#            <p>For account access, privacy, or general support, contact <strong>{FOUNDER_NAME}</strong> at <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>.</p>
#        </div>
#        """,
#        unsafe_allow_html=True,
#    )

with main_right:
    if st.session_state.get("signup_role_reference") not in ROLE_OPTIONS:
        st.session_state.signup_role_reference = ROLE_OPTIONS[0]

    tab_login, tab_signup = st.tabs(["Sign in", "Create account"])

    with tab_login:
        st.markdown("### Sign in to Dr. Charlotte")
        st.caption("Use your username or email address together with your password.")

        login_identifier = st.text_input("Email or username", key="login_identifier")
        login_password = st.text_input("Password", type="password", key="login_password")
        login_submitted = st.button(
            "Sign in",
            type="primary",
            use_container_width=True,
            key="login_submit",
        )

        if login_submitted:
            if not login_identifier or not login_password:
                st.error("Enter your email or username and your password to continue.")
            elif UserStore.authenticate(login_identifier, login_password):
                resolved_user = UserStore.resolve_login_username(login_identifier)
                if not resolved_user:
                    st.error("We could not open your account. Please try again.")
                else:
                    UserStore.update_last_login(resolved_user)
                    st.session_state.current_user = resolved_user
                    st.session_state.history_user = None
                    st.success("Sign-in successful. Opening your account...")
                    st.switch_page("pages/2_Chatbot.py")
            else:
                st.error("The email, username, or password you entered is incorrect.")

    with tab_signup:
        st.markdown("### Create your account")
        st.caption("Choose the role that best describes how this account will be used.")

        selected_role = st.selectbox(
            "Account role",
            ROLE_OPTIONS,
            key="signup_role_selector",
            help="The selected role determines which terms and conditions you must accept before creating the account.",
        )

        if st.session_state.signup_role_reference != selected_role:
            st.session_state.signup_role_reference = selected_role
            st.session_state.signup_accept_role_terms = False

        render_role_terms(selected_role)
        render_privacy_notice()

        signup_name = st.text_input("Full name (optional)", key="signup_name")
        signup_email = st.text_input("Email address", key="signup_email")
        signup_username = st.text_input("Username", key="signup_username")

        signup_org = ""
        if is_clinician_role(selected_role):
            signup_org = st.text_input("Organisation (optional)", key="signup_org")

        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm password", type="password", key="signup_confirm")

        st.markdown(
            """
            <div class="pw-requirements-box">
                <div class="pw-req-header">Password requirements</div>
                <div class="pw-req-row">
                    <span class="pw-req-badge">&#10003;</span>
                    <span>Minimum <strong>8 characters</strong></span>
                </div>
                <div class="pw-req-row">
                    <span class="pw-req-badge">&#10003;</span>
                    <span>Must match the confirmation field exactly</span>
                </div>
                <div class="pw-req-row">
                    <span class="pw-req-badge"></span>
                    <span>Stored using a one-way cryptographic hash</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        role_terms = get_terms_for_role(selected_role)
        accept_role_terms = st.checkbox(
            role_terms["acknowledgement"],
            key="signup_accept_role_terms",
        )
        accept_privacy = st.checkbox(
            f"I have read the privacy notice and I understand that support and account questions can be sent to {SUPPORT_EMAIL}.",
            key="signup_accept_privacy",
        )

        signup_submitted = st.button(
            "Create account",
            type="primary",
            use_container_width=True,
            key="signup_submit",
            disabled=not (accept_role_terms and accept_privacy),
        )

        if signup_submitted:
            accepted_at = datetime.now(timezone.utc).isoformat()

            if not signup_email or not signup_username or not signup_password or not signup_confirm:
                st.error("Email, username, password, and password confirmation are required.")
            elif not is_valid_email(signup_email):
                st.error("Enter a valid email address before creating the account.")
            elif signup_password != signup_confirm:
                st.error("The password and confirmation fields must match.")
            elif len(signup_password) < 8:
                st.error("Use a password with at least 8 characters.")
            else:
                created = UserStore.create_user(
                    signup_username,
                    signup_password,
                    display_name=signup_name.strip() or signup_username,
                    email=signup_email,
                    care_context=default_care_context_for_role(selected_role),
                    role=selected_role,
                    clinical_role=selected_role,
                    organization=signup_org,
                    terms_version=TERMS_VERSION,
                    terms_role=selected_role,
                    terms_accepted_at=accepted_at,
                    privacy_accepted_at=accepted_at,
                )

                if not created:
                    st.error("That username or email is already in use, or the account details do not meet policy.")
                else:
                    resolved_user = UserStore.resolve_login_username(signup_username)
                    if resolved_user:
                        UserStore.update_last_login(resolved_user)
                        st.session_state.current_user = resolved_user
                        st.session_state.history_user = None
                        st.success("Account created successfully. Opening your account...")
                        st.switch_page("pages/2_Chatbot.py")
                    else:
                        st.success("Account created successfully. Please sign in to continue.")

    st.markdown(
        f"""
        <div class="surface-card trust-card">
            <div class="feature-eyebrow">Contact</div>
            <h3>{FOUNDER_NAME}</h3>
            <p>Support contact: <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
