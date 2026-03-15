from datetime import datetime, timezone
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

st.set_page_config(
    page_title="Dr. Charlotte",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_custom_css()


# ── CONSENT GATE ──────────────────────────────────────────────────────────────
def render_consent_page() -> None:
    """Render the full-page GDPR data-notice and consent form."""

    # Brand header
    st.markdown(
        """
        <div class="consent-shell">
            <div class="consent-brand">
                <div class="consent-brand-icon">✚</div>
                <div class="consent-brand-name">Dr. Charlotte</div>
                <div class="consent-brand-sub">Clinical-grade health intelligence</div>
            </div>
            <div style="text-align:center;margin-top:0.4rem;">
                <span class="eyebrow-pill" style="font-size:13px;">
                    Data Privacy Notice &amp; Informed Consent — Please read carefully before continuing
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="consent-intro">
            Dr. Charlotte is a health information assistant. Before you create an account or sign in,
            we are required under the <strong>General Data Protection Regulation (GDPR)</strong> and
            applicable data-protection law to explain clearly what personal data we collect, how we
            use it, where it is stored, and what your rights are. <strong>You must actively accept
            these terms to use the service.</strong>
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Card grid ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(
            """
            <div class="consent-card">
                <div class="consent-card-icon"></div>
                <h3>What data we collect</h3>
                <ul>
                    <li><strong>Account information</strong> — username, display name, optional email address, role, and organisation.</li>
                    <li><strong>Authentication credentials</strong> — your password is <em>never</em> stored in plain text; it is hashed using a one-way cryptographic algorithm (bcrypt) before being written to storage.</li>
                    <li><strong>Conversation history</strong> — messages you send and the responses you receive are saved to your account so the session can resume across logins.</li>
                    <li><strong>Uploaded health documents</strong> — any files you upload (e.g. lab reports, discharge summaries) are processed locally. Before indexing, personal identifiers (names, dates, phone numbers, addresses, ID numbers) are automatically redacted by our anonymisation pipeline.</li>
                    <li><strong>Interaction traces and audit logs</strong> — timestamped records of evidence retrieval steps are retained to support clinical governance, audit, and traceability requirements.</li>
                    <li><strong>Session metadata</strong> — last-login timestamps and usage context to personalise your experience.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="consent-card">
                <div class="consent-card-icon"></div>
                <h3>How we protect your data</h3>
                <ul>
                    <li><strong>Password security</strong> — all passwords are hashed with bcrypt before storage. Plain-text passwords are never written to disk or logged.</li>
                    <li><strong>Document anonymisation</strong> — uploaded records are automatically processed to redact names, dates of birth, addresses, phone numbers, email addresses, and patient identifiers before any content is indexed or passed to the AI model.</li>
                    <li><strong>Access control</strong> — each user account is strictly isolated. You can only access your own data after successful authentication.</li>
                    <li><strong>Audit trails</strong> — all significant actions (login, upload, query, export) are logged with a timestamp so the data lifecycle remains traceable and reviewable.</li>
                    <li><strong>No advertising use</strong> — your health data is never sold, rented, or used for advertising purposes.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="consent-card">
                <div class="consent-card-icon"></div>
                <h3>Where your data is stored</h3>
                <ul>
                    <li><strong>Primary storage</strong> — your account data, conversation history, and audit records are held in a secure database on the server that hosts this application.</li>
                    <li><strong>Uploaded documents</strong> — files you upload are written to an <code>uploads/</code> directory on the application server and indexed locally for retrieval. They are not sent to any external cloud storage provider.</li>
                    <li><strong>Data location</strong> — storage remains within the server environment where Dr. Charlotte is deployed. If you are using a cloud-hosted deployment, data resides in the data centre selected by the operator.</li>
                    <li><strong>No third-party data warehousing</strong> — your personal data is not replicated to or shared with any third-party analytics or data-brokerage platforms.</li>
                </ul>
                <div class="consent-highlight">
                    Third-party AI services (OpenAI) receive only the anonymised text of your query and relevant anonymised document excerpts in order to generate a response. No persistent user profile is held by these providers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="consent-card">
                <div class="consent-card-icon"></div>
                <h3>Your GDPR rights</h3>
                <p>Under the General Data Protection Regulation you have the following rights, exercisable at any time by contacting the data controller:</p>
                <ul>
                    <li><strong>Right of access</strong> — you may request a copy of all personal data held about you. Use the <em>Download audit JSON</em> button in your workspace to export your data immediately.</li>
                    <li><strong>Right to rectification</strong> — you may correct inaccurate personal information at any time via your profile settings.</li>
                    <li><strong>Right to erasure ("right to be forgotten")</strong> — you may request deletion of your account and all associated data.</li>
                    <li><strong>Right to data portability</strong> — you may download your full data snapshot in JSON format from your workspace sidebar.</li>
                    <li><strong>Right to restrict processing</strong> — you may request that we limit how your data is used while a complaint is investigated.</li>
                    <li><strong>Right to withdraw consent</strong> — you may withdraw consent at any time. Withdrawal does not affect the lawfulness of processing carried out before withdrawal.</li>
                    <li><strong>Right to lodge a complaint</strong> — you have the right to lodge a complaint with your national data-protection authority.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Retention & third-party services ──────────────────────────────────
    st.markdown(
        """
        <div class="consent-card consent-card-wide">
            <div class="consent-card-icon"></div>
            <h3>Data retention &amp; third-party services</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                <div>
                    <strong>Retention periods</strong>
                    <ul>
                        <li>Account and profile data — retained for the lifetime of your account, or until you request deletion.</li>
                        <li>Conversation history — retained indefinitely within your account so sessions can resume; you may delete your chat history at any time via the workspace.</li>
                        <li>Uploaded documents — retained until you remove them or request account deletion.</li>
                        <li>Audit logs — retained for a minimum of 12 months to support clinical governance requirements.</li>
                    </ul>
                </div>
                <div>
                    <strong>Third-party services used</strong>
                    <ul>
                        <li><strong>OpenAI GPT-4o-mini</strong> — used to generate AI responses. Queries are sent as anonymised text; no user account data is transmitted. Subject to OpenAI's data-processing terms.</li>
                        <li><strong>PubMed / NCBI (US National Library of Medicine)</strong> — used to retrieve published medical literature citations. Only search queries are sent; no personal data is transmitted.</li>
                        <li><strong>No other third-party processors</strong> receive your personal or health data.</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Password requirements notice ───────────────────────────────────────
    st.markdown(
        """
        <div class="consent-card consent-card-wide consent-pw-card">
            <div class="consent-card-icon"></div>
            <h3>Password requirements</h3>
            <p>When you create an account, your password must meet the following requirements to protect your health data:</p>
            <div class="pw-req-grid">
                <div class="pw-req-item pw-req-ok">Minimum <strong>8 characters</strong> in length</div>
                <div class="pw-req-item pw-req-ok">Must be entered <strong>identically twice</strong> (confirmation field)</div>
                <div class="pw-req-item pw-req-ok">Stored as a <strong>one-way bcrypt hash</strong> — never in plain text</div>
                <div class="pw-req-item pw-req-ok">Never displayed, logged, or transmitted unencrypted</div>
            </div>
            <p style="margin-top:0.5rem;font-size:13px;color:var(--text-soft);">
                We strongly recommend using a unique password that you do not use for any other service,
                and a password manager to generate and store it securely.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Acceptance form ────────────────────────────────────────────────────
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="consent-accept-box">
            <strong>By clicking "I Accept &amp; Continue" you confirm that:</strong>
            <ul>
                <li>You have read and understood this data privacy notice.</li>
                <li>You consent to the collection, storage, and processing of your data as described above.</li>
                <li>You understand your GDPR rights and how to exercise them.</li>
                <li>You are at least 18 years of age, or accessing this service under appropriate supervision.</li>
            </ul>
            <p style="margin:0;font-size:13px;color:var(--text-soft);">
                Consent is required to use Dr. Charlotte. If you decline, no data will be collected and
                you will not be able to access the service.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    agreed = st.checkbox(
        "I have read the data privacy notice above and I give my informed consent to the processing of my data as described.",
        key="consent_checkbox",
    )

    btn_accept, btn_decline = st.columns([2, 1], gap="small")
    with btn_accept:
        accept_clicked = st.button(
            "I Accept & Continue to Dr. Charlotte",
            type="primary",
            use_container_width=True,
            disabled=not agreed,
        )
    with btn_decline:
        decline_clicked = st.button(
            "Decline",
            use_container_width=True,
        )

    if accept_clicked and agreed:
        st.session_state.consent_given = True
        st.session_state.consent_timestamp = datetime.now(timezone.utc).isoformat()
        st.rerun()

    if decline_clicked:
        st.session_state.consent_declined = True
        st.rerun()


def render_consent_declined() -> None:
    st.markdown(
        """
        <div class="consent-shell">
            <div class="consent-brand">
                <div class="consent-brand-icon">✚</div>
                <div class="consent-brand-name">Dr. Charlotte</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(
        "**Access declined.** You have chosen not to accept the data privacy terms. "
        "Dr. Charlotte cannot be used without your informed consent. No data has been collected."
    )
    st.markdown(
        """
        <div class="consent-card" style="max-width:520px;margin:1rem auto;text-align:center;">
            <p>If you change your mind, you can review the privacy notice and accept at any time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Review the privacy notice again", type="primary"):
        st.session_state.consent_declined = False
        st.rerun()


# Handle declined state
if st.session_state.get("consent_declined"):
    render_consent_declined()
    st.stop()

# Show consent gate if not yet accepted
if not st.session_state.get("consent_given"):
    render_consent_page()
    st.stop()

# ── MAIN LANDING PAGE (post-consent) ─────────────────────────────────────────

current_user = st.session_state.get("current_user")
current_profile = UserStore.get_user_profile(current_user) if current_user else {}


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


st.markdown(
    """
    <div class="hero-shell login-hero">
        <div class="hero-copy">
            <div class="eyebrow-pill">Clinical-grade health intelligence</div>
            <h1 class="brand-hero-title">Dr. Charlotte</h1>
            <p class="brand-hero-tagline">Health conversations with evidence, continuity, and traceability.</p>
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

            # Password requirements — always visible inside the signup form
            st.markdown(
                """
                <div class="pw-requirements-box">
                    <div class="pw-req-header"> Password requirements</div>
                    <div class="pw-req-row">
                        <span class="pw-req-badge">✓</span>
                        <span>Minimum <strong>8 characters</strong> long</span>
                    </div>
                    <div class="pw-req-row">
                        <span class="pw-req-badge">✓</span>
                        <span>Must <strong>match</strong> the confirmation field exactly</span>
                    </div>
                    <div class="pw-req-row">
                        <span class="pw-req-badge"></span>
                        <span>Stored as a <strong>one-way cryptographic hash</strong> — never in plain text</span>
                    </div>
                    <div class="pw-req-row">
                        <span class="pw-req-badge"></span>
                        <span>We recommend a <strong>unique password</strong> not used on any other site</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            signup_submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)

        if signup_submitted:
            if not signup_username or not signup_password:
                st.error("Please provide both a username and password.")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match. Please re-enter both fields.")
            elif len(signup_password) < 8:
                st.error("Password is too short. Please use at least 8 characters.")
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