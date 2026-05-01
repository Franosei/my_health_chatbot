import html

import requests
import streamlit as st

from app_ui.theme import format_timestamp, inject_custom_css
from backend.clinical_trials import (
    build_trial_search_profile,
    find_matching_trials,
    location_label,
)
from backend.product_config import PRODUCT_NAME, SUPPORT_EMAIL
from backend.user_store import UserStore


st.set_page_config(
    page_title=f"Find Clinical Trials - {PRODUCT_NAME}",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()


def clean(value: object, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def join_items(items: list[str], limit: int = 5) -> str:
    cleaned = []
    seen = set()
    for item in items:
        text = clean(item)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return ", ".join(cleaned[:limit])


def match_band(score: int) -> str:
    if score >= 90:
        return "Very strong record match"
    if score >= 80:
        return "Strong record match"
    if score >= 65:
        return "Possible match"
    return "Needs manual review"


def render_contact(contact: dict) -> str:
    pieces = [
        contact.get("name", ""),
        contact.get("role", ""),
        contact.get("source", ""),
        contact.get("phone", ""),
        contact.get("email", ""),
    ]
    return " - ".join(piece for piece in [clean(part) for part in pieces] if piece)


def render_coverage_bar(found_for: list, total: int) -> str:
    if not total:
        return ""
    count = len(found_for)
    pct = int(count / total * 100)
    terms_escaped = " &middot; ".join(html.escape(t) for t in found_for[:6])
    return (
        f'<div class="coverage-bar-wrap">'
        f'<span class="coverage-label">Matched {count} of {total} condition search(es) ({pct}%)</span>'
        f'<div class="coverage-bar"><div class="coverage-fill" style="width:{pct}%"></div></div>'
        f'<div class="coverage-terms">{terms_escaped}</div>'
        f'</div>'
    )


def render_trial_card(trial: dict, index: int) -> None:
    score = int(trial.get("match_score", 0))
    best_location = trial.get("best_location") or {}
    contacts = trial.get("contacts", [])
    officials = trial.get("officials", [])
    title = clean(trial.get("title"), "Untitled trial")
    nct_id = clean(trial.get("nct_id"))
    url = clean(trial.get("url"))
    location = location_label(best_location) if best_location else "No trial site listed in the returned record"
    contact_line = render_contact(contacts[0]) if contacts else "No public contact listed"
    physician_line = "No physician or official listed"
    if officials:
        official = officials[0]
        physician_line = " - ".join(
            part for part in [
                clean(official.get("name")),
                clean(official.get("role")),
                clean(official.get("affiliation")),
            ] if part
        )

    found_for = trial.get("found_for_conditions", [])
    total_searched = trial.get("total_conditions_searched", 0)
    coverage_html = render_coverage_bar(found_for, total_searched)

    st.markdown(
        f"""
        <div class="trial-result-card">
            <div class="trial-result-head">
                <span>#{index} | {html.escape(match_band(score))}</span>
                <strong>{score}%</strong>
            </div>
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(join_items(trial.get("conditions", []), limit=6) or "Condition not listed")}</p>
            {coverage_html}
            <div class="trial-chip-row">
                <span>{html.escape(clean(trial.get("status"), "Recruiting"))}</span>
                <span>{html.escape(clean(trial.get("phase"), "Phase not listed"))}</span>
                <span>{html.escape(clean(trial.get("study_type"), "Study type not listed"))}</span>
            </div>
            <div class="trial-detail-grid">
                <div>
                    <strong>Location</strong>
                    <p>{html.escape(location)}</p>
                </div>
                <div>
                    <strong>Hospital / site</strong>
                    <p>{html.escape(best_location.get("facility") or "Not listed")}</p>
                </div>
                <div>
                    <strong>Physician / official</strong>
                    <p>{html.escape(physician_line)}</p>
                </div>
                <div>
                    <strong>Contact</strong>
                    <p>{html.escape(contact_line)}</p>
                </div>
            </div>
            {f"<a href='{html.escape(url)}' target='_blank'>Open ClinicalTrials.gov record ({html.escape(nct_id)})</a>" if url else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Why this score? How was this trial matched?", expanded=False):
        coverage_count = trial.get("condition_coverage", 0)
        total_searched = trial.get("total_conditions_searched", 0) or 1
        ali = trial.get("alignment_score", 0)
        cov = trial.get("coverage_score", 0)
        loc = trial.get("location_score", 0)

        def _bar(value: int, maximum: int) -> str:
            pct = min(100, int(value / maximum * 100))
            return (
                f'<div style="background:rgba(22,59,71,.08);border-radius:999px;'
                f'height:8px;margin:3px 0 10px">'
                f'<div style="width:{pct}%;height:100%;border-radius:999px;'
                f'background:linear-gradient(90deg,var(--primary,#163b47),var(--accent,#2ab5a0))'
                f'"></div></div>'
            )

        def _status_badge(status: str) -> str:
            colours = {
                "included": ("background:#d4edda;color:#155724", "INCLUDED"),
                "excluded": ("background:#f8d7da;color:#721c24", "EXCLUDED"),
                "unknown":  ("background:#fff3cd;color:#856404", "UNKNOWN"),
            }
            style, label = colours.get(status, colours["unknown"])
            return (
                f'<span style="{style};padding:2px 8px;border-radius:4px;'
                f'font-size:.75rem;font-weight:700">{label}</span>'
            )

        age_status = trial.get("age_status", "unknown")
        sex_status = trial.get("sex_status", "unknown")
        age_reason = html.escape(trial.get("age_reason", ""))
        sex_reason = html.escape(trial.get("sex_reason", ""))

        st.markdown(
            f"""
<div style="font-size:.9rem;line-height:1.8">

<strong>How the score is calculated</strong>
<table style="width:100%;border-collapse:collapse;margin:.5rem 0 1rem">
<tr><td style="padding:2px 8px 2px 0;color:#555;white-space:nowrap">Condition alignment (LLM)</td>
    <td style="width:100%">{_bar(ali, 50)}</td>
    <td style="padding:2px 0 2px 12px;white-space:nowrap;font-weight:600">{ali}/50</td></tr>
<tr><td style="padding:2px 8px 2px 0;color:#555;white-space:nowrap">Multi-condition coverage</td>
    <td>{_bar(cov, 30)}</td>
    <td style="padding:2px 0 2px 12px;white-space:nowrap;font-weight:600">{cov}/30</td></tr>
<tr><td style="padding:2px 8px 2px 0;color:#555;white-space:nowrap">Location</td>
    <td>{_bar(loc, 20)}</td>
    <td style="padding:2px 0 2px 12px;white-space:nowrap;font-weight:600">{loc}/20</td></tr>
</table>

<strong>Condition alignment</strong><br>
This trial appeared in <strong>{coverage_count} of {total_searched}</strong> separate condition
searches. The LLM assessed how clinically relevant this trial is to your health profile.
<br><br>

<strong>Age eligibility</strong> &nbsp;{_status_badge(age_status)}<br>
<span style="color:#555">{age_reason}</span>
<br><br>

<strong>Biological sex eligibility</strong> &nbsp;{_status_badge(sex_status)}<br>
<span style="color:#555">{sex_reason}</span>
<br><br>

<strong>Contact availability</strong> (not scored — for action only)<br>
<span style="color:#555">{"Public contact details are listed for this trial." if contacts else "No public contact details are listed."}</span>

</div>
""",
            unsafe_allow_html=True,
        )

        reasoning = trial.get("llm_reasoning", "")
        if reasoning:
            st.info(f"**Clinical assessment:** {reasoning}")

        aligned = trial.get("aligned_conditions", [])
        if aligned:
            st.markdown("**Your conditions that align with this trial's inclusion criteria:**")
            for item in aligned:
                st.markdown(f"- {item}")

        exclusion_risks = trial.get("exclusion_risks", [])
        if exclusion_risks:
            st.warning("**Potential exclusion factors to discuss with the study team:**")
            for item in exclusion_risks:
                st.markdown(f"- {item}")

        if found_for:
            st.markdown("**Your condition searches that found this trial:**")
            for term in found_for:
                st.markdown(f"- {term}")

        if trial.get("interventions"):
            st.markdown("**Interventions being studied:** " + ", ".join(trial["interventions"]))

        eligibility_text = clean(trial.get("eligibility"), "")
        if eligibility_text:
            st.markdown("**Full eligibility criteria**")
            st.text(eligibility_text[:2500])

        if trial.get("locations"):
            st.markdown("**Trial sites**")
            for location_item in trial["locations"][:8]:
                st.markdown(f"- {location_label(location_item)}")

        if contacts:
            st.markdown("**Public contacts**")
            for contact in contacts[:6]:
                st.markdown(f"- {render_contact(contact)}")


current_user = st.session_state.get("current_user")
if not current_user:
    st.warning("Please sign in to find clinical trials.")
    st.session_state.auth_panel = "Sign in"
    st.switch_page("pages/1_Landing.py")

profile = UserStore.get_user_profile(current_user)
memory = UserStore.get_longitudinal_memory(current_user)
symptom_logs = UserStore.get_symptom_logs(current_user, limit=None)
medications = UserStore.get_medications(current_user)
allergies = UserStore.get_allergies(current_user)
vitals = UserStore.get_vitals(current_user, limit=None)
triage_summaries = UserStore.get_triage_summaries(current_user, limit=None)

# Restore persisted search result into session state on page load
# so results survive app restarts without requiring a new search
if "trial_search_result" not in st.session_state or st.session_state.get("_trial_result_user") != current_user:
    persisted = UserStore.get_trial_search_result(current_user)
    st.session_state.trial_search_result = persisted
    st.session_state["_trial_result_user"] = current_user

trial_profile = build_trial_search_profile(
    profile=profile,
    memory=memory,
    symptom_logs=symptom_logs,
    medications=medications,
    allergies=allergies,
    vitals=vitals,
    triage_summaries=triage_summaries,
)

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-profile">
            <div class="feature-eyebrow">Clinical trials</div>
            <h2>{html.escape(clean(profile.get("display_name"), current_user))}</h2>
            <span class="clinical-role-badge">{html.escape(clean(profile.get("clinical_role") or profile.get("role"), "Patient / Individual"))}</span>
            <p>{html.escape(clean(profile.get("care_context"), "Personal health guidance"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Workspace", use_container_width=True):
        st.switch_page("pages/2_Workspace.py")
    if st.button("Open chat", use_container_width=True):
        st.switch_page("pages/2_Chatbot.py")
    if st.button("Health timeline", use_container_width=True):
        st.switch_page("pages/3_Health_Timeline.py")
    if st.button("Sign out", use_container_width=True):
        st.session_state.current_user = None
        st.session_state.history_user = None
        st.session_state.chat_history = []
        st.session_state.auth_panel = "Sign in"
        st.switch_page("pages/1_Landing.py")
    st.caption(f"Support: {SUPPORT_EMAIL}")

st.markdown(
    f"""
    <div class="workspace-hero trial-hero">
        <div class="feature-eyebrow">{PRODUCT_NAME}</div>
        <h1>Find Clinical Trials</h1>
        <p>
            Search recruiting ClinicalTrials.gov studies using saved health context and your location. Results are ranked by record match and location, then shown for clinical discussion.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Trial matching is a pre-screening aid only. A study team or clinician must confirm eligibility before any enrolment decision."
)

summary_cols = st.columns(4, gap="small")
summary_cols[0].metric("Symptoms", len(symptom_logs))
summary_cols[1].metric("Medications", len(medications))
summary_cols[2].metric("Vitals/labs", len(vitals))
summary_cols[3].metric("Triage records", len(triage_summaries))


_COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Argentina", "Australia", "Austria",
    "Bangladesh", "Belgium", "Bolivia", "Brazil", "Bulgaria", "Cambodia",
    "Cameroon", "Canada", "Chile", "China", "Colombia", "Croatia", "Czechia",
    "Denmark", "Ecuador", "Egypt", "Ethiopia", "Finland", "France", "Germany",
    "Ghana", "Greece", "Guatemala", "Hungary", "India", "Indonesia", "Iran",
    "Iraq", "Ireland", "Israel", "Italy", "Japan", "Jordan", "Kenya",
    "Malaysia", "Mexico", "Morocco", "Nepal", "Netherlands", "New Zealand",
    "Nigeria", "Norway", "Pakistan", "Peru", "Philippines", "Poland",
    "Portugal", "Romania", "Russia", "Saudi Arabia", "Serbia", "Singapore",
    "South Africa", "South Korea", "Spain", "Sri Lanka", "Sweden",
    "Switzerland", "Taiwan", "Tanzania", "Thailand", "Turkey", "Uganda",
    "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
    "Vietnam", "Zimbabwe",
]

with st.form("clinical_trial_search_form"):
    location = st.selectbox(
        "Your country",
        options=_COUNTRIES,
        index=_COUNTRIES.index("United Kingdom"),
        help="Country is used to rank nearby trial sites higher in the results.",
    )
    submitted = st.form_submit_button("Find recruiting trials", type="primary", use_container_width=True)

if submitted:
    with st.spinner("Analysing your health data and searching ClinicalTrials.gov..."):
        try:
            result_new = find_matching_trials(
                profile=trial_profile,
                location_query=location,
                max_results=10,
            )
            st.session_state.trial_search_result = result_new
            UserStore.save_trial_search_result(current_user, result_new)
        except requests.RequestException as exc:
            error_result = {
                "error": f"ClinicalTrials.gov search failed: {exc}",
                "trials": [],
                "condition_terms": [],
                "medication_terms": [],
                "location": location,
                "searched_at": "",
            }
            st.session_state.trial_search_result = error_result

result = st.session_state.get("trial_search_result")
if result:
    if result.get("error"):
        st.error(result["error"])
    else:
        searched_at = format_timestamp(result.get("searched_at", ""))
        st.markdown(
            f"""
            <div class="toolbar-card">
                <span>Recruiting only</span>
                <span>Top {len(result.get("trials", []))} ranked matches</span>
                <span>Country: {html.escape(result.get("location", ""))}</span>
                <span>{html.escape(searched_at or "Fresh search")}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cond_terms = result.get("condition_terms", [])
        med_terms = result.get("medication_terms", [])
        if cond_terms or med_terms:
            all_terms = [f"**{t}**" for t in cond_terms] + [f"**{t}** (drug)" for t in med_terms]
            st.caption("Separate searches run for: " + " · ".join(all_terms))

        strong_matches = [trial for trial in result.get("trials", []) if trial.get("match_score", 0) >= 80]
        if strong_matches:
            st.success(f"{len(strong_matches)} returned trial(s) scored at least 80% on record and location match.")
        else:
            st.warning("No returned trial reached the 80% strong-match threshold. The cards below need careful manual review.")

        for index, trial in enumerate(result.get("trials", []), start=1):
            render_trial_card(trial, index)

        if not result.get("trials"):
            st.info("No recruiting trials were returned for this saved health context and location.")
