"""
Dr. Charlotte MCP Server.

Exposes clinical tools via the Model Context Protocol.

Deployed (Railway):
  The FastAPI app mounts this at /mcp automatically via streamable HTTP.
  Set MCP_API_KEY in Railway environment variables to restrict access.
  Claude Desktop config:
    {
      "mcpServers": {
        "dr-charlotte": {
          "url": "https://<your-app>.railway.app/mcp",
          "headers": { "Authorization": "Bearer <MCP_API_KEY>" }
        }
      }
    }

Local (stdio — Claude Desktop direct):
  python -m backend.mcp_server

Tools:
  get_patient_context           — full patient profile, vitals, meds, conditions
  extract_article_evidence      — structured evidence extraction from a medical article
  generate_clinical_note        — generate a SOAP note from a consultation
  send_health_email             — send clinical note or urgent alert by email
  search_trials_for_patient     — search ClinicalTrials.gov for this patient
"""
from __future__ import annotations

import json
import sys

from dotenv import load_dotenv

load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "ERROR: mcp package not installed.\n"
        "Install with:  pip install mcp\n"
        "Then re-run:   python -m backend.mcp_server"
    )
    sys.exit(1)

from backend.clinical_notes import generate_soap_note
from backend.email_service import send_clinical_note_email, send_urgent_care_alert
from backend.summarizer import LLMHelper
from backend.user_store import UserStore

mcp = FastMCP("Dr. Charlotte Clinical Tools")
_llm = LLMHelper()


# ─── Tool: get_patient_context ────────────────────────────────────────────────

@mcp.tool()
def get_patient_context(username: str) -> str:
    """
    Get the complete health context for a patient.

    Returns structured JSON with:
    - profile (demographics, role, care context)
    - conditions (active and past)
    - medications (with dose and schedule)
    - allergies
    - vitals and lab readings (last 20)
    - recent symptom logs
    - latest triage summary
    - longitudinal clinical memory
    """
    profile = UserStore.get_user_profile(username)
    if not profile:
        return json.dumps({"error": f"User '{username}' not found"})

    triage_list = UserStore.get_triage_summaries(username, limit=1)
    return json.dumps(
        {
            "username": username,
            "profile": profile,
            "conditions": UserStore.get_conditions(username),
            "medications": UserStore.get_medications(username),
            "allergies": UserStore.get_allergies(username),
            "vitals": UserStore.get_vitals(username, limit=20),
            "symptom_logs": UserStore.get_symptom_logs(username, limit=10),
            "latest_triage": triage_list[0] if triage_list else {},
            "longitudinal_memory": UserStore.get_longitudinal_memory(username),
        },
        default=str,
    )


# ─── Tool: extract_article_evidence ──────────────────────────────────────────

@mcp.tool()
def extract_article_evidence(
    article_title: str,
    article_text: str,
    patient_question: str,
    patient_conditions: str,
    patient_medications: str,
    patient_age: str = "",
    evidence_tier: int = 3,
) -> str:
    """
    Extract structured, patient-specific evidence from a medical article.

    Parameters:
    - article_title: title of the article
    - article_text: the article body or abstract (first 1200 chars used)
    - patient_question: the patient's health question
    - patient_conditions: comma-separated list of conditions
    - patient_medications: comma-separated list of medications
    - patient_age: age string (optional)
    - evidence_tier: 1=NHS/NICE, 2=systematic review, 3=primary research

    Returns JSON ArticleEvidence object with:
    - question_facts: facts that answer the question
    - patient_aligned_facts: facts matched to patient profile
    - contraindications: relevant warnings
    - patient_relevant_summary: concise patient-specific summary
    - alignment_confidence: 0-1 quality score
    """
    from backend.evidence_extractor import _extract_one_article

    source = {
        "source_id": "mcp-extract",
        "title": article_title,
        "snippet": article_text[:1200],
        "evidence_tier": evidence_tier,
        "tier_label": {1: "NHS/NICE", 2: "Systematic Review", 3: "Research"}.get(evidence_tier, "Research"),
    }
    conditions_list = [c.strip() for c in patient_conditions.split(",") if c.strip()]
    meds_list = [m.strip() for m in patient_medications.split(",") if m.strip()]
    patient_summary = (
        f"Age: {patient_age}; "
        f"Conditions: {patient_conditions}; "
        f"Medications: {patient_medications}"
    )

    result = _extract_one_article(
        _llm, source, patient_question, patient_summary, meds_list, conditions_list
    )
    return result.model_dump_json(indent=2)


# ─── Tool: generate_clinical_note ────────────────────────────────────────────

@mcp.tool()
def generate_clinical_note(
    username: str,
    patient_question: str,
    conversation_summary: str,
    urgency_level: str = "routine",
    next_step: str = "",
) -> str:
    """
    Generate a standard SOAP clinical note from a Dr. Charlotte consultation.

    Parameters:
    - username: the patient's account username
    - patient_question: the question they asked
    - conversation_summary: summary of the conversation context
    - urgency_level: routine | elevated | high | urgent | crisis
    - next_step: recommended action (from triage)

    Returns the generated note as JSON and saves it to the patient's record.
    Note fields: note_id, subjective, objective, assessment, plan,
                 urgency_level, requires_gp_visit, gp_visit_reason.
    """
    triage = (
        {"urgency_level": urgency_level, "next_step": next_step}
        if urgency_level != "routine"
        else None
    )

    note = generate_soap_note(
        username=username,
        conversation_summary=conversation_summary,
        question=patient_question,
        triage_summary=triage,
        llm=_llm,
    )
    UserStore.save_clinical_note(username, note)
    return json.dumps(note, default=str, indent=2)


# ─── Tool: send_health_email ─────────────────────────────────────────────────

@mcp.tool()
def send_health_email(
    username: str,
    email_type: str,
    note_id: str = "",
    urgency_level: str = "high",
    reason: str = "",
) -> str:
    """
    Send a health email to the user.

    email_type:
    - "clinical_note" — sends a saved SOAP note (requires note_id)
    - "urgent_alert"  — sends an urgent care alert (uses urgency_level + reason)

    Returns {"ok": true, "sent_to": "email"} or {"error": "message"}.
    """
    profile = UserStore.get_user_profile(username)
    if not profile:
        return json.dumps({"error": f"User '{username}' not found"})

    email_address = profile.get("email", "")
    display_name = profile.get("display_name", username)

    if not email_address:
        return json.dumps({"error": "User has no email address on file"})

    try:
        if email_type == "clinical_note":
            notes = UserStore.get_clinical_notes(username)
            note = next((n for n in notes if n["note_id"] == note_id), None)
            if not note:
                return json.dumps({"error": f"Note '{note_id}' not found"})
            send_clinical_note_email(email_address, display_name, note)
            UserStore.mark_note_email_sent(username, note_id)
            return json.dumps({"ok": True, "sent_to": email_address})

        if email_type == "urgent_alert":
            if not reason:
                return json.dumps({"error": "Provide a reason for the urgent alert"})
            send_urgent_care_alert(email_address, display_name, reason, urgency_level)
            return json.dumps({"ok": True, "sent_to": email_address})

        return json.dumps({"error": f"Unknown email_type '{email_type}'. Use 'clinical_note' or 'urgent_alert'"})

    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── Tool: search_trials_for_patient ─────────────────────────────────────────

@mcp.tool()
def search_trials_for_patient(
    username: str, location: str = "United Kingdom", max_results: int = 10
) -> str:
    """
    Search ClinicalTrials.gov for recruiting trials matched to a patient's conditions.

    Builds a structured search profile from the patient's saved conditions,
    medications, and symptom logs, then ranks returning trials by profile match.

    Returns JSON array of ranked trial results.
    """
    from backend.clinical_trials import build_trial_search_profile, find_matching_trials

    profile = UserStore.get_user_profile(username)
    if not profile:
        return json.dumps({"error": f"User '{username}' not found"})

    conditions = UserStore.get_conditions(username)
    medications = UserStore.get_medications(username)
    symptom_logs = UserStore.get_symptom_logs(username, limit=20)

    try:
        search_profile = build_trial_search_profile(
            profile=profile,
            conditions=conditions,
            medications=medications,
            symptom_logs=symptom_logs,
        )
        results = find_matching_trials(
            search_profile, location=location, max_results=max_results
        )
        return json.dumps(results, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
