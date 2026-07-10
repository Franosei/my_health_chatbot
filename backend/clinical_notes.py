"""
Clinical note generation and management.

Produces standard SOAP notes (Subjective / Objective / Assessment / Plan)
from a FlynnMed conversation and the patient's stored profile.
Notes are stored per-user and can be edited by clinicians before sharing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from backend.user_store import UserStore, compute_current_age


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_section(value) -> str:
    """Convert LLM output to a clean markdown string regardless of type returned."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for i, item in enumerate(value, 1):
            if isinstance(item, str):
                line = item.strip()
                # already numbered?
                parts.append(line if (line[:2].rstrip('.').isdigit()) else f"{i}. {line}")
            elif isinstance(item, dict):
                parts.append(f"{i}. " + "; ".join(f"**{k}**: {v}" for k, v in item.items()))
        return "\n".join(parts)
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            label = k.replace("_", " ").title()
            if isinstance(v, list):
                lines.append(f"**{label}:**")
                for item in v:
                    lines.append(f"- {item}")
            elif isinstance(v, dict):
                lines.append(f"**{label}:**")
                for sk, sv in v.items():
                    lines.append(f"- {sk.replace('_', ' ').title()}: {sv}")
            else:
                lines.append(f"**{label}:** {v}")
        return "\n".join(lines)
    return str(value).strip()


def _build_objective_section(
    user_profile: dict,
    vitals: List[Dict],
    medications: List[Dict],
    conditions: List[Dict],
    allergies: List[Dict],
) -> str:
    """Build the Objective section from structured patient data."""
    lines: List[str] = []

    age = compute_current_age(user_profile.get("date_of_birth", ""))
    sex = user_profile.get("biological_sex", "Not stated")
    dob = user_profile.get("date_of_birth", "")
    demo = " | ".join(filter(None, [f"Age {age}" if age else "", sex, dob]))
    if demo:
        lines.append(f"Demographics: {demo}")

    active_conditions = [c["name"] for c in conditions if c.get("status") == "active"]
    if active_conditions:
        lines.append(f"Active conditions: {', '.join(active_conditions)}")

    past_conditions = [c["name"] for c in conditions if c.get("status") in ("past", "resolved")]
    if past_conditions:
        lines.append(f"Past conditions: {', '.join(past_conditions[:4])}")

    if medications:
        med_list = [
            f"{m['name']}{' ' + m['dose'] if m.get('dose') else ''}"
            f"{' (' + m['schedule'] + ')' if m.get('schedule') else ''}"
            for m in medications[:10]
        ]
        lines.append(f"Current medications: {'; '.join(med_list)}")

    if allergies:
        allergy_list = [
            f"{a['name']} ({a.get('severity', 'unknown severity')}, {a.get('allergy_type', 'unknown type')})"
            for a in allergies[:6]
        ]
        lines.append(f"Allergies / ADRs: {'; '.join(allergy_list)}")

    if vitals:
        seen_types: set = set()
        vital_lines: List[str] = []
        for v in vitals:
            vtype = v.get("type", "")
            if vtype and vtype not in seen_types:
                seen_types.add(vtype)
                label = vtype.replace("_", " ").title()
                recorded = v.get("recorded_on", "")[:10]
                vital_lines.append(
                    f"{label}: {v.get('value', '')} {v.get('unit', '')} (recorded {recorded})"
                )
        if vital_lines:
            lines.append("Recent vitals / labs:\n  " + "\n  ".join(vital_lines[:10]))

    return "\n".join(lines) if lines else "No objective data recorded in this account."


def generate_soap_note(
    username: str,
    conversation_summary: str,
    question: str,
    triage_summary: Optional[Dict],
    llm,
    vitals: Optional[List[Dict]] = None,
    medications: Optional[List[Dict]] = None,
    conditions: Optional[List[Dict]] = None,
    allergies: Optional[List[Dict]] = None,
    trace_id: Optional[str] = None,
) -> Dict:
    """
    Generate a SOAP note from the conversation context and patient profile.
    Returns a note dict ready for storage. Does NOT save — caller decides.
    """
    from backend.summarizer import LLMHelper

    user_profile = UserStore.get_user_profile(username) or {}
    resolved_vitals = vitals if vitals is not None else UserStore.get_vitals(username, limit=20)
    resolved_meds = medications if medications is not None else UserStore.get_medications(username)
    resolved_conditions = conditions if conditions is not None else UserStore.get_conditions(username)
    resolved_allergies = allergies if allergies is not None else UserStore.get_allergies(username)

    objective_section = _build_objective_section(
        user_profile, resolved_vitals, resolved_meds, resolved_conditions, resolved_allergies
    )

    urgency = "routine"
    requires_gp = False
    gp_reason = ""
    if triage_summary:
        urgency = triage_summary.get("urgency_level", "routine").lower()
        next_step = triage_summary.get("next_step", "")
        requires_gp = urgency in ("high", "urgent", "crisis") or "gp" in next_step.lower()
        gp_reason = next_step

    role_key = (UserStore.get_user_profile(username) or {}).get("clinical_role", "doctor")
    role_guidance = {
        "doctor": (
            "Use full UK GP/hospital SOAP format. "
            "Assessment: differential diagnosis, clinical impression, risk stratification. "
            "Plan: numbered investigations, referrals, medications, follow-up, safety-netting."
        ),
        "nurse": (
            "Use a nursing SOAP format. "
            "Objective: observations, NEWS2 score if relevant, pressure area/falls risk. "
            "Assessment: nursing diagnosis, risk scores. "
            "Plan: nursing interventions, care tasks, patient education, escalation criteria, handover notes."
        ),
        "midwife": (
            "Use a midwifery SOAP format. "
            "Objective: maternal observations, fetal assessment (movement, CTG if relevant), gestation. "
            "Assessment: maternal and fetal risk assessment. "
            "Plan: maternity care pathway, referrals, birth plan considerations."
        ),
        "physiotherapist": (
            "Use a physiotherapy SOAP format. "
            "Objective: range of motion, strength grades, special orthopaedic tests, functional assessment. "
            "Assessment: clinical impression, problem list, functional diagnosis. "
            "Plan: treatment goals, exercise programme, manual therapy, home exercise plan, review date."
        ),
    }.get(str(role_key).lower(), (
        "Use a standard UK clinical SOAP format appropriate for this clinician's role."
    ))

    prompt = (
        "You are a clinical note writer for FlynnMed. "
        f"Generate a SOAP note for a {role_key} in UK clinical format.\n"
        f"Role guidance: {role_guidance}\n\n"
        f"CONSULTATION SUMMARY:\n{conversation_summary}\n\n"
        f"PATIENT QUESTION: {question}\n\n"
        f"OBJECTIVE DATA (use exactly this data, formatted cleanly):\n{objective_section}\n\n"
        f"URGENCY LEVEL: {urgency}\n\n"
        "IMPORTANT: ALL field values MUST be plain strings (no lists, no dicts, no JSON objects inside).\n"
        "Return ONLY a JSON object with these exact string fields:\n"
        "{\n"
        '  "subjective": "2-4 sentences of patient narrative in clinical language.",\n'
        '  "objective": "Formatted text of objective findings — demographics, conditions, meds, vitals each on a new line.',
        ' Use plain text, not nested JSON.",\n'
        '  "assessment": "2-4 sentences: clinical impression, key findings, risk level, differentials.",\n'
        '  "plan": "Numbered steps as a single string, each step on a new line:\\n1. Step one\\n2. Step two\\n3. Step three"\n'
        "}\n\n"
        "Do not wrap values in lists or dicts. Each value must be a plain text string."
    )

    try:
        response = llm.client.chat.completions.create(
            model=LLMHelper.AUX_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=900,
        )
        raw = response.choices[0].message.content or "{}"
        sections = json.loads(raw)
    except Exception as exc:
        print(f"[ClinicalNotes] SOAP generation failed: {exc}")
        sections = {
            "subjective": f"Patient enquired about: {question}. Conversation context: {conversation_summary[:200]}",
            "objective": objective_section,
            "assessment": "Unable to auto-generate assessment — please complete manually.",
            "plan": "Please complete this section manually based on clinical judgement.",
        }

    note_id = f"note-{uuid4().hex[:12]}"
    now = _utc_now()
    display_name = user_profile.get("display_name", username)

    return {
        "note_id": note_id,
        "created_at": now,
        "updated_at": now,
        "username": username,
        "display_name": display_name,
        "trace_id": trace_id or "",
        "question": question[:300],
        "subjective": _coerce_section(sections.get("subjective", "")),
        "objective": _coerce_section(sections.get("objective", objective_section)),
        "assessment": _coerce_section(sections.get("assessment", "")),
        "plan": _coerce_section(sections.get("plan", "")),
        "role_key": role_key,
        "urgency_level": urgency,
        "requires_gp_visit": requires_gp,
        "gp_visit_reason": gp_reason,
        "generated_by": "flynnmed_ai",
        "edited_by": None,
        "email_sent": False,
        "email_sent_at": None,
    }
