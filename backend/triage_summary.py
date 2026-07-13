from __future__ import annotations

from typing import Dict


TRIAGE_STEPS = {
    "self-care", "gp", "primary-care clinician", "local urgent-care service",
    "111", "same-day review", "immediate review", "999", "local emergency services",
}
TRIAGE_STEP_RANK = {
    "self-care": 1,
    "gp": 2,
    "same-day review": 3,
    "111": 3,
    "immediate review": 4,
    "999": 5,
    "primary-care clinician": 2,
    "local urgent-care service": 3,
    "local emergency services": 5,
}


def build_default_triage(intent, policy_decision) -> Dict:
    risk_level = (getattr(intent, "risk_level", "") or "routine").lower()
    escalation_reason = (getattr(intent, "escalation_reason", "") or "").strip()
    policy_action = (getattr(policy_decision, "action", "") or "allow").lower()

    if risk_level == "crisis" or getattr(intent, "crisis_detected", False):
        return {
            "urgency_level": "Emergency",
            "next_step": "local emergency services",
            "what_to_monitor": [
                "Any further deterioration while waiting for emergency help",
            ],
            "rationale": escalation_reason or "Emergency symptom pattern detected.",
        }

    if risk_level == "urgent":
        return {
            "urgency_level": "Urgent",
            "next_step": "local urgent-care service",
            "what_to_monitor": [
                "Worsening pain, breathing difficulty, new neurological symptoms, or reduced responsiveness",
            ],
            "rationale": escalation_reason or "Urgent symptom review is warranted.",
        }

    if risk_level == "elevated" or policy_action != "allow":
        return {
            "urgency_level": "Prompt",
            "next_step": "Primary-care clinician",
            "what_to_monitor": [
                "Persistent symptoms, worsening severity, or new red flags",
            ],
            "rationale": escalation_reason or "A clinician review is appropriate.",
        }

    return {
        "urgency_level": "Routine",
        "next_step": "Self-care",
        "what_to_monitor": [
            "Whether symptoms settle, worsen, or new warning signs appear",
        ],
        "rationale": "Current question appears suitable for self-care guidance unless symptoms change.",
    }


def normalize_triage_output(payload: Dict, fallback: Dict) -> Dict:
    merged = dict(fallback)
    merged.update({key: value for key, value in (payload or {}).items() if value})

    next_step = str(merged.get("next_step") or fallback.get("next_step") or "").strip()
    normalized_next_step = next_step.lower()
    if normalized_next_step not in TRIAGE_STEPS:
        normalized_next_step = str(fallback.get("next_step") or "Self-care").lower()
    fallback_step = str(fallback.get("next_step") or "Self-care").strip().lower()
    if TRIAGE_STEP_RANK.get(normalized_next_step, 0) < TRIAGE_STEP_RANK.get(fallback_step, 1):
        normalized_next_step = fallback_step
    merged["next_step"] = {
        "self-care": "Self-care",
        "gp": "GP",
        "111": "111",
        "same-day review": "Same-day review",
        "immediate review": "Immediate review",
        "999": "999",
        "primary-care clinician": "Primary-care clinician",
        "local urgent-care service": "Local urgent-care service",
        "local emergency services": "Local emergency services",
    }.get(normalized_next_step, "GP")

    monitor = merged.get("what_to_monitor", [])
    if not isinstance(monitor, list):
        monitor = []
    cleaned_monitor = [str(item).strip() for item in monitor if str(item).strip()]
    merged["what_to_monitor"] = cleaned_monitor or fallback.get("what_to_monitor", [])

    for field in ("immediate_actions", "escalation_triggers", "communication_points"):
        items = merged.get(field, [])
        if not isinstance(items, list):
            items = []
        merged[field] = [str(item).strip() for item in items if str(item).strip()]

    merged["urgency_level"] = str(merged.get("urgency_level") or fallback.get("urgency_level") or "").strip()
    merged["rationale"] = str(merged.get("rationale") or fallback.get("rationale") or "").strip()
    return merged
