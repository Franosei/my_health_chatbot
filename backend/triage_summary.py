from __future__ import annotations

from typing import Dict


TRIAGE_STEPS = {"self-care", "gp", "111", "999"}
TRIAGE_STEP_RANK = {"self-care": 1, "gp": 2, "111": 3, "999": 4}


def build_default_triage(intent, policy_decision) -> Dict:
    risk_level = (getattr(intent, "risk_level", "") or "routine").lower()
    escalation_reason = (getattr(intent, "escalation_reason", "") or "").strip()
    policy_action = (getattr(policy_decision, "action", "") or "allow").lower()

    if risk_level == "crisis" or getattr(intent, "crisis_detected", False):
        return {
            "urgency_level": "Emergency",
            "next_step": "999",
            "what_to_monitor": [
                "Any further deterioration while waiting for emergency help",
            ],
            "rationale": escalation_reason or "Emergency symptom pattern detected.",
        }

    if risk_level == "urgent":
        return {
            "urgency_level": "Urgent",
            "next_step": "111",
            "what_to_monitor": [
                "Worsening pain, breathing difficulty, new neurological symptoms, or reduced responsiveness",
            ],
            "rationale": escalation_reason or "Urgent symptom review is warranted.",
        }

    if risk_level == "elevated" or policy_action != "allow":
        return {
            "urgency_level": "Prompt",
            "next_step": "GP",
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
    merged["next_step"] = (
        "Self-care"
        if normalized_next_step == "self-care"
        else normalized_next_step.upper()
        if normalized_next_step.isdigit()
        else "GP"
    )

    monitor = merged.get("what_to_monitor", [])
    if not isinstance(monitor, list):
        monitor = []
    cleaned_monitor = [str(item).strip() for item in monitor if str(item).strip()]
    merged["what_to_monitor"] = cleaned_monitor or fallback.get("what_to_monitor", [])

    merged["urgency_level"] = str(merged.get("urgency_level") or fallback.get("urgency_level") or "").strip()
    merged["rationale"] = str(merged.get("rationale") or fallback.get("rationale") or "").strip()
    return merged
