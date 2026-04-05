"""
Role-specific response templates, escalation banners, evidence tier labels,
and clinical safety disclaimers for the product UI.
"""
from __future__ import annotations
from typing import List

from backend.product_config import PRODUCT_NAME


# ── Crisis template ────────────────────────────────────────────────────────────
CRISIS_RESPONSE = f"""\
## ⚠ Urgent Safety Notice

Based on what you have described, this may be an emergency situation.

**Please act immediately:**
- **Call 999** (UK emergency services) or your local emergency number right now
- If in the US, call **911**
- If you are in immediate danger, go to your nearest Emergency Department
- Tell them exactly what you have told me

---

**Crisis support lines (available 24/7):**
- **Samaritans (UK):** 116 123 (free, 24/7)
- **Crisis Text Line (UK):** Text SHOUT to 85258
- **NHS 111:** For urgent medical advice that is not an emergency
- **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/

---

{PRODUCT_NAME} is not able to provide emergency care. Please reach out to a real person right now.
"""

# ── Evidence tier labels ───────────────────────────────────────────────────────
TIER_LABELS = {
    1: "Tier 1 — Formal Guidance",
    2: "Tier 2 — Review Evidence",
    3: "Tier 3 — Primary Research",
}

TIER_DESCRIPTIONS = {
    1: "NHS, NICE, MHRA or equivalent formal clinical guidance",
    2: "Systematic reviews, meta-analyses, or trusted evidence summaries",
    3: "Primary research from PubMed / Europe PMC",
}


def build_tier_badge(tier: int) -> str:
    label = TIER_LABELS.get(tier, f"Tier {tier}")
    return f"[{label}]"


def get_tier_description(tier: int) -> str:
    return TIER_DESCRIPTIONS.get(tier, "Evidence source")


# ── Escalation banners ─────────────────────────────────────────────────────────
def build_escalation_banner(reason: str, role_key: str = "patient") -> str:
    """Returns a prominent escalation notice to prepend to an answer."""
    if role_key in ("doctor", "nurse", "midwife", "physiotherapist"):
        return (
            f"> **Clinical escalation flag:** {reason}\n"
            "> Please apply your clinical judgement and consider immediate review.\n\n"
        )
    return (
        f"> **Important:** {reason}\n"
        "> Please seek appropriate medical help as described below.\n\n"
    )


def build_vulnerability_notice(flags: List[str]) -> str:
    """Returns a safety note for vulnerable population groups."""
    if not flags:
        return ""

    flag_text = ", ".join(flags)
    return (
        f"> **Vulnerable population notice:** This response has been reviewed with heightened "
        f"safety thresholds for: **{flag_text}**. Clinical judgement and appropriate specialist "
        f"referral should be considered.\n\n"
    )


def build_no_diagnosis_disclaimer(role_key: str = "patient") -> str:
    """Returns a role-appropriate no-diagnosis disclaimer."""
    if role_key == "patient":
        return (
            "\n\n---\n"
            f"*{PRODUCT_NAME} provides evidence-based health information and education. "
            "This is not a diagnosis and does not replace advice from your doctor or a qualified clinician. "
            "If you have concerns about your symptoms, please see a healthcare professional.*"
        )
    if role_key in ("doctor", "nurse", "midwife", "physiotherapist"):
        return (
            "\n\n---\n"
            "*This summary is for clinical decision-support only. Evidence confidence and "
            "applicability to individual patients should be verified using your clinical judgement "
            "and current local guidelines.*"
        )
    return (
        "\n\n---\n"
        "*This information is for educational purposes only and is not a substitute for "
        "professional clinical advice.*"
    )


# ── Role-specific section headings ─────────────────────────────────────────────
ROLE_SECTION_HEADINGS: dict[str, List[str]] = {
    "patient": [
        "## What This May Mean",
        "## What To Do Now",
        "## When To Seek Urgent Help",
        "## When To Book Routine Care",
        "## Evidence Basis",
    ],
    "doctor": [
        "## Clinical Summary",
        "## Differential Considerations",
        "## Red Flags",
        "## Evidence Summary",
        "## Uncertainty & Limitations",
    ],
    "nurse": [
        "## Practical Interpretation",
        "## Monitoring Priorities",
        "## Escalation Thresholds",
        "## Patient Education Points",
        "## Evidence Basis",
    ],
    "midwife": [
        "## Clinical Interpretation",
        "## Warning Signs & Red Flags",
        "## Referral Triggers",
        "## Patient Education Points",
        "## Evidence Basis",
    ],
    "physiotherapist": [
        "## Likely MSK Interpretation",
        "## Red Flags & Contraindications",
        "## Initial Management Principles",
        "## Movement Advice",
        "## Referral Triggers",
        "## Evidence Basis",
    ],
    "caregiver": [
        "## What This May Mean",
        "## What To Do Now",
        "## When To Seek Urgent Help",
        "## Caregiver Support Points",
        "## Evidence Basis",
    ],
}

# Fallback headings for unknown roles
DEFAULT_SECTION_HEADINGS = [
    "## Clinical Takeaway",
    "## What This Means In Practice",
    "## Evidence Snapshot",
    "## Recommended Next Step",
]


def get_section_headings(role_key: str) -> List[str]:
    return ROLE_SECTION_HEADINGS.get(role_key, DEFAULT_SECTION_HEADINGS)


def get_section_headings_text(role_key: str) -> str:
    headings = get_section_headings(role_key)
    return "\n".join(headings)


# ── Role-specific persona blocks ───────────────────────────────────────────────
ROLE_PERSONA_BLOCKS: dict[str, str] = {
    "patient": (
        "You are speaking with a patient or individual seeking health information. "
        "Use plain, accessible language — avoid unexplained medical jargon. "
        "Structure your response with clear, actionable guidance on what to do. "
        "Always include when to seek urgent help before detailed educational content. "
        "Use a warm, reassuring, but honest tone."
    ),
    "doctor": (
        "You are supporting a qualified medical doctor. "
        "Use precise clinical terminology. "
        "Present differentials concisely, clearly label evidence quality, and surface "
        "clinical uncertainty explicitly. "
        "Avoid over-explaining basic clinical concepts. "
        "Include relevant NICE/SIGN guideline references where applicable. "
        "Do not overstate the strength of evidence — be explicit about limitations."
    ),
    "nurse": (
        "You are supporting a registered nurse. "
        "Use intermediate clinical language appropriate for nursing practice. "
        "Focus on practical interpretation, monitoring parameters, and escalation thresholds. "
        "Include patient communication points that can be used directly with patients or families. "
        "Reference NICE guidelines and trust protocol considerations where relevant."
    ),
    "midwife": (
        "You are supporting a registered midwife. "
        "Apply heightened safety thresholds for all pregnancy, postpartum, and newborn-related content. "
        "Use maternity-specific clinical terminology. "
        "Always include obstetric red flags, referral triggers, and RCOG or NICE guidelines where relevant. "
        "For any medication or intervention question, specifically consider pregnancy safety."
    ),
    "physiotherapist": (
        "You are supporting a physiotherapist. "
        "Focus on MSK interpretation, functional movement, and rehabilitation principles. "
        "Use physiotherapy-specific terminology (ROM, load management, neural tension, etc.). "
        "Always include neurovascular red flags and non-mechanical warning signs that require "
        "onward referral. Reference NICE MSK guidelines and NICE CKS where relevant."
    ),
    "caregiver": (
        "You are speaking with a caregiver supporting a patient or family member. "
        "Use accessible language with empathetic framing. "
        "Focus on what the caregiver can practically do and when to seek professional help. "
        "Include caregiver-specific support considerations. "
        "Always include clear escalation guidance."
    ),
}

DEFAULT_PERSONA_BLOCK = (
    f"You are {PRODUCT_NAME}, a senior clinical information specialist. "
    "Provide polished, evidence-grounded explanations using the supplied evidence dossier."
)


def get_persona_block(role_key: str) -> str:
    return ROLE_PERSONA_BLOCKS.get(role_key, DEFAULT_PERSONA_BLOCK)
