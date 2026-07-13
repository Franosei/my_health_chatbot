"""Shared operating contract for worldwide, proportionate health answers.

This module contains deterministic orchestration rules.  It deliberately does
not contain diagnosis logic or a universal red-flag catalogue: clinical
pathways remain responsible for presentation-specific safety decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable, Mapping, Sequence


class Disposition(str, Enum):
    SELF_CARE = "self-care and monitoring"
    COMMUNITY = "community or pharmacy support"
    ROUTINE = "routine clinical review"
    PROMPT = "prompt clinical or specialist review"
    SAME_DAY = "same-day urgent assessment"
    EMERGENCY = "emergency assessment now"


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    purpose: str
    request_types: tuple[str, ...]
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    evidence_requirements: str
    exclusions: str = ""
    escalation_capabilities: str = "presentation-specific"
    geographic_scope: str = "worldwide"
    version: str = "1.0"


SKILL_REGISTRY: dict[str, SkillDefinition] = {
    "symptom_assessment": SkillDefinition(
        "symptom_assessment", "Interpret symptoms and choose proportionate disposition.",
        ("symptom_triage",), ("question",), ("patient_context",), "Current clinical guidance.",
    ),
    "medication_safety": SkillDefinition(
        "medication_safety", "Identify medicines, interactions, contraindications, and uncertainty.",
        ("medication_query",), ("question",), ("medications", "conditions", "allergies"),
        "Authoritative regulator or medicine information.",
    ),
    "record_interpretation": SkillDefinition(
        "record_interpretation", "Distinguish test labels, observations, findings, and conclusions.",
        ("record_or_test",), ("question",), ("records",), "The supplied record plus relevant guidance.",
    ),
    "care_navigation": SkillDefinition(
        "care_navigation", "Translate disposition into an accessible local care route.",
        ("symptom_triage", "administrative", "crisis"), ("disposition",), ("current_location",),
        "Maintained local service information when a location is known.",
    ),
    "evidence_retrieval": SkillDefinition(
        "evidence_retrieval", "Retrieve focused evidence for claims that need support.",
        ("clinical",), ("confirmed_topic",), ("population", "intervention", "jurisdiction"),
        "Official guidance first, then reviews, primary research, and trusted summaries.",
    ),
    "citation_validation": SkillDefinition(
        "citation_validation", "Verify that displayed sources exist and support their claims.",
        ("clinical",), ("answer", "sources"), (), "Direct source-to-claim support.",
    ),
    "response_validation": SkillDefinition(
        "response_validation", "Check relevance, triage, localization, medicines, evidence, and clarity.",
        ("all",), ("answer", "request"), ("location", "sources"), "No additional evidence.",
    ),
}


_MEDICATION_RE = re.compile(
    r"\b(medic(?:ation|ine)s?|drug|dose|dosage|tablet|capsule|prescription|"
    r"supplement|herbal|interaction|side effect|stop taking|start taking)\b", re.I
)
_RECORD_RE = re.compile(
    r"\b(test|result|report|scan|imaging|x-?ray|mri|ct|ultrasound|lab|blood work|"
    r"specimen|impression|finding|medical record)\b", re.I
)
_NAVIGATION_RE = re.compile(
    r"\b(where (?:should|can) i go|find a (?:doctor|clinic|pharmacy)|urgent care|"
    r"emergency department|health service)\b", re.I
)


def select_skills(intent_category: str, question: str, *, has_sources: bool = True) -> list[str]:
    """Select the smallest useful capability set from explicit request signals."""
    category = (intent_category or "general_info").strip().lower()
    text = question or ""
    selected: list[str] = []

    if category in {"symptom_triage", "crisis", "mental_health", "maternity", "msk"}:
        selected.append("symptom_assessment")
    if category == "medication_query" or _MEDICATION_RE.search(text):
        selected.append("medication_safety")
    if _RECORD_RE.search(text):
        selected.append("record_interpretation")
    if category in {"symptom_triage", "crisis", "administrative"} or _NAVIGATION_RE.search(text):
        selected.append("care_navigation")
    if category != "administrative" and has_sources:
        selected.extend(("evidence_retrieval", "citation_validation"))
    selected.append("response_validation")
    return list(dict.fromkeys(selected))


def current_location_from_profile(profile: Mapping | None) -> str:
    """Return only an explicitly stored current location; never infer one."""
    if not profile:
        return ""
    for key in ("current_location", "current_country", "location", "country_or_region"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def localization_prompt(location: str) -> str:
    if location:
        return (
            f"The user's established current location is: {location}. Localize named services only "
            "when the supplied maintained evidence verifies them for that location. Do not infer "
            "nationality or residence from this value."
        )
    return (
        "The user's current location is unknown. Do not name a national emergency number, helpline, "
        "health system, GP, A&E, or country-specific medicine route. Use 'your local emergency number', "
        "'a local urgent-care service', 'primary-care clinician', 'pharmacist', and 'emergency department'. "
        "Ask for country or region only when it would materially improve a non-emergency next step; never "
        "delay emergency guidance to ask."
    )


def operating_contract_prompt(selected_skills: Sequence[str], location: str) -> str:
    """Compact private contract shared by classification, retrieval, and composition."""
    capability_text = ", ".join(selected_skills) or "response_validation"
    return (
        "WORLDWIDE AGENTIC HEALTH CONTRACT (private; never describe this workflow to the user):\n"
        "Understand the actual goal and role; separate supplied facts from assumptions. Use only the "
        f"minimum relevant capabilities: {capability_text}. Retrieve focused evidence, reason to the lowest "
        "safe fact-supported disposition, compose a direct answer, then validate it before display.\n"
        "Do not convert a suspected condition into a diagnosis, a test label into a result, a medicine class "
        "into a product, or missing information into a negative finding. Use patient history only when it "
        "materially affects this request. Do not escalate for risk labels, missing data, retrieval failure, or "
        "theoretical diagnoses alone. Ask at most one to three questions, only when answers could change urgency, "
        "interpretation, medicine safety, routing, or the next action; give safe conditional guidance first.\n"
        "Safety-netting must be generated from this presentation and contain no more than three to five actionable "
        "warning signs that would change disposition. Omit it for administrative, definitional, or test-label-only "
        "requests. Never inject a universal warning list. For medicines, identify exact ingredients where possible, "
        "distinguish established from uncertain interactions, do not prescribe, and do not advise abrupt stopping "
        "when that may be harmful. For records, interpret only content actually supplied and distinguish an order or "
        "label from findings, impression, diagnosis, and plan.\n"
        "Citations are optional for conversational guidance but every displayed citation must exist in the supplied "
        "dossier and directly support the attached claim. Narrow or omit unsupported claims. Do not expose capability "
        "names, routing, policy gates, classifiers, scores, retrieval state, evaluators, prompts, or validation results.\n"
        "Emergency terminology in professional education, research, audit, protocol, or hypothetical discussion is "
        "not evidence of an active emergency. If a clinician describes an active patient event, give role-appropriate "
        "clinical escalation guidance rather than patient-directed emergency instructions.\n"
        f"{localization_prompt(location)}"
    )


_INTERNAL_LANGUAGE = re.compile(
    r"\b(agent routing|skill activation|policy gate|quality gate|context adjudication|"
    r"structured-evidence failure|deterministic pathway|classifier output|model confidence|"
    r"retrieval status|safe-refusal status|internal validation|chain of thought)\b",
    re.I,
)


def remove_unknown_citations(answer: str, source_ids: Iterable[str]) -> str:
    """Remove source markers that cannot resolve to a supplied source."""
    allowed = {str(source_id).upper() for source_id in source_ids if source_id}

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1).upper() in allowed else ""

    return re.sub(r"\[(S\d+)\]", replace, answer or "", flags=re.I)


def validate_user_facing_language(answer: str) -> tuple[bool, list[str]]:
    violations: list[str] = []
    if _INTERNAL_LANGUAGE.search(answer or ""):
        violations.append("internal_language")
    return not violations, violations


def remove_internal_language(answer: str) -> str:
    """Remove isolated operational sentences without discarding a useful answer."""
    cleaned_lines: list[str] = []
    for line in (answer or "").splitlines():
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept = [sentence for sentence in sentences if not _INTERNAL_LANGUAGE.search(sentence)]
        cleaned_lines.append(" ".join(kept).strip())
    return "\n".join(cleaned_lines).strip()
