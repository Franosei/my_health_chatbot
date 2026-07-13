"""Evidence-first patient-context adjudication.

This module is deliberately small and deterministic. It does not diagnose a
patient and it does not provide treatment recommendations. Its job is to
answer a narrower safety question before any model, search tool, or UI is
allowed to act:

    "What does this patient record actually establish, and is the requested
     topic compatible with that evidence?"

The distinction matters for measurements whose names are reused across
specialties. A bare ``peak flow`` must never silently become respiratory
peak-expiratory-flow when the record contains a urology Qmax measurement.

This works generically for ANY pair of similarly-named vital types recorded
for a patient -- there is no fixed list of specialty names or per-term
keyword vocabulary here. A vital only ever becomes a "candidate" for the
current request through plain word overlap with its own recorded ``type``
string, so the exact same logic that resolves peak-flow ambiguity also
resolves ambiguity for any other vital type this app comes to support,
without code changes. When the patient's own record has nothing to check
against, this module defers entirely to the general LLM-based ambiguity
classifier (``IntentRiskClassifier``) rather than guessing from a hardcoded
term list.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from backend.utils import render_vital_for_prompt, vital_display_label

_WORD_RE = re.compile(r"[a-z0-9]+")
_LAST_USER_TURN_RE = re.compile(r"(?:^|\n)\s*(?:user|patient)\s*:\s*(.+)", re.I)
_NEGATION_RE = re.compile(r"\b(?:not|no|without|rather than|instead of|does not|isn't|isn t)\b", re.I)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        value = _text(value)
        if value and value.lower() not in seen:
            seen.add(value.lower())
            output.append(value)
    return output


def _last_user_clarification(chat_summary: str) -> str:
    """Only the patient's own most recent reply may resolve an open
    ambiguity -- earlier turns (which may include a previous, possibly wrong,
    assistant answer or unrelated messages) must not leak in."""
    matches = _LAST_USER_TURN_RE.findall(_text(chat_summary))
    return matches[-1].strip() if matches else ""


def _vital_signature(entry: Dict) -> str:
    return f"{_text(entry.get('type')).lower()}|{_text(entry.get('unit')).lower()}"


def _find_candidates(request_tokens: set, vitals: Optional[List[Dict]]) -> Dict[str, Dict]:
    """A vital is a candidate for the current request purely by token overlap
    with its own recorded ``type`` string -- no specialty vocabulary. Requires
    at least two shared significant words (or full containment for a
    one/two-word type), so an unrelated vital sharing one common word doesn't
    falsely match."""
    candidates: Dict[str, Dict] = {}
    for entry in vitals or []:
        vtype = _text(entry.get("type"))
        if not vtype:
            continue
        vtype_tokens = _tokens(vtype)
        if not vtype_tokens:
            continue
        shared = request_tokens & vtype_tokens
        is_match = len(shared) >= 2 or (len(vtype_tokens) <= 2 and vtype_tokens.issubset(request_tokens))
        if is_match:
            candidates.setdefault(_vital_signature(entry), entry)
    return candidates


def _sentence_is_negated(sentence: str, term_tokens: set) -> bool:
    lower = sentence.lower()
    positions = [lower.find(tok) for tok in term_tokens if tok and tok in lower]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return False
    before = lower[: min(positions)]
    return bool(_NEGATION_RE.search(before))


@dataclass
class ClinicalContextDecision:
    """A machine-readable decision shared by chat, plans, trials, and MCP."""

    status: str = "insufficient"  # confirmed | ambiguous | insufficient
    topic: str = ""
    domain: str = ""  # the matched vital's own recorded `type` string, e.g. "peak_urinary_flow_rate"
    confidence: float = 0.0
    requested_topic: str = ""
    direct_facts: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    query_terms: List[str] = field(default_factory=list)
    clarifying_question: str = ""
    clarification_options: List[Dict[str, str]] = field(default_factory=list)

    @property
    def requires_clarification(self) -> bool:
        return self.status == "ambiguous"

    @property
    def safe_for_generation(self) -> bool:
        return self.status == "confirmed"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "topic": self.topic,
            "domain": self.domain,
            "confidence": self.confidence,
            "requested_topic": self.requested_topic,
            "direct_facts": list(self.direct_facts),
            "reasons": list(self.reasons),
            "blocked_domains": list(self.blocked_domains),
            "query_terms": list(self.query_terms),
            "requires_clarification": self.requires_clarification,
        }

    def as_prompt_block(self) -> str:
        if self.status == "insufficient":
            return (
                "No record-specific interpretation is established for this request. Answer the "
                "current question normally. Do not claim that a record was reviewed, and mention "
                "missing details only when they are necessary to answer safely."
            )

        lines = [
            "Clinical context adjudication (binding safety decision):",
            f"- Status: {self.status}",
            f"- Confirmed topic: {self.topic or 'not confirmed'}",
            f"- Confidence: {self.confidence:.2f}",
        ]
        if self.direct_facts:
            lines.append("- Direct record facts: " + "; ".join(self.direct_facts[:8]))
        if self.reasons:
            lines.append("- Why: " + "; ".join(self.reasons[:5]))
        if self.blocked_domains:
            lines.append("- Do not describe this as: " + ", ".join(self.blocked_domains))
        if self.requires_clarification:
            lines.append(
                "- Generation rule: pause and ask the clarification question; do not produce "
                "condition-specific advice or trial matches."
            )
        else:
            lines.append(
                "- Generation rule: use only this confirmed meaning. A reused or bare term "
                "must not be expanded into another recorded reading."
            )
        return "\n".join(lines)

    def correction_message(self) -> str:
        """Safe fallback used if a downstream model still violates the decision."""
        if self.topic:
            return (
                "## Correct interpretation\n\n"
                f"Your record identifies this as **{self.topic}**. "
                "I have stopped the response because the requested interpretation does not "
                "match the reading recorded in your history.\n\n"
                "Please ask the clinician or service who ordered the test to explain what the "
                "result means in the context of your symptoms and the full report. If you meant "
                "a different test, provide its report title and unit."
            )
        return (
            "## I need to check the result first\n\n"
            "The information available uses a term that can refer to different clinical tests. "
            "I will not guess which one you mean. Please confirm the test name, unit, and the "
            "clinic or report section that produced it."
        )


def decision_from_dict(data: Optional[Dict[str, Any]]) -> Optional[ClinicalContextDecision]:
    """Rehydrate the serialisable form used in trial results and MCP calls."""
    if not isinstance(data, dict) or not data:
        return None
    return ClinicalContextDecision(
        status=_text(data.get("status")) or "insufficient",
        topic=_text(data.get("topic")),
        domain=_text(data.get("domain")),
        confidence=float(data.get("confidence", 0) or 0),
        requested_topic=_text(data.get("requested_topic")),
        direct_facts=[_text(value) for value in data.get("direct_facts", []) if _text(value)],
        reasons=[_text(value) for value in data.get("reasons", []) if _text(value)],
        blocked_domains=[_text(value) for value in data.get("blocked_domains", []) if _text(value)],
        query_terms=[_text(value) for value in data.get("query_terms", []) if _text(value)],
    )


def adjudicate_patient_context(
    *,
    question: str = "",
    requested_topic: str = "",
    conditions: Optional[List[Dict]] = None,
    medications: Optional[List[Dict]] = None,
    vitals: Optional[List[Dict]] = None,
    allergies: Optional[List[Dict]] = None,
    triage_summaries: Optional[List[Dict]] = None,
    document_summaries: Optional[List[Dict]] = None,
    longitudinal_memory: str = "",
    chat_summary: str = "",
) -> ClinicalContextDecision:
    """Resolve cross-specialty measurement ambiguity using only the patient's
    own structured vitals -- no hardcoded specialty names or per-term keyword
    lists. A vital becomes a "candidate" purely via token overlap with its own
    recorded ``type`` string (see ``_find_candidates``), so this generalises to
    any vital type without code changes.

    ``conditions``/``medications``/``allergies``/``triage_summaries``/
    ``document_summaries``/``longitudinal_memory`` are accepted for call-site
    compatibility but are not the primary signal here -- structured vitals are
    the only deterministic, data-grounded source of "what test does this
    patient's record actually establish."
    """
    requested = _text(requested_topic)
    latest_reply = _last_user_clarification(chat_summary)
    request_tokens = _tokens(f"{_text(question)} {requested} {latest_reply}")

    if not request_tokens:
        return ClinicalContextDecision(status="insufficient", requested_topic=requested, topic=requested)

    candidates = _find_candidates(request_tokens, vitals)

    if not candidates:
        return ClinicalContextDecision(
            status="insufficient",
            requested_topic=requested,
            topic=requested,
            reasons=["No recorded vital matches the terms in this request; deferring to general classification."],
        )

    if len(candidates) == 1:
        entry = next(iter(candidates.values()))
        vtype = _text(entry.get("type"))
        return ClinicalContextDecision(
            status="confirmed",
            topic=vital_display_label(vtype),
            domain=vtype,
            confidence=0.9,
            requested_topic=requested,
            direct_facts=[render_vital_for_prompt(entry)],
            reasons=["Exactly one recorded reading on file matches this request."],
            query_terms=_unique([vital_display_label(vtype), vtype.replace("_", " ")]),
        )

    # 2+ distinct recorded readings match. Try to resolve using the patient's
    # own latest reply, scored purely by token overlap against each
    # candidate's own type/label -- no fixed vocabulary.
    if latest_reply:
        reply_tokens = _tokens(latest_reply)
        scored = []
        for sig, entry in candidates.items():
            vtype = _text(entry.get("type"))
            # Score against the vital's own bare type words only -- the
            # human-readable label can carry incidental prose (e.g. a "not a
            # respiratory measurement" disclaimer) whose common words would
            # otherwise cause false matches against an unrelated reply.
            scored.append((len(reply_tokens & _tokens(vtype)), sig, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            _, chosen_sig, chosen_entry = scored[0]
            vtype = _text(chosen_entry.get("type"))
            blocked = [
                vital_display_label(_text(other.get("type")))
                for sig, other in candidates.items()
                if sig != chosen_sig
            ]
            return ClinicalContextDecision(
                status="confirmed",
                topic=vital_display_label(vtype),
                domain=vtype,
                confidence=0.9,
                requested_topic=requested,
                direct_facts=[render_vital_for_prompt(chosen_entry)],
                reasons=["Resolved from the patient's own reply to the earlier clarification question."],
                blocked_domains=blocked,
                query_terms=_unique([vital_display_label(vtype), vtype.replace("_", " ")]),
            )

    facts = [render_vital_for_prompt(entry) for entry in candidates.values()]
    options = [
        {
            "display": vital_display_label(_text(entry.get("type"))),
            "prompt": (
                f"This refers to my {vital_display_label(_text(entry.get('type')))} reading "
                f"({render_vital_for_prompt(entry)})."
            ),
        }
        for entry in candidates.values()
    ]
    return ClinicalContextDecision(
        status="ambiguous",
        topic=requested or _text(question),
        confidence=0.0,
        requested_topic=requested,
        direct_facts=facts,
        reasons=[f"{len(candidates)} different recorded readings on file match this request."],
        clarifying_question="Your records include more than one reading that matches this. Which one does this refer to?",
        clarification_options=options,
    )


def incompatible_terms(text: str, decision: Optional[ClinicalContextDecision]) -> List[str]:
    """Return the confirmed decision's blocked labels (other recorded readings
    that were not the one confirmed) mentioned in generated/source text,
    ignoring negated mentions (e.g. "not a ... reading"). Only meaningful when
    there was an actual competing reading on the patient's own record
    (``decision.blocked_domains``) -- there is no external specialty
    vocabulary to check against otherwise."""
    if not text or not decision or not decision.blocked_domains:
        return []
    hits: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence_tokens = _tokens(sentence)
        for label in decision.blocked_domains:
            label_tokens = _tokens(label)
            if not label_tokens:
                continue
            overlap = label_tokens & sentence_tokens
            if len(overlap) >= max(1, len(label_tokens) - 1) and not _sentence_is_negated(sentence, label_tokens):
                hits.append(label)
    return _unique(hits)


def validate_generated_answer(answer: str, decision: Optional[ClinicalContextDecision]) -> Dict[str, Any]:
    violations = incompatible_terms(answer, decision)
    return {"valid": not violations, "violations": violations}


def source_matches_context(title: str, content: str, decision: Optional[ClinicalContextDecision]) -> bool:
    return not incompatible_terms(f"{title}. {content}", decision)


def build_review_required_plan(decision: ClinicalContextDecision) -> Dict[str, Any]:
    """Return a non-prescriptive plan when generation fails the context gate."""
    now = datetime.now(timezone.utc).isoformat()
    topic = decision.topic or decision.requested_topic or "the requested health concern"
    return {
        "id": uuid.uuid4().hex,
        "condition": topic,
        "title": f"Review required: {topic}",
        "status": "paused",
        "created_at": now,
        "updated_at": now,
        "goals": [{
            "id": uuid.uuid4().hex[:12],
            "text": "Confirm the test meaning and next steps with the clinician who ordered it.",
            "metric": "Clinician-confirmed interpretation",
        }],
        "daily_tasks": [],
        "weekly_tasks": [],
        "medication_reminders": [],
        "lab_reminders": [],
        "escalation_thresholds": [],
        "lifestyle": {},
        "missed_care_checklist": [],
        "evidence_summary": "No condition-specific plan was issued because the patient context did not support a safe interpretation.",
        "safety_notes": (
            "This plan is paused because the requested topic conflicts with the confirmed "
            "patient record. Confirm the test name, unit, and next step with the clinician "
            "who ordered it before starting condition-specific tasks."
        ),
        "clinical_context": decision.as_dict(),
        "validation": {"status": "review_required", "violations": []},
        "after_visit_notes": [],
        "gp_prep_summary": None,
    }


def validate_care_plan(plan: Dict[str, Any], decision: Optional[ClinicalContextDecision]) -> Dict[str, Any]:
    if not decision or not decision.blocked_domains:
        return {"valid": True, "violations": []}
    text_parts: List[str] = []
    for key in ("condition", "title", "evidence_summary", "safety_notes"):
        text_parts.append(_text(plan.get(key)))
    for key in ("goals", "daily_tasks", "weekly_tasks", "medication_reminders", "lab_reminders", "escalation_thresholds", "lifestyle"):
        text_parts.append(str(plan.get(key, "")))
    violations = incompatible_terms(" ".join(text_parts), decision)
    return {"valid": not violations, "violations": violations}
