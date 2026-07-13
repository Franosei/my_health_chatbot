"""Detects a self-stated clinical role from a case's own conversation text.

FlynnMed's role-aware behaviour (`backend/role_router.py`) is driven entirely
by the calling account's stored `clinical_role` profile field -- there is no
code path anywhere in the pipeline that infers role from what the person
actually says in a message. Evaluation cases run through an account with
`user=None` (or an empty/default profile) are therefore always evaluated
under patient-mode defaults (most conservative escalation threshold, lay
terminology), even when the conversation itself makes clear the asker is a
clinician.

This module is purely an evaluation-side concern: it decides which of
FlynnMed's existing account roles (see role_router.py's canonical role keys)
a case *should* be run as, so the harness can exercise the pipeline the same
way a real account with that role would. It does not change, patch, or
otherwise touch how FlynnMed itself classifies or responds -- it only
changes which pre-existing, unmodified role configuration the evaluation
routes a case through.

Detection is deliberately conservative (fixed, generic self-identification
phrasing, not per-case content) to avoid false positives -- e.g. a patient
saying "my doctor told me..." must not be misdetected as the patient being a
doctor.
"""

from __future__ import annotations

import re

from evaluations.models import EvalCase

# Mirrors role_router.py's canonical role keys exactly (patient, caregiver,
# doctor, nurse, midwife, physiotherapist) -- deliberately not importing from
# backend/ to keep this module fully independent of pipeline internals; the
# values themselves are just the account role strings FlynnMed already knows
# how to route (see backend/role_router.py's _ALIAS_MAP).
_ROLE_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    (
        "doctor",
        re.compile(
            r"\bI(?:'m| am) an?\s+(?:[a-z]+\s+){0,3}?"
            r"(?:physician|doctor|GP|general practitioner|surgeon|MD)\b",
            re.I,
        ),
    ),
    (
        "nurse",
        re.compile(r"\bI(?:'m| am) an?\s+(?:[a-z]+\s+){0,3}?nurse\b", re.I),
    ),
    (
        "midwife",
        re.compile(r"\bI(?:'m| am) an?\s+(?:[a-z]+\s+){0,3}?midwife\b", re.I),
    ),
    (
        "physiotherapist",
        re.compile(
            r"\bI(?:'m| am) an?\s+(?:[a-z]+\s+){0,3}?(?:physiotherapist|physical therapist)\b",
            re.I,
        ),
    ),
]


_CURLY_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})


def detect_stated_role(case: EvalCase) -> str:
    """Returns the canonical role key implied by the case's own conversation,
    or "patient" (FlynnMed's default) if no clear clinical self-identification
    is present. Only scans user turns -- a role claim inside an assistant/
    reference turn shouldn't count.

    Real HealthBench text commonly uses a curly apostrophe ("I’m a doctor")
    rather than a straight one -- normalised here so every pattern only has
    to be written once, instead of duplicating each with both quote styles.
    """
    text = " ".join(turn.content for turn in case.conversation if turn.role == "user")
    text = text.translate(_CURLY_APOSTROPHES)
    for role_key, pattern in _ROLE_PATTERNS:
        if pattern.search(text):
            return role_key
    return "patient"


def eval_account_username(role: str, case_id: str) -> str:
    """Deterministic, clearly-namespaced, per-case username so eval accounts
    can never collide with (or be mistaken for) a real user, and each case
    gets a fresh, never-reused account -- no cross-case history contamination
    is possible since nothing is ever read back into a second case."""
    safe_case_id = re.sub(r"[^a-zA-Z0-9_-]", "", case_id)[:32]
    return f"eval-harness-{role}-{safe_case_id}"
