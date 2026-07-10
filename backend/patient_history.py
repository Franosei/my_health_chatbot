"""
PatientHistoryContext: extracts and structures a patient's known medical history
for use in clinical decision-making (intent classification + policy gating).

This allows the system to raise risk level and apply safety gates when the current
question -- even if it seems routine in isolation -- is clinically significant given
the patient's known conditions. E.g. a headache in a hypertensive patient, bleeding
in a patient on warfarin, or a cough in an immunocompromised patient.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.user_store import compute_current_age

# Urgency ranking for comparing previous visit severity
URGENCY_RANK: Dict[str, int] = {"routine": 0, "elevated": 1, "urgent": 2, "crisis": 3}

# ── Condition-group patterns (scanned across memory + medication text) ─────────

_CARDIAC = re.compile(
    r"\b(heart (disease|failure|attack|condition|problem)|cardiac|coronary|angina|"
    r"myocardial (infarction|ischaemia)?|arrhythmia|atrial.?fibrillation|\bAF\b|"
    r"pacemaker|ischaemic heart|cardiomyopathy|heart valve)\b",
    re.IGNORECASE,
)
_HYPERTENSION = re.compile(
    r"\b(hypertension|high blood pressure|hypertensive)\b", re.IGNORECASE
)
_DIABETES = re.compile(
    r"\b(diabetes|diabetic|type [12] diabetes|T[12]DM|insulin|"
    r"hyperglycaemia|hypoglycaemia|diabetic (neuropathy|retinopathy|nephropathy))\b",
    re.IGNORECASE,
)
_ANTICOAGULANT = re.compile(
    r"\b(warfarin|rivaroxaban|apixaban|edoxaban|dabigatran|"
    r"heparin|anticoagulant|blood thinner|DOAC)\b",
    re.IGNORECASE,
)
_IMMUNOCOMPROMISED = re.compile(
    r"\b(immunocompromised|immunosuppressed|chemotherapy|transplant patient|"
    r"long.?term steroids|biologics|HIV|AIDS|neutropenia)\b",
    re.IGNORECASE,
)
_RENAL = re.compile(
    r"\b(chronic kidney (disease|failure)|CKD|renal (failure|impairment|disease)|"
    r"dialysis|low eGFR|stage [3-5] (CKD|kidney))\b",
    re.IGNORECASE,
)
_RESPIRATORY = re.compile(
    r"\b(asthma|COPD|emphysema|chronic bronchitis|pulmonary fibrosis|"
    r"bronchiectasis|respiratory (disease|condition|failure))\b",
    re.IGNORECASE,
)
_MENTAL_HEALTH = re.compile(
    r"\b(depression|anxiety disorder|bipolar|schizophrenia|PTSD|"
    r"eating disorder|previous (self.harm|suicide attempt|overdose))\b",
    re.IGNORECASE,
)
_STROKE = re.compile(
    r"\b(stroke|TIA|transient ischaemic attack|cerebrovascular|"
    r"previous stroke|mini.stroke)\b",
    re.IGNORECASE,
)
_LIVER = re.compile(
    r"\b(liver (disease|failure|cirrhosis)|hepatic|hepatitis [BC]|"
    r"chronic liver|alcohol.related liver)\b",
    re.IGNORECASE,
)


@dataclass
class PatientHistoryContext:
    # Demographics (from user profile)
    age: Optional[int] = None
    biological_sex: str = ""

    known_conditions: List[str] = field(default_factory=list)
    known_medications: List[str] = field(default_factory=list)
    known_allergies: List[str] = field(default_factory=list)
    recent_vitals: List[str] = field(default_factory=list)
    highest_previous_urgency: str = "routine"
    previous_escalation_count: int = 0
    history_vulnerable_flags: List[str] = field(default_factory=list)

    # Condition-group flags used by the policy gate
    has_cardiac_history: bool = False
    has_hypertension: bool = False
    has_diabetes: bool = False
    on_anticoagulants: bool = False
    immunocompromised: bool = False
    has_renal_disease: bool = False
    has_respiratory_disease: bool = False
    has_mental_health_history: bool = False
    has_stroke_history: bool = False
    has_liver_disease: bool = False

    def is_empty(self) -> bool:
        return (
            self.age is None
            and not self.biological_sex
            and not self.known_conditions
            and not self.known_medications
            and not self.known_allergies
            and not self.recent_vitals
            and self.highest_previous_urgency == "routine"
            and self.previous_escalation_count == 0
        )

    def as_prompt_block(self) -> str:
        """Compact text block injected into the classifier LLM prompt."""
        if self.is_empty():
            return ""
        parts: List[str] = []
        # Demographics first -- most clinically impactful for classification
        if self.age is not None:
            parts.append(f"Patient age: {self.age} years")
        if self.biological_sex:
            parts.append(f"Biological sex: {self.biological_sex}")
        if self.known_conditions:
            parts.append("Known conditions: " + "; ".join(self.known_conditions[:8]))
        if self.known_medications:
            parts.append("Current medications: " + ", ".join(self.known_medications[:8]))
        if self.known_allergies:
            parts.append("Known allergies / contraindications: " + "; ".join(self.known_allergies[:10]))
        if self.recent_vitals:
            parts.append("Recent vitals: " + "; ".join(self.recent_vitals[:6]))
        if self.highest_previous_urgency not in ("", "routine"):
            parts.append(f"Highest previous urgency level: {self.highest_previous_urgency}")
        if self.previous_escalation_count > 0:
            parts.append(f"Previous escalations: {self.previous_escalation_count}")
        if self.history_vulnerable_flags:
            parts.append("Previously flagged: " + ", ".join(self.history_vulnerable_flags))
        return "\n".join(parts)


def build_patient_history_context(
    longitudinal_memory: str,
    medications: List[Dict],
    triage_summaries: List[Dict],
    user_profile: Optional[Dict] = None,
    allergies: Optional[List[Dict]] = None,
    conditions: Optional[List[Dict]] = None,
    vitals: Optional[List[Dict]] = None,
    max_triages: int = 10,
) -> PatientHistoryContext:
    ctx = PatientHistoryContext()

    # ── Demographics from user profile ──────────────────────────────────────────
    if user_profile:
        dob = (user_profile.get("date_of_birth") or "").strip()
        ctx.age = compute_current_age(dob)
        sex = (user_profile.get("biological_sex") or "").strip()
        if sex and sex != "Prefer not to say":
            ctx.biological_sex = sex

    # ── Current medications ──────────────────────────────────────────────────────
    for med in (medications or []):
        name = (med.get("name") or "").strip()
        if name:
            ctx.known_medications.append(name)

    # ── Known allergies ──────────────────────────────────────────────────────────
    for allergy in (allergies or []):
        name = (allergy.get("name") or "").strip()
        if not name:
            continue
        reaction = (allergy.get("reaction") or "").strip()
        severity = (allergy.get("severity") or "").strip()
        label = name
        if reaction:
            label += f" ({reaction})"
        if severity and severity != "unknown":
            label += f" [{severity}]"
        ctx.known_allergies.append(label)

    # ── Recent vitals ────────────────────────────────────────────────────────────
    for entry in (vitals or [])[:8]:
        vtype = (entry.get("type") or "").strip()
        value = (entry.get("value") or "").strip()
        unit = (entry.get("unit") or "").strip()
        recorded_on = (entry.get("recorded_on") or "").strip()
        if vtype and value:
            label = f"{vtype}: {value}{' ' + unit if unit else ''}"
            if recorded_on:
                label += f" ({recorded_on})"
            ctx.recent_vitals.append(label)

    # ── Conditions from longitudinal memory ─────────────────────────────────────
    condition_names = []
    for condition in conditions or []:
        name = (condition.get("name") or "").strip()
        if not name:
            continue
        status = (condition.get("status") or "").strip()
        label = name
        if status and status != "unknown":
            label += f" ({status})"
        condition_names.append(label)
    ctx.known_conditions = _unique_nonempty(
        condition_names + _extract_condition_lines(longitudinal_memory or "")
    )

    # ── Scan combined text for condition-group flags ─────────────────────────────
    combined = (
        (longitudinal_memory or "")
        + " "
        + " ".join(ctx.known_conditions)
        + " "
        + " ".join(ctx.known_medications)
    )
    ctx.has_cardiac_history    = bool(_CARDIAC.search(combined))
    ctx.has_hypertension       = bool(_HYPERTENSION.search(combined))
    ctx.has_diabetes           = bool(_DIABETES.search(combined))
    ctx.on_anticoagulants      = bool(_ANTICOAGULANT.search(combined))
    ctx.immunocompromised      = bool(_IMMUNOCOMPROMISED.search(combined))
    ctx.has_renal_disease      = bool(_RENAL.search(combined))
    ctx.has_respiratory_disease = bool(_RESPIRATORY.search(combined))
    ctx.has_mental_health_history = bool(_MENTAL_HEALTH.search(combined))
    ctx.has_stroke_history     = bool(_STROKE.search(combined))
    ctx.has_liver_disease      = bool(_LIVER.search(combined))

    # ── Previous triage urgency + escalation count ───────────────────────────────
    escalation_urgencies = {"urgent", "crisis"}
    highest_rank = 0

    for t in (triage_summaries or [])[:max_triages]:
        urgency = (t.get("urgency_level") or "routine").lower().strip()
        rank = URGENCY_RANK.get(urgency, 0)
        if rank > highest_rank:
            highest_rank = rank
            ctx.highest_previous_urgency = urgency
        if urgency in escalation_urgencies:
            ctx.previous_escalation_count += 1
        for flag in (t.get("vulnerable_flags") or []):
            flag = str(flag).strip()
            if flag and flag not in ctx.history_vulnerable_flags:
                ctx.history_vulnerable_flags.append(flag)

    return ctx


def _unique_nonempty(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _extract_condition_lines(memory_text: str) -> List[str]:
    """Pull condition-related lines from structured longitudinal memory sections."""
    target_sections = {
        "patient summary",
        "conditions and history",
        "recent symptoms or active concerns",
        "current treatments and medicines",
    }
    lines: List[str] = []
    in_target = False

    for raw in memory_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith(":"):
            in_target = line[:-1].strip().lower() in target_sections
            continue
        if in_target and len(line) > 5:
            lines.append(line)
            if len(lines) >= 10:
                break

    return lines
