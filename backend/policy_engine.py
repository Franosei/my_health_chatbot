"""
Clinical policy engine: rule-based gate that applies hard safety constraints,
escalation requirements, and vulnerable-population logic.
Pure Python — no LLM calls.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from backend.audit_models import PolicyGateRecord
from backend.intent_risk_classifier import IntentClassification
from backend.role_router import RoleConfig
from backend.response_templates import (
    CRISIS_RESPONSE,
    build_escalation_banner,
    build_vulnerability_notice,
    build_no_diagnosis_disclaimer,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Medication risk patterns ───────────────────────────────────────────────────
_MED_DOSAGE_PATIENT_PATTERN = re.compile(
    r"\b(how much|what dose|dosage|how many (mg|milligrams?|tablets?|pills?))\b",
    re.IGNORECASE,
)
_PREGNANCY_MED_PATTERN = re.compile(
    r"(safe (in|during|for) pregnan|can i take .{0,30} pregnant|"
    r"(medication|drug|tablet) .{0,30} pregnan)",
    re.IGNORECASE,
)
_DIAGNOSIS_SEEKING_PATTERN = re.compile(
    r"\b(do i have|is it|could it be|am i|is this|could this be|"
    r"diagnos|what (?:is|are) (?:wrong|my condition))\b",
    re.IGNORECASE,
)


@dataclass
class PolicyDecision:
    action: str = "allow"                        # "allow" | "escalate_only" | "block"
    gates_applied: List[PolicyGateRecord] = field(default_factory=list)
    context_notes: List[str] = field(default_factory=list)   # injected into LLM context
    escalation_banner: str = ""                  # prepended to answer if non-empty
    vulnerability_notice: str = ""               # appended near top of answer
    disclaimer: str = ""                         # appended at bottom of answer
    crisis_response: str = ""                    # returned verbatim without LLM if non-empty

    def add_gate(self, gate: PolicyGateRecord) -> None:
        self.gates_applied.append(gate)

    def gates_as_dicts(self) -> List[dict]:
        return [g.as_dict() for g in self.gates_applied]


class PolicyEngine:
    """
    Applies hard clinical safety gates based on intent classification and role.
    Call gate() once per request, before LLM generation.
    """

    def gate(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        question: str,
    ) -> PolicyDecision:
        decision = PolicyDecision()

        # Accumulate vulnerability flags from both intent and role config
        all_flags = list(set(intent.vulnerable_flags + role_config.vulnerable_population_flags))

        # Apply gates in priority order
        self._gate_crisis(intent, role_config, decision)
        if decision.action == "escalate_only":
            return decision

        self._gate_urgent_escalation(intent, role_config, decision)
        self._gate_pregnancy(intent, role_config, question, decision)
        self._gate_paediatric(intent, all_flags, decision)
        self._gate_medication_dosage(intent, role_config, question, decision)
        self._gate_diagnosis_request(intent, role_config, question, decision)
        self._gate_elderly_polypharmacy(intent, all_flags, decision)
        self._gate_mental_health(intent, role_config, decision)

        # Build vulnerability notice if applicable
        if all_flags:
            decision.vulnerability_notice = build_vulnerability_notice(all_flags)

        # Always attach a no-diagnosis disclaimer
        decision.disclaimer = build_no_diagnosis_disclaimer(role_config.role_key)

        return decision

    # ── Individual gates ────────────────────────────────────────────────────────

    def _gate_crisis(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        decision: PolicyDecision,
    ) -> None:
        if not intent.crisis_detected:
            return

        gate = PolicyGateRecord(
            gate_name="crisis",
            applied=True,
            reason="Crisis-level risk detected — returning emergency guidance without LLM generation.",
        )
        decision.add_gate(gate)
        decision.action = "escalate_only"
        decision.crisis_response = CRISIS_RESPONSE

    def _gate_urgent_escalation(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        decision: PolicyDecision,
    ) -> None:
        if intent.risk_level not in ("urgent", "crisis"):
            return
        if role_config.escalation_threshold == "high":
            # Clinical roles — add context note but don't force banner
            gate = PolicyGateRecord(
                gate_name="urgent_clinical",
                applied=True,
                reason=f"Urgent intent detected for clinical user ({role_config.role_key}).",
            )
            decision.add_gate(gate)
            decision.context_notes.append(
                "POLICY NOTE: This question has been classified as urgent. "
                "Prioritise red flag information and immediate action guidance in your response."
            )
        else:
            # Non-clinical roles — force escalation banner
            reason = intent.escalation_reason or "Urgent clinical concern detected."
            gate = PolicyGateRecord(
                gate_name="urgent_escalation",
                applied=True,
                reason=reason,
            )
            decision.add_gate(gate)
            decision.escalation_banner = build_escalation_banner(reason, role_config.role_key)
            decision.context_notes.append(
                f"POLICY NOTE: Urgent risk level. Always lead with escalation guidance. "
                f"Reason: {reason}"
            )

    def _gate_pregnancy(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        question: str,
        decision: PolicyDecision,
    ) -> None:
        has_pregnancy_flag = "pregnancy" in (intent.vulnerable_flags + role_config.vulnerable_population_flags)
        has_pregnancy_med = bool(_PREGNANCY_MED_PATTERN.search(question))

        if not (has_pregnancy_flag or has_pregnancy_med or intent.intent_category == "maternity"):
            return

        gate = PolicyGateRecord(
            gate_name="pregnancy_safety",
            applied=True,
            reason="Pregnancy context detected — applying heightened medication and escalation safety.",
        )
        decision.add_gate(gate)
        decision.context_notes.append(
            "POLICY NOTE: Pregnancy context is present. "
            "Apply heightened caution for all medication, dosage, and intervention advice. "
            "Reference NICE/RCOG guidelines specifically. "
            "Never recommend stopping or starting prescription medication without explicit NICE guidance. "
            "Always include obstetric red flags where relevant."
        )
        if role_config.role_key not in ("midwife", "doctor"):
            decision.escalation_banner = build_escalation_banner(
                "Pregnancy-related question — always verify medication safety with your midwife or GP.",
                role_config.role_key,
            )

    def _gate_paediatric(
        self,
        intent: IntentClassification,
        all_flags: List[str],
        decision: PolicyDecision,
    ) -> None:
        if "paediatric" not in all_flags:
            return

        gate = PolicyGateRecord(
            gate_name="paediatric_safety",
            applied=True,
            reason="Paediatric population flag — applying child-specific safety thresholds.",
        )
        decision.add_gate(gate)
        decision.context_notes.append(
            "POLICY NOTE: Paediatric context. "
            "All dosing, weight-based recommendations, and growth milestones must be explicitly age-qualified. "
            "Never extrapolate adult guidance to children without stating it. "
            "Refer to BNFC and NICE paediatric pathways."
        )

    def _gate_medication_dosage(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        question: str,
        decision: PolicyDecision,
    ) -> None:
        if intent.intent_category != "medication_query":
            if not _MED_DOSAGE_PATIENT_PATTERN.search(question):
                return

        if role_config.role_key in ("doctor", "nurse", "midwife", "physiotherapist"):
            # Clinical roles get BNF context note
            gate = PolicyGateRecord(
                gate_name="medication_clinical",
                applied=True,
                reason="Medication query for clinical user — BNF/NICE reference context added.",
            )
            decision.add_gate(gate)
            decision.context_notes.append(
                "POLICY NOTE: Medication query for a clinician. "
                "Reference BNF, NICE, or MHRA guidance. "
                "Show evidence uncertainty for off-label or non-guideline uses explicitly."
            )
        else:
            # Lay roles get pharmacist/GP referral note
            gate = PolicyGateRecord(
                gate_name="medication_lay",
                applied=True,
                reason="Medication dosage question for lay user — pharmacist/GP referral required.",
            )
            decision.add_gate(gate)
            decision.context_notes.append(
                "POLICY NOTE: Medication question from a patient or caregiver. "
                "Do not provide specific dosage advice for prescription medicines. "
                "Always recommend verification with a pharmacist or GP. "
                "Use BNF/NICE as the source basis."
            )

    def _gate_diagnosis_request(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        question: str,
        decision: PolicyDecision,
    ) -> None:
        is_diagnosis_seeking = bool(_DIAGNOSIS_SEEKING_PATTERN.search(question))
        if not is_diagnosis_seeking:
            return
        if role_config.role_key in ("doctor",):
            # Doctor receives differential discussion context
            gate = PolicyGateRecord(
                gate_name="diagnosis_clinical",
                applied=True,
                reason="Diagnostic question for clinician — differential discussion permitted with uncertainty labelling.",
            )
            decision.add_gate(gate)
            decision.context_notes.append(
                "POLICY NOTE: Diagnostic question from a clinician. "
                "Differential discussion is appropriate but must be clearly framed as hypothesis, "
                "not definitive diagnosis. Label evidence quality explicitly."
            )
        else:
            gate = PolicyGateRecord(
                gate_name="no_diagnosis",
                applied=True,
                reason="Diagnosis-seeking language from non-clinician — no-diagnosis policy enforced.",
            )
            decision.add_gate(gate)
            decision.context_notes.append(
                "POLICY NOTE: This user appears to be seeking a diagnosis. "
                "You must NOT provide a definitive diagnosis. "
                "Explain possible causes and what the symptoms may suggest, "
                "but always direct them to see a clinician for diagnosis."
            )

    def _gate_elderly_polypharmacy(
        self,
        intent: IntentClassification,
        all_flags: List[str],
        decision: PolicyDecision,
    ) -> None:
        if "elderly" not in all_flags:
            return
        if intent.intent_category != "medication_query":
            return

        gate = PolicyGateRecord(
            gate_name="elderly_polypharmacy",
            applied=True,
            reason="Elderly patient + medication query — polypharmacy and renal function considerations added.",
        )
        decision.add_gate(gate)
        decision.context_notes.append(
            "POLICY NOTE: Elderly population + medication context. "
            "Highlight polypharmacy risks, renal/hepatic dosing adjustments, and fall risk. "
            "Reference STOPP/START criteria and NICE guidance for older people."
        )

    def _gate_mental_health(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        decision: PolicyDecision,
    ) -> None:
        if intent.intent_category != "mental_health":
            return

        gate = PolicyGateRecord(
            gate_name="mental_health",
            applied=True,
            reason="Mental health topic detected — crisis resources and empathetic framing required.",
        )
        decision.add_gate(gate)
        decision.context_notes.append(
            "POLICY NOTE: Mental health topic. "
            "Always include crisis support resources (Samaritans 116 123, Crisis Text Line). "
            "Use empathetic, non-stigmatising language. "
            "Never minimise distress. "
            "If any self-harm or suicidal ideation is implied, apply crisis response."
        )
