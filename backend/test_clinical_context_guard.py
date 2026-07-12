from backend.clinical_context_guard import (
    adjudicate_patient_context,
    source_matches_context,
    validate_generated_answer,
)


def urinary_vital():
    return {
        "type": "peak_urinary_flow_rate",
        "value": "18",
        "unit": "mL/s",
        "recorded_on": "2026-07-07",
        "notes": "uroflowmetry report",
    }


def respiratory_vital():
    return {
        "type": "peak_expiratory_flow",
        "value": "420",
        "unit": "L/min",
        "recorded_on": "2026-05-01",
        "notes": "asthma clinic",
    }


def test_single_recorded_reading_is_confirmed_without_asking():
    decision = adjudicate_patient_context(
        question="What does my peak flow mean and how do I manage it?",
        vitals=[urinary_vital()],
    )

    assert decision.status == "confirmed"
    assert decision.domain == "peak_urinary_flow_rate"  # the vital's own type, not a hardcoded specialty name
    assert "Qmax" in decision.topic
    assert decision.requires_clarification is False
    assert decision.blocked_domains == []  # nothing else recorded to block against


def test_no_recorded_vitals_defers_to_general_classifier():
    """With nothing on record to check against, this module must not invent an
    ambiguity from the term alone -- that's the general LLM classifier's job."""
    decision = adjudicate_patient_context(question="What does my peak flow mean?")

    assert decision.status == "insufficient"
    assert decision.requires_clarification is False


def test_two_recorded_readings_require_clarification_then_resolve_from_reply():
    vitals = [urinary_vital(), respiratory_vital()]
    first = adjudicate_patient_context(question="What does my peak flow mean?", vitals=vitals)

    assert first.status == "ambiguous"
    assert first.requires_clarification is True
    assert len(first.clarification_options) == 2

    chosen = first.clarification_options[0]
    resolved = adjudicate_patient_context(
        question="What does my peak flow mean?",
        vitals=vitals,
        chat_summary=f"assistant: {first.clarifying_question}\nuser: {chosen['prompt']}",
    )

    assert resolved.status == "confirmed"
    assert resolved.topic == chosen["display"]
    assert len(resolved.blocked_domains) == 1
    assert resolved.blocked_domains[0] != chosen["display"]


def test_reply_that_does_not_clearly_pick_one_reading_keeps_asking():
    vitals = [urinary_vital(), respiratory_vital()]
    decision = adjudicate_patient_context(
        question="What does my peak flow mean?",
        vitals=vitals,
        chat_summary="user: I'm honestly not sure, can you check for me?",
    )

    assert decision.status == "ambiguous"


def test_generic_synthetic_vital_types_are_handled_without_any_hardcoded_terms():
    """Proves the mechanism isn't hardcoded to peak flow / urology / respiratory --
    two made-up vital types sharing no real-world specialty vocabulary at all
    still trigger the same ambiguity + resolution flow."""
    alpha = {"type": "custom_reading_alpha", "value": "5", "unit": "u1", "recorded_on": "2026-01-01"}
    beta = {"type": "custom_reading_beta", "value": "9", "unit": "u2", "recorded_on": "2026-01-02"}

    first = adjudicate_patient_context(question="What does my custom reading mean?", vitals=[alpha, beta])
    assert first.status == "ambiguous"
    assert len(first.clarification_options) == 2

    chosen = next(o for o in first.clarification_options if "Alpha" in o["display"])
    resolved = adjudicate_patient_context(
        question="What does my custom reading mean?",
        vitals=[alpha, beta],
        chat_summary=f"user: {chosen['prompt']}",
    )
    assert resolved.status == "confirmed"
    assert resolved.domain == "custom_reading_alpha"


def test_post_generation_gate_rejects_the_other_recorded_reading_but_allows_disambiguation_and_negation():
    vitals = [urinary_vital(), respiratory_vital()]
    first = adjudicate_patient_context(question="What does my peak flow mean?", vitals=vitals)
    chosen = first.clarification_options[0]  # urinary, since urinary_vital() is listed first
    decision = adjudicate_patient_context(
        question="What does my peak flow mean?",
        vitals=vitals,
        chat_summary=f"user: {chosen['prompt']}",
    )
    assert decision.status == "confirmed"
    other_label = decision.blocked_domains[0]

    wrong = f"Your {other_label} reading is low, so use an inhaler and breathing exercises."
    negated = f"This is not related to your {other_label} reading."
    correct = "Your urinary flow test result is within the expected range for your age."

    assert validate_generated_answer(wrong, decision)["valid"] is False
    assert validate_generated_answer(negated, decision)["valid"] is True
    assert validate_generated_answer(correct, decision)["valid"] is True
    assert source_matches_context("Asthma peak flow guidance", other_label, decision) is False
