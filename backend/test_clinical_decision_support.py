from backend.clinical_decision_support import ClinicalDecisionSupportEngine
from backend.intent_risk_classifier import IntentClassification
from backend.role_router import RoleRouter
from backend.triage_summary import normalize_triage_output


def _decision_for(
    question: str,
    presentation_hint: str,
    role: str = "nurse",
    vulnerable_flags: list[str] | None = None,
):
    engine = ClinicalDecisionSupportEngine()
    router = RoleRouter()
    intent = IntentClassification(
        intent_category="symptom_triage",
        risk_level="routine",
        pathway_hint="general_triage",
        presentation_hint=presentation_hint,
        vulnerable_flags=vulnerable_flags or [],
    )
    return engine.assess(question, intent, router.resolve(role))


def test_thunderclap_headache_pathway_is_immediate_review():
    decision = _decision_for(
        "I have had a severe headache that came on suddenly an hour ago and it is the worst headache I have ever had.",
        "thunderclap_headache",
    )

    assert decision.pathway_id == "thunderclap_headache"
    assert decision.next_step == "Immediate review"
    assert decision.minimum_risk_level == "urgent"
    assert any("neurological observations" in item for item in decision.immediate_actions)


def test_possible_sepsis_pathway_adds_elderly_flag_and_news2_action():
    decision = _decision_for(
        "A 68-year-old patient has become increasingly confused over two days, has a temperature of 38.9 and is passing very little urine.",
        "possible_sepsis",
        vulnerable_flags=["elderly"],
    )

    assert decision.pathway_id == "possible_sepsis"
    assert decision.next_step == "Immediate review"
    assert "elderly" in decision.vulnerable_flags
    assert any("NEWS2" in item for item in decision.immediate_actions)


def test_recurrent_blackout_pathway_is_same_day_review():
    decision = _decision_for(
        "A patient tells me they have been having episodes where everything goes black for a few seconds and they nearly fall. This has happened three times in the past two weeks.",
        "recurrent_blackout",
    )

    assert decision.pathway_id == "recurrent_blackout"
    assert decision.next_step == "Same-day review"
    assert any("12-lead ECG" in item for item in decision.immediate_actions)


def test_chronic_cough_pathway_remains_prompt_without_red_flags():
    decision = _decision_for(
        "I have had a persistent cough for eight weeks. I am a non-smoker, I have not lost weight, and I have no night sweats.",
        "chronic_cough_no_red_flags",
    )

    assert decision.pathway_id == "chronic_cough_no_red_flags"
    assert decision.urgency_level == "Prompt"
    assert decision.next_step == "GP"


def test_normalize_triage_output_preserves_immediate_review_step():
    fallback = {
        "urgency_level": "Urgent",
        "next_step": "111",
        "what_to_monitor": ["Deterioration"],
        "rationale": "Urgent review needed.",
    }
    normalized = normalize_triage_output(
        {
            "urgency_level": "Emergency",
            "next_step": "Immediate review",
            "what_to_monitor": ["NEWS2 change"],
            "rationale": "Deterministic pathway selected.",
        },
        fallback,
    )

    assert normalized["next_step"] == "Immediate review"
    assert normalized["what_to_monitor"] == ["NEWS2 change"]
