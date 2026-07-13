from backend.agentic_health_contract import (
    current_location_from_profile,
    localization_prompt,
    remove_internal_language,
    remove_unknown_citations,
    select_skills,
    validate_user_facing_language,
)


def test_select_skills_uses_minimum_relevant_capabilities():
    selected = select_skills("medication_query", "Can I take this supplement with warfarin?")

    assert "medication_safety" in selected
    assert "symptom_assessment" not in selected
    assert selected[-1] == "response_validation"


def test_test_label_routes_to_record_interpretation_without_symptom_assessment():
    selected = select_skills("general_info", "My record says MRI brain. What does that label mean?")

    assert "record_interpretation" in selected
    assert "symptom_assessment" not in selected


def test_location_is_only_taken_from_explicit_current_profile_fields():
    assert current_location_from_profile({"nationality": "British"}) == ""
    assert current_location_from_profile({"previous_country": "UK"}) == ""
    assert current_location_from_profile({"current_location": "France"}) == "France"


def test_unknown_location_prompt_has_no_country_default():
    prompt = localization_prompt("")

    assert "local emergency number" in prompt
    assert "999" not in prompt
    assert "911" not in prompt
    assert "NHS" not in prompt


def test_unknown_citation_markers_are_removed_before_display():
    answer = "Supported [S1]. Unknown [S7]."

    assert remove_unknown_citations(answer, ["S1", "S2"]) == "Supported [S1]. Unknown ."


def test_internal_operational_language_is_rejected():
    valid, violations = validate_user_facing_language("The policy gate rejected this answer.")

    assert valid is False
    assert violations == ["internal_language"]


def test_internal_sentence_is_removed_without_discarding_useful_answer():
    answer = "Your earache needs review. The policy gate rejected one source. Use simple pain relief meanwhile."

    assert remove_internal_language(answer) == (
        "Your earache needs review. Use simple pain relief meanwhile."
    )
