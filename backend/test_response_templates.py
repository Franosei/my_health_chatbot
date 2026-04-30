from backend.response_templates import (
    DEFAULT_PERSONA_BLOCK,
    get_persona_block,
    get_section_headings,
)


def test_default_persona_is_safe_competent_and_not_senior():
    assert "safe and competent clinical information assistant" in DEFAULT_PERSONA_BLOCK
    assert "senior clinical information specialist" not in DEFAULT_PERSONA_BLOCK


def test_doctor_persona_pushes_management_without_senior_claim():
    persona = get_persona_block("doctor").lower()

    assert "initial management" in persona
    assert "not a senior specialist" in persona
    assert "clear route" in persona


def test_clinician_headings_lead_with_management_sections():
    assert get_section_headings("doctor")[:4] == [
        "## Working Impression",
        "## Immediate Management",
        "## Investigations / Monitoring",
        "## Escalate Now If",
    ]
    assert get_section_headings("nurse")[1] == "## Immediate Nursing Actions"


def test_patient_headings_include_monitoring_and_urgent_route():
    headings = get_section_headings("patient")

    assert "## What To Monitor" in headings
    assert "## Get Urgent Help If" in headings
