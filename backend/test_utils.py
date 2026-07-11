from backend.utils import render_vital_for_prompt, vital_display_label


def test_peak_urinary_flow_rate_and_peak_expiratory_flow_render_distinctly():
    urinary = vital_display_label("peak_urinary_flow_rate")
    respiratory = vital_display_label("peak_expiratory_flow")

    assert urinary != respiratory
    assert "urology" in urinary.lower()
    assert "respiratory" in respiratory.lower()
    assert "urology" not in respiratory.lower()


def test_bare_peak_flow_key_is_flagged_as_ambiguous_not_defaulted_to_respiratory():
    label = vital_display_label("peak_flow")

    assert "ambiguous" in label.lower()
    assert "respiratory" not in label.lower() or "verify" in label.lower()


def test_render_vital_for_prompt_disambiguates_the_reported_bug_scenario():
    urinary_entry = {
        "type": "peak_urinary_flow_rate",
        "value": "18",
        "unit": "ml/s",
        "recorded_on": "2026-07-07",
    }
    rendered = render_vital_for_prompt(urinary_entry)

    assert "18" in rendered
    assert "ml/s" in rendered
    assert "urology" in rendered.lower()
    # The label explicitly rules out the respiratory reading (the bug that was reported) --
    # "respiratory" appearing as part of "NOT a respiratory measurement" is the point.
    assert "not a respiratory measurement" in rendered.lower()


def test_render_vital_for_prompt_date_prefix_variants():
    entry = {"type": "blood_pressure", "value": "120/80", "unit": "mmHg", "recorded_on": "2026-01-01"}

    assert render_vital_for_prompt(entry) == "Blood Pressure: 120/80 mmHg (2026-01-01)"
    assert render_vital_for_prompt(entry, date_prefix="recorded ") == "Blood Pressure: 120/80 mmHg (recorded 2026-01-01)"
    assert render_vital_for_prompt(entry, include_date=False) == "Blood Pressure: 120/80 mmHg"


def test_unambiguous_vital_falls_back_to_title_case():
    assert vital_display_label("blood_pressure") == "Blood Pressure"
