import backend.user_store as user_store_module
from backend.gp_summary import build_gp_summary_pdf
from backend.medication_checker import MedicationInteractionChecker
from backend.symptom_tracker import build_symptom_pattern_summary
from backend.triage_summary import normalize_triage_output
from backend.user_store import UserStore


def _configure_temp_user_store(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    upload_root = data_dir / "uploads"
    user_db_path = tmp_path / "users.json"
    monkeypatch.setattr(user_store_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(user_store_module, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(user_store_module, "USER_DB_PATH", user_db_path)
    monkeypatch.setattr(user_store_module, "_USER_BACKEND", None)


def test_symptom_pattern_summary_highlights_recurrence_and_triggers():
    summary = build_symptom_pattern_summary(
        [
            {
                "symptom": "Headache",
                "logged_for": "2026-04-08",
                "severity": 4,
                "triggers": "stress",
                "notes": "",
            },
            {
                "symptom": "Headache",
                "logged_for": "2026-04-10",
                "severity": 7,
                "triggers": "stress, poor sleep",
                "notes": "worse after work",
            },
        ]
    )

    assert "Headache logged 2 time(s)" in summary
    assert "common triggers: stress, poor sleep" in summary
    assert "trend worsening" in summary


def test_normalize_triage_output_never_lowers_safe_fallback():
    fallback = {
        "urgency_level": "Urgent",
        "next_step": "111",
        "what_to_monitor": ["Breathing difficulty"],
        "rationale": "Urgent review needed.",
    }
    normalized = normalize_triage_output(
        {
            "urgency_level": "Routine",
            "next_step": "Self-care",
            "what_to_monitor": ["Hydration"],
            "rationale": "Model tried to de-escalate.",
        },
        fallback,
    )

    assert normalized["next_step"] == "111"
    assert normalized["what_to_monitor"] == ["Hydration"]


def test_medication_checker_flags_label_warning_without_network():
    checker = MedicationInteractionChecker()
    left = {
        "canonical_name": "Warfarin",
        "aliases": ["warfarin"],
        "sections": [
            {
                "label": "Drug interactions",
                "text": "Avoid concomitant ibuprofen use because of serious bleeding risk.",
            }
        ],
        "api_url": "https://api.fda.gov/example",
        "effective_time": "20260410",
    }
    right = {
        "canonical_name": "Ibuprofen",
        "aliases": ["ibuprofen"],
        "sections": [],
        "api_url": "https://api.fda.gov/example",
        "effective_time": "20260410",
    }

    alert = checker._build_pair_alert(left, right)

    assert alert is not None
    assert alert["severity"] == "high"
    assert "Warfarin + Ibuprofen" == alert["pair"]


def test_gp_summary_pdf_is_created():
    pdf_bytes = build_gp_summary_pdf(
        user_profile={"display_name": "Alex Patient"},
        symptom_logs=[
            {
                "symptom": "Headache",
                "logged_for": "2026-04-10",
                "severity": 6,
                "triggers": "stress",
                "notes": "Evening flare",
            }
        ],
        medications=[{"name": "Warfarin", "dose": "5 mg", "schedule": "daily", "reason": "AF"}],
        uploads=[{"file": "discharge-letter.pdf"}],
        longitudinal_memory="Patient Summary:\nAtrial fibrillation.\nRecent symptoms or active concerns:\nHeadaches.",
        latest_triage={
            "urgency_level": "Prompt",
            "next_step": "GP",
            "what_to_monitor": ["worsening headache"],
            "rationale": "Needs review if persistent.",
        },
    )

    assert pdf_bytes.startswith(b"%PDF")


def test_user_store_round_trip_for_new_health_records(tmp_path, monkeypatch):
    _configure_temp_user_store(tmp_path, monkeypatch)
    created = UserStore.create_user(
        username="tester",
        password="verysecure1",
        display_name="Tester",
        email="tester@example.com",
    )
    assert created

    symptom = UserStore.add_symptom_log(
        "tester",
        symptom="Cough",
        logged_for="2026-04-11",
        severity=5,
        triggers="cold air",
        notes="Dry cough",
    )
    medication = UserStore.save_medication(
        "tester",
        {"name": "Ibuprofen", "dose": "200 mg", "schedule": "as needed"},
    )
    triage = UserStore.save_triage_summary(
        "tester",
        {
            "question": "My cough is getting worse",
            "urgency_level": "Prompt",
            "next_step": "GP",
            "what_to_monitor": ["breathlessness"],
            "rationale": "Persistent symptoms",
        },
    )

    assert symptom is not None
    assert medication is not None
    assert triage is not None
    assert UserStore.get_symptom_logs("tester")[0]["symptom"] == "Cough"
    assert UserStore.get_medications("tester")[0]["name"] == "Ibuprofen"
    assert UserStore.get_latest_triage_summary("tester")["next_step"] == "GP"
