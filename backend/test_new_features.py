import backend.user_store as user_store_module
import fitz
from backend.gp_summary import build_gp_summary_pdf, build_summary_pdf
from backend.medication_checker import MedicationInteractionChecker
from backend.symptom_tracker import build_symptom_pattern_summary
from backend.triage_summary import normalize_triage_output
from backend.user_store import UserStore


def _configure_temp_user_store(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    upload_root = data_dir / "uploads"
    user_db_path = tmp_path / "users.json"
    # This helper specifically exercises the legacy JSON store. CI provides a
    # DATABASE_URL for the separate SQL integration tests, so remove it here
    # before resetting the lazily selected backend.
    monkeypatch.delenv("DATABASE_URL", raising=False)
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


def test_medication_checker_excerpt_marks_continuation_instead_of_hard_cut():
    checker = MedicationInteractionChecker()
    text = (
        "Introductory text. Lithium should be used with caution with lisinopril and hydrochlorothiazide. "
        "Frequent monitoring of serum potassium and renal function is required because toxicity may increase "
        "when sodium balance changes during treatment. Additional explanatory wording continues for a long time "
        "to simulate label content that would otherwise be cut off abruptly in the UI."
    )

    start = text.index("lisinopril")
    end = start + len("lisinopril")
    excerpt = checker._extract_excerpt(text, start, end, window=220, max_chars=140)

    assert "lisinopril and hydrochlorothiazide" in excerpt.lower()
    assert excerpt.endswith("...") or excerpt.endswith(".")


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
        triage_summaries=[{
            "urgency_level": "Prompt",
            "next_step": "GP",
            "what_to_monitor": ["worsening headache"],
            "rationale": "Needs review if persistent.",
            "created_at": "2025-01-01T10:00:00Z",
        }],
    )

    assert pdf_bytes.startswith(b"%PDF")


def test_health_summary_uses_latest_data_before_previous_history():
    pdf_bytes = build_summary_pdf(
        user_profile={"display_name": "Alex Patient"},
        symptom_logs=[
            {"symptom": "Chest tightness", "logged_for": "2026-05-20", "severity": 7},
            {"symptom": "Cough", "logged_for": "2026-04-01", "severity": 3},
        ],
        medications=[{"name": "Amlodipine", "dose": "5 mg", "schedule": "daily"}],
        uploads=[{"file": "blood-results.pdf"}],
        longitudinal_memory=(
            "Patient Summary:\n"
            "Known hypertension.\n"
            "Conditions and history:\n"
            "Previous asthma diagnosis.\n"
        ),
        role_key="Doctor / Physician",
        triage_summaries=[{
            "urgency_level": "Prompt",
            "next_step": "Book GP review within 2 working days",
            "created_at": "2026-05-21T10:00:00Z",
        }],
        conditions=[
            {"name": "Hypertension", "status": "active", "recorded_on": "2025-01-01"},
            {"name": "Asthma", "status": "past", "recorded_on": "2018-01-01"},
        ],
        vitals=[
            {"type": "blood_pressure", "value": "138/86", "unit": "mmHg", "recorded_on": "2026-05-21"},
            {"type": "blood_pressure", "value": "150/95", "unit": "mmHg", "recorded_on": "2026-04-01"},
        ],
    )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)

    assert text.index("Current Clinical Snapshot") < text.index("Relevant Past Medical History")
    assert "Active problem: Hypertension" in text
    assert "BP 138/86 mmHg" in text
    assert "Previous: 150/95 mmHg" in text
    assert "Past medical history: Asthma" in text


def test_patient_summary_uses_plain_language_sections():
    pdf_bytes = build_summary_pdf(
        user_profile={"display_name": "Alex Patient"},
        symptom_logs=[{"symptom": "Headache", "logged_for": "2026-05-21", "severity": 5}],
        medications=[],
        uploads=[],
        longitudinal_memory="Patient Summary:\nMigraine history.",
        role_key="Patient / Individual",
        conditions=[{"name": "Migraine", "status": "active", "recorded_on": "2024-02-01"}],
        vitals=[],
    )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)

    assert "Current Health Snapshot" in text
    assert "Past Health History" in text
    assert "Current Clinical Snapshot" not in text
    assert "Current health issue: Migraine" in text


def test_patient_summary_filters_prompt_repetition_and_source_noise():
    pdf_bytes = build_summary_pdf(
        user_profile={"display_name": "Kwabena Gyawu"},
        symptom_logs=[],
        medications=[],
        uploads=[{"file": "Patient Name_ Francis Osei.pdf"}],
        longitudinal_memory=(
            "Patient Summary:\n"
            "Age: 29 years\n"
            "Biological sex: Male\n"
            "Display Name: Kwabena Gyawu\n"
            "Role: Patient / Individual\n"
            "History of frequent sore throat and severe fever three years ago.\n"
            "Recent symptoms or active concerns:\n"
            "Experiencing sore throat for some time. Difficulty swallowing liquids but can eat solid food without pain.\n"
            "What symptoms would make chest pain an urgent medical review issue?\n"
        ),
        role_key="Patient / Individual",
        triage_summaries=[{
            "question": "What symptoms would make chest pain an urgent medical review issue?",
            "urgency_level": "Prompt",
            "next_step": "GP",
            "pathway_label": "General triage",
            "decision_summary": "No specific high-acuity presentation matched; using computed acuity floor and clinical judgement.",
            "what_to_monitor": ["Persistence, progression, or new red-flag symptoms"],
            "immediate_actions": ["Arrange clinician review and safety-net for deterioration."],
            "escalation_triggers": ["Any new severe symptom, collapse, or rapid deterioration"],
            "created_at": "2026-05-30T10:00:00Z",
        }],
        recent_chats=[
            {
                "role": "user",
                "content": "What symptoms would make chest pain an urgent medical review issue?",
                "timestamp": "2026-05-30T09:59:00Z",
            }
        ],
        conditions=[
            {
                "name": "Acute kidney injury",
                "status": "active",
                "recorded_on": "2026-05-26",
                "notes": "Acute kidney injury pattern [Auto-extracted from Patient Name_ Francis Osei.pdf]",
            }
        ],
        vitals=[
            {"type": "potassium", "value": "6.8", "unit": "mmol/L", "recorded_on": "2026-05-26"},
            {"type": "whitebloodcells", "value": "24.9", "unit": "x10^9/L", "recorded_on": "2026-05-26"},
        ],
    )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)

    assert "Current health issue: Acute kidney injury" in text
    assert "Experiencing sore throat for some time" in text
    assert text.count("Experiencing sore throat for some time") == 1
    assert "Suggested next step: GP" in text
    assert "What symptoms would make chest pain" not in text
    assert "Display Name" not in text
    assert "Biological sex" not in text
    assert "Auto-extracted" not in text
    assert "Patient Name_ Francis" not in text
    assert "Records Used" not in text


def test_gp_summary_uses_medications_mentioned_in_longitudinal_memory():
    pdf_bytes = build_gp_summary_pdf(
        user_profile={"display_name": "Case Study"},
        symptom_logs=[],
        medications=[],
        uploads=[],
        longitudinal_memory=(
            "Patient Summary:\n"
            "Atrial fibrillation.\n"
            "Current treatments and medicines:\n"
            "On warfarin for atrial fibrillation. Recently prescribed ibuprofen for knee pain.\n"
            "Recent symptoms or active concerns:\n"
            "Increasing confusion and reduced urine output.\n"
        ),
        triage_summaries=[],
    )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)

    assert "Medications" in text
    assert "warfarin" in text.lower()
    assert "ibuprofen" in text.lower()


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
