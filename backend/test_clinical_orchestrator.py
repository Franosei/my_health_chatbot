from types import SimpleNamespace

from backend.clinical_orchestrator import AgenticRetrievalLoop, ClinicalOrchestrator
from backend.intent_risk_classifier import IntentClassification


class _FakeMemory:
    def search(self, query, user):
        return []

    def add_entries(self, entries):
        pass


class _FakePubMed:
    def search_article_records(self, query, n):
        return []


class _FakeOfficialGuidance:
    def search(self, queries, top_k, preferred=None):
        return []


class _FakeQueryExpander:
    def expand(self, question):
        return []

    def expand_with_patient_context(self, question, history_context):
        return []


class _FakeModeration:
    def decide(self, question, role_key=None):
        return False, "", "", {}


_AMBIGUOUS_INTENT = IntentClassification(
    intent_category="general_info",
    risk_level="routine",
    pathway_hint="general_triage",
    ambiguous_term_detected=True,
    ambiguous_term="peak flow",
    ambiguity_clarifying_question="Was your peak flow measured with a breathing device or during a urine flow test?",
    ambiguity_reply_options=[
        {
            "display": "Breathing test",
            "prompt": "My peak flow was measured with a breathing/asthma peak flow meter -- what does my reading mean?",
        },
        {
            "display": "Urine flow test",
            "prompt": "My peak flow was measured during a urology urine flow test (uroflowmetry) -- what does my reading mean?",
        },
    ],
)


def _build_orchestrator(monkeypatch) -> ClinicalOrchestrator:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return ClinicalOrchestrator(
        memory=_FakeMemory(),
        pubmed=_FakePubMed(),
        official_guidance=_FakeOfficialGuidance(),
        llm=object(),
        query_expander=_FakeQueryExpander(),
        moderation=_FakeModeration(),
    )


def test_ambiguous_routine_question_short_circuits_before_retrieval(monkeypatch):
    orchestrator = _build_orchestrator(monkeypatch)
    orchestrator.intent_classifier.classify = lambda *a, **kw: _AMBIGUOUS_INTENT

    def _fail_if_called(self, *a, **kw):
        raise AssertionError(
            "AgenticRetrievalLoop.run should not be called when ambiguity gate fires"
        )

    monkeypatch.setattr(AgenticRetrievalLoop, "run", _fail_if_called)

    bundle = orchestrator.prepare_bundle(
        question="What is my peak flow level and what does it mean?",
        user="patient1",
        user_profile={},
        longitudinal_memory_summary="",
    )

    assert bundle["kind"] == "final"
    payload = bundle["payload"]
    assert payload["follow_up_questions"] == _AMBIGUOUS_INTENT.ambiguity_reply_options
    assert payload["trace"]["retrieval_mode"] == "clarification_requested"
    assert payload["trace"]["ambiguous_term"] == "peak flow"
    assert "Was your peak flow measured" in payload["answer_markdown"]


def test_ambiguity_flag_is_ignored_when_risk_level_is_urgent(monkeypatch):
    orchestrator = _build_orchestrator(monkeypatch)
    urgent_intent = IntentClassification(
        intent_category="symptom_triage",
        risk_level="urgent",
        pathway_hint="general_triage",
        ambiguous_term_detected=True,
        ambiguous_term="peak flow",
        ambiguity_clarifying_question="Was your peak flow measured with a breathing device or during a urine flow test?",
        ambiguity_reply_options=[
            {"display": "Breathing test", "prompt": "breathing prompt"},
            {"display": "Urine flow test", "prompt": "urine prompt"},
        ],
    )
    orchestrator.intent_classifier.classify = lambda *a, **kw: urgent_intent

    def _empty_run(self, *a, **kw):
        return {
            "collected_sources": [],
            "personal_context": [],
            "trial_results": [],
            "tool_calls_made": [],
        }

    monkeypatch.setattr(AgenticRetrievalLoop, "run", _empty_run)

    bundle = orchestrator.prepare_bundle(
        question="What is my peak flow level and what does it mean?",
        user="patient1",
        user_profile={},
        longitudinal_memory_summary="",
    )

    assert bundle["kind"] == "answer"


def test_doctor_acls_education_bypasses_crisis_short_circuit(monkeypatch):
    orchestrator = _build_orchestrator(monkeypatch)
    routine_clinical_intent = IntentClassification(
        intent_category="general_info",
        risk_level="routine",
        pathway_hint="general_triage",
    )
    orchestrator.intent_classifier.classify = lambda *a, **kw: routine_clinical_intent

    monkeypatch.setattr(
        AgenticRetrievalLoop,
        "run",
        lambda self, *a, **kw: {
            "collected_sources": [],
            "personal_context": [],
            "trial_results": [],
            "tool_calls_made": [],
        },
    )

    bundle = orchestrator.prepare_bundle(
        question=(
            "I'm an emergency medicine physician. Walk me through the new BLS and ACLS "
            "guideline updates for adult in-hospital cardiac arrest, including airway research."
        ),
        user="doctor1",
        user_profile={"clinical_role": "doctor"},
        longitudinal_memory_summary="",
    )

    assert bundle["kind"] == "answer"
    assert bundle["role_config"].role_key == "doctor"
    assert bundle["intent"].crisis_detected is False


def test_documentation_mode_short_circuits_clinical_classification_and_retrieval(
    monkeypatch,
):
    orchestrator = _build_orchestrator(monkeypatch)

    def _fail(*args, **kwargs):
        raise AssertionError(
            "clinical classification/retrieval must not run for documentation"
        )

    orchestrator.intent_classifier.classify = _fail
    monkeypatch.setattr(AgenticRetrievalLoop, "run", _fail)

    bundle = orchestrator.prepare_bundle(
        question=(
            "Please draft a SOAP note using only these facts: mild fatigue, stable vitals, "
            "pedal oedema, and LVEF 45%."
        ),
        user="patient1",
        user_profile={},
        longitudinal_memory_summary="",
    )

    assert bundle["kind"] == "answer"
    assert bundle["role_config"].role_key == "patient"
    assert bundle["task_mode"].mode == "documentation"
    assert bundle["combined_sources"] == []
    assert bundle["retrieval_mode"] == "controlled_transformation"
    assert bundle["policy_decision"].action == "allow"


def test_translation_continuation_short_circuits_retrieval(monkeypatch):
    orchestrator = _build_orchestrator(monkeypatch)

    def _fail(*args, **kwargs):
        raise AssertionError("retrieval must not run for a translation continuation")

    orchestrator.intent_classifier.classify = _fail
    monkeypatch.setattr(AgenticRetrievalLoop, "run", _fail)
    history = [
        {
            "role": "user",
            "content": "Preciso que traduza os textos para português de Portugal.",
        },
        {"role": "assistant", "content": "Claro."},
    ]

    bundle = orchestrator.prepare_bundle(
        question="Finally, compile the major systematic reviews.",
        user="patient1",
        user_profile={},
        longitudinal_memory_summary="",
        chat_history=history,
    )

    assert bundle["task_mode"].mode == "translation"
    assert bundle["role_config"].role_key == "patient"
    assert bundle["combined_sources"] == []


class _FakeAgentCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="DONE", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")]
        )


def test_exclude_mismatched_sources_strips_source_and_recomputes_report():
    """
    evidence_ranker has no concept of cross-specialty term mismatch (e.g. respiratory
    vs. urology "peak flow") -- only the evidence dossier's per-article LLM extraction
    catches that. This reconciliation must strip a dossier-flagged mismatch out of
    combined_sources (so it can't be cited or shown in the Sources panel) and out of
    evidence_quality_report's counts (so the "use ... for general context" text stops
    covering it).
    """
    combined_sources = [
        {
            "source_id": "S1",
            "title": "Respiratory peak flow guidance",
            "evidence_quality_status": "question_aligned",
            "evidence_quality_score": 0.4,
        },
        {
            "source_id": "S2",
            "title": "Uroflowmetry guidance",
            "evidence_quality_status": "patient_aligned",
            "evidence_quality_score": 0.8,
        },
    ]
    evidence_quality_report = {
        "overall_status": "patient_aligned_evidence_available",
        "accepted_source_count": 2,
        "excluded_source_count": 1,
        "status_counts": {"question_aligned": 1, "patient_aligned": 1},
        "excluded_sources": [
            {"title": "Some other excluded source", "reasons": ["stale"]}
        ],
    }

    kept, report = ClinicalOrchestrator._exclude_mismatched_sources(
        combined_sources, evidence_quality_report, ["S1"]
    )

    assert [s["source_id"] for s in kept] == ["S2"]
    assert report["accepted_source_count"] == 1
    assert report["excluded_source_count"] == 2
    assert report["status_counts"] == {"patient_aligned": 1}
    assert report["overall_status"] == "patient_aligned_evidence_available"
    titles = [s["title"] for s in report["excluded_sources"]]
    assert "Respiratory peak flow guidance" in titles
    assert "Some other excluded source" in titles


def test_exclude_mismatched_sources_no_op_when_nothing_flagged():
    combined_sources = [
        {"source_id": "S1", "title": "x", "evidence_quality_status": "patient_aligned"}
    ]
    report = {
        "accepted_source_count": 1,
        "excluded_source_count": 0,
        "status_counts": {"patient_aligned": 1},
    }

    kept, out_report = ClinicalOrchestrator._exclude_mismatched_sources(
        combined_sources, report, []
    )

    assert kept is combined_sources
    assert out_report is report


def test_agentic_retrieval_loop_prompt_instructs_using_confirmed_meaning():
    fake_llm = SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeAgentCompletions())
        ),
        AUX_MODEL="gpt-4o-mini",
    )
    loop = AgenticRetrievalLoop(
        llm=fake_llm,
        official_guidance=object(),
        pubmed=object(),
        memory=object(),
        user="patient1",
    )

    loop.run(
        question="What does my peak flow of 18 mean?",
        patient_summary="Peak urinary flow rate / Qmax (urology, NOT a respiratory measurement): 18 ml/s",
        role_key="patient",
        pathway_hint="general_triage",
    )

    sent_prompt = fake_llm.client.chat.completions.calls[0]["messages"][0]["content"]
    assert "confirmed meaning" in sent_prompt.lower()
    assert "raw ambiguous wording" in sent_prompt.lower()
