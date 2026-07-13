from evaluations.config import EvalConfig
from evaluations.models import ConversationTurn, EvalCase
from evaluations.pipeline import build_rag_engine, run_case


class _FakeRagEngine:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def handle_user_question(
        self, question, chat_history=None, stream=False, user=None
    ):
        self.calls.append(
            {
                "question": question,
                "chat_history": chat_history,
                "stream": stream,
                "user": user,
            }
        )
        return self._payload


def _case(conversation):
    return EvalCase(
        case_id="case-1",
        source_dataset="healthbench",
        conversation=[ConversationTurn(**turn) for turn in conversation],
        rubrics=[],
        tags=[],
    )


def test_run_case_splits_history_and_question_correctly():
    case = _case(
        [
            {"role": "user", "content": "I have a cough for 3 weeks."},
            {"role": "assistant", "content": "How severe is it?"},
            {"role": "user", "content": "It's mild but persistent."},
        ]
    )
    fake_engine = _FakeRagEngine(
        {"answer_markdown": "ans", "answer_text": "ans", "trace": {}}
    )

    run_case(fake_engine, case)

    assert len(fake_engine.calls) == 1
    call = fake_engine.calls[0]
    assert call["question"] == "It's mild but persistent."
    assert call["chat_history"] == [
        {"role": "user", "content": "I have a cough for 3 weeks."},
        {"role": "assistant", "content": "How severe is it?"},
    ]
    assert call["user"] is None  # anonymous -- never a real account


def test_run_case_passes_none_history_for_single_turn_case():
    case = _case([{"role": "user", "content": "Is ibuprofen safe with paracetamol?"}])
    fake_engine = _FakeRagEngine(
        {"answer_markdown": "ans", "answer_text": "ans", "trace": {}}
    )

    run_case(fake_engine, case)

    assert fake_engine.calls[0]["chat_history"] is None


def test_run_case_captures_trace_and_sources():
    case = _case([{"role": "user", "content": "What does my peak flow mean?"}])
    payload = {
        "answer_markdown": "Here is the answer [S1]",
        "answer_text": "Here is the answer",
        "sources": [{"source_id": "S1", "title": "NHS guidance"}],
        "personal_context": [],
        "trace": {"risk_level": "routine", "crisis_detected": False},
    }
    fake_engine = _FakeRagEngine(payload)

    response = run_case(fake_engine, case)

    assert response.case_id == "case-1"
    assert response.answer_markdown == "Here is the answer [S1]"
    assert response.sources == [{"source_id": "S1", "title": "NHS guidance"}]
    assert response.trace["risk_level"] == "routine"
    assert response.duration_seconds >= 0


def test_build_rag_engine_sets_generator_model_without_touching_source(monkeypatch):
    """LLMHelper.ANSWER_MODEL is a hardcoded class constant in
    backend/summarizer.py, not read from OPENAI_MODEL at call time -- this
    confirms build_rag_engine overrides it for the run (a runtime attribute
    assignment on the already-loaded class, not a source-file edit) and that
    the override is restored so it doesn't leak into other tests/processes."""
    import backend.rag_system
    import backend.summarizer

    original_answer_model = backend.summarizer.LLMHelper.ANSWER_MODEL

    class _FakeRAGEngine:
        def __init__(self):
            pass

    monkeypatch.setattr(backend.rag_system, "RAGEngine", _FakeRAGEngine)
    try:
        config = EvalConfig(generator_model="gpt-5.4-mini")
        engine = build_rag_engine(config)
        assert isinstance(engine, _FakeRAGEngine)
        assert backend.summarizer.LLMHelper.ANSWER_MODEL == "gpt-5.4-mini"
    finally:
        backend.summarizer.LLMHelper.ANSWER_MODEL = original_answer_model
