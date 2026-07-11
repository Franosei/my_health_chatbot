from types import SimpleNamespace

from backend.query_expander import QueryExpander


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="uroflowmetry Qmax interpretation"))]
        )


def _build_expander(monkeypatch) -> QueryExpander:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    expander = QueryExpander()
    expander.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    return expander


def test_expand_with_patient_context_instructs_using_confirmed_meaning(monkeypatch):
    expander = _build_expander(monkeypatch)

    expander.expand_with_patient_context(
        user_question="What does my peak flow of 18 mean?",
        patient_history_summary="Peak urinary flow rate / Qmax (urology, NOT a respiratory measurement): 18 ml/s",
    )

    sent_prompt = expander.client.chat.completions.calls[0]["messages"][0]["content"]
    assert "confirmed meaning" in sent_prompt.lower()
    assert "raw ambiguous wording" in sent_prompt.lower()
