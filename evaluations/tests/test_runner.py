import argparse
import json

from evaluations import runner
from evaluations.config import EvalConfig
from evaluations.models import (
    AdjudicationDecision,
    CaseResult,
    ConversationTurn,
    DeterministicFindings,
    EvalCase,
    GradingResult,
    PipelineResponse,
    RubricItem,
)


def _valid_case(case_id="case-1"):
    return EvalCase(
        case_id=case_id,
        source_dataset="healthbench",
        conversation=[ConversationTurn(role="user", content="I have a headache.")],
        rubrics=[RubricItem(criterion="Advises rest.", points=5, tags=[])],
        tags=[],
    )


def _invalid_case(case_id="bad-case"):
    return EvalCase(
        case_id=case_id,
        source_dataset="healthbench",
        conversation=[
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello"),
        ],
        rubrics=[],
        tags=[],
    )


def test_dry_run_reports_validation_errors_without_calling_models(capsys):
    cases = [_valid_case("case-1"), _valid_case("case-2"), _invalid_case("bad-case")]

    runner.dry_run(cases, EvalConfig())

    captured = capsys.readouterr()
    assert "3 cases loaded, 1 failed validation" in captured.out
    assert "bad-case" in captured.err
    assert "no models were called" in captured.out


def _minimal_case_result(case_id: str) -> CaseResult:
    grade = GradingResult(
        case_id=case_id,
        grader_model="luna",
        rubric_results=[],
        clinical_correctness_score=0.9,
        triage_appropriateness="appropriate",
        potential_harm_level="none",
        confidence=0.9,
        explanation="fine",
        expected_urgency_level="routine",
    )
    findings = DeterministicFindings(
        case_id=case_id,
        expected_urgency_level="routine",
        actual_urgency_level="routine",
        under_triage=False,
        severe_under_triage=False,
        over_triage=False,
        expected_actual_match=True,
        crisis_gate_expected=False,
        crisis_gate_activated=False,
        medication_or_allergy_fabrication=False,
        patient_record_fabrication=False,
        cross_patient_leakage=False,
        citations_present=None,
        citations_supported=None,
        safe_refusal_violated=False,
        clarification_expected=False,
        clarification_observed=False,
        prompt_injection_detected=False,
        prompt_injection_resisted=None,
        deterministic_pass=True,
        failure_reasons=[],
    )
    return CaseResult(
        case=_valid_case(case_id),
        pipeline_response=PipelineResponse(
            case_id=case_id,
            answer_markdown="ans",
            answer_text="ans",
            trace={"risk_level": "routine"},
        ),
        adjudication=AdjudicationDecision(
            case_id=case_id, triggered=False, luna_grade=grade, final_grade=grade
        ),
        deterministic=findings,
        weighted_score=1.0,
        overall_pass=True,
    )


def test_load_completed_case_ids_returns_empty_set_for_missing_file(tmp_path):
    assert runner._load_completed_case_ids(tmp_path / "does_not_exist.jsonl") == set()


def test_load_completed_case_ids_reads_existing_results(tmp_path):
    raw_path = tmp_path / "cases.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        fh.write(_minimal_case_result("case-1").model_dump_json() + "\n")
        fh.write(_minimal_case_result("case-2").model_dump_json() + "\n")

    assert runner._load_completed_case_ids(raw_path) == {"case-1", "case-2"}


def test_load_completed_case_ids_skips_malformed_lines(tmp_path):
    raw_path = tmp_path / "cases.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        fh.write(_minimal_case_result("case-1").model_dump_json() + "\n")
        fh.write("not valid json\n")

    assert runner._load_completed_case_ids(raw_path) == {"case-1"}


def test_run_dataset_applies_sample_limit_and_resume(tmp_path, monkeypatch):
    all_cases = [_valid_case(f"case-{i}") for i in range(5)]
    monkeypatch.setattr(
        runner, "_prepare_cases", lambda dataset_name, force_download: all_cases
    )

    class _FakeRagEngine:
        pass

    monkeypatch.setattr(
        "evaluations.pipeline.build_rag_engine", lambda config: _FakeRagEngine()
    )

    evaluated_ids = []

    def _fake_evaluate_case(case, rag_engine, config):
        evaluated_ids.append(case.case_id)
        return _minimal_case_result(case.case_id)

    monkeypatch.setattr(runner, "evaluate_case", _fake_evaluate_case)

    config = EvalConfig(output_path=tmp_path, sample_limit=3)
    args = argparse.Namespace(
        dry_run=False,
        resume=False,
        run_id="fixed-run",
        force_download=False,
        batch=False,
    )

    runner.run_dataset("healthbench", args, config)

    # Only the first 3 (sample limit) were evaluated.
    assert evaluated_ids == ["case-0", "case-1", "case-2"]

    summary_path = tmp_path / "reports" / "fixed-run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]["total_cases"] == 3

    # Now resume: same run id, sample bumped to 5 -- only the 2 new cases should run.
    evaluated_ids.clear()
    config2 = EvalConfig(output_path=tmp_path, sample_limit=5)
    args2 = argparse.Namespace(
        dry_run=False,
        resume=True,
        run_id="fixed-run",
        force_download=False,
        batch=False,
    )
    runner.run_dataset("healthbench", args2, config2)

    assert evaluated_ids == ["case-3", "case-4"]

    summary_path_2 = tmp_path / "reports" / "fixed-run_summary.json"
    summary2 = json.loads(summary_path_2.read_text(encoding="utf-8"))
    assert summary2["summary"]["total_cases"] == 5


def test_run_dataset_dry_run_never_builds_rag_engine(tmp_path, monkeypatch):
    all_cases = [_valid_case("case-1")]
    monkeypatch.setattr(
        runner, "_prepare_cases", lambda dataset_name, force_download: all_cases
    )

    def _should_not_be_called(config):
        raise AssertionError("build_rag_engine must not be called in --dry-run")

    monkeypatch.setattr("evaluations.pipeline.build_rag_engine", _should_not_be_called)

    config = EvalConfig(output_path=tmp_path)
    args = argparse.Namespace(
        dry_run=True, resume=False, run_id=None, force_download=False, batch=False
    )

    runner.run_dataset("healthbench", args, config)  # should not raise
