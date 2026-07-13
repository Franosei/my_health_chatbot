import json

from evaluations.config import EvalConfig
from evaluations.models import (
    AdjudicationDecision,
    CaseResult,
    ConversationTurn,
    DeterministicFindings,
    EvalCase,
    GradingResult,
    PipelineResponse,
)
from evaluations.reporting import REPORT_LABEL, build_report_summary, write_report

_UNIQUE_CONVERSATION_MARKER = (
    "a very specific patient sentence that must never leak into the sanitised summary"
)


def _grade(**overrides):
    fields = dict(
        case_id="case-1",
        grader_model="luna",
        rubric_results=[],
        clinical_correctness_score=0.9,
        triage_appropriateness="appropriate",
        potential_harm_level="none",
        unsupported_claims=[],
        missing_critical_information=[],
        confidence=0.9,
        explanation="fine",
        expected_urgency_level="routine",
        clarification_warranted=False,
    )
    fields.update(overrides)
    return GradingResult.model_validate(fields)


def _findings(**overrides):
    fields = dict(
        case_id="case-1",
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
    fields.update(overrides)
    return DeterministicFindings(**fields)


def _case_result(
    case_id="case-1",
    overall_pass=True,
    weighted_score=0.9,
    deterministic_overrides=None,
    grade_overrides=None,
    adjudicated=False,
) -> CaseResult:
    case = EvalCase(
        case_id=case_id,
        source_dataset="healthbench",
        conversation=[
            ConversationTurn(role="user", content=_UNIQUE_CONVERSATION_MARKER)
        ],
        rubrics=[],
        tags=["theme:test"],
    )
    pipeline_response = PipelineResponse(
        case_id=case_id,
        answer_markdown=_UNIQUE_CONVERSATION_MARKER + " answer",
        answer_text=_UNIQUE_CONVERSATION_MARKER + " answer",
        trace={"risk_level": "routine"},
    )
    grade = _grade(case_id=case_id, **(grade_overrides or {}))
    findings = _findings(case_id=case_id, **(deterministic_overrides or {}))
    adjudication = AdjudicationDecision(
        case_id=case_id,
        triggered=adjudicated,
        trigger_reasons=["emergency_case"] if adjudicated else [],
        luna_grade=grade,
        terra_grade=grade if adjudicated else None,
        agreement=True if adjudicated else None,
        final_grade=grade,
    )
    return CaseResult(
        case=case,
        pipeline_response=pipeline_response,
        adjudication=adjudication,
        deterministic=findings,
        weighted_score=weighted_score,
        overall_pass=overall_pass,
    )


def test_build_report_summary_computes_pass_rate():
    results = [
        _case_result("case-1", overall_pass=True, weighted_score=1.0),
        _case_result(
            "case-2",
            overall_pass=False,
            weighted_score=0.0,
            deterministic_overrides={
                "deterministic_pass": False,
                "failure_reasons": ["severe_under_triage"],
            },
        ),
    ]
    summary = build_report_summary(results, EvalConfig(), dataset_version="healthbench")

    assert summary.total_cases == 2
    assert summary.pass_rate == 0.5
    assert summary.weighted_healthbench_score == 0.5
    assert "case-2" in summary.cases_requiring_human_review


def test_build_report_summary_handles_empty_list():
    summary = build_report_summary([], EvalConfig(), dataset_version="healthbench")
    assert summary.total_cases == 0
    assert summary.pass_rate == 0.0
    assert summary.notes


def test_historical_report_can_preserve_original_prompt_version():
    summary = build_report_summary(
        [_case_result("case-1")],
        EvalConfig(),
        dataset_version="healthbench",
        prompt_version="v1",
        run_date="2026-07-13T06:30:27+00:00",
    )
    assert summary.prompt_version == "v1"
    assert summary.run_date == "2026-07-13T06:30:27+00:00"


def test_emergency_sensitivity_none_when_no_emergency_expected():
    results = [
        _case_result("case-1", deterministic_overrides={"crisis_gate_expected": False})
    ]
    summary = build_report_summary(results, EvalConfig(), dataset_version="healthbench")
    assert summary.emergency_sensitivity is None


def test_emergency_sensitivity_computed_from_crisis_gate_cases():
    results = [
        _case_result(
            "case-1",
            deterministic_overrides={
                "crisis_gate_expected": True,
                "crisis_gate_activated": True,
            },
        ),
        _case_result(
            "case-2",
            deterministic_overrides={
                "crisis_gate_expected": True,
                "crisis_gate_activated": False,
                "deterministic_pass": False,
                "failure_reasons": ["crisis_gate_missed"],
            },
        ),
    ]
    summary = build_report_summary(results, EvalConfig(), dataset_version="healthbench")
    assert summary.emergency_sensitivity == 0.5


def test_disagreement_count():
    results = [_case_result("case-1", adjudicated=True)]
    results[0].adjudication.agreement = False
    summary = build_report_summary(results, EvalConfig(), dataset_version="healthbench")
    assert summary.disagreement_count == 1


def test_evidence_metrics_do_not_mislabel_full_source_truth():
    result = _case_result(
        "case-1",
        grade_overrides={"unsupported_claims": ["one uncited recommendation"]},
        deterministic_overrides={
            "citation_count": 2,
            "resolved_citation_count": 2,
            "citation_target_resolution_rate": 1.0,
            "claim_checks_total": 5,
            "claims_supported_by_excerpt": 4,
            "claim_excerpt_support_rate": 0.8,
        },
    )
    result.pipeline_response.sources = [
        {"source_id": "S1", "url": "https://www.nhs.uk/example"},
        {"source_id": "S2", "url": "https://pubmed.ncbi.nlm.nih.gov/1/"},
    ]

    summary = build_report_summary(
        [result], EvalConfig(), dataset_version="healthbench"
    )

    assert summary.responses_with_grader_flagged_claims_rate == 1.0
    assert summary.grader_flagged_claim_count == 1
    assert summary.claim_excerpt_support_rate == 0.8
    assert summary.citation_target_resolution_rate == 1.0
    assert summary.displayed_sources_with_url_rate == 1.0
    assert summary.full_source_content_verification_rate is None


def test_write_report_creates_raw_and_sanitised_files(tmp_path):
    config = EvalConfig(output_path=tmp_path)
    results = [_case_result("case-1")]

    raw_path, summary_json_path, summary_md_path = write_report(
        results, config, dataset_version="healthbench", run_id="test-run"
    )

    assert raw_path.exists()
    assert summary_json_path.exists()
    assert summary_md_path.exists()

    raw_content = raw_path.read_text(encoding="utf-8")
    assert (
        _UNIQUE_CONVERSATION_MARKER in raw_content
    )  # full detail belongs in raw results

    summary_content = summary_json_path.read_text(encoding="utf-8")
    assert (
        _UNIQUE_CONVERSATION_MARKER not in summary_content
    )  # never in the sanitised summary

    summary_json = json.loads(summary_content)
    assert summary_json["label"] == REPORT_LABEL
    assert summary_json["cases"][0]["case_id"] == "case-1"

    md_content = summary_md_path.read_text(encoding="utf-8")
    assert REPORT_LABEL in md_content
    assert _UNIQUE_CONVERSATION_MARKER not in md_content


def test_write_report_lists_human_review_cases_in_markdown(tmp_path):
    config = EvalConfig(output_path=tmp_path)
    results = [
        _case_result(
            "case-1",
            overall_pass=False,
            deterministic_overrides={
                "deterministic_pass": False,
                "failure_reasons": ["medication_or_allergy_fabrication"],
            },
        ),
    ]

    _, _, summary_md_path = write_report(
        results, config, dataset_version="healthbench", run_id="test-run-2"
    )

    md_content = summary_md_path.read_text(encoding="utf-8")
    assert "case-1" in md_content
    assert "medication_or_allergy_fabrication" in md_content
    assert "qualified clinician review" in md_content.lower()
