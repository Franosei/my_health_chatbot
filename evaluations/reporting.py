"""Builds the raw results file plus a sanitised JSON + Markdown summary from
a completed evaluation run.

Raw results (full conversations, full answer text, full grading explanations)
are written under `evaluations/results/raw/` -- gitignored, local only. The
sanitised summary/report only ever contains case identifiers, tags, scores,
and flags -- never full conversation or answer text -- and both the JSON and
Markdown outputs are headed with the "automated benchmark evaluation, not
clinical validation" label plus an explicit list of cases needing qualified
clinician review.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from evaluations.config import GRADING_PROMPT_VERSION, EvalConfig
from evaluations.models import CaseResult, ReportSummary

REPORT_LABEL = "Automated benchmark evaluation -- not clinical validation"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pipeline_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def build_report_summary(
    case_results: List[CaseResult],
    config: EvalConfig,
    dataset_version: str,
) -> ReportSummary:
    total = len(case_results)
    if total == 0:
        return ReportSummary(
            dataset_version=dataset_version,
            pipeline_version=pipeline_version(),
            prompt_version=GRADING_PROMPT_VERSION,
            generator_model=config.generator_model,
            primary_grader_model=config.primary_grader_model,
            adjudicator_model=config.adjudicator_model,
            run_date=_utc_now(),
            total_cases=0,
            pass_rate=0.0,
            weighted_healthbench_score=0.0,
            under_triage_rate=0.0,
            severe_under_triage_rate=0.0,
            emergency_sensitivity=None,
            unsupported_claim_rate=0.0,
            adjudication_rate=0.0,
            disagreement_count=0,
            cases_requiring_human_review=[],
            notes=["No cases were evaluated in this run."],
        )

    passed = sum(1 for cr in case_results if cr.overall_pass)
    weighted_scores = [cr.weighted_score for cr in case_results]
    under_triage = sum(1 for cr in case_results if cr.deterministic.under_triage)
    severe_under_triage = sum(
        1 for cr in case_results if cr.deterministic.severe_under_triage
    )
    unsupported_claims = sum(
        1 for cr in case_results if cr.adjudication.final_grade.unsupported_claims
    )
    adjudicated = sum(1 for cr in case_results if cr.adjudication.triggered)
    disagreements = sum(1 for cr in case_results if cr.adjudication.agreement is False)

    emergency_expected = [
        cr for cr in case_results if cr.deterministic.crisis_gate_expected
    ]
    emergency_sensitivity = (
        sum(1 for cr in emergency_expected if cr.deterministic.crisis_gate_activated)
        / len(emergency_expected)
        if emergency_expected
        else None
    )

    review_cases = [
        cr.case.case_id for cr in case_results if cr.requires_human_review()
    ]

    return ReportSummary(
        dataset_version=dataset_version,
        pipeline_version=pipeline_version(),
        prompt_version=GRADING_PROMPT_VERSION,
        generator_model=config.generator_model,
        primary_grader_model=config.primary_grader_model,
        adjudicator_model=config.adjudicator_model,
        run_date=_utc_now(),
        total_cases=total,
        pass_rate=passed / total,
        weighted_healthbench_score=sum(weighted_scores) / total,
        under_triage_rate=under_triage / total,
        severe_under_triage_rate=severe_under_triage / total,
        emergency_sensitivity=emergency_sensitivity,
        unsupported_claim_rate=unsupported_claims / total,
        adjudication_rate=adjudicated / total,
        disagreement_count=disagreements,
        cases_requiring_human_review=review_cases,
    )


def _sanitized_case_entry(case_result: CaseResult) -> dict:
    return {
        "case_id": case_result.case.case_id,
        "source_dataset": case_result.case.source_dataset,
        "tags": case_result.case.tags,
        "resolved_role": case_result.pipeline_response.resolved_role,
        "weighted_score": round(case_result.weighted_score, 4),
        "overall_pass": case_result.overall_pass,
        "adjudicated": case_result.adjudication.triggered,
        "trigger_reasons": case_result.adjudication.trigger_reasons,
        "agreement": case_result.adjudication.agreement,
        "deterministic_pass": case_result.deterministic.deterministic_pass,
        "failure_reasons": case_result.deterministic.failure_reasons,
        "expected_urgency_level": case_result.deterministic.expected_urgency_level,
        "actual_urgency_level": case_result.deterministic.actual_urgency_level,
        "potential_harm_level": case_result.adjudication.final_grade.potential_harm_level,
        "confidence": case_result.adjudication.final_grade.confidence,
        "requires_human_review": case_result.requires_human_review(),
    }


def _markdown_report(summary: ReportSummary, case_results: List[CaseResult]) -> str:
    lines = [
        f"# {REPORT_LABEL}",
        "",
        "This report is produced by an automated benchmark harness. It is **not** a",
        "clinical validation of FlynnMed and must not be represented as one. Severe,",
        "uncertain, or disputed cases are listed below and require review by a",
        "qualified clinician before any conclusion is drawn from them.",
        "",
        "## Run metadata",
        "",
        f"- Dataset version: `{summary.dataset_version}`",
        f"- Pipeline version (git commit): `{summary.pipeline_version}`",
        f"- Grading prompt version: `{summary.prompt_version}`",
        f"- Generator model: `{summary.generator_model}`",
        f"- Primary grader model (Luna): `{summary.primary_grader_model}`",
        f"- Adjudicator model (Terra): `{summary.adjudicator_model}`",
        f"- Run date: `{summary.run_date}`",
        f"- Total cases: {summary.total_cases}",
        "",
        "## Headline metrics",
        "",
        f"- Pass rate: {summary.pass_rate:.1%}",
        f"- Weighted HealthBench score: {summary.weighted_healthbench_score:.3f}",
        f"- Under-triage rate: {summary.under_triage_rate:.1%}",
        f"- Severe under-triage rate: {summary.severe_under_triage_rate:.1%}",
        (
            f"- Emergency sensitivity: {summary.emergency_sensitivity:.1%}"
            if summary.emergency_sensitivity is not None
            else "- Emergency sensitivity: n/a (no cases expected an emergency disposition)"
        ),
        f"- Unsupported-claim rate: {summary.unsupported_claim_rate:.1%}",
        f"- Adjudication rate (Terra invoked): {summary.adjudication_rate:.1%}",
        f"- Luna/Terra disagreements: {summary.disagreement_count}",
        "",
    ]

    if summary.cases_requiring_human_review:
        lines.append("## Cases requiring qualified clinician review")
        lines.append("")
        for case_id in summary.cases_requiring_human_review:
            match = next(
                (cr for cr in case_results if cr.case.case_id == case_id), None
            )
            reasons = (
                ", ".join(match.deterministic.failure_reasons)
                if match and match.deterministic.failure_reasons
                else "low confidence / disagreement"
            )
            lines.append(f"- `{case_id}` -- {reasons}")
        lines.append("")

    if summary.notes:
        lines.append("## Notes")
        lines.append("")
        for note in summary.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def write_report(
    case_results: List[CaseResult],
    config: EvalConfig,
    dataset_version: str,
    run_id: str,
) -> Tuple[Path, Path, Path]:
    """Writes raw results (full detail, gitignored) plus a sanitised JSON and
    Markdown summary. Returns (raw_path, summary_json_path, summary_md_path).
    """
    output_root = Path(config.output_path)
    raw_dir = output_root / "raw" / run_id
    reports_dir = output_root / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "cases.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        for case_result in case_results:
            fh.write(case_result.model_dump_json() + "\n")

    summary = build_report_summary(case_results, config, dataset_version)

    summary_json_path = reports_dir / f"{run_id}_summary.json"
    sanitized = {
        "label": REPORT_LABEL,
        "summary": json.loads(summary.model_dump_json()),
        "cases": [_sanitized_case_entry(cr) for cr in case_results],
    }
    with open(summary_json_path, "w", encoding="utf-8") as fh:
        json.dump(sanitized, fh, indent=2)

    summary_md_path = reports_dir / f"{run_id}_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as fh:
        fh.write(_markdown_report(summary, case_results))

    return raw_path, summary_json_path, summary_md_path
