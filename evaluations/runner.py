"""CLI entry point for the evaluation harness.

    python -m evaluations.runner --dataset healthbench --dry-run
    python -m evaluations.runner --dataset healthbench_hard --sample 8
    python -m evaluations.runner --dataset healthbench_consensus --resume --run-id my-run
    python -m evaluations.runner --dataset all

See evaluations/README.md for full documentation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set

from evaluations.config import (
    DATASET_URLS,
    DOWNLOADED_DIR,
    NORMALIZED_DIR,
    RAG_METRICS_PROMPT_VERSION,
    EvalConfig,
    load_config,
)
from evaluations.datasets.adapter import load_cases, normalize_dataset, normalized_path
from evaluations.datasets.download import dataset_path, download_dataset
from evaluations.deterministic_metrics import compute_deterministic_findings
from evaluations.grading import (
    agreement_between,
    grade_with_primary,
    grade_with_terra,
    should_adjudicate,
)
from evaluations.models import AdjudicationDecision, CaseResult, EvalCase
from evaluations.reporting import write_report
from evaluations.retry import call_with_retry


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def warn_if_adjudication_disabled(config: EvalConfig) -> None:
    """Adjudication is silently skipped whenever the primary grader and the
    adjudicator resolve to the same model (see finalize_healthbench_result).
    That's an intentional cost-saving default, not a bug -- but it means every
    case flagged for a second opinion this run will get none, so results
    should not be treated as independently cross-checked. Print this loudly
    before any case runs rather than letting it hide in per-case trigger
    reasons."""
    if config.primary_grader_model == config.adjudicator_model:
        print(
            "[runner] WARNING: EVAL_PRIMARY_GRADER_MODEL and EVAL_ADJUDICATOR_MODEL "
            f"are both '{config.primary_grader_model}'. Every case that triggers "
            "adjudication this run will be skipped (no independent second opinion) "
            "-- results should not be treated as cross-checked. Set "
            "EVAL_ADJUDICATOR_MODEL to a different model to enable it.",
            file=sys.stderr,
        )


def _prepare_cases(dataset_name: str, force_download: bool) -> List[EvalCase]:
    raw_path = dataset_path(dataset_name, dest_dir=DOWNLOADED_DIR)
    if force_download or not raw_path.exists():
        raw_path = download_dataset(
            dataset_name, dest_dir=DOWNLOADED_DIR, force=force_download
        )

    norm_path = normalized_path(dataset_name, dest_dir=NORMALIZED_DIR)
    normalize_dataset(raw_path, norm_path, source_dataset=dataset_name)
    return load_cases(norm_path)


def _load_completed_case_ids(raw_path: Path) -> Set[str]:
    if not raw_path.exists():
        return set()
    completed = set()
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if (
                    payload.get("rag_metrics")
                    and payload.get("adjudication")
                    and payload.get("deterministic")
                    and payload.get("weighted_score") is not None
                ):
                    completed.add(payload["case"]["case_id"])
            except Exception:
                continue
    return completed


def dry_run(cases: List[EvalCase], config: EvalConfig) -> None:
    """Validates data and prompts without calling any model. Also reports the
    role each case would be routed as (see resolve_case_user/role_detection.py)
    without actually creating any eval account -- purely informational."""
    from evaluations.role_detection import detect_stated_role

    errors = 0
    role_counts: dict = {}
    for case in cases:
        role = detect_stated_role(case)
        role_counts[role] = role_counts.get(role, 0) + 1
        try:
            case.last_user_turn()
        except Exception as exc:
            errors += 1
            print(
                f"[dry-run] case {case.case_id} failed validation: {exc}",
                file=sys.stderr,
            )

    print(f"[dry-run] {len(cases)} cases loaded, {errors} failed validation.")
    print(f"[dry-run] detected roles: {role_counts}")
    print(
        f"[dry-run] generator_model={config.generator_model} "
        f"primary_grader_model={config.primary_grader_model} "
        f"adjudicator_model={config.adjudicator_model} "
        f"rag_metrics_model={config.rag_metrics_model}"
    )
    print("[dry-run] no models were called.")


def resolve_case_user(case: EvalCase) -> tuple[str, Optional[str]]:
    """Detects a self-stated clinical role in the case's own conversation and
    resolves the eval account to run it as (None for patient -- FlynnMed's
    own default for an anonymous/empty profile, so nothing changes for the
    common case). Returns (role, user). See evaluations/role_detection.py and
    evaluations/pipeline.py's module docstring for why this exists."""
    from evaluations.pipeline import ensure_eval_account
    from evaluations.role_detection import detect_stated_role

    role = detect_stated_role(case)
    return role, ensure_eval_account(role, case.case_id)


def run_case_pipeline(case: EvalCase, rag_engine, config: EvalConfig):
    from evaluations.pipeline import run_case

    role, user = resolve_case_user(case)
    return call_with_retry(
        lambda: run_case(rag_engine, case, user=user, role=role),
        max_retries=config.max_retries,
    )


def _add_consistency_repeats(
    case: EvalCase, pipeline_response, rag_engine, config: EvalConfig
):
    """Periodically repeat the exact production call when explicitly enabled."""

    for _ in range(config.consistency_repeats):
        repeated = run_case_pipeline(case, rag_engine, config)
        pipeline_response.consistency_answers.append(repeated.answer_text)
    return pipeline_response


def _attach_rag_metrics(case_result: CaseResult, config: EvalConfig) -> CaseResult:
    from evaluations.rag_metrics import grade_rag_metrics, unavailable_rag_metrics

    try:
        case_result.rag_metrics = call_with_retry(
            lambda: grade_rag_metrics(
                case_result.case, case_result.pipeline_response, config
            ),
            max_retries=config.max_retries,
        )
    except Exception as exc:  # metric failure must not discard the core grade
        case_result.rag_metrics = unavailable_rag_metrics(
            case_result.case.case_id, config, exc
        )
    return case_result


def finalize_healthbench_result(
    case: EvalCase, pipeline_response, luna_grade, config: EvalConfig
) -> CaseResult:
    """Apply deterministic checks and independent secondary adjudication.

    Both graders receive the same captured FlynnMed answer and physician-authored
    rubrics. The adjudicator never receives the primary output. The final weighted score is
    computed locally from exact rubric points, never supplied by either model.
    """
    preliminary = compute_deterministic_findings(case, pipeline_response, luna_grade)
    triggered, reasons = should_adjudicate(
        case, pipeline_response, luna_grade, preliminary, config
    )

    terra_grade = None
    agreement = None
    adjudication_skipped = False
    adjudication_error = None
    final_grade = luna_grade
    final_findings = preliminary
    if triggered:
        if config.adjudicator_model == luna_grade.grader_model:
            adjudication_skipped = True
            reasons.append("same_model_adjudication_skipped")
        else:
            try:
                terra_grade = call_with_retry(
                    lambda: grade_with_terra(case, pipeline_response, config),
                    max_retries=config.max_retries,
                )
            except Exception as exc:
                adjudication_error = f"{type(exc).__name__}: {exc}"
                reasons.append("adjudicator_failed")
            else:
                agreement = agreement_between(luna_grade, terra_grade)
                final_grade = terra_grade
                final_findings = compute_deterministic_findings(
                    case, pipeline_response, terra_grade
                )

    adjudication = AdjudicationDecision(
        case_id=case.case_id,
        triggered=triggered,
        trigger_reasons=reasons,
        luna_grade=luna_grade,
        terra_grade=terra_grade,
        agreement=agreement,
        adjudication_skipped=adjudication_skipped,
        adjudication_error=adjudication_error,
        final_grade=final_grade,
    )
    weighted_score = final_grade.weighted_score(case)
    ai_pass = final_grade.potential_harm_level != "severe" and weighted_score >= 0.5
    return CaseResult(
        case=case,
        pipeline_response=pipeline_response,
        adjudication=adjudication,
        deterministic=final_findings,
        weighted_score=weighted_score,
        overall_pass=final_findings.deterministic_pass and ai_pass,
    )


def evaluate_case(case: EvalCase, rag_engine, config: EvalConfig) -> CaseResult:
    pipeline_response = run_case_pipeline(case, rag_engine, config)
    pipeline_response = _add_consistency_repeats(
        case, pipeline_response, rag_engine, config
    )
    luna_grade = call_with_retry(
        lambda: grade_with_primary(case, pipeline_response, config),
        max_retries=config.max_retries,
    )
    result = finalize_healthbench_result(case, pipeline_response, luna_grade, config)
    return _attach_rag_metrics(result, config)


def run_dataset(
    dataset_name: str, args: argparse.Namespace, config: EvalConfig
) -> None:
    if getattr(args, "regrade_rag", False):
        _regrade_saved_rag_metrics(dataset_name, args, config)
        return
    cases = _prepare_cases(dataset_name, force_download=args.force_download)

    random_seed = getattr(args, "random_seed", None)
    if random_seed is not None:
        random.Random(random_seed).shuffle(cases)
        print(f"[{dataset_name}] randomized case order with seed {random_seed}.")

    if config.sample_limit is not None:
        cases = cases[: config.sample_limit]

    if args.dry_run:
        dry_run(cases, config)
        return

    run_id = args.run_id or f"{dataset_name}_{_utc_timestamp()}"
    raw_path = Path(config.output_path) / "raw" / run_id / "cases.jsonl"

    completed_ids: Set[str] = set()
    if args.resume:
        completed_ids = _load_completed_case_ids(raw_path)
        if completed_ids:
            print(
                f"[resume] {len(completed_ids)} cases already completed for run '{run_id}', skipping them."
            )

    remaining = [c for c in cases if c.case_id not in completed_ids]
    if not remaining:
        print(
            f"[{dataset_name}] nothing to do -- all {len(cases)} cases already completed."
        )
        return

    from evaluations.pipeline import build_rag_engine

    rag_engine = build_rag_engine(config)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[CaseResult] = []
    if completed_ids:
        with open(raw_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    result = CaseResult.model_validate_json(line)
                    if (
                        result.rag_metrics
                        and result.adjudication
                        and result.deterministic
                        and result.weighted_score is not None
                    ):
                        results.append(result)

    with open(raw_path, "a", encoding="utf-8") as append_fh:
        _run_synchronous(
            dataset_name, remaining, rag_engine, config, results, append_fh
        )

    _, summary_json_path, summary_md_path = write_report(
        results,
        config,
        dataset_version=dataset_name,
        run_id=run_id,
    )
    print(f"[{dataset_name}] wrote raw results to {raw_path}")
    print(f"[{dataset_name}] wrote report to {summary_json_path} and {summary_md_path}")


def _regrade_saved_rag_metrics(
    dataset_name: str, args: argparse.Namespace, config: EvalConfig
) -> None:
    """Re-run only RAG metrics against immutable saved answers and sources."""
    if not args.run_id:
        raise ValueError("--regrade-rag requires --run-id")
    raw_path = Path(config.output_path) / "raw" / args.run_id / "cases.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Saved run not found: {raw_path}")

    saved: List[CaseResult] = []
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                saved.append(CaseResult.model_validate_json(line))
    case_ids = [result.case.case_id for result in saved]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Saved run contains duplicate case ids; refusing to re-grade.")

    checkpoint_name = f"rag_regrade_{RAG_METRICS_PROMPT_VERSION}.jsonl"
    checkpoint_path = raw_path.parent / checkpoint_name
    completed: dict[str, CaseResult] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                result = CaseResult.model_validate_json(line)
                if result.rag_metrics and not result.rag_metrics.evaluation_error:
                    completed[result.case.case_id] = result
        if completed:
            print(
                f"[{dataset_name}] loaded {len(completed)} completed RAG re-grades "
                f"from {checkpoint_path.name}"
            )

    regraded_by_id = dict(completed)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_mode = "a" if checkpoint_path.exists() else "w"
    checkpoint_fh = open(checkpoint_path, checkpoint_mode, encoding="utf-8")
    try:
        for index, result in enumerate(saved, start=1):
            case_id = result.case.case_id
            if case_id in completed:
                continue
            print(f"[{dataset_name}] RAG re-grade ({index}/{len(saved)}) {case_id}")
            regraded = _attach_rag_metrics(result, config)
            regraded_by_id[case_id] = regraded
            if regraded.rag_metrics and not regraded.rag_metrics.evaluation_error:
                checkpoint_fh.write(regraded.model_dump_json() + "\n")
                checkpoint_fh.flush()
    finally:
        checkpoint_fh.close()

    regraded = [regraded_by_id[result.case.case_id] for result in saved]

    _, summary_json_path, summary_md_path = write_report(
        regraded,
        config,
        dataset_version=dataset_name,
        run_id=args.run_id,
    )
    print(f"[{dataset_name}] preserved generation and re-graded {len(regraded)} cases")
    print(f"[{dataset_name}] wrote report to {summary_json_path} and {summary_md_path}")


def _run_synchronous(
    dataset_name: str,
    remaining: List[EvalCase],
    rag_engine,
    config: EvalConfig,
    results: List[CaseResult],
    append_fh,
) -> None:
    for index, case in enumerate(remaining, start=1):
        print(f"[{dataset_name}] ({index}/{len(remaining)}) {case.case_id}")
        try:
            case_result = evaluate_case(case, rag_engine, config)
        except Exception as exc:
            print(
                f"[{dataset_name}] case {case.case_id} FAILED: {exc}", file=sys.stderr
            )
            traceback.print_exc()
            continue
        results.append(case_result)
        append_fh.write(case_result.model_dump_json() + "\n")
        append_fh.flush()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the FlynnMed HealthBench evaluation harness."
    )
    parser.add_argument(
        "--dataset", choices=[*DATASET_URLS.keys(), "all"], required=True
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and prompts without calling any model.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only run the first N cases (overrides EVAL_SAMPLE_LIMIT).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Shuffle cases reproducibly before applying --sample.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run, skipping already-completed cases.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run id (required to --resume a specific run).",
    )
    parser.add_argument(
        "--regrade-rag",
        action="store_true",
        help="Re-grade only RAG metrics for a saved --run-id without regenerating answers.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset even if a local copy exists.",
    )
    parser.add_argument(
        "--consistency-repeats",
        type=int,
        default=None,
        help="Additional identical production calls per case for periodic consistency scoring.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    if args.sample is not None:
        config.sample_limit = args.sample
    if args.consistency_repeats is not None:
        config.consistency_repeats = max(0, args.consistency_repeats)

    warn_if_adjudication_disabled(config)

    if not args.dry_run:
        from evaluations.grading import EvaluatorAccessError, validate_evaluator_access

        print("[runner] checking access to configured evaluator models...")
        try:
            validate_evaluator_access(config)
        except EvaluatorAccessError as exc:
            print(f"[runner] evaluator access check FAILED: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
        print("[runner] evaluator model access confirmed.")

    dataset_names = (
        list(DATASET_URLS.keys()) if args.dataset == "all" else [args.dataset]
    )
    started = time.perf_counter()
    for dataset_name in dataset_names:
        run_dataset(dataset_name, args, config)
    print(f"[runner] done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
