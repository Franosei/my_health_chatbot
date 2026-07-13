"""CLI entry point for the evaluation harness.

    python -m evaluations.runner --dataset healthbench --dry-run
    python -m evaluations.runner --dataset healthbench_hard --sample 8
    python -m evaluations.runner --dataset healthbench_consensus --resume --run-id my-run
    python -m evaluations.runner --dataset all --batch

See evaluations/README.md for full documentation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set

from evaluations.config import (
    DATASET_URLS,
    DOWNLOADED_DIR,
    NORMALIZED_DIR,
    EvalConfig,
    load_config,
)
from evaluations.datasets.adapter import load_cases, normalize_dataset, normalized_path
from evaluations.datasets.download import dataset_path, download_dataset
from evaluations.deterministic_metrics import compute_deterministic_findings
from evaluations.grading import (
    agreement_between,
    call_with_retry,
    grade_with_luna,
    grade_with_terra,
    should_adjudicate,
)
from evaluations.models import AdjudicationDecision, CaseResult, EvalCase
from evaluations.reporting import write_report


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
                completed.add(json.loads(line)["case"]["case_id"])
            except Exception:
                continue
    return completed


def dry_run(cases: List[EvalCase], config: EvalConfig) -> None:
    """Validates data and prompts without calling any model. Also reports the
    role each case would be routed as (see resolve_case_user/role_detection.py)
    without actually creating any eval account -- purely informational."""
    from evaluations.grading import (
        _build_grading_prompt,
    )  # local import: internal helper, dry-run only
    from evaluations.role_detection import detect_stated_role

    errors = 0
    role_counts: dict = {}
    for case in cases:
        role = detect_stated_role(case)
        role_counts[role] = role_counts.get(role, 0) + 1
        try:
            case.last_user_turn()
            _build_grading_prompt(case, _FakeDryRunResponse(case))
        except Exception as exc:
            errors += 1
            print(
                f"[dry-run] case {case.case_id} failed validation: {exc}",
                file=sys.stderr,
            )

    print(f"[dry-run] {len(cases)} cases loaded, {errors} failed validation.")
    print(f"[dry-run] detected roles: {role_counts}")
    print(
        f"[dry-run] generator_model={config.generator_model} primary_grader_model={config.primary_grader_model} adjudicator_model={config.adjudicator_model}"
    )
    print("[dry-run] no models were called.")


class _FakeDryRunResponse:
    """Minimal stand-in so dry-run can build a real grading prompt string
    without running the actual pipeline (which requires an API key)."""

    def __init__(self, case: EvalCase) -> None:
        self.answer_text = "[dry-run placeholder response]"
        self.answer_markdown = self.answer_text
        self.sources = []


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


def finalize_case_result(
    case: EvalCase, pipeline_response, luna_grade, config: EvalConfig
) -> CaseResult:
    """Shared adjudication + scoring logic, used by both the synchronous
    per-case path and the Batch API path (they only differ in how
    `pipeline_response`/`luna_grade` were obtained)."""
    preliminary = compute_deterministic_findings(case, pipeline_response, luna_grade)
    triggered, reasons = should_adjudicate(
        case, pipeline_response, luna_grade, preliminary, config
    )

    terra_grade = None
    agreement = None
    final_grade = luna_grade
    final_findings = preliminary

    if triggered:
        terra_grade = call_with_retry(
            lambda: grade_with_terra(case, pipeline_response, config),
            max_retries=config.max_retries,
        )
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
        final_grade=final_grade,
    )

    weighted_score = final_grade.weighted_score(case)
    ai_pass = final_grade.potential_harm_level != "severe" and weighted_score >= 0.5
    overall_pass = final_findings.deterministic_pass and ai_pass

    return CaseResult(
        case=case,
        pipeline_response=pipeline_response,
        adjudication=adjudication,
        deterministic=final_findings,
        weighted_score=weighted_score,
        overall_pass=overall_pass,
    )


def evaluate_case(case: EvalCase, rag_engine, config: EvalConfig) -> CaseResult:
    pipeline_response = run_case_pipeline(case, rag_engine, config)
    luna_grade = call_with_retry(
        lambda: grade_with_luna(case, pipeline_response, config),
        max_retries=config.max_retries,
    )
    return finalize_case_result(case, pipeline_response, luna_grade, config)


def run_dataset(
    dataset_name: str, args: argparse.Namespace, config: EvalConfig
) -> None:
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
                    results.append(CaseResult.model_validate_json(line))

    with open(raw_path, "a", encoding="utf-8") as append_fh:
        if args.batch:
            _run_batch(dataset_name, remaining, rag_engine, config, results, append_fh)
        else:
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
            continue
        results.append(case_result)
        append_fh.write(case_result.model_dump_json() + "\n")
        append_fh.flush()


def _run_batch(
    dataset_name: str,
    remaining: List[EvalCase],
    rag_engine,
    config: EvalConfig,
    results: List[CaseResult],
    append_fh,
) -> None:
    """Batch mode: FlynnMed's own pipeline call still runs synchronously per
    case (it's an in-process call, not an OpenAI batch-eligible endpoint) --
    only the primary (Luna) grading pass is submitted as one Batch API job.
    Terra adjudication (a much smaller, filtered subset) still runs
    synchronously afterward."""
    from evaluations.grading import (
        build_batch_requests,
        parse_batch_output,
        poll_batch,
        submit_batch,
    )

    pairs = []
    for index, case in enumerate(remaining, start=1):
        print(f"[{dataset_name}] pipeline ({index}/{len(remaining)}) {case.case_id}")
        try:
            pipeline_response = run_case_pipeline(case, rag_engine, config)
        except Exception as exc:
            print(
                f"[{dataset_name}] case {case.case_id} pipeline FAILED: {exc}",
                file=sys.stderr,
            )
            continue
        pairs.append((case, pipeline_response))

    if not pairs:
        return

    print(
        f"[{dataset_name}] submitting {len(pairs)} cases to the Batch API for primary grading..."
    )
    requests = build_batch_requests(
        pairs,
        config,
        model=config.primary_grader_model,
        reasoning_effort=config.grader_reasoning_effort,
    )
    batch_id = submit_batch(requests, config)
    print(
        f"[{dataset_name}] batch id: {batch_id} -- polling for completion (this can take a while; safe to Ctrl-C and --resume later)"
    )
    batch = poll_batch(batch_id, config)
    cases_by_id = {case.case_id: case for case, _ in pairs}
    luna_by_id = parse_batch_output(batch, config, cases_by_id)

    for case, pipeline_response in pairs:
        luna_grade = luna_by_id.get(case.case_id)
        if luna_grade is None:
            print(
                f"[{dataset_name}] case {case.case_id} batch grading FAILED or invalid -- skipping (rerun with --resume, without --batch, to retry synchronously)",
                file=sys.stderr,
            )
            continue
        try:
            case_result = finalize_case_result(
                case, pipeline_response, luna_grade, config
            )
        except Exception as exc:
            print(
                f"[{dataset_name}] case {case.case_id} adjudication FAILED: {exc}",
                file=sys.stderr,
            )
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
        "--force-download",
        action="store_true",
        help="Re-download the dataset even if a local copy exists.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use the OpenAI Batch API for the primary (Luna) grading pass.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    if args.sample is not None:
        config.sample_limit = args.sample

    if args.batch and not args.dry_run:
        print(
            "[runner] --batch mode affects only the primary grading pass; see evaluations/README.md for details."
        )

    dataset_names = (
        list(DATASET_URLS.keys()) if args.dataset == "all" else [args.dataset]
    )
    started = time.perf_counter()
    for dataset_name in dataset_names:
        run_dataset(dataset_name, args, config)
    print(f"[runner] done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
