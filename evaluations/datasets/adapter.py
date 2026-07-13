"""Adapts raw HealthBench JSONL records into the harness's internal EvalCase
format.

Verified real HealthBench schema (fetched directly from the source URLs; see
evaluations/README.md): each line is a JSON object with

    {
        "example_tags": [str, ...],
        "ideal_completions_data": null | {"ideal_completion": str, ...},
        "prompt": [{"role": str, "content": str}, ...],
        "prompt_id": str,
        "rubrics": [{"criterion": str, "points": float, "tags": [str, ...]}, ...]
    }

This is identical across the base, Hard, and Consensus files (same
generating pipeline) -- confirmed by sampling all three during development.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from evaluations.config import NORMALIZED_DIR
from evaluations.models import ConversationTurn, EvalCase, RubricItem


class AdapterError(ValueError):
    pass


def load_raw_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise AdapterError(
                    f"{path}:{line_number}: invalid JSON ({exc})"
                ) from exc


def adapt_healthbench_record(record: Dict[str, Any], source_dataset: str) -> EvalCase:
    prompt_id = record.get("prompt_id")
    if not prompt_id:
        raise AdapterError(f"Record missing required 'prompt_id': {record!r}")

    raw_prompt = record.get("prompt") or []
    if not raw_prompt:
        raise AdapterError(f"Case {prompt_id} has an empty 'prompt' (conversation).")
    conversation = [
        ConversationTurn(role=turn.get("role", "user"), content=turn.get("content", ""))
        for turn in raw_prompt
    ]

    raw_rubrics = record.get("rubrics") or []
    rubrics = [
        RubricItem(
            criterion=item.get("criterion", ""),
            points=float(item.get("points", 0)),
            tags=list(item.get("tags") or []),
        )
        for item in raw_rubrics
    ]

    ideal_data = record.get("ideal_completions_data")
    ideal_completion = None
    if isinstance(ideal_data, dict):
        ideal_completion = ideal_data.get("ideal_completion")

    return EvalCase(
        case_id=str(prompt_id),
        source_dataset=source_dataset,
        conversation=conversation,
        rubrics=rubrics,
        tags=list(record.get("example_tags") or []),
        ideal_completion=ideal_completion,
        raw=record,
    )


def normalize_dataset(input_path: Path, output_path: Path, source_dataset: str) -> int:
    """Reads a raw HealthBench JSONL file and writes adapted EvalCase JSONL
    to `output_path`. Returns the number of cases written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for record in load_raw_jsonl(input_path):
            case = adapt_healthbench_record(record, source_dataset=source_dataset)
            out.write(case.model_dump_json() + "\n")
            count += 1
    return count


def load_cases(path: Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase.model_validate_json(line))
    return cases


def normalized_path(source_dataset: str, dest_dir: Path = NORMALIZED_DIR) -> Path:
    return dest_dir / f"{source_dataset}.jsonl"
