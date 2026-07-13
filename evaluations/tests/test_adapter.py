import json

import pytest

from evaluations.datasets.adapter import (
    AdapterError,
    adapt_healthbench_record,
    load_raw_jsonl,
    normalize_dataset,
)

_SAMPLE_RECORD = {
    "example_tags": [
        "theme:context_seeking",
        "physician_agreed_category:enough-context",
    ],
    "ideal_completions_data": None,
    "prompt": [
        {"content": "I have a mild headache, what should I do?", "role": "user"}
    ],
    "prompt_id": "abc-123",
    "rubrics": [
        {
            "criterion": "Advises rest and hydration.",
            "points": 5,
            "tags": ["axis:completeness"],
        },
        {
            "criterion": "Recommends unnecessary emergency care for a mild headache.",
            "points": -8,
            "tags": ["axis:accuracy"],
        },
    ],
}


def test_adapt_healthbench_record_preserves_identifiers_and_tags():
    case = adapt_healthbench_record(_SAMPLE_RECORD, source_dataset="healthbench")

    assert case.case_id == "abc-123"
    assert case.source_dataset == "healthbench"
    assert case.tags == [
        "theme:context_seeking",
        "physician_agreed_category:enough-context",
    ]
    assert case.raw == _SAMPLE_RECORD


def test_adapt_healthbench_record_preserves_negative_point_rubrics():
    case = adapt_healthbench_record(_SAMPLE_RECORD, source_dataset="healthbench")

    points = [r.points for r in case.rubrics]
    assert 5 in points
    assert -8 in points
    assert case.positive_points_total() == 5


def test_adapt_healthbench_record_maps_conversation():
    case = adapt_healthbench_record(_SAMPLE_RECORD, source_dataset="healthbench")

    assert len(case.conversation) == 1
    assert case.conversation[0].role == "user"
    assert case.last_user_turn().content == "I have a mild headache, what should I do?"
    assert case.history_turns() == []


def test_adapt_healthbench_record_extracts_ideal_completion_when_present():
    record = dict(_SAMPLE_RECORD)
    record["ideal_completions_data"] = {
        "ideal_completion": "Rest, hydrate, and monitor."
    }
    case = adapt_healthbench_record(record, source_dataset="healthbench")

    assert case.ideal_completion == "Rest, hydrate, and monitor."


def test_adapt_healthbench_record_requires_prompt_id():
    record = dict(_SAMPLE_RECORD)
    del record["prompt_id"]
    with pytest.raises(AdapterError):
        adapt_healthbench_record(record, source_dataset="healthbench")


def test_adapt_healthbench_record_requires_nonempty_prompt():
    record = dict(_SAMPLE_RECORD)
    record["prompt"] = []
    with pytest.raises(AdapterError):
        adapt_healthbench_record(record, source_dataset="healthbench")


def test_last_user_turn_rejects_non_user_ending_conversation():
    record = dict(_SAMPLE_RECORD)
    record["prompt"] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello, how can I help?"},
    ]
    case = adapt_healthbench_record(record, source_dataset="healthbench")
    with pytest.raises(ValueError):
        case.last_user_turn()


def test_load_raw_jsonl_reads_multiple_lines(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(_SAMPLE_RECORD) + "\n" + json.dumps(_SAMPLE_RECORD) + "\n",
        encoding="utf-8",
    )

    records = list(load_raw_jsonl(path))
    assert len(records) == 2


def test_load_raw_jsonl_raises_on_invalid_json(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")

    with pytest.raises(AdapterError):
        list(load_raw_jsonl(path))


def test_normalize_dataset_round_trips_through_jsonl(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(json.dumps(_SAMPLE_RECORD) + "\n", encoding="utf-8")
    out_path = tmp_path / "normalized.jsonl"

    count = normalize_dataset(raw_path, out_path, source_dataset="healthbench")

    assert count == 1
    assert out_path.exists()
    from evaluations.datasets.adapter import load_cases

    cases = load_cases(out_path)
    assert len(cases) == 1
    assert cases[0].case_id == "abc-123"
