# FlynnMed HealthBench and tiered RAG evaluation harness

**This is an automated benchmark evaluation, not clinical validation.**
Scores require qualified clinical review before they support safety or launch claims.

## What it evaluates

Each case runs through FlynnMed's real production pipeline
(`RAGEngine.handle_user_question`) and receives two complementary evaluations:

1. HealthBench rubric scoring, deterministic safety checks, and adjudication.
2. Tier 1-3 retrieval, grounding, correctness, calibration, and safety metrics.

For HealthBench, the configured primary model grades the captured FlynnMed answer directly
against the exact physician-authored rubrics. They are prohibited from drafting
their own response or comparing FlynnMed with an invented ideal answer. Each
rubric decision records evidence from the answer, and the harness rejects a
grade that rewrites, reorders, adds, or drops rubric criteria. The weighted
score is calculated locally from the dataset points.

The default evaluator is `gpt-5.4-mini`. When the primary and adjudicator are
the same model, the redundant second call is skipped to reduce latency and
cost; safety-trigger reasons are still recorded. **With the example `.env`
below, both default to `gpt-5.4-mini`, so adjudication is fully disabled** --
the runner prints a `WARNING` at startup whenever this is the case. Set
`EVAL_ADJUDICATOR_MODEL` to a genuinely different model (e.g. `gpt-4o-mini`)
if you want flagged cases to actually get an independent second opinion.

### Tier 1 - Core

1. Faithfulness / groundedness
2. Context relevance (retrieval precision)
3. Noise robustness (`1.0` means no contamination)
4. Context recall / coverage
5. Answer correctness against a gold answer
6. Calibration / appropriate hedging

### Tier 2 - Important

7. Contradiction / conflict handling
8. Claim-level citation accuracy
9. Context precision ranking (binary nDCG)

### Tier 3 - Periodic monitoring

10. Clinical harmlessness
11. Consistency / reproducibility

## Relevance-first RAG evaluation

The RAG judge operates in two stages:

1. Every displayed source excerpt receives a relevance score, rank, and
   relevant/irrelevant classification.
2. The answer is split into atomic material claims. Every answer quote,
   conversation quote, and source quote must be a verbatim substring of the
   captured data or the claim audit is rejected and retried.
3. Faithfulness and noise robustness are calculated locally from those
   validated claim/evidence relationships.
4. Citation accuracy evaluates only the claim carrying each citation. Citation
   completeness separately measures how many material clinical claims have an
   accurately supporting citation. Uncited advice can lower completeness but
   cannot make a different, correctly attached citation inaccurate.
5. Recall, correctness, calibration, conflict handling, harmlessness, and
   consistency are judged with the validated claim audit visible.

Scores use `1.0` as best and `0.0` as worst. A metric lacking required input is
`n/a` and excluded from its denominator rather than scored zero.
Aggregates below `EVAL_MINIMUM_RELIABLE_SAMPLE_SIZE` assessed items are labelled
`PROVISIONAL - insufficient sample`; their numeric value must not be presented
as a reliable system-wide conclusion.

## Configuration

```env
OPENAI_API_KEY=sk-...
# Optional: a separate project key with access to all evaluator models.
EVAL_API_KEY=sk-...

EVAL_GENERATOR_MODEL=gpt-4o-mini
EVAL_PRIMARY_GRADER_MODEL=gpt-5.4-mini
# Must differ from EVAL_PRIMARY_GRADER_MODEL or the second-opinion adjudicator
# call is skipped for every flagged case (a startup WARNING prints when they match).
EVAL_ADJUDICATOR_MODEL=gpt-5.4-mini
EVAL_RAG_METRICS_MODEL=gpt-5.4-mini
EVAL_FALLBACK_MODEL=gpt-5.4-mini
EVAL_ADJUDICATION_THRESHOLD=0.7
EVAL_DOCUMENT_RELEVANCE_THRESHOLD=0.6
EVAL_MINIMUM_RELIABLE_SAMPLE_SIZE=5
EVAL_GOLD_ANSWERS_PATH=evaluations/datasets/private/gold_answers.jsonl
EVAL_CONSISTENCY_REPEATS=0
EVAL_SAMPLE_LIMIT=
EVAL_OUTPUT_PATH=evaluations/results
EVAL_MAX_RETRIES=5
EVAL_REQUEST_TIMEOUT_SECONDS=120
```

If `EVAL_API_KEY` is unset, evaluators use `OPENAI_API_KEY`. Before any case
generation, the runner makes a small access-check request to each distinct
configured evaluator model. A permission error therefore stops the run before
retrieval and answer generation consume time or tokens. Never commit either key.

HealthBench grading and Tier metrics retry with `EVAL_FALLBACK_MODEL` when it
differs from the primary model. Raw results record the model that actually
produced each completed grade.

Consistency is disabled by default because each repeat is another complete
production-pipeline call.

## Running

```powershell
# Validate data and role detection without API calls
py -m evaluations.runner --dataset healthbench --dry-run

# Reproducible ten-case sample
py -m evaluations.runner --dataset healthbench --sample 10 --random-seed 20260714

# Periodic consistency run with two additional production calls per case
py -m evaluations.runner --dataset healthbench --sample 10 --consistency-repeats 2

# Resume a checkpointed run
py -m evaluations.runner --dataset healthbench --sample 10 --random-seed 20260714 --run-id my-run --resume

# Re-grade saved answers after an evaluator change; generation is not repeated
py -m evaluations.runner --dataset healthbench --run-id my-run --regrade-rag
```

Available datasets are `healthbench`, `healthbench_hard`, and
`healthbench_consensus`; use `--dataset all` for all three.

## Gold answers

Gold answers apply to the Tier 1 answer-correctness metric, not HealthBench
rubric scoring. A private clinician-reviewed answer overrides a dataset ideal
completion. Use one JSON object per line:

```json
{"case_id":"case-id","answer":"Clinician-reviewed answer","provenance":"clinical-panel-v1"}
```

See `evaluations/gold_answers.example.jsonl`. Keep real gold data under
`evaluations/datasets/private/`. Without private gold, a dataset ideal
completion is labelled `dataset_ideal_completion_not_clinician_validated`.

## Role-aware execution

Cases explicitly identifying the user as a doctor, nurse, midwife, or
physiotherapist run through a fresh one-case account with that role. Other
cases run anonymously in patient mode. Reports store the resolved role.

## Output

- Full raw results: `evaluations/results/raw/<run_id>/cases.jsonl`
- Sanitised JSON report: `evaluations/results/reports/<run_id>_summary.json`
- Sanitised Markdown report: `evaluations/results/reports/<run_id>_summary.md`

Reports include HealthBench weighted score, pass and triage signals,
primary/secondary adjudication statistics, Tier 1-3 aggregates, assessed-item
denominators and sufficiency labels, document/claim/citation audit counts,
judge-error counts, and cases requiring review.

## Limitations

The RAG judges see stored source excerpts, not complete publications, so an
"unsupported" result means only "not supported by the captured excerpt". It
must never be reported as proof that the full publication lacks the claim.
LLM-as-judge scores are evaluation signals, not clinical ground truth.
HealthBench weighted score measures compliance with the supplied rubrics; it is
not a measure of completeness outside those rubrics. Dataset ideal completions
are explicitly labelled as non-clinician-validated references.

## Testing

```powershell
py -m ruff format --check evaluations
py -m ruff check evaluations
py -m pytest evaluations/tests -q
```
