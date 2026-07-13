"""Thin wrapper connecting the evaluation harness to FlynnMed's real,
production end-to-end pipeline -- nothing here reimplements clinical logic.

Every case is run through `backend.rag_system.RAGEngine.handle_user_question`,
the exact same call the chat API uses, with `user=None` (anonymous): this
exercises the full pipeline (role routing, patient-history context, crisis
pre-screen, moderation, intent/risk classification, policy gate, pathway
context, agentic evidence retrieval, and response synthesis) without ever
touching a real account or persisting anything (`UserStore` writes in
`_finalize_answer_payload`/`_enrich_prebuilt_payload` are all guarded on a
truthy `user` and are skipped for `user=None` -- verified by reading
backend/rag_system.py).
"""

from __future__ import annotations

import time

from evaluations.config import EvalConfig
from evaluations.models import EvalCase, PipelineResponse


def build_rag_engine(config: EvalConfig):
    """Constructs one RAGEngine for the whole run (expensive to build -- do
    not construct per-case). Sets the response-generation model FlynnMed's
    LLMHelper actually uses for the primary answer.

    Note: `LLMHelper.ANSWER_MODEL` is a hardcoded class constant in
    `backend/summarizer.py` (not read from the `OPENAI_MODEL` environment
    variable at call time), so setting `OPENAI_MODEL` alone would not affect
    the real answer-generation call. This overrides that class attribute for
    the lifetime of this standalone evaluation process only -- it never
    touches the live app process and does not edit any source file.
    """
    from backend.rag_system import RAGEngine
    from backend.summarizer import LLMHelper

    LLMHelper.ANSWER_MODEL = config.generator_model

    return RAGEngine()


def run_case(rag_engine, case: EvalCase) -> PipelineResponse:
    """Runs one EvalCase through the real pipeline and captures the result.

    Raises whatever RAGEngine.handle_user_question raises -- the runner is
    responsible for retry/error handling (see evaluations/grading.py's
    `call_with_retry`, reused for this call too).
    """
    question_turn = case.last_user_turn()
    chat_history = [
        {"role": turn.role, "content": turn.content} for turn in case.history_turns()
    ]

    started = time.perf_counter()
    payload = rag_engine.handle_user_question(
        question=question_turn.content,
        chat_history=chat_history or None,
        stream=False,
        user=None,
    )
    duration = time.perf_counter() - started

    trace = payload.get("trace", {}) or {}
    return PipelineResponse(
        case_id=case.case_id,
        answer_markdown=payload.get("answer_markdown", ""),
        answer_text=payload.get("answer_text", payload.get("answer_markdown", "")),
        sources=payload.get("sources", []) or [],
        personal_context=payload.get("personal_context", []) or [],
        trace=trace,
        full_payload=payload,
        duration_seconds=duration,
    )
