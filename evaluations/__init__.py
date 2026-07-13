"""FlynnMed evaluation harness.

Runs FlynnMed's real production pipeline against HealthBench and produces an
automated benchmark report. This package does not change, import into, or
otherwise affect `backend/` or `frontend/` behaviour -- it only calls their
already-public interfaces (see `evaluations/pipeline.py`).

Results from this harness are an automated benchmark evaluation, not
clinical validation. See `evaluations/README.md`.
"""
