# Contributing to FlynnMed

Thanks for your interest in contributing. This document covers how to set up
the project, the expectations for pull requests, and a few rules specific to
a codebase that generates clinical-facing content.

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
For security vulnerabilities, see [SECURITY.md](SECURITY.md) instead of
opening a public issue.

## Ways to contribute

- Bug reports and reproductions (see [Reporting bugs](#reporting-bugs))
- Fixes to retrieval, evidence-ranking, or policy-gate logic
- Frontend fixes/improvements to the React app
- Test coverage (backend `pytest`, frontend `vitest`)
- Documentation improvements

## Development setup

### Backend (Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pip install pytest ruff
```

Create a `.env` file with at least `OPENAI_API_KEY` set -- see the
[Environment Variables](README.md#environment-variables) section of the
README for the full list.

Run the backend:

```powershell
py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

### Frontend (Node 20+)

```powershell
cd frontend
npm install
npm run dev
```

The dev server proxies `/api/*` to `http://127.0.0.1:8000`, so keep the
backend running in another terminal.

## Running tests

```powershell
# Backend
py -m pytest backend/

# Frontend
cd frontend
npm test
```

Both suites run in CI (`.github/workflows/ci.yml`) on every pull request --
please make sure they pass locally first.

## Code style

- **Backend**: run `ruff check backend/` before committing. Follow existing
  patterns in the module you're editing rather than introducing a new style.
- **Frontend**: TypeScript `strict` mode is on (`frontend/tsconfig.json`).
  Run `npm run build` locally (it runs `tsc -b` before bundling) to catch
  type errors before pushing.
- Keep functions and modules focused; this codebase deliberately avoids
  hardcoded clinical term/keyword lists in favour of LLM-driven classification
  plus data-grounded deterministic checks (see `backend/clinical_context_guard.py`
  for the current pattern) -- if you're adding logic that maps a symptom or
  measurement name to a diagnosis or specialty, prefer deriving it from the
  patient's actual structured record over a hardcoded association.

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Keep PRs focused -- one fix or feature per PR is easier to review than a
   bundle of unrelated changes.
3. Add or update tests for the behavior you're changing.
4. Update `README.md` if you change environment variables, API endpoints, or
   setup steps.
5. Make sure `pytest`, `npm test`, and `npm run build` all pass.
6. Describe *why* the change is needed in the PR description, not just what
   changed -- especially for anything touching triage/safety logic.

### Changes to clinical safety logic

Changes to any of the following deserve extra care and a clear explanation of
the clinical reasoning in the PR description, since they affect what a patient
or clinician is told:

- `backend/policy_engine.py` (the 8 hard safety gates)
- `backend/clinical_decision_support.py` (presentation → triage mapping)
- `backend/intent_risk_classifier.py` (crisis pre-screen, risk classification)
- `backend/clinical_context_guard.py` (cross-specialty ambiguity resolution)
- `backend/pathways/*.py` (specialty-specific safety rules)

If you're unsure whether a change affects triage behavior, ask in the PR --
better to check than to silently change what an at-risk patient is told.

## Reporting bugs

Open a GitHub issue with:

- What you did (steps to reproduce)
- What you expected to happen
- What actually happened (include the exact error message/response if there
  is one)
- Whether it's reproducible with a fresh account/clean local database

For anything involving a security vulnerability, see
[SECURITY.md](SECURITY.md) instead -- don't open a public issue for those.
