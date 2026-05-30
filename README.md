# Dr. Charlotte

Dr. Charlotte is now moving from a Streamlit UI to a mobile-first React app backed by a FastAPI API. It remains a health information assistant for people who want their health records, questions, measurements, symptoms, medicines and evidence in one signed-in workspace.

## React app on the production branch

The production branch keeps the existing Python clinical/retrieval modules and exposes them through `backend/api.py`. The React client lives in `frontend/` and calls `/api/*` for auth, chat, uploads, record tracking, PDF export, timeline data and clinical-trial search.

Run the full app from the repository root after installing Python requirements and frontend packages:

```powershell
pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. For frontend-only development, run `npm run dev` inside `frontend/`; Vite proxies `/api` to `http://127.0.0.1:8000`.

The previous Streamlit pages are still present as legacy reference code while the React migration settles.

It is built to be practical rather than flashy. A user can upload medical documents, ask questions, build a timeline, record symptoms and measurements, check medicines, export a GP-ready summary and find relevant clinical trials. The assistant uses personal context where it is available, but it still keeps answers traceable with public health and biomedical sources.

## Screenshots and flow

![Dr Charlotte workspace screen](image/image.png)

![Dr Charlotte health record screen](image/image%202.png)

![Dr Charlotte processing pipeline](image/pipeline.png)

## Repository architecture

The repo is organised as a layered Streamlit application. `Home.py` is the entry point, `pages/` contains the user-facing screens, `app_ui/` holds shared interface helpers and styling, and `backend/` contains the clinical workflow, retrieval, extraction, persistence and export services.

```mermaid
flowchart TD
    User[Signed-in user] --> Home[Home.py]
    Home --> Landing[pages/1_Landing.py]
    Home --> Workspace[pages/2_Workspace.py]
    Workspace --> Chat[pages/2_Chatbot.py]
    Workspace --> Timeline[pages/3_Health_Timeline.py]
    Workspace --> Trials[pages/4_Find_Clinical_Trials.py]

    Chat --> UploadUI[app_ui/uploader.py]
    Chat --> RAG[backend/rag_system.py]
    Timeline --> Store[backend/user_store.py]
    Trials --> TrialSearch[backend/clinical_trials.py]

    UploadUI --> PDFText[backend/utils.py]
    UploadUI --> Store
    RAG --> Extractor[backend/document_extractor.py]
    RAG --> Orchestrator[backend/clinical_orchestrator.py]
    RAG --> Memory[backend/memory_store.py]
    RAG --> GP[backend/gp_summary.py]
    RAG --> Media[image, video and voice services]

    Orchestrator --> Safety[moderation, policy, intent and role routing]
    Orchestrator --> Evidence[NHS, MedlinePlus, Europe PMC, PubMed Central and openFDA]
    Orchestrator --> LLM[backend/summarizer.py]

    Store --> Local[users.json and data/uploads]
    Store --> Postgres[PostgreSQL when DATABASE_URL is set]
```

The important point is that the Streamlit pages stay mostly concerned with interface state and display. The backend modules hold the health logic, document extraction, evidence retrieval and data persistence.

## Main runtime flows

### Account and consent flow

1. `Home.py` routes signed-out users to `pages/1_Landing.py` and signed-in users to `pages/2_Workspace.py`.
2. `pages/1_Landing.py` handles sign-in, account creation, role selection, full-name capture, date of birth, biological sex, role terms and privacy acceptance.
3. `backend/user_store.py` validates and stores the account. It hashes passwords with a salt and normalises the user record so older accounts still receive newer fields.
4. The selected role is later resolved by `backend/role_router.py` so the assistant can answer differently for patients, caregivers and clinical users.

### Document upload flow

1. The chat page calls `app_ui/uploader.py`.
2. Uploaded PDFs are saved under `data/uploads/<username>/`.
3. The uploader checks the filename and the first part of the extracted PDF text for patient-name candidates.
4. The detected patient name is compared with the full account name saved during sign-up.
5. If the name matches, the file is returned to the chat page for processing.
6. If the name is missing, unclear or mismatched, the file is held in a pending verification state and the user must choose whether to process it.
7. `backend/rag_system.py` ingests the approved files, reads text with `backend/utils.py`, anonymises it, summarises it and saves the document summary.
8. `backend/document_extractor.py` extracts structured health data from the document and `backend/user_store.py` saves it into the correct collections.

### Chat answer flow

1. `pages/2_Chatbot.py` sends the user's question to `backend/rag_system.RAGEngine`.
2. `RAGEngine` restores the user's saved documents, symptoms, medicines, allergies, conditions, vitals and longitudinal memory.
3. `backend/context_graph.py` builds fast patient-specific search hints from the account history.
4. `backend/clinical_orchestrator.py` resolves the user role, screens crisis wording, runs moderation, classifies intent, applies policy gates and chooses the right pathway.
5. The orchestrator expands the question and retrieves live evidence from official guidance and biomedical sources.
6. `backend/memory_store.py` performs embedding search over personal context and retrieved content.
7. `backend/evidence_ranker.py` de-duplicates and ranks sources by relevance and evidence tier.
8. `backend/summarizer.py` asks the model to produce a cited, role-aware answer from the evidence dossier and patient context.
9. `backend/triage_summary.py` creates a structured triage card, and the final trace is saved back to the user record.

### Clinical trial flow

1. `pages/4_Find_Clinical_Trials.py` builds a trial-search profile from the saved account data.
2. `backend/clinical_trials.py` uses an LLM to extract separate condition, symptom and medicine terms from the user's health context.
3. The module searches ClinicalTrials.gov once per term and merges results by NCT ID.
4. A fast deterministic score narrows the candidate list.
5. An LLM then scores clinical alignment, while deterministic checks report age and sex eligibility as included, excluded or unknown.
6. The ranked result is saved by `UserStore` so it can be shown again after a restart.

### Timeline and summary flow

1. `pages/3_Health_Timeline.py` reads symptoms, measurements, medicines, allergies, conditions, uploads and triage summaries from `UserStore`.
2. It builds major events, trend cards and chartable series from the saved records.
3. The chat page can call `RAGEngine.build_summary_pdf_for_user`.
4. `backend/gp_summary.py` creates a concise PDF handover from the user's saved profile, documents, trackers, memory and recent triage summaries.

## What it can do

- Create an account, sign back in and keep a saved workspace.
- Store chat history, uploaded records, health trackers, audit events and clinical trial results per user.
- Upload PDFs such as blood results, discharge summaries, clinic letters and medical records.
- Check whether the patient name found in an uploaded document matches the full name saved on the account before processing.
- Ask the user to verify the document before extraction if the name is missing, unclear or does not match.
- Extract health information from uploaded documents and save it into the right places, including measurements, allergies, medicines, conditions, height, weight, blood pressure and lab values where they are present.
- Let users type a custom measurement type as well as choose from the suggested list.
- Log symptoms over time with dates, severity, triggers and notes.
- Keep a medication list and check openFDA label sections for possible interaction warnings.
- Build a health timeline across symptoms, measurements, triage summaries, medicines and records.
- Search live evidence from NHS guidance, MedlinePlus, Europe PMC, PubMed Central and openFDA.
- Generate structured triage summaries with urgency, next steps and monitoring points.
- Export a short GP handover PDF using the account's saved data.
- Search ClinicalTrials.gov for recruiting trials that fit the user's saved health profile.
- Accept voice input through Whisper transcription when the browser supports recording.
- Generate clinical-style educational images and short demonstration videos when explicitly requested.

## Document uploads and auto-population

The document upload flow is designed to stop records being placed into the wrong account.

When a PDF is uploaded, the app first looks for a patient name in the document text and in the filename. It compares that name with the full name saved during account creation. If the match is clear, the file can be processed. If the name is missing or does not match, the user sees a warning and must verify the document before the app extracts data from it.

After verification, the document text is extracted, anonymised where possible, summarised and added to the user's retrieval context. The app then extracts structured health data and saves it to the relevant trackers. Long documents are processed in smaller chunks so more values can be captured from large records.

The extractor can populate:

- measurements and observations, such as blood pressure, pulse, temperature, height and weight
- lab results, such as HbA1c, eGFR, haemoglobin and cholesterol
- medicines, doses and schedules where they are stated
- allergies and contraindications
- conditions, diagnoses and problem list entries
- dates, notes and document references where they are available

Users should still review auto-populated entries. Medical PDFs vary a lot, and no extraction system should be treated as perfect.

## Backend design in detail

### `backend/user_store.py`

`UserStore` is the persistence boundary for the app. The rest of the code should not need to know whether data is stored in `users.json` or PostgreSQL.

It stores accounts, profiles, uploads, document summaries, chat history, symptoms, medicines, allergies, conditions, measurements, triage summaries, trial search results, audit events, interaction traces and longitudinal memory. The module also normalises records on load, which keeps old saved users compatible when new health fields are added.

### `backend/rag_system.py`

`RAGEngine` is the main service used by the chat page. It restores user context, ingests verified documents, refreshes longitudinal memory, prepares answer bundles, streams answer events, saves traces, builds medication checks and creates GP summary PDFs.

It also coordinates optional media generation. Image generation is handled by `backend/image_generator.py`, video generation by `backend/video_generator.py`, and voice transcription is called from the UI through `backend/voice_transcriber.py`.

### `backend/clinical_orchestrator.py`

The orchestrator is the clinical workflow centre of the app. It keeps the answer pipeline consistent by coordinating:

- role resolution through `backend/role_router.py`
- intent and risk classification through `backend/intent_risk_classifier.py`
- deterministic clinical decision support through `backend/clinical_decision_support.py`
- safety policy checks through `backend/policy_engine.py`
- pathway context from `backend/pathways/`
- official guidance retrieval and biomedical retrieval
- evidence ranking through `backend/evidence_ranker.py`

This means the app does not simply send the user's question straight to a model. It builds a governed bundle first, then the language model writes from that bundle.

### `backend/document_extractor.py`

The extractor reads uploaded document text in chunks of up to 5,000 characters with overlap. Each chunk is sent for structured extraction, then the results are merged and de-duplicated.

The extraction schema covers vitals, lab results, medicines, allergies and conditions. It uses preferred keys for common values, but it can also create a concise custom measurement key when a result is present in the document and is not in the built-in list.

### `backend/memory_store.py`

`MemoryStore` keeps an in-memory embedding index for retrieved biomedical content and user-specific summaries. It uses OpenAI embeddings, caches repeated text embeddings and filters search results by user where needed.

The durable memory still lives in `UserStore`. `MemoryStore` is the searchable runtime layer that helps the assistant find the most relevant personal context and evidence for the current question.

### Retrieval modules

- `backend/official_guidance.py` searches public guidance sources such as NHS and MedlinePlus.
- `backend/pubmed_search.py` retrieves biomedical literature from Europe PMC and PubMed Central.
- `backend/medication_checker.py` checks openFDA drug label sections for medicine interaction evidence.
- `backend/query_expander.py` turns user questions into more retrieval-friendly search variants.
- `backend/evidence_ranker.py` ranks and labels the final source set before it reaches the answer model.

### Safety and governance modules

- `backend/moderation_ml.py` combines rule-based moderation with Detoxify support when available.
- `backend/policy_engine.py` applies extra caution for higher-risk categories such as pregnancy, children, elderly polypharmacy, medicine dosing, diagnosis-seeking and mental health concerns.
- `backend/triage_summary.py` normalises the triage card so the final answer cannot lower the safety level below the fallback route.
- `backend/audit_models.py` defines structured trace records for review.
- `backend/response_templates.py` holds role-specific personas, headings and safety wording.

## Frontend structure

The app uses Streamlit's multipage structure.

- `Home.py` is the small router that sends users to the correct first page.
- `pages/1_Landing.py` handles sign-in, sign-up, consent and the role-specific terms.
- `pages/2_Workspace.py` gives the signed-in overview and links into the main workflows.
- `pages/2_Chatbot.py` is the central workspace for chat, uploads, trackers, GP summary export, audit export, feedback and voice input.
- `pages/3_Health_Timeline.py` turns saved records into a timeline, trend summaries and charts.
- `pages/4_Find_Clinical_Trials.py` searches and renders matched recruiting clinical trials.
- `app_ui/theme.py` injects shared CSS and formats timestamps.
- `app_ui/uploader.py` owns the PDF upload UI, name verification and pending document review state.
- `app_ui/static/styles.css` contains the shared design system for the pages.

## How answers are produced

1. The user signs in and enters the workspace.
2. Saved profile data, health trackers, uploaded document summaries and longitudinal memory are restored.
3. New uploads are checked against the user's full account name before processing.
4. Verified PDFs are parsed with PyMuPDF, summarised and used as personal context.
5. The app extracts structured data from the PDF and updates the relevant health trackers.
6. Each question is checked for crisis language and moderated before retrieval.
7. The question is classified for intent and routed through the right pathway, such as general triage, maternity, musculoskeletal, medication or chronic condition support.
8. The app expands the question into search-friendly terms.
9. It retrieves official guidance, biomedical literature and medicine label data where relevant.
10. Retrieved sources and personal context are ranked with OpenAI embeddings.
11. The model writes a cited answer with a structured triage summary.
12. The answer, sources, trace, triage summary and refreshed memory are saved back to the user account.

If the app cannot retrieve enough reliable evidence, it gives a more limited answer instead of pretending to know more than it does.

## Clinical trial finder

The trial finder searches ClinicalTrials.gov for recruiting studies that may fit the user's saved profile.

It extracts conditions and medicine terms from the user's longitudinal health context, searches for each term separately, merges duplicate trials and ranks the strongest candidates. The final score considers condition alignment, whether a trial covers more than one relevant issue and whether the location is practical.

The page also keeps the most recent search result for the user, so the result can survive restarts and hosted deployments when PostgreSQL is configured.

## Workspace experience

The main workspace is built for repeat use.

- The sidebar handles uploads, symptoms, medicines, measurements, GP summaries and audit export.
- The chat keeps the answer, sources, trace and safety summary together.
- The health timeline gives a chronological view across the account.
- The tracker pages let users correct, remove or add health information manually.
- The document upload panel explains when a file needs name verification before extraction.

## Roles, pathways and safety

Dr. Charlotte adapts its style and safety thresholds based on the selected role. Patients and caregivers get plainer language and clearer escalation advice. Clinicians get more structured clinical framing, while still seeing safety prompts where appropriate.

The app includes:

- crisis phrase screening
- moderation before retrieval
- pregnancy, paediatric, elderly polypharmacy, medicine dosing, diagnosis-seeking and mental health safety checks
- role-aware response templates
- structured triage cards with a minimum safety floor
- audit traces for later review

The safety framing is written with UK care routes in mind.

## Data saved per user

Each account can store:

- profile details and consent choices
- chat history and longitudinal memory
- uploaded document summaries and extracted text
- symptom entries
- medicine entries
- allergy entries
- condition entries
- measurements and lab results
- structured triage summaries
- GP handover content
- clinical trial search results
- audit events and interaction traces

By default this data is stored locally in `users.json`, with uploaded PDFs under `data/uploads/<username>/`. When `DATABASE_URL` is set, account data is stored in PostgreSQL instead.

Passwords are stored as salted bcrypt hashes.

## Tech stack

- Frontend: Streamlit
- Answer generation, extraction and trial scoring: OpenAI Chat Completions
- Embeddings: OpenAI `text-embedding-3-small`
- Voice transcription: OpenAI Whisper
- Image generation: OpenAI `gpt-image-1`
- Video generation: OpenAI `sora-2`
- Official guidance retrieval: NHS and MedlinePlus
- Biomedical literature retrieval: Europe PMC and PubMed Central
- Medicine interaction support: openFDA drug label API
- Clinical trial search: ClinicalTrials.gov API v2
- PDF parsing and GP summary export: PyMuPDF
- Persistence: local JSON or PostgreSQL
- Moderation: rule-based checks with Detoxify support when available

## Requirements

- Python 3.11 or 3.12
- An OpenAI API key
- Optional `DATABASE_URL` for PostgreSQL-backed persistence

Python 3.14 is not recommended for this project because some optional NLP packages can be awkward there.

## Quick Start

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Add your settings to `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=
```

Start the app:

```powershell
py -3.12 -m streamlit run Home.py
```

If PowerShell blocks activation, use this for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## PostgreSQL and hosted use

For local development, `users.json` is usually enough. For hosted or shared use, configure PostgreSQL so data does not disappear between deployments.

1. Create a Neon database or another PostgreSQL database.
2. Add `DATABASE_URL` to the environment or Streamlit secrets.
3. Restart the app.

Example Streamlit secrets file:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
DATABASE_URL = "postgresql://username:password@host:5432/database?sslmode=require"
```

## Project structure

```text
app_ui/
  static/styles.css          app-wide styling
  theme.py                   CSS injection and timestamp helpers
  uploader.py                document upload and verification UI

backend/
  anonymizer.py              document redaction helpers
  clinical_orchestrator.py   main clinical workflow engine
  clinical_trials.py         ClinicalTrials.gov search and scoring
  context_graph.py           health context graph helpers
  document_extractor.py      structured extraction from uploaded documents
  evidence_ranker.py         source ranking and evidence tiers
  gp_summary.py              GP handover PDF generation
  image_generator.py         image generation integration
  medication_checker.py      openFDA interaction checks
  memory_store.py            longitudinal memory refresh
  official_guidance.py       NHS and MedlinePlus retrieval
  patient_history.py         account health context builder
  policy_engine.py           safety policy checks
  pubmed_search.py           Europe PMC and PubMed Central retrieval
  rag_system.py              retrieval, generation and ingestion engine
  role_router.py             role-aware routing
  summarizer.py              document summarisation
  symptom_tracker.py         symptom helpers
  triage_summary.py          structured triage cards
  user_store.py              accounts, profiles and persistence
  video_generator.py         video generation integration
  voice_transcriber.py       Whisper transcription

pages/
  1_Landing.py               consent, sign in and sign up
  2_Chatbot.py               main chat workspace
  2_Workspace.py             health trackers overview
  3_Health_Timeline.py       chronological health timeline
  4_Find_Clinical_Trials.py  clinical trial finder

Home.py                      app entry point
requirements.txt             Python dependencies
```

## Troubleshooting

### `OPENAI_API_KEY not found in environment variables`

Create `.env` from `.env.example`, add a real key and start Streamlit from the project root.

### Accounts or trial results disappear on a hosted deployment

Set `DATABASE_URL` so the app uses PostgreSQL instead of the local `users.json` file.

### A PDF says the name cannot be found

Make sure the full name saved on the account is the patient's real full name. The app checks the document text and filename. If the document is correct but the name cannot be detected, verify it in the upload panel before extraction.

### Auto-populated data from a PDF looks wrong

Review the entries in the trackers and remove anything incorrect. The extractor reads free-text documents and may misread a value, date or unit.

### Clinical trial search returns no results

The trial finder needs saved health context such as conditions, symptoms, medicines or uploaded documents. Also check that the host can make outbound HTTPS requests to ClinicalTrials.gov.

### Voice input is unavailable

Make sure the browser allows microphone access. The app depends on Streamlit's browser audio support.

### Video generation does not appear

Check that the OpenAI account has access to the configured video model. The app skips video output if the model call fails.

### Medication warnings do not appear

The interaction checker uses public openFDA label sections. If a medicine name cannot be resolved, or if the label does not mention the paired medicine, the app may not show a pair-specific warning. A pharmacist or clinician should still review any medicine concern.

## Important note

Dr. Charlotte is for health education, evidence review and decision support. It is not a substitute for emergency care, diagnosis or a clinician's judgement.

If someone may be seriously unwell, use the right urgent care route, such as NHS 111 or 999 in the UK.
