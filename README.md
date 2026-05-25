# Dr. Charlotte

Dr. Charlotte is a Streamlit-based health information assistant with a signed-in workspace, personal document context, live evidence retrieval, traceable sources, and safety guardrails. It is designed for individuals, caregivers, and clinical users who want answers that are readable, inspectable, and consistent across sessions.

This repository is more than a simple chatbot demo. It includes account persistence, document upload and anonymisation, auto-extraction of health data from uploaded records, longitudinal memory, a symptom and vitals timeline, medication safety checks, structured triage output, GP-ready PDF export, a clinical trial finder, audit export, voice input, and a role-aware clinical orchestration layer.

## High-impact features

- **Clinical trial finder**: searches ClinicalTrials.gov using your saved health profile, runs separate API queries per condition, and ranks results using LLM-assessed condition alignment, multi-condition coverage, and location proximity
- **Auto-population from uploaded records**: when a PDF is uploaded, the app uses an LLM to extract vitals, lab results, medications, allergies, and conditions and saves them directly into your health trackers
- **Health timeline**: a visual timeline of symptom logs, vitals, triage summaries, and medication records across your full account history
- **Symptom timeline tracker**: log symptoms over time with dates, severity, triggers, and notes
- **Medication interaction checker**: the app keeps a medication list and checks openFDA label interaction sections for concerning combinations
- **Structured triage output**: each answer includes a scannable urgency level, suggested next step, and monitoring points
- **GP-ready PDF export**: download a one-page summary with tracked symptoms, medications, uploaded records, longitudinal memory, and the latest triage summary

## What the app can do

- Create accounts and sign back in later with the same saved workspace
- Show a UK privacy notice and require explicit per-role consent before access
- Support different user roles: patient, caregiver, doctor, nurse, midwife, physiotherapist, and other clinician
- Save chat history, profile settings, uploads, audit events, and interaction traces per user
- Let users clear chat history without deleting the whole account
- Upload PDF records such as discharge letters, lab reports, and specialist notes
- Extract text from uploaded PDFs, anonymise sensitive information, and summarise the document into retrieval-ready context
- Automatically extract vitals, lab results, medications, allergies, and conditions from uploaded PDFs and save them to the relevant trackers
- Build and refresh a longitudinal patient memory from uploaded records and ongoing conversation
- Log symptom timelines with dates, severity, triggers, and notes
- Detect repeat symptom patterns from tracker entries and feed them into longitudinal memory
- Save a structured medication list and flag label-based interaction warnings from openFDA
- Log vitals and lab results (blood pressure, HbA1c, eGFR, haemoglobin, and many more) with dates and units
- Find recruiting clinical trials on ClinicalTrials.gov matched to your health profile and country, with LLM-scored condition alignment, age and sex eligibility verdicts, and a detailed match breakdown
- Persist the last clinical trial search result so it survives app restarts and redeployments
- Generate a structured triage card after each answer with urgency, next step, and what to monitor
- Export a one-page GP summary PDF covering symptoms, medications, records, and the AI summary
- Search live public sources including NHS guidance, MedlinePlus, Europe PMC / PubMed Central, and openFDA drug labels
- Expand user questions into retrieval-friendly search queries
- Rank evidence with OpenAI embeddings and label sources by evidence tier
- Stream answers into the chat interface as they are generated
- Show clickable citations, source trace panels, personal context used, and audit metadata for each response
- Export a user snapshot as JSON for audit and review
- Accept spoken questions through Whisper transcription
- Generate clinical-style illustrations on request with `gpt-image-1`
- Generate short clinical demonstration videos on request with `sora-2`, with a one-video-per-hour limit per user
- Apply moderation, crisis detection, policy gates, and pathway-specific safety rules before the final answer is produced

## How the answer pipeline works

1. A user signs in and enters the workspace.
2. Uploaded PDFs are saved into that user's own folder.
3. PDF text is extracted with PyMuPDF and anonymised before it is summarised or indexed.
4. For new uploads, an LLM extracts structured health data (vitals, medications, allergies, conditions) and saves it to the relevant trackers automatically.
5. The system restores saved document summaries and longitudinal memory for that user.
6. Each new question is checked for crisis language and passed through moderation.
7. The question is classified for intent and risk, then routed through a relevant pathway: general triage, maternity, musculoskeletal, medication, or chronic condition support.
8. The app restores symptom tracker patterns and saved medications into the user's working memory.
9. The app expands the question into search-friendly variants.
10. It retrieves live official guidance from NHS and MedlinePlus, open-access biomedical evidence from Europe PMC / PubMed Central, and medication label interaction data from openFDA when relevant.
11. Retrieved material and personal context are ranked semantically with OpenAI embeddings.
12. Sources are deduplicated, tiered, and passed to the LLM for a cited answer.
13. The app generates a structured triage summary and stores it with the trace.
14. The final answer, trace, triage summary, and refreshed longitudinal memory are saved back to the user account.

If the system cannot retrieve enough reliable evidence, it falls back to a more limited response rather than pretending to know more than it does.

## Clinical trial finder

The Find Clinical Trials page searches ClinicalTrials.gov for recruiting studies that match the user's saved health profile.

1. The app uses an LLM to extract individual condition and medication terms from the user's full longitudinal health context.
2. It runs a separate API query to ClinicalTrials.gov for each extracted term.
3. Results from all searches are merged by trial ID. Trials that appear in more searches score higher.
4. The top 20 candidates are scored with a second LLM call that assesses clinical relevance against the full patient profile.
5. Each trial is given a deterministic age and sex eligibility verdict (INCLUDED / EXCLUDED / UNKNOWN) based on the trial's stated criteria.
6. Trials are ranked by a score out of 100: condition alignment (up to 50), multi-condition coverage (up to 30), and location proximity (up to 20).
7. Contact details are shown for action but do not affect the score.
8. The most recent search result is persisted per user so it survives app restarts and redeployments — including on Neon PostgreSQL in production.

## Workspace experience

The main chat workspace is designed around longitudinal use, not one-off prompts.

- The sidebar lets users upload records, log symptoms, manage medications, download a GP summary PDF, and export an audit snapshot
- Assistant responses surface a structured triage card before the source trace
- Medication questions can trigger a dedicated interaction panel with openFDA-backed label evidence links
- Symptom tracker data is folded into longitudinal memory so the assistant can reference recurring patterns over time
- The health timeline page gives a chronological view across all tracker types

## Roles, pathways, and safety

The app does not answer every user in the same voice. It adapts response style, evidence framing, and escalation thresholds based on the selected role.

- Patients and caregivers get plainer language and stronger escalation nudges
- Doctors, nurses, midwives, and physiotherapists get more clinically structured responses with clearer disposition and initial management steps
- Crisis patterns are screened early so emergency guidance can be returned immediately
- Policy gates add extra caution for pregnancy, paediatrics, elderly polypharmacy, medication dosing, diagnosis-seeking questions, and mental health topics
- Structured triage is normalised against a fallback safety floor so the final card cannot de-escalate below the minimum safe route
- Moderation runs before retrieval so unsafe requests can be blocked early
- AI-generated images and videos are only triggered by explicit user requests and are filtered away from unsafe visual topics

The pathway and policy layers are written with UK clinical safety framing in mind.

## Data captured per user

Each user account persists:

- chat history and longitudinal memory
- uploaded document summaries and extracted text
- symptom tracker entries
- medication list entries
- allergy entries
- vitals and lab result entries (auto-populated from uploaded PDFs where available)
- saved structured triage summaries
- generated GP handover content
- clinical trial search results (last search persisted until the user runs a new one)
- audit events and interaction traces

## Media features

### Voice input

Users can record a question in the chat page and have it transcribed with OpenAI Whisper. The app stores the resulting text in the conversation, not the raw audio file.

### Illustrations

If a user explicitly asks to see something visually, the app can generate a clinical-style educational illustration with `gpt-image-1`.

### Short videos

If a user explicitly asks for a video or animation, the app can attempt to generate a short demonstration clip with `sora-2`. Video generation is rate-limited to once per hour per user.

## Data and persistence

By default, the app stores user data locally:

- `users.json` holds accounts, hashed passwords, profiles, chat history, document summaries, symptom logs, medication lists, allergy entries, vitals, triage summaries, traces, audit events, longitudinal memory, and the last clinical trial search result
- `data/uploads/<username>/` holds uploaded PDFs for each user

If `DATABASE_URL` is set, the app switches from the local JSON store to PostgreSQL (Neon or any compatible host) for all account persistence. That is the correct option for hosted or shared deployments — all user data including trial search results is stored in Neon automatically.

Passwords are stored as salted bcrypt hashes, not in plain text.

## Tech stack

- Frontend: Streamlit
- Answer generation, extraction, and trial scoring: OpenAI Chat Completions (`gpt-4o-mini` by default)
- Embeddings: OpenAI `text-embedding-3-small`
- Voice transcription: OpenAI Whisper
- Image generation: OpenAI `gpt-image-1`
- Video generation: OpenAI `sora-2`
- Official guidance retrieval: NHS and MedlinePlus
- Biomedical literature retrieval: Europe PMC / PubMed Central
- Medication interaction support: openFDA drug label API
- Clinical trial search: ClinicalTrials.gov API v2
- PDF parsing: PyMuPDF
- GP summary export: PyMuPDF-generated single-page PDF
- Persistence: local JSON or PostgreSQL (Neon)
- Moderation: rules-based checks plus Detoxify support when available

## Requirements

- Python 3.11 or 3.12 is recommended
- An OpenAI API key
- Optional `DATABASE_URL` for PostgreSQL-backed persistence (required for production deployments)

Python 3.14 is not recommended because some optional NLP tooling can be awkward there.

## Quick start

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

Then start the app:

```powershell
py -3.12 -m streamlit run Home.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Running with PostgreSQL (Neon)

For local development, `users.json` is fine. For anything hosted or shared, use PostgreSQL.

1. Create a Neon (or other PostgreSQL) database.
2. Add `DATABASE_URL` to your environment or Streamlit secrets.
3. Restart the app — all user data, including trial search results, routes to PostgreSQL automatically.

Example Streamlit secrets file (`secrets.toml`):

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
DATABASE_URL = "postgresql://username:password@host:5432/database?sslmode=require"
```

## Project structure

```text
app_ui/
  static/styles.css       custom CSS for all pages
  theme.py                CSS injection and timestamp helpers
  uploader.py             document upload UI component
backend/
  anonymizer.py           spaCy NER + regex PII redaction for uploaded docs
  audit_models.py         ClinicalAuditTrace and PolicyGateRecord dataclasses
  clinical_orchestrator.py  central clinical workflow engine
  clinical_trials.py      ClinicalTrials.gov multi-term search and LLM scoring
  document_extractor.py   LLM extraction of vitals/medications/allergies from PDFs
  evidence_ranker.py      TieredSource evidence ranking (NHS/NICE → reviews → research)
  feedback_store.py       anonymised feedback persistence
  gp_summary.py           GP handover PDF generation
  image_generator.py      gpt-image-1 integration
  intent_risk_classifier.py  intent and risk level classification
  medication_checker.py   openFDA drug label interaction checker
  memory_store.py         longitudinal memory building and refresh
  moderation_ml.py        Detoxify + regex moderation (role-adaptive)
  official_guidance.py    NHS and MedlinePlus live retrieval
  patient_history.py      patient history context builder
  policy_engine.py        8 hard safety gates
  product_config.py       product name, roles, terms, privacy notice
  pubmed_search.py        Europe PMC / PubMed Central retrieval
  query_expander.py       search query expansion
  rag_system.py           RAGEngine, streaming, document ingestion
  response_templates.py   role-specific headings, personas, escalation banners
  role_router.py          RoleConfig and RoleRouter
  summarizer.py           document summarisation
  symptom_tracker.py      symptom log helpers
  triage_summary.py       structured triage card generation
  user_store.py           auth, profiles, all per-user data; JSON or Neon backend
  video_generator.py      sora-2 integration
  voice_transcriber.py    Whisper transcription
  pathways/
    chronic_conditions.py
    general_triage.py
    maternity.py
    medications.py
    msk.py
pages/
  1_Landing.py            consent gate, sign in, sign up
  2_Chatbot.py            main chat workspace
  2_Workspace.py          health trackers overview
  3_Health_Timeline.py    chronological health timeline
  4_Find_Clinical_Trials.py  ClinicalTrials.gov trial finder
Home.py                   entry point, redirects to landing or workspace
requirements.txt
```

## Troubleshooting

### `OPENAI_API_KEY not found in environment variables`

Create `.env` from `.env.example`, add a real key, and start Streamlit from the project root.

### Accounts or trial results disappear on a hosted deployment

Set `DATABASE_URL` so the app stores all user data in PostgreSQL instead of the local `users.json` file.

### Clinical trial search returns no results or errors

The trial finder calls ClinicalTrials.gov from the server. Check that outbound HTTPS is allowed in your hosting environment. The search also requires the user to have at least some health data saved (conditions, symptoms, medications, or uploads) so the LLM can extract search terms.

### Voice input is unavailable

Make sure the browser allows microphone access. The app uses Streamlit's audio input where available.

### Video generation does not appear

Check that your OpenAI account has access to the video model. The app will silently skip video generation if the model call fails.

### Medication interaction warnings do not appear

The interaction checker uses public openFDA drug label sections. If a medication name cannot be resolved, or if the label does not explicitly mention the paired drug, the app may show no pair-specific warning even when a pharmacist would still want to review it.

### Auto-populated data from a PDF looks wrong

Review the extracted entries in the health trackers and remove any that are incorrect. The LLM extractor reads free-text documents and may occasionally misread a value or unit — always check auto-populated data before relying on it.

### GP summary PDF is sparse

The PDF is generated from what the user has actually saved. Add symptom logs, medications, upload PDFs, or ask a question first so the account has longitudinal memory and triage content to include.

### spaCy or other optional NLP packages cause install issues

They are not required for the main app flow. Python 3.11 or 3.12 is the safest setup.

## Important note

Dr. Charlotte is for evidence review, health education, and decision support. It is not a substitute for emergency care, diagnosis, or a clinician's judgement.

If someone may be seriously unwell, use the appropriate urgent care route, such as NHS 111 or 999 in the UK.
