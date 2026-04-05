# Dr. Charlotte

Dr. Charlotte is a Streamlit-based health information assistant with a proper signed-in workspace, personal document context, live evidence retrieval, traceable sources, and safety guardrails. It is designed for individuals, caregivers, and clinical users who want answers that are readable, inspectable, and consistent across sessions.

This repository is more than a simple chatbot demo. It includes account persistence, document upload and anonymisation, longitudinal memory, audit export, voice input, optional AI-generated visuals, and a role-aware clinical orchestration layer.

## What the app can do

- Create accounts and let people sign back in later with the same saved workspace
- Show a UK privacy notice and require explicit consent before access
- Support different user roles, including patient, caregiver, doctor, nurse, midwife, physiotherapist, and other clinician
- Save chat history, profile settings, uploads, audit events, and interaction traces per user
- Let users clear chat history without deleting the whole account
- Upload PDF records such as discharge letters, lab reports, and specialist notes
- Extract text from uploaded PDFs, anonymise sensitive information, and summarise the document into retrieval-ready context
- Build and refresh a longitudinal patient memory from uploaded records and ongoing conversation
- Search live public sources for each question, including NHS guidance, MedlinePlus, and Europe PMC / PubMed Central
- Expand user questions into retrieval-friendly search queries
- Rank evidence with OpenAI embeddings and label sources by evidence tier
- Stream answers into the chat interface as they are generated
- Show clickable citations, source trace panels, personal context used, and audit metadata for each assistant response
- Export a user snapshot as JSON for audit and review
- Accept spoken questions through Whisper transcription
- Generate clinical-style illustrations on request with `gpt-image-1`
- Generate short clinical demonstration videos on request with `sora-2`, with a one-video-per-hour limit per user
- Apply moderation, crisis detection, policy gates, and pathway-specific safety rules before the final answer is produced

## How the answer pipeline works

1. A user signs in and enters the workspace.
2. Uploaded PDFs are saved into that user's own folder.
3. PDF text is extracted with PyMuPDF and anonymised before it is summarised or indexed.
4. The system restores any saved document summaries and longitudinal memory for that user.
5. Each new question is checked for crisis language and passed through moderation.
6. The question is classified for intent and risk, then routed through a relevant pathway such as general triage, maternity, musculoskeletal, medication, or chronic condition support.
7. The app expands the question into search-friendly variants.
8. It retrieves live official guidance from NHS and MedlinePlus, and open-access biomedical evidence from Europe PMC / PubMed Central.
9. Retrieved material and personal context are ranked semantically with OpenAI embeddings.
10. Sources are deduplicated, tiered, and passed to the LLM for a cited answer.
11. The final answer, trace, sources, and refreshed longitudinal memory are saved back to the user account.

If the system cannot retrieve enough reliable evidence, it falls back to a more limited response rather than pretending to know more than it does.

## Roles, pathways, and safety

The app does not answer every user in the same voice. It adapts response style, evidence framing, and escalation thresholds based on the selected role.

- Patients and caregivers get plainer language and stronger escalation nudges
- Doctors, nurses, midwives, and physiotherapists get more clinically structured responses
- Crisis patterns are screened early so emergency guidance can be returned immediately
- Policy gates add extra caution for pregnancy, paediatrics, elderly polypharmacy, medication dosing, diagnosis-seeking questions, and mental health topics
- Moderation runs before retrieval so unsafe requests can be blocked early
- AI-generated images and videos are only triggered by explicit user requests and are filtered away from unsafe visual topics

The pathway and policy layers are written with UK clinical safety framing in mind. The live public retrievers in this repo currently fetch NHS, MedlinePlus, and Europe PMC / PubMed Central content.

## Media features

### Voice input

Users can record a question in the chat page and have it transcribed with OpenAI Whisper. The app stores the resulting text in the conversation, not the raw audio file.

### Illustrations

If a user explicitly asks to see something visually, the app can generate a clinical-style educational illustration with `gpt-image-1`.

### Short videos

If a user explicitly asks for a video or animation, the app can attempt to generate a short demonstration clip with `sora-2`. Video generation is rate-limited to once per hour per user. Whether this works in practice depends on the OpenAI account and model access available in your environment.

## Data and persistence

By default, the app stores user data locally:

- `users.json` holds accounts, hashed passwords, profiles, chat history, document summaries, traces, audit events, and longitudinal memory
- `data/uploads/<username>/` holds uploaded PDFs for each user

If `DATABASE_URL` is set, the app switches from the local JSON store to PostgreSQL for account persistence. That is the better option for hosted or shared deployments.

Passwords are stored as salted PBKDF2-SHA256 hashes, not in plain text.

## Tech stack

- Frontend: Streamlit
- Answer generation and query expansion: OpenAI Chat Completions
- Embeddings: OpenAI `text-embedding-3-small`
- Voice transcription: OpenAI Whisper
- Image generation: OpenAI `gpt-image-1`
- Video generation: OpenAI `sora-2`
- Official guidance retrieval: NHS and MedlinePlus
- Biomedical literature retrieval: Europe PMC / PubMed Central
- PDF parsing: PyMuPDF
- Persistence: local JSON or PostgreSQL
- Moderation: rules-based checks plus Detoxify support when available

## Requirements

- Python 3.11 or 3.12 is recommended
- An OpenAI API key
- Optional `DATABASE_URL` if you want PostgreSQL-backed persistence

Python 3.14 is not recommended for this project because some optional NLP tooling can be awkward there.

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

## Running with PostgreSQL

For local development, `users.json` is fine. For anything long-lived or shared, use PostgreSQL.

1. Create a PostgreSQL database.
2. Add `DATABASE_URL` to your environment or Streamlit secrets.
3. Restart the app.

Example:

```toml
OPENAI_API_KEY="your_openai_api_key_here"
DATABASE_URL="postgresql://username:password@host:5432/database?sslmode=require"
```

## Project structure

```text
app_ui/
  static/
  theme.py
  uploader.py
backend/
  anonymizer.py
  clinical_orchestrator.py
  evidence_ranker.py
  image_generator.py
  intent_risk_classifier.py
  memory_store.py
  moderation_ml.py
  official_guidance.py
  policy_engine.py
  pubmed_search.py
  query_expander.py
  rag_system.py
  response_templates.py
  summarizer.py
  user_store.py
  video_generator.py
  voice_transcriber.py
  pathways/
pages/
  1_Landing.py
  2_Chatbot.py
Home.py
requirements.txt
users.json
data/uploads/
```

## Troubleshooting

### `OPENAI_API_KEY not found in environment variables`

Create `.env` from `.env.example`, add a real key, and start Streamlit from the project root.

### Accounts disappear on a hosted deployment

Use `DATABASE_URL` so the app stores users in PostgreSQL instead of relying on local app storage.

### Voice input is unavailable

Make sure your browser allows microphone access. The app uses Streamlit's audio input where available, with a fallback recorder package in older environments.

### Video generation does not appear

Check that your OpenAI account has access to the required video model. The app will silently skip video generation if the model call fails.

### spaCy or other optional NLP packages cause install issues

They are not required for the main app flow. Python 3.11 or 3.12 is the safest setup.

## Important note

Dr. Charlotte is for evidence review, health education, and decision support. It is not a substitute for emergency care, diagnosis, or a clinician's judgement.

If someone may be seriously unwell, use the appropriate urgent care route, such as NHS 111 or 999 in the UK.
