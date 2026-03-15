# My Health Chatbot

My Health Chatbot is a Streamlit-based health research assistant designed as a expert/no expert facing application rather than a prototype. Users can create an account, return with the same login, continue previous chats, upload personal documents, inspect traceable sources, export an audit snapshot, and clear saved chat history when needed.

## What the app does

- Client-facing landing page and chat workspace
- Persistent user accounts with login continuity
- Saved chat history per user, plus a `Delete chat` action
- Per-user document uploads and retrieval-ready personal context
- Live multi-source retrieval for every question
- LLM synthesis across retrieved evidence instead of single-source answers
- Clickable source links and trace information for responses
- Exportable audit snapshot for conversations, uploads, and traces

## Retrieval and answer flow

For each question, the app combines multiple evidence paths instead of relying on one case-specific rule:

1. The user signs in and can upload health documents.
2. Uploaded documents are processed into personal context for that specific user.
3. The question can be expanded into retrieval-friendly variants.
4. The app searches live sources including official guidance pages and Europe PMC / PubMed content.
5. Retrieved evidence is ranked with OpenAI embeddings.
6. The LLM synthesizes the evidence into a readable answer with traceable sources.
7. The chat, source trace, and audit record are saved to the user profile for future sessions.

## Sources used

- Europe PMC / PubMed for research evidence
- NHS for live patient-facing health guidance
- MedlinePlus for live patient-facing health guidance
- User-uploaded records for personal context within that account

## Core stack

- Frontend: Streamlit
- LLM and embeddings: OpenAI API
- Retrieval: Europe PMC, NHS, MedlinePlus
- Document parsing: PyMuPDF
- Local persistence: JSON user store plus per-user upload folders

## Requirements

- Python 3.11 or 3.12 recommended
- OpenAI API key
- `DATABASE_URL` for hosted deployments where account persistence matters

Python 3.14 is not recommended for this project because some NLP-related packages can behave inconsistently there.

## Quick start

From the repo root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Add your OpenAI key to `.env`:

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

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Data persistence

The app stores user-specific information locally:

- `users.json`: users, hashed passwords, profiles, chat history, traces, audit events
- `data/uploads/<username>/`: uploaded files for each signed-in user

Using the same login restores the same chat history and uploaded context.

If `DATABASE_URL` is set, the app uses a PostgreSQL-backed user store instead of `users.json`. This is the recommended mode for Streamlit-hosted deployments because accounts, chat history, traces, and document summaries survive app restarts and browser refreshes.

## Deployment note for Streamlit

For local development, the default JSON store is fine. For a deployed Streamlit link, use a real database:

1. Create a hosted PostgreSQL database such as Supabase, Neon, Azure Database for PostgreSQL, or another managed Postgres service.
2. Add `DATABASE_URL` to your Streamlit app secrets.
3. Redeploy or restart the app.

Example:

```toml
OPENAI_API_KEY="your_openai_api_key_here"
DATABASE_URL="postgresql://username:password@host:5432/database?sslmode=require"
```

With that in place, users can create an account once and sign back in later with the same credentials. Their profile, chat history, audit trail, traces, and document summaries will still be there.

## Project structure

```text
app_ui/
  static/
    assistant.png
    styles.css
    user.png
  theme.py
  uploader.py
backend/
  anonymizer.py
  memory_store.py
  moderation_ml.py
  official_guidance.py
  pubmed_search.py
  query_expander.py
  rag_system.py
  summarizer.py
  user_store.py
pages/
  1_Landing.py
  2_Chatbot.py
Home.py
requirements.txt
.env.example
```

## Troubleshooting

### `OPENAI_API_KEY not found in environment variables`

Create `.env` from `.env.example`, add your real key, and start Streamlit from the project root.

### Accounts disappear on the hosted Streamlit app

Set `DATABASE_URL` in Streamlit secrets so the app uses PostgreSQL for user persistence instead of the local `users.json` file.

### spaCy or Pydantic errors on Python 3.14

The app no longer requires spaCy to run. If spaCy is installed and causes import issues, use Python 3.11 or 3.12 for the cleanest setup.

### Styling changes do not appear immediately

Refresh the browser, or restart Streamlit:

```powershell
py -3.12 -m streamlit run Home.py
```

## Important note

This application is for evidence review, health education, and question support. It is not a substitute for emergency care, diagnosis, or a clinician's judgment.
