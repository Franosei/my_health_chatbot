# MCP Server

FlynnMed exposes clinical tools via MCP -- works both locally and on Railway.

## Tools exposed

### get_patient_context(username)
Returns the patient's full health context: profile, conditions, medications,
allergies, vitals, symptom logs, latest triage, longitudinal memory.

### extract_article_evidence(article_title, article_text, patient_question, patient_conditions, patient_medications, patient_age, evidence_tier)
Extracts structured ArticleEvidence JSON from a medical article matched to a patient.
Returns question_facts, patient_aligned_facts, contraindications, drug_interactions.

### generate_clinical_note(username, patient_question, conversation_summary, urgency_level, next_step)
Generates a SOAP note and saves it to the patient's account.
Returns the full note JSON including note_id.

### send_health_email(username, email_type, note_id, urgency_level, reason)
Sends email to the patient. email_type: "clinical_note" | "urgent_alert"
Requires SMTP configured in .env

### search_trials_for_patient(username, location, max_results)
Searches ClinicalTrials.gov for recruiting trials matched to the patient's
conditions and medications. Returns ranked trial results.

## Connect to Claude Desktop

### Deployed on Railway (recommended)
The MCP server is mounted automatically at `/mcp` on your Railway service.
No separate process needed -- it runs as part of the main app.

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "flynnmed": {
      "url": "https://<your-app>.railway.app/mcp",
      "headers": {
        "Authorization": "Bearer <MCP_API_KEY>"
      }
    }
  }
}
```

Set `MCP_API_KEY` in Railway environment variables to restrict access.

### Local (stdio -- dev only)
```bash
pip install mcp
python -m backend.mcp_server
```
```json
{
  "mcpServers": {
    "flynnmed": {
      "command": "python",
      "args": ["-m", "backend.mcp_server"],
      "cwd": "/path/to/my_health_chatbot"
    }
  }
}
```

## Railway environment variables
```
MCP_API_KEY=some-secret-key   # protects the /mcp endpoint
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@email.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=FlynnMed <your@email.com>
```
