# my_health_chatbot

**my_health_chatbot** is a health-focused chatbot designed to answer user questions using real-time biomedical research. It retrieves and summarizes content from **PubMed Central (PMC)**, providing scientifically grounded responses in plain English.

---

## Features

- Real-time access to biomedical literature from PubMed Central
- Natural language question support
- Evidence-based summarization using Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG) pipeline
- Designed for general users and health professionals

---

## Example Questions

- What are the current treatments for hypertension?
- Is metformin safe for elderly patients?
- What does recent research say about probiotics and gut health?

---

## How It Works

1. The user submits a health-related question.
2. The system reformulates the question into a search query suitable for scientific databases.
3. It retrieves open-access articles from PubMed Central using the Entrez API.
4. Relevant sections of the articles are passed to a language model for summarization.
5. A concise, user-friendly answer is generated, optionally including sources.

---

## Tech Stack

- Frontend: Streamlit
- Backend: Python
- Data Source: PubMed Central Open Access API
- Language Models: OpenAI
- Architecture: Retrieval-Augmented Generation (RAG)

---

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/my_health_chatbot.git
cd my_health_chatbot

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
