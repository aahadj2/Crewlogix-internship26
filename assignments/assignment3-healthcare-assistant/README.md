# Assignment 3 — Healthcare Clinic RAG Assistant

A RAG-based assistant for Dr. Jawad Zaheer's clinic that answers patient questions (appointments, fees, schedules, lab tests, refills, reports, emergencies) strictly from the clinic's policy documents, running fully local on Ollama.

## Stack

- **LLM:** `llama3.2` via Ollama (local, no API keys)
- **Embeddings:** `nomic-embed-text` via Ollama
- **Vector store:** Chroma (persisted to `chroma_db/`, built automatically on first run from `docs/`)
- **Orchestration:** LangGraph agent with a SQLite checkpointer (`chat_history.db`) for multi-turn conversation memory
- **UI:** Streamlit (`streamlit_app.py`) or CLI (`app.py`)

## Features

- Answers only from the 8 clinic documents in `docs/`; falls back with "I could not find this information in the provided clinic documents" for out-of-scope questions
- Safety guardrails: refuses diagnosis/prescription requests and redirects emergency symptoms (chest pain, stroke, etc.) to urgent care
- Every answer ends with a single `[Source: <document>]` citation
- Conversation memory across turns (pronoun references resolve correctly)

## Running

Prerequisites: [Ollama](https://ollama.com) installed with both models pulled:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Install dependencies and run:

```bash
pip install -r requirements.txt

# CLI
python app.py

# Streamlit UI
streamlit run streamlit_app.py
```

The Chroma index and chat-history database are created on first run.
