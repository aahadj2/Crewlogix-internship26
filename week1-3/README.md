# Week 1-3: GenAI / LLM Prompting Basics

Exploratory tasks covering LLM API integration, prompting techniques, structured output extraction, and tool calling using Google Gemini and local Ollama (llama3.2) models.

## Setup

```bash
pip install google-generativeai python-dotenv fastapi uvicorn ollama langchain
```

Create a `.env` file (not committed) with:

```
GOOGLE_API_KEY=your_key_here
```

Tasks 2-7 use a local [Ollama](https://ollama.com) instance running the `llama3.2` model — install Ollama and run `ollama pull llama3.2` before running them.

## Tasks

| File | Description |
|---|---|
| `task1.py` | FastAPI service with a `POST /ask` endpoint that answers questions using Google's Gemini (`gemini-2.0-flash`) API. |
| `task2.py` | Same FastAPI Q&A endpoint as task1, but backed by a local Ollama model (`llama3.2`) instead of a cloud API. |
| `task3.py` | Prompt engineering demo: zero-shot vs. few-shot prompting for a customer-review sentiment analyzer, plus tuning generation parameters (temperature/top_p/top_k). |
| `task4.py` | Few-shot + chain-of-thought prompting demo for a "study coach" assistant that diagnoses a student's learning problem and recommends a strategy. |
| `task5.py` | Extracts structured JSON (role, skills, seniority, salary range) from unstructured Pakistani job-posting text, with JSON validation and formatted table output. |
| `task6.py` | Function/tool-calling demo — routes natural-language math queries to `add`/`subtract`/`multiply` Python functions via Ollama, then generates a natural-language explanation of the result. |
| `task7.py` | One-shot + chain-of-thought customer support ticket classifier that outputs a category (Billing, Technical Issue, etc.) and reason as JSON. |
