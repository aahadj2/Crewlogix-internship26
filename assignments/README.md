# Assignments

This folder contains the standalone assignments completed during the CrewLogix internship, alongside the weekly tasks in `week1-3/`, `week4-6/`, and `week7-9/`.

## Index

| # | Assignment | Description | Status |
|---|-----------|-------------|--------|
| 1 | [Build, Evaluate & Improve a RAG Assistant](assignment1-rag-evaluation/) | RAG pipeline over internship daily-report PDFs with hybrid retrieval (Chroma vector search + BM25), improved with a `bge-reranker-base` cross-encoder. Evaluated baseline vs. improved on an 8-question test set using RAGAS metrics (`task1.py`) and manual overlap metrics (`task2.py`). Fully local on Ollama. | Complete |
| 2 | [Healthcare Clinic RAG Assistant](assignment2-healthcare-assistant/) | RAG-based clinic assistant answering patient questions (appointments, fees, schedules, lab tests, refills, emergencies) strictly from clinic policy documents. Fully local stack: Ollama (llama3.2 + nomic-embed-text), Chroma vector store, LangGraph agent with SQLite conversation memory, Streamlit/CLI interfaces, and safety guardrails with source citations. | Complete |
| 3 | _To be added_ | — | Pending |

## Structure

Each assignment lives in its own folder (`assignmentN-short-name/`) containing the source code, supporting documents, the original assignment brief, and its own README with setup and run instructions.
