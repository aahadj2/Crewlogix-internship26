# Assignment 1 — Build, Evaluate, and Improve a Real RAG Assistant

A RAG pipeline over four internship daily-report PDFs (week 6), evaluated before and after adding a cross-encoder reranker to measure the improvement. Runs fully local on Ollama.

## Pipeline

1. **Load & chunk:** PyPDFLoader over the 4 PDFs, RecursiveCharacterTextSplitter (500 chars, 100 overlap)
2. **Hybrid retrieval:** Chroma vector search (`nomic-embed-text` embeddings) + BM25 keyword retrieval, deduplicated
3. **Improvement:** `BAAI/bge-reranker-base` cross-encoder reranks the merged candidates
4. **Generation:** `llama3.2` via Ollama, answering strictly from retrieved context ("Not found in documents" fallback)

## Evaluation

Both scripts run the same 8-question test set (7 answerable + 1 out-of-scope trap) as baseline vs. reranked, and print a comparison table:

- **`task1.py`** — RAGAS evaluation (faithfulness, answer relevancy) with llama3.2 as the judge, then an interactive Q&A loop
- **`task2.py`** — manual token-overlap metrics (faithfulness, relevancy, context recall) without an LLM judge

Findings are written up in `aireport1.docx`; the original brief is in the `Assignment- Build, Evaluate...` docx.

## Running

Prerequisites: [Ollama](https://ollama.com) with `llama3.2` and `nomic-embed-text` pulled.

```bash
pip install -r requirements.txt
python task1.py   # RAGAS evaluation + interactive Q&A
python task2.py   # manual metric evaluation
```

The Chroma index (`chroma_db/`) is built on first run.
