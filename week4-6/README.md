# Week 4-6: Retrieval-Augmented Generation (RAG)

Progressive build-up of a RAG pipeline: from transformer attention internals, through basic vector retrieval, to hybrid search, reranking, and RAG evaluation with RAGAS. Uses local Ollama (`llama3.2`, `nomic-embed-text`) models throughout.

## Setup

```bash
pip install bertviz transformers torch faiss-cpu sentence-transformers ollama pymupdf pypdf python-docx \
    langchain langchain-community langchain-text-splitters chromadb rank-bm25 ragas datasets fastapi uvicorn
```

Requires a local [Ollama](https://ollama.com) instance: `ollama pull llama3.2` and `ollama pull nomic-embed-text`.

Sample source documents (`Stats.docx`, `M.Sc. Applied Psychology.docx`, `The_Plan_of_the_Giza_Pyramids.pdf`, `new-approaches-and-procedures-for-cancer-treatment.pdf`, `test.pdf`) are included so the RAG scripts run out-of-the-box.

## Tasks

| File | Description |
|---|---|
| `task1.py` | Visualizes BERT's self-attention heads (via `bertviz`) over a sample sentence — foundational look at how transformer attention works. |
| `task2.py` | Basic RAG pipeline: chunks a PDF, embeds chunks with `all-MiniLM-L6-v2`, indexes them in FAISS, and answers interactive questions using retrieved context + Ollama. |
| `task3.py` | Multi-source RAG: loads a PDF, a DOCX, and a live Wikipedia page via LangChain loaders, splits into chunks, embeds with Ollama, and serves an interactive Q&A loop over a FAISS index. |
| `task4.py` | RAG with a persistent ChromaDB store — builds (or reloads) an embedded collection from a PDF and answers questions with a minimum-similarity-score cutoff. |
| `task5.py` | FastAPI RAG service using semantic (embedding-similarity) chunking instead of fixed-size chunking, indexing a PDF + DOCX pair and exposing a `/query` endpoint. |
| `task6.py` | Same semantic-chunking RAG approach as task5, but backed by a persistent Chroma vector store via LangChain, wrapped in a FastAPI `/query` endpoint. |
| `task7.py` | Hybrid retrieval: combines dense vector search (Chroma) with sparse keyword/TF-IDF search, fused via Reciprocal Rank Fusion (RRF), to answer questions. |
| `task8.py` | Hybrid retrieval using Chroma (dense) + BM25 (sparse) combined with RRF fusion — a cleaner/production-style version of task7's hybrid search. |
| `task9.py` | Adds a cross-encoder reranking stage (`BAAI/bge-reranker-base`) on top of hybrid retrieval, and measures Recall@K before/after reranking against a small ground-truth set. |
| `task10.py` | Full RAG evaluation pipeline using RAGAS metrics (faithfulness, answer relevancy) to compare a baseline hybrid-retrieval pipeline against a reranked version on a fixed QA test set. |
