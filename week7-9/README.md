# Week 7-9: Agents, Guardrails, and Production RAG

Builds toward production-grade LLM applications: LangSmith-traced LCEL chains, PII-redaction and prompt-injection guardrails, multi-tool ReAct agents, hybrid retrieval with reranking, LangGraph workflows, and RAGAS-based evaluation. Uses local Ollama (`llama3.2`, `nomic-embed-text`) models throughout.

## Setup

```bash
pip install langchain langchain-community langchain-classic langchain-ollama langgraph python-dotenv \
    chromadb rank-bm25 sentence-transformers pypdf datasets ragas duckduckgo-search
```

Create a `.env` file (not committed) with:

```
LANGCHAIN_API_KEY=your_langsmith_key_here
```

Requires a local [Ollama](https://ollama.com) instance: `ollama pull llama3.2` and `ollama pull nomic-embed-text`.

## Tasks

| File | Description |
|---|---|
| `task.py` | Minimal LCEL chain (`prompt \| model \| parser`) with LangSmith tracing enabled — first step toward observable chains. |
| `task1.py` | Same LCEL chain pattern using `ChatPromptTemplate`, with LangSmith tracing configured via environment variables. |
| `task2.py` | Conversational RAG chatbot over a PDF with input/output guardrails: PII redaction (CNIC, phone, email, IBAN) and a prompt-injection blocklist, plus sliding conversation memory. |
| `task3.py` | Two-tool ReAct agent (Week 8): routes queries to a DuckDuckGo web-search tool (a custom RapidAPI weather tool is scaffolded but disabled), using LangChain's `AgentExecutor`. |
| `task4.py` | Multi-tool ReAct agent combining a PDF RAG Q&A tool with a Pakistani (FBR) income-tax-slab calculator tool. |
| `task5.py` | LUMS University AI Assistant (Week 9): two agents — a RAG policy Q&A agent and a fee-calculator agent — with topic guardrails, PII output scrubbing, LangSmith tracing, and JSON-persisted sliding-window memory. |
| `task6.py` | LangGraph version of the university assistant: a 2-node graph that collects user name/email (saved to `users.csv`) before answering from `lums_policies.txt` via RAG. |
| `task7.py` | LangGraph course registration workflow: validates student info (with retry loop up to 3 attempts), checks eligibility, pauses for advisor approval via `interrupt()`, and registers or rejects the student. Uses `InMemorySaver` checkpointing and `Command(resume=...)` to continue after human review. |
| `taska.py` | Production RAG chatbot (Assignment 2, Week 7): hybrid retrieval (BM25 + dense vector via `EnsembleRetriever`), cross-encoder reranking, multi-turn memory, PII redaction, and safety/output filtering. |
| `taskb.py` | Extended production RAG chatbot (Week 7 extension): adds per-session memory isolation, source citations, and RAGAS evaluation (faithfulness, answer relevancy, context precision) on top of taska's hybrid retrieval + reranking pipeline. |
| `assgn2.ipynb` | Notebook version of `taska.py` — the Assignment 2 production RAG chatbot walkthrough. |

## Data files
- `new-approaches.pdf` — sample source document for the RAG pipelines (tasks 2-4, taska, taskb).
- `lums_policies.txt` — sample knowledge base for the LUMS assistant (tasks 5-6).
- `users.csv`, `lums_chat_history.json` — example runtime output generated while testing task6/task5 (user-submitted names/emails and a sample Q&A session).
