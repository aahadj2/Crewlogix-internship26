"""
Week 10 — Task 6
University AI Assistant using LangGraph
  Node 1: collect_user_info  — asks name & email, saves to users.csv
  Node 2: rag_answer         — answers query from lums_policies.txt
"""

import csv
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

load_dotenv()

# LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# RAG setup
_POLICIES_FILE = Path(__file__).parent / "lums_policies.txt"
_docs = TextLoader(str(_POLICIES_FILE), encoding="utf-8").load()
_chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60).split_documents(_docs)
_vectorstore = FAISS.from_documents(_chunks, OllamaEmbeddings(model="nomic-embed-text"))
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

_rag_chain = (
    ChatPromptTemplate.from_template(
        "You are a helpful university assistant. "
        "Answer using ONLY the context below. "
        "If the answer is not in the context, say you don't have that information.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    | llm
    | StrOutputParser()
)

# CSV file for storing user details
CSV_FILE = Path(__file__).parent / "users.csv"


# Graph state
class State(TypedDict):
    query: str
    name: str
    email: str
    answer: str


# Node 1: Collect user name & email, save to CSV
def collect_user_info(state: State) -> State:
    print("\nBefore answering, please provide your details.")
    name = input("Your name: ").strip()
    email = input("Your email: ").strip()

    file_exists = CSV_FILE.exists()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "email", "query"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"name": name, "email": email, "query": state["query"]})

    print(f"Details saved for {name}.")
    return {**state, "name": name, "email": email}


# Node 2: Answer query using RAG
def rag_answer(state: State) -> State:
    docs = _retriever.invoke(state["query"])
    context = "\n\n".join(d.page_content for d in docs)
    answer = _rag_chain.invoke({"context": context, "question": state["query"]})
    return {**state, "answer": answer}


# Build LangGraph workflow
_builder = StateGraph(State)
_builder.add_node("collect_user_info", collect_user_info)
_builder.add_node("rag_answer", rag_answer)

_builder.set_entry_point("collect_user_info")
_builder.add_edge("collect_user_info", "rag_answer")
_builder.add_edge("rag_answer", END)

app = _builder.compile()


if __name__ == "__main__":
    print("University AI Assistant")
    print("=" * 40)
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = app.invoke({"query": query, "name": "", "email": "", "answer": ""})
        print(f"\nAnswer: {result['answer']}\n")
