import ollama
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_models import ChatOllama
from sentence_transformers import CrossEncoder
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig

PDF_PATHS = [
    "Monday6.pdf",
    "Tuesday6.pdf",
    "Wednesday6.pdf",
    "Thursday6.pdf",

]

def load():
    docs = []
    for path in PDF_PATHS:
        docs.extend(PyPDFLoader(path).load())
    print(f"Loaded {len(docs)} pages from {len(PDF_PATHS)} documents")
    return docs

def chunk(docs):
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    return chunks

def build_stores(chunks):
    vectordb = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory="./chroma_db")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    return vectordb, bm25

def retrieve(query, vectordb, bm25, reranker=None, top_k=5):
    seen, docs = set(), []
    for doc in vectordb.similarity_search(query, k=top_k) + bm25.invoke(query):
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            docs.append(doc)
    if reranker:
        scores = reranker.predict([(query, d.page_content) for d in docs])
        docs = [d for _, d in sorted(zip(scores, docs), reverse=True)][:top_k]
    return docs

def answer(query, docs):
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"Answer using only this context. If not found, say 'Not found in documents'.\n\nContext:\n{context}\n\nQuestion: {query}"
    return ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])["message"]["content"]

def run_ragas(vectordb, bm25, reranker=None):
    qa_pairs = [
        {"question": "What is BM25 and why is it used?",                          "ground_truth": "BM25 is a keyword retrieval algorithm that rewards rare terms, accounts for term frequency and document length normalisation. It is used for exact match retrieval of IDs, codes, names, and numbers."},
        {"question": "What is Reciprocal Rank Fusion?",                            "ground_truth": "RRF combines two ranked lists using the formula 1/(k+rank) per list, summed across lists. It is scale-independent and used to fuse BM25 and vector search results."},
        {"question": "What is a cross-encoder and how does it differ from a bi-encoder?", "ground_truth": "A cross-encoder encodes the query and document together for joint attention, giving more accurate relevance scores. A bi-encoder encodes them independently and compares vectors."},
        {"question": "What is HyDE?",                                              "ground_truth": "HyDE generates a hypothetical answer to the query using an LLM, embeds that answer, and uses it for retrieval instead of the original query to fix vocabulary mismatch."},
        {"question": "What are the 4 RAGAS metrics?",                             "ground_truth": "Faithfulness, Context Precision, Context Recall, and Answer Relevancy."},
        {"question": "What is step-back prompting?",                              "ground_truth": "Step-back prompting rewrites a too-specific query into a broader, more general version before retrieval to improve recall on narrow queries."},
        {"question": "What is ChromaDB persistence?",                             "ground_truth": "ChromaDB persistence saves the collection to disk so documents do not need to be re-embedded on every run. Use PersistentClient with a path."},
        {"question": "What year was the Eiffel Tower built?",                     "ground_truth": "Not found in documents."},
    ]

    questions, answers, contexts, ground_truths = [], [], [], []
    for qa in qa_pairs:
        docs = retrieve(qa["question"], vectordb, bm25, reranker)
        ans  = answer(qa["question"], docs)
        questions.append(qa["question"])
        answers.append(ans)
        contexts.append([d.page_content for d in docs])
        ground_truths.append(qa["ground_truth"])
        print(f"Q: {qa['question']}\nA: {ans[:150]}\n")

    asyncio.set_event_loop(asyncio.new_event_loop())
    return evaluate(
        Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}),
        metrics=[faithfulness, answer_relevancy],
        llm=ChatOllama(model="llama3.2"),
        embeddings=OllamaEmbeddings(model="nomic-embed-text"),
        run_config=RunConfig(timeout=120, max_retries=2, max_workers=1),
    )

if __name__ == "__main__":
    print("Loading docs")
    vectordb, bm25 = build_stores(chunk(load()))

    print("\nBaseline RAGAS")
    baseline = run_ragas(vectordb, bm25)
    print("Baseline:", baseline)

    reranker = CrossEncoder("BAAI/bge-reranker-base")

    print("\nImproved RAGAS")
    improved = run_ragas(vectordb, bm25, reranker)
    print("Improved:", improved)

    print(f"\n{'Metric':<25} {'Baseline':>10} {'Improved':>10}")
    print("-" * 47)
    for metric in ["faithfulness", "answer_relevancy"]:
        b = round(baseline[metric], 3)
        i = round(improved[metric], 3)
        print(f"{metric:<25} {str(b):>10} {str(i):>10}")

    while True:
        query = input("\nAsk (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        docs = retrieve(query, vectordb, bm25, reranker)
        for i, d in enumerate(docs, 1):
            print(f"[{i}] {d.page_content[:100]}...")
        print("\nAnswer:\n" + answer(query, docs))