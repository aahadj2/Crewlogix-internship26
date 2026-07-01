import re
from numpy import rint
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


ground_truth = {
    "what is gene therapy": [
        "Gene therapy is the insertion of a normal copy of a defective gene in the genome to cure a specific disorder."
    ],
    "what is stem cell therapy": [
        "Stem cell therapeutic strategy is also one of the treatment options for cancer which are considered to be safe and effective."
    ],
    "what is ablation therapy": [
        "Ablation is a treatment technique that destroys tumors without removing them."
    ],
}
def load():
    return PyPDFLoader("new-approaches-and-procedures-for-cancer-treatment.pdf").load()

def chunk(docs):
    all_chunks = []
    for doc in docs:
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', doc.page_content.strip()) if s]
        if not sentences:
            continue
        chunks, current = [], [sentences[0]]
        for s in sentences[1:]:
            e1 = ollama.embeddings(model="nomic-embed-text", prompt=" ".join(current))["embedding"]
            e2 = ollama.embeddings(model="nomic-embed-text", prompt=s)["embedding"]
            dot = sum(a*b for a,b in zip(e1,e2))
            mag = (sum(a**2 for a in e1)**0.5) * (sum(b**2 for b in e2)**0.5)
            if (dot/mag if mag else 0) < 0.75:
                chunks.append(" ".join(current))
                current = [s]
            else:
                current.append(s)
        chunks.append(" ".join(current))
        all_chunks.extend([Document(page_content=c, metadata=doc.metadata) for c in chunks])
    return all_chunks

def build_stores(chunks):
    vectordb = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory="./chroma_db")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 20
    return vectordb, bm25

def rrf(lists, k=60):
    scores, docs = {}, {}
    for results in lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            docs[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (rank + k)
    return [docs[d] for d in sorted(scores, key=scores.get, reverse=True)]

def rerank(query, candidates, reranker, top_k=3):
    scores = reranker.predict([(query, d.page_content) for d in candidates])
    return [doc for _, doc in sorted(zip(scores, candidates), reverse=True)][:top_k]

def recall_at_k(retrieved, relevant, k=5):
    top_k = set(d.page_content for d in retrieved[:k])
    return len(top_k & set(relevant)) / len(relevant) if relevant else 0.0

def answer(query, docs):
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"Answer using only this context:\n\n{context}\n\nQuestion: {query}"
    return ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])["message"]["content"]

if __name__ == "__main__":
    print("Loading") 
    docs = load()

    print("Chunking")
    chunks = chunk(docs)
    print(f"{len(chunks)} chunks")

    print("Building stores")
    vectordb, bm25 = build_stores(chunks)
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    while True:
        query = input("Ask (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        candidates = rrf([vectordb.similarity_search(query, k=20), bm25.invoke(query)])[:20]
        reranked   = rerank(query, candidates, reranker)

        relevant = ground_truth.get(query.lower(), [])
        if relevant:
            before = recall_at_k(candidates, relevant, k=5)
            after  = recall_at_k(reranked,   relevant, k=3)
            print(f"Recall@5 before: {before:.3f} | Recall@3 after: {after:.3f}")
        else:
            print("(no ground truth for this query — skipping recall)")
        #print(f"Recall before: {before:.3f} | Recall after: {after:.3f}")

        print("\nAnswer:\n" + answer(query, reranked) + "\n")