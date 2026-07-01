import ollama
import math
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def load_docs():
    return PyPDFLoader("new-approaches-and-procedures-for-cancer-treatment.pdf").load()

def semantic_chunks(docs):
    all_chunks = []
    for doc in docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content.strip())
        sentences = [s for s in sentences if s]
        if not sentences:
            continue
        chunks, current = [], [sentences[0]]
        for sentence in sentences[1:]:
            emb1 = ollama.embeddings(model="nomic-embed-text", prompt=" ".join(current))["embedding"]
            emb2 = ollama.embeddings(model="nomic-embed-text", prompt=sentence)["embedding"]
            dot = sum(a * b for a, b in zip(emb1, emb2))
            mag = (sum(a**2 for a in emb1)**0.5) * (sum(b**2 for b in emb2)**0.5)
            similarity = dot / mag if mag else 0.0
            if similarity < 0.75:
                chunks.append(" ".join(current))
                current = [sentence]
            else:
                current.append(sentence)
        if current:
            chunks.append(" ".join(current))
        all_chunks.extend([Document(page_content=c) for c in chunks])
    return all_chunks

def build_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

def build_keyword_index(chunks):
    corpus = [c.page_content for c in chunks]
    tokenized = [doc.lower().split() for doc in corpus]
    df = {}
    for tokens in tokenized:
        for word in set(tokens):
            df[word] = df.get(word, 0) + 1
    return corpus, tokenized, df

def keyword_search(query, corpus, tokenized, df, k=5):
    N = len(corpus)
    query_tokens = query.lower().split()
    scores = []
    for i, tokens in enumerate(tokenized):
        score = 0
        for word in query_tokens:
            tf = tokens.count(word)
            idf = math.log((N + 1) / (df.get(word, 0) + 1))
            score += tf * idf
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [Document(page_content=corpus[i]) for i, _ in scores[:k]]

def rrf(vector_results, keyword_results, k=60):
    scores = {}
    for rank, doc in enumerate(vector_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(keyword_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:5]

def answer(question, vectorstore, corpus, tokenized, df):
    vector_results = vectorstore.similarity_search(question, k=5)
    keyword_results = keyword_search(question, corpus, tokenized, df, k=5)
    top_chunks = rrf(vector_results, keyword_results)
    context = "\n\n".join(top_chunks)
    prompt = f"Answer using only this context:\n\n{context}\n\nQuestion: {question}"
    return ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])["message"]["content"]

if __name__ == "__main__":
    print("Loading docs")
    docs = load_docs()

    print("Chunking documents ")
    chunks = semantic_chunks(docs)
    print(f"{len(chunks)} chunks created")

    print("Building vector store")
    vectorstore = build_vector_store(chunks)

    print("Building keyword index")
    corpus, tokenized, df = build_keyword_index(chunks)

    while True:
        question = input("Ask a question (or 'exit'): ")
        if question.lower() == "exit":
            break
        print("\n" + answer(question, vectorstore, corpus, tokenized, df) + "\n")