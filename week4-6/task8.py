import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

def load():
    return PyPDFLoader("new-approaches-and-procedures-for-cancer-treatment.pdf").load()

def chunk(docs):
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)

def build_stores(chunks):
    vectordb = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory="./chroma_db")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    return vectordb, bm25

def rrf(results_list, k=60):
    scores, docs = {}, {}
    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc.page_content
            docs[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (rank + k)
    return [docs[d] for d in sorted(scores, key=scores.get, reverse=True)][:5]

def answer(query, vectordb, bm25):
    fused = rrf([vectordb.similarity_search(query, k=5), bm25.invoke(query)])
    context = "\n\n".join(d.page_content for d in fused)
    prompt = f"Answer using only this context:\n\n{context}\n\nQuestion: {query}"
    return ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])["message"]["content"]

if __name__ == "__main__":
    print("Loading...")
    chunks = chunk(load())
    vectordb, bm25 = build_stores(chunks)
    print("Ready!\n")

    while True:
        query = input("Ask: ")
        if query.lower() == "exit":
            break
        print("\n" + answer(query, vectordb, bm25) + "\n")