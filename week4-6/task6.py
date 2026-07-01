from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import ollama
import re

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
app = FastAPI()
class Query(BaseModel):
    question: str

def load_docs():
    pdf = PyPDFLoader("new-approaches-and-procedures-for-cancer-treatment.pdf").load()
    doc = Docx2txtLoader("M.Sc. Applied Psychology.docx").load()
    return pdf + doc

def semantic_chunks(docs):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
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
            mag = (sum(a**2 for a in emb1) ** 0.5) * (sum(b**2 for b in emb2) ** 0.5)
            similarity = dot / mag if mag else 0.0

            if similarity < 0.75:
                chunks.append(" ".join(current))
                current = [sentence]
            else:
                current.append(sentence)

        if current:
            chunks.append(" ".join(current))

        all_chunks.extend([Document(page_content=c) for c in chunks])

    return Chroma.from_documents(all_chunks, embeddings)

def create_app():
    print("Loading")
    docs = load_docs()
    print("Indexing")
    vectorstore = semantic_chunks(docs)
   

    @app.post("/query")
    def query(req: Query):
        context = "\n\n".join(d.page_content for d in vectorstore.similarity_search(req.question, k=3))
        prompt = f"Answer using only this context:\n\n{context}\n\nQuestion: {req.question}"
        return {"answer": ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])["message"]["content"]}

if __name__ == "__main__":
    create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)