# import pypdf
# import ollama

# pdf_path = "new-approaches-and-procedures-for-cancer-treatment.pdf"
# doc_path = "M.Sc, Applied Psychology.docx"

# data = pypdf.PdfReader(pdf_path)
# docdata = ollama.Ollama(doc_path)

# #semantic chunking

# def semantic_chunking(text, chunk_size=500, overlap=50):
#     chunks = []
#     start = 0

   
# def emded_chunking(text, chunk_size=500, overlap=50):


# def cosine_similarity(vec1, vec2):
#     dot_product = sum(a * b for a, b in zip(vec1, vec2))
#     magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
#     magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
#     if magnitude1 == 0 or magnitude2 == 0:
#         return 0.0
#     return dot_product / (magnitude1 * magnitude2)
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
app = FastAPI()
import pypdf
import docx
import ollama
import re
app = FastAPI()
class Query(BaseModel):
    question: str
@app.post("/query")
def query(req: Query):
    return {"answer": answer(req.question, index)}

def load_pdf(path):
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def semantic_chunks(text, threshold=0.75):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]

    chunks, current = [], [sentences[0]]

    for sentence in sentences[1:]:
        combined = " ".join(current + [sentence])
        emb1 = ollama.embeddings(model="nomic-embed-text", prompt=" ".join(current))["embedding"]
        emb2 = ollama.embeddings(model="nomic-embed-text", prompt=sentence)["embedding"]

        dot = sum(a * b for a, b in zip(emb1, emb2))
        mag = (sum(a**2 for a in emb1) ** 0.5) * (sum(b**2 for b in emb2) ** 0.5)
        similarity = dot / mag if mag else 0.0

        if similarity < threshold:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_index(chunks):
    return [{"text": c, "emb": ollama.embeddings(model="nomic-embed-text", prompt=c)["embedding"]} for c in chunks]

def get_top_k(query, index, k=3):
    q_emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag = (sum(x**2 for x in a) ** 0.5) * (sum(y**2 for y in b) ** 0.5)
        return dot / mag if mag else 0.0

    scored = sorted(index, key=lambda x: cosine(q_emb, x["emb"]), reverse=True)
    return [x["text"] for x in scored[:k]]

def answer(query, index):
    context = "\n\n".join(get_top_k(query, index))
    prompt = f"Answer using only this context:\n\n{context}\n\nQuestion: {query}"
    return ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])["message"]["content"]

if __name__ == "__main__":
    text = load_pdf("new-approaches-and-procedures-for-cancer-treatment.pdf")
    text += "\n" + load_docx("M.Sc. Applied Psychology.docx")

    print("Chunking...")
    index = build_index(semantic_chunks(text))
    print(f"{len(index)} chunks indexed")

    print(answer("What are the advantages of stem cell therapy?", index))  
    uvicorn.run(app, host="0.0.0.0", port=8000)