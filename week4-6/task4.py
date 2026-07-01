import chromadb
import pypdf
import ollama
from sentence_transformers import SentenceTransformer

PDF_PATH   = "The_Plan_of_the_Giza_Pyramids.pdf"
DB_PATH    = "./chroma_db"
COLLECTION = "giza_plan"
CHUNK_SIZE = 500
CHUNK_OL   = 50
TOP_K      = 3
MIN_SCORE  = 0.0

emb = SentenceTransformer("all-MiniLM-L6-v2")
db  = chromadb.PersistentClient(path=DB_PATH)


def load_and_chunk(path):
    reader = pypdf.PdfReader(path)
    text   = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = []
    start  = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OL
    return chunks


def get_or_build_collection():
    try:
        col = db.get_collection(COLLECTION)
        if col.count() > 0:
            print(f"Loaded {col.count()} chunks from disk")
            return col
    except Exception:
        pass

    chunks = load_and_chunk(PDF_PATH)
    col    = db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=emb.encode(chunks).tolist(),
    )
    print(f"Embedded and saved {col.count()} chunks")
    return col


def retrieve(query, col):
    results = col.query(query_embeddings=emb.encode([query]).tolist(), n_results=TOP_K, include=["documents", "distances"])
    chunks  = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        if (1 - dist) >= MIN_SCORE:
            chunks.append({"doc": doc, "score": round(1 - dist, 4)})
    return chunks


def ask(query, col):
    chunks = retrieve(query, col)
    if not chunks:
        return "No relevant chunks found. Try lowering MIN_SCORE."
    context = "\n\n---\n\n".join([c["doc"] for c in chunks])
    prompt  = f"Use ONLY the context below. Do not use general knowledge.\n\nContext:\n{context}\n\nQuestion: {query}"
    print(f"Retrieved {len(chunks)} chunks | scores: {[c['score'] for c in chunks]}")
    return ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])["message"]["content"]


col = get_or_build_collection()

while True:
    q = input("\nAsk: ").strip()
    if q.lower() in ("quit", "exit"):
        break
    print("\n" + ask(q, col))