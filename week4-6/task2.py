import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import pymupdf

def load_pdf(path: str, chunk_size: int = 500) -> list[str]:
    doc = pymupdf.open(path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

PDF_PATH = "./The_Plan_of_the_Giza_Pyramids.pdf"
documents = load_pdf(PDF_PATH)
print(f"Loaded {len(documents)} chunks from {PDF_PATH}")


model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.array(model.encode(documents), dtype=np.float32)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"Stored vectors: {index.ntotal}")


while True:
    query = input("\nAsk a question: ")
    if query.lower() == "exit":
        break

    query_vector = np.array(model.encode([query]), dtype=np.float32)
    distances, indices = index.search(query_vector, k=3)

    retrieved = [documents[i] for i in indices[0]]
    context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved))

    print("\nRetrieved chunks:")
    for i, (dist, chunk) in enumerate(zip(distances[0], retrieved)):
        print(f"  [{i+1}] (dist={dist:.3f}) {chunk[:80]}…")

    
    response = ollama.chat(
        model="llama3.2",
        messages=[{
            "role": "user",
            "content": (
                f"Answer the question using only the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )
        }]
    )

    print(f"\nAnswer:\n{response['message']['content']}")