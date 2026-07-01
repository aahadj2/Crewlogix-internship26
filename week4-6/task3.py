import os
import ollama  # Native Ollama python library
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Setup absolute or relative paths
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
pdf_path = os.path.join(current_dir, "test.pdf")
docx_path = os.path.join(current_dir, "Stats.docx")
web_url = "https://en.wikipedia.org/wiki/Cancer"

documents = []

print("Loading data from sources...")
# PDF Loader
if os.path.exists(pdf_path):
    documents.extend(PyPDFLoader(pdf_path).load())
else:
    print(f"Warning: '{pdf_path}' not found.")

# DOCX Loader
if os.path.exists(docx_path):
    documents.extend(Docx2txtLoader(docx_path).load())
else:
    print(f"Warning: '{docx_path}' not found.")

# Web Loader (WebBaseLoader successfully parses direct raw URLs)
try:
    documents.extend(WebBaseLoader(web_url).load())
except Exception as e:
    print(f"Web loading failed: {e}")

print("Total documents loaded:", len(documents))

# 2. Split Data into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)
print("Total chunks created:", len(chunks))

# 3. Create Local Embeddings using your exact model name
print("Generating embeddings via llama3.2:latest and indexing into FAISS...")
embedding_model = OllamaEmbeddings(model="llama3.2:latest")

# 4. Store Embeddings in local Vector Database
vector_db = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)
vector_db.save_local("faiss_vector_database")
print("Vector database saved successfully.")

# 5. Initialize the similarity retriever
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 6. Local RAG Function using ollama.chat format
def answer_question(question):
    # Perform Cosine Similarity retrieval from local FAISS index
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    print("Forwarding to local llama3.2:latest chat thread...")
    
    # Use your preferred native ollama.chat layout structure
    response = ollama.chat(
        model="llama3.2:latest",
        messages=[{
            "role": "user",
            "content": (
                f"You are a helpful assistant.\n\n"
                f"Answer the user question using only the provided context.\n"
                f"If the answer is not present in the context, say:\n"
                f"\"The answer is not available in the provided context.\"\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                f"Final Answer:"
            )
        }]
    )

    print("\n================ FINAL ANSWER ================\n")
    print(response['message']['content'])

    print("\n================ RETRIEVED SOURCES ================\n")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"Source {i}")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:500])
        print("-" * 80)

# 7. Interactive Query Loop
while True:
    query = input("\nAsk a question or type 'exit': ")

    if query.lower() == "exit":
        print("Exiting pipeline setup.")
        break

    if query.strip():
        answer_question(query)