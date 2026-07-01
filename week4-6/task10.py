import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from datasets import Dataset
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import asyncio
def load():
    return PyPDFLoader("new-approaches-and-procedures-for-cancer-treatment.pdf").load()

def chunk(docs):
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    print(f"Chunk size: 500 | Overlap: 100 | Total chunks: {len(chunks)}")
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
        {"question": "What is stem cell therapy?",                             "ground_truth": "Stem cell therapy uses undifferentiated cells from bone marrow to treat cancer by regenerating damaged tissues."},
        {"question": "What is targeted drug therapy?",                         "ground_truth": "Targeted drug therapy uses drugs that interfere with specific growth molecules to block cancer growth and spreading."},
        {"question": "What are the side effects of stem cell therapy?",        "ground_truth": "Side effects include tumorigenesis, drug toxicity, increased immune responses, and viral infection."},
        {"question": "What is cryoablation?",                                  "ground_truth": "Cryoablation ablates tissue by freezing to lethal temperatures, used to treat benign and malignant primary tumors."},
        {"question": "What are natural antioxidants used in cancer?",          "ground_truth": "Vitamins, polyphenols, curcumin, berberine, and quercetin are natural antioxidants used in cancer treatment."},
        {"question": "How many gene therapy trials are ongoing?",              "ground_truth": "Approximately 2900 gene therapy clinical trials are currently ongoing, two-thirds of which are related to cancer."},
        {"question": "What are advantages and disadvantages of targeted therapy?", "ground_truth": "Advantages: high specificity, reduced adverse reactions. Disadvantages: long-term side effects are in question."},
        {"question": "What year was quantum computing invented?",              "ground_truth": "Not found in documents."},
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

    llm        = ChatOllama(model="llama3.2")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    asyncio.set_event_loop(asyncio.new_event_loop())
    return evaluate(
    Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}),
    metrics=[faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings,
    run_config=RunConfig(timeout=120, max_retries=2, max_workers=1),
    )

if __name__ == "__main__":
    print("Loading")
    vectordb, bm25 = build_stores(chunk(load()))

    print("\nBase RAGAS")
    baseline = run_ragas(vectordb, bm25)
    print("Baseline:", baseline)

    reranker = CrossEncoder("BAAI/bge-reranker-base")

    print("\nImproved RAGAS")
    improved = run_ragas(vectordb, bm25, reranker)
    print("Improved:", improved)

    print(f"\n{'Metric':<25} {'Baseline':>10} {'Improved':>10}")
    print("-" * 47)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
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