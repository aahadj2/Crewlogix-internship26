import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

PDF_PATHS = [
    "Monday6.pdf",
    "Tuesday6.pdf",
    "Wednesday6.pdf",
    "Thursday6.pdf",
]

QA_PAIRS = [
    {"question": "What is BM25 and why is it used?",                          "ground_truth": "BM25 is a keyword retrieval algorithm that rewards rare terms and accounts for term frequency and document length."},
    {"question": "What is Reciprocal Rank Fusion?",                           "ground_truth": "RRF combines two ranked lists using 1/(k+rank) per list summed across lists."},
    {"question": "What is a cross-encoder?",                                  "ground_truth": "A cross-encoder encodes the query and document together for joint attention giving more accurate relevance scores."},
    {"question": "What is HyDE?",                                             "ground_truth": "HyDE generates a hypothetical answer using an LLM then embeds that answer for retrieval instead of the original query."},
    {"question": "What are the 4 RAGAS metrics?",                            "ground_truth": "Faithfulness, Context Precision, Context Recall, and Answer Relevancy."},
    {"question": "What is step-back prompting?",                             "ground_truth": "Step-back prompting rewrites a specific query into a broader version before retrieval."},
    {"question": "What is ChromaDB persistence?",                            "ground_truth": "ChromaDB persistence saves the collection to disk so documents do not need to be re-embedded on every run."},
    {"question": "What year was the Eiffel Tower built?",                    "ground_truth": "Not found in documents."},
]

def load():
    docs = []
    for path in PDF_PATHS:
        docs.extend(PyPDFLoader(path).load())
    print(f"Loaded {len(docs)} pages")
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
    prompt = f"Answer using only this context. If not found say 'Not found in documents'.\n\nContext:\n{context}\n\nQuestion: {query}"
    return ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])["message"]["content"]

def score(ans, context, ground_truth):
    a = set(ans.lower().split())
    c = set(context.lower().split())
    g = set(ground_truth.lower().split())
    faith   = len(a & c) / len(a) if a else 0
    relev   = len(a & g) / len(g) if g else 0
    recall  = len(g & c) / len(g) if g else 0
    return round(faith, 3), round(relev, 3), round(recall, 3)

def evaluate(vectordb, bm25, reranker=None):
    faith_scores, relev_scores, recall_scores = [], [], []
    for qa in QA_PAIRS:
        docs = retrieve(qa["question"], vectordb, bm25, reranker)
        ans  = answer(qa["question"], docs)
        ctx  = " ".join(d.page_content for d in docs)
        f, r, rc = score(ans, ctx, qa["ground_truth"])
        faith_scores.append(f)
        relev_scores.append(r)
        recall_scores.append(rc)
        print(f"Q: {qa['question']}\nA: {ans[:150]}\nFaithfulness: {f} | Relevancy: {r} | Context Recall: {rc}\n")
    return (
        round(sum(faith_scores)  / len(faith_scores),  3),
        round(sum(relev_scores)  / len(relev_scores),  3),
        round(sum(recall_scores) / len(recall_scores), 3),
    )
if __name__ == "__main__":
    print("Loading docs")
    vectordb, bm25 = build_stores(chunk(load()))

    print("\nBaseline evaluation")
    bf, br, brc = evaluate(vectordb, bm25)
    print(f"Baseline -> Faithfulness: {bf} | Relevancy: {br} | Context Recall: {brc}")

    reranker = CrossEncoder("BAAI/bge-reranker-base")

    print("\nImproved evaluation")
    if_, ir, irc = evaluate(vectordb, bm25, reranker)
    print(f"Improved -> Faithfulness: {if_} | Relevancy: {ir} | Context Recall: {irc}")

    print(f"\n{'Metric':<20} {'Baseline':>10} {'Improved':>10}")
    print("-" * 42)
    print(f"{'Faithfulness':<20} {bf:>10} {if_:>10}")
    print(f"{'Answer Relevancy':<20} {br:>10} {ir:>10}")
    print(f"{'Context Recall':<20} {brc:>10} {irc:>10}")