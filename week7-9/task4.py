"""
Week 8 — Task 4
RAG + FBR Tax Multi-Tool Agent:
  Tool 1 : answer_from_pdf  — Q&A over new-approaches.pdf via RAG
  Tool 2 : get_fbr_tax_slab — calculates FBR income tax for Pakistani filers
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import AgentExecutor, create_react_agent

load_dotenv()


# RAG Setup
PDF_PATH = os.path.join(os.path.dirname(__file__), "new-approaches.pdf")

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="llama3.2", temperature=0)

rag_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)
rag_chain = rag_prompt | llm | StrOutputParser()


# Tool 1: RAG Q&A
@tool
def answer_from_pdf(question: str) -> str:
    """
    Answer questions about the content of the new-approaches document (PDF).
    Use this tool when the user asks about topics, concepts, strategies, or
    information that may be covered in the uploaded document.
    Input: a question string.
    Returns: an answer grounded in the PDF content.
    """
    retrieved = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in retrieved)
    return rag_chain.invoke({"context": context, "question": question})


# Tool 2: FBR Tax Calculator
@tool
def get_fbr_tax_slab(annual_income_pkr: str) -> str:
    """
    Calculate the applicable FBR income tax slab for a Pakistani individual filer.
    Use this tool when the user asks about income tax, tax brackets,
    tax liability, or FBR tax rates in Pakistan.
    Input: annual income in Pakistani Rupees (PKR) as a number (e.g. 1800000 or 1,800,000).
    Returns: the applicable tax slab, rate, and tax payable as a string.
    """
    import re
    annual_income_pkr = float(re.sub(r"[^\d.]", "", str(annual_income_pkr)))
    if annual_income_pkr <= 600_000:
        return f"PKR {annual_income_pkr:,.0f}: Exempt — no income tax applicable."
    elif annual_income_pkr <= 1_200_000:
        tax = (annual_income_pkr - 600_000) * 0.05
        return (f"PKR {annual_income_pkr:,.0f}: Slab 2 — 5% on amount above 600,000. "
                f"Tax payable: PKR {tax:,.0f}")
    elif annual_income_pkr <= 2_200_000:
        tax = 30_000 + (annual_income_pkr - 1_200_000) * 0.15
        return (f"PKR {annual_income_pkr:,.0f}: Slab 3 — 15% on amount above 1,200,000. "
                f"Tax payable: PKR {tax:,.0f}")
    elif annual_income_pkr <= 3_200_000:
        tax = 180_000 + (annual_income_pkr - 2_200_000) * 0.25
        return (f"PKR {annual_income_pkr:,.0f}: Slab 4 — 25% on amount above 2,200,000. "
                f"Tax payable: PKR {tax:,.0f}")
    else:
        tax = 430_000 + (annual_income_pkr - 3_200_000) * 0.35
        return (f"PKR {annual_income_pkr:,.0f}: Slab 5 — 35% on amount above 3,200,000. "
                f"Tax payable: PKR {tax:,.0f}")


# Agent
tools = [answer_from_pdf, get_fbr_tax_slab]
prompt = PromptTemplate.from_template(
    "Answer the following questions as best you can. "
    "You have access to the following tools:\n\n"
    "{tools}\n\n"
    "Use the following format:\n\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True,
)


def run_agent(query: str) -> str:
    result = agent_executor.invoke({"input": query})
    return result["output"]



if __name__ == "__main__":
    test_queries = [
        "What new approaches are discussed in the document?",
        "My annual income is PKR 1,800,000. What is my FBR income tax?",
        "What is the tax on PKR 500,000 annual salary?",
    ]

    for query in test_queries:
        print("\n" + "=" * 65)
        print(f"USER: {query}")
        print("=" * 65)
        answer = run_agent(query)
        print(f"\nFINAL ANSWER:\n{answer}")
        
