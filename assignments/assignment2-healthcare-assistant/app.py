

import os
import re
import sys
import uuid
from typing import Annotated, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver



DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
CHAT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")

MODEL_ID = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

FALLBACK_MESSAGE = (
    "I could not find this information in the provided clinic documents. "
    "Please contact the clinic staff for confirmation."
)
GENERIC_SAFETY_MESSAGE = (
    "I am not a doctor and cannot provide medical diagnosis or treatment advice. "
    "Please consult a qualified healthcare professional."
)
EMERGENCY_SAFETY_MESSAGE = (
    "I am not a doctor and cannot provide medical diagnosis or treatment advice. "
    "Please seek urgent medical help or contact emergency services."
)

EMERGENCY_SYMPTOM_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "unconscious", "not breathing",
    "can't breathe", "cannot breathe", "severe bleeding", "heavy bleeding",
    "seizure", "overdose", "suicidal", "broken bone", "severe allergic reaction",
    "difficulty breathing", "poisoning", "choking",
]

MEDICAL_ADVICE_PATTERNS = [
    "what medicine should i take", "what medication should i take",
    "what should i take for", "can you diagnose", "diagnose me",
    "prescribe me", "recommend a medicine", "recommend medicine",
    "what drug should i", "what pill should i",
]

SYMPTOM_WORDS = [
    "headache", "fever", "cough", "cold", "stomach ache", "body ache",
    "nausea", "rash", "sore throat", "migraine", "toothache", "back pain",
    "vomiting", "diarrhea",
]


def is_emergency(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in EMERGENCY_SYMPTOM_KEYWORDS)


def is_medical_advice_request(question: str) -> bool:
    q = question.lower()
    if any(p in q for p in MEDICAL_ADVICE_PATTERNS):
        return True
    if any(s in q for s in SYMPTOM_WORDS) and any(
        w in q for w in ["medicine", "medication", "treat", "cure", "prescription", "drug", "pill"]
    ):
        return True
    return False


#rag pipeline

def build_vectorstore() -> Chroma:
    """Load clinic docs, split into chunks, embed locally via Ollama, and persist in Chroma."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    loader = DirectoryLoader(
        DOCS_DIR, glob="*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR,
    )
    return vectorstore


_vectorstore = build_vectorstore()
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 6})


#tools

@tool
def clinic_policy_search(query: str) -> str:
    """Use this tool to search healthcare clinic policies about appointments,
    doctor schedules, lab tests, fees, medicine refills, emergencies, and
    report collection."""
    docs: list[Document] = _retriever.invoke(query)
    if not docs:
        return "NO_RESULTS_FOUND"
    parts = []
    for d in docs:
        source = os.path.basename(d.metadata.get("source", "unknown"))
        parts.append(f"[Source: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


_SELF_CITATION_PATTERN = re.compile(
    r"\n*\(?\[?Source:\s*[^\)\]\n]*\)?\]?\n*", re.IGNORECASE
)


def strip_self_citations(text: str) -> str:
    """Small local models occasionally echo a '(Source: ...)' citation into
    their own answer text (imitating the pattern seen in tool results). The
    app appends its own canonical citation from the retrieved documents, so
    strip any self-generated one first to avoid a duplicate."""
    return _SELF_CITATION_PATTERN.sub("\n", text).strip()


def extract_sources(messages: Sequence[BaseMessage]) -> list[str]:
    sources: list[str] = []
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str):
            for match in re.findall(r"\[Source: ([^\]]+)\]", content):
                if match not in sources:
                    sources.append(match)
    return sources


#agent

SYSTEM_PROMPT = f"""You are a Healthcare Clinic Support Assistant for Dr. Jawad Zaheer
Clinic in Lahore, Pakistan. Patients will ask about appointment booking, doctor
availability, consultation fees, lab test instructions, medicine refill policy,
emergency guidance, patient report collection, and clinic timings.

Rules you must always follow:
1. You MUST call the clinic_policy_search tool for every clinic-related question
   before answering. Never answer from your own general knowledge about the clinic.
2. Base your answer ONLY on the retrieved tool results. Do not invent or assume
   any clinic detail that is not present in the retrieved text.
3. If the retrieved information does not answer the question (or the tool returns
   NO_RESULTS_FOUND), respond with EXACTLY this sentence and nothing else:
   "{FALLBACK_MESSAGE}"
4. Never give a medical diagnosis, treatment plan, or medicine recommendation.
   If asked for one, respond with EXACTLY this sentence and nothing else:
   "{GENERIC_SAFETY_MESSAGE}"
5. Keep answers concise, warm, and professional. When you use retrieved
   information, you do not need to repeat the [Source: ...] tags yourself -
   the application will attach citations automatically.
"""

_llm = ChatOllama(model=MODEL_ID, temperature=0)
_react_agent = create_agent(_llm, tools=[clinic_policy_search], system_prompt=SYSTEM_PROMPT)


#langgraph workflow

class ClinicState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    answer: str
    sources: list[str]
    found_answer: bool
    is_emergency: bool
    needs_generic_safety: bool


def receive_question(state: ClinicState) -> dict:
    question = state["messages"][-1].content
    emergency = is_emergency(question)
    advice = (not emergency) and is_medical_advice_request(question)
    return {
        "question": question,
        "is_emergency": emergency,
        "needs_generic_safety": advice,
    }


def route_after_receive(state: ClinicState) -> str:
    if state["is_emergency"] or state["needs_generic_safety"]:
        return "safety_response"
    return "agent_executor"


def safety_response(state: ClinicState) -> dict:
    text = EMERGENCY_SAFETY_MESSAGE if state["is_emergency"] else GENERIC_SAFETY_MESSAGE
    return {"answer": text, "sources": [], "found_answer": True}


def agent_executor(state: ClinicState) -> dict:
    input_messages = state["messages"]
    result = _react_agent.invoke({"messages": input_messages})
    result_messages = result["messages"]

    new_messages = result_messages[len(input_messages):]
    final_ai_message = result_messages[-1]
    answer_text = strip_self_citations(final_ai_message.content)
    sources = extract_sources(new_messages)
    found = "could not find this information" not in answer_text.lower()
    return {"answer": answer_text, "sources": sources, "found_answer": found}


def save_memory(state: ClinicState) -> dict:

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"PATIENT: {state['question']}\nASSISTANT: {state['answer']}\n\n")
    return {}


def route_after_save_memory(state: ClinicState) -> str:
    return "final_answer" if state["found_answer"] else "fallback_response"


def final_answer(state: ClinicState) -> dict:
    text = state["answer"]
    if state["sources"]:
        text = f"{text}\n\n(Source: {', '.join(state['sources'])})"
    return {"messages": [AIMessage(content=text)]}


def fallback_response(_state: ClinicState) -> dict:
    return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}


def build_workflow() -> StateGraph:
    workflow = StateGraph(ClinicState)

    workflow.add_node("receive_question", receive_question)
    workflow.add_node("agent_executor", agent_executor)
    workflow.add_node("safety_response", safety_response)
    workflow.add_node("save_memory", save_memory)
    workflow.add_node("final_answer", final_answer)
    workflow.add_node("fallback_response", fallback_response)

    workflow.add_edge(START, "receive_question")
    workflow.add_conditional_edges(
        "receive_question",
        route_after_receive,
        {"safety_response": "safety_response", "agent_executor": "agent_executor"},
    )
    workflow.add_edge("safety_response", "save_memory")
    workflow.add_edge("agent_executor", "save_memory")
    workflow.add_conditional_edges(
        "save_memory",
        route_after_save_memory,
        {"final_answer": "final_answer", "fallback_response": "fallback_response"},
    )
    workflow.add_edge("final_answer", END)
    workflow.add_edge("fallback_response", END)

    return workflow


_workflow = build_workflow()


#test

TEST_QUESTIONS = [
    "How can I book an appointment?",
    "What time should I arrive before my appointment?",
    "How long should I fast before a fasting blood sugar test?",
    "Can I drink water while fasting for the test?",
    "What should I do in case of a medical emergency?",
]

EXTRA_DEMO_QUESTIONS = [
    ("What is the consultation fee for Dr. Fatima Jawad?", "follow-up (memory demo, refers to prior context)"),
    ("Do you offer heart surgery at this clinic?", "fallback demo (info not in documents)"),
    ("I have chest pain. What medicine should I take?", "safety/emergency demo"),
]


def run_graph(graph, config, user_text: str) -> str:
    result = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
    return result["messages"][-1].content


def run_test_suite(graph, config) -> None:
    print("=" * 70)
    print("RUNNING REQUIRED TEST QUESTIONS")
    print("=" * 70)
    for q in TEST_QUESTIONS:
        print(f"\nPatient: {q}")
        print(f"Assistant: {run_graph(graph, config, q)}")

    print("\n" + "=" * 70)
    print("EXTRA DEMOS: memory follow-up / fallback / safety")
    print("=" * 70)
    for q, label in EXTRA_DEMO_QUESTIONS:
        print(f"\n[{label}]")
        print(f"Patient: {q}")
        print(f"Assistant: {run_graph(graph, config, q)}")


def main() -> None:
    with SqliteSaver.from_conn_string(CHAT_DB_PATH) as checkpointer:
        graph = _workflow.compile(checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            run_test_suite(graph, config)
            return

        print("Dr. Jawad Zaheer Clinic - Healthcare Support Assistant")
        print("Type 'exit' to quit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            answer = run_graph(graph, config, user_input)
            print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
