"""
Week 9 — Task 5
LUMS University AI Assistant
  Agent 1: RAG Policy Agent   — Q&A on LUMS policies from lums_policies.txt
  Agent 2: Fee Calculator     — estimates student semester fees
  Guardrails: input topic filter + output PII scrubber
  Tracing: LangSmith
  Memory: sliding-window memory + JSON persistence
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import AgentExecutor, create_react_agent

load_dotenv()

# Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "lums-university-assistant"

# LLM
llm = ChatOllama(model="llama3.2", temperature=0)


# Knowledge Base
_POLICIES_FILE = Path(__file__).parent / "lums_policies.txt"
LUMS_DOCUMENTS = TextLoader(str(_POLICIES_FILE), encoding="utf-8").load()


# RAG Setup
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
chunks = splitter.split_documents(LUMS_DOCUMENTS)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

rag_chain = (
    ChatPromptTemplate.from_template(
        "You are a helpful LUMS university assistant. "
        "Answer using ONLY the context below. "
        "If the answer is not in the context, say: "
        "'I don't have specific information about that. "
        "Please contact LUMS at +92-42-111-11-5867 or info@lums.edu.pk.'\n\n"
        "Recent conversation:\n{history}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    | llm
    | StrOutputParser()
)


# Memory
MEMORY_FILE = Path(__file__).parent / "lums_chat_history.json"


class ChatMemory:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []
        self._load()

    def add(self, role, content):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def get_window(self):
        recent = self.history[-(self.window_size * 2):]
        lines = [
            f"{'User' if m['role'] == 'human' else 'Assistant'}: {m['content']}"
            for m in recent
        ]
        return "\n".join(lines) if lines else "No previous conversation."

    def _load(self):
        if MEMORY_FILE.exists():
            try:
                self.history = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.history = []

    def _save(self):
        MEMORY_FILE.write_text(
            json.dumps(self.history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


memory = ChatMemory(window_size=5)


# Guardrails
_UNIVERSITY_KEYWORDS = {
    "lums", "university", "campus", "admission", "student", "course",
    "semester", "gpa", "fee", "scholarship", "hostel", "exam", "credit",
    "faculty", "department", "degree", "program", "academic", "tuition",
    "registration", "class", "lecture", "assignment", "grade", "transcript",
    "policy", "attendance", "withdraw", "probation", "dean", "rector",
    "sdsb", "sbasse", "mgshss", "sahsol", "redc", "mba", "phd",
}

_HARMFUL_KEYWORDS = {
    "hack", "exploit", "bomb", "weapon", "violence", "illegal",
    "drug trafficking", "murder", "attack",
}

_OFF_TOPIC_KEYWORDS = {
    "weather forecast", "stock price", "cricket score", "recipe", "movie review",
    "song lyrics", "celebrity gossip",
}

_PII_PATTERNS = [
    r"\b\d{5}-\d{7}-\d\b",
    r"\b(?:\+92|0)?3\d{2}[-\s]?\d{7}\b",
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
]

_REJECTION_MSG = (
    "I'm the LUMS University Assistant and can only answer questions about "
    "LUMS admissions, academic policies, fees, scholarships, campus services, "
    "or student life. Please ask a university-related question."
)


def input_guardrail(query):
    q = query.lower()
    if any(kw in q for kw in _HARMFUL_KEYWORDS):
        return False, _REJECTION_MSG
    if any(kw in q for kw in _UNIVERSITY_KEYWORDS):
        return True, ""
    if len(query.split()) <= 5:  # short follow-ups pass through
        return True, ""
    if any(kw in q for kw in _OFF_TOPIC_KEYWORDS):
        return False, _REJECTION_MSG
    return True, ""


def output_guardrail(text):
    for pattern in _PII_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


# Agent 1: RAG Policy Agent
@tool
def query_lums_policies(question: str) -> str:
    """
    Answer questions about LUMS university policies, admissions, academic rules,
    attendance, examinations, scholarships, campus facilities, hostel, and
    student conduct using the LUMS knowledge base.
    Input: a question about LUMS.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    history = memory.get_window()
    return rag_chain.invoke({"context": context, "question": question, "history": history})


_RAG_PROMPT = PromptTemplate.from_template(
    "You are Agent 1 — LUMS University Policy Expert. "
    "Answer questions about LUMS admissions, academic rules, scholarships, "
    "campus facilities, hostel, and student conduct.\n\n"
    "Tools available:\n{tools}\n\n"
    "Format:\n"
    "Question: the input question\n"
    "Thought: what to do\n"
    "Action: one of [{tool_names}]\n"
    "Action Input: input to the action\n"
    "Observation: result\n"
    "... (repeat as needed)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)

_rag_agent = create_react_agent(llm=llm, tools=[query_lums_policies], prompt=_RAG_PROMPT)
rag_agent_executor = AgentExecutor(
    agent=_rag_agent,
    tools=[query_lums_policies],
    verbose=True,
    max_iterations=4,
    handle_parsing_errors=True,
)


# Agent 2: Fee Calculator Agent
_FEE_TABLE = {
    "undergraduate": {
        "sbasse": 52_000,
        "sdsb":   52_000,
        "sahsol": 50_000,
        "mgshss": 48_000,
    },
    "graduate": {
        "mba": 70_000,
        "ms":  58_000,
        "phd": 42_000,
    },
    "fixed": {
        "student_services": 30_000,
        "technology":       15_000,
        "health_services":  10_000,
        "library":           8_000,
    },
    "hostel": {"shared": 60_000, "single": 90_000},
    "meal":   {"standard": 25_000, "premium": 35_000},
    "annual_registration": 25_000,
}

_SCHOOL_MAP = {
    "sbasse":  "sbasse", "science":    "sbasse", "engineering": "sbasse",
    "sdsb":    "sdsb",   "business":   "sdsb",   "management":  "sdsb",
    "mgshss":  "mgshss", "arts":       "mgshss", "humanities":  "mgshss",
    "social":  "mgshss",
    "sahsol":  "sahsol", "law":        "sahsol",
    "mba":     "mba",    "ms":         "ms",     "phd":         "phd",
    "masters": "ms",
}


@tool
def calculate_lums_fee(query: str) -> str:
    """
    Calculate a LUMS student's estimated semester fee.
    Input: describe program, school, credit hours, and optional hostel/meal/scholarship.
    Examples: "SBASSE 15 credit hours shared hostel standard meal"
              "MBA 12 credits no hostel 50% scholarship"
    """
    q = query.lower()

    is_grad = any(k in q for k in ("mba", "ms ", " ms,", "phd", "master", "graduate", "grad"))

    school = None
    for alias, key in _SCHOOL_MAP.items():
        if alias in q:
            school = key
            break
    if school is None:
        school = "mba" if is_grad else "sbasse"

    ch_match = re.search(r"(\d{1,2})\s*(?:credit hours?|credits?|ch\b|cr\b)", q)
    if ch_match:
        credit_hours = int(ch_match.group(1))
    else:
        nums = re.findall(r"\b(9|1[0-9]|2[01])\b", q)
        credit_hours = int(nums[0]) if nums else 15
    credit_hours = max(9, min(21, credit_hours))

    if is_grad or school in ("mba", "ms", "phd"):
        rate = _FEE_TABLE["graduate"].get(school, _FEE_TABLE["graduate"]["ms"])
        level_label = "Graduate"
    else:
        rate = _FEE_TABLE["undergraduate"].get(school, _FEE_TABLE["undergraduate"]["sbasse"])
        level_label = "Undergraduate"

    tuition = rate * credit_hours
    fixed_total = sum(_FEE_TABLE["fixed"].values())

    hostel_fee, hostel_label = 0, None
    if "single" in q:
        hostel_fee, hostel_label = _FEE_TABLE["hostel"]["single"], "Single room"
    elif any(k in q for k in ("shared", "hostel", "accommodation", "dorm", "room")):
        hostel_fee, hostel_label = _FEE_TABLE["hostel"]["shared"], "Shared room"

    meal_fee, meal_label = 0, None
    if "premium meal" in q or "premium" in q:
        meal_fee, meal_label = _FEE_TABLE["meal"]["premium"], "Premium"
    elif any(k in q for k in ("meal", "food", "standard meal", "dining")):
        meal_fee, meal_label = _FEE_TABLE["meal"]["standard"], "Standard"

    scholarship_pct = 0
    for pct in (100, 75, 50, 25):
        if f"{pct}%" in q or f"{pct} percent" in q:
            scholarship_pct = pct
            break

    gross_total = tuition + fixed_total + hostel_fee + meal_fee
    discount = int(tuition * scholarship_pct / 100)
    net_total = gross_total - discount

    lines = [
        f"LUMS Fee Estimate - {level_label} | {school.upper()} | {credit_hours} credit hours",
        f"Tuition: PKR {tuition:,}",
        f"Fixed fees (services/tech/health/library): PKR {fixed_total:,}",
    ]
    if hostel_label:
        lines.append(f"Hostel ({hostel_label}): PKR {hostel_fee:,}")
    if meal_label:
        lines.append(f"Meal Plan ({meal_label}): PKR {meal_fee:,}")
    lines.append(f"Gross Total: PKR {gross_total:,}")
    if scholarship_pct:
        lines.append(f"Scholarship Discount ({scholarship_pct}%): PKR {discount:,}")
        lines.append(f"Net Payable: PKR {net_total:,}")
    lines.append(f"Note: Annual registration fee PKR {_FEE_TABLE['annual_registration']:,} charged once per year.")
    return "\n".join(lines)


_FEE_PROMPT = PromptTemplate.from_template(
    "You are Agent 2 — LUMS Fee Calculator. "
    "Help students estimate their semester fees based on school, credit hours, "
    "hostel, meal plan, and scholarship.\n\n"
    "Tools available:\n{tools}\n\n"
    "Format:\n"
    "Question: the input question\n"
    "Thought: what to do\n"
    "Action: one of [{tool_names}]\n"
    "Action Input: input to the action\n"
    "Observation: result\n"
    "... (repeat as needed)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)

_fee_agent = create_react_agent(llm=llm, tools=[calculate_lums_fee], prompt=_FEE_PROMPT)
fee_agent_executor = AgentExecutor(
    agent=_fee_agent,
    tools=[calculate_lums_fee],
    verbose=True,
    max_iterations=4,
    handle_parsing_errors=True,
)


# Router
_FEE_INTENT_KEYWORDS = {
    "calculate", "how much", "total fee", "fee breakdown", "cost",
    "per semester", "credit hour rate", "hostel fee",
    "tuition fee", "semester fee", "annual fee",
}


def _route(query):
    q = query.lower()
    fee_hits = sum(1 for kw in _FEE_INTENT_KEYWORDS if kw in q)
    if fee_hits >= 1 and any(k in q for k in ("credit", "semester", "tuition", "hostel", "meal")):
        return "fee"
    if any(k in q for k in ("calculate my fee", "how much will i pay", "what is the fee for")):
        return "fee"
    return "rag"


# Main chat function
def chat(query):
    allowed, rejection = input_guardrail(query)
    if not allowed:
        return rejection

    memory.add("human", query)

    agent_type = _route(query)
    try:
        executor = fee_agent_executor if agent_type == "fee" else rag_agent_executor
        result = executor.invoke({"input": query})
        response = result.get("output", "I could not process your request. Please try rephrasing.")
    except Exception as exc:
        response = (
            f"An error occurred ({str(exc)[:120]}). "
            "Please try rephrasing or contact LUMS at +92-42-111-11-5867."
        )

    response = output_guardrail(response)
    memory.add("assistant", response)
    return response


# Demo
if __name__ == "__main__":
    demo_queries = [
        # RAG agent queries
        "What are the undergraduate admission requirements at LUMS?",
        "What is the GPA threshold for academic probation at LUMS?",
        "What scholarships does LUMS offer for financially disadvantaged students?",
        "What is the attendance policy and what happens if I miss too many classes?",
        # Fee calculator queries
        "Calculate my fee: SBASSE undergraduate, 18 credit hours, shared hostel, standard meal plan",
        "How much will an MBA student pay for 12 credit hours with no hostel and 50% scholarship?",
        "What is the semester fee for MGSHSS 15 credit hours with single room and premium meal?",
        # Guardrail tests
        "What is the weather forecast for Lahore today?",
        "How do I hack into the LUMS SIS portal?",
    ]

    for query in demo_queries:
        print(f"\nUSER: {query}")
        answer = chat(query)
        print(f"ASSISTANT: {answer}\n")
