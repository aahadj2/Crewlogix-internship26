"""
Task 7 - LangGraph University Course Registration Workflow
Concepts: StateGraph, TypedDict, conditional edges, loop, interrupt, InMemorySaver, Command
"""

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

MAX_ATTEMPTS = 3

# State
class State(TypedDict):
    messages:   Annotated[list[str], operator.add]  # reducer: appends each node's messages
    name:       str
    student_id: str
    gpa:        Optional[float]
    credits:    Optional[int]
    course:     str
    status:     str       # pending | incomplete | eligible | review | rejected | registered
    attempts:   int
    missing:    list[str]

# Nodes
def collect_student_info(state: State) -> dict:
    return {"messages": [f"Received info for: {state.get('name', '?')}"]}

def validate_info(state: State) -> dict:
    missing = []
    if not state.get("name"):        missing.append("name")
    if not state.get("student_id"):  missing.append("student_id")
    if state.get("gpa") is None:     missing.append("gpa")
    if state.get("credits") is None: missing.append("credits")
    if not state.get("course"):      missing.append("course")

    if missing:
        return {"messages": [f"Missing: {missing}"], "missing": missing, "status": "incomplete"}
    return {"messages": ["All fields valid."], "missing": [], "status": "pending"}

def ask_for_missing_info(state: State) -> dict:
    # Simulate student providing the missing data
    attempt = state.get("attempts", 0) + 1
    updates = {"messages": [f"Attempt {attempt}: filling {state['missing']}"], "attempts": attempt}
    defaults = {"gpa": 3.2, "credits": 75, "name": "Unknown", "student_id": "S000", "course": "GEN101"}
    for f in state["missing"]:
        updates[f] = defaults[f]
    return updates

def check_eligibility(state: State) -> dict:
    gpa, credits = state["gpa"], state["credits"]
    if gpa >= 3.0 and credits >= 60:
        return {"messages": [f"Eligible (GPA={gpa}, Credits={credits})"], "status": "eligible"}
    elif gpa >= 2.5 and credits >= 30:
        return {"messages": [f"Needs review (GPA={gpa}, Credits={credits})"], "status": "review"}
    return {"messages": [f"Rejected (GPA={gpa}, Credits={credits})"], "status": "rejected"}

def human_review(state: State) -> dict:
    # Pauses here — resume with: app.invoke(Command(resume="approved"), config=config)
    decision = interrupt("Advisor: approve or reject?")
    if decision == "approved":
        return {"messages": ["Advisor approved."], "status": "eligible"}
    return {"messages": ["Advisor rejected."], "status": "rejected"}

def register_course(state: State) -> dict:
    return {"messages": [f"REGISTERED: {state['name']} -> {state['course']}"], "status": "registered"}

def generate_rejection(state: State) -> dict:
    return {"messages": [f"REJECTED: {state['name']} -> {state['course']} ({state['status']})"]}

# Routing
def route_validate(state: State) -> str:
    if state["status"] == "incomplete":
        return "generate_rejection" if state.get("attempts", 0) >= MAX_ATTEMPTS else "ask_for_missing_info"
    return "check_eligibility"

def route_eligibility(state: State) -> str:
    return {"eligible": "register_course", "review": "human_review"}.get(state["status"], "generate_rejection")

def route_review(state: State) -> str:
    return "register_course" if state["status"] == "eligible" else "generate_rejection"

# Build graph
builder = StateGraph(State)
for name, fn in [
    ("collect_student_info", collect_student_info),
    ("validate_info",        validate_info),
    ("ask_for_missing_info", ask_for_missing_info),
    ("check_eligibility",    check_eligibility),
    ("human_review",         human_review),
    ("register_course",      register_course),
    ("generate_rejection",   generate_rejection),
]:
    builder.add_node(name, fn)

builder.add_edge(START, "collect_student_info")
builder.add_edge("collect_student_info", "validate_info")
builder.add_conditional_edges("validate_info", route_validate)
builder.add_edge("ask_for_missing_info", "validate_info")
builder.add_conditional_edges("check_eligibility", route_eligibility)
builder.add_conditional_edges("human_review", route_review)
builder.add_edge("register_course",   END)
builder.add_edge("generate_rejection", END)

app = builder.compile(checkpointer=InMemorySaver())

# Helper
def run(label, student, config):
    print(f"\n{label}")
    result = app.invoke(student, config=config)
    for m in result["messages"]:
        print(" ", m)

def make_student(name, sid, gpa, credits, course):
    return {"name": name, "student_id": sid, "gpa": gpa, "credits": credits,
            "course": course, "messages": [], "status": "pending", "attempts": 0, "missing": []}

# Test cases
if __name__ == "__main__":

    # Test 1: Direct Approval — high GPA and credits, goes straight to registration
    run("TEST 1: Direct Approval",
        make_student("Ali Khan", "S001", 3.7, 90, "CS401"),
        {"configurable": {"thread_id": "t1"}})

    # Test 2: Missing Info Loop — gpa and credits missing, filled in on retry
    run("TEST 2: Missing Info Loop",
        make_student("Sara Ahmed", "S002", None, None, "Math301"),
        {"configurable": {"thread_id": "t2"}})

    # Test 3: Human Review — mid GPA triggers interrupt, advisor approves
    cfg = {"configurable": {"thread_id": "t3"}}
    print("\nTEST 3: Human Review (advisor approves)")
    app.invoke(make_student("Bilal Raza", "S003", 2.7, 45, "ENG201"), config=cfg)
    result = app.invoke(Command(resume="approved"), config=cfg)
    for m in result["messages"]:
        print(" ", m)
