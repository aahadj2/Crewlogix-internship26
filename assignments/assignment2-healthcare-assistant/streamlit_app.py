"""
Streamlit web interface for the Dr. Jawad Zaheer Clinic Healthcare Support
Assistant. Reuses the exact same RAG pipeline, retriever tool, agent, and
LangGraph workflow defined in app.py - this file is only a UI layer.

Run with: streamlit run streamlit_app.py
"""

import sqlite3
import uuid

import streamlit as st

from app import _workflow, CHAT_DB_PATH, run_graph
from langgraph.checkpoint.sqlite import SqliteSaver

st.set_page_config(page_title="Dr. Jawad Zaheer Clinic Assistant", page_icon="🏥")

st.title("🏥 Dr. Jawad Zaheer Clinic - Healthcare Support Assistant")
st.caption(
    "Ask about appointment booking, doctor availability, consultation fees, "
    "lab test instructions, medicine refills, emergency guidance, report "
    "collection, or clinic timings. Answers are grounded only in the clinic's "
    "own policy documents."
)


@st.cache_resource
def get_checkpointer_and_graph():
    # Streamlit can rerun the script from different internal threads, so the
    # sqlite3 connection must tolerate cross-thread use. Building the
    # connection directly (instead of the from_conn_string context manager,
    # which closes the connection when its "with" block exits) keeps it open
    # for the lifetime of the cached resource.
    conn = sqlite3.connect(CHAT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = _workflow.compile(checkpointer=checkpointer)
    return graph


graph = get_checkpointer_and_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        with st.spinner("Checking clinic documents..."):
            answer = run_graph(graph, config, user_input)
        st.markdown(answer)

    st.session_state.history.append(("assistant", answer))

with st.sidebar:
    st.subheader("Session")
    st.write(f"Thread ID: `{st.session_state.thread_id[:8]}...`")
    if st.button("Start new conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    st.subheader("Try a test question")
    for q in [
        "How can I book an appointment?",
        "What time should I arrive before my appointment?",
        "How long should I fast before a fasting blood sugar test?",
        "Can I drink water while fasting for the test?",
        "What should I do in case of a medical emergency?",
    ]:
        st.code(q, language=None)
