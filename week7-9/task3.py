"""
Week 8 — Task 3
Two-tool ReAct Agent:
  Tool 1 : DuckDuckGoSearchRun  — answers any general web query
  Tool 2 : get_weather (custom) — fetches live weather via RapidAPI
The AgentExecutor runs the ReAct loop and returns a Final Answer.
"""

import os
# import requests
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_core.tools import tool
from langchain_classic import hub

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

# Add your RapidAPI key to .env as:  RAPIDAPI_KEY=<your_key>
# Sign up at https://rapidapi.com and subscribe to "WeatherAPI.com"
# RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")


# ── Tool 1 : Web Search Agent ─────────────────────────────────────────────────
web_search = DuckDuckGoSearchRun(
    name="web_search",
    description=(
        "Search the web for any user query. "
        "Use this for general knowledge, news, facts, or any topic "
        "that does NOT involve weather. "
        "Input: a search query string."
    ),
)


# ── Tool 2 : Weather Agent (custom RapidAPI tool) ─────────────────────────────
# @tool
# def get_weather(city: str) -> str:
#     """
#     Return current weather details for a given city using RapidAPI WeatherAPI.
#     Input must be a city name, e.g. 'Lahore', 'London', 'New York'.
#     Use this whenever the user asks about weather, temperature, humidity, or climate.
#     """
#     if not RAPIDAPI_KEY:
#         return (
#             "RAPIDAPI_KEY is not set. "
#             "Add RAPIDAPI_KEY=<your_key> to your .env file "
#             "and subscribe to WeatherAPI.com on RapidAPI."
#         )
#
#     url = "https://weatherapi-com.p.rapidapi.com/current.json"
#     headers = {
#         "X-RapidAPI-Key": RAPIDAPI_KEY,
#         "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com",
#     }
#     params = {"q": city}
#
#     try:
#         resp = requests.get(url, headers=headers, params=params, timeout=10)
#         resp.raise_for_status()
#         data = resp.json()
#
#         if "error" in data:
#             return f"WeatherAPI error: {data['error']['message']}"
#
#         loc = data["location"]
#         cur = data["current"]
#         return (
#             f"Weather in {loc['name']}, {loc['region']}, {loc['country']}:\n"
#             f"  Condition   : {cur['condition']['text']}\n"
#             f"  Temperature : {cur['temp_c']}°C  /  {cur['temp_f']}°F\n"
#             f"  Feels like  : {cur['feelslike_c']}°C\n"
#             f"  Humidity    : {cur['humidity']}%\n"
#             f"  Wind        : {cur['wind_kph']} kph {cur['wind_dir']}\n"
#             f"  Visibility  : {cur['vis_km']} km\n"
#             f"  UV Index    : {cur['uv']}"
#         )
#     except requests.exceptions.HTTPError as e:
#         return f"HTTP error fetching weather: {e}"
#     except Exception as e:
#         return f"Weather lookup failed: {e}"


# ── LLM ───────────────────────────────────────────────────────────────────────
# temperature=0 → deterministic output so ReAct Thought/Action format parses correctly
llm = ChatOllama(model="llama3.2", temperature=0)

# ── Agent setup ───────────────────────────────────────────────────────────────
tools = [web_search]  # get_weather commented out

# Standard ReAct prompt from LangChain Hub (hwchase17/react)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # prints Thought / Action / Observation trace
    max_iterations=6,      # prevents infinite loops
    handle_parsing_errors=True,  # returns error msg instead of crashing on bad format
)


# ── Runner ────────────────────────────────────────────────────────────────────
def run_agent(query: str) -> str:
    """Invoke the agent and return its Final Answer."""
    result = agent_executor.invoke({"input": query})
    return result["output"]


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        # Web search tool expected
        "What are the latest developments in quantum computing?",
        # Weather tool expected (commented out — re-enable get_weather first)
        # "What is the current weather in Lahore?",
        # Requires both: search for context + weather (commented out — re-enable get_weather first)
        # "What is the weather in London today, and what are the top tourist spots there?",
    ]

    for query in test_queries:
        print("\n" + "=" * 65)
        print(f"USER: {query}")
        print("=" * 65)
        answer = run_agent(query)
        print(f"\nFINAL ANSWER:\n{answer}")
        print("-" * 65)
