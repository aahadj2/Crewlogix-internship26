# import os
# import ollama

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")


# llm = ollama.Ollama(model="llama3.2")
# prompt = "tell me about today's {topic} and what {level} is it"
# parser = StrOutputParser()
# chain = prompt | llm | parser


import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")  # use LANGCHAIN_API_KEY, not LANGSMITH_API_KEY


llm = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template("tell me about today's {topic} and what {level} is it")
parser = StrOutputParser()
chain = prompt | llm | parser

result = chain.invoke({"topic": "AI", "level": "beginner"})
print(result)