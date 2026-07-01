import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
#from langchain.memory import ConversationBufferMemory
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
prompt = PromptTemplate.from_template(
    "tell me about today's {topic} and what {level} is it"
)
model = OllamaLLM(model="llama3.2")
parser = StrOutputParser()
 #memory = ConversationBufferMemory()
chain = prompt | model | parser

history = []

response = chain.invoke({"topic": "fifa", "level": "beginner"})
history.append({"input": "fifa, beginner", "output": response})

print(response)
print(history)