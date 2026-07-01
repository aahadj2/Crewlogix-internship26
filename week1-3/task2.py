from ollama import chat
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(body: Question):
    try:
        prompt = f"answer the following question: {body.question}"

        response = chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        
        return {"question": body.question, "answer": response.message.content}
    except Exception as e:
        return {"error": str(e)}