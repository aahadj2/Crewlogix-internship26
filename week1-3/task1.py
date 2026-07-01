import google.generativeai as genai
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")
app = FastAPI()

#the format we will get the post req  in
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(body: Question):
    try:
        prompt = f"answer the following question : {body.question}"
        response = model.generate_content(prompt)
        return {"question": body.question, "answer": response.text}
    except Exception as e:
        return {"error": str(e)}