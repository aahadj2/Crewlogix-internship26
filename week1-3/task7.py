from ollama import chat
import json

one_shot = """
Example:
Input: "I was charged twice for my subscription"
Output: {"category": "Billing", "reason": "User was charged twice,this is a billing issue."}
"""

prompt = """
You are a customer support classifier.

Classify the user message into exactly one category:
Billing, Technical Issue, Account Access, Refund, or Other.

{one_shot}

Think step by step, then return ONLY raw JSON.

Input: "{message}"
"""

tickets = [
    "I can't log into my account, it says password is wrong",
    "The app keeps crashing on my phone",
    "My invoice shows wrong amount"
]

for ticket in tickets:
    response = chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a support classifier. Return raw JSON only."},
            {"role": "user",   "content": prompt.format(one_shot=one_shot, message=ticket)}
        ],
        options={"temperature": 0.3,"top_k": 10}
    )
    clean = response.message.content.strip().replace("}}", "}")
    data  = json.loads(clean)

    print(f"\nTicket   : {ticket}")
    print(f"Category : {data.get('category', 'N/A')}")
    print(f"Reason   : {data.get('reason', 'N/A')}")