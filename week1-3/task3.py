from ollama import chat

role = "You are a customer review analysis assistant that helps businesses analyze customer reviews"
instruction =" Analyze the following product review and return the customers sentiment, a short reason  and a suggest support reply.Do not ask questions or add any extra text."
inputdata = "The product is great but the delivery was late" 
outputformat = "Sentiment: (Positive,Negative,Neutral), Reason: <reason>, Support Reply: <support_reply>"
constraints = "Max 50 words for reason and support reply"

prompt1 = f"{role}\nInstruction: {instruction}\nInput: {inputdata}\nOutput Format: {outputformat}\nConstraints: {constraints}"

response = chat(model="llama3.2", messages=[{"role": "user", "content": prompt1}])
print(response.message.content)

few_shot = """
Example 1:
Input: "love this blender"
Sentiment: Positive, Reason: Customer praises performance and usability with no complaints., Support Reply: So glad you love it! Enjoy blending. Reach out anytime.

Example 2:
Input: "Item arrived broken and customer service never responded."
Sentiment: Negative, Reason: Product damaged on arrival and support was unresponsive., Support Reply: We sincerely apologize. Please share your order number so we can send a replacement.

Example 3:
Input: "Works fine, nothing special. Does the job but packaging could be better."
Sentiment: Neutral, Reason: Product meets expectations but leaves no strong impression., Support Reply: Thanks for the feedback! We'll pass your note to our team.
"""

finalprompt = f"{role}\nInstruction: {instruction}\nExamples: {few_shot}\nInput: {inputdata}\nOutput Format: {outputformat}\nConstraints: {constraints}"

response = chat(
    model="llama3.2",
    messages=[{"role": "user", "content": finalprompt}],
    options={
        "temperature": 0.5,
        "top_p": 0.8,
        "top_k": 30
    }
)
print("\nafter fewshot\n")
print(response.message.content)
