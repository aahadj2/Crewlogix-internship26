from ollama import chat


role = "You are an AI agent that helps students identify study problems and recommend personalized learning strategies."
instruction =" Analyze the following student reply and identify the issue and give best learning strategies.Do not ask questions or add any extra text."
inputdata = "I forget topics after reading them once" 
output_format = "Thinking: <reasoning>\nFeedback: <recommendation>"
constraints = "Max 50 words for reason and support reply"

fewshot = """
Example 1:
Input: "I can't focus when I study."
Thinking: The student struggles with sustained attention. Short timed intervals with breaks help train focus gradually.
Feedback: Try the Pomodoro Technique: study for 25 minutes, then take a 5-minute break. Repeat to build focus over time.

Example 2:
Input: "I get overwhelmed with too much information."
Thinking: The student is experiencing cognitive overload. Chunking material and scheduling reduces overwhelm.
Feedback: Break material into small sections and tackle one topic at a time using a structured study schedule.

Example 3:
Input: "I have trouble remembering what I read."
Thinking: Passive reading leads to poor retention. Active recall and spaced repetition force memory consolidation.
Feedback: After reading, summarize key points from memory. Review the material at increasing intervals to boost retention.
"""
chain_of_thought = "think step by step and give the best learning strategies for the student"
# chain_of_thought = (
#     "Step 1: Identify the core study problem from the input.\n"
#     "Step 2: Reason about why this problem occurs.\n"
#     "Step 3: Match the problem to the most effective learning strategy.\n"
#     "Step 4: Give a clear, concise recommendation."
# )
finalprompt = f"{role}\nInstruction: {instruction}\nExamples: {fewshot}\nInput: {inputdata}\nOutput Format: {output_format}\nConstraints: {constraints}\nChain of Thought: {chain_of_thought}"

response = chat(
    model="llama3.2",
    messages=[{"role": "user", "content": finalprompt}],
    options={
        "temperature": 0.5,
        "top_p": 0.8,
        "top_k": 30
    }
)
print(response.message.content)