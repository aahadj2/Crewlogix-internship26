from ollama import chat
from langchain import PromptTemplate

def add_numbers(a, b):
    return int(a) + int(b)

def subtract_numbers(a, b):
    return int(a) - int(b)

def multiply_numbers(a, b):
    return int(a) * int(b)

cot= "think step by step and call the appropriate tool to answer the query."
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Adds two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "subtract_numbers",
            "description": "Subtracts b from a",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply_numbers",
            "description": "Multiplies two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

queries = ["Add 15 and 15", "Subtract 9 from 10", "Multiply 6 and 7"]

for query in queries:
    print(f"\n{'='*40}")
    print(f"Query  : {query}")

    response = chat(model="llama3.2", messages=[{"role": "user", "content": query}], tools=tools)

    if response.message.tool_calls:
        tool  = response.message.tool_calls[0]
        name  = tool.function.name
        args  = tool.function.arguments

        if name == "add_numbers":       result = add_numbers(**args)
        elif name == "subtract_numbers": result = subtract_numbers(**args)
        elif name == "multiply_numbers": result = multiply_numbers(**args)

        print(f"Tool   : {name}({args})")
        print(f"Result : {result}")

        final = chat(
            model="llama3.2",
            messages=[{"role": "user", "content": f"The result of {query} is {result}. Explain briefly."}]
        )
        print(f"Answer : {final.message.content.strip()}")
    else:
        print("No tool called")