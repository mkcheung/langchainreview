import os
import random
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq


from langchain_core.messages import AIMessage, HumanMessage
from pprint import pprint

messages=[AIMessage(content=f"Please tell me how can I help.", name="LLMModel")]
messages.append(HumanMessage(content=f"I want to learn coding", name="Mars"))
messages.append(AIMessage(content=f"Which programming language you want to learn", name="LLMModel"))
messages.append(AIMessage(content=f"I want to learn the python programming languages", name="Mars"))

for message in messages:
    message.pretty_print()

llm_groq=ChatGroq(model="openai/gpt-oss-120b")
result = llm_groq.invoke(messages)

print(result.response_metadata)
