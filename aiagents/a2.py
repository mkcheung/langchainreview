import os
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# model = init_chat_model("gpt-4.1")

# response = model.invoke("Hello, how are you?")
# print(response.content)

## Gemini Integrations
# model = init_chat_model("google_genai:gemini-3.5-flash-lite")
# response = model.invoke("Why do parrots talk?")
# print(response.content)
# model = ChatGoogleGenerativeAI("google_genai:gemini-3.5-flash-lite")
# response = model.invoke("Why do parrots talk?")
# print(response.content)


## ChatOpenAI Integrations
# model = ChatOpenAI("gpt-5")
# response = model.invole("Hello, how are you?")
# print(response.content)

## Groq Model Integrations
# model = init_chat_model("openai/gpt-oss-120b")
# response = model.invoke("Why do parrots talk?")
# print(response.content)
model = ChatGroq(model="openai/gpt-oss-120b")
# response = model.invoke("Why do parrots talk?")
response = model.stream("Why do parrots talk?")
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
