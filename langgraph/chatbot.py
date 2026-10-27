import os
from dotenv import load_dotenv
load_dotenv()
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.tools import tool



class State(TypedDict):
    messages:Annotated[list, add_messages]

llm=ChatOpenAI(model="gpt-4o")
llm.invoke("Hello")

llm_groq=ChatGroq(model="openai/gpt-oss-120b")
llm_groq.invoke("Hey I am Mars and I like to play cricket")

def superbot(state:State):
    return {"messages":[llm_groq.invoke(state['messages'])]}

## Construct the graph
graph=StateGraph(State)

### attach all nodes to the graph
graph.add_node("Superbot", superbot)
graph.add_edge(START, "Superbot")
graph.add_edge("Superbot", END)

### Compile the graph
graph_builder=graph.compile()


### View
display(Image(graph_builder.get_graph().draw_mermaid_png()))

for event in graph_builder.stream({ "messages":"Hi, my name is Mars and I like BJJ"}, stream_mode="values"):
    print(event)