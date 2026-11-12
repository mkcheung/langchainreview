import os
import random
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AnyMessage, SystemMessage, HumanMessage, AIMessage,ToolMessage
from langgraph.graph.message import add_messages # reducer
from typing import Annotated
from pprint import pprint
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from IPython.display import Image, display

class State(TypedDict):
    topic:str
    story:str
    improved_story:str
    final_story:str

llm = ChatOpenAI(model="gpt-5.4-mini")

def generate_story(state:State):
    msg = llm.invoke(f"Write a one sentence story promise about {state['topic']}")
    return {"story":msg.content}

def check_conflict(state:State):
    if "?" in state['story'] or  "!" in state['story']:
        return "Fail"
    return "Pass"

def improve_story(state:State):
    msg=llm.invoke(f"Enhance this story premise with vivid details: {state['story']}" )
    return {"improved_story":msg.content}

def polish_story(state:State):
    msg = llm.invoke(f"Add an unexepcted twist to this story premise: {state['improved_story']}")
    return {"final_story":msg.content}

graph=StateGraph(State)
graph.add_node("generate", generate_story)
graph.add_node("improve", improve_story)
graph.add_node("polish", polish_story)

graph.add_edge(START, 'generate')
graph.add_conditional_edges('generate', check_conflict, {'Pass':"improve", "Fail":
"generate"})
graph.add_edge('improve', 'polish')
graph.add_edge('polish', END)

compiled_graph = graph.compile()

graph_image = compiled_graph.get_graph().draw_mermaid_png()
display(Image(graph_image))

state = {"topic": "Agentic AI Systems"}
result = compiled_graph.invoke(state)
for key, value in result.items():
    print(f"\n=== {key.replace('_', ' ').upper()} ===")
    print(value)