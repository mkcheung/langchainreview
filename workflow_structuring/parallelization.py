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
    characters: str
    settings:str
    premises:str
    story_intro:str

llm = ChatOpenAI(model="gpt-5.4-mini")

def generate_characters(state:State):
    """Generate character descriptions"""
    msg = llm.invoke(f"Create two character names and brief traits for a story about {state['topic']}")
    return {"characters":msg.content}

def generate_setting(state:State):
    """Generate a story setting"""
    msg = llm.invoke(f"Describe a vivid setting for a story about {state['topic']}")
    return {"settings":msg.content}

def generate_premise(state:State):
    """Generate a story premise"""
    msg = llm.invoke(f"Write a one-sentence plot premise for a story about {state['topic']}")
    return {"premises":msg.content}

def combine_elements(state:State):
    """Combine characters, setting, and premise into an intro"""
    msg = llm.invoke(
        f"Write a short story introduction using these elements: \n"
        f"Characters: {state['characters']} \n"
        f"Setting: {state['settings']} \n"
        f"Premises: {state['premises']} \n"
    )
    return {"story_intro":msg.content}

graph=StateGraph(State)
graph.add_node("character", generate_characters)
graph.add_node("setting", generate_setting)
graph.add_node("premise", generate_premise)
graph.add_node("combine", combine_elements)

graph.add_edge(START, 'character')
graph.add_edge(START, 'setting')
graph.add_edge(START, 'premise')
graph.add_edge('character', 'combine')
graph.add_edge('setting', 'combine')
graph.add_edge('premise', 'combine')
graph.add_edge('combine', END)

compiled_graph = graph.compile()

graph_image = compiled_graph.get_graph().draw_mermaid_png()
display(Image(graph_image))

state = {"topic": "time travel"}
result = compiled_graph.invoke(state)
for key, value in result.items():
    print(f"\n=== {key.replace('_', ' ').upper()} ===")
    print(value)