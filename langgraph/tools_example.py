import os
import random
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage,ToolMessage
from langgraph.graph.message import add_messages # reducer
from typing import Annotated
from pprint import pprint
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display


def add(a:int, b:int) -> int:
    """ Add a and b
    Args:
        a (int): first int
        b (int): second int

    Returns:
        int  
    """
    return a+b

llm = ChatGroq(model="openai/gpt-oss-120b")

llm_with_tools = llm.bind_tools([add])

tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 2 plus 2?", name = "mars")])


class State(TypedDict):
    messages:Annotated[list[AnyMessage], add_messages]
    name: str

### chatbot node functionality
def llm_tool(state:State):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }

#### Reducers with add_messages
initial_messages=[AIMessage(content=f"Please tell me how can I help.", name="LLMModel")]
initial_messages.append(HumanMessage(content=f"I want to learn coding", name="Mars"))
print(initial_messages)

ai_message = AIMessage(content=f"Which programming language you want to learn", name="LLMModel")
print(ai_message)
add_messages(initial_messages, ai_message)

tools = [add]
builder=StateGraph(State)
builder.add_node('llm_tool', llm_tool)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, "llm_tool")
builder.add_conditional_edges(
    "llm_tool",
    # If the latest message (result) from assistant is a tool call -> tools condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools condition routes to END
    tools_condition
)
builder.add_edge("tools", END)

graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

messages = graph.invoke({"messages":"What is 2 plus 2?"})
for message in messages['messages']:
    message.pretty_print()


messages = graph.invoke({"messages":"What is Machine Learning?"})
for message in messages['messages']:
    message.pretty_print()