import os
import random
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage,ToolMessage
from langgraph.graph.message import add_messages # reducer
from typing import Annotated
from pprint import pprint
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
print(arxiv.name)

api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
print(wiki.name)

# result = arxiv.invoke("Attention is all you need")
# print(result)

# result = wiki.invoke("What is Machine Learning?")
# print(result)

tavily = TavilySearchResults()

# result = tavily.invoke("Provide me the recent AI news for march 3rd, 2025")
# print(result)

# tools=[arxiv,wiki,tavily]


# result = llm_with_tools.invoke([HumanMessage(content=f"What is the recent AI News?")]).tool_calls
# print(result)

class State(TypedDict):
    messages:Annotated[list[AnyMessage], add_messages]

def multiply(a:int, b:int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    
    """
    return a * b

def add(a:int, b:int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    
    """
    return a + b

def divide(a:int, b:int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    
    """
    return a / b

tools=[arxiv, wiki, add, multiply, divide]
llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)

def tool_calling_llm(state):
    return {"messages":[llm_with_tools.invoke(state['messages'])]}


builder=StateGraph(State)
builder.add_node('tool_calling_llm', tool_calling_llm)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools condition routes to END
    tools_condition
)

builder.add_edge("tools", 'tool_calling_llm')

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))

config={"configurable":{"thread_id":"1"}}

for chunk in graph.stream({'messages':'Hi, my name Mars and I like cricket'}, config, stream_mode='updates'):
    print(chunk)

for chunk in graph.stream({'messages':'I also like football'}, config, stream_mode='updates'):
    print(chunk)

messages = [HumanMessage(content="Add 12 and 13.")]

messages = graph.invoke({"messages":messages}, config=config)
for message in messages['messages']:
    message.pretty_print()

messages=[HumanMessage(content="Divide that by 5")]

messages = graph.invoke({"messages":messages}, config=config)
for message in messages['messages']:
    message.pretty_print()


messages=[HumanMessage(content="then multiply that number by 2")]

messages = graph.invoke({"messages":messages}, config=config)
for message in messages['messages']:
    message.pretty_print()
