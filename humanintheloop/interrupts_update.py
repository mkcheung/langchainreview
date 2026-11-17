import operator
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.types import Send
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from typing import Literal
from typing_extensions import TypedDict


load_dotenv()

console = Console()

llm = ChatOpenAI(model="gpt-5.4-mini")

result = llm.invoke('Hello')
print(result)
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

tools = [add,multiply,divide]

llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state:MessagesState):
    return {"messages":[llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder=StateGraph(MessagesState)

builder.add_node('assistant', assistant)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'assistant')
builder.add_conditional_edges(
    'assistant',
    tools_condition
 )
builder.add_edge('tools', 'assistant')

memory = MemorySaver()

graph = builder.compile(interrupt_before=["assistant"], checkpointer = memory)
thread = {"configurable":{"thread_id":123}}
initial_input={"messages":HumanMessage(content="Multiple 2 and 3")}

for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

state = graph.get_state(thread)

print(state.next)
print(graph.get_state_history(thread))

graph.update_state(thread,{"messages":[HumanMessage(content="No, please multiple 15 and 6")]})

new_state = graph.get_state(thread).values


for m in new_state['messages']:
    m.pretty_print()

for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
