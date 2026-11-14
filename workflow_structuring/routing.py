import os
import random
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing_extensions import Literal,TypedDict
from pydantic import BaseModel,Field
from langchain_core.messages import BaseMessage, AnyMessage, SystemMessage, HumanMessage, AIMessage,ToolMessage
from langgraph.graph.message import add_messages # reducer
from typing import Annotated
from pprint import pprint
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from IPython.display import Image, display

llm = ChatOpenAI(model="gpt-5.4-mini")
#Schema from structured output to use as routing logic
class Route(BaseModel):
    step:Literal["poem", "story", "joke"]=Field(description="The next step in the routing process")

#augement the llm with scemea for structured output
router = llm.with_structured_output(Route)
class State(TypedDict):
    input:str
    decision: str
    output:str

def llm_call_1(state:State):
    """Write a story"""
    result = llm.invoke(state["input"])
    return {"output":result.content}

def llm_call_2(state:State):
    """Write a joke"""
    result = llm.invoke(state["input"])
    return {"output":result.content}

def llm_call_3(state:State):
    """Write a poem"""
    result = llm.invoke(state["input"])
    return {"output":result.content}

def llm_call_router(state:State):
    """Route the input that is given to the appropriate node"""

    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input to story, joke or poem based on the user's request."
            ),
            HumanMessage(content=state["input"])
        ]
    )
    return{"decision": decision.step}

def route_decision(state:State):
    if state["decision"] == 'story':
        return "llm_call_1"
    elif state['decision'] == 'joke':
        return "llm_call_2"
    elif state['decisio'] == 'poem':
        return 'llm_call_3'

router_builder=StateGraph(State)
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

router_builder.add_edge(START, 'llm_call_router')
router_builder.add_conditional_edges(
    'llm_call_router',
    route_decision,
    {
        "llm_call_1":"llm_call_1",
        "llm_call_2":"llm_call_2",
        "llm_call_3":"llm_call_3",
    }
)
router_builder.add_edge('llm_call_1', END)
router_builder.add_edge('llm_call_2', END)
router_builder.add_edge('llm_call_3', END)

router_workflow = router_builder.compile()

state = router_workflow.invoke({"input":"Write me a joke about Agentic AI system"})
print(state['output'])