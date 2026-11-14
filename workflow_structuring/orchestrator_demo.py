import os
import operator
import random
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing_extensions import Literal,TypedDict
from pydantic import BaseModel,Field
from langchain_core.messages import BaseMessage, AnyMessage, SystemMessage, HumanMessage, AIMessage,ToolMessage
from langgraph.graph.message import add_messages # reducer
from langgraph.types import Send
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown

console = Console()

llm = ChatOpenAI(model="gpt-5.4-mini")
#Schema from structured output to use as routing logic
class Section(BaseModel):
    name:str=Field(description="Name for this section of the report.")
    description:str=Field(description="Brief Overview of main topics and concepts of the section")

class Sections(BaseModel):
    sections:list[Section]=Field(
        description="Sections of the report"
    )

#augment the llm with scemea for structured output
planner = llm.with_structured_output(Sections)

### creating workers dynamically in langgraph

class State(TypedDict):
    topic:str
    sections: list[Section] # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]
    final_report: str

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}")
        ]
    )
    console.print("\nReport outline", style="bold cyan")
    for index, section in enumerate(report_sections.sections, start=1):
        console.print(f"{index}. {section.name}", markup=False)
    console.print()
    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """Worker writes a section of the report"""

    section = llm.invoke(
        [
            SystemMessage(content="Write a report section following the provided name and description. Format it as Markdown, starting with a level-two heading (##) containing the section name. Include no preamble for each section."),
            HumanMessage(content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}")
        ]
    )

    return {"completed_sections": [section.content]}

def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    return [Send("llm_call", {"section": s}) for s in state['sections']]

def synthesizer(state: State):
    """Synthesize full report from sections"""

    completed_sections = state["completed_sections"]

    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report":completed_report_sections}

orchestrator_worker_builder = StateGraph(State)

orchestrator_worker_builder.add_node('orchestrator', orchestrator)
orchestrator_worker_builder.add_node('llm_call', llm_call)
orchestrator_worker_builder.add_node('synthesizer', synthesizer)

orchestrator_worker_builder.add_edge(START, 'orchestrator')
orchestrator_worker_builder.add_conditional_edges(
    'orchestrator', assign_workers, ['llm_call']
)

orchestrator_worker_builder.add_edge('llm_call', 'synthesizer')
orchestrator_worker_builder.add_edge('synthesizer', END)

orchestrator_worker = orchestrator_worker_builder.compile()

state = orchestrator_worker.invoke({'topic': "Create a report on Agentic AI RAGS"})

console.rule("Final report")
console.print(Markdown(state["final_report"]))
