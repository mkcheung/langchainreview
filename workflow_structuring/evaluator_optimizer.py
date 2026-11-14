import operator
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from typing import Literal
from typing_extensions import TypedDict


load_dotenv()

console = Console()

llm = ChatOpenAI(model="gpt-5.4-mini")

### creating workers dynamically in langgraph

class State(TypedDict):
    joke:str
    topic:str
    feedback:str
    funny_or_not:str

class Feedback(BaseModel):
    grade: Literal['funny', 'not funny'] = Field(description="Decide if the joke is funny or not"),
    feedback: str = Field(description="If the joke is not funny, provide feedback on how to improve it")

#augment the llm with scemea for structured output
evaluator = llm.with_structured_output(Feedback)

def llm_call_generator(state:State):
    """LLM generates a joke"""

    if state.get('feedback'):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""

    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""
    if state['funny_or_not'] == 'funny':
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


optimizer_builder = StateGraph(State)

optimizer_builder.add_node('llm_call_generator', llm_call_generator)
optimizer_builder.add_node('llm_call_evaluator', llm_call_evaluator)

optimizer_builder.add_edge(START, 'llm_call_generator')
optimizer_builder.add_edge('llm_call_generator', 'llm_call_evaluator')
optimizer_builder.add_conditional_edges(
    'llm_call_evaluator', 
    route_joke, 
    {
        'Accepted': END,
        "Rejected + Feedback": "llm_call_generator",
    }
)

optimizer_worker = optimizer_builder.compile()

state = optimizer_worker.invoke({'topic': "Agentic AI System"})

console.rule("Joke")
console.print(Markdown(state["joke"]))
