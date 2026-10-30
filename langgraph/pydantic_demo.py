### pydantic data validation

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from IPython.display import Image, display


class State(BaseModel):
    name:str

def example_node(state:State):
    return {"name":"Hello"}

builder=StateGraph(State)
builder.add_node("example_node",example_node)
builder.add_edge(START, "example_node")
builder.add_edge("example_node", END)

graph=builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

print(graph.invoke({"name": "Mars"}))