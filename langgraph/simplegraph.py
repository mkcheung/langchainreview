from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
import random
from typing import Literal
from typing_extensions import TypedDict

class State(TypedDict):
    graph_info:str

def start_play(state:State):
    print("Start_Play node has been called.")
    return {
        "graph_info":state['graph_info']
        + "I am planning to play"
    }

def cricket(state:State):
    print("My Cricket node has been called")
    return {
        "graph_info":state['graph_info']
        + " Cricket"
    }

def badminton(state:State):
    print("My Badminton node has been called")
    return {
        "graph_info":state['graph_info']
        + " Badminton"
    }

def random_play(state:State) -> Literal['cricket', 'badminton']:
    graph_info=state['graph_info']

    if random.random() > 0.5:
        return "cricket"
    else:
        return "badminton"

## Construct the graph
graph=StateGraph(State)

### attach all nodes to the graph
graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)

### establish the edge - the conduits from node to node
graph.add_edge(START, "start_play")
graph.add_conditional_edges("start_play", random_play)
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)

### Compile the graph
graph_builder=graph.compile()

### View
display(Image(graph_builder.get_graph().draw_mermaid_png()))

graph_builder.invoke({
    "graph_info":"Hey my name is mars"
})