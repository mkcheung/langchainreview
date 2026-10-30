import random
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Literal

class TypedDictState(TypedDict):
    name: str
    game: Literal["cricket", "badminton"]

def play_game(state: TypedDictState):
    print("---Play Game node has been called---")
    return {"name":state['name'] + " wants to play "}

def cricket(state:TypedDictState):
    print("--- Cricket node has been called ---")
    return {"name": state['name'] + "cricket" ,"game": "cricket"}

def badminton(state:TypedDictState):
    print("--- Badminton node has been called ---")
    return {"name": state['name'] + "badminton" ,"game": "badminton"}

def decide_play(state:TypedDictState)->Literal["cricket", "badminton"]:
    if random.random() < 0.5:
        return "cricket"
    else:
        return "badminton"

builder = StateGraph(TypedDictState)
builder.add_node("playgame", play_game)
builder.add_node("cricket", cricket)
builder.add_node("badminton", badminton)


### establish the edge - the conduits from node to node
builder.add_edge(START, "play_game")
builder.add_conditional_edges("play_game", random_play)
builder.add_edge("cricket", END)
builder.add_edge("badminton", END)

graph=builder.compile()

display(Image(graph_builder.get_graph().draw_mermaid_png()))

graph.invoke({
    "name":"Mars"
})