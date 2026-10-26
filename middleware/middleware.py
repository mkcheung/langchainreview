import os
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# Message based summarization 
# agent=create_agent(
#     model="gpt-4o-mini",
#     checkpointer=InMemorySaver(),
#     middleware=[
#         SummarizationMiddleware(
#             model="gpt-4o-mini",
#             trigger=("messages", 10),
#             keep=("messages", 4)
#         )
#     ]
# )

# config={"configurable":{"thread_id": "test-1"}}

# questions = [
#     "What is 2+2?",
#     "What is 10*5?",
#     "What is 100/4?",
#     "What is 15-7?",
#     "What is 3*3?",
#     "What is 4*4?",
# ]

# for q in questions:
#     response=agent.invoke({"messages":[HumanMessage(content=q)]}, config)
#     print(f"Messages: {response}")
#     print(f"Messages: {len(response['messages'])}")



# Token based summarization 

@tool
def search_hotels(city: str)->str:
    """Search hotels - returns long response to use more tokens."""
    return f"""Hotels in {city}:
    1. Grand Hotel - 5 star, $350/night, spa, pool, gym
    2. City Inn - 4 star, $180/night business center
    3. Budget Stay - 3 star, $75/night, free wifi"""


agent=create_agent(
    model="gpt-4o-mini",
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 550),
            keep=("tokens", 200)
        )
    ]
)

config={"configurable":{"thread_id": "test-1"}}

def count_tokens(messages):
    total_chars = sum(len(str(m.content)) for m in messages)
    return total_chars // 4

cities = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]

for city in cities:
    response = agent.invoke(
        {"messages": [HumanMessage(content=f"Find hotels in {city}")]},
        config=config
    )

    tokens = count_tokens(response["messages"])
    print(f"{city}: ~{tokens}, {len(response['messages'])} messages")
    print(f"{(response['messages'])}")