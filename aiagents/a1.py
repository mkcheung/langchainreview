import os
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city:str) -> str:
    """Get the weather for a city"""
    return f"The wearther in {city} is sunny."

agent = create_agent(
    model="gpt-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistent",
    debug=True
)
response = agent.invoke({"messages":[{"role":"user", "content":"What is the weather like in New York?"}]})
print(response['messages'][-1])
