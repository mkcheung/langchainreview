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
)
# response = agent.invoke("What's the weather like in Boston?")
# for tool_call in response.tool_calls:
#     print(f"Tool: {tool_call['name']}")
#     print(f"Args: {tool_call['args']}")

# Tool Execution Loop
messages = [
    {"role": "user", "content": "What is the weather like in Boston?"}
]

result = agent.invoke({"messages": messages})
print(result["messages"][-1].content)