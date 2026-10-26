import os
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

@tool
def read_email_tool(email_id: str)->str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

@tool
def send_email_tool(recipient: str, subject:str, body:str)-> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model="gpt-4o",
    tools=[read_email_tool,send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool":{
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "read_email_tool":False
            }
        )
    ]
)

config = {"configurable": {"thread_id":"test_approve"}}

result = agent.invoke(
    {"messages": [HumanMessage(content=f"Send email to john@test.com with subject 'Hello' and body 'How are you?'")]},
    config=config
)

if "__interrupt__" in result:
    print("Paused! Approving...")
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "approve"}
                ]
            }
        ),
        config=config
    )

    print(f"Result: {result['messages'][-1].content}")