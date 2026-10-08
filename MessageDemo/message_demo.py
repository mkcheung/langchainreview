import os
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage,ToolMessage
from langchain.tools import tool


model = init_chat_model("google_genai:gemini-3.5-flash-lite")

model.invoke("Please tell me what is artificial intelligence")

# messages=[
#     SystemMessage("You are a poetry expert"),
#     HumanMessage("Write a poem on artificial intellegence")
# ]

# response=model.invoke(messages)
# print(response.content)

############ System Message - detailed information to the LLM through system message ############

# sys_message = SystemMessage("""You are a senior Python developer with expertise in web frameworks. 
# Always provide code examples and explain your reasoning. Be concise but thorought in your explanations
# """)

# messages=[
#     sys_message,
#     HumanMessage("How do I create a REST API?")
# ]

# response=model.invoke(messages)
# print(response.content)


############ Human Messages ############

# human_msg = HumanMessage(
#     content="Hello!",
#     name="alice",
#     id="msg_123"
# )

# response=model.invoke([
#     human_msg
# ])
# print(response)

############ Fudged AI Message Insertion with System Message Context and Human Input ############
# ai_msg = AIMessage("I'd be happy to help you with that question!")

# messages=[
#     SystemMessage("You are a helpful assistant"),
#     HumanMessage("Can you help me?"),
#     ai_msg,
#     HumanMessage("Great! What's 2+2?")
# ]

# response = model.invoke(messages)
# print(response.content)

############ Tool Messages ############
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        'name':'get_weather',
        'args':{'locaton': 'San Francisco'},
        'id': 'call_123'
    }]
)

weather_result = "Sunny, 72 degrees Farenheit"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"
)


messages=[
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,
    tool_message
]

response = model.invoke(messages)

print(response.content)