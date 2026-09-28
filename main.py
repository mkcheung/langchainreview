from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from operator import itemgetter
import os
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model='openai/gpt-oss-20b', groq_api_key=os.getenv("GRQQ_API_KEY_2"))
store = {}

prompt=ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant. Answer all questions to the best of your ability in {language} '),
        MessagesPlaceholder(variable_name="messages")
    ]
)

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
trimmer.invoke(messages)


chain=(
    RunnablePassthrough.assign(messages=itemgetter('messages')|trimmer)
    | prompt
    | model
)

with_message_history=RunnableWithMessageHistory(chain, get_session_history, input_messages_key='messages')
config = {"configurable":{"session_id":"chat1"}}

response = with_message_history.invoke(
    {'messages': messages, 'language': 'English'},
    config=config
)

response=with_message_history.invoke(
    {
        'messages':[HumanMessage(content='What is my favorite ice cream?')],
        'language':'English'
    },
    config=config
)
response=with_message_history.invoke(
    {
        'messages':[HumanMessage(content='What math problem did I ask?')],
        'language':'English'
    },
    config=config
)

print(response)