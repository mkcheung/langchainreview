from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model='openai/gpt-oss-20b', groq_api_key=os.getenv("GRQQ_API_KEY_2"))
store = {}
def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt=ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant. Answer all questions to the best of your ability in {language} '),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain=prompt|model

with_message_history=RunnableWithMessageHistory(chain, get_session_history, input_messages_key='messages')
config = {"configurable":{"session_id":"chat1"}}
response = with_message_history.invoke(
    {
        'messages': 
        [
            HumanMessage(content="Hi My name is Mars."),
        ],
        'language':'hebrew'
    },
    config=config
)

print(response.content)
response = with_message_history.invoke(
    {
        'messages': 
        [
            HumanMessage(content="What's my name?"),
        ],
        'language':'hebrew'
    },
    config=config
)
print(response.content)




