# from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, ArxivLoader, WikipediaLoader
# from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, HTMLHeaderTextSplitter
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage,SystemMessage
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.chains import create_retrieval_chain
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from langserve import add_routes
# import bs4
# from fastapi import FastAPI
# import wikipedia

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
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

with_message_history=RunnableWithMessageHistory(model, get_session_history)
config = {"configurable":{"session_id":"chat1"}}
response = with_message_history.invoke(
    [
        HumanMessage(content="Hi, my name is Mars and I am a programmer/martial-artist/philosopher"),
    ],
    config
)

print(response.content)
response = with_message_history.invoke(
    [
        HumanMessage(content="Hi, my name is Mars and I am a programmer/martial-artist/philosopher"),
    ],
    config
)
print(response.content)
# load_dotenv()
# model = ChatGroq(model='openai/gpt-oss-20b', groq_api_key=os.getenv("GRQQ_API_KEY_2"))

# result = model.invoke(
#     [
#         HumanMessage(content="Hi, my name is Mars and I am a programmer/martial-artist/philosopher"),
#         AIMessage(content="Hi Mars, it's nice to meet you!"),
#         HumanMessage(content="Hey, what's my name and what do I do?"),
#     ]
# )

# print(result)

