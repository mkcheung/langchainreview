import operator
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTetSplitter
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain.tools.retriver import create_retriever_tool
from langgraph.types import Send
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from typing import Literal
from typing_extensions import TypedDict


load_dotenv()

console = Console()

llm = ChatOpenAI(model="gpt-5.4-mini")

result = llm.invoke('Hello')

urls=[
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/how-tos/map-reduce"
]

docs=[WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]
print(docs_list)
text_splitter = RecursiveCharacterTetSplitter(
    chunk_size=1000, check_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)
print(doc_splits)

vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retriever_vector_db_blog",
    "Search and run information about Langgraph"
)

langchain_urls=[
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history/"
]

docs=[WebBaseLoader(url).load() for url in langchain_urls]
docs_list = [item for sublist in docs for item in sublist]
print(docs_list)
text_splitter = RecursiveCharacterTetSplitter(
    chunk_size=1000, check_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)
print(doc_splits)

vectorstorelangchain = FAISS.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings()
)

retrieverlangchain = vectorstorelangchain.as_retriever()


retriever_tool_langchain = create_retriever_tool(
    retrieverlangchain,
    "retriever_vector_langchain_blog",
    "Search and run information about Langchain"
)

tools=[retriever_tool, retriever_tool_langchain]