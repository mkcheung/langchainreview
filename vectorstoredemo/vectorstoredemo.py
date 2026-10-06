from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from operator import itemgetter
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model='Llama3-8b-8192', groq_api_key=os.getenv("GRQQ_API_KEY_2"))
store = {}

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={'source':'mammal-pets-doc'},
    ),
    Document(
        page_content="Cats are independence pets that often enjoy their own space.",
        metadata={'source':'mammal-pets-doc'},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={'source':'mammal-pets-doc'},
    ),
    Document(
        page_content='Parrots are intelligent birds capable of mimicking human speech.',
        metadata={'source':'mammal-pets-doc'},
    ),
    Document(
        page_content='Rabbits are social animals that need plenty of space to hop around.',
        metadata={'source':'mammal-pets-doc'},
    ),
]

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore=Chroma.from_documents(documents, embedding=embeddings)

# vectorstore.similarity_search("cat")

# see the matching scores
# result = vectorstore.similarity_search_with_score('cat')

# demonstrate use of the retriever
# retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
# result = retriever.batch(["cat", "dog"])

# retriever from vector store
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":1}
)

# print(retriever.batch(["cat", "dog"]))

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain={"context":retriever,"question":RunnablePassthrough()}|prompt|llm

response=rag_chain.invoke("tell me about dogs")
print(response.context)