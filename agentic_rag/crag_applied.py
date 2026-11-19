import operator
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.types import Send
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from typing import Annotated, Sequence, List
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()

console = Console()

urls=[
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/how-tos/map-reduce"
]

docs=[WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]
print(docs_list)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
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
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
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

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

llm = ChatOpenAI(model="gpt-5.4-mini")
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give it a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system),
        ('human', "Retrieved document: \n\n {document} \n\n User {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
# print(retrieval_grade.invoke({'question':question, 'document':doc_txt}))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using the supplied context. "
    "If the context doesn't provide the answer, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

llm_rewriter = ChatOpenAI(model="gpt-6.2-sol")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()

system = """You are a question re-writer that converts an input question to a better version that is optimized \n
    for web search. Look at the input and try to reason about the underlying sematic intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            'human',
            "Here is the initial question: \n\n {question} \n Formulate an improved question."
        )
    ]
)
question_rewriter = re_write_prompt | llm_rewriter | StrOutputParser()
question_rewriter.invoke({'question':question_rewriter})

web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    """
    Represent the state of our graph.

    Attributes:
        question: question
        generation:LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    print("---RETRIEVE---")
    question = state['question']
    documents = retriever.invoke(question)
    return {'documents': documents, 'question':question}

def generate(state):
    """
    Generate answer

    Args:
        state(dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state['question']
    documents = state['documents']

    generation = rag_chain.invoke({"context": documents, "question":question})
    return {"documents": documents, "question":question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrueved documents are relevant to the question.

    Args:
        state (dict): The current graph state
    
    Returns: 
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state['question']
    documents = state['documents']

    filtered_docs = []
    web_search = 'No'
    for d in documents:
        score = retrieval_grader.invoke(
            {"question":question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == 'yes':
            print("---GRADE: DOCUMENT RELEVANT ---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT ---")
            web_search = 'Yes'
            continue
    return {'documents': filtered_docs, 'question':question, 'web_search':web_search}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Updates question key wth a re-phrased question
    """

    print("--- TRANSFORM QUERY ---")
    question = state['question']
    documents = state['documents']

    better_question = question_rewriter.invoke({'question':question})
    return {'documents': documents, 'question':better_question}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Update documents key with appended web results
    """
    print("--- Web Search ---")
    question = state['question']
    documents = state['documents']

    docs = web_search_tool.invoke({'query':question})
    web_results = "\n".join([d['content'] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {'documents': documents, 'question':question}

def decide_to_generate(state):
    """
    Determines whether to generate an answer,or re-generate a question.

    Args:
        state (dict): The current graph state
    
    Returns:
        str: Binary decision for next node to call
    """
    print("--- ASSESS GRADED DOCUMENTS ---")
    web_search = state['web_search']

    if web_search == 'Yes':
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return 'transform_query'
    else:
        print('---DECISION: GENERATE---')
        return 'generate'

workflow = StateGraph(GraphState)

workflow.add_node('retrieve', retrieve)
workflow.add_node('grade_documents', grade_documents)
workflow.add_node('generate', generate)
workflow.add_node('transform_query', transform_query)
workflow.add_node('web_search_node', web_search)

workflow.add_edge(START, 'retrieve')
workflow.add_edge('retrieve', 'grade_documents')
workflow.add_conditional_edges(
    'grade_documents',
    decide_to_generate,
    {
        'transform_query': "transform_query",
        'generate':'generate'
    } 
)

workflow.add_edge('transform_query', 'web_search_node')
workflow.add_edge('web_search_node', 'generate')

app = workflow.compile()
result = app.invoke({"question":"What are the types of agent memory?"})
console.print(result)
