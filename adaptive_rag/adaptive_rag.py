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
from typing import Annotated, List, Literal, Sequence
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()

console = Console()

urls=[
    "https://lilianweng-ai.github.io/posts/2023-06-23-agent/",
    "https://lilianweng-ai.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng-ai.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs=[WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)
print(doc_splits)

vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal['vectorstore', 'web_search'] = Field(
        ...,
        decsription="Given a user question choose to route it to web search or a vectorstore."
    )
   
llm = ChatOpenAI(model="gpt-5.4-mini")
structured_llm_router = llm.with_structured_output(RouteQuery) 

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system),
        ('human', "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

print(
    question_router.invoke(
        {'question': "Who will the bears draft first in the NFL draft?"}
    )
)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

llm_rewriter = ChatOpenAI(model="gpt-6.2-sol")
structured_llm_grader = llm_rewriter.with_structured_output(GradeDocuments)
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
question = 'agent memory'
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question":question, "document":doc_txt}))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using the supplied context. "
    "If the context doesn't provide the answer, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

rag_chain = prompt | llm | StrOutputParser()

generation = rag_chain.invoke({'context': docs, "question":question})
print(generation)

class GradeHallucinations(BaseModel):
    """Binary score for halluncination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

halluncination_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system),
        ('human', "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

halluncination_grader = halluncination_prompt | structured_llm_grader
halluncination_grader.invoke({'documents': docs, 'generation':generation})

class GradeAnswer(BaseModel):
    """Binary score to assess if the answer answers the question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether am amswer addresses / resolves a question \n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system),
        ('human', "User question: \n\n {question} \n\n LLM generation. {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({'question': question, 'generation':generation})

system = """You are a question re-writer that converst an input question to a better version that is optimized \n
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system),
        ('human', "Here is the inital question: \n\n {question} \n Formulate an improved question."),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({'question':question})

web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    """
    Represent the state of our graph.

    Attributes:
        question: question
        generation:LLM generation
        documents: list of documents
    """
    question: str
    generation: str
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

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

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

def grade_generation_v_documents_and_question(state):
    """
    Determines with the generation is grounded in the document and answer question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUNCINATIONS---")
    question = state['question']
    documents = state['documents']
    generation = state['generation']

    score = halluncination_grader.invoke(
        {'documents': documents, 'generation': generation}
    )
    grade = score.binary_score

    if grade == 'yes':
        print("---DECISION: GENRATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs. QUESTION---")
        score = answer_grader.invoke({'question': question, "generation": generation})
        grade = score.binary_score
        if grade == 'yes':
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return 'useful'
        else:
            print("---DECISION: GENERATION DOS NOT ADDRESS QUESTION---")
            return 'not useful'
    else:
        pprint('---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---')
        return 'not supported'

workflow = StateGraph(GraphState)

workflow.add_node('web_search', web_search)
workflow.add_node('retrieve', retrieve)
workflow.add_node('grade_documents', grade_documents)
workflow.add_node('generate', generate)
workflow.add_node('transform_query', transform_query)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        'web_search': "web_search",
        'vectorstore':'retrieve'
    } 
)
workflow.add_edge('web_search', 'generate')
workflow.add_edge('retrieve', 'grade_documents')
workflow.add_conditional_edges(
    'grade_documents',
    decide_to_generate,
    {
        'transform_query': "transform_query",
        'generate':'generate'
    } 
)
workflow.add_edge('transform_query', 'retrieve')
workflow.add_conditional_edges(
    'generate',
    grade_generation_v_documents_and_question,
    {
        'not supported': "generate",
        'useful':END,
        'not useful': 'transform_query'
    } 
)

app = workflow.compile()

result = app.invoke({'question':'What is machine learning?'})
console.print(result)