import operator
from typing import Annotated

from dotenv import load_dotenv
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
from typing import Annotated, Sequence
from typing_extensions import TypedDict


load_dotenv()

console = Console()

# llm = ChatOpenAI(model="gpt-5.4-mini")
# result = llm.invoke('Hello')

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


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent(state):
    """ Invokes the agent model to generate a response based on the
    current state. Given the question, it will decide to retrieve using
    the retriever tool, or simply end.

    Args:
        state (messages): The current state
    
    Returns:
        dict: The updated state with the agent response appended to messages
    """

    print("---CALL AGENT---")
    messages = state['messages']
    
    model = ChatOpenAI(model="gpt-5.4-mini")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {'messages': [response]}

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """
    print("---CHECK RELEVANCE---")

    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    model = ChatOpenAI(model="gpt-5.4-mini")

    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    )

    chain = prompt | llm_with_tool

    messages = state['messages']
    last_meassage = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT")
        return 'generate'
    else:
        print("---DECISION: DOCS NOT RELEVANT")
        print(score)
        return 'rewrite'

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
        dict: The updated message
    """

    print("---GENERATE---")
    messages = state['messages']
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    prompt = hub.pull(
        'rlm/rag-prompt'
    )

    llm = ChatOpenAI(model="gpt-5.4-mini")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({'context':docs, 'question': question})
    return {'messages':[response]}

def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state
    
    Returns:
        dict: The updated state with re-phrased question

    """

    print("--- TRANSFORM QUERY ---")
    messages = state['messages']
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f"""\n
            Look at the input and try to reason about the underlying semantic intent / meaning. \n
            Here is the initial question:
            \n ------- \n
            {question}
            \n ------- \n
            Formulate an improved questions: """,
        )
    ]
    model = ChatOpenAI(model="gpt-5.4-mini")
    response = model.invoke(msg)
    return {'messages': [response]}

workflow = StateGraph(AgentState)

workflow.add_node('agent', agent)
retrieve = ToolNode([retriever_tool, retriever_tool_langchain])
workflow.add_node('retrieve', retrieve)
workflow.add_node('rewrite', rewrite)
workflow.add_node(
    'generate', generate
)
workflow.add_edge(START,agent)

workflow.add_conditional_edges(
    'agent',
    tools_condition,
    {
        'tools': 'retrieve',
        END:END
    }
)

workflow.add_conditional_edges(
    'retrieve',
    grade_documents,
    {
        'generate':'generate',
        'rewrite':'rewrite'
    }
)

workflow.add_edge('generate', END)
workflow.add_edge('rewrite', 'agent')

graph = workflow.compile()

graph.invoke({'messages':'What is Langgraph?'})