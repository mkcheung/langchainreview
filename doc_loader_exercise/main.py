from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, ArxivLoader, WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, HTMLHeaderTextSplitter
import bs4
import wikipedia
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

wikipedia.set_user_agent("langchaincourse-exercise/1.0 (mars.kwong.cheung@gmail.com)")

# Text Loader
# loader=TextLoader('speech.txt')
# text_documents=loader.load()
# print(text_documents)

# Demonstrate use of recusrive character splitter
# speech: str = ""
# with open("speech.txt") as f:
#     speech = f.read()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
# text = text_splitter.create_documents([speech])
# print(text[0])
# print(text[1])

# Demonstrate use of  character splitter
# loader=TextLoader('speech.txt')
# docs=loader.load()
# text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=50)
# text = text_splitter.split_documents(docs)
# print(text[0])
# print(text[1])

# Demonstrate use of HTML Header Splitter
# manual html string
# html_string = """
# <!DOCTYPE html>
# <html>
# <body>
#     <div>
#         <h1>Foo</h1>
#         <p>Some intro text about Foo.</p>
#         <div>
#             <h2>Bar main section</h2>
#             <p>Some intro text about Bar.</p>
#             <h3>Bar subsection 1</h3>
#             <p>Some text about the first subtopic of Bar.</p>
#             <h3>Bar subsection 2</h3>
#             <p>Some text about the second subtopic of Bar.</p>
#         </div>
#         <div>
#             <h2>Baz</h2>
#             <p>Some text about Baz</p>
#         </div>
#         <br>
#         <p>Some concluding text about Foo</p>
#     </div>
# </body>
# </html>
# """

# headers_to_split_on=[
#     ("h1","Header 1"),
#     ("h2","Header 2"),
#     ("h3","Header 3")
# ]

# html_splitter=HTMLHeaderTextSplitter(headers_to_split_on)
# html_header_splits=html_splitter.split_text(html_string)
# print(html_header_splits)

# Demonstrate use of HTML Header Splitter
# data from url
# url = 'https://plato.stanford.edu/entries/goedel/'

# headers_to_split_on=[
#     ("h1","Header 1"),
#     ("h2","Header 2"),
#     ("h3","Header 3"),
#     ("h4","Header 4")
# ]
# html_splitter=HTMLHeaderTextSplitter(headers_to_split_on)
# html_header_splits=html_splitter.split_text_from_url(url)
# print(html_header_splits)

# PyPDF Loader
# loader = PyPDFLoader('attention.pdf')
# docs=loader.load()
# print(docs)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# final_documents = text_splitter.split_documents(docs)
# # print(final_documents)
# print(final_documents[0])
# print(final_documents[1])



# Web Based Loader
# loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
#                         class_=('post-title', 'post-content', 'post-header')
#                     )))
# print(loader.load())

#ArxivLoader - for loading research papers
# docs = ArxivLoader(query="1605.08386", load_max_docs=2).load()
# len(docs)
# print(docs)

#Wikipedia Loader
# loader = WikipediaLoader(query="Artificial Intelligence", load_max_docs=2)
# docs = loader.load()
# for i, doc in enumerate(docs):
#     print(f"--- Document {i+1} ---")
#     print(f"Source URL: {doc.metadata['source']}")
#     print(f"Title: {doc.metadata['title']}")
#     print(f"Content Preview: {doc.page_content[:200]}...\n")

################################################################################################################################################################
# Demonstrate use of recursive character splitter and the embedding
# into Chroma. Showcase query retrival of documents
# loader=TextLoader('speech.txt')
# docs=loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# final_documents = text_splitter.split_documents(docs)
# embeddings_1024 = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
# db=Chroma.from_documents(final_documents, embeddings_1024)

# query="It will be all the easier for us to conduct ourselves as belligerents in a high spirit of right and fairness because we act without animus, not in enmity toward a people or with the desire to bring any injury or disadvantage"
# retrieved_results = db.similarity_search(query)
# print(retrieved_results)
################################################################################################################################################################

################################################################################################################################################################
# Demonstrate basic LCEL chaining. Output results with StrOutputParser
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', "You are an expert AI Engineer. Provide me answers based on the question"),
        ('user', "{input}")
    ]
)
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(model="gpt-4o")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser
response = chain.invoke({"input":"Can you tell me about Langsmith?"})
print(response)
################################################################################################################################################################