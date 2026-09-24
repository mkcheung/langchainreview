from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, ArxivLoader, WikipediaLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import bs4
import wikipedia

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
loader=TextLoader('speech.txt')
docs=loader.load()
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=50)
text = text_splitter.split_documents(docs)
print(text[0])
print(text[1])

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