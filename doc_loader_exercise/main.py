from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, ArxivLoader, WikipediaLoader
import bs4
import wikipedia

wikipedia.set_user_agent("langchaincourse-exercise/1.0 (mars.kwong.cheung@gmail.com)")

# Text Loader
# loader=TextLoader('speech.txt')
# text_documents=loader.load()
# print(text_documents)

# PyPDF Loader
# loader = PyPDFLoader('attention.pdf')
# docs=loader.load()
# print(docs)

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

loader = WikipediaLoader(query="Artificial Intelligence", load_max_docs=2)
docs = loader.load()
for i, doc in enumerate(docs):
    print(f"--- Document {i+1} ---")
    print(f"Source URL: {doc.metadata['source']}")
    print(f"Title: {doc.metadata['title']}")
    print(f"Content Preview: {doc.page_content[:200]}...\n")