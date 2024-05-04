#pip install python-dotenv

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Access TogetherAI API key
togetherai_api_key = os.getenv("TogetherAI_API_KEY")


pinecone_api_key = os.getenv("Pinecone_API_KEY")
pinecone_env = os.getenv("Pinecone_ENV")
pinecone_index = os.getenv("Pinecone_INDEX")




prompt = f"""
Based on the following information:\n\n
1. {context}\n\n
2. {response2}\n\n
3. {response3}\n\n
Please provide a detailed answer to the question: {question}.
Your answer should integrate the essence of all three responses, providing a unified answer that leverages the \
diverse perspectives or data points provided by three responses. \
If the responses are irrelevant to the question then respond by saying that I couldn't find a good response to your query in the database. 
"""


import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

#### INDEXING ####

# Load pdf Documents


# Creates Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create embeddings
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What is Task Decomposition?")