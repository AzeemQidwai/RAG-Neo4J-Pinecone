#pip install python-dotenv
#pip install sentence_transformers
#pip install gpt4all

import openai
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import sentence_transformers

# Load environment variables from .env file
load_dotenv()


# Initialize the OpenAI client with your API key
openai_api_key = os.getenv("OPENAI_API_KEY")
HF_Token = os.getenv("Huggingface_TOKEN")


def lc_openai_embedding(text):
     model="text-embedding-3-large" ## 3072 dimension embeddings
     embeddings = OpenAIEmbeddings(
            model = model,
            dimensions = 1536,
            openai_api_type=openai_api_key).embed_query(text)
     return embeddings

def openai_embedding(text):
    model="text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(
        model = model,
        openai_api_type=openai_api_key).embed_query(text)
    return embeddings           


def generate_huggingface_embeddings(text):
    model_name = "nomic-ai/nomic-embed-text-v1"  #nomic-ai/nomic-embed-text-v1 #sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja   #https://huggingface.co/BAAI/bge-small-en-v1.5
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        ).embedDocuments(text)
    return embeddings

 

def generate_gpt4all(text):
   embeddings = GPT4AllEmbeddings().embed_query(text) # dim 384
   return embeddings




# with open('corpus.txt', 'r', encoding='utf-8') as file:
#     content = file.read()
#     print(content)

# get_openai_embedding(content)

# generate_gpt4all(content)

#text="How are you doin?"

#embedding = generate_gpt4all(text)
#len(embedding)