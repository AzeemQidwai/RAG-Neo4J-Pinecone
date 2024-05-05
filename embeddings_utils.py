#pip install python-dotenv

import openai
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import GPT4AllEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Initialize the OpenAI client with your API key
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_openai_embedding(text_to_embed):
	# Embed a line of text
	response = openai.Embedding.create(
    	model= "text-embedding-ada-002",  ##you can use other models 1. `text-similarity-babbage-001`, `text-similarity-curie-001`, `text-embedding-ada-002`
    	input=[text_to_embed]
	)
	# Extract the AI output embedding as a list of floats
	embedding = response["data"][0]["embedding"]
    
	return embedding


def generate_huggingface_embeddings(text):
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
    return embeddings

def generate_gpt4all(text):
   embeddings = GPT4AllEmbeddings().embed_query(text)
   return embeddings


with open('corpus.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    print(content)

get_openai_embedding(content)

generate_gpt4all(content)
