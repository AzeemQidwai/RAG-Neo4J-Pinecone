#pip install python-dotenv
#pip install sentence_transformers
#pip install gpt4all
#pip install --upgrade --quiet  spacy

import openai
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import spacy
import numpy as np
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
import sentence_transformers

#Langchain Embeddings info at https://js.langchain.com/docs/integrations/text_embedding

#spacy.load('en_core_web_lg')
spacy.load('en_core_web_sm')

# Load environment variables from .env file
load_dotenv()


# Initialize the OpenAI client with your API key
openai_api_key = os.getenv("OPENAI_API_KEY")
HF_Token = os.getenv("Huggingface_TOKEN")


def lc_openai_embedding(): ## default 3072 dimension embeddings
     model="text-embedding-3-large" 
     embeddings = OpenAIEmbeddings(
            model = model,
            #dimensions = 1536,
            openai_api_key=openai_api_key)
     return embeddings

def openai_embedding():  ## default 1536 dimension embeddings
    model="text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(
        model = model,
        openai_api_key=openai_api_key)
    return embeddings     


def spacy_embedding():  ## default 96 dimension embeddings
    model ="en_core_web_sm"
    embeddings = SpacyEmbeddings(model_name=model)
    return embeddings



def generate_huggingface_embeddings(): ## default 768 dimension embeddings
    model_name = "BAAI/bge-base-en-v1.5"   #https://huggingface.co/BAAI/bge-small-en-v1.5  (contains many embedding function)
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
    return embeddings


def generate_gpt4all(): # default 384 dimension embeddings
   embeddings = GPT4AllEmbeddings(openai_api_key=openai_api_key)
   return embeddings




##EXAMPLES 

# content = ["How are you doin?","I am great"]
# embedding = generate_gpt4all().embed_documents(content)
# len(embedding)

################################################
# text= "How are you doin?"
# embedding = generate_gpt4all().embed_query(text)
# len(embedding)



# def lc_openai_embedding(text): ## default 3072 dimension embeddings
#      model="text-embedding-3-large" 
#      embeddings = OpenAIEmbeddings(
#             model = model,
#             #dimensions = 1536,
#             openai_api_key=openai_api_key).embed_documents(text)
#      return embeddings


# def lc_openai_embedding(text): ## default 3072 dimension embeddings
#      model="text-embedding-3-large" 
#      embeddings = OpenAIEmbeddings(
#             model = model,
#             #dimensions = 1536,
#             openai_api_key=openai_api_key).embed_query(text)
#      return embeddings