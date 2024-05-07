#pip install nltk
#pip install --quiet langchain_experimental langchain_openai


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
import nltk
from nltk.tokenize import sent_tokenize
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os



# Load environment variables from .env file
load_dotenv('.env')

openai_api_key = os.getenv("OPENAI_API_KEY")

def recursive(txt_doc):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len,
        is_separator_regex = False,
    )
    # text splitter
    splits = text_splitter.split_documents(txt_doc)
    return splits


def character(txt_doc):
    text_splitter = CharacterTextSplitter(
    separator = ".",
    chunk_size = 800,
    chunk_overlap = 80 #always less than chunk size
    )
    characters = text_splitter.split_text(txt_doc)
    return characters


def sentence(txt_doc):
    nltk.download('punkt')
    sentences = sent_tokenize(txt_doc)
    return sentences


def paragraphs(text):
    paragraphs = text.split('\n\n')  # Assuming paragraphs are separated by two newlines
    return paragraphs

def semantic(text):
    if not isinstance(text, str):
        raise TypeError("The 'text' argument must be a string.")
    semantic_splits = SemanticChunker(OpenAIEmbeddings(openai_api_key=openai_api_key))
    docs = semantic_splits.create_documents([text])
    return docs



##EXAMPLE
# with open('corpus.txt', 'r', encoding='utf-8') as file:
#     content = file.read()
#     print(content)

#text = "Hello world. This is an example text to demonstrate sentence splitting. Enjoy using this script!"
# characters = character(text)

# for idx, character in enumerate(characters):
#     print(f"Character {idx+1}: {character}")