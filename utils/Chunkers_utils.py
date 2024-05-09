#pip install nltk
#pip install --quiet langchain_experimental langchain_openai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
import nltk
from nltk.tokenize import sent_tokenize
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.schema.document import Document

##These splitter function works well with pdfloader 

# Load environment variables from .env file
load_dotenv('.env')

openai_api_key = os.getenv("OPENAI_API_KEY")

def recursive(txt_doc):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 100,
        chunk_overlap  = 5,
        length_function = len,
        is_separator_regex = False,
    )
    # text splitter
    splits = text_splitter.split_documents(txt_doc)
    return splits


def character(txt_doc):
    text_splitter = CharacterTextSplitter(
    separator = ".",
    chunk_size = 200,
    chunk_overlap = 20 #always less than chunk size
    )
    characters = text_splitter.split_documents(txt_doc)
    return characters


def sentence(txt_doc):
    nltk.download('punkt')
    text = "".join([page.page_content for page in txt_doc])
    print(len(text))
    sentences = sent_tokenize(text)
    split_docs = [Document(page_content=x) for x in sentences]
    return split_docs


def paragraphs(txt_doc):
    text = "".join([page.page_content for page in txt_doc])
    print(len(text))
    paragraphs = text.split('\n') # Assuming paragraphs are separated by two newlines
    split_docs = [Document(page_content=x) for x in paragraphs]
    return split_docs

def semantic(txt_doc):
    semantic_splits = SemanticChunker(OpenAIEmbeddings(openai_api_key=openai_api_key))
    docs = semantic_splits.split_documents(txt_doc)
    return docs



##EXAMPLE

# text = pdf_to_text('../input/Constitution.pdf')
# text1 = pdf_to_text_plumber('../input/Constitution.pdf')
# text2 = pdfloader('../input/Constitution.pdf')


# chunks = semantic(text)
# len(chunks)
# chunks


# for index, chunk in enumerate(chunks):
#     embedding= generate_gpt4all().embed_query(chunk.page_content)
#     print(len(embedding))

