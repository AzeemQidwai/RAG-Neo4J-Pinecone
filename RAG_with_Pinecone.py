#pip install python-dotenv
#pip install -q pinecone-client
#pip install --upgrade -q pinecone-client


from langchain import hub
import hashlib
from typing import List
import pinecone
from pinecone import Pinecone
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from pdf_utils import pdf_to_text, pdf_to_text_plumber, pdfloader
from Chunkers_utils import recursive, character, sentence, paragraphs
from embeddings_utils import get_openai_embedding, generate_huggingface_embeddings, generate_gpt4all
from LoadingData_Pinecone import upload_to_pinecone
from LLM_utils import infer_Mixtral, infer_llama3, infer_llama2, infer_Qwen, infer_gpt4


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

##API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
togetherai_api_key = os.getenv("TogetherAI_API_KEY")
pinecone_api_key = os.getenv("Pinecone_API_KEY")
pinecone_env = os.getenv("Pinecone_ENV")
pinecone_index = os.getenv("Pinecone_INDEX")


#Select Options
chunker = 'recursive'  #recursive, character, sentence, paragraphs
embeddingtype = 'openai' #openai, HF, gpt4all
question =""

###INDEXING###

## Load pdf Documents
text = pdfloader('source/Constitution')


# Creates Chunks
if chunker == 'recursive':
    chunks = recursive(text)
elif chunker == 'character':
    chunks = character(text)
elif chunker == 'sentence':
    chunks = sentence(text)
else:
    chunks = paragraphs(text)


## Loading data to Pinecone
upload_to_pinecone('Constitution', chunks, embeddingtype)

##Question embeddings
if embeddingtype == 'openai':
    query_embeds = get_openai_embedding(question)
elif embeddingtype == 'HF':
    query_embeds = generate_huggingface_embeddings(question)
else:
    query_embeds = generate_gpt4all(question)


#### RETRIEVAL & GENERATION####

##Return Context##

def filter_matching_docs(query_embeds: str, top_chunks: int = 3, get_text: bool = False) -> List:
    """
    Semnatic search between user content and vector DB

    @param
    question: user question
    top_chunks: number of most similar content ot be filtered
    get_text: if True, return only the text not the document

    @return
    list of similar content
    """
    
    index = os.getenv("Pinecone_INDEX")
    response = index.query(query_embeds,top_k = top_chunks,include_metadata = True)

    #get the data out
    filtered_data = []
    filtered_text = []

    for content in response["matches"]:
        #save the meta data as a dictionary
        info = {}
        info["id"] = content.get("id", "")
        info["score"] = content.get("score", "")
        # get the saved metadat info
        content_metadata = content.get("metadata","")
        # combine it it info
        info["filename"] = content_metadata.get("doc_name", "")
        info["chunk"] = content_metadata.get("chunk", "")
        info["text"] = content_metadata.get("text", "")
        filtered_text.append(content_metadata.get("text", ""))

        #append the data
        filtered_data.append(info)

    if get_text:
        similar_text = " ".join([text for text in filtered_text])
        print(similar_text)
        return similar_text

    print(filtered_data)

    return filtered_data



retreived_content = filter_matching_docs(query_embeds, 3, get_text = True)
#print(f"Retreived content: {retreived_content}")



##Creating prompt##

prompt = f"""
You are an AI assistant that is expert in Pakistan Constitution.
Based on the following CONTEXT: \n\n
{retreived_content}
Please provide a detailed answer to the question: {question}.
Please be truthful. Keep in mind, you will lose the job, if you answer out of CONTEXT questions.
If the responses are irrelevant to the question then respond by saying that I couldn't find a good response to your query in the database. 
"""


#QA WITH LLM#

Response = infer_gpt4(prompt=prompt)

