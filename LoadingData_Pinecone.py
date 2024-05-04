#pip install -q pinecone-client
#pip install --upgrade -q pinecone-client

import hashlib
import os
import uuid
from typing import List
import pinecone
from pinecone import Pinecone
import openai

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv


load_dotenv('.env')



pinecone_api_key = os.getenv("Pinecone_API_KEY")
pinecone_env = os.getenv("Pinecone_ENV")
pinecone_index = os.getenv("Pinecone_INDEX")

pc = Pinecone()
pc.list_indexes()



def upload_to_pinecone(text_document: str, file_name, chunker,embeddings) -> None:
    """
    Upload the text content to pinecone

    @params
    text_document: text content needs to upload
    file_name: name of the filed to be included as metadata
    chunk_type: chunk size to split the data

    @return
    None
    """

    MODEL = "text-embedding-ada-002"

    texts = chunker(text_document)

    for index, sub_docs in enumerate(texts):
        document_hash = hashlib.md5(sub_docs.page_content.encode("utf-8"))
        embedding = openai.embeddings.create(model= MODEL,input=sub_docs.page_content).data[0].embedding
        metadata = {"doc_name":file_name, "chunk": str(uuid.uuid4()), "text": sub_docs.page_content, "doc_index":index}
        pinecone_index.upsert([(document_hash.hexdigest(), embedding, metadata)])
        print("{} ==> Done".format(index))

    print("Done!")

    return True


def filter_matching_docs(question: str, top_chunks: int = 3, get_text: bool = False) -> List:
    """
    Semnatic search between user content and vector DB

    @param
    question: user question
    top_chunks: number of most similar content ot be filtered
    get_text: if True, return only the text not the document

    @return
    list of similar content
    """
    
    index=pinecone.Index(os.environ['PINECONE_INDEX_NAME'])

    question_embed_call = openai.embeddings.create(input = question ,model = MODEL)
    query_embeds = question_embed_call.data[0].embedding
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


def QA_with_your_docs(user_question: str, text_list: List[str], chain_type: str = "stuff") -> str:
    """
    This is the main function to chat with the content you have

    @param
    user_question: question or context user wants to figure out
    text: list of similar texts
    chat_type: Type of chain run (stuff is cost effective)

    @return
    answers from the LLM
    """
    llm = OpenAI(temperature=0, openai_api_key = os.environ['OPENAI_API_KEY'])
    chain = load_qa_with_sources_chain(llm, chain_type = chain_type, verbose = False)

    all_docs = []
    for doc_content in text_list:
        metadata = {}
        doc_text = doc_content.get("text", "")
        metadata["id"] = doc_content.get("id", "")
        metadata["score"] = doc_content.get("score", "")
        metadata["filename"] = doc_content.get("filename", "")
        metadata["chunk"] = doc_content.get("chunk", "")
        chunk_name = doc_content.get("filename", "UNKNOWN")
        offset=", OFFSET="+str(doc_content.get("chunk","UNKNOWN"))
        metadata["source"] = chunk_name + offset
        all_docs.append(Document(page_content = doc_text, metadata = metadata))

    chain_response = chain.run(input_documents = all_docs, question = user_question )
    print(chain_response)

    return chain_response