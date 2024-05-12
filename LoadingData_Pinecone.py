#pip install -q pinecone-client
#pip install --upgrade -q pinecone-client
#pip install utils

import hashlib
from pinecone import PodSpec
import os
import uuid
from typing import List
import pinecone
from pinecone import Pinecone
import openai
import os
from dotenv import load_dotenv
from utils import embeddings_utils, Chunkers_utils, pdf_utils, LLM_utils
from utils.embeddings_utils import  lc_openai_embedding, spacy_embedding, openai_embedding, generate_huggingface_embeddings, generate_gpt4all




load_dotenv('.env')


pinecone_api_key = os.getenv("Pinecone_API_KEY")
pinecone_env = os.getenv("Pinecone_ENV")
pinecone_index = os.getenv("Pinecone_INDEX")


pc = Pinecone(api_key=pinecone_api_key)
pc.list_indexes()
pc_index = pc.Index(pinecone_index)


def upload_to_pinecone(file_name, chunks, embeddingtype) -> None:
    """
    Upload the text content to pinecone

    @params
    text_document: text content needs to upload
    file_name: name of the filed to be included as metadata
    chunk_type: chunk type to split the data
    embeddingtype: type of embedding model

    @return
    None
    """

    for index, sub_docs in enumerate(chunks):
        document_hash = hashlib.md5(sub_docs.page_content.encode("utf-8"))
        if embeddingtype == 'openai':
            embedding = openai_embedding().embed_documents(sub_docs.page_content)  ## default 1536 dimension embeddings
        elif embeddingtype == 'HF':
            embedding = generate_huggingface_embeddings().embed_documents(sub_docs.page_content)  ## default 768 dimension embeddings
        elif embeddingtype == 'langchain':
            embedding = lc_openai_embedding().embed_documents(sub_docs.page_content) ## default 3072 dimension embeddings
        elif embeddingtype == 'spacy':
            embedding = spacy_embedding().embed_documents(sub_docs.page_content)  ## default 96 dimension embeddings
        else:
            embedding = generate_gpt4all().embed_documents(sub_docs.page_content) # default 384 dimension embeddings


        metadata = {"doc_name":file_name, "chunk": str(uuid.uuid4()), "text": sub_docs.page_content, "doc_index":index}
        pc_index.upsert([(document_hash.hexdigest(), embedding[0], metadata)])
        print("{} ==> Done".format(index))

    print("Done!")

    return True


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

    response = pc_index.query(vector=query_embeds, top_k=top_chunks, include_metadata=True)


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
        #print(similar_text)
        return similar_text

    #print(filtered_data)
    return filtered_data

# ##ONLY RUN IF YOU HAVEN'T CREATED THE INDEX ON PINECONE
# #This represents the configuration used to deploy a pod-based index.
# index_name =  pinecone_index                            # create a new index called "langchain"

# if index_name not in pc.list_indexes().names():
#     print(f'Creating index {index_name}')
#     pc.create_index(
#         name = index_name,
#         dimension = 1536,  #This is the default dimension for text-embedding-3-small(one of the recommended OpenAI's embedding models.)
#         metric = 'cosine',  # This is the algorithm used to calculate the distance between vectors.
#         spec = PodSpec(
#             environment = 'gcp-starter'
#         ) 
#     )
#     print('Index created! :)')
# else:
#      print(f'Index {index_name} already exists!')


##ONLY RUN IF YOU WANT TO DELETE THE INDEX ON PINECONE
# # index_name = pinecone_index
# # if index_name in pc.list_indexes().names():
# #     print(f'Deleting index {index_name} ...')
# #     pc.delete_index(index_name)
# #     print('Done')
# # else:
# #     print(f'Index {index_name} does not exists!')