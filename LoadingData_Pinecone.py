#pip install -q pinecone-client
#pip install --upgrade -q pinecone-client

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
from embeddings_utils import get_openai_embedding, generate_huggingface_embeddings, generate_gpt4all
from Chunkers_utils import recursive, character, sentence, paragraphs



load_dotenv('.env')

pinecone_api_key = os.getenv("Pinecone_API_KEY")
pinecone_env = os.getenv("Pinecone_ENV")
pinecone_index = os.getenv("Pinecone_INDEX")


pc = Pinecone()
pc.list_indexes()

##ONLY RUN IF YOU HAVEN'T CREATED THE INDEX ON PINECONE
#This represents the configuration used to deploy a pod-based index.
index_name = 'RagCP'                             # create a new index called "langchain"

if index_name not in pc.list_indexes().names():
    print(f'Creating index {index_name}')
    pc.create_index(
        name = index_name,
        dimension = 1536,  #This is the default dimension for text-embedding-3-small(one of the recommended OpenAI's embedding models.)
        metric = 'cosine',  # This is the algorithm used to calculate the distance between vectors.
        spec = PodSpec(
            environment = 'gcp-starter'
        ) 
    )
    print('Index created! :)')
else:
     print(f'Index {index_name} already exists!')


# index_name = 'RagCP'
# if index_name in pc.list_indexes().names():
#     print(f'Deleting index {index_name} ...')
#     pc.delete_index(index_name)
#     print('Done')
# else:
#     print(f'Index {index_name} does not exists!')



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
            embedding = get_openai_embedding(sub_docs.page_content)
        elif embeddingtype == 'HF':
            embedding = generate_huggingface_embeddings(sub_docs.page_content)
        else:
            embedding = generate_gpt4all(sub_docs.page_content)


        metadata = {"doc_name":file_name, "chunk": str(uuid.uuid4()), "text": sub_docs.page_content, "doc_index":index}
        pinecone_index.upsert([(document_hash.hexdigest(), embedding, metadata)])
        print("{} ==> Done".format(index))

    print("Done!")

    return True


