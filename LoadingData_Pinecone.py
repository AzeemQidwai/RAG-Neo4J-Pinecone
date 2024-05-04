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
from embeddings_utils import get_openai_embedding, generate_huggingface_embeddings, generate_gpt4all
from Chunkers_utils import recursive, character, sentence, paragraphs



load_dotenv('.env')

pinecone_api_key = os.getenv("Pinecone_API_KEY")
pinecone_env = os.getenv("Pinecone_ENV")
pinecone_index = os.getenv("Pinecone_INDEX")


pc = Pinecone()
pc.list_indexes()



def upload_to_pinecone(text_document: str, file_name, chunker, embeddingtype) -> None:
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

    if chunker == 'recursive':
        texts = recursive(text_document)
    elif chunker == 'character':
        texts = character(text_document)
    elif chunker == 'sentence':
        texts = sentence(text_document)
    else:
        texts = paragraphs(text_document)

    for index, sub_docs in enumerate(texts):
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


