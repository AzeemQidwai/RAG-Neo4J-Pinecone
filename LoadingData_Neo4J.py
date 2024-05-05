#pip install neo4j 

from pathlib import Path
from typing import List

from langchain.chains.openai_functions import create_structured_output_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_text_splitters import TokenTextSplitter
from neo4j.exceptions import ClientError

import os
from dotenv import load_dotenv
from embeddings_utils import get_openai_embedding, generate_huggingface_embeddings, generate_gpt4all
from Chunkers_utils import recursive, character, sentence, paragraphs



# Load environment variables from .env file
load_dotenv('.env')


# Access Neo4j credentials
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_url = os.getenv("NEO4J_URL")
neo4j_db = os.getenv("NEO4J_DB")



# Global constants
VECTOR_INDEX_NAME = 'RagCP'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'


from neo4j import GraphDatabase

for index, sub_docs in enumerate(chunks):
    document_hash = hashlib.md5(sub_docs.page_content.encode("utf-8"))
    if embeddingtype == 'openai':
        embedding = get_openai_embedding(sub_docs.page_content)
    elif embeddingtype == 'HF':
        embedding = generate_huggingface_embeddings(sub_docs.page_content)
    else:
        embedding = generate_gpt4all(sub_docs.page_content)
    









# class EmbeddingStore:
#     def __init__(self, uri, user, password):
#         self.driver = GraphDatabase.driver(uri, auth=(user, password))

#     def close(self):
#         self.driver.close()

#     def create_node_with_embedding(self, node_id, embedding, document_name):
#         with self.driver.session() as session:
#             session.write_transaction(self._create_and_set_embedding, node_id, embedding, document_name)

#     @staticmethod
#     def _create_and_set_embedding(tx, node_id, embedding, document_name):
#         query = (
#             "CREATE (n:Node {id: $node_id, embedding: $embedding, document_name: $document_name}) "
#             "RETURN n"
#         )
#         result = tx.run(query, node_id=node_id, embedding=embedding, document_name=document_name)
#         try:
#             return result.single()[0]
#         except Exception as e:
#             print("Error creating node:", e)
#             return None

# # Example usage
# if __name__ == "__main__":
#     uri = "bolt://localhost:7687"  # Neo4j bolt URI
#     user = "neo4j"                 # Neo4j username
#     password = "password"          # Neo4j password
#     store = EmbeddingStore(uri, user, password)
    
#     # Example data
#     embeddings = {
#         'node1': ([0.1, 0.2, 0.3, 0.4], 'Document1'),
#         'node2': ([0.5, 0.6, 0.7, 0.8], 'Document2')
#     }
    
#     for node_id, (embedding, document_name) in embeddings.items():
#         store.create_node_with_embedding(node_id, embedding, document_name)
    
#     store.close()
