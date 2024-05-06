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
from neo4j import GraphDatabase

import os
from dotenv import load_dotenv
from utils.embeddings_utils import get_openai_embedding, generate_huggingface_embeddings, generate_gpt4all
from utils.Chunkers_utils import recursive, character, sentence, paragraphs

# Load environment variables from .env file
load_dotenv('.env')


# Access Neo4j credentials
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_url = os.getenv("NEO4J_URL")
neo4j_db = os.getenv("NEO4J_DB")
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader

# Load the text file
  # Ensures loader uses metadata, adjust according to actual implementation


pdfpath="source/Constitution.pdf"
loader = PyPDFLoader(pdfpath)   
documents = loader.load()

graph = Neo4jGraph(
    url=neo4j_url, username=neo4j_username, password=neo4j_password
    )
parent_documents = documents 

for i, parent in enumerate(parent_documents):
    # Create child splits from each parent page if necessary
    child_documents = child_splitter.split_documents([parent])
    page_metadata = parent.metadata  # Access metadata, adjust according to your data structure
    params = {
        "parent_text": parent.page_content,
        "parent_id": f"{page_metadata['source']}-{page_metadata['page']}",  # Unique ID using source and page number
        "parent_embedding": embeddings.embed_query(parent.page_content),
        "children": [
            {
                "text": c.page_content,
                "id": f"{page_metadata['source']}-{page_metadata['page']}-{ic}",
                "embedding": embeddings.embed_query(c.page_content),
            }
            for ic, c in enumerate(child_documents)
        ],
    }
    # Ingest data with updated query using metadata in IDs
    graph.query(
        """
        MERGE (p:Parent {id: $parent_id})
        SET p.text = $parent_text
        WITH p
        CALL db.create.setVectorProperty(p, 'embedding', $parent_embedding)
        YIELD node
        WITH p 
        UNWIND $children AS child
        MERGE (c:Child {id: child.id})
        SET c.text = child.text
        MERGE (c)<-[:HAS_CHILD]-(p)
        WITH c, child
        CALL db.create.setVectorProperty(c, 'embedding', child.embedding)
        YIELD node
        RETURN count(*)
        """,
        params,
    )














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
