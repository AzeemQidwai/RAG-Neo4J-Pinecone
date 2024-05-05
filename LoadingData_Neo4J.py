#pip install neo4j 

import os
from dotenv import load_dotenv
import os
import logging
from retry import retry
from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain_community.graphs import Neo4jGraph
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

# Load environment variables from .env file
load_dotenv('.env')


# Access Neo4j credentials
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_url = os.getenv("NEO4J_URL")
neo4j_db = os.getenv("NEO4J_DB")

# Configure the logging module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


LOGGER = logging.getLogger(__name__)


def upload_to_neo4j():
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
    # Connect to the knowledge graph instance using LangChain
    graph = Neo4jGraph(
        url=neo4j_url, 
        username=neo4j_username, 
        password=neo4j_username
        )
    










# Instantiate Neo4j vector from documents
neo4j_vector = Neo4jVector.from_documents(
    documents,
    OpenAIEmbeddings(),
    url=neo4j_url,
    username=neo4j_username,
    password=neo4j_password
)