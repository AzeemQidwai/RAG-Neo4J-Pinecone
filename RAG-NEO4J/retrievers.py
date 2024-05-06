from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
import os
from dotenv import load_dotenv

load_dotenv('../.env')
# Access the variables

openai_api_key = os.getenv("OPENAI_API_KEY")
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
db = os.getenv("NEO4J_DB")

# Typical RAG retriever

typical_rag = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(openai_api_key=openai_api_key), index_name="typical_rag",
    url=uri,
    username=username,
    password=password)

# Parent retriever

parent_query = """
MATCH (node)<-[:HAS_CHILD]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata LIMIT 1
"""

parent_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    index_name="parent_document",
    retrieval_query=parent_query,
    url=uri,
    username=username,
    password=password
)

# Hypothetic questions retriever

hypothetic_question_query = """
MATCH (node)<-[:HAS_QUESTION]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata
"""

hypothetic_question_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    index_name="hypothetical_questions",
    retrieval_query=hypothetic_question_query,
    url=uri,
    username=username,
    password=password
)
# Summary retriever

summary_query = """
MATCH (node)<-[:HAS_SUMMARY]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata
"""

summary_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    index_name="summary",
    retrieval_query=summary_query,
    url=uri,
    username=username,
    password=password
)