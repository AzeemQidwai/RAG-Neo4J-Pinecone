from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector




# Configure the Neo4j connection
neo4j_config = {
    "uri": "neo4j+s://60ef2a6c.databases.neo4j.io:7687",
    "username": "neo4j",
    "password": "zFAH7hipCN030a67Cf9GfD1ADeYGavu51Umh9DJtffA"
}


# Typical RAG retriever

typical_rag = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(open), index_name="neo4j",
    config=neo4j_config
)

# Parent retriever

parent_query = """
MATCH (node)<-[:HAS_CHILD]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata LIMIT 1
"""

parent_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    index_name="parent_document",
    retrieval_query=parent_query,
)

# Hypothetic questions retriever

hypothetic_question_query = """
MATCH (node)<-[:HAS_QUESTION]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata
"""

hypothetic_question_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    index_name="hypothetical_questions",
    retrieval_query=hypothetic_question_query,
)
# Summary retriever

summary_query = """
MATCH (node)<-[:HAS_SUMMARY]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata
"""

summary_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    index_name="summary",
    retrieval_query=summary_query,
)