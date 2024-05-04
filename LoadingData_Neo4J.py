import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')



# Access Neo4j credentials
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_uri = os.getenv("NEO4J_URI")