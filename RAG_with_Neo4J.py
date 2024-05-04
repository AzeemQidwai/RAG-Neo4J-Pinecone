#pip install python-dotenv
import openai
from openai import OpenAI
import together
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv('.env')

# Access OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Access TogetherAI API key
togetherai_api_key = os.getenv("TOGETHER_API_KEY")

# Access Neo4j credentials
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_uri = os.getenv("NEO4J_URI")




prompt = f"""
Based on the following information:\n\n
1. {context}\n\n
2. {response2}\n\n
3. {response3}\n\n
Please provide a detailed answer to the question: {question}.
Your answer should integrate the essence of all three responses, providing a unified answer that leverages the \
diverse perspectives or data points provided by three responses. \
If the responses are irrelevant to the question then respond by saying that I couldn't find a good response to your query in the database. 
"""