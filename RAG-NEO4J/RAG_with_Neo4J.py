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

embeddingtype = 'openai' #openai, HF, gpt4all



from chain import chain

if __name__ == "__main__":
    original_query = "What is the plot of the Dune?"
    print(  # noqa: T201
        chain.invoke(
            {"question": original_query},
            {"configurable": {"strategy": "parent_document"}},
        )
    )

