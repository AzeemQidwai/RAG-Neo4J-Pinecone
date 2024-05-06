#pip install python-dotenv

import openai
from openai import OpenAI
import os
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv('.env')

# Access OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")





from chain import chain

if __name__ == "__main__":
    original_query = "In what scenarios is the delay of general elections acceptable?"
    print(  # noqa: T201
        chain.invoke(
            {"question": original_query},
            {"configurable": {"strategy": "parent_document"}},
        )
    )

