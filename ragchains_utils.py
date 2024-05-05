from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# prompt_guidance=f"""
# Please guide the user with the following information:
# {retreived_content}
# The user's question was: {question}
# """ 

# # Chain
# rag_chain = (
#     {"context": retreived_content | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

#Context = retreived documents


##CHAIN 1##
def rangchain1(user_question: str, context: List[str], chain_type: str = "stuff") -> str:
    """
    This is the main function to chat with the content you have

    @param
    user_question: question or context user wants to figure out
    text: list of similar texts
    chat_type: Type of chain run (stuff is cost effective)

    @return
    answers from the LLM
    """
    llm = OpenAI(temperature=0, openai_api_key = openai_api_key)
    chain = load_qa_with_sources_chain(llm, chain_type = chain_type, verbose = False)

    chain_response = chain.run(input_documents = context, question = user_question )
    print(chain_response)

    return chain_response


##CHAIN 2
def ragchain2(user_question: str, context: List[str])

template = """
User: You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

Keep in mind, you will lose the job, if you answer out of CONTEXT questions

CONTEXT: {context}
Query: {question}

Remember only return AI answer
Assistant:
"""
prompt = ChatPromptTemplate.from_template(template)

# G- Generator
llm = Ollama(model="llama2",callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

print("Generating the response....")
output_parser = StrOutputParser()
chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | output_parser
)

print("\nAI response:.... \n")
query = "what is the project goal?"
print(chain.invoke(query))