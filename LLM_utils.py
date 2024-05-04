#%pip install --upgrade --quiet langchain-together

import together
import openai
from openai import OpenAI
from langchain_together import Together
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceEndpoint


load_dotenv('.env')

###TOGETHERAI MODELS
openai_api_key = os.getenv("OPENAI_API_KEY")
togetherai_api_key = os.getenv("TOGETHER_API_KEY")
client = OpenAI()

def infer_Mixtral(prompt):
    llm = Together(
    model="mistralai/Mixtral-8x22B",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=togetherai_api_key
    )
    candidate = llm.invoke(prompt)
    return candidate


def infer_llama3(prompt):
    llm = Together(
    model="meta-llama/Llama-3-8b-chat-hf", #"meta-llama/Llama-3-8b-hf",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=togetherai_api_key
    )
    candidate = llm.invoke(prompt)
    return candidate

def infer_llama2(prompt):
    llm = Together(
    model="togethercomputer/LLaMA-2-7B-32K", 
    temperature=0.5,
    max_tokens=1024,
    together_api_key=togetherai_api_key
    )
    candidate = llm.invoke(prompt)
    return candidate

def infer_Qwen(prompt):
    llm = Together(
    model="Qwen/Qwen1.5-1.8B-Chat",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=togetherai_api_key
    )
    candidate = llm.invoke(prompt)
    return candidate


def infer_gpt4(prompt):
    llm = ChatOpenAI()
    prompt=prompt
    candidate = llm.invoke(prompt,model='gpt-4-turbo-preview')
    return candidate.content


# # Example usage
# prompt_text = "Translate the following English text to French: 'Hello, how are you?'"
# response_text = infer_llama2(prompt_text)
# print(response_text)



##HUGGINGFACE

from getpass import getpass
Huggingface_TOKEN = getpass()

HF_token = os.getenv("Huggingface_TOKEN")



#repo_id = "HuggingFaceH4/zephyr-7b-beta"
#repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
repo_id = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8"
#repo_id = "google/gemma-7b"

def infer_HF(prompt):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, token=HF_token
        )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(prompt)


# prompt_text = "Translate the following English text to French: 'Hello, how are you?'"
# response_text = infer_HF(prompt_text)
# print(response_text)