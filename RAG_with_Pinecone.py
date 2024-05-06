#pip install python-dotenv
#pip install -q pinecone-client
#pip install --upgrade -q pinecone-client
#pip install utils


from langchain import hub
import hashlib
from typing import List
import pinecone
from pinecone import Pinecone
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from utils.pdf_utils import pdf_to_text, pdf_to_text_plumber, pdfloader
from utils.Chunkers_utils import recursive, character, sentence, paragraphs, semantic
from utils.embeddings_utils import  lc_openai_embedding, openai_embedding, generate_huggingface_embeddings, generate_gpt4all
from LoadingData_Pinecone import upload_to_pinecone, filter_matching_docs
from utils.LLM_utils import infer_Mixtral, infer_llama3, infer_llama2, infer_Qwen, infer_gpt4

import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

##API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
togetherai_api_key = os.getenv("TogetherAI_API_KEY")
pinecone_api_key = os.getenv("Pinecone_API_KEY")
pinecone_env = os.getenv("Pinecone_ENV")
pinecone_index = os.getenv("Pinecone_INDEX")



#Select Options
chunker = 'recursive'  #recursive, character, sentence, paragraphs, semantic
embeddingtype = 'openai' #openai, HF, gpt4all, 'langchain'


###INDEXING###

## Load pdf Documents
text = pdfloader('source/Constitution.pdf')

# Creates Chunks
if chunker == 'recursive':
    chunks = recursive(text)
elif chunker == 'character':
    chunks = character(text)
elif chunker == 'sentence':
    chunks = sentence(text)
elif chunker == 'semantic':
    chunks = semantic(text)
else:
    chunks = paragraphs(text)


## Loading data to Pinecone
upload_to_pinecone('Constitution', chunks, embeddingtype)

#QuestionEmbeddings

##load questions
# Path to the JSON file
file_path = 'output/questions.json'

# Open the file and load the data
with open(file_path, 'r') as file:
    data = json.load(file)

print(data)
questions = data['question']


#### RETRIEVAL & GENERATION####


question_responses = {}
##Question embeddings
for question in questions:
    if embeddingtype == 'openai':
        q_embedding = openai_embedding(question)
    elif embeddingtype == 'HF':
        q_embedding = generate_huggingface_embeddings(question)
    elif embeddingtype == 'langchain':
        q_embedding = lc_openai_embedding(question)
    else:
        q_embedding = generate_gpt4all(question)


    ##Return Context##

    retrieved_content = filter_matching_docs(q_embedding, 3, get_text = True)
    #print(f"Retreived content: {retrieved_content}")

    ##Creating prompt##

    prompt = f"""
    You are an AI assistant that is expert in Pakistan Constitution.
    Based on the following CONTEXT: \n\n
    {retrieved_content} \n\n
    Please provide a detailed answer to the question: {question}.\n\n
    Please be truthful. Keep in mind, you will lose the job, if you answer out of CONTEXT questions.
    If the responses are irrelevant to the question then respond by saying that I couldn't find a good response to your query in the database. 
    """

    #QA WITH LLM#
    response = infer_gpt4(prompt=prompt)

    # Store retrieved content and response in dictionary
    question_responses[question] = {
        'retrieved_content': retrieved_content,
        'response': response
    }
    print(response)


def save_to_json(question_responses, json_output_file):
    # Create a list to hold the results in the required structure
    results = {"questions": [], "answers": [], "contexts": []}

    # Iterate through the question_responses dictionary
    for question, data in question_responses.items():
        results["questions"].append(question)
        results["answers"].append(data['response'])
        results["contexts"].append(data['retrieved_content'])
    
    # Write the results to a JSON file
    with open(json_output_file, 'w') as file:
        json.dump(results, file, indent=4)

# Output JSON file path
json_output_file = 'output/qa_results_pc.json'

# Execute the function
save_to_json(question_responses, json_output_file)