#pip install python-dotenv
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from pdf_utils import pdf_to_text, pdf_to_text_plumber, pdfloader
from Chunkers_utils import recursive, character, sentence, paragraphs
from embeddings_utils import get_openai_embedding, generate_huggingface_embeddings, generate_gpt4all
from LoadingData_Pinecone import upload_to_pinecone
from LLM_utils import infer_Mixtral, infer_llama3, infer_llama2, infer_Qwen, infer_gpt4


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

##API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
togetherai_api_key = os.getenv("TogetherAI_API_KEY")
pinecone_api_key = os.getenv("Pinecone_API_KEY")
pinecone_env = os.getenv("Pinecone_ENV")
pinecone_index = os.getenv("Pinecone_INDEX")


#Select Options
chunker = 'recursive'  #recursive, character, sentence, paragraphs
embeddingtype = 'openai' #openai, HF, gpt4all


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



# Load pdf Documents
text = pdfloader('')


# Creates Chunks


# Create embeddings




# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What is Task Decomposition?")


#### RETRIEVAL and GENERATION ####

def filter_matching_docs(question: str, top_chunks: int = 3, get_text: bool = False) -> List:
    """
    Semnatic search between user content and vector DB

    @param
    question: user question
    top_chunks: number of most similar content ot be filtered
    get_text: if True, return only the text not the document

    @return
    list of similar content
    """
    
    index = os.getenv("Pinecone_INDEX")

    question_embed_call = openai.embeddings.create(input = question ,model = MODEL)
    query_embeds = question_embed_call.data[0].embedding
    response = index.query(query_embeds,top_k = top_chunks,include_metadata = True)

    #get the data out
    filtered_data = []
    filtered_text = []

    for content in response["matches"]:
        #save the meta data as a dictionary
        info = {}
        info["id"] = content.get("id", "")
        info["score"] = content.get("score", "")
        # get the saved metadat info
        content_metadata = content.get("metadata","")
        # combine it it info
        info["filename"] = content_metadata.get("doc_name", "")
        info["chunk"] = content_metadata.get("chunk", "")
        info["text"] = content_metadata.get("text", "")
        filtered_text.append(content_metadata.get("text", ""))

        #append the data
        filtered_data.append(info)

    if get_text:
        similar_text = " ".join([text for text in filtered_text])
        print(similar_text)
        return similar_text

    print(filtered_data)

    return filtered_data


def QA_with_your_docs(user_question: str, text_list: List[str], chain_type: str = "stuff") -> str:
    """
    This is the main function to chat with the content you have

    @param
    user_question: question or context user wants to figure out
    text: list of similar texts
    chat_type: Type of chain run (stuff is cost effective)

    @return
    answers from the LLM
    """
    llm = OpenAI(temperature=0, openai_api_key = os.environ['OPENAI_API_KEY'])
    chain = load_qa_with_sources_chain(llm, chain_type = chain_type, verbose = False)

    all_docs = []
    for doc_content in text_list:
        metadata = {}
        doc_text = doc_content.get("text", "")
        metadata["id"] = doc_content.get("id", "")
        metadata["score"] = doc_content.get("score", "")
        metadata["filename"] = doc_content.get("filename", "")
        metadata["chunk"] = doc_content.get("chunk", "")
        chunk_name = doc_content.get("filename", "UNKNOWN")
        offset=", OFFSET="+str(doc_content.get("chunk","UNKNOWN"))
        metadata["source"] = chunk_name + offset
        all_docs.append(Document(page_content = doc_text, metadata = metadata))

    chain_response = chain.run(input_documents = all_docs, question = user_question )
    print(chain_response)

    return chain_response