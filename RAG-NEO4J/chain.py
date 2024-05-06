from operator import itemgetter

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import ConfigurableField, RunnableParallel

from retrievers import (
    hypothetic_question_vectorstore,
    parent_vectorstore,
    summary_vectorstore,
    typical_rag,
)




template = """
Answer the question based only on the following CONTEXT:
{context}

Question: {question}

Please be truthful. Keep in mind, you will lose the job, if you answer out of CONTEXT questions.
If the responses are irrelevant to the question then respond by saying that I couldn't find a good response to your query in the database.
"""
prompt = ChatPromptTemplate.from_template(template)

retriever = typical_rag.as_retriever().configurable_alternatives(
    ConfigurableField(id="strategy"),
    default_key="typical_rag",
    parent_strategy=parent_vectorstore.as_retriever(),
    hypothetical_questions=hypothetic_question_vectorstore.as_retriever(),
    summary_strategy=summary_vectorstore.as_retriever(),
)

chain = (
    RunnableParallel(
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    question: str


chain = chain.with_types(input_type=Question)