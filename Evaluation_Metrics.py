#pip install bert-score
#pip install langchain openai ragas
#pip install --upgrade openai (run only when you get this error: AttributeError: module 'openai' has no attribute 'OpenAI' )
#pip install pandas openpyxl

import ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, answer_relevancy, faithfulness, context_precision, context_recall, answer_similarity
import json
import pandas as pd
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv('.env')
openai_api_key = os.getenv("OPENAI_API_KEY")

#Structure of the json file
# {
#     "question": ["When was the first super bowl?", "Who won the most super bowls?"],
#     "answer": ["The first superbowl was held on Jan 15, 1967", "The most super bowls have been won by The New England Patriots"],
#     "contexts": [
#         ["The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"],
#         ["The Green Bay Packers...Green Bay, Wisconsin.", "The Packers compete...Football Conference"]
#     ],
#     "ground_truth": ["On Jan 15, 1967", "The New England Patriots"]
# }


# Path to the JSON file
file_path = 'output/qa_results_simpleneo4j_langchain_cosine_recursive_gpt4.json'
#file_path = 'output/qa_results_neo4j.json'

# Open the file and load the data
with open(file_path, 'r') as file:
    data = json.load(file)

print(data)


# Process existing contexts to split each context into sentences
updated_contexts = []
for context in data.get('contexts', []):
    # Split context into multiple sentences based on period (".")
    sentences = [sentence.strip() for sentence in context.split('.') if sentence.strip()]
    updated_contexts.append(sentences)


data['contexts'] = updated_contexts

####Only run for Neo4j###############################

data['ground_truth'] = data['ground_truth'][0]

#####################################################

# Create a new dictionary with the desired key order
updated_data = {
    'question': data.get('question', []),
    'answer': data.get('answer', []),
    'contexts': data.get('contexts', []),
    'ground_truth': data.get('ground_truth', [])
}
data=updated_data

dataset = Dataset.from_dict(data)


score = evaluate(dataset,metrics=[answer_similarity, 
                                   faithfulness, 
                                   context_precision,
                                   context_recall,
                                   answer_correctness,
                                   answer_relevancy])
df = score.to_pandas()

# Write the DataFrame to an Excel file
df.to_excel('output/scores/evaluation_scores.xlsx', index=False)







