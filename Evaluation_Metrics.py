#pip install bert-score
#pip install langchain openai ragas
#pip install --upgrade openai (run only when you get this error: AttributeError: module 'openai' has no attribute 'OpenAI' )

import ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_similarity
import json
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

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



openai_api_key = os.getenv("OPENAI_API_KEY")
# Path to the JSON file
file_path = 'output/qa_results_pc.json'
#file_path = 'output/qa_results_neo4j.json'

# Open the file and load the data
with open(file_path, 'r') as file:
    data = json.load(file)

print(data)

dataset = Dataset.from_dict(data)

score = evaluate(dataset)
score.to_pandas()


score1 = evaluate(dataset,metrics=[answer_similarity, 
                                   faithfulness, 
                                   context_precision,
                                   context_recall])
score1.to_pandas()






