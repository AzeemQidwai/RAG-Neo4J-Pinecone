#pip install bert-score
#pip install langchain openai weaviate-client ragas
#pip install --upgrade openai (run only when you get this error: AttributeError: module 'openai' has no attribute 'OpenAI' )

import ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_similarity
import json
import pandas as pd

# Path to the JSON file
file_path = 'extra/test2.json'

# Open the file and load the data
with open(file_path, 'r') as file:
    data = json.load(file)

print(data)

dataset = Dataset.from_dict(data)

score0 = evaluate(dataset,metrics=[faithfulness])
score0.to_pandas()


score1 = evaluate(dataset,metrics=[context_precision])
score1.to_pandas()


score2 = evaluate(dataset,metrics=[context_recall])
score2.to_pandas()


score3 = evaluate(dataset,metrics=[answer_similarity])
score3.to_pandas()





