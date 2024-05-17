# import json
# import torch
# import random
# from tqdm import tqdm
# import ollama

# # Provided ModelInference class
# class ModelInference:
#     def __init__(self, model_name):
#         self.model_name = model_name
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     def generate_one_completion(self, prompt):
#         try:
#             response = ollama.chat(model=self.model_name, messages=[
#                 {'role': 'user', 'content': prompt},
#             ])
#             completion = response['message']['content']
#             return {"completion": completion}
#         except Exception as e:
#             print(f"Error with model {self.model_name}: {str(e)}")
#             return {"completion": f"Failed to generate response with {self.model_name}"}

# # List of models to iterate over
# models = [
#     'mistral:instruct',
#     'gemma:instruct',
#     'phi3:instruct',
# ]

# QA_generation_prompt = """
# Your task is to write a factoid question and an answer given a context.
# Your factoid question should be answerable with a specific, concise piece of factual information from the context.
# Your factoid question should be formulated in the same style as questions users could ask in a search engine.
# This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

# Provide your answer as follows:

# Output:::
# Factoid question: (your factoid question)
# Answer: (your answer to the factoid question)

# Now here is the context.

# Context: {context}\n
# Output:::
# """

# # Load the chunked WikiSource data
# with open("chunked_wikisource_data.json", 'r', encoding='utf-8') as file:
#     chunked_data = json.load(file)

# # Initialize the model inference instances
# model_inferences = {model_name: ModelInference(model_name) for model_name in models}

# qa_dataset = []

# # Randomly select a portion of the dataset
# portion_size = 100  # Specify the size of the random portion
# random_portion = random.sample(chunked_data, min(portion_size, len(chunked_data)))

# # Generate QA pairs for each chunk of text in the random portion
# for chunk in tqdm(random_portion, desc="Generating QA pairs"):
#     context = chunk['title'] + " " + chunk["text"]
#     for model_name, model_inference in model_inferences.items():
#         prompt = QA_generation_prompt.format(context=context)
#         completion_result = model_inference.generate_one_completion(prompt)
#         completion = completion_result["completion"]
        
#         # Extract the question and answer from the completion
#         if "Factoid question:" in completion and "Answer:" in completion:
#             factoid_question = completion.split("Factoid question:")[1].split("Answer:")[0].strip()
#             answer = completion.split("Answer:")[1].strip()
            
#             qa_pair = {
#                 "custom_id": chunk["custom_id"],
#                 "context": context,
#                 "model": model_name,
#                 "factoid_question": factoid_question,
#                 "answer": answer
#             }
#             qa_dataset.append(qa_pair)
#             break  # Only keep the first successful model's response to avoid duplicates

# # Save the QA dataset to a JSON file
# output_file_path = "./qa_dataset.json"
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     json.dump(qa_dataset, file, ensure_ascii=False, indent=4)

# print(f"QA dataset has been saved to {output_file_path}")



import json
import torch
import random
from tqdm import tqdm
import ollama

# Provided ModelInference class
class ModelInference:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def generate_one_completion(self, prompt):
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt},
            ])
            completion = response['message']['content']
            return {"completion": completion}
        except Exception as e:
            print(f"Error with model {self.model_name}: {str(e)}")
            return {"completion": f"Failed to generate response with {self.model_name}"}

# List of models to iterate over
models = [
    'mistral:instruct',
    'gemma:instruct',
    'phi3:instruct',
]

QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::
"""

# Load the chunked WikiSource data
with open("chunked_wikisource_data.json", 'r', encoding='utf-8') as file:
    chunked_data = json.load(file)

# Initialize the model inference instances
model_inferences = {model_name: ModelInference(model_name) for model_name in models}

# List to store the QA pairs from all models
qa_dataset = []

# Randomly select a portion of the dataset
portion_size = 100  # Specify the size of the random portion

for model_name, model_inference in model_inferences.items():
    print(f"Generating QA pairs for model {model_name}")
    
    # Generate unique random portions for each model
    random_portion = random.sample(chunked_data, min(portion_size, len(chunked_data)))
    
    for chunk in tqdm(random_portion, desc=f"Generating QA pairs for {model_name}"):
        context = chunk['title'] + " " + chunk["text"]
        prompt = QA_generation_prompt.format(context=context)
        completion_result = model_inference.generate_one_completion(prompt)
        completion = completion_result["completion"]
        
        # Extract the question and answer from the completion
        if "Factoid question:" in completion and "Answer:" in completion:
            factoid_question = completion.split("Factoid question:")[1].split("Answer:")[0].strip()
            answer = completion.split("Answer:")[1].strip()
            
            qa_pair = {
                "custom_id": chunk["custom_id"],
                "context": context,
                "model": model_name,
                "factoid_question": factoid_question,
                "answer": answer
            }
            qa_dataset.append(qa_pair)

# Save the QA dataset to a single JSON file
output_file_path = "./qa_dataset.json"
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(qa_dataset, file, ensure_ascii=False, indent=4)

print(f"QA dataset has been saved to {output_file_path}")
