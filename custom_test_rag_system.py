
import json
import os
import torch
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, AutoConfig, AutoModel
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from llm2vec import LLM2Vec
from rag import Rag, Document
# Initialize Weights & Biases
wandb.init(project="rag_retrieval_testing", entity='byyoung3')

# List of index directories and corresponding nicknames
index_dirs = ["./embeddings/BAAI_bge-small-en-v1.5", "./embeddings/McGill-NLP_LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse"]
index_nicknames = ["BAAI_bge-small-en-v1.5", "McGill-NLP_LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse"]

# Load the QA dataset
qa_dataset_path = "./qa_dataset.json"
with open(qa_dataset_path, 'r', encoding='utf-8') as file:
    qa_dataset = json.load(file)


# Function to test RAG system's ability to retrieve the correct item and log metrics to W&B
def test_rag_system(rag_model, index_nickname, qa_dataset):
    # Load stored embeddings from file
    rag_model.load_embeddings(f"./embeddings/{index_nickname}/embeddings.json")

    total_queries = len(qa_dataset)
    pass_at_1 = 0
    pass_at_3 = 0
    pass_at_5 = 0

    for qa_pair in tqdm(qa_dataset, desc=f"Testing RAG system for {index_nickname}"):
        question = qa_pair["factoid_question"]
        ground_truth_id = qa_pair["custom_id"]

        # Query the index
        retrieved_docs = rag_model.retrieve_texts(question, k=5)

        # Check pass@1, pass@3, pass@5
        for k in range(1, 6):
            if k <= len(retrieved_docs):
                retrieved_id = retrieved_docs[k-1][0]
                if retrieved_id == ground_truth_id:
                    if k == 1:
                        pass_at_1 += 1
                    if k <= 3:
                        pass_at_3 += 1
                    if k <= 5:
                        pass_at_5 += 1
                    break

    metrics = {
        "pass@1": (pass_at_1 / total_queries) * 100,
        "pass@3": (pass_at_3 / total_queries) * 100,
        "pass@5": (pass_at_5 / total_queries) * 100
    }

    # Log metrics to W&B
    wandb.log({f"{index_nickname}_pass@1": metrics["pass@1"],
               f"{index_nickname}_pass@3": metrics["pass@3"],
               f"{index_nickname}_pass@5": metrics["pass@5"]})

    return metrics

# Initialize RAG models for both SentenceTransformer and LLM2Vec models
models = {
    "BAAI_bge-small-en-v1.5": SentenceTransformer("BAAI/bge-small-en-v1.5"),
    "McGill-NLP_LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse": LLM2Vec(
        PeftModel.from_pretrained(
            AutoModel.from_pretrained(
                "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
                config=AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True),
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            ),
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse"
        ),
        AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"),
        pooling_mode="mean",
        max_length=2048
    )
}

results = {}
for index_nickname, model in models.items():
    model.eval()
    rag_model = Rag(model_name=index_nickname, model=model)

    # Test the RAG system
    results[index_nickname] = test_rag_system(rag_model, index_nickname, qa_dataset)

# Log combined results to W&B
wandb.log({
    "BAAI_bge-small-en-v1.5_pass@1": results["BAAI_bge-small-en-v1.5"]["pass@1"],
    "BAAI_bge-small-en-v1.5_pass@3": results["BAAI_bge-small-en-v1.5"]["pass@3"],
    "BAAI_bge-small-en-v1.5_pass@5": results["BAAI_bge-small-en-v1.5"]["pass@5"],
    "McGill-NLP_LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse_pass@1": results["McGill-NLP_LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse"]["pass@1"],
    "McGill-NLP_LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse_pass@3": results["McGill-NLP_LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse"]["pass@3"],
    "McGill-NLP_LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse_pass@5": results["McGill-NLP_LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse"]["pass@5"]
})

# Print the results
for index_nickname, metrics in results.items():
    print(f"Results for {index_nickname}:")
    print(f"pass@1: {metrics['pass@1']:.2f}%")
    print(f"pass@3: {metrics['pass@3']:.2f}%")
    print(f"pass@5: {metrics['pass@5']:.2f}%")
