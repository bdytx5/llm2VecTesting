

import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from llm2vec import LLM2Vec

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
import torch
from llm2vec import LLM2Vec
from sentence_transformers import SentenceTransformer
from rag import Rag, Document


if __name__ == "__main__":
    # Load the chunked WikiSource data
    with open("chunked_wikisource_data.json", 'r', encoding='utf-8') as file:
        chunked_data = json.load(file)

    # Using SentenceTransformer model
    model_name = "BAAI/bge-small-en-v1.5"
    model = SentenceTransformer(model_name)
    rag_model = Rag(model_name=model_name, model=model)

    # Create documents
    documents = []
    for item in tqdm(chunked_data, desc="Generating documents"):
        document = Document(text=item['title'] + " " + item['text'], metadata=item)
        documents.append(document)

    # Generate and store embeddings individually
    for document in tqdm(documents, desc="Storing embeddings for documents"):
        rag_model.store_embedding(document)

    # Save embeddings to file under the directory corresponding to the model name
    base_path = "./embeddings"
    rag_model.save_embeddings(base_path)



    model_name = "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse"

    tokenizer = AutoTokenizer.from_pretrained(
        "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
    )
    config = AutoConfig.from_pretrained(
        "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        trust_remote_code=True,
        
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Loading MNTP (Masked Next Token Prediction) model.
    model = PeftModel.from_pretrained(
        model,
        "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-unsup-simcse",
    )

    # Wrapper for encoding and pooling operations
    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=2048)

    rag_model = Rag(model_name=model_name, model=l2v)

    # Create documents
    documents = []
    for item in tqdm(chunked_data, desc="Generating documents"):
        document = Document(text=item['title'] + " " + item['text'], metadata=item)
        documents.append(document)

    # Generate and store embeddings individually
    for document in tqdm(documents, desc="Storing embeddings for documents"):
        rag_model.store_embedding(document)

    # Save embeddings to file under the directory corresponding to the model name
    base_path = "./embeddings"
    rag_model.save_embeddings(base_path)
