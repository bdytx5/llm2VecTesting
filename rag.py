import json
import os
import torch

import wandb
from sentence_transformers import SentenceTransformer



class Document:
    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata
        self.embedding = None

class Rag:
    def __init__(self, model_name: str, model: SentenceTransformer, device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.embeddings = {}

    def generate_embedding(self, text: str) -> torch.Tensor:
        embedding = self.model.encode([text], convert_to_tensor=True, device=self.device)
        return embedding.squeeze()

    def store_embedding(self, document: Document):
        embedding = self.generate_embedding(document.text)
        document.embedding = embedding
        self.embeddings[document.metadata['custom_id']] = embedding

    def save_embeddings(self, base_path: str):
        model_dir = os.path.join(base_path, self.model_name.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, "embeddings.json")
        with open(file_path, 'w') as file:
            json.dump({key: embed.tolist() for key, embed in self.embeddings.items()}, file)

    def load_embeddings(self, file_path: str):
        with open(file_path, 'r') as file:
            embeddings = json.load(file)
        self.embeddings = {key: torch.tensor(embed) for key, embed in embeddings.items()}

    def retrieve_texts(self, query: str, k: int = 5) -> list:
        query_embedding = self.generate_embedding(query)
        query_embedding_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
        stored_embeddings = torch.stack(list(self.embeddings.values()))
        stored_embeddings_norm = torch.nn.functional.normalize(stored_embeddings, p=2, dim=1)
        
        cos_sim = torch.mm(query_embedding_norm.unsqueeze(0), stored_embeddings_norm.transpose(0, 1)).squeeze()
        top_k_indices = cos_sim.topk(k).indices
        results = [(list(self.embeddings.keys())[idx], cos_sim[idx].item()) for idx in top_k_indices]
        return results
