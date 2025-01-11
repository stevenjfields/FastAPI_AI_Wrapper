from typing import List
import numpy as np
from pydantic import BaseModel
import torch
from transformers import AutoModel, AutoTokenizer
import os
from sklearn.preprocessing import normalize

from models.rerank import ScoredDocument

model_dir = "./weights"

class EmbeddingResult(BaseModel):
    embedding: list[float]
    token_count: int

class StellaEmbed:
    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim
        vector_linear_directory = f"2_Dense_{vector_dim}"
        self.embedding_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.vector_linear = torch.nn.Linear(in_features=self.embedding_model.config.hidden_size, out_features=vector_dim)
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
        }
        self.vector_linear.load_state_dict(vector_linear_dict)
        self.vector_linear.cuda()


    def embed(self, input_str: str):
        with torch.no_grad():
            input_data = self.embedding_tokenizer(input_str, padding="longest", truncation=True, max_length=8192, return_tensors="pt")
            input_data = {k: v.cuda() for k, v in input_data.items()}
            attention_mask = input_data["attention_mask"]
            last_hidden_state = self.embedding_model(**input_data)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            text_embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            text_embeddings = normalize(self.vector_linear(text_embeddings).cpu().numpy())

        return EmbeddingResult(
            embedding=text_embeddings[0].tolist(),
            token_count=input_data["input_ids"][0].nonzero().shape[0]
        )
    
    def embed_batch(self, text: List[str]):
        with torch.no_grad():
            input_data = self.embedding_tokenizer(text, padding="longest", truncation=True, max_length=8192, return_tensors="pt")
            input_data = {k: v.cuda() for k, v in input_data.items()}
            attention_mask = input_data["attention_mask"]
            last_hidden_state = self.embedding_model(**input_data)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            text_embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            text_embeddings = normalize(self.vector_linear(text_embeddings).cpu().numpy())

        return [
            EmbeddingResult(
                embedding=text_embeddings[i].tolist(),
                token_count=input_data["input_ids"][i].nonzero().shape[0]
            )
            for i in range(len(text))
        ]

    def rerank_embeddings(self, query: str, documents: List[str]):
        query_embedding = self.embed(query).embedding
        
        scored_documents = []
        for document in documents:
            scored_documents.append(
                ScoredDocument(
                    text=document.text,
                    score=np.dot(query_embedding, np.array(document.embedding).T)
                )
            )
        scored_documents.sort(key=lambda x: x.score, reverse=True)
        return scored_documents
    
    def rerank_text(self, query: str, documents: List[str]):
        query_embedding = self.embed(query).embedding
        documents_embeddings = self.embed_batch(documents)
        scored_documents = []
        for i, document in enumerate(documents):
            scored_documents.append(
                ScoredDocument(
                    text=document,
                    score=np.dot(query_embedding, np.array(documents_embeddings[i].embedding).T)
                )
            )
        scored_documents.sort(key=lambda x: x.score, reverse=True)
        return scored_documents
