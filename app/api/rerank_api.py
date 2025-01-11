import fastapi
from fastapi import APIRouter
import numpy as np
import torch
from sklearn.preprocessing import normalize

from models.rerank import RerankRequest, RerankResponse, ScoredDocument, EmbeddingRerankRequest

router = APIRouter()

@router.post(
    "/rerank/text",
    response_model=RerankResponse,
    tags=["rerank"]
)
def rerank_text(request: fastapi.Request, rerank_request: RerankRequest):
    stella_embed = request.scope["stella_embed"]
    scored_documents = stella_embed.rerank_text(rerank_request.query, rerank_request.documents)
    return RerankResponse(query=rerank_request.query, documents=scored_documents)

@router.post(
    "/rerank/embeddings",
    response_model=RerankResponse,
    tags=["rerank"]
)
def rerank_embeddings(request: fastapi.Request, rerank_request: EmbeddingRerankRequest):
    stella_embed = request.scope["stella_embed"]
    scored_documents = stella_embed.rerank_embeddings(rerank_request.query, rerank_request.documents)
    return RerankResponse(query=rerank_request.query, documents=scored_documents)