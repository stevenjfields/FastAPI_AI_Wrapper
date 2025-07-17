import fastapi
from fastapi import APIRouter
import torch
from sklearn.preprocessing import normalize

from models.embed import Embedding, BatchedEmbedResponse, EmbedRequest, BatchedEmbedRequest

router = APIRouter()


@router.post(
    "/embed",
    response_model=Embedding,
    tags=["embed"]
)
def embed(request: fastapi.Request, body: EmbedRequest):
    stella_embed = request.scope["stella_embed"]
    embedding = stella_embed.embed(body.text)
    return Embedding(
        text=body.text,
        token_count=embedding.token_count,
        embedding=embedding.embedding
    )

@router.post(
    "/embed_batch",
    response_model=BatchedEmbedResponse,
    tags=["embed"]
)
def embed_batch(request: fastapi.Request, body: BatchedEmbedRequest):
    stella_embed = request.scope["stella_embed"]
    embeddings = stella_embed.embed_batch(body.texts)
    embeddings = [
        Embedding(
            text=body[i],
            token_count=embeddings[i].token_count,
            embedding=embeddings[i].embedding
        )
        for i in range(len(body))
    ]
    return BatchedEmbedResponse(embeddings=embeddings)
