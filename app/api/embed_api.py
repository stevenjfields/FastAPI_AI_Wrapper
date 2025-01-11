import fastapi
from fastapi import APIRouter
import torch
from sklearn.preprocessing import normalize

from models.embed import Embedding, EmbedResponse

router = APIRouter()

@router.post(
    "/embed",
    response_model=EmbedResponse,
    tags=["embed"]
)
def embed(request: fastapi.Request, body: list[str]):
    stella_embed = request.scope["stella_embed"]
    embeddings = stella_embed.embed_batch(body)
    embeddings = [
        Embedding(
            text=body[i],
            token_count=embeddings[i].token_count,
            embedding=embeddings[i].embedding
        )
        for i in range(len(body))
    ]
    return EmbedResponse(embeddings=embeddings)
