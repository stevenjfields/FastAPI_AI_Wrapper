import fastapi
from fastapi import APIRouter

from models.embed import EmbedRequest, EmbedResponse

router = APIRouter()

@router.post("/embed")
def embed(request: fastapi.Request, body: EmbedRequest):
    embedding_model = request.scope["embedding_model"]
    tokenizer = request.scope["tokenizer"]

    tokens = tokenizer(body.text, return_tensors="pt", device="cuda")
    embeddings = embedding_model(**tokens).last_hidden_state.mean(dim=1)

    return EmbedResponse(embedding=embeddings.tolist())

