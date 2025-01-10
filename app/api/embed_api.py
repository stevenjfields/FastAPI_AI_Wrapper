import fastapi
from fastapi import APIRouter
import torch
from sklearn.preprocessing import normalize

from models.embed import Embedding, EmbedResponse


router = APIRouter()

@router.post("/embed")
def embed(request: fastapi.Request, body: list[str]) -> EmbedResponse:
    embedding_model = request.scope["embedding_model"]
    tokenizer = request.scope["tokenizer"]
    vector_linear = request.scope["vector_linear"]
    with torch.no_grad():
        input_data = tokenizer(body, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = embedding_model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        text_embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        text_embeddings = normalize(vector_linear(text_embeddings).cpu().numpy())

    embeddings = []
    for i in range(len(body)):
        embeddings.append(
            Embedding(
                text=body[i],
                token_count=input_data["input_ids"][i].nonzero().shape[0],
                embedding=text_embeddings[i].tolist()
            )
        )
    return EmbedResponse(embeddings=embeddings)
