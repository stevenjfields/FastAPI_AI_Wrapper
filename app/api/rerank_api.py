import fastapi
from fastapi import APIRouter
import numpy as np
import torch
from sklearn.preprocessing import normalize

from models.rerank import RerankRequest, RerankResponse, ScoredDocument, RankedDocument

router = APIRouter()

@router.post(
    "/rerank",
    response_model=RerankResponse,
    tags=["rerank"]
)
def rerank(request: fastapi.Request, rerank_request: RerankRequest):
    embedding_model, tokenizer, vector_linear = request.scope["embedding_models"].values()
    with torch.no_grad():
        input_data = tokenizer(rerank_request.query, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = embedding_model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        query_embedding = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        query_embedding = normalize(vector_linear(query_embedding).cpu().numpy())

    scored_documents = []
    for document in rerank_request.documents:
        scored_documents.append(
            ScoredDocument(
                text=document.text,
                embedding=document.embedding,
                score=np.dot(query_embedding, np.array(document.embedding).T)
            )
        )
    scored_documents.sort(key=lambda x: x.score, reverse=True)

    ranked_documents = []
    for i, scored_document in enumerate(scored_documents):
        ranked_documents.append(
            RankedDocument(
                text=scored_document.text,
                score=scored_document.score,
                rank=i + 1
            )
        )

    return RerankResponse(documents=ranked_documents)
