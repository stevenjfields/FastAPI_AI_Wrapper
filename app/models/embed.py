from pydantic import BaseModel

class EmbedRequest(BaseModel):
    text: str

class BatchedEmbedRequest(BaseModel):
    texts: list[str]

class Embedding(BaseModel):
    text: str
    token_count: int
    embedding: list[float]
    

class BatchedEmbedResponse(BaseModel):
    embeddings: list[Embedding]

