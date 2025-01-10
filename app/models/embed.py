from pydantic import BaseModel

class Embedding(BaseModel):
    text: str
    token_count: int
    embedding: list[float]
    
class EmbedResponse(BaseModel):
    embeddings: list[Embedding]

