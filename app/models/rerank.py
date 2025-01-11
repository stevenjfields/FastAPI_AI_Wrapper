from pydantic import BaseModel

class Document(BaseModel):
    text: str

class EmbeddingDocument(Document):
    embedding: list[float]

class RerankRequest(BaseModel):
    query: str
    documents: list[str]

class EmbeddingRerankRequest(BaseModel):
    query: str
    documents: list[EmbeddingDocument]

class ScoredDocument(BaseModel):
    text: str
    score: float

class RerankResponse(BaseModel):
    query: str
    documents: list[ScoredDocument]
