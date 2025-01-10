from pydantic import BaseModel

class Document(BaseModel):
    text: str
    embedding: list[float]

class RerankRequest(BaseModel):
    query: str
    documents: list[Document]

class ScoredDocument(BaseModel):
    text: str
    score: float

class RerankResponse(BaseModel):
    query: str
    documents: list[ScoredDocument]
