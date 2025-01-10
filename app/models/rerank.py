from pydantic import BaseModel

class Document(BaseModel):
    text: str
    embedding: list[float]

class RerankRequest(BaseModel):
    query: str
    documents: list[Document]

class ScoredDocument(Document):
    score: float

class RankedDocument(BaseModel):
    text: str
    score: float
    rank: int

class RerankResponse(BaseModel):
    documents: list[RankedDocument]