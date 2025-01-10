from pydantic import BaseModel
import torch
class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list[float]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(embedding=tensor.tolist())
