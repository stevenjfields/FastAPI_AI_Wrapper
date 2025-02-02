from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

from ai.llama import Message

class MessageRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 1.0  # Default temperature of 1.0
    max_tokens: Optional[int] = 1024

class MessageResponse(BaseModel):
    messages: list[Message]
    input_tokens: int
    output_tokens: int
