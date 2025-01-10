from enum import Enum
from pydantic import BaseModel

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: MessageRole
    content: str

class MessageRequest(BaseModel):
    messages: list[Message]

class MessageResponse(BaseModel):
    messages: list[Message]
    input_tokens: int
    output_tokens: int
