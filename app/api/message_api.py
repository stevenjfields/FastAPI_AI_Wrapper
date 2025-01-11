import fastapi
from fastapi import APIRouter
import torch

from models.message import Message, MessageRequest, MessageResponse, MessageRole

router = APIRouter()

@router.post(
    "/message",
    response_model=MessageResponse,
    tags=["message"]
)
def message(request: fastapi.Request, message_request: MessageRequest):
    llama_model = request.scope["llama_model"]
    input_str = "\n".join([f"{message.content}\n" for message in message_request.messages])
    output = llama_model.generate(input_str, message_request.temperature, message_request.max_tokens)
    messages = message_request.messages.copy()  
    messages.append(Message(
        role=MessageRole.ASSISTANT,
        content=output.content
    ))
    return MessageResponse(
        messages=messages,
        input_tokens=output.input_tokens,
        output_tokens=output.output_tokens
    )

