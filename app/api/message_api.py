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
    llama_model, tokenizer = request.scope["llama_models"].values()
    input_str = "\n".join([f"{message.content}\n" for message in message_request.messages])
    with torch.no_grad():
        input_data = tokenizer(input_str, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        input_length = input_data["input_ids"].shape[1]
        output = llama_model.generate(**input_data, max_new_tokens=1024)
        output = output[0][input_length:].tolist()
    messages = message_request.messages.copy()  
    messages.append(Message(
        role=MessageRole.ASSISTANT,
        content=tokenizer.decode(output, skip_special_tokens=True)
    ))
    return MessageResponse(
        messages=messages,
        input_tokens=input_data["input_ids"].shape[1],
        output_tokens=len(output)
    )

