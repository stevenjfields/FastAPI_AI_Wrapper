import fastapi
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json

from models.message import Message, MessageRequest, MessageResponse, StreamingContentResponse, StreamingMetadataResponse, StreamingStartResponse, StreamingEndResponse
from ai.llama import Role

router = APIRouter()

@router.post(
    "/message",
    response_model=MessageResponse,
    tags=["message"]
)
def message(request: fastapi.Request, message_request: MessageRequest):
    llama_model = request.scope["llama_model"]
    output = llama_model.generate(message_request.messages, message_request.temperature, message_request.max_tokens)
    messages = message_request.messages.copy()  
    messages.append(Message(
        role=Role.ASSISTANT,
        content=output.content
    ))
    return MessageResponse(
        messages=messages,
        input_tokens=output.input_tokens,
        output_tokens=output.output_tokens
    )

@router.post(
    "/message/stream",
    tags=["message"]
)
async def message_stream(request: fastapi.Request, message_request: MessageRequest):
    llama_model = request.scope["llama_model"]
    
    async def generate_stream():
        input_tokens = 0
        output_tokens = 0
        
        # Send start message
        start_response = StreamingStartResponse()
        yield f"event: start\ndata: {start_response.model_dump_json()}\n\n"
        
        # Get input token count
        chat_messages = [message.to_dict() for message in message_request.messages]
        if llama_model.system_prompt:
            chat_messages.insert(0, {"role": "system", "content": llama_model.system_prompt})
        
        input_str = llama_model.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_data = llama_model.tokenizer(input_str, return_tensors="pt")
        input_tokens = input_data["input_ids"].shape[1]
        
        # Stream content tokens
        async for token in llama_model.generate_stream(
            message_request.messages, 
            message_request.temperature, 
            message_request.max_tokens
        ):
            output_tokens += 1
            
            # Send content token as SSE with event name
            content_response = StreamingContentResponse(content=token)
            yield f"event: content\ndata: {content_response.model_dump_json()}\n\n"
        
        # Send metadata as final SSE event with event name
        metadata_response = StreamingMetadataResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        yield f"event: metadata\ndata: {metadata_response.model_dump_json()}\n\n"
        
        # Send end message
        end_response = StreamingEndResponse()
        yield f"event: end\ndata: {end_response.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

