import asyncio
from enum import Enum
from threading import Thread
from typing import List, Optional, AsyncGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import TextIteratorStreamer
from pydantic import BaseModel
from transformers import BitsAndBytesConfig
import torch

class LlamaModel(Enum):
    LLaMA_3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"
    LLaMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"

class LlamaOutput(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    def __str__(self):
        return self.value

class Message(BaseModel):
    role: Role
    content: str

    def to_dict(self):
        return {"role": self.role.value, "content": self.content}


class Llama:
    def __init__(self, model: LlamaModel, device: str = "cuda", quantization_config: Optional[BitsAndBytesConfig] = None, system_prompt: Optional[str] = None):
        self.model = model.value
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model,
            device_map=device,
            torch_dtype=torch.float16,
            quantization_config=quantization_config
        )
        self.system_prompt = system_prompt

    def generate(self, messages: List[Message], temperature: float = 1.0, max_tokens: int = 1024, streamer: TextIteratorStreamer = None):
        chat_messages = [message.to_dict() for message in messages]
        
        if self.system_prompt:
            chat_messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        input_str = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        with torch.no_grad():
            input_data = self.tokenizer(input_str, return_tensors="pt")
            input_data = {k: v.cuda() for k, v in input_data.items()}
            input_length = input_data["input_ids"].shape[1]
            output = self.model.generate(**input_data, max_new_tokens=max_tokens, temperature=temperature, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, streamer=streamer)
            output = output[0][input_length:].tolist()
            input_tokens = input_data["input_ids"].shape[1]
            output_tokens = len(output)
            del input_data
            torch.cuda.empty_cache()
        return LlamaOutput(
            content=self.tokenizer.decode(output, skip_special_tokens=True),
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

    async def generate_stream(self, messages: List[Message], temperature: float = 1.0, max_tokens: int = 1024) -> AsyncGenerator[str, None]:
        chat_messages = [message.to_dict() for message in messages]
        
        if self.system_prompt:
            chat_messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        input_str = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        with torch.no_grad():
            input_data = self.tokenizer(input_str, return_tensors="pt")
            input_data = {k: v.cuda() for k, v in input_data.items()}
            input_length = input_data["input_ids"].shape[1]
            
            # Create streamer for real-time token streaming
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Run generation in a separate thread
            generation_kwargs = dict(
                **input_data,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer
            )
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream tokens as they're generated
            for text in streamer:
                yield text
                
            thread.join()
            
            # Clean up
            del input_data
            torch.cuda.empty_cache()
