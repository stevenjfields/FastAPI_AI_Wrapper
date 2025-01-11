from enum import Enum
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
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

class SpecialTokens(Enum):
    BEGIN_OF_TEXT = "<|begin_of_text|>"
    END_OF_TEXT = "<|end_of_text|>"
    START_HEADER_ID = "<|start_header_id|>"
    END_HEADER_ID = "<|end_header_id|>"
    END_OF_TURN = "<|eot_id|>"

    def __str__(self):
        return self.value


class Message(BaseModel):
    role: Role
    content: str

    def to_prompt(self):
        return f"{SpecialTokens.START_HEADER_ID}{self.role}{SpecialTokens.END_HEADER_ID}{self.content}{SpecialTokens.END_OF_TURN}"


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

    def generate(self, messages: List[Message], temperature: float = 1.0, max_tokens: int = 1024):
        if self.system_prompt:
            messages.insert(0, Message(role=Role.SYSTEM, content=self.system_prompt))
        input_str = " ".join([message.to_prompt() for message in messages])

        input_str += f"{SpecialTokens.START_HEADER_ID}{Role.ASSISTANT}{SpecialTokens.END_HEADER_ID}"

        with torch.no_grad():
            input_data = self.tokenizer(input_str, return_tensors="pt")
            input_data = {k: v.cuda() for k, v in input_data.items()}
            input_length = input_data["input_ids"].shape[1]
            output = self.model.generate(**input_data, max_new_tokens=max_tokens, temperature=temperature, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
            output = output[0][input_length:].tolist()
        return LlamaOutput(
            content=self.tokenizer.decode(output, skip_special_tokens=True),
            input_tokens=input_data["input_ids"].shape[1],
            output_tokens=len(output)
        )
