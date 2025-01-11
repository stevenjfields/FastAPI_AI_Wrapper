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

class Llama:
    def __init__(self, model: LlamaModel, device: str = "cuda", quantization_config: Optional[BitsAndBytesConfig] = None):
        self.model = model.value
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model,
            device_map=device,
            torch_dtype=torch.float16,
            quantization_config=quantization_config
        )

    def generate(self, input_str: str, temperature: float = 1.0, max_tokens: int = 1024):
        with torch.no_grad():
            input_str = f"{input_str}\n"
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
