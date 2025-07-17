import os

import fastapi
import uvicorn
from fastapi.middleware import Middleware
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from ai.llama import Llama, LlamaModel
from ai.stella_embed import StellaEmbed
from api import embed_api, rerank_api, message_api

model_dir = "./weights"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

vector_dim = 256
stella_embed = StellaEmbed(vector_dim)

llama_model = Llama(
    LlamaModel.LLaMA_3_2_3B,
    device="cuda:0",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
)

app = fastapi.FastAPI()

def configure_routing(app: fastapi.FastAPI):
    app.include_router(embed_api.router)
    app.include_router(rerank_api.router)
    app.include_router(message_api.router)

@app.middleware("http")
async def add_models(request: fastapi.Request, call_next):
    request.scope["stella_embed"] = stella_embed
    request.scope["llama_model"] = llama_model
    response = await call_next(request)
    return response

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    configure_routing(app)
    uvicorn.run(app, host="0.0.0.0", port=8000)
