import os

import fastapi
import uvicorn
from fastapi.middleware import Middleware
from transformers import AutoModel, AutoTokenizer
import torch
from api.embed_api import router as embed_api

model_dir = "./weights"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Defined here and added to request scope to keep the model and tokenizer in memory
vector_dim = 256
vector_linear_directory = f"2_Dense_{vector_dim}"
embedding_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
vector_linear = torch.nn.Linear(in_features=embedding_model.config.hidden_size, out_features=vector_dim)
vector_linear_dict = {
    k.replace("linear.", ""): v for k, v in
    torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
}
vector_linear.load_state_dict(vector_linear_dict)
vector_linear.cuda()

app = fastapi.FastAPI()

def configure_routing(app: fastapi.FastAPI):
    app.include_router(embed_api)

@app.middleware("http")
async def add_models(request: fastapi.Request, call_next):
    request.scope["embedding_model"] = embedding_model
    request.scope["tokenizer"] = tokenizer
    request.scope["vector_linear"] = vector_linear
    response = await call_next(request)
    return response

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    configure_routing(app)
    uvicorn.run(app, host="0.0.0.0", port=8000)
