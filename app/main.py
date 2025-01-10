import os

import fastapi
import uvicorn
from fastapi.middleware import Middleware
from transformers import AutoModel, AutoTokenizer

from api.embed_api import router as embed_api

model_dir = "./weights"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

vector_dim = 256
vector_linear_directory = f"2_Dense_{vector_dim}"
embedding_model = AutoModel.from_pretrained("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_400M_v5", trust_remote_code=True)

app = fastapi.FastAPI()

def configure_routing(app: fastapi.FastAPI):
    app.include_router(embed_api)

@app.middleware("http")
async def add_models(request: fastapi.Request, call_next):
    request.scope["embedding_model"] = embedding_model
    request.scope["tokenizer"] = tokenizer
    response = await call_next(request)
    return response

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    configure_routing(app)
    uvicorn.run(app, host="0.0.0.0", port=8000)
