import os

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from message_type import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    CompletionTokensDetails,
    DeltaMessage,
    ModelList,
    PromptTokensDetails,
)

qwen_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

# reference:
# https://fastapi.tiangolo.com/zh/tutorial/cors/#corsmiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


"""

 `response_model=ModelList`:
 This specifies that the response from this endpoint should be validated and serialized using the `ModelList` Pydantic model.
 This ensures that the response data conforms to the structure defined by `ModelList`.

"""


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelCard(id="qwen/qwen1.5-7b-chat")])


@app.get("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

