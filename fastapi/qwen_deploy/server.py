import os
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from message_type import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    CompletionTokensDetails,
    DeltaMessage,
    ModelList,
    PromptTokensDetails,
    Usage,
)
from qwen_api import Qwen

qwen_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qwen_model
    qwen_model = Qwen(model_path="./Qwen/Qwen1.5-7B-Chat", device="cuda")
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
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")

    # TODO: implement the following logic
    if request.stream:
        HTTPException(status_code=400, detail="stream not supported")

    """
    # construct messages template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    """
    messages = []
    for message in request.messages:
        messages.append({"role": message.role, "content": message.content})

    # call qwen generate
    response = qwen_model.generate_response(messages)
    # make
    print(response)
    content = response["generated_text"]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=content),
        finish_reason="stop",
    )
    usage_data = Usage(
        completion_tokens=response["completion_tokens"],
        prompt_tokens=response["prompt_tokens"],
        total_tokens=response["prompt_tokens"] + response["completion_tokens"],
    )
    print(usage_data)
    res = ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        usage=usage_data,
        object="chat.completion",
    )
    return res


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    input: JSON String of {"query": "", "history": []}
    output: JSON String of {"response": "", "history": [], "status": 200}
        status 200 stand for response ended, else not
    """
    # TODO: implement this
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            print("receive json_request from request is\n", type(json_request))
            query = json_request["query"]
            history = json_request["history"]
            print("receive history from request is\n", history)
            await websocket.send_text(f"At {time.time()} Message text was: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
