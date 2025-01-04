import os
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from message_type import *
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


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")

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

    # TODO: implement the following logic
    if request.stream:
        # HTTPException(status_code=400, detail="stream not supported")
        generate = stream_chat_warpper(messages, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")
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


async def stream_chat_warpper(message, model_id: str):
    global model, tokenizer

    # 3. 初始化响应数据
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )

    # - 创建一个初始的 `ChatCompletionResponseStreamChoice` 对象，表示助手角色的响应。
    # - 创建一个 `ChatCompletionResponse` 对象，并将其转换为 JSON 格式后返回。
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    # 4. 初始化当前长度
    current_length = 0

    # 5. 处理模型的流式响应
    for new_response in qwen_model.stream_chat(message):
        # - 如果新响应的长度与当前长度相同，则跳过。
        if len(new_response) == current_length:
            continue
        print(new_response)
        # - 否则，提取新文本并更新当前长度。
        new_text = new_response[current_length:]
        current_length = len(new_response)

        # - 创建新的 `ChatCompletionResponseStreamChoice`
        #  和 `ChatCompletionResponse` 对象，并将其转换为 JSON 格式后返回。
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield "[DONE]"


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
