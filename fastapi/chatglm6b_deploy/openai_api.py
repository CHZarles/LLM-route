# coding=utf-8
# Implements API for ChatGLM2-6B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.


import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from transformers import AutoModel, AutoTokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# https://platform.openai.com/docs/api-reference/models/object
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


# https://platform.openai.com/docs/api-reference/models/list
class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


# https://platform.openai.com/docs/api-reference/chat/create
# 这个格式对应的是 https://platform.openai.com/docs/api-reference/chat/completions 里面的 Request body 里的 messages 类型


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


# https://platform.openai.com/docs/api-reference/chat/streaming
# 这个应该是对应 stream 模式的 “The chat completion chunk object” 里面的 delta 类型
class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


# 这个格式对应的是 https://platform.openai.com/docs/api-reference/chat/completions 整个请求
"""
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "developer",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
"""


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


# 对应 https://platform.openai.com/docs/api-reference/chat/object
# The chat completion object 里面的 choices 类型
"""
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hi there! How can I assist you today?",
        "refusal": null
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
"""


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


# 对应 https://platform.openai.com/docs/api-reference/chat/streaming
# The chat completion chunk object 里面的 choices 类型
"""
{
    "id": "chatcmpl-123",
    ...
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": ""
            },
            "logprobs": null,
            "finish_reason": null
        }
    ]
}
"""


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


# 这个结构对应 https://platform.openai.com/docs/api-reference/chat/object
# 整个 The chat completion object 对象
"""
{
  ...
  "created": 1728933352,
  "model": "gpt-4o-2024-08-06",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hi there! How can I assist you today?",
        "refusal": null
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  ...
}
"""


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")

    """

    1. `prev_messages = request.messages[:-1]`：从请求的消息列表中获取除最后一条消息之外的所有消息，赋值给 `prev_messages` 变量。
    2. `if len(prev_messages) > 0 and prev_messages[0].role == "system":`：检查 `prev_messages` 是否非空，并且第一条消息的角色是否为 "system"。
    3. `query = prev_messages.pop(0).content + query`：如果条件满足，将第一条系统消息的内容与当前查询内容拼接起来，并更新 `query` 变量。

    """
    query = request.messages[-1].content
    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query
    """
    `history` 列表将包含所有成对的用户消息和助手回复，便于后续处理或分析。

    """
    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if (
                prev_messages[i].role == "user"
                and prev_messages[i + 1].role == "assistant"
            ):
                history.append([prev_messages[i].content, prev_messages[i + 1].content])

    if request.stream:
        # 主意 这里返回的是一个协程对象
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop",
    )

    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )


async def predict(query: str, history: List[List[str]], model_id: str):
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
    for new_response, _ in model.stream_chat(tokenizer, query, history):
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


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "./model/chatglm-6b-int4", trust_remote_code=True
    )
    model = (
        AutoModel.from_pretrained("./model/chatglm-6b-int4", trust_remote_code=True)
        .half()
        .cuda()
    )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "THUDM/chatglm2-6b-int4", trust_remote_code=True
    # )
    # model = AutoModel.from_pretrained(
    #     "THUDM/chatglm2-6b-in4", trust_remote_code=True
    # ).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
    model.eval()

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
