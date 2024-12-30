# coding=utf-8
# Implements API for ChatGLM2-6B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.


from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from chatglm_api import ChatGLMModel
from message_type import (  # ChatCompletionResponseStream,; ChatCompletionResponseStreamChoice,; DeltaMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    Usage,
)


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


model = ChatGLMModel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        HTTPException(status_code=400, detail="stream not supported")
        # 主意 这里返回的是一个协程对象
        # generate = predict(query, history, request.model)
        # return EventSourceResponse(generate, media_type="text/event-stream")

    response = model.generate_response(query, history)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
