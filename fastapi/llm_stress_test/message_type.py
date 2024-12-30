import time
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


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


class PromptTokensDetails(BaseModel):
    cached_tokens: int


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[PromptTokensDetails] = None
    completion_tokens_details: Optional[CompletionTokensDetails] = None


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[Usage] = None
