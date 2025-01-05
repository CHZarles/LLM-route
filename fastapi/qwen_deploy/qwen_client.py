import asyncio
import json
import os
import re
from typing import List, Tuple

import requests
import websockets

from message_type import *

# ChatCompletionRequest
# url: localhost:8000/v1/chat/completions


# [{"role": "user", "content": prompt}]
def request_http(messages: List[Dict[str, str]], stream: bool = False):
    MODEL_UID = "qwen1.5-chat-7b"
    URL = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": MODEL_UID,
        "n": 1,
        "temperature": 0,
        "top_p": 1.0,
        "messages": messages,
        "max_tokens": 8192,
        "stream:": stream,
    }

    response = requests.post(URL, json=payload)
    # requests
    if not stream:
        if response.status_code == 200:
            completion = response.json()
            print("Response:", completion)
        else:
            print("Failed to get response:", response.status_code, response.text)
    else:
        # NOTE: 实际测试中这个代码还是会一次把结果打印回来,我期望的效果是下面这种
        """
        curl -X POST  http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json" `
        -d '{
            "model": "qwen1.5-7b-chat",
            "messages": [
              {
                "role": "system",
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
        if response.status_code == 200:
            # handle steam response
            for chunk in response.iter_content(chunk_size=1024):
                # 处理响应内容
                print(chunk.decode("utf-8"))
        else:
            print("Failed to get response:", response.status_code, response.text)


async def request_websocket(query: str, history: List[Tuple[str, str]]):
    # wscat ws://localhost:8000/chat/completions/ws
    uri = "ws://localhost:8000/chat/completions/ws"
    # receive the response until the server send messages with status 200
    async with websockets.connect(uri) as websocket:
        # Prepare the input JSON string
        input_data = {"query": query, "history": history}
        await websocket.send(json.dumps(input_data))

        while True:
            response = await websocket.recv()
            output_data = json.loads(response)
            print(f"Response: {output_data['response']}")
            print(f"History: {output_data['history']}")
            print(f"Status: {output_data['status']}")
            if output_data["status"] == 200:
                break


if __name__ == "__main__":
    # construct a messages with history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "一拳超人埼玉的朋友是谁"},
        {
            "role": "assistant",
            "content": "在《一拳超人》这部漫画中，埼玉（Saitama）最常被提及的朋友是杰诺斯（Genos）。杰诺斯是一位改造人，他在埼玉的帮助下从爆炸中生还，之后便成为了埼玉的弟子和室友。杰诺斯对埼玉非常尊敬，经常试图学习埼玉的“秘诀”，尽管埼玉本人认为自己只是通过普通的锻炼变得强大。",
        },
        {"role": "user", "content": "他们两个谁的等级高"},
    ]

    # request the HTTP using curl
    playload = {"model": "qwen1.5-7b-chat", "messages": messages, "stream": False}
    curl_command = f"curl -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{json.dumps(playload)}'"
    print("\n ===========> request http with curl command stream=False <============ ")
    os.system(curl_command)
    playload["stream"] = True
    print("\n ===========> request http with curl command stream=True <============ ")
    curl_command = f"curl -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{json.dumps(playload)}'"
    os.system(curl_command)

    # request the HTTP using the requests library
    print("\n =============> request_http with stream=False <============= ")
    request_http(messages, stream=False)
    print("\n =============> request_http with stream=True  <============= ")
    request_http(messages, stream=True)

    # request the websocket
    print(" ===========> request websocket <============ ")
    query = messages[-1]["content"]
    history = [(m["role"], m["content"]) for m in messages[:-1]]
    asyncio.run(request_websocket(query, history))
