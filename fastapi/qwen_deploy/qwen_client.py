import requests

from message_type import *

# ChatCompletionRequest
# url: localhost:8000/v1/chat/completions

MODEL_UID = "qwen1.5-chat-7b"
URL = "http://localhost:8000/v1/chat/completions"
prompt = input("Enter your prompt: ")
stream = False
payload = {
    "model": MODEL_UID,
    "n": 1,
    "temperature": 0,
    "top_p": 1.0,
    "messages": [{"role": "user", "content": prompt}],
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
    if response.status_code == 200:
        # handle steam response
        for chunk in response.iter_content(chunk_size=1024):
            # 处理响应内容
            print(chunk.decode("utf-8"))
    else:
        print("Failed to get response:", response.status_code, response.text)
