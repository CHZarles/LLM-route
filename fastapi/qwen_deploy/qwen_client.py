import requests

from message_type import *

# ChatCompletionRequest
# url: localhost:8000/v1/chat/completions

MODEL_UID = "qwen1.5-chat-7b"
URL = "http://localhost:8000/v1/chat/completions"
prompt = input("Enter your prompt: ")
payload = {
    "model": MODEL_UID,
    "n": 1,
    "temperature": 0,
    "top_p": 1.0,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 8192,
}

# requests
response = requests.post(URL, json=payload)
if response.status_code == 200:
    completion = response.json()
    print("Response:", completion)
else:
    print("Failed to get response:", response.status_code, response.text)
