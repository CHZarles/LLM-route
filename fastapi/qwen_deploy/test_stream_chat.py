from threading import Thread

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

device = "cuda"  # 加载模型的设备
model_path = "./Qwen/Qwen1.5-7B-Chat"  # 模型路径
generation_config = GenerationConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "请介绍一下基于人类反馈的强化学习HFRL方法"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=8192)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

generated_text = ""
for new_text in streamer:
    generated_text += new_text
# yield generated_text
print("####################################")
print(generated_text)
