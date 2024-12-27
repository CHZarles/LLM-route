from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model_name = "D:\qwen\Qwen-1_8B-Chat"
# 1. 使用`AutoTokenizer.from_pretrained()`方法初始化分词器。
# 2. 使用`AutoModelForCausalLM.from_pretrained()`方法初始化模型。
# 3. 调用模型的`chat()`方法来生成对输入提示的回复。
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "北京奥运会啥时候？", history=None)
print(response)
