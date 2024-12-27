import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# 创建FastAPI应用
app = FastAPI()
# 定义请求体模型
class Query(BaseModel):
    text: str
# 定义路由
@app.post("/chat")
async def chat(query: Query):
    # 声明全局变量以便在函数内部使用模型和分词器
    global model, tokenizer  
    response, history = model.chat(tokenizer, query.text, history=None)
    return {
            "result": response,  # 返回生成的响应
    }

def load_model(model_name):
    model_name = model_name
    # 声明全局变量以便在函数内部使用模型和分词器
    global model, tokenizer  
    # 加载预训练的分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # 加载分词器
     # 加载模型并设置为评估模式
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    ).eval() 
    model.generation_config = GenerationConfig.from_pretrained(model_name,trust_remote_code=True)  
    # 设置模型为评估模式
    model.eval() 

# 主函数入口
if __name__ == '__main__':
    # 模型路径
    model_name = "D:\qwen\Qwen-1_8B-Chat"
    # 设置0号GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    load_model(model_name=model_name)
    uvicorn.run(app, host='127.0.0.1', port=8080, workers=1)