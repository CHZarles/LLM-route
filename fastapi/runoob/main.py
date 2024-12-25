import time
from typing import Union

from fastapi import (Cookie, Depends, FastAPI, File, Form, Header,
                     HTTPException, UploadFile)
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

model_info = {
    0 : {
        "name":"Resnet50",
        "description":"This is model 0",
        "input_shape":[1, 28, 28]
                },
    1 : {
        "name":"Yolov5",
        "description":"This is model 1",
        "input_shape":[1, 30, 30]
                },
}



app=FastAPI()

# 基本路由
@app.get("/")
def read_root():
    return {"Desc":"Welcome to LLMROUTE : FastAPI"}



@app.get("/models/{model_id}")
def read_model_info(model_id:int , query: Union[str,None]=None):
    # check if model_id is valid
    if model_id not in model_info:
        # return {"Error":"Model not found"}
        raise HTTPException(status_code=404, detail="Error: Model not found")

    # check if query is valid
    if query:
        if query not in model_info[model_id]:
            raise HTTPException(status_code=404, detail="Error: Query not found")
            # return {"Error":"Query not found"}
        return {"model_id":model_id, "query":query, "info":model_info[model_id][query]}

    return {"model_id":model_id, "info":model_info[model_id]}



class ModelQueryRequest(BaseModel):
    query:str = "description" # query key
    limit:int = 50 # limit of result

@app.put("/models/{model_id}")
# 这里用了pydantic 所以用put请求，因为put请求可以有请求体
def read_model_info(model_id:int , query:ModelQueryRequest):
    # check if model_id is valid
    if model_id not in model_info:
        # return {"Error":"Model not found"}
        raise HTTPException(status_code=404, detail="Error: Model not found")

    # check if query is valid
    if query.query not in model_info[model_id]:
        raise HTTPException(status_code=404, detail="Error: Query not found")
        # return {"Error":"Query not found"}
    return {"model_id":model_id, "query":query.query, "info":model_info[model_id][query.query][:query.limit]}


@app.get("/models/")
def read_model_info(model_id:int, query:str, limit:int):
    # check if model_id is valid
    if model_id not in model_info:
        raise HTTPException(status_code=404, detail="Error: Model not found")

    # check if query is valid
    if query not in model_info[model_id]:
        raise HTTPException(status_code=404, detail="Error: Query not found")
        # return {"Error":"Query not found"}
    return {"model_id":model_id, "query":query, "info":model_info[model_id][query][:limit]}


# 请求头和cookie
@app.get("/header_and_cookie/")
def catch_header_and_cookie(user_agent: str = Header(None), session_token:str=Cookie(None)):
    return {"User-Agent": user_agent, "Session-Token": session_token}

# 重定向
@app.get("/request_state/")
def read_request_state():
    return RedirectResponse(url="/header_and_cookie/")

# 自定义响应头
@app.get("/check/{model_name}")
def check_model(model_name:str):
    # check if model exists
    flag = False
    for model_id in model_info:
        if model_info[model_id]["name"] == model_name:
            flag = True
            break
    if not flag:
        #responses 404 not found
        headers = {"Server": "FastAPI",
                "X-Custom-Header": "This is a custom header"
                   }
        return JSONResponse(headers=headers ,status_code=404, content={"Error":"Model not found, Please try other model name"})
    else:
        return {"Model":model_name}


# 路径依赖项
def get_model_info(model_name:str):
    for model_id in model_info:
        if model_info[model_id]["name"] == model_name:
            return model_info[model_id]
    return None

async def add_timestamp(model_info:dict = Depends(get_model_info)):
    if model_info is None:
        return None
    model_info["timestamp"] = time.time()
    return model_info

@app.get("/search/{model_name}")
def search_model(model_info:dict = Depends(add_timestamp)):
    if model_info is None:
        raise HTTPException(status_code=404, detail="Error: Model not found")
    return model_info

# 表单数据
class ModelInfo(BaseModel):
    name:str = Field(..., title="Model Name", description="Name of the model")
    description:str = Field(..., title="Model Description", description="Description of the model")
    input_shape:list = Field(..., title="Input Shape", description="Shape of the input tensor")

@app.post("/update/{model_id}")
async def update_model(model_id:int, model_name: str = Form(...), description:str = Form(...), input_shape:list = Form(...)):
    if model_id in model_info:
        # model id exists
        raise HTTPException(status_code=400, detail="Error: Model already exists")
    model_info[model_id] = {"name":model_name, "description":description, "input_shape":input_shape}
    return {"model_id":model_id, "model_info":model_info[model_id]}


# 上传文件
@app.post("/model_weight/{model_id}")
async def upload_model_weight(model_id:int, model_weight:UploadFile = File(...)):
    if model_id not in model_info:
        # model id not exists
        raise HTTPException(status_code=400, detail="Error: Model not found")
    # get file size (byte)
    file_size = str((len(model_weight.file.read()) / (1024 * 1024))) + " MB"
    
    return {"model_id":model_id, "save successfully":True, "model_name":model_info[model_id]["name"] ,  "model_weight":model_weight.filename,"file_size":file_size}
