from typing import Union

from fastapi import FastAPI

# NOTE: feature 1: 创建server主体
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


# NOTE: feature 2: 路径参数 + 数据校验
@app.get("/f2/{item_id}")
# 这段代码把路径参数 item_id 的值传递给路径函数的参数 item_id。
# 本例把 item_id 的类型声明为 int。
# FastAPI 使用 Python 类型声明实现了数据校验。
async def read_int_item(item_id: int):
    return {"item_id": item_id}


# NOTE: feature 3: 预设路径参数
from enum import Enum


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/f3/{model_name}")
async def get_model(model_name: ModelName):
    if model_name == ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}
    # 即使嵌套在 JSON 请求体里（例如， dict），也可以从路径操作返回枚举元素
    return {"model_name": model_name, "message": "Have some residuals"}


# NOTE: feature 4:包含文件名的路径参数
@app.get("/f4/{file_path:path}")
# 本例中的 URL 是 /files//home/johndoe/myfile.txt
async def read_file(file_path: str):
    return {"file_path": file_path}


# NOTE: feature 5:查询参数
# 声明的参数不是路径参数时，路径操作函数会把该参数自动解释为查询参数。
fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/f5/defeaut_item/")
# http://127.0.0.1:8000/f5/?skip=0&limit=10
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]


@app.get("/users/{user_id}/f5/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: Union[str, None] = None, short: bool = False
):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item


"""
  File "/home/charles/LLM-route/fastapi/server.py", line 61, in <module>
    user_id: int, item_id: str, q: str | None = None, short: bool = False
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
"""


@app.get("/f5/request_item/{item_id}")
# http://127.0.0.1:8000/f5/foo-item?needy=sooooneedy
async def read_user_item_(item_id: str, needy: str):
    item = {"item_id": item_id, "needy": needy}
    return item


# NOTE: feature 6:请求体
"""
请求体是客户端发送给 API 的数据。响应体是 API 发送给客端的数据。
API 基本上肯定要发送响应体，但是客户端不一定发送请求体。
使用 Pydantic 模型声明请求体，能充分利用它的功能和优点。
"""
from pydantic import BaseModel


class Item(BaseModel):
    # 把数据模型声明为继承 BaseModel 的类。
    # 与声明查询参数一样，包含默认值的模型属性是可选的，否则就是必选的。
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None


@app.post("/f6/")
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict


# NOTE: feature 7:请求体 + 路径参数 [+ 查询参数]
@app.put("/f7/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}


@app.put("/f7/q/{item_id}")
async def update_item_(item_id: int, item: Item, q: Union[str, None] = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result


# NOTE: feature 8:查询参数 + 字符串校验
from fastapi import Query


@app.get("/f8/")
# 添加约束条件：即使 q 是可选的，但只要提供了该参数，则该参数值不能超过50个字符的长度。
async def read_items(q: Union[str, None] = Query(default=None, max_length=50)):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


@app.get("/f8/regrex/")
# 支持正则表达式
async def read_items(
    q: Union[str, None] = Query(
        default=None, min_length=3, max_length=50, pattern="^fixedquery$"
    ),
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


@app.get("/f8/default/")
async def read_items(q: str = Query(default="fixedquery", min_length=3)):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


# NOTE: feature 9:路径参数 + 数据校验
from fastapi import Path, Query


@app.get("/f9/{item_id}")
# 1. 可以使用 Path 为路径参数声明相同类型的校验和元数据。
# 2.要声明路径参数 item_id的 title 元数据值，你可以输入 ....
# 3. 传递 * 作为函数的第一个参数。Python 不会对该 * 做任何事情，但是它将知道之后的所有参数都应作为关键字参数（键值对），也被称为 kwargs，来调用。
# 即使它们没有默认值。
# 4. 你可以使用 ge 和 le 来声明最小值和最大值。
async def read_items(
    *,
    item_id: int = Path(title="The ID of the item to get", ge=0, le=1000),
    q: str,
    size: float = Query(gt=0, lt=10.5),
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    if size:
        results.update({"size": size})
    return results


try:
    from typing_extensions import Annotated
except ImportError:
    from typing import Annotated

from typing import List, Literal

# NOTE: feature 10:查询参数模型
from pydantic import BaseModel, Field


class FilterParams(BaseModel):
    model_config = {"extra": "forbid"}

    limit: int = Field(100, gt=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: List[str] = []


@app.get("/f10/")
# 限制你要接收的查询参数。
async def read_items(filter_query: Annotated[FilterParams, Query()]):
    return filter_query
