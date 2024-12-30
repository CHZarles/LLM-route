## FastApi 学习

- [菜鸟教程-fastapi](https://www.runoob.com/fastapi/fastapi-tutorial.html)
  - [practice source code](./runoob/)
- [official document](https://fastapi.tiangolo.com/zh/tutorial/#_1)

### 扩展阅读

- [open api](https://www.bilibili.com/video/BV16a411A7pt/?spm_id_from=333.337.search-card.all.click&vd_source=27d3b33a76014ebb5a906ad40fa382de)
- [pydantic](https://pydantic.com.cn/#:~:text=Pydantic%20%E6%98%AFPython%20%E4%B8%AD%E4%BD%BF%E7%94%A8,Pydantic%20%E5%AF%B9%E5%85%B6%E8%BF%9B%E8%A1%8C%E9%AA%8C%E8%AF%81%E3%80%82)

## python 协程

### 引言

fastapi的异步并发web框架, https://fastapi.tiangolo.com/zh/async/#_1, 框架十分推荐使用定义协程函数的方式来定义“路由函数”

> 如果你正在使用一个第三方库和某些组件（比如：数据库、API、文件系统...）进行通信，第三方库又不支持使用 await （目前大多数数据库三方库都是这样），这种情况你可以像平常那样使用 def 声明一个路径操作函数，就像这样：
>
> ```python
> @app.get('/')
> def results():
> results = some_library()
> return results
> ```
>
> 如果你的应用程序不需要与其他任何东西通信而等待其响应，请使用 async def。

所以要用好这个框架，还是最好要先学会什么是python的 ”协程“

### 经典协程

> 相关背景知识 [Generator](./generator)

- [A Curious Course on Coroutines and Concurrency 2009](http://www.dabeaz.com/coroutines/)
  > 看完Part1 就能知道 协程函数 这个概念的来源
  > 看完Part2 能知道协程函数组合pipeline的用法
  > Part3,4 ... 都是基于协程的高级封装，我还没看，如果要开发协程框架，那应该看一下

#### 协程函数例子

- 首先是要注意看上面材料 “Code Samples” 部分的例子 （上面的例子是python2的
- 然后是要理解 “描述性编程” 这个概念。
- 这些协程函数组合起来，也只是起到 pipeline 的作用，没法做到“乱序”的并发

#### Core Conception

- " If you use yield more generally, you get a coroutine"
- " Instead, functions can consume values sent to it"

### 原生协程

> 建议看一下 《流畅的python》chapter 21 ，里面比较详细地讲了 "原生协程" 和 "经典协程"

我这里直接摘录书中的内容

> 原生协程
> 　　使用 async def 定义的协程函数。在原生协程内可以使用
> await 关键字委托另一个原生协程，这类似于在经典协程中使用
> yield from。async def 语句定义的始终是原生协程，即使主体中
> 没有使用 await 关键字。await 关键字不能在原生协程外部使用。

#### Asyncio 框架

> 推荐书籍 《using-asyncio-python-understanding-asynchronous》

这本书给出了使用asyncio的指导思想:

> Yury Selivanov, the author of PEP 492 and all-round major contributor to async
> Python, explained in his PyCon 2016 talk “async/await in Python 3.5 and Why It Is
> Awesome,” that many of the APIs in the asyncio module are really intended for
> framework designers, not end-user developers. In that talk, he emphasized the main
> features that end users should care about. These are a small subset of the whole
> asyncio API and can be summarized as follows
>
> - Starting the asyncio event loop
> - Calling async/await functions
> - Creating a task to be run on the loop
> - Waiting for multiple tasks to complete
> - Closing the loop after all concurrent tasks have completed

书中还对ayncio的api做了分层,到时候按需要去了解就好

| Tier   | Level Concept Implementation                                                                           |
| ------ | ------------------------------------------------------------------------------------------------------ |
| Tier 9 | Network: streams `StreamReader`, `StreamWriter`, `asyncio.open_connection()`, `asyncio.start_server()` |
| Tier 8 | Network: TCP & UDP Protocol                                                                            |
| Tier 7 | Network: transports `BaseTransport`                                                                    |
| Tier 6 | Tools `asyncio.Queue`                                                                                  |
| Tier 5 | Subprocesses & threads `run_in_executor()`, `asyncio.subprocess`                                       |
| Tier 4 | Tasks `asyncio.Task`, `asyncio.create_task()`                                                          |
| Tier 3 | Futures `asyncio.Future`                                                                               |
| Tier 2 | Event loop `asyncio.run()`, `BaseEventLoop`                                                            |
| Tier 1 | (Base) Coroutines `async def`, `async with`, `async for`, `await`                                      |

个人心得：

- 多线程模型无法预测task切换的代码位置，但是用协程异步可以，因为协程函数的切换位置就是 await 调用处

#### 实战

参考这里面和 async 相关的 https://github.com/mCodingLLC/VideosSampleCode/tree/master

## Fastapi + LLM 实践

intro: https://www.bilibili.com/video/BV1UfWCeREy5/?vd_source=27d3b33a76014ebb5a906ad40fa382de

### OpenAI标准服务制作方法

把llm推理服务部的对外接口署成和Openai api的协议形式。

补充阅读：

[OpenAI api 国内中文文档](https://openai.xiniushu.com/docs/guides/chat)

[OpenAI API (二) 对话补全（Chat completions)](https://zhuanlan.zhihu.com/p/647532219)

[方糖技能栈-gpt](https://github.com/easychen/openai-gpt-dev-notes-for-cn-developer)

[手把手教会你如何通过ChatGPT API实现上下文对话](https://juejin.cn/post/7217360688263004217)

#### OpenAI 的Create chat completion 里面的 streaming 是指什么？

[OpenAI API （五）Chat Completion中的stream处理](https://zhuanlan.zhihu.com/p/651129737)

要调试这些api的用法，可以在tb买一些api自己调试着玩，api都是类似的,只是可能没有openai官方的那么全

> backup: https://apifox.com/apidoc/project-4798344/doc-5309262

结合pydantic + fastapi 实现openai api的具体例子见[openai_api.py](./chatglm6b_deploy/openai_api.py),

#### 实践

开启server

```bash
python ./chatglm6b_deploy/openai_api.py
```

测试请求

```bash
curl -X 'POST' \
  'http://localhost:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
        {  "role": "system",
           "content":"你是人工智能助手"
         },
        {
            "role": "user",
            "content":"海贼王作者是谁"
        },
       {
        "role": "assistant",
        "content": "海贼王的作者是中国漫画家藤本树。"
       },
       {"role":"user", "content":"他出生时间是什么"}
    ],
    "stream": false,
    "model": "gpt-3.5-turbo",
    "temperature": 0.5,
    "max_length": 100,
    "top_p": 1
}
```

### websocket流式输出（长连接）

查看源码: ./chatglm6b_deploy/websocket_api.py

这个代码有个问题就算是同一时间只能有一个client查询,server也没办法同时回应多个response

如果要用postman测试这个代码，请求体的内容要按下面的格式来

```
{
  "query": "I want to drive car",
  "history": [
    [
      "hhh",
      "I'm sorry, but I'm not sure what you're asking me. Can you please provide more context or clarify your question?"
    ],
    [
      "how are you",
      "As an AI language model, I don't have feelings in the same way that humans do, but I'm functioning properly and ready to assist you with any questions or tasks you may have. How can I assist you today?"
    ]
  ]
}

```

fastapi 的 websocket 的写法也很简单，参考 ./chatglm6b_deploy/test_websocket.py

### LLM 部署 压力测试

参考：https://blog.csdn.net/liuzhenghua66/article/details/139332747

> 这个压测sample用来qwen模型，但是qwen这个模型太大了，我想着用chatglm
> 但前者是预训练模型返回的是tokenid？反正就是一个数，后者是推理结果。

#### LLM压测的必要性

1. **性能评估**：压测能揭示模型在高负载下的响应时间和吞吐量，找出性能瓶颈。
2. **系统稳定性**：模拟高负载场景，确保模型在突发流量时稳定运行。
3. **扩展能力**：评估系统在增加负载时的可扩展性，检查扩展过程中的瓶颈。
4. **资源利用率优化**：分析高负载下的资源使用，优化配置，提高效率。
5. **发现潜在问题**：找出在高负载下可能出现的问题，如内存泄漏和连接超时，为优化提供依据。
