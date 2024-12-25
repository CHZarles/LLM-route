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
  > .. 未看

#### 协程函数例子

- 首先是要注意看上面材料 “Code Samples” 部分的例子 （上面的例子是python2的
- 然后是要理解 “描述性编程” 这个概念。
- 这些协程函数组合起来，也只是起到 pipeline 的作用，没法做到“乱序”的并发

#### Core Conception

- " If you use yield more generally, you get a coroutine"
- " Instead, functions can consume values sent to it"

### 协程（新）

## Fastapi + LLM 实践

intro: https://www.bilibili.com/video/BV1UfWCeREy5/?vd_source=27d3b33a76014ebb5a906ad40fa382de
