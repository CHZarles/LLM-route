## prepare

llm model : Qwen/Qwen1.5-7B-Chat
benchmark test_data : ShareGPT_V3_unfiltered_cleaned_split.json

## start the server

### command

```bash
python server.py

```

## run client

This clietn supports both HTTP and WebSocket protocols to send chat messages and receive responses from the server.

The client can send chat messages in three ways:

1. Using HTTP requests via the `requests` library.
2. Using HTTP requests via `curl` commands.
3. Using WebSocket connections via the `websockets` library.

```bash
python qwen_client.py

```

### output demostration

```
 ===========> request http with curl command stream=False <============
{"model":"qwen1.5-7b-chat","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"在《一拳超人》的故事中，埼玉（Saitama）的等级是非常高的，几乎无人能敌。他被称为\"最强的男人\"，在任何时候都能轻易击败任何强大的对手，甚至包括其他英雄协会的S级英雄。相比之下，杰诺斯
（Genos）虽然经过了改造，实力也相当强大，但他始终无法达到埼玉的程度。杰诺斯的等级在漫画中被标记为S+级，但即使如此，与埼玉相比还是有显著差距。"},"finish_reason":"stop"}],"created":1736056053,"usage":{"prompt_tokens":132,"completion_tokens":109,"total_tokens":241,"prompt_tokens_details":null,"completion_tokens_details":null}}
 ===========> request http with curl command stream=True <============
data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"在"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"故事"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"中"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"，埼"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"玉"},"finish_reason":null}]}

......

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"还是"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"无法"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"超越"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"埼"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"玉"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"。"},"finish_reason":null}]}

data: {"model":"qwen1.5-7b-chat","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]


 =============> request_http with stream=False <=============
Response: {'model': 'qwen1.5-chat-7b', 'object': 'chat.completion', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '在《一拳超
人》的故事中，埼玉（Saitama）的等级几乎是无敌的。他被称为“最强的男人”，无论对手有多强，只要被他一拳就能击败。相比之下，杰诺斯（Genos）虽然经过了改造
，力量和战斗技巧都非常高，但他的实力在埼玉面前还是显得微不足道。杰诺斯通常会挑战埼玉进行训练或者测试自己的进步，但他从未真正击败过埼玉。所以，可以说
埼玉的等级远高于杰诺斯。'}, 'finish_reason': 'stop'}], 'created': 1736056061, 'usage': {'prompt_tokens': 132, 'completion_tokens': 113, 'total_tokens': 245, 'prompt_tokens_details': None, 'completion_tokens_details': None}}

 =============> request_http with stream=True  <=============
{"model":"qwen1.5-chat-7b","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"在《一拳超人》的故事里，埼玉（Saitama）是无敌的存在，他的力量和速度几乎达到了人类的极限。无论面对多么强大的敌人，他都能一击击败，因此没有明确的等级来衡量他的实力。相比之下，杰诺斯（
Genos）虽然经过了改造，但他的力量和能力在埼玉面前仍然显得微不足道。杰诺斯在故事初期是一个超级英雄，后来在埼玉的影响下，他变得更加成熟和理解力量的意义
。\n\n简单来说，埼玉的等级是无法用常规的英雄等级来衡量的，因为他几乎无敌；而杰诺斯则是在埼玉的指导下成长的，他的等级相对于埼玉来说是非常低的。"},"finish_reason":"stop"}],"created":1736056066,"usage":{"prompt_tokens":132,"completion_tokens":153,"total_tokens":285,"prompt_tokens_details":null,"completion_tokens_details":null}}
 ===========> request websocket <============
Response: 在
Response: 在
Response: 在《一
Response: 在《一拳
Response: 在《一拳超
Response: 在《一拳超人
Response: 在《一拳超人
Response: 在《一拳超人》的故事
Response: 在《一拳超人》的故事线
Response: 在《一拳超人》的故事线中
Response: 在《一拳超人》的故事线中
Response: 在《一拳超人》的故事线中，埼
......
Response: 在《一拳超人》的故事线中，埼玉（Saitama）的等级几乎是无敌的。他被称为"最强的男人"，无论面对多么强大的敌人或怪兽，只要一拳就能解决，几乎没有
任何对手能与之抗衡。相比之下，杰诺斯（Genos）虽然经过了改造，实力也相当强大，但他尚未达到埼玉那种几乎无法打败的程度。杰诺斯通常会使用高科技武器和战斗
技巧，而他的力量和速度都是经过严格训练的，但与埼玉相比，他还是存在差距
Response: 在《一拳超人》的故事线中，埼玉（Saitama）的等级几乎是无敌的。他被称为"最强的男人"，无论面对多么强大的敌人或怪兽，只要一拳就能解决，几乎没有
任何对手能与之抗衡。相比之下，杰诺斯（Genos）虽然经过了改造，实力也相当强大，但他尚未达到埼玉那种几乎无法打败的程度。杰诺斯通常会使用高科技武器和战斗
技巧，而他的力量和速度都是经过严格训练的，但与埼玉相比，他还是存在差距
Response: 在《一拳超人》的故事线中，埼玉（Saitama）的等级几乎是无敌的。他被称为"最强的男人"，无论面对多么强大的敌人或怪兽，只要一拳就能解决，几乎没有
任何对手能与之抗衡。相比之下，杰诺斯（Genos）虽然经过了改造，实力也相当强大，但他尚未达到埼玉那种几乎无法打败的程度。杰诺斯通常会使用高科技武器和战斗
技巧，而他的力量和速度都是经过严格训练的，但与埼玉相比，他还是存在差距。

```

## Performence

### latency

```bash
python ./benchmark_latency.py

```

```bash
Total time: 143.41 s
Throughput: 0.07 requests/s
Average latency: 14.34 s
Average latency per token: 0.03 s
Average latency per output token: 0.04 s
```

### concurrency

```bash
python ./benchmark_serving.py

```

测试参数是 num_request = 10, concurrency = 3
测试环境是 单卡 v100 32G

```bash
Total time: 107.65 s
Throughput: 0.09 requests/s
Average latency: 29.44 s
Average latency per token: 0.09 s
Average latency per output token: 0.16 s
Throughput: 26.9574756281125 tokens/s
```
