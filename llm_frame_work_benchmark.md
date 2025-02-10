## technical idea

目的：统一指标维度，了解社区框架功能

1. 框架参考：vllm/trt-llm/sglang/lmdeploy 等
2. 维度：BS/Req/TTFT/Throughtput 等
3. 其他

## Sglang

ref:
https://www.bilibili.com/video/BV1Se41117SM?spm_id_from=333.788.videopod.sections&vd_source=27d3b33a76014ebb5a906ad40fa382de:w
https://www.bilibili.com/video/BV1neqDYVEVr/?spm_id_from=333.337.search-card.all.click&vd_source=27d3b33a76014ebb5a906ad40fa382de

### quick start

- server

```bash
# unset http_proxy ; unset https_proxy
python -m sglang.launch_server --model-path /data1/rcai/Qwen2.5-14B-Instruct --port 30000 --host 0.0.0.0 --tp 4

```

- client

```python3
import openai
client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
response = client.chat.completions.create(
    #model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(response)
```

> openai standard api

### benchmark and profiler

#### static single batch

Benchmark the latency of running a single static batch without a server.
Note that this is a simplified test script without a dynamic batching server, so it may run out of memory for a batch size that a real server can handle.
A real server truncates the prefill into several batches, while this simplified script does not.

[This script does not launch a server and uses the low-level APIs.](https://github.com/sgl-project/sglang/blob/cddb1cdf8fd85538003990dd12ab4f686d3da064/python/sglang/bench_one_batch.py#L4)

##### example

```bash

python -m sglang.bench_one_batch \
    --model-path /data1/rcai/Qwen2.5-32B-Instruct  \
    --tp 8 \
    --batch 4 8 \
    --input-len 1024 \
    --output-len 1 1024 \
    --result-filename benchmark.log \
    --run-name test_run \
```

> source: https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_one_batch.py
> cli options：https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_one_batch.py#L78
> 默认是做latency test，如果指定 --correctness_test, 则会做正确性测试 [详见](https://github.com/sgl-project/sglang/blob/cddb1cdf8fd85538003990dd12ab4f686d3da064/python/sglang/bench_one_batch.py#L259)

content of result file `benchmark.log`

```json
# benchmark.log

{
   "run_name":"test_run",
   "batch_size":4,
   "input_len":1024,
   "output_len":1,
   "prefill_latency":3.7364211082458496,
   "prefill_throughput":1096.2361793108923,
   "total_latency":3.7364211082458496,
   "overall_throughput":1097.3067224547506
}{
   "run_name":"test_run",
   "batch_size":4,

   "input_len":1024,
   "output_len":1024,
   "prefill_latency":3.58247447013855,
   "prefill_throughput":1143.3438072320973,
   "median_decode_latency":0.018887758255004883,
   "median_decode_throughput":211.77738225975435,
   "total_latency":22.969415187835693,
   "overall_throughput":356.64817467091535
}{
   "run_name":"test_run",
   "batch_size":8,

   "input_len":1024,
   "output_len":1,
   "prefill_latency":7.195620775222778,
   "prefill_throughput":1138.4702245855046,
   "total_latency":7.195620775222778,

   "overall_throughput":1139.5820119142015
}{

   "run_name":"test_run",
   "batch_size":8,
   "input_len":1024,
   "output_len":1024,
   "prefill_latency":7.162742376327515,
   "prefill_throughput":1143.6960272470678,
   "median_decode_latency":0.023103952407836914,
   "median_decode_throughput":346.2611010783757,
   "total_latency":30.811896800994873,

   "overall_throughput":531.7426611486958

}

```

terminal output

```bash
max_total_num_tokens=355636
Warmup ...
Prefill. latency: 4.32211 s, throughput:    947.69 token/s
Decode.  latency: 2.03403 s, throughput:      1.97 token/s
Decode.  latency: 0.02052 s, throughput:    194.90 token/s
Decode.  latency: 0.01973 s, throughput:    202.71 token/s
Decode.  latency: 0.02056 s, throughput:    194.52 token/s
Decode.  latency: 0.01969 s, throughput:    203.17 token/s
Decode.  median latency: 0.01973 s, median throughput:    202.71 token/s
Total. latency:  6.476 s, throughput:    637.43 token/s
Benchmark ...
Prefill. latency: 3.73642 s, throughput:   1096.24 token/s
Total. latency:  3.736 s, throughput:   1097.31 token/s
Prefill. latency: 3.58247 s, throughput:   1143.34 token/s
Decode.  latency: 0.03126 s, throughput:    127.94 token/s
Decode.  latency: 0.01914 s, throughput:    209.03 token/s
Decode.  latency: 0.01905 s, throughput:    209.94 token/s
Decode.  latency: 0.01889 s, throughput:    211.75 token/s
Decode.  latency: 0.01888 s, throughput:    211.87 token/s
Decode.  median latency: 0.01889 s, median throughput:    211.78 token/s
Total. latency: 22.969 s, throughput:    356.65 token/s
Prefill. latency: 7.19562 s, throughput:   1138.47 token/s
Total. latency:  7.196 s, throughput:   1139.58 token/s
Prefill. latency: 7.16274 s, throughput:   1143.70 token/s
Decode.  latency: 0.03835 s, throughput:    208.63 token/s
Decode.  latency: 0.02299 s, throughput:    347.90 token/s
Decode.  latency: 0.02286 s, throughput:    349.94 token/s
Decode.  latency: 0.02305 s, throughput:    347.04 token/s
Decode.  latency: 0.02290 s, throughput:    349.29 token/s
Decode.  median latency: 0.02310 s, median throughput:    346.26 token/s
Total. latency: 30.812 s, throughput:    531.74 token/s
```

##### explain

###### prefill_latency

prefill_latency = latency(set up token pool) + latency(setup kv cache pool) + latency(setup config of framework) + latency(generate the first token)

```python
#line 227 ------------
@torch.no_grad  # 禁用梯度计算，用于推理阶段
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,  # 请求列表
        req_to_token_pool=model_runner.req_to_token_pool,  # request到token的映射池
        token_to_kv_pool=model_runner.token_to_kv_pool,  # token到KV cache的映射池
        tree_cache=None,  # 树结构缓存（未使用）
        model_config=model_runner.model_config,  # 模型配置
        enable_overlap=False,  # 是否启用重叠优化
        spec_algorithm=SpeculativeAlgorithm.NONE,  # 不使用推测性解码
        enable_custom_logit_processor=False,  # 不使用自定义logit处理器
    )
    # 准备进行扩展操作
    batch.prepare_for_extend()
    # 获取用于模型处理的批次数据
    model_worker_batch = batch.get_model_worker_batch()
    # 初始化前向传播批次
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    # 执行模型前向传播，获取输出logits
    logits_output = model_runner.forward(forward_batch)
    # 根据logits采样下一个token
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    # 返回下一个token的ID、logits值和批次对象
    return next_token_ids, logits_output.next_token_logits, batch


# line 352 -------------
tic = time.time()
next_token_ids, _, batch = extend(reqs, model_runner)
synchronize(device)
prefill_latency = time.time() - tic
```

##### prefill_throughput

prefill_throughput = input_len \* batch_size / prefill_latency

> https://github.com/sgl-project/sglang/blob/cddb1cdf8fd85538003990dd12ab4f686d3da064/python/sglang/bench_one_batch.py#L358

##### decode latency

decode_latency = latency(prepare input batch) + latency(generate ith token)

```python

# line 248 ---------------------------
@torch.no_grad  # 禁用梯度计算，用于推理阶段
def decode(input_token_ids, batch, model_runner):
    # 将输入 token ID 设置为批次的输出 ID
    batch.output_ids = input_token_ids
    # 准备进行解码操作
    batch.prepare_for_decode()
    # 获取用于模型处理的批次数据
    model_worker_batch = batch.get_model_worker_batch()
    # 初始化前向传播批次
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    # 执行模型前向传播，获取输出 logits
    logits_output = model_runner.forward(forward_batch)
    # 根据 logits 采样下一个 token
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    # 返回下一个 token 的 ID 和 logits 值
    return next_token_ids, logits_output.next_token_logits


# line 369 ---------------------------
tic = time.time()
next_token_ids, _ = decode(next_token_ids, batch, model_runner) # next_token_ids 是上一次推理生成的token
synchronize(device)
latency = time.time() - ti

```

##### decode throughput

decode_throughput = batch_size / decode_latency

> https://github.com/sgl-project/sglang/blob/cddb1cdf8fd85538003990dd12ab4f686d3da064/python/sglang/bench_one_batch.py#L358

#### offline

```
python3 -m sglang.bench_offline_throughput --model-path /data1/rcai/Qwen2.5-32B-Instruct  --num-prompts 10
```

这个命令本质上也是先启动一个临时的server,然后在调用request来测试benchmark, 默认情况下会下载 `hareGPT_V3_unfiltered_cleaned_split.json` 到 ``\tmp` 目录
