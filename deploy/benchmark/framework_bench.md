# Vllm

# quick start

## basic api
```bash
# SPDX-License-Identifier: Apache-2.0
from vllm import LLM, SamplingParams
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="/data1/rcai/Qwen2.5-7B-Instruct", tensor_parallel_size=2)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```
## launch server
```python
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

```bash
curl http://localhost:8000/v1/completions     -H "Content-Type: application/json"     -d '{
        "model": "/data1/rcai/Qwen2.5-7B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

# benchmark

vllm 没有提供 benchmark 工具，脚本在 benchmarks 目录下。

```bash
# 针对不同场景的 benchmark 脚本
 backend_request_func.py
 benchmark_guided.py
 benchmark_latency.py
 benchmark_long_document_qa_throughput.py
 benchmark_prefix_caching.py
 benchmark_prioritization.py
 benchmark_serving_guided.py
 benchmark_serving.py
 benchmark_throughput.py
```

## 测试 latency

```bash
python benchmark_latency.py  \
    --model /data1/rcai/Qwen2.5-32B-Instruct \
    --batch-size 8 \
    --input-len 1024 \
    --output-len 1024 \
    --num-iters 10 \
    --tensor_parallel_size 8 \
    --output-json latency_8_1024_1024.json
```

> 最常见的option是这些，更复杂的特性见
> `python benchmark_latency.py --help`

```
# benchmark.json
{
    "avg_latency": 32.35338248407934,
    "latencies": [
        32.30754439602606,
        32.3486586951185,
        32.33120731706731,
        32.31845103390515,
        32.32320077391341,
        32.576691009802744,
        32.324395106872544,
        32.33950181095861,
        32.3329295089934,
        32.331245188135654
    ],
    "percentiles": {
        "10": 32.31736037011724,
        "25": 32.323499357153196,
        "50": 32.33122625260148,
        "75": 32.33785873546731,
        "90": 32.371461926586925,
        "99": 32.55616810148116
    }
}

```

> throughput = 8 \* (1024 + 1024) / 32.35 = 506.4

#### Metrics

- latency 统计的区间是 `llm.beam_search`

```
# benchmark_latency.py ----- line 45
def llm_generate():
    if not args.use_beam_search:
        llm.generate(dummy_prompts,
                     sampling_params=sampling_params,
                     use_tqdm=False)
    else:
        llm.beam_search(
            dummy_prompts,
            BeamSearchParams(
                beam_width=args.n,
                max_tokens=args.output_len,
                ignore_eos=True,
            ))

def run_to_completion(profile_dir: Optional[str] = None):
    if profile_dir:
        ......
    else:
        start_time = time.perf_counter()
        llm_generate()
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

```

## 测试 throughput

基本用法

```bash
python benchmark_throughput.py  \
    --model /data1/rcai/Qwen2.5-32B-Instruct \
    --input-len 1024 \
    --output-len 1024 \
    --num-prompts 8 \
    --tensor_parallel_size 8 \
    --output-json throughput_1024_1024.json
```

> 8个prompt，相当于batch_size=8,iter=1

```
# throughput_1024_1024.json
{
    "elapsed_time": 31.930586914066225,
    "num_requests": 8,
    "total_num_tokens": 16384,
    "requests_per_second": 0.2505434686036353,
    "tokens_per_second": 513.1130237002451
}
```

这个统计出来的值会比上面根据latency计算的稍微高一点

#### Metrics

- elapsed_time 统计的区间是

```python
start = time.perf_counter()
llm.beam_search(
    prompts,
    BeamSearchParams(
        beam_width=n,
        max_tokens=output_len,
        ignore_eos=True,
    ))
end = time.perf_counter()
```

- total_num_tokens = num_prompts \* (input_len + output_len)
- requests_per_second = num_prompts / elapsed_time
- tokens_per_second = total_num_tokens / elapsed_time
