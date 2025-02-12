# Tensorrt (v0.16)

# Quick Start

## enter docker

```bash
# Open a docker image
cd TensorRT-LLM
sudo docker run -dit --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all --volume ${PWD}:/code/tensorrt_llm -v /data1/rcai:/test_data --workdir /code/tensorrt_llm --runtime=nvidia --name v016_tensorrtllm_benchmark tensorrt_v016 /bin/bash

# enter docker
sudo docker exec -it v016_tensorrtllm_benchmark bash
```

## convert model to tensorrt-llm checkpoint format

```bash
# Qwen
cd TensorRT-LLM/examples/qwen

# Convert the model into TensorRT-LLM checkpoint format
python convert_checkpoint.py --model_dir  /test_data/Qwen2.5-32B-Instruct/  \
                              --output_dir /tmp/tllm_checkpoint_8gpu_fp16 \
                              --tp 8 \
                              --dtype float16
```

## Compile model

```bash
trtllm-build --checkpoint_dir tllm_checkpoint_8gpu_fp16 \
    --output_dir /tmp/tllm_engine_bs16_tp8_fp16 \
    --gemm_plugin auto \
    --max_batch_size 16 \
    --max_num_tokens 16384 \
    --max_input_len 1024
```

> 这个不同的参数配置会影响不同的性能

## run

```bash
cd TensorRT-LLM/examples
export  OMPI_ALLOW_RUN_AS_ROOT=1=1
export  OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun -n 8 python3 run.py --engine_dir ./qwen/tllm_engine_bs16_tp8_fp16 --max_output_len 100 --tokenizer_dir /test_data/Qwen2.5-32B-Instruct --input_text "How do I count to nine in French?"
```

# benchmark

## perf-benchmark

https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-benchmarking.md

TensorRT-LLM 提供了 trtllm-bench CLI，一个打包的基准测试工具。但是这个工具只支持部分的模型。

```
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-70b-hf
tiiuae/falcon-180B
EleutherAI/gpt-j-6b
meta-llama/Meta-Llama-3-8B
meta-llama/Llama-3.1-8B
meta-llama/Meta-Llama-3-70B
meta-llama/Llama-3.1-70B
meta-llama/Llama-3.1-405B
mistralai/Mixtral-8x7B-v0.1
mistralai/Mistral-7B-v0.1
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.1-70B-Instruct
meta-llama/Llama-3.1-405B-Instruct
mistralai/Mixtral-8x7B-v0.1-Instruct
```

> ref: https://github.com/NVIDIA/TensorRT-LLM/blob/16d2467ea8e3b45d7d9555010c2ad018d006d5c1/docs/source/performance/perf-benchmarking.md#throughput-benchmarking

> 具体支持的模型配置 , 详见： https://github.com/NVIDIA/TensorRT-LLM/blob/v0.16.0/tensorrt_llm/bench/build/benchmark_config.yml

### benchmark.py

> 详见 benchmark/python/benchmark.py

#### example

这里直接用上面quick start转出来的engine演示, 这个engine是针对

- batch_size = 16
- input_len = 1024
- output_len = 1024
- tp = 8

准备的

```bash
cd benchmarks/python

mpirun -n 8 python benchmark.py \
--model dec \
--engine_dir /tmp/tllm_engine_bs16_tp8_fp16 \
--batch_size 16 \
--input_output_len "1024,1024" \
--num_runs 10
```

运行结果

```bash
[BENCHMARK] engine_dir tllm_engine_bs16_tp8_fp16 world_size 8 num_heads 40 num_kv_heads 8 num_layers 64 hidden_size 5120 vocab_size 152064 precision float16
batch_size 16
gpu_weights_percent 1.0
input_length 1024
output_length 1024
gpu_peak_mem(gb) 0.0
build_time(s) None
tokens_per_sec 257.84
percentile95(ms) 63916.113 percentile99(ms) 63916.113
latency(ms) 63543.434
compute_cap sm86
quantization QuantMode.0
generation_time(ms) 49542.179
total_generated_tokens 16368.0
generation_tokens_per_second 330.385
```

#### metrics

- latency, 这就是平均时延

```python
# line 161 --------- gpt_benchmark.py
def run(self, inputs, config, benchmark_profiler=None):
    batch_size, inlen, outlen = config[0], config[1], config[2]
    self.decoder.setup(batch_size, inlen, outlen, beam_width=self.num_beams)
    if self.remove_input_padding:
        self.decoder.decode_batch(inputs[0],
                                  self.sampling_config,
                                  benchmark_profiler=benchmark_profiler)
    else:
        self.decoder.decode(inputs[0],
                            inputs[1],
                            self.sampling_config,
                            benchmark_profiler=benchmark_profiler)
    torch.cuda.synchronize()
```

```python
# line 262 --------- benchmarker.py
start_time = time()
while iter_idx < args.num_runs or cur_duration < args.duration:
    start.record()
    benchmarker.run(inputs,
                    config,
                    benchmark_profiler=benchmark_profiler)
    end.record()

    torch.cuda.synchronize()
    latencies.append(start.elapsed_time(end))

# line 305 -------- benchmarker.py
latency = round(sum(latencies) / iter_idx, 3)
```

- tokens_per_sec

```python
# line 186 --- gpt_benchmark.py
tokens_per_sec = round(batch_size * outlen / (latency / 1000), 2)
```

- generation_time
  这个指标大概相当于 timestamp(last_token) - timestamp(first_token)

```python
# line 3784 ------- generation.py
def profile_fn(benchmark_profiler_obj, step_count):
    if benchmark_profiler_obj is not None:
        benchmark_profiler_obj.record_cuda_event('last_token')
        benchmark_profiler_obj.record_elapsed_time(
            'first_token', 'last_token', 'generation_time')
        benchmark_profiler_obj.add_aux_info('generation_step_count',
                                            step_count)

#  line 60 -------- benchmark_profiler.py
def record_elapsed_time(self, start_event_name: str, end_event_name: str,
                        timer_name: str):
    if timer_name not in self.timer_dict.keys():
        self.timer_dict[timer_name] = 0.0
    if not self.started:
        return
    self.get_cuda_event(start_event_name).synchronize()
    self.get_cuda_event(end_event_name).synchronize()
    self.timer_dict[timer_name] += self.get_cuda_event(
        start_event_name).elapsed_time(self.get_cuda_event(end_event_name))
```

- total_generated_tokens
  这个是框架内部统计的生成token的数量

  > TODO: 要注意的是，这个数值不等于 1024 \* 16，具体是什么含义还要认真看代码。

- generation_tokens_per_second
  generation_tokens_per_second = total_generated_tokens / generation_time
