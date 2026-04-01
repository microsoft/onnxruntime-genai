# Windows ML
## Setup venv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2) or any you like.

```
uv venv -p 3.12
.venv\Scripts\activate
uv pip install -r ./docs/requirements.txt
```

## Convert

```
uv run src/python/py/models/builder.py --model_name Qwen/Qwen3-4B -o docs/4b -p int4 -e cpu
[WIP] uv run src/python/py/models/builder.py --model_name Qwen/Qwen3-8B -o docs/8b-trtrtx -p fp16 -e NvTensorRtRtx
uv run src/python/py/models/builder.py --model_name Qwen/Qwen3-8B -o docs/8b-cuda -p int4 -e cuda
```

## Run

```
uv run examples/python/model-qa.py -m docs/4b
[WIP] uv run examples/python/model-qa.py -m docs/8b-trtrtx --use_winml --ep_path dummy
```

## Benchmark

```
uv run benchmark/python/benchmark_e2e.py -i docs/4b --chat_template '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'
```

## Result

### Qwen/Qwen3-4B + CPU

CPU Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz

| Metric | Value | Unit |
|---|---:|---|
| Average Tokenization Latency (per token) | 0.009268000023439527 | ms |
| Average Tokenization Throughput (per token) | 107898.1438790374 | tps |
| Average Prompt Processing Latency (per token) | 17.330111199931707 | ms |
| Average Prompt Processing Throughput (per token) | 57.70303424273127 | tps |
| Average Token Generation Latency (per token) | 121.97182760163986 | ms |
| Average Token Generation Throughput (per token) | 8.198614546188496 | tps |
| Average Sampling Latency (per token) | 0.2144200014299713 | ms |
| Average Sampling Throughput (per token) | 4663.744022623729 | tps |
| Average Wall Clock Time | 30.455924940109252 | s |
| Average Wall Clock Throughput | 9.226447745474118 | tps |

# onnxruntime CUDA

## Setup venv

```
uv venv -p 3.12 venv-cuda
.\venv-cuda\Scripts\activate
uv pip install -r .\docs\requirements-cuda.txt
```

[TODO: find a better source]

Download dlls from https://github.com/microsoft/windows-ai-studio-templates/releases?q=0.0.7&expanded=true to C:\Users\XXX\.aitk\bin\libonnxruntime_cuda_windows\0.0.7 (or other place)

## Run

```
$env:PATH += ";C:\Users\XXX\.aitk\bin\libonnxruntime_cuda_windows\0.0.7"
uv run examples/python/model-qa.py -m docs/8b-cuda -e cuda --ep_path "C:\Users\XXX\.aitk\bin\libonnxruntime_cuda_windows\0.0.7\onnxruntime_providers_cuda.dll"
```

## Benchmark

```
$env:PATH += ";C:\Users\XXX\.aitk\bin\libonnxruntime_cuda_windows\0.0.7"
uv run benchmark/python/benchmark_e2e.py -i docs/8b-cuda --chat_template '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n' -e cuda -epl "C:\Users\XXX\.aitk\bin\libonnxruntime_cuda_windows\0.0.7\onnxruntime_providers_cuda.dll"
```

## Result

### Qwen/Qwen3-8B + CUDA

| Metric | Value | Unit |
|---|---:|---|
| Average Tokenization Latency (per token) | 0.007992400031071156 | ms |
| Average Tokenization Throughput (per token) | 125118.86243336323 | tps |
| Average Prompt Processing Latency (per token) | 5.049365999933798 | ms |
| Average Prompt Processing Throughput (per token) | 198.0446654120757 | tps |
| Average Token Generation Latency (per token) | 29.319571991847656 | ms |
| Average Token Generation Throughput (per token) | 34.10690989206975 | tps |
| Average Sampling Latency (per token) | 0.10375999700045213 | ms |
| Average Sampling Throughput (per token) | 9637.625567738234 | tps |
| Average Wall Clock Time | 7.342488265037536 | s |
| Average Wall Clock Throughput | 38.270405053015565 | tps |
