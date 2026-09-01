# ONNX Runtime GenAI Engine Sample

## Models and intent

This sample runs text models through the ONNX Runtime GenAI
continuous-batching Engine. It is intended to make it easy to:

- submit one or more requests to the same Engine;
- compare throughput at different batch sizes;
- measure time to first token, prompt throughput, and decode throughput;
- use built-in CUDA or a CUDA plugin execution provider;
- try custom prompts without changing the Python code.

The sample is model-agnostic. These two Qwen 3.8 27B models are available for
testing in the private `foundrylocalmodels` staging account:

| Model | Azure path | When to use it |
| --- | --- | --- |
| INT4 weights, INT8 KV | `paged-attention/qwen3.8-27b-int4-int8-kv/` | Recommended default. Similar short-context speed with more KV-cache capacity. |
| INT4 weights, FP16 KV | `paged-attention/qwen3.8-27b-int4-fp16-kv/` | Use when you want FP16 KV-cache values. |

The models were tested on an NVIDIA A100 80GB with CUDA 12.8.

Run the commands below from the `samples/qwen3.8` directory.

## Install the runtime

Create a Python environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install ONNX Runtime GenAI from the package feed. Replace the placeholders with
the feed URL and package version:

```bash
python -m pip install \
  --extra-index-url "<FEED_URL>" \
  "onnxruntime-genai-cuda==<PACKAGE_VERSION>"
```

For the CUDA plugin EP, install:

```bash
python -m pip install \
  --extra-index-url "<FEED_URL>" \
  "onnxruntime-ep-cuda12==0.2.0.dev20260831164052"
```

Use package versions published together in the same feed.

## Download a model

Install Azure CLI and AzCopy, then sign in:

```bash
az login --tenant 72f988bf-86f1-41af-91ab-2d7cd011db47
az account set --subscription 00c06639-6ee4-454e-8058-8d8b1703bd87

export AZCOPY_AUTO_LOGIN_TYPE=AZCLI
export AZCOPY_TENANT_ID=72f988bf-86f1-41af-91ab-2d7cd011db47
```

Download the INT8-KV model:

```bash
mkdir -p models/qwen3.8-27b-int4-int8-kv

azcopy copy \
  "https://foundrylocalmodels.blob.core.windows.net/staging/paged-attention/qwen3.8-27b-int4-int8-kv/*" \
  "models/qwen3.8-27b-int4-int8-kv" \
  --recursive=true
```

Or download the FP16-KV model:

```bash
mkdir -p models/qwen3.8-27b-int4-fp16-kv

azcopy copy \
  "https://foundrylocalmodels.blob.core.windows.net/staging/paged-attention/qwen3.8-27b-int4-fp16-kv/*" \
  "models/qwen3.8-27b-int4-fp16-kv" \
  --recursive=true
```

Each model is about 17 GB. Storage access requires a data-plane role such as
**Storage Blob Data Reader**.

## Run the sample

The repository includes eight short prompts in `prompts.json`. Run one request:

```bash
python engine_example.py \
  --model models/qwen3.8-27b-int4-int8-kv \
  --batch-size 1 \
  --metrics
```

Try different batch sizes:

```bash
python engine_example.py \
  --model models/qwen3.8-27b-int4-int8-kv \
  --batch-size 4 \
  --max-new-tokens 128 \
  --metrics
```

The sample cycles through the bundled prompt list when the batch size is larger
than the number of prompts.

## Use the CUDA plugin EP

Add `--cuda-plugin` to discover and register the installed
`onnxruntime_ep_cuda` package:

```bash
python engine_example.py \
  --model models/qwen3.8-27b-int4-int8-kv \
  --cuda-plugin \
  --batch-size 4 \
  --metrics
```

You can also register a plugin library directly:

```bash
python engine_example.py \
  --model <MODEL_DIR> \
  --cuda-plugin-library /path/to/libonnxruntime_providers_cuda.so \
  --batch-size 4 \
  --metrics
```

## Use custom prompts

Pass one prompt:

```bash
python engine_example.py \
  --model <MODEL_DIR> \
  --prompt "Explain continuous batching." \
  --metrics
```

Repeat `--prompt` to create a prompt pool:

```bash
python engine_example.py \
  --model <MODEL_DIR> \
  --prompt "Review this function." \
  --prompt "Explain this test failure." \
  --batch-size 4 \
  --metrics
```

Or provide a JSON array:

```bash
python engine_example.py \
  --model <MODEL_DIR> \
  --prompt-file my_prompts.json \
  --batch-size 8 \
  --metrics
```

Prompts over 4,096 tokens are rejected by default. Raise
`--max-prompt-tokens` explicitly for a deliberate long-context test.

## Metrics

With `--metrics`, the sample prints:

- per-request time to first token;
- time until all requests receive their first token;
- effective prompt tokens per second, including Engine queueing;
- aggregate steady decode tokens per second;
- input/output token counts and total wall-clock time.

Model loading and prompt tokenization are excluded from metric timing.
