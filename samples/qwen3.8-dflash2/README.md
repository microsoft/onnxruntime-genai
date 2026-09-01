# Qwen3.8 Speculative Decoding Engine Sample

## Models and intent

This sample runs a Qwen3.8-27B NVFP4 model through the ONNX Runtime GenAI
continuous-batching Engine with a **block drafter** attached, so every target
forward pass can emit several tokens instead of one. It is intended to make it
easy to:

- see the drafter's acceptance rate and tokens per target forward;
- compare speculative decoding against a non-speculative baseline built from the
  same model directory (`--no-drafter`);
- compare throughput at different batch sizes;
- measure time to first token, prompt throughput, and decode throughput;
- use built-in CUDA or a CUDA plugin execution provider.

For a non-speculative baseline, run this same sample with `--no-drafter`: it drops
the drafter session and decodes one token per target forward from the same model
directory, so the two runs are directly comparable.

The sample is drafter-agnostic. It reads `genai_config.json` and picks up
whichever of `model.dflash2`, `model.dspark`, or `model.mtp` declares a
`filename`, so it works unchanged as the drafter is swapped.

| Model directory | Target KV cache | Exported `max_batch_size` |
| --- | --- | --- |
| `~/models_local/qwen3.8-27b-nvfp4-fp16-kv-dflash2` | FP16 | 8 |
| `~/models_local/qwen3.8-27b-nvfp4-int8-kv-dflash2` | INT8 per-channel | 8 |

Both carry the same DFlash 2 drafter: 5 layers, an 8-token draft block over a
2,048-token sliding window, proposing up to 7 tokens per target forward. They
differ only in the target's KV-cache format, so they are a controlled A/B on
cache dtype. Pass the same `--max-batch-size` to both when comparing them; the
exported defaults differ.

Both targets are 64-layer hybrids: 16 full-attention layers backed by a paged KV
cache, and 48 GatedDeltaNet layers backed by fixed recurrent state. Tested on an
NVIDIA H200 with CUDA 13.0.

Run the commands below from the `samples/qwen3.8-dflash2` directory.

## Install the runtime

Create a Python environment and install a CUDA build of ONNX Runtime GenAI that
includes the Engine event-buffer API (`Engine.create_event_buffer`,
`og.EngineEventFlags`). To build it from this repository:

```bash
source /path/to/venv/bin/activate
export CUDA_HOME=/path/to/cuda
export ORT_HOME=/path/to/onnxruntime          # provides lib/ and include/
export LD_LIBRARY_PATH=$ORT_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDAARCHS=90                            # 90 = H100/H200

cd ../..
python build.py --use_cuda --ort_home "$ORT_HOME" --config Release --parallel
python -m pip install --force-reinstall \
  build/Linux/Release/wheel/onnxruntime_genai_cuda-*.whl
```

Verify the API the sample needs:

```bash
python -c "import onnxruntime_genai as og; print(hasattr(og.Engine, 'create_event_buffer'))"
```

## Run the sample

The sample bundles eight short prompts in `prompts.json`. Run one request:

```bash
CUDA_VISIBLE_DEVICES=0 python engine_example.py \
  --model ~/models_local/qwen3.8-27b-nvfp4-int8-kv-dflash2 \
  --batch-size 1 \
  --metrics
```

Add `--warmup` whenever you intend to read the numbers. It runs one short
throwaway request first, so lazy weight prepack and first-touch allocations are
not charged to the measured requests. Without it the first measured request pays
those costs, which inflates time to first token and can make speculative
decoding look slower than the baseline. Warmup activity is excluded from the
reported drafter statistics.

```bash
CUDA_VISIBLE_DEVICES=0 python engine_example.py \
  --model ~/models_local/qwen3.8-27b-nvfp4-int8-kv-dflash2 \
  --batch-size 1 --warmup --metrics
```

Try different batch sizes:

```bash
CUDA_VISIBLE_DEVICES=0 python engine_example.py \
  --model ~/models_local/qwen3.8-27b-nvfp4-int8-kv-dflash2 \
  --batch-size 4 \
  --max-new-tokens 256 \
  --metrics
```

The sample cycles through the bundled prompt list when the batch size is larger
than the number of prompts.

## Reasoning mode

Qwen3.8 emits a reasoning block before its answer, and the export enables it by
default at `xhigh` effort. The sample makes that explicit: `--thinking` is on by
default and pins the effort with `--reasoning-effort` (default `low`), because at
`xhigh` a short token budget is consumed entirely by reasoning and never reaches
the answer. `--max-new-tokens` therefore defaults to 2048.

Turn reasoning off for short, direct answers:

```bash
CUDA_VISIBLE_DEVICES=0 python engine_example.py \
  --model ~/models_local/qwen3.8-27b-nvfp4-int8-kv-dflash2 \
  --no-thinking --max-new-tokens 256 --metrics
```

Raise the budget for harder prompts:

```bash
CUDA_VISIBLE_DEVICES=0 python engine_example.py \
  --model ~/models_local/qwen3.8-27b-nvfp4-int8-kv-dflash2 \
  --reasoning-effort high --max-new-tokens 8192 --metrics
```

Reasoning mode changes answer quality substantially, so keep it fixed when
comparing runs. The published accuracy figures for these models were measured
with reasoning on.

## Measure the speculative speedup

`--no-drafter` overlays an empty drafter filename, which drops the drafter
session and decodes one token per target forward. Everything else, including the
weights and the KV-cache format, is unchanged, so the two runs are directly
comparable:

```bash
CUDA_VISIBLE_DEVICES=0 python engine_example.py \
  --model ~/models_local/qwen3.8-27b-nvfp4-int8-kv-dflash2 \
  --batch-size 1 --metrics --no-drafter
```

Use `--warmup` on both sides of this comparison, and give both the same
`--max-batch-size`.

Decode greedily (the default) when reading the acceptance rate: under greedy
decoding an accepted draft token is exactly the token the target would have
produced, so acceptance measures a lossless match. `--do-sample` is available but
makes acceptance depend on the sampling trajectory.

`--max-draft-tokens` overrides how many tokens the drafter proposes per target
forward. It defaults to the `num_draft_tokens` value in `genai_config.json` (7
for both models above).

## Control cache capacity

The exported `engine.dynamic_batching` settings can be overridden per run:

```bash
CUDA_VISIBLE_DEVICES=0 python engine_example.py \
  --model ~/models_local/qwen3.8-27b-nvfp4-int8-kv-dflash2 \
  --batch-size 8 \
  --max-batch-size 8 \
  --num-blocks 1024 \
  --metrics
```

- `--max-batch-size` is not only a scheduler limit. The Engine eagerly allocates
  fixed GatedDeltaNet recurrent state for every configured slot before it sizes
  the paged KV pool, at roughly 604 MiB per slot. Set it to the concurrency you
  actually serve; leaving it at 100 reserves about 59 GiB before any request runs.
- `--num-blocks` sets the paged KV pool explicitly and takes precedence over
  `--gpu-utilization`. With INT8 KV a 256-token block costs 8 MiB; with FP16 KV
  it costs 16 MiB.

One process holds tens of GiB of cache pool, so run cells sequentially rather
than putting two models on one GPU.

## Use the CUDA plugin EP

Add `--cuda-plugin` to discover and register the installed `onnxruntime_ep_cuda`
package:

```bash
python engine_example.py \
  --model <MODEL_DIR> \
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

Pass one prompt, repeat `--prompt` to build a pool, or supply a JSON array:

```bash
python engine_example.py --model <MODEL_DIR> --prompt "Explain speculative decoding." --metrics

python engine_example.py --model <MODEL_DIR> \
  --prompt "Review this function." \
  --prompt "Explain this test failure." \
  --batch-size 4 --metrics

python engine_example.py --model <MODEL_DIR> --prompt-file my_prompts.json --batch-size 8 --metrics
```

Prompts over 4,096 tokens are rejected by default. Raise `--max-prompt-tokens`
explicitly for a deliberate long-context test.

## Metrics

With `--metrics`, the sample prints per-request time to first token, time until
all requests receive their first token, effective prompt tokens per second,
aggregate steady decode tokens per second, and total wall-clock time. Model
loading and prompt tokenization are excluded from metric timing.

When a drafter is active it adds a `speculative` section:

| Field | Meaning |
| --- | --- |
| `acceptance_rate` | Accepted draft tokens over evaluated draft tokens. |
| `draft_tokens_proposed` / `_evaluated` / `_accepted` | Raw drafter counters. |
| `target_forward_passes` | Target model forwards, the cost speculation reduces. |
| `draft_forward_passes` | Drafter forwards, the cost speculation adds. |
| `output_tokens_per_target_forward` | End-to-end speedup factor over one token per forward. |
| `acceptance_length_histogram` | How many rounds accepted 0, 1, 2, ... draft tokens. |

Read `output_tokens_per_target_forward` as the headline number;
`acceptance_rate` explains it. Both are workload-specific and are not a
substitute for a task-level quality evaluation.

## Sweep contexts and batch sizes

`benchmark.py` runs a matrix of workloads and batch sizes and, by default,
runs both the speculative and the `--no-drafter` arm so every cell gets a
speedup. It has two ways to source prompts, and the choice matters more than it
looks: acceptance depends on how predictable the text is, so the prompt content
is part of the measurement, not just its length.

### Synthetic contexts (default)

Prompts are synthesized from a pool of varied passages up to a requested token
length, which makes context length the controlled variable. Use this to see how
a single workload scales with context and batch size.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark.py \
  --model ~/models_local/qwen3.8-27b-nvfp4-int8-kv-dflash2 \
  --context-length 512 --context-length 2048 --context-length 8192 \
  --batch-size 1 --batch-size 4 \
  --generate-length 256 \
  --max-batch-size 4 --num-blocks 512 \
  --json-out bench.json
```

The passages are deliberately varied and the pool is rotated per cycle, because
filler built by repeating one sentence is trivially predictable: the drafter
copies it with an induction head and acceptance comes out far higher than any
real workload would give. `--context-file` substitutes your own text. Treat
synthetic numbers as a scaling curve, not as an acceptance rate you can quote.

### Real prompts with `--dataset`

For acceptance and speedup numbers meant to represent a workload, drive the
sweep from a speculative-decoding prompt set. `--dataset` reads a `.jsonl` or
`.parquet` file with `category` and `turns` fields and makes each category one
row of the sweep:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark.py \
  --model ~/models_local/qwen3.8-27b-nvfp4-int8-kv-dflash2 \
  --dataset ~/data/spec_bench/question.jsonl \
  --dataset-category math_reasoning --dataset-category summarization \
  --dataset-category qa \
  --dataset-limit 8 --batch-size 1 --generate-length 256
```

Two public sets use this schema:

- [Spec-Bench](https://github.com/hemingkx/Spec-Bench) (Apache-2.0),
  `data/spec_bench/question.jsonl`, 480 items across translation,
  summarization, qa, math_reasoning and rag.
- [nvidia/SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench)
  (parquet, needs `pyarrow`). Its qualitative split is built for acceptance
  measurement; its throughput split has fixed input-length buckets. Some records
  ship as placeholders and must be filled in with NVIDIA's `prepare_data.py`
  first; the loader skips any that are still empty.

Neither dataset is redistributed here and neither is downloaded automatically —
point `--dataset` at a local copy and check its license. The loader uses only
the first turn of each item, so results are not comparable to those projects'
leaderboards; the point is a realistic prompt mix, not a reproduced score.

Category spread is the reason to bother. On the same engine and generation
length, real prompts separate by more than 1.5x in accepted tokens per target
forward, and a synthetic run reports one number that hides it:

| workload | prompt tok | accept | tok/target fwd |
| --- | ---: | ---: | ---: |
| math_reasoning | 67 | 0.888 | 5.12 |
| qa | 22 | 0.759 | 3.67 |
| summarization | 689 | 0.710 | 3.20 |
| synthetic ctx=512 | 512 | 0.777 | 3.76 |

### Reading the output

It prints a markdown table and optionally writes the full per-cell record as
JSON:

| workload | batch | arm | prompt tok | TTFT s | decode tok/s | accept | tok/target fwd | speedup |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | speculative | 512 | 0.115 | 184.5 | 0.777 | 3.76 | 2.516x |
| 2048 | 1 | speculative | 2046 | 0.317 | 195.3 | 0.800 | 4.00 | 2.669x |
| 512 | 1 | baseline | 512 | 0.097 | 73.3 | - | - | - |

`--arms speculative` or `--arms baseline` runs a single arm; the speedup column
needs both. Each arm loads the model once and reuses one engine across its cells,
so `--max-batch-size` and `--num-blocks` must cover the largest cell.

The script warms up at the largest workload and batch size before measuring.
Without that, the first measured cell absorbs first-touch allocation and reads
roughly twice as slow as it should. `--no-warmup` disables it, which is only
useful for measuring cold-start cost deliberately. Keep `--generate-length` at a
few hundred tokens: the decode-rate window is the span after every request has
produced its first token, and very short generations make it noisy.

### Requirements for speculation to engage

The engine silently decodes one token per forward unless all of these hold, so a
benchmark that violates one measures the baseline while reporting the
speculative arm:

- greedy decoding, or sampling with a positive `top_k`;
- `repetition_penalty == 1` and `no_repeat_ngram_size == 0`;
- the sequence is already past `min_length`.

The last one is easy to trip: pinning `min_length` to `max_length` to force a
fixed output length disables drafting for the entire run. `benchmark.py`
therefore sets only `max_length`, and reports `completed_full_length` per cell so
a request that stopped early on EOS is visible instead of quietly skewing the
rate.

## Interpreting FP16 versus INT8 KV

INT8 KV cache is first of all a capacity win. It halves the paged pool (8 MiB
versus 16 MiB per 256-token block), which is what makes very long contexts fit on
a single H200.

It is competitive on speed too, once the speculative paged XQA kernel is
available for a quantized cache. Measured on H200 with generation 256 and a
draft width of 7, decode throughput in tokens per second:

| Context | batch | INT8 KV | FP16 KV |
| ---: | ---: | ---: | ---: |
| 512 | 1 | 156.6 | 364.6 |
| 2048 | 1 | 233.1 | 184.2 |
| 8192 | 1 | 161.2 | 133.9 |
| 32768 | 1 | 178.5 | 119.0 |

INT8 leads from 2K upward and the gap widens with context; FP16 leads at very
short context. Treat the cross-dtype numbers as indicative only: the two arms
follow different token trajectories and their acceptance rates differ (0.72
versus 0.98 in the 512/1 cell above), so throughput mixes engine speed with
acceptance luck. `output_tokens_per_target_forward` is the acceptance-aware
figure, and a fixed `--num-blocks` on both sides is what makes the comparison
controlled.
