# Exporting DeepSeek-V4-Flash to ONNX

Step-by-step guide for turning the
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
checkpoint into an 8-GPU ONNX model with the ONNX Runtime GenAI model builder,
and running it.

The model is 284 B parameters, so a single export produces roughly 172 GiB of
weights split across eight tensor-parallel ranks. Every step below is
reproducible on one 8x H200 node.

Scripts referenced here live in
[examples/python/deepseek_v4_flash_0731](../examples/python/deepseek_v4_flash_0731):

| script | role |
| --- | --- |
| `check_weights.py` | validates the checkpoint adapter and the fp8 operator against reference math |
| `export.py` | exports one graph per rank, all ranks in parallel by default |
| `dsv4_engine.py` | the serving layer: process launch, per-GPU KV cache, lockstep decode |
| `dsv4_prompt.py` | tokenizer plus the checkpoint's own message encoder |
| `run.py` | command-line front end for the engine |

---

## 1. What the builder produces

The builder lives in
[src/python/py/models/builders/deepseek_v4.py](../src/python/py/models/builders/deepseek_v4.py)
and is selected automatically for `architectures: ["DeepseekV4ForCausalLM"]`.
Two helper modules sit next to it:

| file | role |
| --- | --- |
| `builders/safetensors_store.py` | streaming reader for the sharded checkpoint, plus the external-data writer that lets a 150 GiB model be emitted layer by layer |
| `builders/deepseek_v4_weights.py` | serves the builder's key names over the raw checkpoint, expands the block-scaled fp8 scales, and repacks the fp4 experts into the QMoE layout |

### Weight formats

The checkpoint mixes three numeric formats, and each one takes a different path
into the graph:

| checkpoint format | tensors | what the builder emits |
| --- | --- | --- |
| block-scaled fp8 (`e4m3` weights + `ue8m0` scales, 128x128 tiles) | all dense projections: `wq_a`, `wq_b`, `wkv`, `wo_b`, shared-expert `sw1/sw2/sw3` | `com.microsoft.MatMulBlockQuantizedFp8Weight` consuming the fp8 bytes verbatim; only the scale is expanded from `[N/128, K/128]` to the operator's per-row `[N, K/128]` |
| fp4 (two codes per byte + `ue8m0` block scales) | the 256 routed experts per layer | `com.microsoft.QMoE` with `expert_weight_bits=4`, gate and up pre-interleaved into `fc1` (`swiglu_fusion=1`) |
| bf16 | embedding, lm_head, norms, router, compressor, hyper-connection tables | plain initializers; rmsnorm scales and anything feeding an fp32 subgraph are cast to `float` |

Nothing is dequantized on the way out, so the exported model is the same
accuracy as the original checkpoint.

### Parallelism

The export is **tensor parallel** across attention and the dense FFN and
**expert parallel** across the routed experts. Each rank keeps `256 / world`
experts and its own column slice of the router logits; the QMoE kernel
renormalizes over the local experts, the builder multiplies the rank output by
the local routing mass, and an `AllReduce` sums the ranks back into the exact
dense result.

Size budget at `world=8`:

```
routed experts   277.0 G params -> fp4 + scales   137.1 GiB
dense / attn / shared 6.0 G params -> fp8           5.6 GiB
embedding + lm_head   1.1 G params -> bf16          2.1 GiB
                                     ---------------------
per-rank ONNX model                                21.5 GiB
total on disk                                       172 GiB
```

An H200 has 143 GiB, so 21.5 GiB per rank leaves ample room for the KV cache.

---

## 2. Prerequisites

### ONNX Runtime

You need an ONNX Runtime GPU build with:

* NCCL enabled (`--use_nccl`) for `com.microsoft.AllReduce`
* bfloat16 registered on `AllReduce` / `AllGather`
* `com.microsoft.MatMulBlockQuantizedFp8Weight` (SM80+)
* `com.microsoft.QMoE` with 4-bit experts and `swiglu_fusion`

Verify the installed package:

```bash
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

`check_weights.py` in step 1 fails loudly if the fp8 operator is missing.

### Python packages

```
torch          >= 2.9 with torch.float8_e8m0fnu
onnx           == 1.19.1
onnx_ir        == 0.2.1
safetensors
transformers
numpy
```

### Disk and checkpoint

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir ~/DeepSeek-V4-Flash-0731
```

311 GB for the checkpoint plus ~200 GiB for the export.

### Environment

```bash
export CKPT=~/DeepSeek-V4-Flash-0731
export EX=$PWD/examples/python/deepseek_v4_flash_0731   # from the repo root
cd /tmp                                                 # run from a neutral directory
unset CUDA_VISIBLE_DEVICES                              # every script pins its own GPU

# The CUDA EP needs cuDNN 9 and CUDA 13 on the loader path. Without them the
# session silently falls back to CPU, which has no bfloat16 kernels, and the
# first bf16 node is blamed instead ("Could not find an implementation for
# Expand(13)"). `dsv4_engine.py` checks for this and says so.
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

> Run Python from a directory such as `/tmp`. A checkout named `onnxruntime` in
> the current working directory shadows the installed package.

---

## 3. Builder options

Pass these through `extra_options` (`--extra_options k=v` on the model-builder
CLI, or a dict when calling the builder directly). `export.py` sets them for
you.

| option | default | meaning |
| --- | --- | --- |
| `dsv4_checkpoint` | *(none)* | path to the safetensors checkpoint. Without it the builder expects a state dict to be supplied by the caller. |
| `dsv4_tp_world` | `1` | number of ranks |
| `dsv4_tp_rank` | `0` | this rank's index |
| `dsv4_moe_impl` | `qmoe` | `qmoe` for the fused 4-bit kernel, `dense` for an unquantized reference graph |
| `dsv4_max_seq_len` | `4096` | static upper bound baked into the KV cache shapes |
| `dsv4_stream_weights` | `1` | append every initializer to the external-data blob as it is produced. Turn off only for tiny models. |
| `dsv4_repack_device` | `cpu` | device used to repack the fp4 experts. `cuda` is much faster. |
| `dsv4_num_layers` | `0` (all) | export only the first N layers, for smoke tests |
| `filename` | `model.onnx` | output file name |

---

## 4. Step-by-step

### Step 1 - weight-format checks (~1 minute)

Validate the two format conversions against `inference/convert.py`, the
reference code shipped inside the checkpoint, and run one real projection
through the fp8 operator:

```bash
python $EX/check_weights.py --ckpt $CKPT
```

Expected:

```
  attn.wq_a.weight             max|d|=0.000e+00 OK
  attn.wkv.weight              max|d|=0.000e+00 OK
  ffn.sw2.weight               max|d|=0.000e+00 OK
  w1 mxfp4_dequantize          max|d|=0.000e+00 OK
  w1 pack_for_qmoe             max|d|=0.000e+00 OK
  w2 mxfp4_dequantize          max|d|=0.000e+00 OK
  w2 pack_for_qmoe             max|d|=0.000e+00 OK
  fc1 slot w1                  max|d|=0.000e+00 OK
  fc1 slot w3                  max|d|=0.000e+00 OK
  renamed keys                 OK
  MatMulBlockQuantizedFp8Weight max|d|=1.554e-02 rel=2.495e-03 OK
WEIGHT CHECK PASS
```

The fp8 scale expansion and the fp4 nibble order and transpose are checked
bit-exactly; only the operator probe has tolerance, since it accumulates in
bf16.

### Step 2 - four-layer smoke test (~90 seconds)

Export a prefix of the layers onto a single GPU and run it. This exercises every
node type and every initializer dtype without any collectives, and it is the
fastest way to catch graph type errors:

```bash
rm -rf /tmp/dsv4_w1
python $EX/export.py --ckpt $CKPT --out /tmp/dsv4_w1 --world 1 --layers 4
python $EX/run.py --model /tmp/dsv4_w1 --world 1 --max-new-tokens 2
```

Expected:

```
rank 0 exit=0: [rank 0] done in 46s, 15.4 GiB of weights
[rank 0] session ready in 29s
[rank 0] step 0: S=8 finite=True max=19.125 min=-21.375 argmax=98751
[rank 0] step 1: S=1 finite=True max=15.875 min=-20.125 argmax=17524
tokens: [98751, 17524]
ranks exited with [0]
```

The generated text is nonsense - only four of 43 layers are present - but
`finite=True` and a stable argmax mean the graph is type-correct and every
weight arrived in the right layout.

### Step 3 - full 8-rank export (~90 seconds)

Ranks are independent, so `export.py` runs all eight at once, one per GPU:

```bash
python $EX/export.py --ckpt $CKPT --out ~/dsv4_onnx --world 8
```

Expected:

```
rank 0 exit=0: [rank 0] done in 85s, 21.5 GiB of weights
rank 1 exit=0: [rank 1] done in 87s, 21.5 GiB of weights
... (x8)
```

Layout:

```
~/dsv4_onnx/
  rank_0/model.onnx        graph only, a few MB
  rank_0/model.onnx.data   21.5 GiB of external weights
  export_rank0.log
  ...
  rank_7/
```

The exporter never holds more than one tensor at a time: `SafeTensorStore` reads
from the safetensors shards on demand, and `ExternalDataWriter` appends each
initializer to `model.onnx.data` the moment it is built. Peak host memory is a
few GiB regardless of model size.

Add `--rank R` to export a single rank, `--repack-device cpu` if a GPU is short
on memory, and `--max-seq-len N` to change the context bound.

The equivalent invocation through the stock model-builder CLI is:

```bash
python -m onnxruntime_genai.models.builder \
  -i $CKPT -o ~/dsv4_onnx/rank_0 -p bf16 -e cuda -c /tmp/dsv4_cache \
  --extra_options dsv4_checkpoint=$CKPT dsv4_tp_world=8 dsv4_tp_rank=0 \
                  dsv4_repack_device=cuda
```

`export.py` additionally fills in the config fields the checkpoint omits
(`rope_scaling["rope_type"]`, `tie_word_embeddings`, `_name_or_path`).

### Step 4 - run the model

```bash
python $EX/run.py --model ~/dsv4_onnx --world 8 --tokenizer $CKPT \
  --prompt "The capital city of France is called" --raw --max-new-tokens 8
```

Expected:

```
rank 0 is listening; starting the remaining ranks
all 8 ranks ready in 77s (max_seq_len=4096, kv cache 76.8 MiB/rank)
prompt 8 tokens, generated 8 (max_new_tokens); prefill 8.02s, decode 2.23 tok/s
tokens: [11111, 16, 983, 344, 270, 9152, 4593, 295]
text: ' Paris. It is the largest city in'
```

`--raw` tokenizes the prompt verbatim, which is what you want for a plain
continuation. Drop it to send the prompt as a user turn through the
checkpoint's message encoder, optionally with `--system`, `--thinking-mode
thinking` and `--reasoning-effort`:

```bash
python $EX/run.py --model ~/dsv4_onnx --tokenizer $CKPT \
  --prompt "Why is the sky blue?" --thinking-mode thinking --max-new-tokens 512
```

`--tokenizer` is optional: without it, `--prompt` must be a JSON list of token
ids and the output is not decoded.

---

## 5. Serving

`run.py` is a thin shell over `dsv4_engine.DSV4Engine`, which is the piece to
reuse:

```python
from dsv4_engine import DSV4Engine
from dsv4_prompt import DSV4Prompt

prompt = DSV4Prompt(CKPT, thinking_mode="chat")
with DSV4Engine("~/dsv4_onnx", world=8) as engine:
    ids = prompt.encode_messages([{"role": "user", "content": "2+2?"}])
    out = engine.generate(ids, max_new_tokens=64, eos_token_ids=prompt.eos_token_ids)
    print(prompt.decode(out["tokens"]), out["decode_tok_s"])
```

The engine stays resident across requests, so the 80 s of session creation is
paid once.

### KV cache across eight GPUs

There is no cross-device cache and no host round trip. Each rank allocates its
own cache in its own GPU's memory and binds it into the session with
`IOBinding`:

* Every `past_*` graph input gets a CUDA tensor allocated **once**, at session
  start, from the shapes ORT reports. For `max_seq_len=4096` that is 38.4 MiB
  per rank - the model is MQA with a single 512-wide latent KV row, a 128-slot
  sliding window, and a compressed stream of `max_seq_len / ratio + 2` rows, so
  the cache is three orders of magnitude smaller than the weights.
* Two such sets exist per rank. A step binds set *A* as the `past_*` inputs and
  set *B* as the `present_*` outputs, then swaps. Nothing is allocated or copied
  between steps; the only per-step host work is rebinding pointers.
* `past_len` is a scalar input, so a request resets by zeroing the cache and
  setting `past_len = 0`.

### Why the ranks stay in lockstep

A rank cannot run alone - it blocks in the two `AllReduce` nodes of every layer
until the other seven arrive. A request is therefore a collective: the engine
sends the same prompt to all eight workers over a `socketpair` and drives the
decode loop token by token.

Greedy sampling is decided centrally. Every rank computes its own argmax, the
engine takes rank 0's and broadcasts it back before the next step. `lm_head` is
replicated on every rank and the inputs to it are post-`AllReduce`, so the
logits agree bit for bit almost always - but **not quite always**, roughly once
per several hundred low-confidence tokens. Letting each rank pick its own token
desynchronizes the group permanently the first time that happens, because the
losing rank feeds a different token into the next `AllReduce`. The engine counts
these events and returns them as `rank_disagreements`; the round trip costs
about 0.1 ms against a 100 ms step.

### How the launcher works

`DSV4Engine` forks one process per rank and sets, for rank `r`:

| variable | value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `r` |
| `LOCAL_RANK` | `r` |
| `LOCAL_WORLD_SIZE` | `world` |
| `RANK0_IP` | `127.0.0.1` |
| `RANK0_PORT` | `19555` (or `port=`) |
| `DSV4_FD` | inherited socket used for the request protocol |

Two details matter:

* `RANK0_IP` must be `127.0.0.1`, not `localhost`. ORT's NCCL bootstrap resolves
  the name with `AF_UNSPEC` and connects to the first address family it can open
  a socket for, which may be IPv6 while rank 0 bound IPv4.
* Launch is **staged**. Rank 0 binds its socket deep inside session
  initialization, which takes over a minute with 21.5 GiB of weights, while
  peers give up after 40 seconds. The launcher polls `/proc/net/tcp{,6}` until
  rank 0 is in state `0A` (listening) before starting ranks 1-7.

Per-rank logs land in `/tmp/dsv4_rank{r}.log` (`log_dir=` to change).

---

## 6. Evaluating with the OpenAI evals harness

`evals/completion_fns/dsv4.py` in the [evals](https://github.com/openai/evals)
fork drives the engine, and `evals/registry/completion_fns/dsv4.yaml` registers
three variants. The model owns every GPU on the node, so the engine is a
process-wide singleton and the harness must run single-threaded:

```bash
OPENAI_API_KEY=dummy EVALS_THREADS=1 \
  oaieval dsv4/flash-0731-mcq10 match_mmlu_pro_800 \
    --record_path ~/dsv4_evals/mmlu_pro_800_chat.jsonl
```

`match_mmlu_pro_800` is a stratified 800-question subset of MMLU-Pro, drawn
across all 14 categories by largest-remainder allocation. `flash-0731-mcq10`
uses `thinking_mode=chat`, which closes the reasoning block immediately so the
model answers with a bare letter; `flash-0731-think-mcq10` leaves it open and
raises the token budget.

---

## 7. Benchmarking

`dsv4_benchmark.py` measures TTFT and prefill/decode throughput using the same
metric definitions as `benchmark/python/benchmark_e2e.py`: TTFT is prompt
processing plus first-token sampling, and decode throughput excludes that first
token. One engine is reused across every configuration, and EOS is disabled so
that each repetition decodes exactly `--gen-length` tokens.

```bash
python examples/python/deepseek_v4_flash_0731/dsv4_benchmark.py \
    --model ~/dsv4_onnx --gen-length 128 --reps 5
```

On 8x H200 at batch 1, with `max_seq_len=4096`:

| prompt | gen | TTFT (ms) | prefill tps | decode tps | ms/token |
|-------:|----:|----------:|------------:|-----------:|---------:|
|    128 | 128 | 144.3 | 887 | 10.29 | 97.2 |
|    512 | 128 | 227.1 | 2255 | 10.26 | 97.5 |
|   1024 | 128 | 344.7 | 2971 | 10.25 | 97.6 |
|   2048 | 128 | 593.4 | 3451 | 10.23 | 97.7 |
|   3072 | 128 | 875.3 | 3510 | 10.26 | 97.5 |

Decode latency is flat at ~97.5 ms/token across a 24x range of context length,
so neither attention nor memory bandwidth is the constraint - the cost is
per-pass kernel launch overhead spread over the graph's 25,757 nodes. Fitting
the prompt sweep gives `TTFT = 97.5 ms + 0.253 ms/token`, and the intercept
equals the decode latency: every forward pass pays a fixed ~97.5 ms, while real
prefill compute is a quarter of a millisecond per token. Prefill amortizes that
overhead (89% of TTFT is compute at 3072 tokens), decode does not amortize it at
all. Fusing the hyper-connection, KV-compressor and attention subgraphs, or
capturing the decode step as a CUDA graph, is what would move decode throughput.

Because prompts are random token ids, the engine occasionally reports a rank
argmax disagreement with a top-2 gap of exactly zero. That is a tie on nonsense
input, not numerical drift.

> Do not add a GPU-memory sampler thread to the benchmark process. Forking
> `nvidia-smi` concurrently with the engine's `Popen(pass_fds=...)` rank spawn
> corrupts NCCL bootstrap: rank 0 dies at the first `AllReduce` with
> "NCCL failure 1: unhandled cuda error" and the other seven hang.

---

## 8. Troubleshooting

**`Type Error: Type (tensor(float)) of output arg (...) does not match expected type (tensor(bfloat16))`**
An initializer feeding an fp32 subgraph was registered in bf16. Register it with
`to=ir.DataType.FLOAT`. This applies to all rmsnorm scales, the router
`gate_weight`, the compressor `wkv`/`wgate`, and the hyper-connection `*_fn`
tables. Step 2 catches these in about a minute.

**`Type parameter (T) of Optype (MatMul) bound to different types`**
Same cause: one side of the MatMul is bf16 and the other is fp32.

**`com.microsoft:MatMulBlockQuantizedFp8Weight(-1) is not a registered function/op`**
The installed ONNX Runtime predates the operator, or the build is CPU-only.

**NCCL `Connection refused`, or ranks hanging at startup**
Rank 0 was not listening yet. Use `run.py`'s staged launch rather than starting
the ranks yourself, and check `/tmp/run_rank0.log` for `session ready`.

**`rank 0 never listened on port ...`**
Session creation failed; the real error is at the end of `/tmp/run_rank0.log`.

**An export rank exits non-zero**
The traceback is in `<out>/export_rank<r>.log`.

**`ModuleNotFoundError`, or the wrong `onnxruntime` is imported**
Run from `/tmp`, not from a directory containing an `onnxruntime` or
`onnxruntime-genai` checkout.

**`path traversal attack` / `hard link attack` when loading the model**
`onnx_ir` refuses symlinked or hardlinked external data. Keep `model.onnx.data`
a regular file next to `model.onnx`.

**Out of memory while repacking experts**
Use `--repack-device cpu`. It is slower, but the repack is a small fraction of
the total export time.

---

## 9. Known limitations

* The lightning indexer is exported as a dense `window ∪ all-compressed` read
  set rather than a top-512 gather. This is numerically exact below roughly 2 K
  tokens and conservative above it.
* `lm_head` is replicated on every rank rather than vocabulary-sharded with an
  `AllGather`.
* DSpark speculative layers (40-42) are exported as ordinary layers; the
  multi-token-prediction runtime is not wired up.
* The graph is composed of primitive ONNX ops. Fusing the hyper-connections,
  attention, and the KV compressor into dedicated contrib operators would cut
  the node count by roughly an order of magnitude, but that is a performance
  concern only.
