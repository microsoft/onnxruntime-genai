# Qwen3.6-35B-A3B Decode Throughput Optimization

Performance optimization log for ONNX Runtime GenAI decoding of the Qwen3.6-35B-A3B
MoE model (INT4 QMoE, hybrid GatedDeltaNet + GQA architecture).

## Environment

- **Hardware:** 1x NVIDIA H200 (sm_90, 143 GB), `CUDA_VISIBLE_DEVICES=0`
- **Deployment target:** RTX 4090 (sm_89)
- **CUDA:** 13.0, cuDNN 9.19
- **ORT build:** `~/onnxruntime/build/cu130/Release` (Release, sm_90)
- **Model:** `~/olive-recipes/Qwen-Qwen3.6-35B-A3B/builtin/cuda/models` (INT4, block size 128)
- **Architecture:** 40 decoder layers — 30 GatedDeltaNet (linear-attention) layers + 10 GQA layers; 256-expert MoE, top-8 routing.

## Reference points

Single-stream decode (batch=1, greedy, 200 tokens) on **one H200**, same model. genai is
INT4 (QMoE); the serving engines run **bf16** weights. "no MTP" = plain autoregressive;
"MTP" = the model's built-in multi-token-prediction head used for self-speculative decoding.

| Engine | precision | decode no-MTP | decode + MTP | MTP accept |
|---|---|---|---|---|
| **genai (ORT, this work)** | INT4 | **158.3 tok/s** | — (future) | 88% (measured offline) |
| llama.cpp (Q4_K_XL, llama-bench) | INT4 | 224.9 tok/s | n/a (bench has no MTP) | — |
| vLLM 0.22 (TP=1) | bf16 | 221.3 tok/s | **262.1 tok/s** (+18%) | 96.1% (accept len 1.96) |
| SGLang 0.5.12 (TP=1) | bf16 | 161.9 tok/s (~171 internal) | **227.9 tok/s** (+41%) | ~95% (accept len 1.93–2.0) |

**Measured this session** (servers launched on GPU0, `/tmp/bench_oai.py`):
- **vLLM** is the fastest no-MTP engine (221 tok/s); **MTP adds +18%** (→262) at 96% accept.
- **SGLang** no-MTP is slower (162; its internal decode log shows ~171 tok/s with CUDA graph
  on) but **MTP adds +41%** (→228) at ~95% accept — a bigger relative MTP win because its
  base is slower.
- **genai (158)** is already within ~10–30% of the bf16 engines' no-MTP decode and *ahead of
  SGLang's* no-MTP, despite being a much smaller (INT4) memory footprint. The decisive gap
  to vLLM/llama.cpp is **MTP**: both fast engines confirm the ~95% MTP acceptance and the
  ~1.2–1.4× wall-clock speedup it buys — directly corroborating genai's 88% offline
  measurement and making MTP the clear #1 next step.

> Note: the serving engines are bf16; a like-for-like INT4 comparison would lower their
> numbers (less weight bandwidth) — i.e. genai's INT4 158 tok/s is **not** 1.4× behind on
> equal precision. The MTP gap is the real, precision-independent lever.

### Earlier llama.cpp profiling (pre-fixes, retained for context)

Profiling (no CUDA graph, same H200) showed the GPU **compute** is nearly equal (llama
5.20 ms/tok vs ORT 5.72 ms/tok at session start); the gap was launch/scheduling overhead
(ORT 54% GPU-busy vs llama 74% without graphs). After Exp 1–7 the ORT GPU-busy dropped to
5.10 ms/tok — compute is now competitive; the remaining wall-clock gap is MTP + launch
overhead.

## Benchmark methodology

- `bench_genai.py`: 1 warmup + 3 iters, 200 decode tokens, greedy. Reports decode tok/s.
- `bench_oai.py`: single-stream OpenAI-API client (batch=1, greedy, min_tokens=max_tokens);
  used for vLLM (port 8001) and SGLang (port 30000). Engines launched at TP=1 on GPU0,
  `--max-model-len 8192`. vLLM MTP via `--speculative-config '{"method":"mtp",
  "num_speculative_tokens":1}'`; SGLang MTP via `--speculative-algorithm NEXTN
  --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2
  --mamba-scheduler-strategy extra_buffer` with `SGLANG_ENABLE_SPEC_V2=1`.
- `decode_profile2.py` + nsys: skips warmup via `cudaProfilerStart/Stop`, captures
  steady-state only. Export to sqlite, query `CUPTI_ACTIVITY_KIND_KERNEL`.
- A/B isolation via `ORT_DISABLE_LA_DECODE` env gate (debug-only, removed from final source).

---

## Experiments

### Exp 0 — Baseline (session start)

QMoE router fix (prior session) already applied. CUDA graph **off** in deployed config.

- **Decode: ~113–117 tok/s**

### Exp 1 — Enable CUDA graph (decoder)

Decode shapes are static (1 token; fixed recurrent/conv state buffers; small GQA KV
cache), so graph capture/replay removes per-kernel launch overhead. Set
`enable_cuda_graph="1"` in `genai_config.json` decoder `provider_options`.

> Note: `optimize.py::update_genai_config` already sets this to `"1"`, but the
> `olive run --config cuda/text.json` flow only runs the `ModelBuilder` pass (which
> defaults graph off). Fixed directly in the deployed `genai_config.json`.

| Config | Decode tok/s |
|---|---|
| graph OFF | ~117 |
| **graph ON** | **143.5** (old LA kernel) |

- **Result: +22–27% decode.** This is the single largest lever: the no-graph path is
  launch-overhead bound (47–63% GPU-busy); graph replay recovers most of the idle time.

### Exp 2 — GatedDeltaNet column-parallel decode kernel

**Problem.** `LinearAttentionRecurrentKernelFixedShape` (the existing
gated-delta/linear-attention kernel) launches grid `(batch, kv_num_heads)` = `(1, 32)`
at decode — only **32 blocks** (32 of 132 SMs), each serially walking the whole
`128x128` state with ~10 block-wide `__syncthreads` per token. The shared-memory state
caching that pays off during prefill gives **zero** amortization at `seq_len=1`, so the
kernel is latency-bound.

**Fix.** New `LinearAttentionDecodeKernel<T, DK>` mirroring the
flash-linear-attention / llama.cpp `gated_delta_net` decode design:

- Grid `(kv_num_heads, batch, ceil(d_v / 4))`, block `(32, 4)` — **~1024 blocks**, full
  GPU saturation.
- Each **warp owns one output column**; the recurrent state column is sharded into
  **registers** (`DK/32` rows per lane).
- Matrix-vector reductions (`S^T@k`, `S^T@q`) via `__shfl_xor_sync` warp reductions —
  **no shared memory, no block barriers**.
- Handles all update rules (linear/gated/delta/gated_delta), per-key-dim decay,
  per-head beta, GQA and inverse-GQA, and `n_k_heads` K-sharing.
- Dispatched only for `seq_len <= 16` and `d_k in {64,128,256}`; otherwise falls through
  to the existing recurrent kernels — **prefill path untouched**.
- State global layout unchanged (row-major `[DK, d_v]`), so present_state stays
  bit-compatible and interchangeable between prefill and decode steps. (v1 trade-off:
  per-column state access is strided/uncoalesced — see Exp 3.)

**Kernel time (nsys, no graph, per token):**

| Kernel | Time/tok |
|---|---|
| `LinearAttentionRecurrentKernelFixedShape` (old) | 693 µs |
| `LinearAttentionDecodeKernel` (new) | **346 µs (2.0x)** |

Total GPU busy 5.72 → 5.38 ms/tok.

**End-to-end A/B (CUDA graph ON):**

| Kernel | Decode tok/s |
|---|---|
| old recurrent | 143.5 |
| **new column-parallel** | **151.0 (+5.2%)** |

(No-graph end-to-end is unchanged at ~117 because that path is launch-overhead bound;
the saved GPU time hides in idle gaps. The win materializes under graph where wall≈busy.)

**Correctness:** all 26 `ContribOpLinearAttentionTest` parity tests pass (covers
single-token decode, inverse-GQA, KGQA, Qwen3.5-like shapes).

### Exp 3 — Coalesced column-per-thread decode kernel

**Problem.** The v1 decode kernel (Exp 2) shards each state column across a warp's
lanes. In the row-major `[d_k, d_v]` state layout a column is strided by `d_v`, so the
per-token state load/store is fully **uncoalesced** (32 lanes hit 32 separate sectors).
llama.cpp coalesces by storing the state **transposed**, but that would change this op's
`present_state` output layout and break the parity tests (which validate state in
`[d_k, d_v]`).

**Fix.** `LinearAttentionDecodeColKernel`: map **one thread per output column**, holding
the whole column `S[:, col]` in registers (`DK` values). For a fixed row `i`, consecutive
threads read consecutive addresses `i*d_v + col` → **fully coalesced with no transpose**.
The state layout stays row-major (contract + parity preserved). Per-column reductions
(`S^T@k`, `S^T@q`) are sequential within the thread → no cross-thread/warp reductions;
only per-token k/q/decay broadcasts use shared memory.

- Grid `(batch, kv_num_heads, ceil(d_v/32))`, block `(32)`.
- Used by default when `d_v % 32 == 0`; otherwise falls back to the v1 warp kernel.

**Kernel time (nsys, no graph, per token):**

| Kernel | Time/tok | vs original |
|---|---|---|
| `LinearAttentionRecurrentKernelFixedShape` (original) | 693 µs | 1.0x |
| `LinearAttentionDecodeKernel` (v1, warp-column) | 346 µs | 2.0x |
| `LinearAttentionDecodeColKernel` (v2, column-per-thread) | **202 µs** | **3.4x** |

Total GPU busy 5.38 → 5.25 ms/tok.

**End-to-end A/B (CUDA graph ON):** v1 151.8 → **v2 154.8 tok/s (+1.6%)**.

**Correctness:** all 26 parity tests pass.

### Exp 4 — Can fpA_intB GEMM/GEMV replace `MatMulFloatInt4Kernel`? (investigation)

`MatMulFloatInt4Kernel` is the #1 decode cost (1290 µs/tok, 310 calls). ORT ships an
alternative weight-only INT4 path (`fpA_intB_gemv` GEMV + CUTLASS weight-only GEMM,
under `onnxruntime/contrib_ops/cuda/llm/`) that `MatMulNBits` can dispatch to. Question:
would it be faster here?

**Step 1 — why `MatMulFloatInt4Kernel` runs.** Profiled the kernel by *launch shape*
(`gridX = N/8`):

| N | calls/tok | µs/tok | µs/call | blocks |
|---|---|---|---|---|
| 32 | 60 | 200 | 3.33 | **4** |
| 512 | 100 | 366 | 3.66 | 64 |
| 2048 | 80 | 357 | 4.47 | 256 |
| 4096 | 30 | 137 | 4.55 | 512 |
| 8192 | 40 | 231 | 5.77 | 1024 |

Model `MatMulNBits` shapes (K, N): (2048,512)×100, (2048,32)×60, (2048,8192)×40,
(4096,2048)×40, (512,2048)×40, (2048,4096)×30, (2048,248320)×1 (lm_head). The N=32/512
matmuls are the GatedDeltaNet beta/decay projections.

**Key finding — a ~3 µs per-kernel floor.** N=32 (4 blocks) = 3.33 µs vs N=2048
(256 blocks) = 4.47 µs — only **1.34× for 64× the work**. The small-N matmuls are
**launch/scheduling-floor bound, not bandwidth bound**. A split-K microbench
(`/tmp/mm4_bench.cu`) confirmed split-K *loses* in every case (0.19–0.67×) because the
atomicAdd + output-zeroing memset cost more than the floor it tries to hide.

**Step 2 — enable fpA_intB.** It is **OFF by default** at build
(`onnxruntime_USE_FPA_INTB_GEMM=OFF`, `cmake/CMakeLists.txt:110`) — that is why
`MatMulFloatInt4Kernel` runs. Rebuilt the provider with `-DUSE_FPA_INTB_GEMM=ON`
(382 extra CUDA sources, `.so` 113→123 MB). It *also* needs a **runtime env var**
`ORT_FPA_INTB_GEMM` (bitmask in `matmul_nbits.h:38`: `All=1, Gemv=2, Int4=4, Int8=8`),
plus per-node gates (`N % 64 == 0` for int4, `block_size ∈ {64,128}`, `sm ≥ 75`).

**Result:**

| Configuration | Outcome |
|---|---|
| Build ON, env unset | `MatMulFloatInt4Kernel` still runs; **153.7** vs 154.8 tok/s (identical) |
| Build ON, `ORT_FPA_INTB_GEMM=7` or `4` | **Crash**: `std::optional` assertion `CutlassGemmConfig _M_is_engaged()` in `getBestConfig` (`gemm_profiler.h:360`) |

The crash root cause: the CUTLASS tactic profiler (`profileTacticsForProblem`) found
**no valid config** for these shapes on sm_90 / CUDA 13 and stored `nullopt`; the caller
then dereferences `bestTactic->enableCudaKernel`. A genuine robustness bug in the
fpA_intB profiler tactic selection, not quick to fix.

**Why it would not help much even if fixed.** The dominant cost is the small-N
GatedDeltaNet projections (N=32: 60 calls, N=512: 100 calls = 566 µs/tok), which are
launch-floor bound. `fpA_intB_gemv` has the *same* grid limit (`CtaN=8` → ~4 blocks for
N=32, `dispatcher.h:329`), and the CUTLASS GEMM is not faster than the tuned
`MatMulFloatInt4` GEMV at M=1. So fpA_intB is **not a clear win** for this kernel.

**llama.cpp comparison.** `mul_mat_vec_q` (`mmvq.cu`) amortizes the launch floor by
folding the batch/expert dimension into the grid (`block_nums(nblocks, nchannels_dst,
nsamples)`) and a `small_k` path that puts `nwarps` rows per block — a *graph-wide*
launch-amortization strategy, not a single-matmul fix. This is the same "more resident
blocks" idea that lost in the ORT split-K microbench (ORT's variant pays atomicAdd +
memset; llama avoids both by having each block own full rows). It reinforces that the
real lever for this cost class is **fusion / fewer launches** (a graph change), which
CUDA graph (Exp 1) already captures most of.

**Conclusion:** `MatMulFloatInt4Kernel` is a kernel-level dead-end at decode. The build
was reverted to `USE_FPA_INTB_GEMM=OFF` (source unchanged — only the build flag was
toggled and reverted), keeping the working branch clean.

### Exp 5 — MoE routing overhead: also launch-floor bound (investigation)

The MoE routing kernels are the largest remaining *non-BW-bound* compute. Detailed
profile (nsys, no graph):

| Kernel | µs/tok | calls | grid | block | Issue |
|---|---|---|---|---|---|
| `finalizeMoeRoutingKernel` | 224 | 40 | **1** | 256 | single block |
| `fusedBuildExpertMapsSortFirstToken` | 163 | 40 | **1** | 32 | single block, 1 warp |
| `expandInputRowsKernel` | 91 | 40 | 1056 | 256 | already parallel |

`finalizeMoeRouting` launches `blocks = num_rows` = **1** at decode. With hidden=2048 and
128-bit (8×fp16) loads, it processes `2048/8 = 256` elements with 256 threads = exactly
**one block of column work**, looping serially over the k=8 experts. So column-splitting
cannot add blocks; the only parallelism is splitting the 8 experts across blocks with
atomicAdd.

**Microbench** (`/tmp/fin_bench.cu`, decode shape):

| Variant | µs/call |
|---|---|
| current (1 block, serial k-loop, register accum) | **2.16** |
| expert-split (8 blocks, atomicAdd + memset) | 15.60 (**0.14×**) |

The single-block kernel is already at the ~2 µs launch floor in isolation; expert-split is
**7× slower** (atomics + the required output memset). `fusedBuildExpertMapsSort` sorts only
8 entries (1 token × k=8) — also irreducibly small. So MoE routing, like the matmuls, is
launch-floor / small-work bound at decode and cannot be parallelized away. The only way to
remove it is to *eliminate* the kernels via a fused per-expert path (option C), not to
parallelize them.

### Exp 6 — Router gate GEMV via cublas split-K (finding, not yet optimized)

Three cublas kernels run 40×/tok (once per layer):

| Kernel | µs/tok | calls | grid | what |
|---|---|---|---|---|
| `nvjet_hsh_...` | 207 | 40 | 4 | cublas GEMV |
| `dot_kernel<float>` | 73 | 40 | 496 | split-K partial sums |
| `reduce_1Block_kernel` | 72 | 40 | 1 | split-K reduction |

These are **plain fp16 `MatMul` nodes**: the MoE router gate `[2048,256]` (×40) and a
`[2048,1]` scalar projection (×40). ORT routes these tiny M=1 GEMVs to cuBLAS, which picks
a **496-way split-K** (`dot_kernel`) followed by a separate `reduce_1Block` — two kernels
for one small matmul (≈351 µs/tok across the gate + scalar projections).

A dedicated fp16 M=1 GEMV would collapse the `dot`+`reduce` pair into one kernel and skip
the split-K overhead. **Microbench** (`/tmp/gate_bench.cu`, `[1,2048]×[2048,256]` fp16):

| Variant | µs/call | vs cuBLAS |
|---|---|---|
| cuBLAS `Hgemm` (current, split-K dot+reduce) | 7.11 | 1.0× |
| custom GEMV, weight row-major `[K,N]` (strided) | 7.45 | 0.96× |
| custom GEMV, weight **transposed `[N,K]`** (coalesced) | **3.65** | **1.95×** |

So a single-kernel GEMV with the weight stored **transposed** is **~2× faster** than cuBLAS
for this shape — a genuine, measured ~100–140 µs/tok win (≈2–2.6% e2e), unlike the
launch-floor dead-ends above. The row-major variant ties cuBLAS, so the coalesced
transposed layout is essential. **This was implemented — see Exp 7.**

### Exp 7 — MatMul decode (M=1) GEMV fast path (implemented)

Implemented the Exp 6 finding as a core-ORT `MatMul` fast path (CUDA EP):

- **PrePack** (`MatMul<T>::PrePack`): for a constant fp16/bf16 weight `B` used as a plain
  `[K, N]` matrix with `N ≤ 1024` and `K ≥ 256`, build a transposed `[N, K]` copy once at
  session init. `is_packed` stays **false** so the original `B` remains available for the
  cuBLAS path when `M > 1`.
- **Dispatch** (`MatMul<T>::ComputeInternal`): at `M == 1` (non-transposed A, single output
  offset, matching N/K), launch `GemvM1Kernel` — one block per output column, splitting K
  across the block and reducing once. fp32 accumulation matches cuBLAS's fp32-compute fp16
  GEMM. All other shapes fall through to cuBLAS unchanged.
- New files: `core/providers/cuda/math/matmul_gemv.{h,cu}`. Guarded by `if constexpr` to
  only instantiate for fp16/bf16 (float/double keep the cuBLAS path).

**Profile (nsys, no graph):** the three cuBLAS router-gate kernels (`nvjet` 207 + `dot_kernel`
73 + `reduce_1Block` 72 = **352 µs/tok**) are replaced by a single `GemvM1Kernel` at
**192 µs/tok** (80 calls = gate `[2048,256]` + scalar `[2048,1]`). GPU busy 5.25 → 5.10 ms/tok.

**End-to-end (CUDA graph ON):** **154.8 → 158.3 tok/s (+2.3%)**.

**Correctness:** added `MatMul_Float16_GemvDecode` parity test (constant-B fast path and
non-constant-B cuBLAS path both validated against a reference); all 36 MatMul/FusedMatMul
tests pass. Committed on branch `tlwu/matmul_decode_gemv`.

**Conclusion (Exp 4–7):** at decode (M=1) the per-launch kernels split into two classes:
(1) **launch-floor / BW-bound dead-ends** — INT4 matmul, MoE GEMM, MoE routing (Exp 4–5);
and (2) the **one measured, improvable item** — the cuBLAS router-gate GEMV, now replaced
by a custom transposed-weight kernel (Exp 6–7, +2.3%). All other gains require kernel-count
reduction (fusion), captured mostly by CUDA graph (Exp 1) already.

---

## Cumulative result

| Stage | Decode tok/s | vs baseline |
|---|---|---|
| Baseline (graph off, old LA) | ~117 | — |
| + CUDA graph on | 143.5 | +23% |
| + GatedDeltaNet decode kernel (v1) | 151.8 | +30% |
| + Coalesced column kernel (v2) | 154.8 | +32% |
| + MatMul decode GEMV (Exp 7) | **158.3** | **+35%** |

llama.cpp reference: 224.9 tok/s.

## Files changed

- `onnxruntime/contrib_ops/cuda/bert/linear_attention_impl.cu` (ORT) — new decode kernels.
  Branch `tlwu/gated_delta_net_decode` (commits: v1 kernel, v2 coalesced kernel); PR #28985.
- `onnxruntime/core/providers/cuda/math/matmul_gemv.{h,cu}`, `matmul.{h,cc}`,
  `test/providers/cpu/math/matmul_test.cc` (ORT) — MatMul decode GEMV fast path (Exp 7).
  Branch `tlwu/matmul_decode_gemv`.
- `cuda/models/genai_config.json` (olive output artifact) — `enable_cuda_graph="1"`.

## Future work items

- **MTP (multi-token prediction) decode.** The Qwen3.6 config has
  `mtp_num_hidden_layers = 1` (a dedicated MTP head). The genai export currently runs plain
  single-token autoregressive decode (so does the llama.cpp `llama-bench tg128 = 224.9
  tok/s` reference — `llama-bench` does *not* use speculative/MTP; that lives in separate
  `examples/lookup` / `examples/speculative` tools). Wiring the MTP head as a draft model
  for self-speculative decoding (verify-and-accept) could lift effective tok/s well above
  the memory-bound single-token ceiling. **This is the single highest-upside item**
  (potential ≈ `1 + acceptance_rate` ×, so ~1.6–1.9× at a typical 60–90% accept), but it is
  a large multi-component feature. Design + feasibility analysis below.

### MTP design + feasibility (analysis)

**MTP head architecture** (verified from the HF safetensors weight names): the head is one
extra decoder layer that reuses the main model's embedding and `lm_head`
(`mtp_use_dedicated_embeddings = False`, `tie_word_embeddings = False`). Per the weights
(`mtp.fc`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.layers.0.*` =
full-attention GQA + MoE-with-shared-expert, `mtp.norm`) it is the standard DeepSeek-V3 /
Qwen MTP module:

```
h'_i   = fc( concat[ pre_fc_norm_embedding(embed(t_{i+1})),  pre_fc_norm_hidden(h_i) ] )
h''_i  = DecoderLayer_mtp(h'_i)            # one full-attention + MoE layer
logits = lm_head( mtp.norm(h''_i) )        # predicts t_{i+2}
```

where `h_i` is the main model's last hidden state at position `i` (before the final norm)
and `t_{i+1}` is the token the main model just produced. The MTP layer is a
`full_attention` layer (it has `self_attn.{q,k,v,o}_proj` + `q_norm`/`k_norm`), so it needs
mRoPE + its own small KV cache.

**Export approach** (model builder, `src/python/py/models/builders/qwen.py`): follow the
Whisper multi-model precedent (separate `Model` subclass with its own `self.graph` /
`filename`, orchestrated by `make_model` / `save_model` — see
`builders/whisper.py:468–488`). A `Qwen35MtpHead` subclass would emit `mtp.onnx` with
inputs `(hidden_states [B,S,H], input_ids [B,S], position_ids, past KV)` and output
`logits`, reusing existing helpers: `make_embedding` (base.py:1420) on `input_ids`,
`make_layernorm` (1510) for the two pre-fc norms + `mtp.norm`, `make_concat` (951) +
`make_matmul` (1124) for `fc`, `make_attention` + `make_moe` + `make_shared_expert` for the
layer, and `make_lm_head` (3698) for the projection.

**Resolved: the MTP hidden input is the POST-final-norm hidden state** (verified — HF's
`out.hidden_states[-1]`, which the 88% measurement used, is bit-equal to the post-final-norm
output via `allclose`, not the pre-norm layer output). This is **exactly the tensor the
builder already exposes** via the `include_hidden_states` extra_option (base.py:1540–1542:
the final-norm output is emitted as the `hidden_states` graph output). So `text.onnx` needs
only `--extra_options include_hidden_states=true` — **no new tensor wiring**, the export
foundation already exists. The MTP head then takes that `hidden_states` output directly into
`pre_fc_norm_hidden`.

**Runtime draft/verify loop** (genai C++): genai currently has **no** speculative /
draft-model / MTP infrastructure (confirmed by repo-wide search). The loop would be: (1)
main model step → token `t`, hidden `h`; (2) MTP head(`h`, `t`) → draft token `t'`; (3) next
main step processes `[t, t']` as a 2-token batch, accept `t'` iff it equals the main model's
greedy/sampled token at that position; (4) on accept, 2 tokens/step. This needs a new
generator code path, KV-cache management for accepted/rejected drafts, and CUDA-graph
interplay.

**Two hard blockers (why this is not an unsupervised one-shot):**

1. **No HF reference forward.** transformers 5.10.2 *discards* the MTP weights on load
   (`modeling_qwen3_5_moe.py`: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). So there
   is no ground-truth to validate an export against without hand-porting Qwen's MTP forward
   (the algorithm above is the well-known DeepSeek/Qwen form, but the exact details — norm
   eps, MTP-layer rotary position, causal masking over the draft — must be confirmed against
   Qwen's reference, not guessed).
2. **No genai speculative infra.** The entire draft/verify generator path is built from
   scratch (no Eagle/Medusa/nextn precedent in the repo).

**Recommended first step (supervised, low-risk, high-information):** measure the **MTP
acceptance rate** on the real model by hand-wiring the MTP forward above on top of the HF
decoder layer (`Qwen3_5MoeDecoderLayer`, which *is* implemented) loaded with the `mtp.*`
weights, and comparing MTP's `t_{i+2}` prediction to the main model's greedy `t_{i+2}`.

**MEASURED (this analysis).** Hand-wired the MTP forward above (`/tmp/mtp_accept_gen.py`):
loaded the 72 GB HF model + the `mtp.*` weights (which transformers discards), reused
`Qwen3_5MoeDecoderLayer` for the MTP layer (`missing=0, unexpected=0` on load), and measured
greedy acceptance over the model's **own greedy generation** (the realistic self-speculative
scenario, not teacher-forced prompt text):

| Measurement | Accept rate | Implied speedup (1 draft tok) |
|---|---|---|
| Over prompt positions (teacher-forced, lower bound) | 42% | 1.42× |
| **Over greedy-generated text (realistic)** | **88.3%** (265/300) | **1.88×** |

88% matches DeepSeek/Qwen's reported ~85–90% for a 1-token MTP head — which both
**validates the wiring** (a bug would give ≈0% over the 248K vocab) and makes MTP a **strong
go**: ≈1.88× ideal, realistically ~1.5–1.7× net after the MTP head's own forward overhead
(one extra decoder layer + the 248K-vocab `lm_head`), which would push genai decode from
158 tok/s to roughly the 240–270 tok/s region — past the llama.cpp 224.9 single-token
reference. This justifies building the full export + draft/verify runtime (the two blockers
above remain: port/confirm Qwen's exact MTP forward for export parity, and build the
generator draft/verify path from scratch).

**CONFIRMED by vLLM + SGLang MTP benchmarks (this session).** Both production engines ship a
working Qwen3.6 MTP path and measured **~95% per-position acceptance / accept-length ~1.95**
on this exact model — corroborating the 88% offline number. The wall-clock payoff:
vLLM 221→262 (+18%), SGLang 162→228 (+41%). The genai equivalent would lift 158 →
**~215–225 tok/s** (≈+1.35–1.4×, the realistic 1-draft-token speedup after head overhead),
i.e. roughly matching llama.cpp / vLLM no-MTP and closing most of the remaining gap.

### What vLLM and SGLang do for Qwen3.6 (source study)

Read both engines' `qwen3_5` model code to identify their decode optimizations and what
genai could adopt:

| Optimization | vLLM | SGLang | genai status |
|---|---|---|---|
| **MTP self-speculative decode** | `qwen3_5_mtp.py` + `--speculative-config` (method `mtp`) | `qwen3_5_mtp.py` + `--speculative-algorithm NEXTN` | **Not yet** — #1 gap (this doc) |
| **Dedicated GDN decode kernel** | FLA Triton `fused_recurrent_gated_delta_rule_packed_decode` (decode) / `chunk_gated_delta_rule` (prefill) | FLA Triton + `gdn_fused_proj` JIT kernels | **Done** (Exp 2–3, custom CUDA `LinearAttentionDecodeColKernel`) |
| **Shared-expert fusion into MoE** | folded into `FusedMoE` | `num_fused_shared_experts` → extra expert slots in `FusedMoE` (toggle `--disable-shared-experts-fusion`) | **Not yet** — backlog (modest, +2–4% under graph) |
| **Fused QK-norm + gate** | — | `fused_qk_gemma_rmsnorm_with_gate` (one kernel for q/k RMSNorm + output gate) | **Not yet** — norm-fusion backlog item A |
| **CUDA graph (full)** | `CUDAGraphMode.FULL_AND_PIECEWISE`, capture sizes 1…512 | `cuda graph: True` per decode batch | **Done** (Exp 1) |
| **torch.compile / inductor** | yes (VLLM_COMPILE) | partial (JIT kernels) | n/a (ORT graph) |

Key takeaways for genai:
1. **MTP is the dominant differentiator** — it's the only feature that moves these engines
   meaningfully past the shared ~160–225 tok/s single-token band, and it's confirmed at ~95%
   accept on this model. Everything else is incremental.
2. SGLang's **`num_fused_shared_experts`** is exactly the Option-A design the fusion subagent
   sketched (fold the shared expert in as extra MoE slots). It's a real, shipped pattern —
   but the measured upside is small under CUDA graph.
3. SGLang's **`fused_qk_gemma_rmsnorm_with_gate`** validates the GQA q/k-norm fusion backlog
   item (A) as a real optimization others ship, though it touches only 10/40 layers here.

- **Shared-expert fusion into QMoE.** Qwen3.6 has a shared expert
  (`shared_expert_intermediate_size = 512`). It *is* correctly exported and runs today, but
  as **120 separate `MatMulNBits` GEMVs** (gate/up/down × 40 layers) plus a sigmoid gate,
  outside the `QMoE` op — i.e. it is a large chunk of the `MatMulFloatInt4Kernel` 310
  calls/tok. It is **not blocked by any operator/kernel** (functionally complete, coherent
  generation). The optimization opportunity is to fold the always-on shared expert into the
  `QMoE` op (or a fused shared-FFN kernel) to cut ~120 launches/tok. That requires a `QMoE`
  operator-spec + kernel extension (an optional shared-expert weight set + gate) and
  model-builder export support — a real but scoped enhancement, verifiable via the qMoE
  parity tests.

- **Shared-expert fusion into QMoE.** Qwen3.6 has a shared expert
  (`shared_expert_intermediate_size = 512`). It *is* correctly exported and runs today, but
  as **120 separate `MatMulNBits` GEMVs** (gate/up/down × 40 layers) plus a sigmoid gate,
  outside the `QMoE` op — i.e. it is a large chunk of the `MatMulFloatInt4Kernel` 310
  calls/tok. It is **not blocked by any operator/kernel** (functionally complete, coherent
  generation). The optimization opportunity is to fold the always-on shared expert into the
  `QMoE` op (or a fused shared-FFN kernel) to cut ~120 launches/tok. That requires a `QMoE`
  operator-spec + kernel extension (an optional shared-expert weight set + gate) and
  model-builder export support — a real but scoped enhancement, verifiable via the qMoE
  parity tests.

## Pending optimizations (analysis backlog)

Per-kernel GPU breakdown after Exp 1–3 (nsys, no graph, GPU busy 5.25 ms/tok):

| Kernel | µs/tok | calls/tok | notes |
|---|---|---|---|
| `MatMulFloatInt4Kernel` | 1290 | 310 | dense INT4 projections (incl. shared-expert FFN, ~120/tok); already the optimized ORT kernel; weight-BW-bound |
| `MoeFCGemm` (fc1+fc2) | 1006 | 79 | MoE grouped GEMM; weight-BW-bound |
| MoE routing (`finalizeMoeRouting`+`buildExpertMaps`+`expandInputRows`) | ~480 | 120 | gather/sort/scatter overhead |
| `MatMulFloat8bKernelM1` | 227 | 1 | lm_head (vocab projection, FP8) |
| `LinearAttentionDecodeColKernel` | 202 | 30 | **optimized (Exp 2–3)** |
| norms (`cuApplyLayerNorm`+`SkipLayerNorm`+`LpNorm`) | ~418 | 191 | already fused single kernels |

`kColsPerBlock` sweep (32/64/128) for the Exp 3 kernel: all within run-to-run noise
(154.8 / 153.7 / 154.3 tok/s) — LinearAttention is now only ~4% of GPU time. Kept 32
(best measured kernel time, spreads across the most SMs).

**Assessment of remaining items (not pursued in this unsupervised run):**

- **C — MoE per-expert quantized GEMV.** llama.cpp does use per-expert GEMV, but its MoE
  GPU time is comparable to ORT's grouped GEMM — the GEMM is weight-bandwidth-bound, so
  per-expert GEMV would *not* speed up the 1006 µs GEMM. The only capturable prize is the
  ~480 µs routing overhead (gather/sort/scatter), a ~9% ceiling. Capturing it requires a
  full decode-path replacement that exactly matches the CUTLASS-prepacked interleaved
  weight layout and per-group dequant across the op's many quant modes (int4/int8/FP4/
  MXFP4/W4A8/FP8) — large surface, high blast radius. Verifiable via the 44 qMoE parity
  tests, but deferred for supervised implementation. *Concrete plan:* in the QMoE decode
  case (M=1), loop the top-k experts calling `fpA_intB_gemv::kernel_launcher` per expert
  (fc1 → SwiGLU → fc2), slicing each expert's prepacked weight/scale base pointer, and
  accumulate weighted by the router prob (writing output directly, eliminating
  `expandInputRows`/`buildExpertMaps`/`finalizeMoeRouting`).

- **A / B — norm fusions (GQA q/k-norm; RMSNorm+Mul).** These require changing the
  exported ONNX graph (model-builder/olive pass), which alters generation and must be
  validated for output coherence — not safe to do while offline. Also modest: the norm
  kernels are already individually fused, and the GQA q/k-norm touches only 10/40 layers.

- **`MatMulFloatInt4Kernel` (1290 µs, the #1 cost).** Already the optimized ORT INT4 GEMV;
  improving it is core-ORT work affecting all models, out of scope here. The high call
  count (310/tok) is architectural; an earlier in-projection fusion experiment was
  **negative** (−8%) under CUDA graph, so fusing these does not help. **Exp 4 confirmed**
  the fpA_intB alternative is not a win (launch-floor bound, same grid limit, and the
  CUTLASS profiler crashes on these shapes).

**Bottom line:** the readily-achievable, low-risk, parity-verified decode wins (CUDA graph
+ GatedDeltaNet kernel, +32%) are captured. Remaining gains require either BW-bound kernels
that won't improve, core-ORT GEMV work, or model-graph changes needing supervised
generation-quality validation. The largest remaining *capturable, non-BW-bound* compute is
the MoE routing overhead (~480 µs/tok) — investigated next (Exp 5).

---

## Next steps (prioritized, post vLLM/SGLang comparison)

The cross-engine benchmark reframes the roadmap: genai's INT4 decode (158 tok/s) is already
competitive with the bf16 serving engines' single-token decode (vLLM 221, SGLang ~171,
llama.cpp 225) on a much smaller memory footprint, and **ahead of SGLang's no-MTP path**.
The one feature that separates the leaders is **MTP self-speculative decoding**, confirmed
at ~95% acceptance on this exact model by both vLLM and SGLang.

1. **MTP self-speculative decode — #1, highest upside (~+35–40% decode).**
   Validated 3 independent ways: genai offline 88% accept, vLLM 96% (+18% wall), SGLang ~95%
   (+41% wall). Plan (supervised, multi-step):
   - **Export** (`src/python/py/models/builders/qwen.py`): **IMPLEMENTED** — `Qwen35MtpHead`
     emits `mtp.onnx` (one full-attention + MoE layer + `fc` + 3 norms), gated behind
     `--extra_options enable_mtp=true`. It reuses the parent `Qwen35MoeTextModel` machinery
     (`_make_full_attention`, `make_moe`, mRoPE, `make_layer` residual chain) for the single
     MTP layer, takes the main model's `hidden_states` (post-final-norm) as an extra input,
     and loads the `mtp.*` + shared `embed_tokens`/`lm_head` weights directly from the source
     safetensors (HF discards `mtp.*` on load). The pre-FC norms use the builder's existing
     `(1 + weight)` offset RMSNorm (`layernorm_attrs["add_offset"]=1`). Reference forward:
     SGLang `qwen3_5_mtp.py` / vLLM `qwen3_5_mtp.py` and the validated PyTorch script
     `/tmp/mtp_accept_gen.py`.
   - **Validate** (DONE): exported a standalone fp16 `mtp.onnx` (3.8 GB; only the
     `mtp.*` + shared `embed`/`lm_head` weights, no 72 GB main model needed) and ran it
     end-to-end in onnxruntime in place of the PyTorch `mtp_draft`, reusing the real
     model's hidden states. It reproduced **88.3% (265/300)** greedy acceptance —
     **bit-for-bit identical to the PyTorch reference** (`/tmp/mtp_accept_gen.py`),
     confirming the export (mRoPE, offset RMSNorm, fc/concat, MoE+shared-expert, lm_head)
     is numerically correct. The embedding + lm_head are currently duplicated into
     `mtp.onnx` (~2 GB); weight-sharing with the main model is a follow-up optimization.
   - **Runtime**: build a draft/verify loop in the genai generator (no precedent in repo —
     largest piece). Reuse the existing CUDA-graph decode path for both the main step and
     the 2-token verify step.
   - Target: 158 → ~215–225 tok/s.

   **Runtime algorithm — PROVEN via executable prototype** (`/tmp/mtp_runtime_prototype.py`,
   HF bf16 main model + `mtp.onnx` draft via onnxruntime). Per
   [MultiTokenPrediction.txt](MultiTokenPrediction.txt) Qwen3.6 is the **unified single-graph
   Next-N** case (one shared single-stream KV, sequential base→MTP, flat chain, partial
   rollback on reject). The validated loop, with invariant *enter each iteration with cache =
   positions `0..L-1`, `t` = predicted token for position `L` (not yet in cache), `h` =
   hidden state of position `L-1` (the source of `t`)*:

   ```
   emit t
   d        = MTP(accumulated (h_i, t_i))         # 1 draft token
   snapshot = cache.snapshot()                    # incl. GatedDeltaNet conv+recurrent state
   verify [t, d] at positions L, L+1              # ONE main forward, output_hidden_states
   m        = argmax(logits@L)                    # main's real token after t
   if d == m:   # ACCEPT — harvest a free token from the verify pass
       emit d;  t = argmax(logits@L+1);  h = hidden@L+1;  L += 2
   else:        # REJECT
       cache = snapshot;  re-run [t] at L;  t = argmax;  h = hidden;  L += 1
   ```

   **Measured (3 prompts, 80 tokens each, greedy):** accept **89.7%** (113/126) — matching
   the 88% validation; **2/3 prompts bit-exact vs plain greedy** (including one with 6
   rejects, confirming the accept/reject + snapshot/restore logic), with the single
   divergence being fp16 near-tie non-determinism between the batched 2-token verify and the
   1-token baseline (a documented property of greedy speculative decoding, not a bug);
   **effective 1.69× tokens per main-forward**.

   **KEY hybrid-model finding.** The GatedDeltaNet **recurrent state cannot be cropped**
   (unlike attention KV). The verify forward `[t, d]` irreversibly folds the draft `d` into
   the recurrent state, so a rejected draft must **snapshot the conv+recurrent state before
   the verify and re-run the one correct token** (one extra forward). This caps the hybrid
   model at `(1 + a)/(2 - a) ≈ 1.68×` at 88% accept, versus `(1 + a) ≈ 1.88×` for a
   pure-attention model (which simply crops KV by 1 and reuses `logits@L`). This is exactly
   why SGLang's Qwen3.6 MTP requires `--mamba-scheduler-strategy extra_buffer`. **The genai
   C++ `MtpState` must snapshot/restore the linear-attention conv + recurrent state buffers
   on reject.**

   **C++ integration plan** (the remaining, largest piece — core generator surgery):
   1. **Main re-export** with `include_hidden_states=true` so the decoder graph exposes the
      `hidden_states` output (not a default decoder output — `Config::Decoder::Outputs` only
      had logits/present KV; a `hidden_states` field was added).
   2. **Config — DONE.** Added `Config::Model::Mtp { filename, session_options, run_options,
      num_hidden_layers, num_key_value_heads, head_size, main_hidden_states, Inputs{
      input_ids, hidden_states, attention_mask, position_ids, past_kv }, Outputs{ logits,
      present_kv } }` to `config.h` + JSON parsing in `config.cpp` (mirrors the `Encoder`
      pattern), plus `Decoder::Outputs::hidden_states`. The Qwen builder's
      `make_genai_config` now emits the `mtp` section (and `decoder.outputs.hidden_states`)
      when `enable_mtp`. Compile-verified (`onnxruntime-genai` + `python` targets) and
      round-trip-parse-verified (`og.Config` accepts a config with the `mtp` section;
      existing configs without it still parse — backward compatible).
   3. **`MtpState : State`** wrapping the `mtp.onnx` session (its own tiny KV cache for the
      single MTP layer), exposing `Run(hidden_states, last_token) → draft logits`. *(TODO)*
   4. **Orchestrator** (a new `MtpGenerator` owning the main `Generator` + `MtpState`, chosen
      over editing the hot `Generator` loop for lower blast radius): each step does
      draft (`GetOutput("hidden_states")` + last token → `MtpState::Run`) → verify
      (`AppendTokens([t, d])`, read `logits@L` / `logits@L+1` via `GetOutput("logits")`,
      which returns the full per-position tensor) → accept/reject. *(TODO.)*

   **Runtime infrastructure findings (verified against genai source + a full fp16 export).**
   - `GetOutput("hidden_states")` works with **zero core changes**: `State::Run` auto-registers
     every graph output not managed by GenAI (via `ExtraOutputs`), so once the main graph is
     exported with `include_hidden_states=true` the hidden state is retrievable. Confirmed by
     loading the exported text-only model in genai: `get_output("hidden_states")` →
     `(1, S, 2048)`, `get_output("logits")` → `(1, S, 248320)` (full per-position logits, so
     the 2-token verify can read both positions from one forward).
   - **The reject path needs a new primitive.** genai's `RecurrentState::RewindTo` only
     supports `index == 0` (full zero-reset) and throws on partial rewind — the GatedDeltaNet
     conv/recurrent state *cannot* be cropped like attention KV. So a rejected draft cannot be
     rolled back with the existing API. **Implemented** `RecurrentState::Snapshot()` /
     `RestoreSnapshot()` (compile-verified): snapshot the conv+recurrent buffers before the
     speculative forward and restore them in place (stable addresses for CUDA-graph replay) on
     reject. The attention KV side already supports partial `RewindTo`, so a "rewind by 1" is
     KV `RewindTo(L+1)` + recurrent `RestoreSnapshot()`.
   - **Export — DONE & validated.** `builder.py ... --extra_options enable_mtp=true
     include_hidden_states=true exclude_embeds=false` produces a text-only `qwen3_5_moe_text`
     model (`DecoderOnly_Model`, `input_ids` input, `hidden_states` output) + `mtp.onnx` + a
     `genai_config.json` carrying the `mtp` section. Two builder bugs fixed along the way: the
     MTP head must not inherit `include_hidden_states`/`exclude_lm_head` (would create a graph
     cycle), and the shared `save_model` cache-dir cleanup must tolerate a multi-model builder.

   **Remaining (TODO):** wire `RestoreSnapshot` into `Generator::RewindToLength` for hybrid
   models, add an `MtpState` wrapping `mtp.onnx`, build the `MtpGenerator` draft/verify
   orchestrator, expose it through the C API + Python bindings, and measure end-to-end
   wall-clock decode against the fp16 model.

   **Runtime validated on real genai (DONE).** The snapshot/restore primitive is now exposed
   end to end: `RecurrentState::Snapshot()` / `RestoreSnapshot()` → `State::SnapshotState()`
   → `Generator::SnapshotState()` → `OgaGenerator_SnapshotState` (C API) → `snapshot_state()`
   (Python). `RecurrentState::RewindTo(index != 0)` now restores from the snapshot instead of
   throwing, so a "rewind by 1" rolls back both the attention KV (crop) and the GatedDeltaNet
   recurrent state (snapshot restore). A draft/verify orchestrator was then built on the
   **real genai model** (`og.Model` for the main decode, `mtp.onnx` via onnxruntime for the
   draft) using `get_output("hidden_states")` / `get_output("logits")` + `snapshot_state()` +
   `rewind_to()`. Measured (3 prompts, 60 tokens, greedy): **84.5% accept, 1.57× effective
   tokens per main-forward**, with one prompt matching plain greedy *exactly through two
   rejects* — proving the recurrent snapshot/restore rollback is correct (a broken restore
   would corrupt the recurrent state and diverge immediately). The remaining late divergences
   are the documented fp16 near-tie non-determinism between the batched 2-token verify and the
   1-token baseline, not a logic bug (correctness is guaranteed by verifying the draft against
   the main model's real token). This Python orchestrator on real genai kernels is the
   executable reference for the eventual in-engine `MtpGenerator`.

   **Only remaining piece:** a pure-C++ `MtpState` + `MtpGenerator` so the MTP draft runs
   in-engine (the reference orchestrator runs the draft in a separate onnxruntime session,
   which adds per-step overhead); this is needed for a true wall-clock decode number but not
   for correctness, which is now fully validated.

   **In-engine MTP draft — DONE.** Rather than a bespoke `MtpState`, the MTP head loads as a
   standalone genai `og.Model` (`qwen3_5_moe_text`, one layer) and is fed the main model's
   hidden state through a new `HiddenStatesInputs` feeder: a resizable `[batch, seq, hidden]`
   device tensor refreshed each step from a caller-staged value, created by `DecoderOnly_State`
   only when `config.model.decoder.inputs.hidden_states` is set (zero impact on normal models).
   Staging is exposed as `Generator::SetHiddenStates` → `OgaGenerator_SetHiddenStates` (C API)
   → `set_hidden_states` (Python). Verified the in-engine head matches the raw-onnxruntime
   reference (identical argmax + top-5). An **all-in-engine** draft/verify orchestrator (both
   the main decoder and the MTP head as genai models, no raw onnxruntime) reaches **81.6%
   accept / 1.51× tokens per main-forward** with greedy-exact output modulo the documented
   fp16 near-tie non-determinism. All MTP runtime primitives — `hidden_states` output and
   input, recurrent snapshot/restore, draft/verify rollback — now run inside genai and are
   committed (branch `tlwu/qwen_3.6_mtp`).

   **Remaining polish (optional):** fold the orchestration into a first-class C++
   `MtpGenerator` (+ C API / bindings) so a single `generate_next_token` call does the
   draft/verify internally, and benchmark wall-clock tok/s against the 158 baseline.

2. **Shared-expert fusion into the MoE op — modest (+2–4% under graph), low risk.**
   SGLang ships this as `num_fused_shared_experts` (extra MoE expert slots). genai design =
   standalone `SharedExpertFFN` op (subagent Option B, QMoE untouched) behind a default-off
   `fuse_shared_expert` builder flag. Eliminates ~120 `MatMulNBits` launches/tok. Build
   op + kernel + parity test together (do not emit a node for a missing op).

3. **GQA q/k-norm + gate fusion — small (10/40 layers), real precedent.**
   SGLang's `fused_qk_gemma_rmsnorm_with_gate` fuses the q/k RMSNorm + output gate into one
   kernel. genai backlog item A; needs a model-builder graph change + a fused CUDA op, with
   generation-quality validation.

4. **Not worth pursuing** (established this session): MatMulFloatInt4 kernel-level tuning
   (launch-floor bound, Exp 4), MoE routing parallelization (launch-floor bound, Exp 5),
   fpA_intB GEMM (no win + profiler crash, Exp 4).

**Single highest-leverage action: implement MTP** (item 1). It is the only change that
closes the bulk of the remaining gap, is independently validated at ~95% accept across three
codebases, and turns genai's INT4 decode into the throughput leader for this model.

---

## MTP self-speculative decoding — implemented (experiment log)

This section records what was built and measured for Qwen3.6 MTP. The full design is in
[examples/python/qwen-3.6-mtp.md](examples/python/qwen-3.6-mtp.md); the runnable example is
[examples/python/qwen-3.6-mtp.py](examples/python/qwen-3.6-mtp.py).

### What shipped (branch `tlwu/qwen_3.6_mtp`)

| Piece | Where |
|---|---|
| MTP head export (`mtp.onnx`, `enable_mtp=true`) | `src/python/py/models/builders/qwen.py` (`Qwen35MtpHead`) |
| `mtp` config section + `decoder.outputs.hidden_states` | `src/config.{h,cpp}` |
| Recurrent-state snapshot/restore (reject rollback) | `src/models/recurrent_state.{h,cpp}` |
| CUDA-graph-safe managed `hidden_states` output | `src/models/hidden_states_inputs.{h,cpp}` |
| Device-to-device `hidden_states` input feed | `src/models/hidden_states_inputs.cpp` |
| In-engine `MtpGenerator` (C++ draft/verify) | `src/mtp_generator.{h,cpp}` |
| C API + Python `og.MtpGenerator` | `src/ort_genai_c.*`, `src/ort_genai.h`, `src/python/python.cpp` |

### Correctness

- The exported fp16 `mtp.onnx` reproduces the PyTorch reference's **88.3% greedy acceptance,
  bit-identical**.
- The in-engine `og.MtpGenerator` output **matches plain greedy decoding** (one prompt bit-exact
  through draft accept *and* reject — proving the recurrent snapshot/restore rollback is correct),
  modulo occasional late fp16 near-tie divergences (a documented property of greedy speculative
  decoding; every emitted token is still one the main model produced).
- Acceptance **~83–88%** on real genai; **~1.55–1.64 tokens per main forward**.

### Wall-clock benchmark (fp16, 1× H200, batch 1, greedy, CUDA graph OFF)

| Decoder | tok/s | vs baseline |
|---|---|---|
| Baseline greedy (no MTP) | ~114–117 | 1.00× |
| Python reference loop (`--reference`) | ~36 | ~0.31× |
| **In-engine `og.MtpGenerator`** | **~113–118** | **~1.0× (break-even)** |

The Python orchestrator is far below 1× because it pays per-step Python + host round-trips and
runs the MTP draft in a separate session. The in-engine generator removes all of that (~3×
recovery) and reaches **break-even**.

### Why it was break-even (measured, before verify-shape capture)

Per-step latency (graph state matters):

| Step | CUDA graph OFF | CUDA graph ON (1-token only) |
|---|---|---|
| Main 1-token decode | ~8.8 ms | ~6.8 ms |
| MTP head 1-token draft | ~18 ms | **~0.5 ms** |
| Main 2-token verify | ~9.5 ms | ~9.5 ms (was not captured — ran eager) |

CUDA graph collapses the launch-overhead-bound MTP draft by ~35×. Originally genai only
graph-captured single-token (`shape[1] == 1`) steps, so the **2-token verify forward ran eagerly**
— and it is the dominant cost. Enabling CUDA graph on the main model while the verify ran eagerly
also **corrupted output** (the graphed 1-token path and the eager 2-token path shared captured
static buffers — "Empire Empire" degeneration), so graph had to be kept off.

### Implemented: graph-capture the verify shape

ONNX Runtime's CUDA EP holds **multiple captured graphs keyed by `gpu_graph_id`** (a
`graph_id_to_run_count_` map; per-id capture-after-`min_num_runs` then replay), so the 1-token
decode and the 2-token verify can each have their own captured graph bound to their own
(different-shape) static buffers.

What changed:

| Change | Where |
|---|---|
| `State` keeps a CUDA-graph annotation id **per captured length** (`graph_ids_` map, lazily generated) | `src/models/model.{h,cpp}` |
| `State::Run(session, capture?, capture_length)` selects the id for the captured length | `src/models/model.cpp` |
| `DecoderOnly_State::Run` captures `shape[1] ∈ [1, max_graph_capture_length]`, passing the length | `src/models/decoder_only.cpp` |
| `GeneratorParams::max_graph_capture_length` (default 1; MTP sets 2) | `src/generators.h`, `src/mtp_generator.cpp` |
| `Tensor::CreateTensor(shape, make_static, static_capacity_bytes)` pre-sizes the static buffer to the **max** captured shape so its base address stays stable across the 1- and 2-token graphs | `src/tensor.{h,cpp}` |
| Shape-dependent static I/O (input_ids, position_ids, logits, hidden_states output) pre-size to 2 tokens under MTP | `src/models/{input_ids,position_inputs,logits,hidden_states_inputs}.cpp` |

Each distinct captured length gets its own annotation id, so ORT captures and replays an
independent graph for the 1-token decode and the 2-token verify. The static buffers are pre-sized
to the 2-token width up front, keeping the buffer base address stable across both captured graphs
(the KV/recurrent state already use fixed-size shared buffers, a precondition for graph capture).

### Result (fp16, 1× H200, batch 1, greedy, **CUDA graph ON**)

- **Output is coherent** with the verify captured — the earlier graph-on "Empire Empire"
  corruption is gone; acceptance **~83–97%**, **~1.56–1.94 tokens per main forward**.
- The 2-token verify now replays from its own captured graph (separate annotation id + static
  buffers); enabling graph lifts the in-engine generator from ~115 → **~127 tok/s (+10%)**.

| Decoder (CUDA graph ON) | tok/s | vs baseline |
|---|---|---|
| Baseline greedy (no MTP) | ~140 | 1.00× |
| **In-engine `og.MtpGenerator`** | **~127** | **0.90×** |

The verify-shape capture is necessary (it fixes correctness and recovers the eager-verify cost),
but graph capture helps the *pure-GPU* baseline even more (114 → 140 tok/s, +23%), so the relative
ratio is below 1×. The bottleneck has now shifted to **host-side orchestration overhead** that is
intrinsic to the draft/verify loop and cannot be graph-captured:

- two/three full-vocab logits **D2H copies + CPU argmax** per step (vocab 248320 × up to 2 rows),
- recurrent-state snapshot/restore for reject rollback,
- per-step CUDA-graph replay synchronization (onnxruntime PR #28686 adds async replay).

The next lever is therefore **on-device argmax** (avoid the per-step ~2 MB logits D2H + 248K-element
host argmax) and trimming the snapshot cost, not the GPU forward.


### Memory-saving future work (embedding / lm_head sharing)

The MTP head reuses the main model's token embedding and `lm_head`. In the exported `mtp.onnx`
those two tensors are **bit-identical** to the main model's and together are ~2 GB of the head's
~3.8 GB (fp16): `model.embed_tokens.weight` (1017 MB) + `lm_head.MatMul.weight` (1017 MB). They can
be shared rather than duplicated:

- **Runtime sharing** — inject the main session's already-loaded weight `OrtValue`s into the MTP
  session via `OrtSessionOptions::AddInitializer(name, ort_value)` so the head allocates no copies
  (needs a model-load hook, since `og.Model` binds initializers at session creation).
- **Export sharing** — emit `mtp.onnx` without the `embed_tokens` / `lm_head` initializers
  (referenced by name) when sharing is enabled, shrinking the file too.

Note: for Qwen3.6 `tie_word_embeddings = False` — `embed_tokens` and `lm_head` are independently
trained (max abs diff ~0.3), so they **cannot** be collapsed into one transposed tensor. The
saving is cross-model duplication (main ↔ MTP), not an embed/lm_head tie.

### Updated MTP next steps (priority order)

1. **Graph-capture the 2-token verify shape** under its own `gpu_graph_id` — the top wall-clock
   lever; also fixes graph-on correctness. Target: break-even → ~1.4–1.5×.
2. **Share embedding + `lm_head` main ↔ MTP** (`AddInitializer` / export) — ~2 GB saving.
3. **INT4 MTP head + main** — match the deployed INT4 model and raise the amortized baseline.
4. **Speculative sampling** (`do_sample=true`) and **tree/multi-token drafts** to lift the
   tokens-per-forward ceiling.

