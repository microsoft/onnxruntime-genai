---
name: debugging-cuda-multimodal
description: >
  Debug and fix CUDA crashes, segfaults, and incorrect behavior in
  multimodal ONNX models running on ORT GenAI with CUDAExecutionProvider.
  Covers all issues discovered during Gemma4 multimodal CUDA enablement
  on H200 GPUs: missing CUDA kernel registrations, stale OrtValue pointers,
  CUTLASS alignment crashes, GQA shared-memory limits, KV-shared attention
  failures, f32/f16 input mismatches, TopK kernel crashes with large vocabs,
  memory pattern shape mismatches, and genai_config session requirements.
  Includes systematic debugging methodology with GDB, compute-sanitizer,
  ORT logging, and GPU memory monitoring.
---

# Skill: Debugging CUDA Issues in Multimodal ORT GenAI Models

## When to use

Use this skill when:

- A multimodal model segfaults or crashes only under CUDA EP (works on CPU)
- ORT inserts unexpected `MemcpyFromHost`/`MemcpyToHost` causing crashes
- CUTLASS FMHA kernels crash with specific head dimensions or sequence lengths
- The model works for prefill but crashes on decode (or vice versa)
- GenAI TopK sampling crashes with large vocabulary sizes
- You see `cudaErrorInvalidValue`, `cudaErrorIllegalAddress`, or error 717
- Stale GPU pointers cause silent corruption or segfaults between runs
- A model produces correct output on CPU but wrong output or crashes on CUDA

## Quick diagnostic flow

```
1. Does it crash or produce wrong output?
   ├─ Crash → segfault? CUDA error? Which kernel?
   │   ├─ Segfault in ORT op → Check kernel registration (§1) or input device (§6)
   │   ├─ CUDA illegal address → Check CUTLASS alignment (§3) or KV-shared (§5)
   │   ├─ cudaErrorInvalidValue → Check shared memory limits (§4)
   │   └─ Crash on 2nd run → Check stale OrtValue (§2) or MemPattern (§8)
   └─ Wrong output → Check f32/f16 mismatch (§6) or genai_config (§9)
```

## Issue catalog

All issues below were discovered and fixed during Gemma4 multimodal
CUDA enablement on H200 GPUs. Each entry includes root cause, symptoms,
diagnosis steps, and the fix.

---

### §1. Equal opset 19 — missing CUDA kernel registration

**Symptom:** Segfault in `Equal` op during model execution on CUDA EP.
GDB backtrace shows the crash inside a CPU kernel reading a CUDA device
pointer.

**Root cause:** ONNX bumped the `Equal` op from opset 13 to opset 19.
ORT's CUDA EP only had a kernel registered for opset 13. When the model
uses opset 19 `Equal`, ORT falls the node back to CPU. However, GenAI
allocates `input_ids` on CUDA (`p_device_inputs_ = p_device_`), so the
CPU `Equal` kernel dereferences a CUDA pointer → segfault.

**Why it's subtle:** The op doesn't fail at graph partitioning — ORT
silently falls it to CPU. There's no warning unless you check verbose
logs for node placement. The crash looks like a random memory corruption.

**Diagnosis:**

```bash
# 1. Check ORT logs for CPU fallback
# Set log_severity_level=0 and grep for Equal placement
grep -i "Equal.*placed on.*CPU" ort_verbose.log
grep -i "kernel not found.*Equal" ort_verbose.log

# 2. GDB to confirm the crash site
gdb -batch -ex "run" -ex "bt 10" --args python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])
# ... run inference
"
```

**Fix (in ORT):**

File: `onnxruntime/core/providers/cuda/math/binary_elementwise_ops.cc`
```cpp
// Add opset 19 registration for Equal
BINARY_LOGICALOP_REGISTER_UZILHFD(Equal, 19)
```

File: `onnxruntime/core/providers/cuda/cuda_execution_provider.cc`
```cpp
// Version Equal: change @13 to @13-18, and add @19
// Before:
//   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, ...)>
// After:
//   BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, 18, ...)>
//   BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 19, ...)>
```

**General pattern:** Whenever a new ONNX opset bumps an op version, check
that ORT's CUDA EP has a kernel registered for the new version. If not,
the op silently falls to CPU and any GPU-resident inputs cause a segfault.
Search for the op in `cuda_execution_provider.cc` and check the version
ranges.

---

### §2. ClearOutput for stale OrtValue between runs

**Symptom:** Model works on first run (prefill) but crashes or produces
garbage on second run (decode). Segfault or CUDA illegal address in the
embedding output tensor.

**Root cause:** GenAI's `ExtraOutputs::Update()` only cleared outputs
after `extra_outputs_start_`. The embedding model's output at index 0
was never cleared between runs. After the first run, the `OrtValue` at
index 0 held a GPU pointer to a buffer that ORT may have freed or
reallocated. The second run read the stale pointer.

**Diagnosis:**

```bash
# Crash on second inference call but not first
# GDB shows crash in ORT memory access during Run()
# The output OrtValue from a previous Run() still points to freed memory
```

**Fix (in GenAI):**

File: `src/models/embeddings.h`
```cpp
// Before each Run(), null out the output to prevent stale pointer reuse
output_value_.ClearOutput();  // Add before session_.Run()
```

File: `src/models/multi_modal.cpp`
```cpp
// Same pattern: clear outputs before each Run() call
// Ensures ORT allocates fresh buffers for each execution
```

**General pattern:** In GenAI, any `OrtValue` that persists across
`Session::Run()` calls must be explicitly cleared before the next run.
ORT may free or reallocate GPU buffers between runs, making old pointers
invalid. Always `ClearOutput()` before `Run()`.

---

### §3. CUTLASS BiasLoader alignment crash

**Symptom:** CUDA illegal address access in CUTLASS FMHA (fused
multi-head attention) kernel. Crash is intermittent and depends on
sequence length and head dimensions.

**Root cause:** The CUTLASS FMHA `BiasLoader` used a hardcoded 128-bit
(8-element for f16) vector load width regardless of the kernel's actual
alignment. When the unaligned kernel path was selected (`kAlignmentA=4`),
the `BiasLoader` still performed 8-element loads, reading past the end
of the bias buffer or from misaligned addresses.

**Diagnosis:**

```bash
# Use compute-sanitizer to identify the exact illegal access
compute-sanitizer --tool memcheck python3 run_model.py
# Look for: "Invalid __global__ read" in the FMHA kernel

# Check if crash correlates with specific sequence lengths
# Unaligned lengths (not divisible by 8) are more likely to crash
```

**Fix (in ORT):**

File: `onnxruntime/contrib_ops/cuda/bert/cutlass_fmha/kernel_forward.h`
```cpp
// Before: hardcoded 8-element vector load
// static constexpr int kLoadCount = 8;

// After: use the kernel's alignment parameter
// static constexpr int kLoadCount = kAlignmentA;
```

The `BiasLoader`'s vector width must match the kernel's alignment
parameter (`kAlignmentA`), not a hardcoded value. When `kAlignmentA=4`,
loads should be 4 elements wide.

**General pattern:** CUTLASS kernels have multiple template
instantiations for different alignment levels. Any loader or iterator
that hardcodes a vector width instead of deriving it from the alignment
template parameter will crash on the unaligned paths.

---

### §4. GQA CUTLASS FMHA shared memory overflow (head_dim > 256)

**Symptom:** `cudaErrorInvalidValue` when launching GQA attention for
layers with `head_dim > 256`. The error occurs at kernel launch, not
during execution.

**Root cause:** The GQA CUTLASS FMHA kernel's shared memory requirement
scales with `queries_per_block × head_dim`. When `head_dim > 256`
(e.g., `head_dim=512` in some Gemma4 layers), the required shared memory
exceeds the SM's maximum (typically 48KB or 96KB depending on GPU arch).
`cudaErrorInvalidValue` is returned by the kernel launch.

**Behavior:** ORT gracefully falls back to unfused (loop-based) attention
when the FMHA launch fails. The unfused path produces correct results but
is slower.

**Diagnosis:**

```bash
# Check ORT verbose logs for FMHA fallback
grep -i "fmha\|fused.*attention\|unfused" ort_verbose.log

# Calculate shared memory requirement:
# shared_bytes ≈ queries_per_block × head_dim × sizeof(element)
# If shared_bytes > 48KB (or 96KB with opt-in), kernel will fail
```

**Status:** ORT PR #28374 was filed but closed — the guard condition
didn't match the real failure mode. The current behavior (graceful
fallback to unfused attention) is acceptable for correctness. Performance
optimization for large head_dim remains open.

**General pattern:** When adding models with unusually large head
dimensions, expect CUTLASS FMHA to fail and verify that the unfused
fallback path works correctly. Don't assume all attention layers use
the same head_dim — some architectures (e.g., Gemma4) use different
head dimensions for different layer types.

---

### §5. KV-shared Attention — CUTLASS aligned kernel crash

**Symptom:** CUDA illegal address in GQA attention kernel for KV-shared
layers. These are layers where all heads share the same K and V
(effectively `num_kv_heads=1` with full KV reuse).

**Root cause:** KV-shared layers pass the full KV cache as K
(`past_key=nullptr`, K comes directly from the input). The CUTLASS
aligned kernel path requires `total_kv % 8 == 0`. When the total KV
length doesn't satisfy this alignment, the aligned kernel crashes.
The unaligned kernel path runs but produces numerically divergent results
compared to the reference implementation.

**Diagnosis:**

```bash
# Identify KV-shared layers in the model
# These typically have num_kv_heads=1 or kv_shared=True in the config
python3 -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('model_id')
print(config)  # Look for layers with shared KV
"

# Check if crash occurs only for specific layers
# KV-shared layers may alternate with regular attention layers
```

**Workaround (in mobius):**

Use ONNX `Attention` op (not `GroupQueryAttention`) for KV-shared layers.
The standard `Attention` op handles shared KV correctly.

```python
# Feature flag to control behavior:
# MOBIUS_USE_GQA_FOR_KV_SHARED=1  → use GQA (crashes on CUTLASS aligned path)
# MOBIUS_USE_GQA_FOR_KV_SHARED=0  → use Attention (default, safe)
```

**General pattern:** The GQA op in ORT assumes standard grouped-query
attention with separate KV heads. Non-standard attention patterns
(shared KV, cross-attention with unequal lengths) should use the
standard `Attention` op until GQA explicitly supports them.

---

### §6. Vision/audio f32 input vs. f16 model — Cast CPU fallback crash

**Symptom:** Segfault or CUDA error when running the vision encoder or
audio encoder. The crash occurs in a `Cast` node that ORT inserted
automatically.

**Root cause:** GenAI's image/audio preprocessor outputs `float32`
tensors (pixel_values, audio_features). When the model is exported in
`float16`, ORT inserts a `Cast(f32 → f16)` node at the model entry.
If this `Cast` node falls to CPU (e.g., due to type combination not
registered on CUDA), it reads the GPU-resident input tensor from CPU →
segfault.

**Diagnosis:**

```bash
# Check for Cast nodes in ORT logs
grep -i "Cast.*placed on.*CPU" ort_verbose.log

# Verify input tensor dtype vs model expectation
python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('vision_model.onnx', providers=['CUDAExecutionProvider'])
for inp in sess.get_inputs():
    print(f'{inp.name}: {inp.type}')
# If input expects f16 but preprocessor provides f32, Cast is inserted
"
```

**Fix (in mobius):**

PR #265: Add an explicit `Cast` node at the model entry point in the
ONNX graph. The model input remains `float32` (matching the preprocessor
output), and the `Cast(f32 → f16)` is performed inside the model where
ORT can guarantee CUDA placement.

```python
# In the model's forward():
# Accept f32 input, cast explicitly inside the model
pixel_values_f16 = op.Cast(pixel_values, to=ir.DataType.FLOAT16)
```

**General pattern:** When a model is exported in f16/bf16 but receives
f32 inputs from a preprocessor, always add an explicit `Cast` inside the
model graph rather than relying on ORT's automatic insertion. ORT's
auto-inserted Cast may fall to CPU if the specific type combination lacks
a CUDA kernel.

---

### §7. TopK 262K vocab — benchmark kernel crash

**Symptom:** GenAI's TopK benchmark crashes with `vocab_size=262144`.
CUDA error 717 (`cudaErrorCooperativeLaunchTooLarge`) from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`. The error corrupts the
CUDA context, causing all subsequent CUDA calls to fail.

**Root cause:** The `Distributed_Select` kernel in GenAI's TopK
implementation uses cooperative groups. With very large vocab sizes
(262K), the required block count exceeds what the GPU can support for
cooperative launches. The occupancy query itself returns an error, but
the error was not checked — the kernel launched anyway with invalid
parameters, corrupting CUDA state.

**Diagnosis:**

```bash
# Reproduce with large vocab benchmark
# Look for error 717 or "cooperative launch" errors in output

# Check CUDA context health after crash
nvidia-smi  # If this hangs or errors, CUDA context is corrupted
```

**Partial fix (in GenAI):**

File: `src/cuda/cuda_topk_common.cuh`
```cpp
// Add error checking in IsSupportedCooperative()
// Return false (graceful fallback) instead of launching with bad params
```

File: `src/cuda/cuda_topk_benchmark.cuh`
```cpp
// Add cudaDeviceSynchronize() in BENCHMARK_KERNEL macro
// Prevents error propagation from corrupting subsequent benchmarks
```

**General pattern:** Always check the return value of
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` before launching
cooperative kernels. Large vocab sizes (>100K) can exceed cooperative
launch limits on some GPUs. Provide a graceful fallback path.

---

### §8. DisableMemPattern for multimodal decoder

**Symptom:** Crash on the second inference call (decode) after a
successful first call (prefill). ORT reports shape mismatch or buffer
overrun in the decoder session.

**Root cause:** ORT's memory pattern optimization pre-allocates GPU
buffers based on the shapes observed during the first `Run()`. In
multimodal models, the `inputs_embeds` tensor changes shape dramatically
between prefill and decode:

- Prefill: `(batch_size, seq_len, hidden_size)` — e.g., `(1, 1024, 3072)`
- Decode:  `(batch_size, 1, hidden_size)` — e.g., `(1, 1, 3072)`

The pre-allocated buffer from prefill is sized for `seq_len=1024`. On
decode, ORT tries to reuse this pattern but the internal tensor shapes
don't match, causing a crash.

**Fix (in GenAI):**

```cpp
// In MultiModalLanguageModel constructor:
// Disable memory pattern for the decoder session when running multimodal
session_options.DisableMemPattern();
```

This is only needed for the decoder session in multimodal models. The
vision and embedding sessions have consistent shapes and can keep memory
patterns enabled.

**Diagnosis:**

```bash
# If the model works for one-shot inference but crashes on multi-turn:
# 1. Check if DisableMemPattern is set
python3 -c "
# Add to session options before creating session:
# session_options.DisableMemPattern()
"

# 2. Verify shape changes between runs
# Log inputs_embeds shape on each Run() call
```

**General pattern:** Any ORT session where input tensor shapes change
significantly between runs must use `DisableMemPattern()`. This is
common in multimodal models (prefill vs decode), variable-length
sequence models, and any model with dynamic batching.

---

### §9. genai_config.json CUDA session requirements

**Symptom:** Model runs on CPU despite requesting CUDA. Or model crashes
because some sessions run on CPU while others run on CUDA, causing
device mismatch on shared tensors.

**Root cause:** In `genai_config.json`, each session (decoder, embedding,
vision, audio) independently specifies its execution provider. An empty
`provider_options: []` defaults to CPU, not CUDA. All sessions must
explicitly specify CUDA.

**Correct genai_config.json structure:**

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "provider_options": [
          { "cuda": {} }
        ],
        "enable_mem_pattern": false
      }
    },
    "vision": {
      "session_options": {
        "provider_options": [
          { "cuda": {} }
        ]
      }
    },
    "embedding": {
      "session_options": {
        "provider_options": [
          { "cuda": {} }
        ]
      }
    },
    "audio": {
      "session_options": {
        "provider_options": [
          { "cuda": {} }
        ]
      }
    }
  }
}
```

**Key requirements:**

1. **All sessions need `{"cuda": {}}`** — even if the top-level config
   mentions CUDA, each session's `provider_options` must include it.
2. **Decoder needs `enable_mem_pattern: false`** for multimodal models
   (see §8).
3. **Gemma4 requires `<bos>` token** at prompt start. Without it, the
   model produces garbled output. Ensure the tokenizer/prompt template
   includes the BOS token.

**Diagnosis:**

```bash
# Check current config
cat genai_config.json | python3 -m json.tool

# Verify each session has CUDA provider
python3 -c "
import json
with open('genai_config.json') as f:
    config = json.load(f)
for session_name in ['decoder', 'vision', 'embedding', 'audio']:
    session = config.get('model', {}).get(session_name, {})
    providers = session.get('session_options', {}).get('provider_options', [])
    has_cuda = any('cuda' in p for p in providers)
    print(f'{session_name}: CUDA={has_cuda}')
"
```

---

## Debugging methodology

### GDB for segfaults

```bash
# Quick backtrace on crash
gdb -batch -ex "run" -ex "bt 10" --args python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])
# ... minimal repro
"

# Interactive debugging
gdb --args python3 script.py
(gdb) catch throw
(gdb) run
(gdb) bt
```

### compute-sanitizer for CUDA memory errors

```bash
# Detect illegal memory accesses
compute-sanitizer --tool memcheck python3 run_model.py

# Detect race conditions
compute-sanitizer --tool racecheck python3 run_model.py

# Detect uninitialized memory
compute-sanitizer --tool initcheck python3 run_model.py
```

### ORT verbose logging

```python
import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.log_severity_level = 0  # VERBOSE

sess = ort.InferenceSession('model.onnx', sess_options=session_options,
                            providers=['CUDAExecutionProvider'])
```

```bash
# Check node placement (which ops fell to CPU)
grep "placed on \[CPU\]" ort_verbose.log

# Check for missing CUDA kernels
grep "CUDA kernel not found" ort_verbose.log

# Check for Memcpy insertions (CPU↔GPU transfers)
grep -i "memcpy" ort_verbose.log
```

### GPU memory monitoring

```bash
# One-shot VRAM check
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Continuous monitoring (every 1s)
watch -n 1 nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Check for CUDA context corruption
nvidia-smi  # If this hangs, CUDA context is corrupted; reboot may be needed
```

### ORT profiling

```python
session_options = ort.SessionOptions()
session_options.enable_profiling = True
session_options.profile_file_prefix = "ort_profile"

sess = ort.InferenceSession('model.onnx', sess_options=session_options,
                            providers=['CUDAExecutionProvider'])
# ... run inference

# Profile JSON is written to ort_profile_<timestamp>.json
# Open in chrome://tracing or Perfetto
```

## Common pitfalls checklist

Before debugging a CUDA multimodal issue, verify these first:

- [ ] All sessions in genai_config.json have `{"cuda": {}}` provider
- [ ] Decoder session has `enable_mem_pattern: false`
- [ ] Model inputs match expected dtype (f32 inputs with f16 model need explicit Cast)
- [ ] OrtValue outputs are cleared between Run() calls
- [ ] No ops falling to CPU (check ORT verbose logs)
- [ ] CUTLASS FMHA is compatible with model's head_dim and alignment
- [ ] KV-shared layers use Attention (not GQA) op
- [ ] Prompt includes required special tokens (e.g., `<bos>` for Gemma4)

## Cross-references

- **Debugging Memcpy nodes:** See the `debugging-memcpy` skill for
  reducing CPU↔GPU transfers in ONNX models
- **Debugging multimodal output:** See the `debugging-multimodal` skill
  for stage-by-stage parity checking (vision → embedding → decoder)
- **ORT GenAI config:** See the `ort-genai-config` skill for full
  genai_config.json format and model type registry

### 10. Speech session provider_options missing CUDA

**Symptom**: Audio CUDA crashes with `MlasCastF32ToF16KernelAvx2` segfault.
Text and image CUDA work fine.

**Root cause**: In `genai_config.json`, the `speech` session has
`provider_options: []` (CPU) while decoder/embedding/vision have
`[{"cuda": {}}]`. GenAI allocates audio features on GPU (via
`p_device_`), but the speech session runs on CPU EP. ORT inserts
`InsertedPrecisionFreeCast` on CPU which reads GPU memory → crash.

**Key insight**: `speech` is a SEPARATE config key from `audio` in
genai_config.json. Setting CUDA for `audio` does NOT set it for
`speech`.

**Fix**: Ensure `speech.session_options.provider_options` includes
`[{"cuda": {}}]` in genai_config.json:

```json
{
  "model": {
    "speech": {
      "session_options": {
        "provider_options": [{"cuda": {}}]
      }
    }
  }
}
```

**Mobius**: The genai_config generator already does this correctly when
`--ep cuda` is specified (line 345 in `genai_config.py` calls
`_make_session_options(self.ep)` for the speech session). The issue
only appears with manually constructed configs.

### 11. Audio encoder SkipSimplifiedLayerNormalization 1D skip

**Symptom**: Audio encoder CUDA crashes with "skip is expected to have
3 or 2 dimensions, got 1". CPU works fine.

**Root cause**: PR #253 changed `_Gemma4ScaleFreeRMSNorm` from manual
ops to `op.RMSNormalization(stash_type=1)`. This enabled ORT's
SkipLayerNorm fusion pattern: `Add(output_proj.bias [1D]) +
RMSNormalization → SkipSimplifiedLayerNormalization`. The 1D bias
ends up in the "skip" input position. CUDA kernel rejects 1D skip.

**Fix**: Inline manual RMSNorm primitive ops for the audio encoder's
`pre_projection_norm` to prevent ORT from recognizing the fusion
pattern. Applied in mobius PR #269.

**Detection**: Look for SkipSimplifiedLayerNormalization nodes with
1D initializer inputs in position 1 (skip).
