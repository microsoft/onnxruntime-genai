# Qwen hybrid WebGPU coding export profile

This is an opt-in reproduction recipe, not a new provider default or a claim of
qualification on every WebGPU device. Use a supported Qwen3.5-family hybrid
target and a compatible DFlash 2 checkpoint. Historical artifact names alone
are not architecture identifiers: newer sparse Qwen3.8 exporters have different
contracts and must not be substituted for a dense hybrid checkpoint.

Pin the target and drafter revisions locally. Record their file SHA256 hashes,
the GenAI source commit, the builder command, and the runtime and plugin hashes.
Retain the same checkpoint, quantization policy, and weight bytes when comparing
chunk policies. Do not patch exported ONNX graphs.

From the repository root (replace the checkpoint paths and cache capacity):

```bash
python src/python/py/models/builder.py \
  -i target_checkpoint -o output -p int4 -e webgpu \
  --drafter_options '{"drafter_type":"dflash2","path":"drafter_checkpoint","quant_config":{"io_dtype":"fp16"}}' \
  --runtime_config '{"engine":{"dynamic_batching":{"num_blocks":128,"max_batch_size":1,"max_scheduled_tokens":512,"prefix_caching":false}},"search":{"chunk_size":512},"speculative":{"max_draft_tokens":7,"min_adaptive_k":0,"cooldown":false}}' \
  --extra_options use_paged_attention=true prune_lm_head=true \
    paged_block_size=256 paged_chunk_size=512 \
    dflash2_num_draft_tokens=7 webgpu_gdn_chunk_size=16
```

For an installed builder replace `src/python/py/models/builder.py` with
`-m onnxruntime_genai.models.builder`. A drafter checkpoint must support at least
seven proposals. The chosen block budget must cover target and auxiliary cache
reservations as well as prompt plus output; `128` is illustrative, not automatic
capacity sizing. This recipe does not enable graph capture.

The three independent controls are:

* `prune_lm_head=true`: existing builder support exports `logits_indices` and
  gathers only the target rows the Engine needs, including speculative
  verification rows. It does not prune the vocabulary or change head weights.
* `search.chunk_size=512` / `max_scheduled_tokens=512`: per-request prompt
  contribution and whole-batch scheduled-token budget, respectively.
* `webgpu_gdn_chunk_size=16`: internal GatedDeltaNet chunk attribute, rather than
  the emitter's unchanged default of 64. FP32 gates, recurrent state,
  accumulation, and state-update capture bindings remain unchanged.

Inspect the emitted config and graph before running: `logits_indices` must be
bound, GatedDeltaNet nodes must carry `chunk_size=16`, and recurrent inputs and
outputs must remain FP32. Compare generated token IDs, acceptance counters,
and fixed-state snapshots against the 64-chunk export on the same runtime.
Changing chunk arithmetic can change floating-point rounding; qualify output
and application quality rather than assuming bit identity.

Use fresh requests with prefix caching disabled, deterministic sampling, an
identical prompt-token hash, and at most 2048 output tokens. Report model-load
time separately from first-request TTFT and warm-process TTFT/decode rate.
Warm process does not mean a cold filesystem cache.

## Runtime prerequisites

The loaded plugin must implement the selected GatedDeltaNet chunk path.
GenAI #2773 (convolution replay) and #2775 (phase-boundary copies) are separately
owned prerequisites for the integrated performance configuration, not changes
implemented by this recipe. ORT #33567 provides the companion copy-vector
submission optimization. Do not invent a package pin: adopt a published,
mutually compatible core/plugin pair only after operator, rollback, output,
and platform qualification. Builder-only tests do not establish GPU support.
