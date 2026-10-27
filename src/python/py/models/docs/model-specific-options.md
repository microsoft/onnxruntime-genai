# Model-specific Export Options

These options and workflows apply to particular model families and speculative drafters. See the [Model Builder README](../README.md) for shared configuration, quantization, and export options, and [Qwen3.8 design](qwen3_8.md) for component ownership.

## Build a DFlash 2 Block Drafter

Set `dflash2_path` to a DFlash 2 checkpoint to export an auxiliary `dflash2.onnx` block drafter beside a Qwen3.5 or Qwen3.8 target model. The target must use paged attention. SpecForge identifies the target layers whose outputs are tapped, while `aux_hidden_state_layers` identifies residual streams entering layers, so each configured auxiliary layer must be one greater than the corresponding `target_layer_ids` entry in the DFlash checkpoint. The drafter reuses the target's embedding and LM-head initializers, so both checkpoints must use compatible tensors.

`dflash2_num_draft_tokens` optionally overrides how many tokens the drafter proposes per step. It must be a positive integer no greater than the draft checkpoint's block size minus the anchor token; that checkpoint limit is also the default.

`max_draft_tokens` writes `speculative.max_draft_tokens` into `genai_config.json`, capping how many drafted tokens the engine verifies each step. It must be between 1 and 16, and defaults to unset, which leaves the runtime default of 4 in effect. This differs from `dflash2_num_draft_tokens`: the drafter's exported block costs the same to run no matter how many of its tokens are verified, so raising this value buys extra accepted tokens for free until the wider verification step costs more than it saves. The best value is workload-specific and must be measured; it can be retuned on an already-exported model by editing the config.

The legacy `dflash2_precision` option accepts `bf16` (default), `int4`, or `int8`. With builder configuration version 2, `drafter_options.quant_config` independently controls the DFlash 2 body and additionally supports `weights.type=int2`. Integer modes quantize the attention and MLP weights while keeping the small dynamic-convolution and selector projections dense. Body activations and KV caches default to BF16 on CUDA; WebGPU requires FP16. Structured `drafter_options.quant_config.io_dtype` can select FP16 on CUDA but cannot select BF16 on WebGPU. This option does not quantize the drafter's KV cache. INT2 requires an ONNX Runtime build with 2-bit `MatMulNBits` support.

INT4/INT8 body weights honor the drafter's explicit `quant_config.format.matmulnbits_weights_prepacked` target setting. INT2 body weights retain portable raw blockwise storage in the generic packing pass; set `runtime_config.model.dflash2.session_options["ep.cuda.fpa_intb_gemm"]` to `"1"` to let the runtime prepack eligible raw weights. INT2 requires block size 64 or 128 and output width divisible by 128 for that kernel path. Unsupported projections stay raw. BF16 x INT2 requires a full fpA-intB kernel build.

Normally the LM head is not quantized separately: the drafter adopts the target's saved head, bytes and layout alike, so the two always agree and are deduplicated into one copy on disk. Its precision and block size remain the target's. If a BF16 target uses offline-prepacked weights, the drafter instead keeps a private raw INT4 head because the prepacked kernel requires FP16 activations. When the target's head uses a format the drafter cannot address by name, such as asymmetric, `use_qdq`, or `rtn`/`k_quant` layouts, the drafter's head stays dense. A head the checkpoint supplies already quantized (FP8) overrides `--precision` for the target and for the drafter alike. The embedding table works the same way: `op_types_to_quantize=MatMul/Gather` turns the target's `Gather` into `GatherBlockQuantized`, and the drafter adopts that table rather than keeping a dense copy. It has to, because the two graphs are deduplicated by initializer name — a target that renames the table while the drafter keeps a dense `Gather` costs more than the target saved. Under `shared_embeddings` the target gathers from its LM-head weight instead of a table of its own, so the drafter's embedding stays dense.

WebGPU DFlash 2 uses FP16 activations and KV caches. Residual addition and `SkipSimplifiedLayerNormalization` execute in FP32, with persistent residual sums kept in FP32. Attention/MLP convolution **finish** multiply, mask, and accumulation paths also stay FP32 through the following skip normalization. Only normalized outputs are cast back to FP16. Convolution prepare paths and coefficients, attention/KV caches, and INT4 `MatMulNBits` activations remain FP16. This policy is automatic and requires re-exporting older WebGPU drafters.

```bash
python -m onnxruntime_genai.models.builder -i path_to_target_model -o path_to_output_folder -p int4 -e webgpu --extra_options use_paged_attention=true num_blocks=1024 state_update_capacity=7 aux_hidden_state_layers=6,20,34,48,62 dflash2_path=path_to_dflash2_checkpoint dflash2_precision=int4
```

```bash
python -m onnxruntime_genai.models.builder -i path_to_target_model -o path_to_output_folder -p int4 -e cuda --extra_options use_paged_attention=true aux_hidden_state_layers=2,12,22 dflash2_path=path_to_dflash2_checkpoint dflash2_precision=int4 max_draft_tokens=7
```

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_target_model -o path_to_output_folder -p fp16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true aux_hidden_state_layers=2,12,22 dflash2_path=path_to_dflash2_checkpoint dflash2_num_draft_tokens=4 dflash2_precision=int4 max_draft_tokens=4

# From source:
python builder.py -i path_to_target_model -o path_to_output_folder -p fp16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true aux_hidden_state_layers=2,12,22 dflash2_path=path_to_dflash2_checkpoint dflash2_num_draft_tokens=4 dflash2_precision=int4 max_draft_tokens=4
```

For a BF16 x INT2 DFlash 2 body using CUDA runtime prepacking, use builder configuration version 2 and an ONNX Runtime build with the full fpA-intB kernel set:

```json
{
  "builder_config_version": 2,
  "target_options": {
    "attention": {"implementation": "paged"}
  },
  "drafter_options": {
    "drafter_type": "dflash2",
    "path": "path_to_dflash2_checkpoint",
    "num_draft_tokens": 7,
    "quant_config": {
      "io_dtype": "bf16",
      "weights": {"type": "int2", "block_size": 64}
    }
  },
  "runtime_config": {
    "model": {
      "dflash2": {
        "session_options": {"ep.cuda.fpa_intb_gemm": "1"}
      }
    },
    "speculative": {"max_draft_tokens": 7}
  }
}
```

## Build a DSpark Block Drafter

Set `dspark_path` to a DSpark checkpoint to export an auxiliary `dspark.onnx` block drafter beside a Qwen3.5 or Qwen3.8 target model. The target must use paged attention. SpecForge identifies the target layers whose outputs are tapped, while `aux_hidden_state_layers` identifies residual streams entering layers, so each configured auxiliary layer must be one greater than the corresponding `target_layer_ids` entry in the DSpark checkpoint. DSpark has no precision option of its own: its body is always BF16, and its embedding and LM head follow whatever format the target saved them in, adopted under the target's initializer names on the terms described for [DFlash 2](#build-a-dflash-2-block-drafter). `dspark_path` and `dflash2_path` are mutually exclusive, and selecting DSpark replaces rather than accompanies the target's MTP head.

`dspark_num_draft_tokens` optionally overrides how many tokens the drafter proposes per step. It must be at least `2` and no greater than the checkpoint's trained `block_size`; the default is that checkpoint block size. `dspark_top_k` controls how many candidates the lattice keeps per block slot; it defaults to `16` and must be a positive integer no greater than the drafter vocabulary size. The score tensor and its host copy grow as `dspark_top_k` squared, so keep this value near the default unless a larger lattice has been benchmarked for the intended workload.

The DSpark transformer body executes in BF16 even when the target uses FP16, so the selected execution provider and hardware must support BF16. DSpark export on WebGPU is rejected because WebGPU PagedAttention supports only FP16 activations and KV caches. Its KV cache consumes additional GPU memory: a full-attention DSpark layer adds a key/value entry for every target cache block and therefore reduces the target's maximum resident context under a fixed memory budget. A checkpoint configured uniformly for sliding-window attention uses its positive `sliding_window` instead; because the exported graph carries a single window for every layer, a checkpoint that windows only some of its layers (`max_window_layers` other than `0`, or a mixed `layer_types`) is rejected.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_target_model -o path_to_output_folder -p fp16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true aux_hidden_state_layers=1,11,21 dspark_path=path_to_dspark_checkpoint dspark_num_draft_tokens=4 dspark_top_k=16

# From source:
python builder.py -i path_to_target_model -o path_to_output_folder -p fp16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true aux_hidden_state_layers=1,11,21 dspark_path=path_to_dspark_checkpoint dspark_num_draft_tokens=4 dspark_top_k=16
```

## Export a ModelOpt or compressed-tensors NVFP4/FP8 Checkpoint

This scenario is for when your Qwen3.6 MoE checkpoint has already been quantized by NVIDIA TensorRT Model Optimizer and you want to carry its NVFP4/FP8 tensors into the ONNX model instead of dequantizing to fp16 and re-quantizing to int4.

```bash
# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p bf16 -e cuda -c cache_dir_to_store_temp_files
```

When `config.json` declares `quant_method=modelopt` or `quant_method=compressed-tensors`, the builder preserves the checkpoint's original quantized data types automatically: routed experts use native NVFP4 `QMoE`, dense NVFP4 modules use `MatMulBlockQuantizedFp4Weight`, and FP8 attention projections use `MatMulBlockQuantizedFp8Weight`. The compressed-tensors loader accepts packed NVFP4 weights with reciprocal global scales and scalar or per-channel FP8 weight scales. KV-cache quantization remains explicit and requires calibrated scales supplied through `kv_cache_quant_scheme` and `kv_cache_scale_file`. CUDA linear-attention gate fusion is enabled automatically.

The `--precision` argument controls the unquantized tensors and model I/O; it does not change the checkpoint's native FP8/NVFP4 tensors. ModelOpt and compressed-tensors export require the CUDA EP and an ONNX Runtime build that provides the corresponding contrib ops. For CPU, CUDA, and WebGPU, the builder replaces each shared-expert output `Mul` and routed/shared `Add` pair with `com.microsoft::GatedAdd`; other execution providers retain the portable `Mul` + `Add` graph.

NVIDIA's `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint is also supported. Its routed main-model experts remain native NVFP4, attention/shared experts/hyper-connections follow `--precision`, and the sharded PLE table retains its FP8 bytes and per-tensor scale. The loader concatenates PLE shards in numeric order without loading the checkpoint through Hugging Face's eager ModelOpt quantizer. Vision weights are loaded only when exporting the multimodal components.

The official [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) checkpoint is supported on CUDA and WebGPU through the native loader, without Hugging Face's eager FP8 quantizer. Decoder and MTP experts retain their E4M3 weight bytes and 128x128 `weight_scale_inv` block multipliers. CUDA preserves BF16 expert scales; WebGPU widens them exactly to FP32 to match its QMoE kernel. Engram retains its E4M3 table and widens its scalar scale exactly to FP32 for the annotated CPU lookup, on both CUDA and WebGPU exports. No export-time weight dequantization, reciprocal-scale conversion, or requantization is performed. Use `bf16` to retain the source precision of unquantized tensors as well. The existing FP8 `QMoE` path is weight-only, not the official checkpoint's dynamic-A8 arithmetic. Multimodal decoder exports preserve temporal, height, and width coordinates: Q/K normalization and MRoPE run before sparse attention, rather than reducing `position_ids` to its temporal row. The sparse indexer receives request-specific cosine/sine tables recomposed from the complete coordinate history.

```bash
# From source; repository downloads reuse the specified cache:
python builder.py -m Qwen/Qwen3.8-Flash-Next-FP8 \
  -o path_to_output_folder -p bf16 -e cuda \
  -c path_to_cache_dir --extra_options text_only=true
```

An already-downloaded checkpoint directory can instead be supplied with `-i`. MTP and the standalone Engram graph are included by default. Omit `text_only=true` to include the vision and embedding graphs.

Multimodal main and MTP graphs require an int64 `rotary_position_ids` input with shape `[3, batch_size, rotary_sequence_length]`, mapped in their config `inputs`. It contains T/H/W positions for every chronological cache row, including the current tokens. The existing `position_ids` input remains current-step-only: `[3, batch_size, sequence_length]` for dense execution or `[3, num_tokens]` for paged execution. Paged history rows follow request order in `cumulative_sequence_lengths`; shorter histories can be zero-padded to the longest request. The runtime must retain the prompt's actual image/video coordinates across decode, rewind, and speculative rounds, not reconstruct them as ordinary token indices. Text-only exports retain their existing ABI. This binding requires the matching runtime support in the stacked execution PR.

Native NVFP4 expert layout is explicit and independent of the export host GPU. For `quant_type="nvfp4"` on CUDA, omitted/default `qmoe_weights_prepacked=-1` and explicit `0` select legacy N-packed raw storage; `1` opts into K-packed row-major storage and requires an ONNX Runtime build supporting the NVFP4 meaning of `weights_prepacked`. Mode `2` is not supported. Integer QMoE has different meanings: `-1`/`1` select its CUTLASS layout and `0` is raw. These distinct contracts are why the two original per-op packing options are retained instead of one ambiguous `prepack_layout` flag. Gate/up rows are interleaved without dequantization; FP8 block scales and FP32 global scales are unchanged. Legacy raw-layout Qwen3.8 exports retain `session.disable_prepacking=1` as a memory-safety default; the MTP session does not inherit it. Re-export to change layouts: changing only an ONNX attribute does not rearrange the stored bytes.

Qwen3.8 Flash exports always include `engram.onnx` and `engram.onnx.data`, including paged exports with `text_only=true` or `exclude_mtp=true`. The standalone Engram component owns hashing and table lookup for both execution layouts; neither the decoder nor MTP stores the table or duplicates hashing/lookup. Non-CPU exports annotate the `GatherBlockQuantized` lookup for CPU execution and widen its scalar scale exactly to FP32. Only gathered activations are cast to model I/O precision. Engram retains the native FP8 table without dequantizing its stored weights. Text-only export does not require vision or embedding companion graphs.

Each exported ONNX model owns exactly one separate external-data file named `<model filename>.data`, including embedding, vision, Engram, MTP, block-drafter, CLM, and KEV graphs. A graph never references another graph's data file. Small `Constant` tensor attributes remain inline because ONNX Runtime shape inference cannot read external shape/axis constants; initializers and other supported tensor attributes use the graph's own data file.

Engram token history follows the decoder's `past.%d.ple_tokens` / `present.%d.ple_tokens` naming convention, using the zero-based PLE layer ID. Its config uses `inputs.past_ple_token_names` and `outputs.present_ple_token_names` with concrete bindings such as `past.1.ple_tokens` and `present.1.ple_tokens`. The runtime also accepts the legacy `past_tokens` and `present_tokens` keys for older exports.

`model.decoder.ple_token_pad_id` is always emitted as a scalar integer. If the checkpoint lists multiple EOS IDs, the first is the Engram reset/padding ID. The decoder also records `indexer_compress_ratio` from the architecture config so dense speculative rollback does not assume a four-token compression block.

For paged decoding, the runtime invokes the original standalone `NGramHashMapping` lookup pipeline per request and packs its `[batch_size, sequence_length, ple_embed_dim]` output into the decoder's `[num_tokens, ple_embed_dim]` `engram_embeddings` input. It forwards the component's updated token history through the matching `present.%d.ple_tokens` decoder input/output binding. No varlen hashing adaptation or additional shape-processing subgraph is required in the standalone component. Compact speculative-state capture retains the separate standard-ONNX gathers for accepted-prefix rollback snapshots rather than duplicating the hash operator or lookup.

```bash
python builder.py -i path_to_local_checkpoint \
  -o path_to_output_folder -p fp16 -e cuda \
  -c path_to_cache_dir --extra_options text_only=true
```

Use `--extra_options text_only=true` for supported text exports, or additionally `exclude_mtp=true` to omit MTP. Qwen3.8 MTP `checkpoint_policy=requantize` is explicitly rejected until conversion is supported. NVIDIA's MTP experts retain their original E4M3 FP8 weights and 128x128 `weight_scale_inv` block multipliers without export-time dequantization or requantization. Gate, up, and down projections are separate `QMoE` inputs with `quant_type="fp8"`, `block_size=128`, `activation_type="silu"`, and `swiglu_fusion=0`. This requires the CUDA block-scaled FP8 support from [ORT PR #32887](https://github.com/microsoft/onnxruntime/pull/32887).

## MTP Head (Qwen3.6/Qwen3.8)

For CUDA decoder exports, standalone Engram uses a CUDA session with `session.layer_assignment_settings="cpu(=cpu_embedding)"`. Only its `GatherBlockQuantized` node carries `layer_ann=cpu_embedding`; hashing and other supported operations remain on CUDA. The FP8 table stays unchanged, and its scalar scale is widened exactly to FP32 for the CPU lookup kernel.

Paged Qwen3.8 QSA MTP exports include IndexShare in `mtp.onnx` by default. There are no separate extend/decode model files and no ONNX `If` node. Index sharing is an inherent property of the paged main/MTP graphs, not a separate feature flag or conversion tool. `max_draft_tokens` controls the draft budget (default `7` for this contract, exported capacity at most `7`). Runtime budgets `1` through the exported limit all use the same MTP session; larger budgets are rejected. Existing older exports are unchanged.

The unversioned IndexShare contract adds CPU `mode`, GPU int64 `projection_rows`, frozen `past_indices`/`past_counts`, `base_row_indices`, and `range_starts`/`range_ends`. Refresh selects all projection rows and supplies zero merge rows; reuse selects zero projection rows and merges one query per active request. A standalone `PackedSparseAttentionIndexerMerge` feeds the extended `PackedSparseAttentionIndexer`, which accepts packed Q/K without a Split. Every forward uses the same session, decoder body, output names and state interface, including `present_indices`, `present_counts`, and `indexer_status`, produced directly by the indexer without Identity aliases. Export removes unused Constant nodes and initializers while preserving live constants and external weight payloads. The MTP config maps `inputs.past_indices`/`past_counts` and `outputs.present_indices`/`present_counts`/`indexer_status` to those graph names. The requested draft limit is specified only by `speculative.max_draft_tokens`; the runtime derives the exported IndexShare capacity from the graph rather than a second MTP draft limit. Older IndexShare config aliases are not accepted. Packages using the former IndexShare binding names must be renamed or re-exported for a GenAI runtime using this contract. There is no output pruning, special executor option, ONNX If, or duplicated decoder body. This interface requires the extended CUDA operators; unsupported runtimes/providers fail rather than silently ignoring reuse.

The dynamic Engine captures the final extend row's QSA selection per request, freezes it for the iteration, and appends draft positions without rerunning indexer projection, scoring, or TopK. Extend already includes the bonus token, so decode capacity is `base_capacity + max_draft_tokens - 1` (2057 for a 2051-entry QSA base and seven drafts). Target verification is unchanged. A single draft uses the same extend path, with no subsequent reuse call. A runtime draft budget above the exported limit is rejected. All merge statuses are checked before publishing the chain. External expert tensors retain their original storage without changing their checkpoint bytes or quantized formats.

To enable IndexShare for an older export, re-export the checkpoint with Model Builder and `use_paged_attention=true`. Use the dynamic `Engine`, not the separate legacy `MtpGenerator` API. Full native-model greedy parity and H200/GB10 performance gates remain unverified on the available hardware.

The CUDA operators also support CSA reuse: `append_indices` merges compressed entry IDs, and the indexer's default CSA reuse policy advances key/gate state without query rotation, scoring or TopK. Required auxiliary K/V production and the attention-owned local window remain outside projection gating. This is an operator contract, not a complete DeepSeek V4 exporter or MTP integration; its model-specific cache ownership, visibility and rollback still need validation.

When a Qwen3.5 MoE configuration declares one or more MTP layers with `mtp_num_hidden_layers`, the builder exports the multi-token-prediction head for self-speculative decoding. An auxiliary `mtp.onnx` (plus its `mtp.onnx.data`) is generated alongside the main model, and the main model automatically exposes the hidden states consumed by the MTP head. Models without declared MTP layers do not produce this file or an MTP section in `genai_config.json`.

Qwen3.8 Flash Next checkpoints (`Qwen4ExpForConditionalGeneration`) use their native one-layer MTP head automatically. The exporter preserves the target's hyper-connection streams, applies the checkpoint's separate `mtp.fc_embedding` and `mtp.fc_hidden` projections, and exports the QSA indexer cache bindings required by the draft layer. Identical `model.embed_tokens.weight` and LM-head tensors are declared as shared initializers for runtime reuse, but each graph retains its own external-data copy. Adopting a target tensor copies its bytes into the drafter's own data file rather than creating a cross-graph reference. Both the original floating-point checkpoint and NVIDIA's mixed NVFP4 checkpoint are supported, with the FP8 MTP reconstruction described above.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files
```

Qwen3.5 MoE checkpoints (`Qwen3_5MoeForConditionalGeneration`) that declare MTP layers must ship `mtp.*` weights in their safetensors. For those models, the builder rejects `exclude_lm_head=true` and `prune_lm_head=true` because the exported MTP workflow requires the main LM head. The MTP weights are read directly from the source safetensors because Hugging Face `transformers` discards them on load. Hugging Face repository IDs are resolved to their downloaded snapshot before the safetensors are scanned.

Set `exclude_mtp=true` to skip the head entirely, which is how such a checkpoint is built with `prune_lm_head=true` for a deployment that does not speculate. A block drafter (`dflash2_path` / `dspark_path`) already supersedes the head and needs no extra option.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_mtp=true prune_lm_head=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_mtp=true prune_lm_head=true
```

To keep the exported head available but prevent the dynamic Engine from loading and running it automatically, set `model.mtp.enabled` to `false` in `genai_config.json` while preserving the rest of the section. Alternatively, remove the entire `model.mtp` section. Both methods avoid rebuilding the ONNX files; recreate the Model and Engine after changing the configuration. The flag defaults to `true` and does not affect an `MtpGenerator` constructed explicitly by the application.

By default the MTP head inherits the main model's settings. For a ModelOpt or compressed-tensors checkpoint, the builder preserves each original MTP tensor format: native NVFP4 linears and experts remain NVFP4, FP8 attention projections remain FP8, and unquantized tensors follow the requested graph precision.

To configure the MTP model independently, use `mtp_quant_config` with an inline JSON object or a JSON file using the structured `QuantConfig` schema. Its `io_dtype`, `weights`, `moe`, and `runtime` targets are independent. For example, `mtp_quant_config='{"io_dtype":"bf16","weights":{"type":"int4","block_size":64},"moe":{"type":"nvfp4"}}'` exports INT4 dense MTP MatMuls, NVFP4 MTP experts, and BF16 I/O.

Supplying `mtp_quant_config` explicitly dequantizes native ModelOpt or compressed-tensors MTP tensors before applying the MTP configuration. Dense `weights.type` supports integer or unquantized formats; use `moe.type=mxfp4/nvfp4` to select FP4 experts independently. Without an explicit MTP configuration, pre-quantized NVFP4 dense weights remain `MatMulBlockQuantizedFp4Weight` and native FP8 attention projections remain `MatMulBlockQuantizedFp8Weight`.

The head always exports `hidden_states_out` (its own post-final-norm hidden state), which a multi-token loop feeds back as the next chained draft's `hidden_states` input. It is required for `num_speculative_tokens > 1` and ignored otherwise.

A multi-token verify forward can additionally carry a window of recurrent/conv states so a partial accept can be handled by cropping instead of replaying the main model. Pass `state_window=W` (with `W >= num_speculative_tokens + 1`) to widen `past/present_key_values.%d.{conv,recurrent}_state` to `[W, B, ...]` and emit the matching attribute on `CausalConvWithState` / `LinearAttention`. This requires ONNX Runtime kernels that understand the attribute; leave it at the default `0` otherwise.

## Compact State Updates (Qwen3.5/3.8)

Paged Qwen3.5/3.8 exports can capture compact convolution and GatedDeltaNet transitions for speculative tokens instead of returning full recurrent-state checkpoints. Set `state_update_capacity=N` to reserve updates for up to `N` tokens. The capacity defaults to `0` (disabled) and requires `use_paged_attention=true`. It must be an integer from `0` through `8`, matching the kernel and runtime-parser bound, because the kernel packs every captured transition for a layer into a single fixed-width capsule output. Paged Qwen3.5/3.8 exports use GatedDeltaNet and support CUDA and WebGPU with `fp16` model I/O; CUDA also supports `bf16`. The Engine supports compact state updates on CPU, CUDA, and WebGPU. WebGPU currently stages compact partial-acceptance replay through CPU for correctness, so it is not yet performance optimized. When enabled, `genai_config.json` records the capacity, the `state_update_capture_count` and `state_update_active` input bindings, and the per-layer convolution-value and recurrent-capsule output templates. All compact state-update inputs and outputs are omitted when `state_update_capacity=0`.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p bf16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true state_update_capacity=3

# From source:
python builder.py -m model_name -o path_to_output_folder -p bf16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true state_update_capacity=3
```

## Qwen3.5/3.8 Recurrent Operator

Qwen3.5/3.8 hybrid layers emit `CausalConvWithState` + `GatedDeltaNet` with an FP32 V-major recurrent state and native Qwen gate arithmetic from the raw `A_log`/`dt_bias` initializers. GatedDeltaNet export supports CUDA and WebGPU, with `state_window=0` or at least `2`; paged exports use CUDA with `fp16`/`bf16` or WebGPU with `fp16` model I/O. The equivalent `LinearAttention` + `LinearAttentionGate` decomposition is retained internally in the Qwen builders rather than exposed as a model export option.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p fp16 -e webgpu --extra_options use_paged_attention=true num_blocks=128

# From source:
python builder.py -m model_name -o path_to_output_folder -p fp16 -e webgpu --extra_options use_paged_attention=true num_blocks=128
```

## GPT-OSS Precision Advisory

The builder emits a precautionary FP16 I/O warning for exports with 36 decoder layers and 128 experts per layer, matching the [published GPT-OSS-120B configuration](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json). The [original overflow report](https://github.com/microsoft/onnxruntime-genai/pull/2444) describes NaN logits after activation growth through layer 35, but does not identify the checkpoint revision or complete reproduction settings. It does not establish that every 120B checkpoint or FP16 export overflows.

The [published GPT-OSS-20B configuration](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json) has 24 layers and 32 experts per layer and does not receive this warning. The advisory also excludes BF16/FP32 I/O and exports truncated with `num_hidden_layers` to fewer than 36 layers. It neither rejects FP16 nor changes the requested precision or existing defaults. Absence of this warning is not a guarantee of numerical accuracy.

On CUDA / TRT-RTX, explicitly select the wider BF16 range with `--precision bf16` or, for INT4 weights, `--precision int4 --extra_options use_cuda_bf16=true`:

```bash
# From wheel (INT4 weights and BF16 I/O on CUDA):
python -m onnxruntime_genai.models.builder -m openai/gpt-oss-120b -o path_to_output_folder -p int4 -e cuda --extra_options use_cuda_bf16=true

# From source:
python builder.py -m openai/gpt-oss-120b -o path_to_output_folder -p int4 -e cuda --extra_options use_cuda_bf16=true
```

`use_cuda_bf16` does not change I/O precision on other providers. Consider FP32 I/O where supported, or a provider with BF16 support. Check finite logits and output quality using the actual checkpoint, quantization settings, provider, and representative inputs; configuration-only regression tests do not qualify full-model numerical behavior.
