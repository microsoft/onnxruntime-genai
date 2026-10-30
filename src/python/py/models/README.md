# ONNX Runtime GenAI Model Builder

This folder contains the model builder for quickly creating optimized and quantized ONNX models within a few minutes that run with ONNX Runtime GenAI.

# Contents

- [Model-specific Export Options](docs/model-specific-options.md)

- [Current Support](#current-support)
- [Usage](#usage)
  - [Full Usage](#full-usage)
  - [Structured Builder Configuration](#structured-builder-configuration)
  - [Original PyTorch Model from Hugging Face](#original-pytorch-model-from-hugging-face)
  - [Original PyTorch Model from Disk](#original-pytorch-model-from-disk)
  - [Customized or Finetuned PyTorch Model](#customized-or-finetuned-pytorch-model)
  - [Quantized PyTorch Model](#quantized-pytorch-model)
  - [GGUF Model](#gguf-model)
  - [Extra Options](#extra-options)
    - [Number of Hidden Layers](#number-of-hidden-layers)
    - [Filename](#filename)
    - [Config Only](#config-only)
    - [Hugging Face Authentication](#hugging-face-authentication)
    - [Hugging Face Remote Code](#hugging-face-remote-code)
    - [Exclude Embedding Layer](#exclude-embedding-layer)
    - [Exclude Language Modeling Head](#exclude-language-modeling-head)
    - [Prune Language Modeling Head](#prune-language-modeling-head)
    - [Include Last Hidden States Output](#include-last-hidden-states-output)
    - [Include Auxiliary Hidden States Output](#include-auxiliary-hidden-states-output)
    - [Build with Paged Attention](#build-with-paged-attention)
    - [Disable Windowed KV Cache](#disable-windowed-kv-cache)
    - [Enable Shared Embeddings](#enable-shared-embeddings)
    - [Enable CUDA Graph Capture](#enable-cuda-graph-capture)
    - [MTP Head (Qwen3.6/Qwen3.8)](#mtp-head-qwen3638)
    - [Select the Qwen3.5/3.8 Recurrent Operator](#select-the-qwen3538-recurrent-operator)
    - [Enable WebGPU Graph Capture](#enable-webgpu-graph-capture)
    - [Configure QKV Projections Fusion](#configure-qkv-projections-fusion)
    - [Disable QK Norm GQA Fusion in CUDA or WebGPU](#disable-qk-norm-gqa-fusion-in-cuda-or-webgpu)
    - [Quantization Options](#quantization-options)
      - [Accuracy Level](#accuracy-level)
      - [MatMul Block Size](#matmul-block-size)
      - [QMoE Block Size](#qmoe-block-size)
      - [QMoE Weights Prepacked](#qmoe-weights-prepacked)
      - [MatMulNBits Weights Prepacked](#matmulnbits-weights-prepacked)
      - [Is Symmetric](#is-symmetric)
      - [Op Types To Quantize](#op-types-to-quantize)
      - [Nodes To Exclude](#nodes-to-exclude)
      - [Algo Config](#algo-config)
      - [Int8 Bit Placement](#int8-bit-placement)
      - [Use QDQ Pattern for Quantization](#use-qdq-pattern-for-quantization)
      - [Use 8 Bits Quantization in QMoE](#use-8-bits-quantization-in-qmoe)
      - [Use FP4 Quantization in QMoE](#use-fp4-quantization-in-qmoe)
      - [Quantize the KV Cache](#quantize-the-kv-cache)
    - [FP32 I/O for WebGPU EP](#fp32-io-for-webgpu-ep)
    - [BF16 I/O for CUDA EP](#bf16-io-for-cuda-ep)
    - [LoRA Models](#lora-models)
  - [Unit Testing Models](#unit-testing-models)
    - [Option 1: Use the model builder directly](#option-1-use-the-model-builder-directly)
    - [Option 2: Edit the config.json file](#option-2-edit-the-configjson-file-on-disk-and-then-run-the-model-builder)
- [Design](#design)

## Current Support

The tool currently supports the following model architectures.

- AMD OLMo
- ChatGLM
- DeepSeek
- ERNIE 4.5
- Gemma
- gpt-oss
- Granite
- Granite MoE Hybrid
- HunYuan Dense V1
- InternLM2
- LFM2 (text, LFM2-VL / LFM2.5-VL / d1-3B with their vision models, and the decoders of LFM2-Audio / LFM2.5-Audio)
- LFM2 MoE
- Llama
- Mistral
- Nemotron
- Phi
- Qwen
- SmolLM3
- Whisper

It is intended for supporting the latest, popular state-of-the-art models.

## Usage

### Full Usage

For all available options, please use the `-h/--help` flag.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder --help

# From source:
python builder.py --help
```

### Structured Builder Configuration

Schema version 2 is an **experimental implementation** of the [shared configuration design](../../../../docs/ModelBuilderConfiguration.md). It normalizes legacy quantization syntax before applying structured overrides, rejects unsupported or conflicting drafter policies, validates runtime overlays against exported capabilities, and checks borrowed quantized-head layouts before adoption. Olive integration, target checkpoint conversion policy, and INT8 embedding export remain pending.

Each structured CLI option accepts an inline JSON object or a JSON file path. Relative paths use the process working directory, including nested checkpoint and calibration paths. Omitting `builder_config_version` selects version 2 when a target, drafter, speculative, or runtime field is present. Version 1 cannot be combined with those fields. `search` alone does not select version 2.

The following target-only invocation template requires a supported dense, unquantized checkpoint and its tokenizer. It makes the CPU I/O dtype explicit and does not exercise checkpoint conversion, sharing, or speculative limits:

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_dense_checkpoint -o output -e cpu \
  --builder_config_version 2 \
  --target_options '{"quant_config":{"io_dtype":"fp32","weights":{"type":"int4","block_size":32}}}' \
  --drafter_options '{"drafter_type":"none"}' \
  --runtime_config '{"search":{"max_length":128}}'

# From source, at the repository root:
python src/python/py/models/builder.py -i path_to_dense_checkpoint -o output -e cpu \
  --builder_config_version 2 \
  --target_options '{"quant_config":{"io_dtype":"fp32","weights":{"type":"int4","block_size":32}}}' \
  --drafter_options '{"drafter_type":"none"}' \
  --runtime_config '{"search":{"max_length":128}}'
```

`target_options` routes `quant_config` and `attention` to the existing exporter. Target MLP gate/up, QKV, and eligible Q/K normalization fusion are automatic and have no configuration options. `quant_config.format` is the canonical graph-layout key; `runtime` remains a parsing alias. The target rejects an explicit checkpoint policy until its loaders implement both paths. Root CLI `precision` is optional when target weight type is explicit.

DFlash2 and DSpark selection requires a local checkpoint `path` and paged target attention. DSpark uses BF16 body I/O; DFlash2 defaults to BF16 but also supports FP16 through `drafter_options.quant_config.io_dtype`. Omitted target taps are inferred from checkpoint `target_layer_ids + 1` before construction. DFlash2 parses `auto`, `required`, and `off` sharing modes and validates adopted quantized-head node attributes. MTP/DSpark currently accept only `auto`. Structured drafter selections that conflict with legacy drafter paths are rejected.

Runtime fragments are applied after composite configuration generation. Objects merge recursively; arrays replace whole. The validator rejects absent engine or speculative capabilities, invalid allocation and draft limits, provider changes, and changes to graph-required session options. WebGPU paged attention requires a positive `num_blocks` in the effective allocation policy. This may come from legacy extra options or `runtime_config.engine.dynamic_batching.num_blocks`; the runtime overlay takes precedence. Combining `gpu_utilization_factor` with a fixed capacity is rejected; WebGPU always requires a positive fixed capacity.

`runtime_config.engine.dynamic_batching.dflash2_max_snapshots` limits retained windowed DFlash 2 prefix snapshots for a hybrid target with fixed state and prefix caching enabled. It defaults to `1`, accepts integers from `0` through `2147483647`, and uses `0` to disable drafter snapshots without disabling target prefix caching. The runtime caps the count by the shared cache memory budget and target checkpoint capacity (`max_batch_size`). The same field is accepted in `runtime_profiles[].overlay.engine.dynamic_batching`. For a supported paged DFlash 2 export, use either form in the structured builder invocation:

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i target_checkpoint -o output -p int4 -e cuda \
  --builder_config_version 2 --extra_options use_paged_attention=true \
  --drafter_options '{"drafter_type":"dflash2","path":"drafter_checkpoint"}' \
  --runtime_config '{"engine":{"dynamic_batching":{"dflash2_max_snapshots":4}}}'

# From source, at the repository root:
python src/python/py/models/builder.py -i target_checkpoint -o output -p int4 -e cuda \
  --builder_config_version 2 --extra_options use_paged_attention=true \
  --drafter_options '{"drafter_type":"dflash2","path":"drafter_checkpoint"}' \
  --runtime_config '{"runtime_profiles":[{"id":"larger-snapshot-pool","eligibility":{"minimum_total_device_memory_bytes":34359738368},"overlay":{"engine":{"dynamic_batching":{"dflash2_max_snapshots":4}}}}]}'
```

`component_options` is omitted by default. When present, it exports a final-hidden-state backbone without its LM/MTP heads and packages either generic pre-built ONNX heads or one pinned model-specific artifact. Generic declarations require a non-empty `heads` array with unique safe names and filenames; every logical input/output binding must name an actual graph input/output. Revision fields are rejected for generic heads. Model-specific declarations use `model_source` instead of `heads` and require non-empty `artifact_revision` and `base_revision` pins. Component mode cannot be combined with `drafter_options`.

```bash
# Generic head from a wheel:
python -m onnxruntime_genai.models.builder -m Qwen/Qwen3-8B -o output -e cpu \
  --component_options \
  '{"backbone":{"filename":"backbone.onnx"},"heads":[{"name":"classifier","source":"classifier.onnx","inputs":{"hidden_states":"hidden_states"},"outputs":{"scores":"scores"}}]}'

# Equivalent source invocation:
python src/python/py/models/builder.py -m Qwen/Qwen3-8B -o output -e cpu \
  --component_options component-options.json
```

Generic head graphs and confined external-data files are copied only after the complete package plan passes path, symlink, collision, and graph-binding validation. Imported prebuilt graphs retain their original serialized representation and external-data layout; the one-private-data-file export rule applies to generated or reserialized models, not unchanged prebuilt artifacts. A model-specific configuration resolves its separate base model, tokenizer, adapter/head artifact, and immutable revisions before backbone export.

For memory-dependent INT4/INT8 KV-cache graphs, use the [KV-cache variant authoring workflow](../../../../docs/ModelBuilderConfiguration.md#authoring-kv-cache-variants) to export each variant directly from the checkpoint with its own KV scheme and external-data file.

Python callers should pass structured dictionaries to `parse_extra_options` before calling `create_model` with its returned options. The legacy options parameter still takes a list of `KEY=VALUE` strings, not a dictionary. Pass `precision=None` explicitly when deriving it from the structured target. A standalone `create_model` call without prepared Hugging Face metadata still fails.

### Original PyTorch Model from Hugging Face

This scenario is where your PyTorch model is not downloaded locally (either in the default Hugging Face cache directory or in a local folder on disk).

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_save_hf_files

# From source:
python builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_save_hf_files
```

### Original PyTorch Model from Disk

This scenario is where your PyTorch model is already downloaded locally (either in the default Hugging Face cache directory or in a local folder on disk).

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved

# From source:
python builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved
```

### Customized or Finetuned PyTorch Model

This scenario is where your PyTorch model has been customized or finetuned for one of the currently supported model architectures and your model can be loaded in Hugging Face.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files
```

### Quantized PyTorch Model

This scenario is where your PyTorch model is one of the currently supported model architectures, has already been quantized to INT4 precision, and your model can be loaded in the Hugging Face style via [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) or [AutoAWQ](https://github.com/casper-hansen/AutoAWQ).

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p int4 -e execution_provider -c cache_dir_to_store_temp_files

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p int4 -e execution_provider -c cache_dir_to_store_temp_files
```

### GGUF Model

This scenario is where your float16/float32 GGUF model is already on disk.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -i path_to_gguf_file -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files

# From source:
python builder.py -m model_name -i path_to_gguf_file -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files
```

### Extra Options

This scenario is for when you want to have control over some specific settings. The below example shows how you can pass key-value arguments to `--extra_options`.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options filename=decoder.onnx

# From source:
python builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options filename=decoder.onnx
```

To see all available options through `--extra_options`, please use the `help` commands in the `Full Usage` section above.

#### Number of Hidden Layers

This scenario is for when you want to manually set the number of hidden layers that the model builder exports.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider --extra_options num_hidden_layers=4

# From source:
python builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider --extra_options num_hidden_layers=4
```

#### Filename

This scenario is for when you want to use a custom ONNX filename instead of the default `model.onnx`.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider --extra_options filename=decoder.onnx

# From source:
python builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider --extra_options filename=decoder.onnx
```

#### Config Only

This scenario is for when you already have your optimized and/or quantized ONNX model and you need to create the config files to run with ONNX Runtime GenAI.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options config_only=true

# From source:
python builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options config_only=true
```

Afterwards, please open the `genai_config.json` file in the output folder and modify the fields as needed for your model. You should store your ONNX model in the output folder as well.

#### Hugging Face Authentication

This scenario is for when you need to disable the Hugging Face authentication or use a different authentication token than the one stored in [huggingface-cli login](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#huggingface-cli-login).

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options hf_token=false

# From source:
python builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options hf_token=false
```

#### Hugging Face Remote Code

This scenario is for when you need to enable trusting remote code from a Hugging Face repo. The default is `hf_remote=false`, which means `trust_remote_code=False` is used for `transformers.*.from_pretrained()` calls and any Python code shipped inside the repository (referenced by its `auto_map` field) will **not** be executed. Set `hf_remote=true` only for repositories you fully trust, because doing so is equivalent to running arbitrary code from that repository as the current user.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options hf_remote=true

# From source:
python builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options hf_remote=true
```

#### Exclude Embedding Layer

This scenario is for when you want to exclude the embedding layer from your ONNX model.

For LFM2-VL models (`Lfm2VlForConditionalGeneration`, such as LFM2-VL, LFM2.5-VL and d1-3B), this option builds the whole image pipeline: next to the decoder, the builder writes the embedding model (`embeddings.onnx`), the vision model (`vision.onnx`) and `processor_config.json`, and adds their sections to `genai_config.json`. The vision and embedding models are never quantized. The vision model uses the decoder's input/output precision, except that it uses FP16 when the decoder uses BF16.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_embeds=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_embeds=true
```

#### Exclude Language Modeling Head

This scenario is for when you want to exclude the language modeling head from your ONNX model.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_lm_head=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_lm_head=true
```

#### Prune Language Modeling Head

LM-head pruning is disabled by default. Set `prune_lm_head=true` to compute only the logits needed for generation. Standard models then project the hidden state of each row's last real token and output `[batch_size, 1, vocab_size]` logits. The graph takes each row's last real token (the last 1 in `attention_mask`), so padded batches work. Pruned models exported by older builders take the last column instead, which is wrong for the shorter rows of a right-padded batch, so export them again if you run padded batches. Such an export has a `/lm_head/prune/Gather` node and no `/lm_head/prune/GatherElements` node. Paged-attention models add a `logits_indices` input and output `[num_logits, vocab_size]`: the Engine selects one final packed row per request and every row needed for speculative verification, without projecting discarded prompt rows.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options prune_lm_head=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options prune_lm_head=true
```

#### Include Last Hidden States Output

This scenario is for when you want to include the last hidden states as an output to your ONNX model.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options include_hidden_states=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options include_hidden_states=true
```

Note that this is the same as outputting embeddings since the last hidden states are also known as the embeddings.

#### Include Auxiliary Hidden States Output

Set `aux_hidden_state_layers` to a comma-separated list of decoder layer indices to expose the residual streams entering those layers. The selected streams are concatenated along the hidden dimension into an `aux_hidden_states` output for speculative block drafters such as EAGLE3 or DFlash. The default is empty (disabled), and every index must be in `[1, num_hidden_layers)`.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options aux_hidden_state_layers=5,19,33

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options aux_hidden_state_layers=5,19,33
```

#### Build with Paged Attention

This scenario is for when you want to build a model that uses the `PagedAttention` operator so it can be served by ONNX Runtime GenAI's continuous-batching engine. When enabled, the builder replaces `GroupQueryAttention` with `PagedAttention`, packs all sequences of the batch into a single flattened token axis (`input_ids` becomes 1D), stores the KV-cache in paged `[num_blocks, block_size, num_key_value_heads, head_size]` buffers, and removes the `attention_mask` input in favor of the `block_table`, `cumulative_sequence_lengths`, and `past_sequence_lengths` metadata inputs. It also removes `position_ids` when RoPE is fused into attention; architectures that require an external MRoPE op retain packed position IDs (for example, Qwen3.5/3.8 uses `[3, num_tokens]`). Set `prune_lm_head=true` to add a `logits_indices` input, gather the packed hidden states consumed by generation or draft verification, and output `[num_logits, vocab_size]` logits. By default, it projects every packed hidden state and outputs `[num_tokens, vocab_size]` logits.

Paged attention supports CUDA with `fp16` or `bf16` precision and WebGPU with `fp16` precision. Paged exports include the CPU `attention_metadata` input used by the runtime to provide stable query and KV bounds without downloading device sequence lengths in every attention layer. Paged attention cannot be combined with `exclude_embeds` or `exclude_lm_head`. `paged_block_size` defaults to `256` and must be a power of two and at least `16`, matching what the ONNX Runtime PagedAttention op accepts; for models with short and long rotary caches, it must also evenly divide `original_max_position_embeddings`. The vendored FlashAttention paged kernel needs the block to be a multiple of its tile as well (256 for `head_size <= 64`, 128 for `head_size <= 128`, otherwise 64), so a smaller block stays valid but makes ORT fall back to another attention backend. A quantized KV cache is exempt from the tile requirement alone: FlashAttention still serves it, through a dense dequantized path with no page alignment to satisfy. A block drafter (`dflash2_path`/`dspark_path`) shares the target's block size and usually has the smaller head size, so it reaches its tile at a larger block than the target does. `gpu_utilization_factor` defaults to `0.6` and must be greater than `0` and at most `1`. `max_batch_size` defaults to `100` and must be a positive integer no greater than `256`. `paged_chunk_size` must be a positive integer and is written to `search.chunk_size`. It caps the prompt tokens a single request contributes to one step, whereas `max_scheduled_tokens` caps the step as a whole; a value at or above `max_scheduled_tokens` therefore has no effect, and a smaller one lets concurrent prefills interleave rather than letting one request consume the step budget on its own. Models whose sliding-window layers are served from a ring of blocks hold only `paged_chunk_size + window_size - 1` positions, so they require chunked prefill and default to `paged_block_size`. For every other paged model it is written only when passed.

Prefix caching is enabled by default for dynamic batching and can be disabled by setting `engine.dynamic_batching.prefix_caching` to `false` in `genai_config.json`. The builder writes this opt-out automatically for paged sliding-window KV rings, MTP, DSpark, and full-attention DFlash 2 because those layouts mirror auxiliary state per target block and do not yet support prefix caching. DFlash 2 with a positive sliding window uses a fixed-size auxiliary pool and retains the default target prefix caching behavior.

`max_scheduled_tokens` and `num_blocks` are additional `engine.dynamic_batching` knobs and are written only when passed. `max_scheduled_tokens` caps the tokens in one dynamically batched forward pass and therefore caps the peak prefill activation, which is the largest transient in a long-context deployment. `num_blocks` sets the total block budget before auxiliary-cache reservations. The target's resolved pool can be smaller when MTP or a full-attention block drafter reserves cache memory, and all resident requests share that pool, so `num_blocks * paged_block_size` is only the single-request upper bound when the target owns every configured block. `num_blocks` is mutually exclusive with `gpu_utilization_factor`, which is omitted from the config when `num_blocks` is set.

Both options require positive integers. Auxiliary drafter caches share this memory budget, so they can reduce the target's allocated block count.

```bash
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e cuda --extra_options use_paged_attention=true max_scheduled_tokens=2048 num_blocks=128
```

Paged builds can describe non-legacy decoder state in `model.decoder.state_groups`. The Qwen hybrid builder emits exact logical layer IDs for sparse paged KV, fixed convolution state, and fixed recurrent state. Tensor name templates are emitted once under the decoder's `inputs` and `outputs`. Legacy models whose every decoder layer uses paged KV omit the manifest and preserve the existing implicit contract. The hybrid state manifest is experimental and its schema is not yet stable. GenAI's Engine supports Qwen3.8 hybrid paged-KV, fixed convolution, and recurrent-state execution on WebGPU, including 3-plane MRoPE position IDs shaped `[3, num_tokens]`. Packed 3-plane position IDs currently support text-only models; multimodal coordinates are not supported by the Engine.

Qwen3.8 sparse paged layers also expose per-layer `sparse_attention.%d.selected_indices` (`int32`, `[num_tokens, indexer_budget + indexer_compress_ratio - 1]`) and `sparse_attention.%d.selected_counts` (`int32`, `[num_tokens]`) inputs. Their name templates are recorded in the ONNX metadata as `qwen4_exp.selected_index_names` and `qwen4_exp.selected_count_names`. A packed-aware selector must populate these inputs before `SparsePagedAttention`; the current GenAI runtime does not produce them. Non-paged exports compute selections internally with `SparseAttentionIndexer` and do not have this external contract.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e cuda -c cache_dir_to_store_temp_files --extra_options use_paged_attention=true prune_lm_head=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e cuda -c cache_dir_to_store_temp_files --extra_options use_paged_attention=true prune_lm_head=true
```

WebGPU requires a fixed cache capacity. Set `num_blocks` through legacy extra options or `runtime_config.engine.dynamic_batching.num_blocks`. Structured runtime settings take precedence, and the final configuration must retain a positive integer block count.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e webgpu -c cache_dir_to_store_temp_files --extra_options use_paged_attention=true num_blocks=1024

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e webgpu -c cache_dir_to_store_temp_files --extra_options use_paged_attention=true num_blocks=1024
```

```bash
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e webgpu --runtime_config '{"engine":{"dynamic_batching":{"num_blocks":1024}}}' --extra_options use_paged_attention=true
```

#### Disable Windowed KV Cache

By default, sliding-window layers use a reduced KV cache on supported execution providers. With paged attention, eligible local layers use a ring of blocks when the exported model also contains at least one full-context layer. Set `windowed_kv_cache=false` to give every layer a full-length KV cache, which is useful for performance comparisons or compatibility testing. The option defaults to `true` and applies to both paged and non-paged models.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e cuda -c cache_dir_to_store_temp_files --extra_options use_paged_attention=true windowed_kv_cache=false

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e cuda -c cache_dir_to_store_temp_files --extra_options use_paged_attention=true windowed_kv_cache=false
```

#### Enable Shared Embeddings

This scenario is for when you want to enable weight sharing between the embedding layer and the language modeling head. This reduces model size and can improve memory efficiency, especially useful for models with tied embeddings (where `tie_word_embeddings=true` in config.json). For Qwen models with an exported MTP head, compatible non-native LM-head formats also let the MTP embedding lookup reuse the head's LM-head initializer, allowing the shared initializers to be deduplicated across the main and MTP graphs. Native NVFP4 and FP8 LM-head storage cannot be consumed by the generic embedding gather, so those formats retain a separate embedding initializer. Shared embeddings are only valid for models with tied embeddings; setting `shared_embeddings=true` for a model with `tie_word_embeddings=false` will raise a `ValueError`. Shared embeddings are automatically enabled if `tie_word_embeddings=true` in the model's config.json (can be overridden with `shared_embeddings=false`), but cannot be used with `exclude_embeds=true` or `exclude_lm_head=true`.

##### Example 1: INT4 weights + INT4 embeddings (for RTN and K-Quant)

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true algo_config=k_quant

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true algo_config=k_quant
```

##### Example 2: INT4 weights + INT8 embeddings (for RTN Last and K-Quant Last)

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true algo_config=k_quant_last

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true algo_config=k_quant_last
```

##### Example 3: INT4 weights + FP16 embeddings

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true algo_config=rtn nodes_to_exclude=/lm_head/MatMul

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true algo_config=rtn nodes_to_exclude=/lm_head/MatMul
```

##### Example 4: FP16 weights + FP16 embeddings

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p fp16 -e cuda --extra_options shared_embeddings=true

# From source:
python builder.py -m model_name -o path_to_output_folder -p fp16 -e cuda --extra_options shared_embeddings=true
```

#### Enable CUDA Graph Capture

This scenario is for when you want to enable CUDA graph capture for your ONNX model.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options enable_cuda_graph=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options enable_cuda_graph=true
```

#### Enable WebGPU Graph Capture

This scenario is for when you want to enable WebGPU graph capture for your ONNX model.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options enable_webgpu_graph=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options enable_webgpu_graph=true
```

#### Quantization Options

These options apply when exporting weight-only quantized models (`-p int4` for 4-bit weights or `-p int8` for 8-bit weights). Both precisions produce `MatMulNBits` ops and share the quantization options below; the `-p int8` build simply runs the final `MatMulNBits` quantization pass with 8-bit weights (and quantizes MoE experts to 8-bit to match).

> **Note:** A checkpoint that is already quantized keeps its own weight format. Model Builder re-exports those tensors unchanged, so against an FP8/NVFP4 checkpoint `-p int4` reaches only the weights the checkpoint left in floating point, and the exported model is correspondingly larger. The build prints a warning when it detects this. Start from an unquantized checkpoint to quantize the whole model.

> **Note:** These weight-only quantization options were previously prefixed with `int4_` (e.g. `int4_algo_config`, `int4_block_size`). Because they now apply to both int4 and int8 (and future) precisions, the prefix has been dropped (`algo_config`, `block_size`, `is_symmetric`, `accuracy_level`, `op_types_to_quantize`, `nodes_to_exclude`). The old `int4_`-prefixed names are not accepted as deprecated aliases anymore and have been removed.


##### Accuracy Level

This scenario is for when you want to control the accuracy level used for MatMul activation handling.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options accuracy_level=4

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options accuracy_level=4
```

##### MatMul Block Size

This scenario is for when you want to set the block size for MatMul quantization.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options block_size=32

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options block_size=32
```

##### QMoE Block Size

This scenario is for when you want to set the block size for QMoE expert weights. Set `qmoe_block_size` to `0` or a negative value for per-channel quantization. Block-wise QMoE on CPU, CUDA, and WebGPU supports only `32`, `64`, or `128`; TRT-RTX also accepts `16` and `256`. The default is `32` except for TRT-RTX, which defaults to `128`. INT2 and mixed-width CUDA QMoE require block size 64 or 128 and both `hidden_size` and `moe_intermediate_size` to be divisible by it. WebGPU block-wise QMoE requires both `hidden_size` and `moe_intermediate_size` to be divisible by `qmoe_block_size`. Raw block-wise INT4 QMoE requires both dimensions to be even.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options qmoe_block_size=128

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options qmoe_block_size=128
```

On CPU and WebGPU the block-wise expert weights use ONNX Runtime's MatMulNBits encoding with signed block scales: each block's max-magnitude element maps exactly to the lowest quantized value, so no extreme is clipped. This encoding has been validated end to end against the Hugging Face reference on CPU, and the WebGPU QMoE kernel consumes the MatMulNBits layout directly. TRT-RTX keeps the builder's original symmetric encoding with positive block scales until the signed-scale grid has been measured on that execution provider.

The CPU `QMoE` kernel dequantizes the experts to fp32 on every call unless the environment variable `ORT_USE_MLAS_Q4_GEMM_MOE=1` is set at runtime, which enables its AVX-512 MLAS Q4 fast path. That fast path re-quantizes to the same grid as this encoding, so enabling it only changes results at fp32 rounding level.

ONNX Runtime builds that include [microsoft/onnxruntime#32644](https://github.com/microsoft/onnxruntime/pull/32644) run block-wise experts on the MLAS QNBit GEMM (`MatMulNBits`) kernels directly, without the environment variable. For LFM2.5-8B-A1B int4 on a 30-core x86 server, CPU decode went from about 0.8 s per token with the default dequantize-then-GEMM path to 35 ms per token with `ORT_USE_MLAS_Q4_GEMM_MOE=1` and 15 ms per token with #32644.

##### QMoE Weights Prepacked

`qmoe_weights_prepacked` accepts `-1` (default), `0`, or `1`. For integer CUDA QMoE, `-1`/`1` export the CUTLASS layout and `0` exports raw weights. Other EPs retain their existing raw encodings. NVFP4 uses legacy raw storage for `-1`/`0` and requires explicit `1` for its newer K-packed layout. FP8 block weights retain their native loader layout. This option never queries the export host GPU.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options qmoe_weights_prepacked=0

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options qmoe_weights_prepacked=0
```

##### MatMulNBits Weights Prepacked

`matmulnbits_weights_prepacked` accepts `0` (default, raw), `1` (SM80 layout), or `2` (exact SM90 layout). It applies only to CUDA FP16/BF16 inputs without `g_idx`, and does not depend on the export host GPU. Select the layout for the deployment GPU: mode `2` is Hopper SM90-only, not Blackwell SM100/SM120/SM121. Ineligible nodes remain raw, including non-default zero points, shared weight buffers, and SM90 block size `32` (SM90 supports `64`/`128`).

The builder writes `ep.cuda.fpa_intb_gemm=1` automatically for prepacked exports so that nodes the prepack pass skipped because their `N`, `K`, or `block_size` is unsupported use the same kernel family. This setting is optional for the eligible nodes that were prepacked.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options matmulnbits_weights_prepacked=1

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options matmulnbits_weights_prepacked=1
```

Set `enable_cuda_fpa_intb_gemm=true` to select the same CUDA kernel family independently of offline weight packing. The default is `false`; the option writes `ep.cuda.fpa_intb_gemm=1` to the decoder session options and only applies to the CUDA EP.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options enable_cuda_fpa_intb_gemm=true

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options enable_cuda_fpa_intb_gemm=true
```

##### Device Allocator for Initializers

`use_device_allocator_for_initializers=true` opts into `session.use_device_allocator_for_initializers=1` (default `false`). Paged Qwen3.8 sessions enable it automatically. Initializers then bypass the ONNX Runtime arena, avoiding retention of the original allocation when a kernel replaces a weight during `PrePack`.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options use_device_allocator_for_initializers=true

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options use_device_allocator_for_initializers=true
```

##### Independent Token Embedding Precision

`embedding_quant_type=int4|int8` requests independent token-embedding precision, defaulting to the model precision. INT4 currently requires INT4 model precision; INT8 requires symmetric quantization. An explicit request enables Gather quantization and is rejected if it conflicts with an embedding exclusion. An INT8 request does not reuse an INT4 shared LM-head buffer.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options embedding_quant_type=int8

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options embedding_quant_type=int8
```

##### Is Symmetric

This scenario is for when you want to choose symmetric (`int4`) or asymmetric (`uint4`) weight quantization.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options is_symmetric=false

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options is_symmetric=false
```

##### Op Types To Quantize

This scenario is for when you want to target specific operator types for quantization.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options op_types_to_quantize=MatMul/Gather

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options op_types_to_quantize=MatMul/Gather
```

##### Nodes To Exclude

This scenario is for when you want to skip quantizing specific nodes.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options nodes_to_exclude=/lm_head/MatMul,/model/embed_tokens/Gather

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options nodes_to_exclude=/lm_head/MatMul,/model/embed_tokens/Gather
```

##### Algo Config

This scenario is for when you want to select the base quantization algorithm mode.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options algo_config=default

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options algo_config=default
```

Supported base values are: `default`, `rtn`, `k_quant`.

The legacy compound values `rtn_last`, `k_quant_last`, `k_quant_mixed`, and `k_quant_linear` are still accepted as aliases for a base method plus a `matmul_mixed_precision`.

##### Mixed Precision

This scenario is for when you want to quantize selected MatMul groups with a different quant type than the int4 body, independently from the base quantization algorithm.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options algo_config=default matmul_mixed_precision=last_matmul:int8

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options algo_config=k_quant matmul_mixed_precision=last_matmul:int8,mixed_layers:int8
```

`matmul_mixed_precision` is a comma-separated list of `selector:quant_type` pairs. Supported selectors are:

- `last_matmul`: The last MatMul, such as `/lm_head/MatMul` (the single largest, output-sensitive weight).
- `mixed_layers`: The most quantization-sensitive layers, using the mixed strategy from llama.cpp.
- `linear_attn`: Linear-attention projections and their MLPs, for hybrid attention models.

Supported quant types are `int4` and `int8`. Using a quant-type name (rather than a bare bit count) lets new schemes such as `fp8`/`fp4` be added without introducing a new option. `matmul_mixed_precision` is orthogonal to `algo_config` and can be combined with any base method.

##### Use QDQ Pattern for Quantization

This scenario is for when you want to use the QDQ pattern when quantizing the model.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options use_qdq=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options use_qdq=true
```

This option is not supported with `-p int8` because 8-bit `MatMulNBits` is QOperator-only.

##### Choose the MoE Quantization Type in QMoE

This scenario is for when you want to select the quantization scheme for MoE (QMoE) layers via the single `moe_quant_type` option. Supported values include:

- `int2`: 2-bit integer QMoE weights on CUDA (`expert_weight_bits=2`, `quant_type="int"`). Requires block size 64 or 128 that divides both `hidden_size` and `moe_intermediate_size`.
- `int4`: 4-bit integer QMoE weights (`expert_weight_bits=4`, `quant_type="int"`).
- `int8`: 8-bit integer QMoE weights (`expert_weight_bits=8`, `quant_type="int"`).
- `mxfp4`: MXFP4 QMoE weights on the CUDA EP (`quant_type="fp4"`, `expert_weight_bits=4`, `block_size=32`): 4-bit e2m1 weights with ue8m0 (float8e8m0) block scales and a per-expert float32 global scale. Requires an ONNX Runtime build with `onnxruntime_USE_FP4_QMOE=ON`, `precision=int4` or `precision=int8` with symmetric integer quantization, and is only supported on the CUDA EP.

This single option replaces the older per-type flags so new quantization schemes can be added without introducing a new flag each time. The `use_8bits_moe` flag is deprecated (use `moe_quant_type=int8`).

```bash
# From wheel (8-bit integer QMoE):
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options moe_quant_type=int8

# From source (8-bit integer QMoE):
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options moe_quant_type=int8
```

```bash
# From wheel (MXFP4 QMoE on CUDA):
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p int8 -e cuda -c cache_dir_to_store_temp_files --extra_options moe_quant_type=mxfp4

# From source (MXFP4 QMoE on CUDA):
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p int8 -e cuda -c cache_dir_to_store_temp_files --extra_options moe_quant_type=mxfp4
```

##### Quantize the KV Cache

This scenario is for when you want to quantize the KV cache via the `kv_cache_quant_scheme` option. Quantized KV cache is only supported for the CPU and CUDA execution providers. Supported values are:

- `none` (default): no KV cache quantization.
- `int8_per_tensor` / `int8_per_channel`: 8-bit integer KV cache.
- `int4_per_tensor` / `int4_per_channel`: 4-bit integer KV cache.
- `fp8_per_tensor` / `fp8_per_channel`: FP8 (float8e4m3fn) KV cache.
- `int4_per_token` / `int8_per_token`: dynamic per-token, per-head scales with CUDA paged attention.

The `int8`/`int4`/`fp8` prefix selects the KV cache bit width and the `per_tensor`/`per_channel` suffix selects the scale granularity.

Static per-tensor and per-channel scales are supplied through a required calibration file. Per-token schemes instead allocate FP16 scale caches, compute their values on device, and reject `kv_cache_scale_file`:

- `kv_cache_scale_file`: path to a JSON file with calibrated per-layer scales in the form `{"scales": {"k_scales": [...per layer...], "v_scales": [...per layer...]}, "layer_ids": [...model layer IDs...]}`. Each per-layer entry is a scalar (`per_tensor`) or a length-`(num_kv_heads * head_size)` vector (`per_channel`). `layer_ids` maps each scale entry to its model layer; it is contiguous for dense models and sparse for hybrid models where only full-attention layers own a KV cache. This option is required when `kv_cache_quant_scheme` is enabled.

A calibrated scale is `threshold / qmax`, so it depends on the bit width it was calibrated for. The file may record that divisor as a top-level `"qmax"` (128 for int8, 8 for int4, 448 for fp8 e4m3); the builder then rescales the entries to the requested scheme, so a single calibration file can serve several bit widths. Without `"qmax"` the scales are used as-is and must already match the requested scheme — reusing an int8 file for `int4_*` without it silently under-scales by 16x and clips the cache.

The scale file is produced by the `kv_cache_calibration` module, which runs a baseline (non-quantized) build of the same model over a calibration corpus and captures the `present.*.key`/`present.*.value` tensors:

```bash
# 1. Build the baseline (no kv_cache_quant_scheme) used for calibration:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_baseline_folder -p precision -e cuda -c cache_dir_to_store_temp_files

# 2. Calibrate the scales:
python -m onnxruntime_genai.models.quantization.kv_cache_calibration --model path_to_baseline_folder --tokenizer path_to_local_folder_on_disk --out path_to_scales.json --quant-type int8_per_channel
```

Then rebuild with the quantized KV cache:

```bash
# From wheel (int8 per-channel KV cache with calibrated scales):
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e cuda -c cache_dir_to_store_temp_files --extra_options kv_cache_quant_scheme=int8_per_channel kv_cache_scale_file=path_to_scales.json

# From source (int8 per-channel KV cache with calibrated scales):
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e cuda -c cache_dir_to_store_temp_files --extra_options kv_cache_quant_scheme=int8_per_channel kv_cache_scale_file=path_to_scales.json
```

##### Quantize the KV Cache with Paged Attention

`kv_cache_quant_scheme` can be combined with `use_paged_attention=true`. In that case the paged KV cache blocks (`[num_blocks, block_size, num_kv_heads, head_size]`) are allocated in the quantized element type and the `PagedAttention` op receives the `k_scale`/`v_scale` initializers plus the matching `k_quant_type`/`v_quant_type` attributes.

The CUDA paged path supports INT4, INT8, and FP8. INT4 requires an ORT build with `onnxruntime_USE_INT4_KV_CACHE=ON` and stores two signed values per UINT8 byte, with physical cache width `(head_size + 1) // 2`. Per-channel scales retain shape `(num_kv_heads, 1, head_size)`. The `int4_per_token` and `int8_per_token` schemes use separate FP16 scale caches of shape `[num_blocks, block_size, num_kv_heads]`, with input/output aliases emitted in `genai_config.json`. They require a GenAI runtime that supports the scale-cache bindings.

`kv_cache_rotation=hadamard` rotates Q/K after normalization and RoPE, rotates V before caching, and applies the inverse transform to the attention output. The default is `none`. Rotation is CUDA-paged-only, requires head size 16, 32, 64, 128, or 256, and rejects per-channel scales. This is an opt-in quantization mode; application quality must be evaluated on representative inputs.

```bash
# From wheel (no calibration file):
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e cuda --extra_options use_paged_attention=true kv_cache_quant_scheme=int4_per_token kv_cache_rotation=hadamard

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e cuda --extra_options use_paged_attention=true kv_cache_quant_scheme=int4_per_token kv_cache_rotation=hadamard
```

```bash
# From wheel (paged attention + int8 per-channel KV cache):
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e cuda -c cache_dir_to_store_temp_files --extra_options use_paged_attention=true kv_cache_quant_scheme=int8_per_channel kv_cache_scale_file=path_to_scales.json

# From source (paged attention + fp8 per-tensor KV cache):
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e cuda -c cache_dir_to_store_temp_files --extra_options use_paged_attention=true kv_cache_quant_scheme=fp8_per_tensor kv_cache_scale_file=path_to_scales.json
```

#### FP32 I/O for WebGPU EP

This scenario is for when you want to force FP32 model I/O for WebGPU (useful for GPUs without FP16 support on WebGPU, such as GTX 10xx).

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p int4 -e webgpu -c cache_dir_to_store_temp_files --extra_options use_webgpu_fp32=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p int4 -e webgpu -c cache_dir_to_store_temp_files --extra_options use_webgpu_fp32=true
```

#### BF16 I/O for CUDA EP

This scenario is for when you want to use BF16 I/O precision in quantized ONNX models for CUDA / TRT-RTX.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p int4 -e cuda -c cache_dir_to_store_temp_files --extra_options use_cuda_bf16=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p int4 -e cuda -c cache_dir_to_store_temp_files --extra_options use_cuda_bf16=true
```

#### LoRA Models

This scenario is where you have a finetuned model with LoRA adapters and your model can be loaded in the Hugging Face style via [PEFT](https://github.com/huggingface/peft).

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e execution_provider -c cache_dir_to_store_temp_files --extra_options adapter_path=path_to_adapter_files

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e execution_provider -c cache_dir_to_store_temp_files --extra_options adapter_path=path_to_adapter_files
```

Base weights should be located in `path_to_local_folder_on_disk` and adapter weights should be located in `path_to_adapter_files`.

### Unit Testing Models

This scenario is where your PyTorch model is already downloaded locally (either in the default Hugging Face cache directory or in a local folder on disk). If it is not already downloaded locally, here is an example of how you can download it.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
cache_dir = "cache_dir_to_save_hf_files"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
model.save_pretrained(cache_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.save_pretrained(cache_dir)
```

#### Option 1: Use the model builder directly

This option is the simplest but it will download another copy of the PyTorch model onto disk to accommodate the change in the number of hidden layers.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider --extra_options num_hidden_layers=4

# From source:
python builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider --extra_options num_hidden_layers=4
```

#### Option 2: Edit the config.json file on disk and then run the model builder

1. Navigate to where the PyTorch model and its associated files are saved on disk.
2. Modify `num_hidden_layers` in `config.json` to your desired target (e.g. 4 layers).
3. Run the below command for the model builder.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved

# From source:
python builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved
```

## Design

Please read the [design document](DESIGN.md) for more details and for how to contribute.
