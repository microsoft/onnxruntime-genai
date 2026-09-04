# ONNX Runtime GenAI Model Builder

This folder contains the model builder for quickly creating optimized and quantized ONNX models within a few minutes that run with ONNX Runtime GenAI.

# Contents

- [Current Support](#current-support)
- [Usage](#usage)
  - [Full Usage](#full-usage)
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
    - [Build a DFlash 2 Block Drafter](#build-a-dflash-2-block-drafter)
    - [Build a DSpark Block Drafter](#build-a-dspark-block-drafter)
    - [Disable Windowed KV Cache](#disable-windowed-kv-cache)
    - [Enable Shared Embeddings](#enable-shared-embeddings)
    - [Enable CUDA Graph Capture](#enable-cuda-graph-capture)
    - [Export a ModelOpt or compressed-tensors NVFP4/FP8 Checkpoint](#export-a-modelopt-or-compressed-tensors-nvfp4fp8-checkpoint)
    - [MTP Head (Qwen3.6)](#mtp-head-qwen36)
    - [Compact State Updates (Qwen3.5/3.8)](#compact-state-updates-qwen3538)
    - [Select the Qwen3.5/3.8 Recurrent Operator](#select-the-qwen3538-recurrent-operator)
    - [Enable WebGPU Graph Capture](#enable-webgpu-graph-capture)
    - [Disable QKV Projections Fusion](#disable-qkv-projections-fusion)
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

LM-head pruning is disabled by default. Set `prune_lm_head=true` to compute only the logits needed for generation. Standard models then project the final hidden state and output `[batch_size, 1, vocab_size]` logits. Paged-attention models project the final packed hidden state for each sequence and output `[batch_size, vocab_size]` logits.

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

This scenario is for when you want to build a model that uses the `PagedAttention` operator so it can be served by ONNX Runtime GenAI's continuous-batching engine. When enabled, the builder replaces `GroupQueryAttention` with `PagedAttention`, packs all sequences of the batch into a single flattened token axis (`input_ids` becomes 1D), stores the KV-cache in paged `[num_blocks, block_size, num_key_value_heads, head_size]` buffers, and removes the `attention_mask` input in favor of the `block_table`, `cumulative_sequence_lengths`, and `past_sequence_lengths` metadata inputs. It also removes `position_ids` when RoPE is fused into attention; architectures that require an external MRoPE op retain packed position IDs (for example, Qwen3.5/3.8 uses `[3, num_tokens]`). Set `prune_lm_head=true` to select the final packed hidden state for each sequence before the LM head and output `[batch_size, vocab_size]` logits. By default, it projects every packed hidden state and outputs `[num_tokens, vocab_size]` logits.

Paged attention supports CUDA with `fp16` or `bf16` precision and WebGPU with `fp16` precision. Paged exports include the CPU `attention_metadata` input used by the runtime to provide stable query and KV bounds without downloading device sequence lengths in every attention layer. Paged attention cannot be combined with `exclude_embeds` or `exclude_lm_head`. `paged_block_size` defaults to `256` and must be a positive multiple of `256`; for models with short and long rotary caches, it must evenly divide `original_max_position_embeddings`. `gpu_utilization_factor` defaults to `0.6` and must be greater than `0` and at most `1`. `max_batch_size` defaults to `100` and must be a positive integer no greater than `256`. `paged_chunk_size` defaults to `paged_block_size`, must be a positive integer, and is written to `search.chunk_size`; it applies only to models whose sliding-window layers are served from a ring of blocks, which hold `paged_chunk_size + window_size - 1` positions and therefore require chunked prefill.

Paged builds can describe non-legacy decoder state in `model.decoder.state_groups`. The Qwen hybrid builder emits exact logical layer IDs for sparse paged KV, fixed convolution state, and fixed recurrent state. Tensor name templates are emitted once under the decoder's `inputs` and `outputs`. Legacy models whose every decoder layer uses paged KV omit the manifest and preserve the existing implicit contract. The hybrid state manifest is experimental and its schema is not yet stable. It requires coordinated Engine runtime work beyond the current onnxruntime-genai#2454 head and is not compatible with the merged runtime on its own. In particular, the runtime must supply packed multimodal position IDs with shape `[3, num_tokens]`; the current `VarlenDecoderIO` does not create that input.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e cuda -c cache_dir_to_store_temp_files --extra_options use_paged_attention=true prune_lm_head=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e cuda -c cache_dir_to_store_temp_files --extra_options use_paged_attention=true prune_lm_head=true
```

#### Build a DFlash 2 Block Drafter

Set `dflash2_path` to a DFlash 2 checkpoint to export an auxiliary `dflash2.onnx` block drafter beside a Qwen3.5 MoE target model. The target must use paged attention, and `aux_hidden_state_layers` must exactly match the drafter checkpoint's `target_layer_ids`. The drafter reuses the target's embedding and LM-head initializers, so both checkpoints must use compatible tensors.

`dflash2_num_draft_tokens` optionally overrides how many tokens the drafter proposes per step. It must be a positive integer no greater than the draft checkpoint's block size minus the anchor token; that checkpoint limit is also the default.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_target_model -o path_to_output_folder -p fp16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true aux_hidden_state_layers=1,11,21 dflash2_path=path_to_dflash2_checkpoint dflash2_num_draft_tokens=4

# From source:
python builder.py -i path_to_target_model -o path_to_output_folder -p fp16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true aux_hidden_state_layers=1,11,21 dflash2_path=path_to_dflash2_checkpoint dflash2_num_draft_tokens=4
```

#### Build a DSpark Block Drafter

Set `dspark_path` to a DSpark checkpoint to export an auxiliary `dspark.onnx` block drafter beside a Qwen3.5 or Qwen3.8 target model. The target must use paged attention. SpecForge identifies the target layers whose outputs are tapped, while `aux_hidden_state_layers` identifies residual streams entering layers, so each configured auxiliary layer must be one greater than the corresponding `target_layer_ids` entry in the DSpark checkpoint. The drafter reuses the target's embedding and LM-head initializers. `dspark_path` and `dflash2_path` are mutually exclusive, and selecting DSpark replaces rather than accompanies the target's MTP head.

`dspark_num_draft_tokens` optionally overrides how many tokens the drafter proposes per step. It must be at least `2` and no greater than the checkpoint's trained `block_size`; the default is that checkpoint block size. `dspark_top_k` controls how many candidates the lattice keeps per block slot; it defaults to `16` and must be a positive integer no greater than the drafter vocabulary size. The score tensor and its host copy grow as `dspark_top_k` squared, so keep this value near the default unless a larger lattice has been benchmarked for the intended workload.

The DSpark transformer body executes in BF16 even when the target uses FP16, so the selected execution provider and hardware must support BF16. Its KV cache consumes additional GPU memory: a full-attention DSpark layer adds a key/value entry for every target cache block and therefore reduces the target's maximum resident context under a fixed memory budget. A checkpoint configured uniformly for sliding-window attention uses its positive `sliding_window` instead; because the exported graph carries a single window for every layer, a checkpoint that windows only some of its layers (`max_window_layers` other than `0`, or a mixed `layer_types`) is rejected.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_target_model -o path_to_output_folder -p fp16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true aux_hidden_state_layers=1,11,21 dspark_path=path_to_dspark_checkpoint dspark_num_draft_tokens=4 dspark_top_k=16

# From source:
python builder.py -i path_to_target_model -o path_to_output_folder -p fp16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true aux_hidden_state_layers=1,11,21 dspark_path=path_to_dspark_checkpoint dspark_num_draft_tokens=4 dspark_top_k=16
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

This scenario is for when you want to enable weight sharing between the embedding layer and the language modeling head. This reduces model size and can improve memory efficiency, especially useful for models with tied embeddings (where `tie_word_embeddings=true` in config.json). Shared embeddings are only valid for models with tied embeddings; setting `shared_embeddings=true` for a model with `tie_word_embeddings=false` will raise a `ValueError`. Shared embeddings are automatically enabled if `tie_word_embeddings=true` in the model's config.json (can be overridden with `shared_embeddings=false`), but cannot be used with `exclude_embeds=true` or `exclude_lm_head=true`.

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

#### Export a ModelOpt or compressed-tensors NVFP4/FP8 Checkpoint

This scenario is for when your Qwen3.6 MoE checkpoint has already been quantized by NVIDIA TensorRT Model Optimizer and you want to carry its NVFP4/FP8 tensors into the ONNX model instead of dequantizing to fp16 and re-quantizing to int4.

```bash
# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p bf16 -e cuda -c cache_dir_to_store_temp_files
```

When `config.json` declares `quant_method=modelopt` or `quant_method=compressed-tensors`, the builder preserves the checkpoint's original quantized data types automatically: routed experts use native NVFP4 `QMoE`, dense NVFP4 modules use `MatMulBlockQuantizedFp4Weight`, and FP8 attention projections use `MatMulBlockQuantizedFp8Weight`. The compressed-tensors loader accepts packed NVFP4 weights with reciprocal global scales and scalar or per-channel FP8 weight scales. KV-cache quantization remains explicit and requires calibrated scales supplied through `kv_cache_quant_scheme` and `kv_cache_scale_file`. CUDA linear-attention gate fusion is enabled automatically.

The `--precision` argument controls the unquantized tensors and model I/O; it does not change the checkpoint's native FP8/NVFP4 tensors. ModelOpt and compressed-tensors export require the CUDA EP and an ONNX Runtime build that provides the corresponding contrib ops. For CPU, CUDA, and WebGPU, the builder replaces each shared-expert output `Mul` and routed/shared `Add` pair with `com.microsoft::GatedAdd`; other execution providers retain the portable `Mul` + `Add` graph.

#### MTP Head (Qwen3.6)

When a Qwen3.5 MoE configuration declares one or more MTP layers with `mtp_num_hidden_layers`, the builder exports the multi-token-prediction head for self-speculative decoding. An auxiliary `mtp.onnx` (plus its `mtp.onnx.data`) is generated alongside the main model, and the main model automatically exposes the hidden states consumed by the MTP head. Models without declared MTP layers do not produce this file or an MTP section in `genai_config.json`.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files
```

Qwen3.5 MoE checkpoints (`Qwen3_5MoeForConditionalGeneration`) that declare MTP layers must ship `mtp.*` weights in their safetensors. For those models, the builder rejects `exclude_lm_head=true` and `prune_lm_head=true` because the exported MTP workflow requires the main LM head. The MTP weights are read directly from the source safetensors because Hugging Face `transformers` discards them on load. To disable MTP during inference, remove the `model.mtp` section from `genai_config.json`; rebuilding the ONNX models is not required.

By default the MTP head inherits the main model's settings. For a ModelOpt or compressed-tensors checkpoint, the builder preserves each original MTP tensor format: native NVFP4 linears and experts remain NVFP4, FP8 attention projections remain FP8, and unquantized tensors follow the requested graph precision.

To configure the MTP model independently, use `mtp_quant_config` with an inline JSON object or a JSON file using the structured `QuantConfig` schema. Its `io_dtype`, `weights`, `moe`, and `runtime` targets are independent. For example, `mtp_quant_config='{"io_dtype":"bf16","weights":{"type":"int4","block_size":64},"moe":{"type":"nvfp4"}}'` exports INT4 dense MTP MatMuls, NVFP4 MTP experts, and BF16 I/O.

Supplying `mtp_quant_config` explicitly dequantizes native ModelOpt or compressed-tensors MTP tensors before applying the MTP configuration. Dense `weights.type` supports integer or unquantized formats; use `moe.type=mxfp4/nvfp4` to select FP4 experts independently. Without an explicit MTP configuration, pre-quantized NVFP4 dense weights remain `MatMulBlockQuantizedFp4Weight` and native FP8 attention projections remain `MatMulBlockQuantizedFp8Weight`.

The head always exports `hidden_states_out` (its own post-final-norm hidden state), which a multi-token loop feeds back as the next chained draft's `hidden_states` input. It is required for `num_speculative_tokens > 1` and ignored otherwise.

A multi-token verify forward can additionally carry a window of recurrent/conv states so a partial accept can be handled by cropping instead of replaying the main model. Pass `state_window=W` (with `W >= num_speculative_tokens + 1`) to widen `past/present_key_values.%d.{conv,recurrent}_state` to `[W, B, ...]` and emit the matching attribute on `CausalConvWithState` / `LinearAttention`. This requires ONNX Runtime kernels that understand the attribute; leave it at the default `0` otherwise.

#### Compact State Updates (Qwen3.5/3.8)

Paged Qwen3.5/3.8 exports can capture compact convolution and GatedDeltaNet transitions for speculative tokens instead of returning full recurrent-state checkpoints. Set `state_update_capacity=N` to reserve updates for up to `N` tokens. The capacity defaults to `0` (disabled) and requires `use_paged_attention=true`. It must be an integer from `0` through `8`, matching the kernel and runtime-parser bound, because the kernel packs every captured transition for a layer into a single fixed-width capsule output. Paged Qwen3.5/3.8 exports use GatedDeltaNet regardless of `linear_attn_op` and support CUDA with `fp16` or `bf16` model I/O. When enabled, `genai_config.json` records the capacity, the `state_update_capture_count` and `state_update_active` input bindings, and the per-layer convolution-value and recurrent-capsule output templates. All compact state-update inputs and outputs are omitted when `state_update_capacity=0`.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p bf16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true state_update_capacity=3

# From source:
python builder.py -m model_name -o path_to_output_folder -p bf16 -e cuda -c cache_dir_for_hf_files --extra_options use_paged_attention=true state_update_capacity=3
```

#### Select the Qwen3.5/3.8 Recurrent Operator

This scenario is for when you want to choose which contrib operator implements the linear-attention layers of a non-paged Qwen3.5/3.8 export. `linear_attn_op` accepts `linear_attention` (the default), which emits `CausalConvWithState` + `LinearAttention`, or `gated_delta_net`, which emits `CausalConvWithState` + `GatedDeltaNet` with an FP32 V-major recurrent state and native Qwen gate arithmetic from the raw `A_log`/`dt_bias` initializers. `gated_delta_net` is CUDA-only, requires `state_window=0`, and supports `fp16` or `bf16` model I/O. Paged exports (`use_paged_attention=true`) always use GatedDeltaNet and therefore also require CUDA; they ignore this option. The default `linear_attention` path requires an ONNX Runtime kernel that implements the selected contrib operator.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p bf16 -e cuda -c cache_dir_for_hf_files --extra_options linear_attn_op=gated_delta_net

# From source:
python builder.py -m model_name -o path_to_output_folder -p bf16 -e cuda -c cache_dir_for_hf_files --extra_options linear_attn_op=gated_delta_net
```

#### Enable WebGPU Graph Capture

This scenario is for when you want to enable WebGPU graph capture for your ONNX model.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options enable_webgpu_graph=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options enable_webgpu_graph=true
```

#### Disable QKV Projections Fusion

This scenario is for when you want to keep Q/K/V projections in the attention layer separate instead of fusing them into a single packed MatMul operation.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options disable_qkv_fusion=true

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options disable_qkv_fusion=true
```

#### Disable QK Norm GQA Fusion in CUDA or WebGPU

QK Norm GQA fusion is enabled by default for CUDA and WebGPU when GroupQueryAttention is used and rotary embedding can be fused into the attention op. In this mode, Q/K norm weights are passed directly into GroupQueryAttention instead of emitting explicit Q/K normalization nodes.

This scenario is for when you want to disable that fusion and keep explicit Q/K normalization nodes in the graph.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e cuda -c cache_dir_to_store_temp_files --extra_options fuse_qk_norm_gqa=false

# From source:
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e webgpu -c cache_dir_to_store_temp_files --extra_options fuse_qk_norm_gqa=false
```

#### Quantization Options

These options apply when exporting weight-only quantized models (`-p int4` for 4-bit weights or `-p int8` for 8-bit weights). Both precisions produce `MatMulNBits` ops and share the quantization options below; the `-p int8` build simply runs the final `MatMulNBits` quantization pass with 8-bit weights (and quantizes MoE experts to 8-bit to match).

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

This scenario is for when you want to set the block size for QMoE expert weights.
Set `qmoe_block_size` to `0` or a negative value for per-channel quantization. CUDA block-wise QMoE supports only `32`, `64`, or `128`; the default is `32` except for TRT-RTX, which defaults to `128`.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options qmoe_block_size=128

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options qmoe_block_size=128
```

##### QMoE Weights Prepacked

This scenario is for when you want to control the CUDA QMoE expert weight layout. The default value is `-1`, which lets the builder choose the layout automatically. Use `0` to export raw weights and let CUDA prepack them at runtime, or `1` to export CUTLASS-prepacked weights.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options qmoe_weights_prepacked=0

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options qmoe_weights_prepacked=0
```

##### MatMulNBits Weights Prepacked

This scenario is for when you want to control the CUDA MatMulNBits (int4/int8) weight layout. The default value is `0`, which exports raw blockwise weights. Use `1` to export the SM80/Ampere `fpA_intB` prepacked layout, or `2` to export the SM90/Hopper `fpA_intB` prepacked layout. This only applies to the CUDA EP, and an offline-prepacked model must be run with `ORT_FPA_INTB_GEMM` enabling the relevant nbits.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options matmulnbits_weights_prepacked=1

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options matmulnbits_weights_prepacked=1
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

This scenario is for when you want to select the quantization scheme for MoE (QMoE) layers via the single `moe_quant_type` option. Supported values are `int4` (default), `int8`, and `mxfp4`:

- `int4`: 4-bit integer QMoE weights (`expert_weight_bits=4`, `quant_type="int"`).
- `int8`: 8-bit integer QMoE weights (`expert_weight_bits=8`, `quant_type="int"`).
- `mxfp4`: MXFP4 QMoE weights on the CUDA EP (`quant_type="fp4"`, `expert_weight_bits=4`, `block_size=32`): 4-bit e2m1 weights with ue8m0 (float8e8m0) block scales and a per-expert float32 global scale. Requires an ONNX Runtime build with `onnxruntime_USE_FP4_QMOE=ON`, `precision=int4` with symmetric INT4 quantization, and is only supported on the CUDA EP.

This single option replaces the older per-type flags so new quantization schemes can be added without introducing a new flag each time. The `use_8bits_moe` flag is deprecated (use `moe_quant_type=int8`).

```bash
# From wheel (8-bit integer QMoE):
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options moe_quant_type=int8

# From source (8-bit integer QMoE):
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options moe_quant_type=int8
```

```bash
# From wheel (MXFP4 QMoE on CUDA):
python -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p int4 -e cuda -c cache_dir_to_store_temp_files --extra_options moe_quant_type=mxfp4

# From source (MXFP4 QMoE on CUDA):
python builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p int4 -e cuda -c cache_dir_to_store_temp_files --extra_options moe_quant_type=mxfp4
```

##### Quantize the KV Cache

This scenario is for when you want to quantize the KV cache via the `kv_cache_quant_scheme` option. Quantized KV cache is only supported for the CPU and CUDA execution providers. Supported values are:

- `none` (default): no KV cache quantization.
- `int8_per_tensor` / `int8_per_channel`: 8-bit integer KV cache.
- `int4_per_tensor` / `int4_per_channel`: 4-bit integer KV cache.
- `fp8_per_tensor` / `fp8_per_channel`: FP8 (float8e4m3fn) KV cache.

The `int8`/`int4`/`fp8` prefix selects the KV cache bit width and the `per_tensor`/`per_channel` suffix selects the scale granularity.

The scales applied to the KV cache are supplied through a required calibration file:

- `kv_cache_scale_file`: path to a JSON file with calibrated per-layer scales in the form `{"scales": {"k_scales": [...per layer...], "v_scales": [...per layer...]}, "layer_ids": [...model layer IDs...]}`. Each per-layer entry is a scalar (`per_tensor`) or a length-`(num_kv_heads * head_size)` vector (`per_channel`). `layer_ids` maps each scale entry to its model layer; it is contiguous for dense models and sparse for hybrid models where only full-attention layers own a KV cache. This option is required when `kv_cache_quant_scheme` is enabled.

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

`kv_cache_quant_scheme` can be combined with `use_paged_attention=true`. In that case the paged KV cache blocks
(`[num_blocks, block_size, num_kv_heads, head_size]`) are allocated in the quantized element type and the
`PagedAttention` op receives the `k_scale`/`v_scale` initializers plus the matching `k_quant_type`/`v_quant_type`
attributes.

Only `int8_*` and `fp8_*` are supported on the paged path; `int4_*` is rejected because `PagedAttention` has no
sub-byte cache backend. Per-channel scales are emitted with the `(num_kv_heads, 1, head_size)` shape that
`PagedAttention` requires.

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
