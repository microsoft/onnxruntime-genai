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
    - [Enable Shared Embeddings](#enable-shared-embeddings)
    - [Enable CUDA Graph Capture](#enable-cuda-graph-capture)
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

This scenario is for when you want to prune the language modeling head to only compute the last token's logits.

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

#### Enable Shared Embeddings

This scenario is for when you want to enable weight sharing between the embedding layer and the language modeling head. This reduces model size and can improve memory efficiency, especially useful for models with tied embeddings (where `tie_word_embeddings=true` in config.json). Shared embeddings are automatically enabled if `tie_word_embeddings=true` in the model's config.json (can be overridden with `shared_embeddings=false`), but cannot be used with `exclude_embeds=true` or `exclude_lm_head=true`.

##### Example 1: INT4 weights + INT4 embeddings (for RTN and K-Quant)

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true int4_algo_config=k_quant

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true int4_algo_config=k_quant
```

##### Example 2: INT4 weights + INT8 embeddings (for RTN Last and K-Quant Last)

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true int4_algo_config=k_quant_last

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true int4_algo_config=k_quant_last
```

##### Example 3: INT4 weights + FP16 embeddings

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true int4_algo_config=rtn int4_nodes_to_exclude=/lm_head/MatMul

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e cuda --extra_options shared_embeddings=true int4_algo_config=rtn int4_nodes_to_exclude=/lm_head/MatMul
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

##### Accuracy Level

This scenario is for when you want to control the accuracy level used for MatMul activation handling.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_accuracy_level=4

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_accuracy_level=4
```

##### MatMul Block Size

This scenario is for when you want to set the block size for MatMul quantization.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_block_size=32

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_block_size=32
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
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_is_symmetric=false

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_is_symmetric=false
```

##### Op Types To Quantize

This scenario is for when you want to target specific operator types for quantization.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_op_types_to_quantize=MatMul/Gather

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_op_types_to_quantize=MatMul/Gather
```

##### Nodes To Exclude

This scenario is for when you want to skip quantizing specific nodes.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_nodes_to_exclude=/lm_head/MatMul,/model/embed_tokens/Gather

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_nodes_to_exclude=/lm_head/MatMul,/model/embed_tokens/Gather
```

##### Algo Config

This scenario is for when you want to select the base quantization algorithm mode.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_algo_config=default

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_algo_config=default
```

Supported base values are: `default`, `rtn`, `k_quant`.

The legacy compound values `rtn_last`, `k_quant_last`, `k_quant_mixed`, and `k_quant_linear` are still accepted as aliases for a base method plus a `matmul_mixed_precision`.

##### Mixed Precision

This scenario is for when you want to quantize selected MatMul groups with a different quant type than the int4 body, independently from the base quantization algorithm.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_algo_config=default matmul_mixed_precision=last_matmul:int8

# From source:
python builder.py -m model_name -o path_to_output_folder -p int4 -e execution_provider --extra_options int4_algo_config=k_quant matmul_mixed_precision=last_matmul:int8,mixed_layers:int8
```

`matmul_mixed_precision` is a comma-separated list of `selector:quant_type` pairs. Supported selectors are:

- `last_matmul`: The last MatMul, such as `/lm_head/MatMul` (the single largest, output-sensitive weight).
- `mixed_layers`: The most quantization-sensitive layers, using the mixed strategy from llama.cpp.
- `linear_attn`: Linear-attention projections and their MLPs, for hybrid attention models.

Supported quant types are `int4` and `int8`. Using a quant-type name (rather than a bare bit count) lets new schemes such as `fp8`/`fp4` be added without introducing a new option. `matmul_mixed_precision` is orthogonal to `int4_algo_config` and can be combined with any base method.

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
