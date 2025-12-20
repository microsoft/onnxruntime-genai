# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Run the model builder to create the desired ONNX model.
"""

import argparse
import os
import textwrap
from typing import Any

import onnx_ir as ir
import torch
from builders import (
    ChatGLMModel,
    ErnieModel,
    Gemma2Model,
    Gemma3Model,
    GemmaModel,
    GPTOSSModel,
    GraniteModel,
    LlamaModel,
    MistralModel,
    Model,
    NemotronModel,
    OLMoModel,
    Phi3MiniLongRoPEModel,
    Phi3MiniModel,
    Phi3MoELongRoPEModel,
    Phi3SmallLongRoPEModel,
    Phi3SmallModel,
    Phi3VModel,
    Phi4MMModel,
    PhiModel,
    Qwen3Model,
    Qwen25VLTextModel,
    QwenModel,
    SmolLM3Model,
)
from transformers import (
    AutoConfig,
)


def check_extra_options(kv_pairs, execution_provider):
    """
    Check key-value pairs and set values correctly
    """
    bools = [
        "int4_is_symmetric",
        "exclude_embeds",
        "exclude_lm_head",
        "include_hidden_states",
        "enable_cuda_graph",
        "enable_webgpu_graph",
        "use_8bits_moe",
        "use_qdq",
        "use_webgpu_fp32",
        "use_cuda_bf16",
        "shared_embeddings",
        "hf_remote",
        "disable_qkv_fusion",
    ]
    for key in bools:
        if key in kv_pairs:
            if kv_pairs[key] in {"false", "False", "0"}:
                kv_pairs[key] = False
            elif kv_pairs[key] in {"true", "True", "1"}:
                kv_pairs[key] = True
            else:
                raise ValueError(f"{key} must be false/False/0 or true/True/1.")

    if "int4_op_types_to_quantize" in kv_pairs:
        op_types_to_quantize = ()
        for op_type in kv_pairs["int4_op_types_to_quantize"].split("/"):
            op_types_to_quantize += (op_type,)
        kv_pairs["int4_op_types_to_quantize"] = op_types_to_quantize

    if "int4_nodes_to_exclude" in kv_pairs:
        nodes_to_exclude = []
        for node in kv_pairs["int4_nodes_to_exclude"].split(","):
            nodes_to_exclude.append(node)
        kv_pairs["int4_nodes_to_exclude"] = nodes_to_exclude

    if "exclude_lm_head" in kv_pairs and "include_hidden_states" in kv_pairs:
        # 'exclude_lm_head' is for when 'hidden_states' are outputted and 'logits' are not outputted
        # 'include_hidden_states' is for when 'hidden_states' are outputted and 'logits' are outputted
        raise ValueError(
            "Both 'exclude_lm_head' and 'include_hidden_states' cannot be used together. Please use only one of them at once."
        )

    if kv_pairs.get("enable_webgpu_graph", False) and execution_provider != "webgpu":
        print(
            "WARNING: enable_webgpu_graph is only supported with WebGPU execution provider. Disabling enable_webgpu_graph."
        )
        kv_pairs["enable_webgpu_graph"] = False


def parse_extra_options(kv_items, execution_provider):
    """
    Parse key-value pairs that are separated by '='
    """
    kv_pairs = {}

    if kv_items:
        for kv_str in kv_items:
            kv = kv_str.split("=")
            kv_pairs[kv[0].strip()] = kv[1].strip()

    print(f"Extra options: {kv_pairs}")
    check_extra_options(kv_pairs, execution_provider)
    return kv_pairs


def parse_hf_token(hf_token):
    """
    Returns the authentication token needed for Hugging Face.
    Token is obtained either from the user or the environment.
    """
    if hf_token.lower() in {"false", "0"}:
        # Default is `None` for disabling authentication
        return None

    if hf_token.lower() in {"true", "1"}:
        # Return token in environment
        return True

    # Return user-provided token as string
    return hf_token


def set_io_dtype(precision, execution_provider, extra_options) -> ir.DataType:
    int4_cpu = precision == "int4" and execution_provider == "cpu"
    fp32_webgpu = execution_provider == "webgpu" and extra_options.get("use_webgpu_fp32", False)
    bf16_cuda = precision == "int4" and execution_provider == "cuda" and extra_options.get("use_cuda_bf16", False)

    if precision in {"int8", "fp32"} or int4_cpu or fp32_webgpu:
        # FP32 precision
        return ir.DataType.FLOAT

    if precision == "bf16" or bf16_cuda:
        # BF16 precision
        return ir.DataType.BFLOAT16

    # FP16 precision
    return ir.DataType.FLOAT16


def set_onnx_dtype(precision: str, extra_options: dict[str, Any]) -> ir.DataType:
    if precision == "int4":
        return ir.DataType.INT4 if extra_options.get("int4_is_symmetric", True) else ir.DataType.UINT4

    to_onnx_dtype = {
        "fp32": ir.DataType.FLOAT,
        "fp16": ir.DataType.FLOAT16,
        "bf16": ir.DataType.BFLOAT16,
    }
    return to_onnx_dtype[precision]


@torch.no_grad
def create_model(
    model_name,
    input_path,
    output_dir,
    precision,
    execution_provider,
    cache_dir,
    **extra_options,
):
    if execution_provider == "NvTensorRtRtx":
        execution_provider = "trt-rtx"
        extra_options["use_qdq"] = True

    # Create cache and output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Load model config
    extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": cache_dir}
    hf_name = input_path if os.path.isdir(input_path) else model_name
    hf_token = parse_hf_token(extra_options.get("hf_token", "true"))
    hf_remote = extra_options.get("hf_remote", True)

    config = AutoConfig.from_pretrained(hf_name, token=hf_token, trust_remote_code=hf_remote, **extra_kwargs)
    if "adapter_path" in extra_options:
        from peft import PeftConfig

        peft_config = PeftConfig.from_pretrained(
            extra_options["adapter_path"],
            token=hf_token,
            trust_remote_code=hf_remote,
            **extra_kwargs,
        )
        config.update(peft_config.__dict__)

    # Set input/output precision of ONNX model
    io_dtype = set_io_dtype(precision, execution_provider, extra_options)
    onnx_dtype = set_onnx_dtype(precision, extra_options)
    config_only = "config_only" in extra_options

    # List architecture options in alphabetical order
    if config.architectures[0] == "ChatGLMForConditionalGeneration" or config.architectures[0] == "ChatGLMModel":
        # Quantized ChatGLM model has ChatGLMForConditionalGeneration as architecture whereas HF model as the latter
        config.bos_token_id = 1
        config.hidden_act = "swiglu"
        onnx_model = ChatGLMModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
        onnx_model.model_type = "chatglm"
    elif config.architectures[0] == "Ernie4_5_ForCausalLM":
        onnx_model = ErnieModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "GemmaForCausalLM":
        onnx_model = GemmaModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Gemma2ForCausalLM":
        print(
            "WARNING: This model loses accuracy with float16 precision. It is recommended to set `--precision bf16` or `--precision int4 --extra_options use_cuda_bf16=true` by default."
        )
        onnx_model = Gemma2Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Gemma3ForCausalLM":
        print(
            "WARNING: This model loses accuracy with float16 precision. It is recommended to set `--precision bf16` or `--precision int4 --extra_options use_cuda_bf16=true` by default."
        )
        onnx_model = Gemma3Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
        onnx_model.model_type = "gemma3_text"
    elif config.architectures[0] == "Gemma3ForConditionalGeneration":
        text_config = config.text_config
        for key in text_config:
            if not hasattr(config, key):
                setattr(config, key, getattr(text_config, key))
        print(
            "WARNING: This model loses accuracy with float16 precision. It is recommended to set `--precision bf16` or `--precision int4 --extra_options use_cuda_bf16=true` by default."
        )
        print(
            "WARNING: This is only generating the text component of the model. Setting `--extra_options exclude_embeds=true` by default."
        )
        extra_options["exclude_embeds"] = True
        onnx_model = Gemma3Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "GptOssForCausalLM":
        print("WARNING: This model only supports symmetric quantization for `QMoE`.")
        delattr(config, "quantization_config")
        onnx_model = GPTOSSModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "GraniteForCausalLM":
        onnx_model = GraniteModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "LlamaForCausalLM":
        onnx_model = LlamaModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "MistralForCausalLM":
        onnx_model = MistralModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "NemotronForCausalLM":
        onnx_model = NemotronModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "OlmoForCausalLM":
        onnx_model = OLMoModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "PhiForCausalLM":
        onnx_model = PhiModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif (
        config.architectures[0] == "Phi3ForCausalLM"
        and config.max_position_embeddings == config.original_max_position_embeddings
    ):
        onnx_model = Phi3MiniModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif (
        config.architectures[0] == "Phi3ForCausalLM"
        and config.max_position_embeddings != config.original_max_position_embeddings
    ):
        onnx_model = Phi3MiniLongRoPEModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif (
        config.architectures[0] == "PhiMoEForCausalLM"
        and config.max_position_embeddings != config.original_max_position_embeddings
    ):
        print(
            "WARNING: This model only works for CUDA currently because `MoE` is only supported for CUDA in ONNX Runtime. Setting `--execution_provider cuda` by default."
        )
        print(
            "WARNING: This model currently only supports the quantized version. Setting `--precision int4` by default."
        )
        execution_provider = "cuda"
        onnx_dtype = set_onnx_dtype("int4", extra_options)
        onnx_model = Phi3MoELongRoPEModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif (
        config.architectures[0] == "Phi3SmallForCausalLM"
        and config.max_position_embeddings == config.original_max_position_embeddings
    ):
        onnx_model = Phi3SmallModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif (
        config.architectures[0] == "Phi3SmallForCausalLM"
        and config.max_position_embeddings != config.original_max_position_embeddings
    ):
        onnx_model = Phi3SmallLongRoPEModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Phi3VForCausalLM":
        print(
            "WARNING: This is only generating the text component of the model. Setting `--extra_options exclude_embeds=true` by default."
        )
        extra_options["exclude_embeds"] = True
        onnx_model = Phi3VModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Phi4MMForCausalLM":
        print(
            "WARNING: This is only generating the text component of the model. Setting `--extra_options exclude_embeds=true` by default."
        )
        extra_options["exclude_embeds"] = True
        onnx_model = Phi4MMModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Qwen2ForCausalLM":
        onnx_model = QwenModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Qwen3ForCausalLM":
        onnx_model = Qwen3Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "SmolLM3ForCausalLM":
        onnx_model = SmolLM3Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Qwen2_5_VLForConditionalGeneration":
        text_config = config.text_config
        for key in text_config:
            if not hasattr(config, key):
                setattr(config, key, getattr(text_config, key))
        print(
            "WARNING: This is only generating the text component of the model. Setting `--extra_options exclude_embeds=true` by default."
        )
        extra_options["exclude_embeds"] = True
        onnx_model = Qwen25VLTextModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config_only:
        # Create base Model class to guess model attributes
        onnx_model = Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    else:
        raise NotImplementedError(f"The {hf_name} model is not currently supported.")

    if not config_only:
        # Make ONNX model
        onnx_model.make_model(input_path)

        # Save ONNX model
        onnx_model.save_model(output_dir)

    # Make GenAI config
    onnx_model.make_genai_config(hf_name, extra_kwargs, output_dir)

    # Copy Hugging Face processing files to output folder
    onnx_model.save_processing(hf_name, extra_kwargs, output_dir)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        default=None,
        help="Model name in Hugging Face. Do not use if providing an input path to a Hugging Face directory in -i/--input.",
    )

    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default="",
        help=textwrap.dedent("""\
            Input model source. Currently supported options are:
                hf_path: Path to folder on disk containing the Hugging Face config, model, tokenizer, etc.
                gguf_path: Path to float16/float32 GGUF file on disk containing the GGUF model
            """),
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to folder to store ONNX model and additional files (e.g. GenAI config, external data files, etc.)",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        choices=["int4", "bf16", "fp16", "fp32"],
        help="Precision of model",
    )

    parser.add_argument(
        "-e",
        "--execution_provider",
        required=True,
        choices=["cpu", "cuda", "dml", "webgpu", "NvTensorRtRtx"],
        help="Execution provider to target with precision of model (e.g. FP16 CUDA, INT4 CPU, INT4 WebGPU)",
    )

    parser.add_argument(
        "-c",
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_dir"),
        help="Cache directory for Hugging Face files and temporary ONNX external data files",
    )

    parser.add_argument(
        "--extra_options",
        required=False,
        metavar="KEY=VALUE",
        nargs="+",
        help=textwrap.dedent("""\
            Key value pairs for various options. Currently supports:
                int4_accuracy_level = 1/2/3/4: Specify the minimum accuracy level for activation of MatMul in int4 quantization.
                    4 is int8, which means input A of int4 quantized MatMul is quantized to int8 and input B is upcasted to int8 for computation.
                    3 is bf16.
                    2 is fp16.
                    1 is fp32.
                    Default is 4 for the CPU EP and 0 for non-CPU EPs.
                int4_block_size = 16/32/64/128/256: Specify the block size for int4 quantization.
                    Default value is 32.
                int4_is_symmetric = Quantize the weights symmetrically. Default is true.
                    If true, quantization is done to int4. If false, quantization is done to uint4.
                int4_op_types_to_quantize = MatMul/Gather: Specify op types to target for int4 quantization.
                    Use this option when you want to quantize specific ops.
                    Separate the op types with a '/' when passing them here (e.g. int4_op_types_to_quantize=MatMul/Gather)
                int4_nodes_to_exclude = Specify nodes to exclude from int4 quantization.
                    Use this option when you want to exclude certain nodes from being quantized.
                    Separate the node names with a ',' when passing them here (e.g. int4_nodes_to_exclude=/lm_head/MatMul,/model/embed_tokens/Gather)
                int4_algo_config = Method for int4 quantization. Default is 'default'.
                    Currently supported options are: 'default', 'rtn', 'rtn_last', 'k_quant', 'k_quant_mixed', 'k_quant_last'.
                    default = algo_config passed to MatMulNBitsQuantizer is None. Quantizer uses default RTN algorithm. All MatMuls are quantized as int4.(different node naming conventions to `rtn`)
                    rtn = RTN algorithm for int4 quantization.
                    rtn_last = RTN algorithm where only the last MatMul (/lm_head/MatMul) is quantized as int8. Other MatMuls are quantized as int4.
                    k_quant = k_quant algorithm for int4 quantization.
                    k_quant_mixed = k_quant algorithm with mixed precision (int4 + int8).
                    k_quant_last = k_quant algorithm where only the last MatMul (/lm_head/MatMul) is quantized as int8. Other MatMuls are quantized as int4.
                shared_embeddings = Enable weight sharing between embedding and LM head layers. Default is false.
                    Use this option to share weights and reduce model size by eliminating duplicate weights.
                    For quantized models (INT4/UINT4): Shares quantized weights using GatherBlockQuantized. Only works with rtn and k_quant algorithms, and cannot be used if LM head is excluded.
                    For float models (FP16/FP32/BF16): Shares float weights using Gather. Works for pure FP models or INT4 models where LM head is excluded from quantization.
                num_hidden_layers = Manually specify the number of layers in your ONNX model.
                    Used for unit testing purposes.
                filename = Filename for ONNX model (default is 'model.onnx').
                    For models with multiple components, each component is exported to its own ONNX model.
                    The filename for each component will be '<filename>_<component-name>.onnx' (ex: '<filename>_encoder.onnx', '<filename>_decoder.onnx').
                config_only = Generate config and pre/post processing files only.
                    Use this option when you already have your optimized and/or quantized ONNX model.
                hf_token = false/token: Use this to manage authentication with Hugging Face.
                    Default behavior is to use the authentication token stored by `huggingface-cli login`.
                    If false, authentication with Hugging Face will be disabled.
                    If token, you can provide a custom authentication token that differs from the one stored in your environment.
                    If you have already authenticated via `huggingface-cli login`, you do not need to use this flag because Hugging Face has already stored your authentication token for you.
                hf_remote = Use this to manage trusting remote code in Hugging Face repos.
                    Default behavior is set to true. If false, remote code stored in Hugging Face repos will not be used.
                exclude_embeds = Remove embedding layer from your ONNX model.
                    Use this option when you want to remove the embedding layer from within your ONNX model.
                    Instead of `input_ids`, you will have `inputs_embeds` as the input to your ONNX model.
                exclude_lm_head = Remove language modeling head from your ONNX model.
                    Use this option when you want to remove the language modeling head from within your ONNX model.
                    Instead of `logits`, you will have `hidden_states` as the output to your ONNX model.
                include_hidden_states = Include hidden states as output from your ONNX model.
                    Use this option when you want to have the hidden states as an output from your ONNX model.
                    In addition to `logits`, you will have `hidden_states` as an output to your ONNX model.
                enable_cuda_graph = Enable CUDA graph capture during inference. Default is false.
                    If enabled, all nodes being placed on the CUDA EP is the prerequisite for the CUDA graph to be used correctly.
                    It is not guaranteed that CUDA graph be enabled as it depends on the model and the graph structure.
                enable_webgpu_graph = Enable WebGPU graph capture during inference. Default is false.
                    If enabled, the model structure will be optimized for WebGPU graph execution.
                    This affects attention mask reformatting and position IDs handling.
                use_8bits_moe = Use 8-bit quantization for MoE layers. Default is false.
                    If true, the QMoE op will use 8-bit quantization. If false, the QMoE op will use 4-bit quantization.
                use_qdq = Use the QDQ decomposition for ops.
                    Use this option when you want to use quantize-dequantize ops. For example, you will have a quantized MatMul op instead of the MatMulNBits op.
                use_webgpu_fp32 = Use FP32 I/O precision for WebGPU EP.
                    Use this option to enable GPUs that do not support FP16 on WebGPU (e.g. GTX 10xx).
                use_cuda_bf16 = Use BF16 I/O precision in quantized ONNX models for CUDA EP.
                    Use this option to create quantized ONNX models that use BF16 precision.
                adapter_path = Path to folder on disk containing the adapter files (adapter_config.json and adapter model weights).
                    Use this option for LoRA models.
            """),
    )

    args = parser.parse_args()
    print(
        "Valid precision + execution provider combinations are: FP32 CPU, FP32 CUDA, FP16 CUDA, FP16 DML, BF16 CUDA, FP16 TRT-RTX, INT4 CPU, INT4 CUDA, INT4 DML, INT4 WebGPU"
    )
    return args


if __name__ == "__main__":
    args = get_args()
    extra_options = parse_extra_options(args.extra_options, args.execution_provider)
    create_model(
        args.model_name,
        args.input,
        args.output,
        args.precision,
        args.execution_provider,
        args.cache_dir,
        **extra_options,
    )
