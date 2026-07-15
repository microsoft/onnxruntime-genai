# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime_genai as og
from onnx import TensorProto
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


DEFAULT_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"


def _load_config(model_path):
    with open(Path(model_path) / "genai_config.json", "r") as config_file:
        return json.load(config_file)


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_pixel_values(processor, image_path):
    with Image.open(image_path) as image:
        processed = processor(images=image.convert("RGB"), return_tensors="np")
    if "pixel_values" not in processed:
        raise RuntimeError("Processor output does not contain pixel_values.")
    return _as_numpy(processed["pixel_values"])


def _model_input_dtype(model_path, input_name):
    dtype_map = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
    }
    model = onnx.load(model_path, load_external_data=False)
    for input_value in model.graph.input:
        if input_value.name == input_name:
            element_type = input_value.type.tensor_type.elem_type
            if element_type not in dtype_map:
                type_name = TensorProto.DataType.Name(element_type)
                raise RuntimeError(f"Unsupported {input_name} type: {type_name}")
            return dtype_map[element_type]
    raise RuntimeError(f"Input {input_name} was not found in {model_path}.")


def _tokenizer(model_path, processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


def _initial_decoder_ids(model_path, processor, task_prompt, decoder_start):
    if not task_prompt:
        return np.array([[decoder_start]], dtype=np.int32)

    processed = processor(text=task_prompt, return_tensors="np")
    if "input_ids" in processed:
        prompt_ids = np.asarray(processed["input_ids"], dtype=np.int32)
    else:
        encoded = _tokenizer(model_path, processor)(
            task_prompt, return_tensors="np", add_special_tokens=True
        )
        prompt_ids = np.asarray(encoded["input_ids"], dtype=np.int32)

    start_ids = np.full((prompt_ids.shape[0], 1), decoder_start, dtype=np.int32)
    return np.concatenate([start_ids, prompt_ids], axis=1)


def _configure_model(args, model_dir):
    provider = args.execution_provider
    if args.ep_path:
        if provider in {"follow_config", "cpu"}:
            raise ValueError("--ep_path requires an explicit non-CPU execution provider")
        og.register_execution_provider_library(provider, args.ep_path)

    config = og.Config(str(model_dir))
    if provider != "follow_config":
        config.clear_providers()
        if provider != "cpu":
            config.append_provider(provider)

    if provider == "NvTensorRTRTXExecutionProvider":
        config.set_provider_option(provider, "enable_cuda_graph", "1")
        if args.cache_dir:
            cache_dir = Path(args.cache_dir).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            config.set_provider_option(provider, "nv_runtime_cache_path", str(cache_dir))

    return og.Model(config)


def run_parse(args):
    model_dir = Path(args.model_path).resolve()
    config_json = _load_config(model_dir)
    model_config = config_json["model"]
    decoder_config = model_config["decoder"]
    vision_config = model_config["vision"]

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    encoder_path = model_dir / vision_config.get("filename", "encoder.onnx")
    pixel_values = _load_pixel_values(processor, args.image_file)
    pixel_values = np.ascontiguousarray(
        pixel_values.astype(_model_input_dtype(encoder_path, "pixel_values"), copy=False)
    )

    decoder_start = int(model_config.get("bos_token_id", 2))
    input_ids = np.ascontiguousarray(
        _initial_decoder_ids(
            model_dir, processor, args.task_prompt, decoder_start
        )
    )
    prompt_length = input_ids.shape[1]
    cache_capacity = int(
        decoder_config.get("cache_sequence_length", model_config["context_length"])
    )
    max_length = min(prompt_length + args.max_new_tokens, cache_capacity)
    if max_length <= prompt_length:
        raise ValueError("The prompt already fills the configured KV-cache capacity")

    model = _configure_model(args, model_dir)
    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=max_length,
        num_beams=1,
        past_present_share_buffer=True,
    )
    generator = og.Generator(model, params)

    inputs = og.NamedTensors()
    inputs["input_ids"] = input_ids
    inputs["pixel_values"] = pixel_values
    generator.set_inputs(inputs)

    generated = []
    while not generator.is_done():
        generator.generate_next_token()
        generated.append(int(generator.get_next_tokens()[0]))

    tokens = np.asarray(generated, dtype=np.int32)
    print(_tokenizer(model_dir, processor).decode(tokens, skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser(
        description="Run a cached Nemotron Parse package through ONNX Runtime GenAI."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Folder containing encoder.onnx, decoder_prefill.onnx, decoder.onnx, and genai_config.json.",
    )
    parser.add_argument("--image_file", required=True, help="Input image to parse.")
    parser.add_argument(
        "-e",
        "--execution_provider",
        default="follow_config",
        choices=[
            "follow_config",
            "cpu",
            "cuda",
            "CUDAExecutionProvider",
            "NvTensorRTRTXExecutionProvider",
        ],
        help="Execution provider for the complete encoder/prefill/decode pipeline.",
    )
    parser.add_argument(
        "--ep_path", default="", help="Path to an execution provider plug-in library."
    )
    parser.add_argument(
        "--cache_dir", default="", help="Optional TRT-RTX runtime cache directory."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024, help="Maximum tokens to decode."
    )
    parser.add_argument(
        "--task_prompt", default=DEFAULT_TASK_PROMPT, help="Nemotron Parse task prompt."
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_file):
        raise FileNotFoundError(args.image_file)
    run_parse(args)


if __name__ == "__main__":
    main()
