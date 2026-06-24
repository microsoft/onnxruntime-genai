# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


_REGISTERED_EP_LIBS = set()


def _shape_profile(feed):
    return ",".join(f"{name}:{'x'.join(str(dim) for dim in value.shape)}" for name, value in feed.items())


def _provider_name(execution_provider):
    if execution_provider == "cpu":
        return "CPUExecutionProvider"
    if execution_provider == "cuda":
        return "CUDAExecutionProvider"
    return execution_provider


def _make_session(model_path, execution_provider, ep_path, feed, max_feed=None, cache_dir=None):
    provider = _provider_name(execution_provider)
    if ep_path and (provider, ep_path) not in _REGISTERED_EP_LIBS:
        ort.register_execution_provider_library(provider, ep_path)
        _REGISTERED_EP_LIBS.add((provider, ep_path))

    session_options = ort.SessionOptions()
    if provider == "NvTensorRTRTXExecutionProvider":
        max_feed = feed if max_feed is None else max_feed
        options = {
            "enable_cuda_graph": "0",
            "nv_profile_min_shapes": _shape_profile(feed),
            "nv_profile_opt_shapes": _shape_profile(feed),
            "nv_profile_max_shapes": _shape_profile(max_feed),
        }
        if cache_dir:
            options["nv_runtime_cache_path"] = str(cache_dir)

        ep_devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
        if not ep_devices:
            raise RuntimeError(f"{provider} was not found after registration.")
        session_options.add_provider_for_devices(ep_devices, options)
        return ort.InferenceSession(str(model_path), sess_options=session_options)

    return ort.InferenceSession(str(model_path), sess_options=session_options, providers=[provider])


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_pixel_values(processor, image_path, image_height, image_width):
    image = Image.open(image_path).convert("RGB").resize((image_width, image_height))
    processed = processor(images=image, return_tensors="np")
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
            elem_type = input_value.type.tensor_type.elem_type
            if elem_type not in dtype_map:
                raise RuntimeError(f"Unsupported input type {TensorProto.DataType.Name(elem_type)} for {input_name}.")
            return dtype_map[elem_type]
    raise RuntimeError(f"Input {input_name} was not found in {model_path}.")


def _load_config(model_path):
    config_path = Path(model_path) / "genai_config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def _decode_tokens(model_path, processor, token_ids):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def run_parse(args):
    model_dir = Path(args.model_path).resolve()
    config = _load_config(model_dir)
    model_config = config["model"]
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    encoder_model_path = model_dir / model_config["vision"].get("filename", "encoder.onnx")
    image_height = int(model_config["vision"].get("image_height", 768))
    image_width = int(model_config["vision"].get("image_width", 768))
    pixel_values = _load_pixel_values(processor, args.image_file, image_height, image_width)
    pixel_values = pixel_values.astype(_model_input_dtype(encoder_model_path, "pixel_values"), copy=False)
    encoder_feed = {"pixel_values": pixel_values}
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None

    encoder_session = _make_session(
        encoder_model_path,
        args.execution_provider,
        args.ep_path,
        encoder_feed,
        cache_dir=cache_dir / "encoder" if cache_dir else None,
    )
    encoder_hidden_states = encoder_session.run(None, encoder_feed)[0]

    decoder_start = int(model_config.get("bos_token_id", model_config.get("decoder_start_token_id", 2)))
    eos_token_id = model_config.get("eos_token_id", model_config.get("pad_token_id"))
    if isinstance(eos_token_id, list):
        eos_token_ids = {int(token) for token in eos_token_id}
    elif eos_token_id is None:
        eos_token_ids = set()
    else:
        eos_token_ids = {int(eos_token_id)}

    decoder_ids = np.array([[decoder_start]], dtype=np.int64)
    decoder_feed = {
        "decoder_input_ids": decoder_ids,
        "decoder_attention_mask": np.ones_like(decoder_ids, dtype=np.int64),
        "encoder_hidden_states": encoder_hidden_states,
    }
    decoder_max_ids = np.ones((decoder_ids.shape[0], args.max_new_tokens + 1), dtype=np.int64)
    decoder_max_feed = {
        "decoder_input_ids": decoder_max_ids,
        "decoder_attention_mask": np.ones_like(decoder_max_ids, dtype=np.int64),
        "encoder_hidden_states": encoder_hidden_states,
    }
    decoder_session = _make_session(
        model_dir / model_config["decoder"].get("filename", "decoder.onnx"),
        args.execution_provider,
        args.ep_path,
        decoder_feed,
        max_feed=decoder_max_feed,
        cache_dir=cache_dir / "decoder" if cache_dir else None,
    )

    generated = []
    for _ in range(args.max_new_tokens):
        decoder_feed = {
            "decoder_input_ids": decoder_ids,
            "decoder_attention_mask": np.ones_like(decoder_ids, dtype=np.int64),
            "encoder_hidden_states": encoder_hidden_states,
        }
        logits = decoder_session.run(None, decoder_feed)[0]
        next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        if next_token in eos_token_ids:
            break
        generated.append(next_token)
        decoder_ids = np.concatenate([decoder_ids, np.array([[next_token]], dtype=np.int64)], axis=1)

    text = _decode_tokens(model_dir, processor, generated)
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Run an exported Nemotron Parse encoder/decoder ONNX package.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Folder containing encoder.onnx, decoder.onnx, and genai_config.json.",
    )
    parser.add_argument("--image_file", type=str, required=True, help="Input image to parse.")
    parser.add_argument("-e", "--execution_provider", type=str, default="cuda",
                        choices=["cpu", "cuda", "CUDAExecutionProvider", "NvTensorRTRTXExecutionProvider"],
                        help="Execution provider for encoder.onnx and decoder.onnx.")
    parser.add_argument("--ep_path", type=str, default="", help="Path to an execution provider plug-in DLL.")
    parser.add_argument("--cache_dir", type=str, default="", help="Optional TRT-RTX runtime cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to decode.")
    args = parser.parse_args()

    if not os.path.exists(args.image_file):
        raise FileNotFoundError(args.image_file)
    run_parse(args)


if __name__ == "__main__":
    main()
