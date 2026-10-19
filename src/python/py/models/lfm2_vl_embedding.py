# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Finish an LFM2-VL export: build its embedding model and wire in the vision model.

python -m onnxruntime_genai.models.lfm2_vl_embedding -i HF_CHECKPOINT -o MODEL [--vision vision_encoder.onnx]

MODEL is a model builder output made with --extra_options exclude_embeds=true, with the vision model
(and its data files) already copied into it. Writes embeddings.onnx, processor_config.json and the
embedding and vision sections of genai_config.json. Rerun it whenever the builder rewrites MODEL.

This is a separate tool, not builder output, because the builder never sees the vision model: it comes
from a published ONNX export, and the tool has to run after it is in MODEL to check its output type and
width. Like split_cpu_embedding.py, it finishes a builder output and leaves the builder unchanged.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open

EMBEDDING_WEIGHT = "model.language_model.embed_tokens.weight"


def tensor_type(values, name):
    value = next((v for v in values if v.name == name), None)
    if value is None:
        raise ValueError(f"No tensor named {name}")
    return value.type.tensor_type


def load_table(checkpoint, dtype):
    index = checkpoint / "model.safetensors.index.json"
    filename = json.loads(index.read_text())["weight_map"][EMBEDDING_WEIGHT] if index.exists() else "model.safetensors"
    with safe_open(checkpoint / filename, "pt") as f:
        table = f.get_tensor(EMBEDDING_WEIGHT)
    return table.to({TensorProto.FLOAT: torch.float32, TensorProto.FLOAT16: torch.float16}[dtype]).numpy()


def make_embedding_model(table, image_token_id, features_type):
    """input_ids, image_features -> inputs_embeds, with the features at the image token positions."""
    hidden_size = table.shape[1]
    embeds_type = helper.np_dtype_to_tensor_dtype(table.dtype)

    def const(name, value):
        return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.array(value, np.int64)))

    features = "image_features"
    nodes = []
    if features_type != embeds_type:
        nodes.append(helper.make_node("Cast", [features], ["cast_features"], to=embeds_type))
        features = "cast_features"
    nodes += [
        helper.make_node("Gather", ["embed_tokens.weight", "input_ids"], ["embeds"], axis=0),
        helper.make_node("Shape", ["embeds"], ["embeds_shape"]),
        const("rows", [-1, hidden_size]),
        helper.make_node("Reshape", ["embeds", "rows"], ["flat_embeds"]),
        const("image_token_id", image_token_id),
        helper.make_node("Equal", ["input_ids", "image_token_id"], ["is_image"]),
        const("flat", [-1]),
        helper.make_node("Reshape", ["is_image", "flat"], ["flat_is_image"]),
        helper.make_node("NonZero", ["flat_is_image"], ["positions"]),
        helper.make_node("Transpose", ["positions"], ["indices"], perm=[1, 0]),
        helper.make_node("ScatterND", ["flat_embeds", "indices", features], ["flat_out"]),
        helper.make_node("Reshape", ["flat_out", "embeds_shape"], ["inputs_embeds"]),
    ]
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("image_features", features_type, ["num_image_tokens", hidden_size]),
    ]
    outputs = [
        helper.make_tensor_value_info("inputs_embeds", embeds_type, ["batch_size", "sequence_length", hidden_size])
    ]
    graph = helper.make_graph(
        nodes, "lfm2_vl_embedding", inputs, outputs, [numpy_helper.from_array(table, "embed_tokens.weight")]
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def make_processor_config(image_processor):
    """The Hugging Face Lfm2VlImageProcessor without image splitting (tiling)."""
    interpolation = {2: "LINEAR", 3: "CUBIC"}  # PIL resample codes of the Hugging Face image processor
    resample = image_processor["resample"]
    if resample not in interpolation:
        raise ValueError(f"Unsupported resample {resample}: expected 2 (bilinear) or 3 (bicubic)")
    patch_size, merge_size = image_processor["encoder_patch_size"], image_processor["downsample_factor"]
    pixels_per_token = (patch_size * merge_size) ** 2

    def step(name, kind, **attrs):
        return {"operation": {"name": name, "type": kind, **({"attrs": attrs} if attrs else {})}}

    size = image_processor["size"]
    transforms = [
        step("decode_image", "DecodeImage", color_space="RGB"),
        step(
            "resize",
            "Resize",
            height=size["height"],
            width=size["width"],
            interpolation=interpolation[resample],
            smart_resize=1,
            min_pixels=image_processor["min_image_tokens"] * pixels_per_token,
            max_pixels=image_processor["max_image_tokens"] * pixels_per_token,
            patch_size=patch_size,
            merge_size=merge_size,
        ),
        step("rescale", "Rescale", rescale_factor=image_processor["rescale_factor"]),
        step("normalize", "Normalize", mean=image_processor["image_mean"], std=image_processor["image_std"]),
        step("to_channel_first", "Permute3D", dims=[2, 0, 1]),
        step("image_sizes", "PixtralImageSizes"),
    ]
    return {"processor": {"name": "lfm2_vl_image_processor", "transforms": transforms}}


def convert(checkpoint, output, vision="vision_encoder.onnx"):
    checkpoint, output = Path(checkpoint), Path(output)
    hf_config = json.loads((checkpoint / "config.json").read_text())
    # LFM2.5-VL nests the image processor settings in processor_config.json, LFM2-VL has preprocessor_config.json.
    processor_path = checkpoint / "processor_config.json"
    image_processor = json.loads(processor_path.read_text()).get("image_processor") if processor_path.exists() else None
    if image_processor is None:
        image_processor = json.loads((checkpoint / "preprocessor_config.json").read_text())

    config = json.loads((output / "genai_config.json").read_text())
    model = config["model"]
    if model["type"] != "lfm2_vl":
        raise ValueError(
            f'Expected type "lfm2_vl", got "{model["type"]}": build with --extra_options exclude_embeds=true'
        )
    decoder = onnx.load(output / model["decoder"]["filename"], load_external_data=False)
    embeds_type = tensor_type(decoder.graph.input, model["decoder"]["inputs"]["inputs_embeds"]).elem_type
    if embeds_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
        raise ValueError("Only float32 and float16 decoders are supported")
    features = tensor_type(onnx.load(output / vision, load_external_data=False).graph.output, "image_features")
    hidden_size = model["decoder"]["hidden_size"]
    if features.shape.dim[-1].dim_value != hidden_size:
        raise ValueError(f"{vision} does not produce {hidden_size}-wide image_features for this decoder")

    processor = make_processor_config(image_processor)
    table = load_table(checkpoint, embeds_type)
    if table.shape[1] != hidden_size:
        raise ValueError(f"The checkpoint's embedding table is {table.shape[1]} wide, the decoder {hidden_size}")
    embedding = make_embedding_model(table, hf_config["image_token_id"], features.elem_type)

    # The embedding and vision sessions follow the decoder's EP. Graph capture replays one recorded run, so it
    # stays on the decoder: the other sessions get new shapes per prompt (CUDA/TensorRT RTX, DML/AMDGPU, WebGPU).
    graph_capture = {"enable_cuda_graph", "enable_graph_capture", "enableGraphCapture"}
    provider_options = [
        {name: {k: v for k, v in options.items() if k not in graph_capture}}
        for provider in model["decoder"]["session_options"].get("provider_options", [])
        for name, options in provider.items()
    ]
    session_options = {"log_id": "onnxruntime-genai", "provider_options": provider_options}
    model["embedding"] = {
        "filename": "embeddings.onnx",
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": session_options,
    }
    model["vision"] = {
        "filename": vision,
        "config_filename": "processor_config.json",
        "patch_size": image_processor["encoder_patch_size"],
        "spatial_merge_size": image_processor["downsample_factor"],
        "max_num_patches": image_processor["max_num_patches"],
        "inputs": {
            "pixel_values": "pixel_values",
            "attention_mask": "pixel_attention_mask",
            "image_sizes": "spatial_shapes",
        },
        "outputs": {"image_features": "image_features"},
        "session_options": session_options,
    }

    # onnx.save appends to an existing data file, so a rerun would keep the old table too.
    (output / "embeddings.onnx.data").unlink(missing_ok=True)
    onnx.save(embedding, output / "embeddings.onnx", save_as_external_data=True, location="embeddings.onnx.data")
    (output / "processor_config.json").write_text(json.dumps(processor, indent=2) + "\n")
    (output / "genai_config.json").write_text(json.dumps(config, indent=4) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Hugging Face checkpoint directory")
    parser.add_argument("-o", "--output", required=True, help="Model builder output directory (exclude_embeds=true)")
    parser.add_argument("--vision", default="vision_encoder.onnx", help="Vision model file inside the output directory")
    args = parser.parse_args()
    print(convert(args.input, args.output, args.vision))
