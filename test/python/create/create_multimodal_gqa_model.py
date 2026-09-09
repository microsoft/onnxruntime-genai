# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Small non-recurrent VLM with genuine GroupQueryAttention past/present buffers.

The same graphs serve Python and native tests: CPU FP32, CUDA FP16, and WebGPU
FP32/FP16. No Concat node implements the decoder cache. The only allowed
CPU partition on an accelerator is the named attention-length metadata chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from create.create_gqa_model import HEAD_SIZE, HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, gqa_node
from create.create_multimodal_turn_test_model import (
    QWEN_FAMILIES,
    VOCAB_SIZE,
    _info,
    make_config,
    make_embedding_model,
    make_vision_model,
)
from create.create_multimodal_turn_test_model import (
    _tensor as _init,
)
from onnx import TensorProto as T
from onnx import helper, numpy_helper

# GQA consumes these on the host even when its numerical kernel runs on a GPU.
# Exact node-name/op-type pairs, not a blanket allowance for shape/arithmetic ops.
CPU_METADATA_NODES = {
    "metadata.mask_i32": "Cast",
    "metadata.length_plus_one": "ReduceSum",
    "metadata.seqlens_k": "Sub",
    "metadata.mask_shape": "Shape",
    "metadata.total_i64": "Gather",
    "metadata.total_sl": "Cast",
}
GPU_DEVICES = ("cuda", "webgpu")


def supported_dtypes(device):
    # CUDA GQA registers MLFloat16/BFloat16, not FP32; media tensors use FP16.
    return ("fp16",) if device == "cuda" else ("fp32", "fp16") if device == "webgpu" else ("fp32",)


def _convert_float_model(model, dtype):
    tensor_type = T.FLOAT16 if dtype == np.float16 else T.FLOAT
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.type.tensor_type.elem_type == T.FLOAT:
            value.type.tensor_type.elem_type = tensor_type
    for index, value in enumerate(model.graph.initializer):
        if value.data_type == T.FLOAT:
            model.graph.initializer[index].CopyFrom(_init(value.name, numpy_helper.to_array(value), dtype))
    for node in model.graph.node:
        for attr in node.attribute:
            if node.op_type == "Cast" and attr.name == "to" and attr.i == T.FLOAT:
                attr.i = tensor_type
    return model


def _media_models(family, dtype):
    vision = make_vision_model(family)
    if family == "mistral3":
        vision.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "height"
        vision.graph.input[0].type.tensor_type.shape.dim[3].dim_param = "width"
        vision.graph.node[0].input[0] = "pixel_patches"
        vision.graph.node.insert(
            0, helper.make_node("Reshape", ["pixels_nhwc", "patch_shape"], ["pixel_patches"], name="vision.patches")
        )
        vision.graph.node.insert(
            0,
            helper.make_node(
                "Transpose", ["pixel_values"], ["pixels_nhwc"], name="vision.channels_last", perm=[0, 2, 3, 1]
            ),
        )
        vision.graph.initializer.append(_init("patch_shape", [-1, 3], np.int64))
        for index, value in enumerate(vision.graph.initializer):
            if value.name == "pixel_axes":
                vision.graph.initializer[index].CopyFrom(_init("pixel_axes", [1], np.int64))
    vision.graph.node[-1].output[0] = "scalar_features"
    vision.graph.node.append(
        helper.make_node("MatMul", ["scalar_features", "feature_projection"], ["image_features"], name="vision.project")
    )
    vision.graph.initializer.append(
        _init("feature_projection", np.linspace(-0.01, 0.02, HIDDEN_SIZE).reshape(1, -1), np.float32)
    )
    vision.graph.output[0].type.tensor_type.shape.dim[1].dim_value = HIDDEN_SIZE
    embedding = make_embedding_model(family)
    for index, value in enumerate(embedding.graph.initializer):
        if value.name == "empty_feature":
            embedding.graph.initializer[index].CopyFrom(_init("empty_feature", np.zeros((1, HIDDEN_SIZE)), np.float32))
        elif value.name in ("two", "three"):
            embedding.graph.initializer[index].CopyFrom(
                _init(value.name, 0.02 if value.name == "two" else 0.03, np.float32)
            )
    embedding.graph.input[1].type.tensor_type.shape.dim[1].dim_value = HIDDEN_SIZE
    embedding.graph.output[0].type.tensor_type.shape.dim[2].dim_value = HIDDEN_SIZE
    return _convert_float_model(vision, dtype), _convert_float_model(embedding, dtype)


def make_decoder(family, dtype, *, capture=False):
    tensor_type = T.FLOAT16 if dtype == np.float16 else T.FLOAT
    qwen = family in QWEN_FAMILIES
    rng = np.random.default_rng(20260908)
    inits = [
        _init("axis_1", [1], np.int64),
        _init("axis_2", [2], np.int64),
        _init("one", [1], np.int32),
        _init("shape_index", 1, np.int64),
        _init("position_projection", np.linspace(-0.003, 0.005, HIDDEN_SIZE).reshape(1, -1), dtype),
    ]
    nodes = [
        helper.make_node("Cast", ["attention_mask"], ["mask_i32"], name="metadata.mask_i32", to=T.INT32),
        helper.make_node(
            "ReduceSum", ["mask_i32", "axis_1"], ["length_plus_one"], name="metadata.length_plus_one", keepdims=0
        ),
        helper.make_node("Sub", ["length_plus_one", "one"], ["seqlens_k"], name="metadata.seqlens_k"),
        helper.make_node("Shape", ["attention_mask"], ["mask_shape"], name="metadata.mask_shape"),
        helper.make_node("Gather", ["mask_shape", "shape_index"], ["total_i64"], name="metadata.total_i64", axis=0),
        helper.make_node("Cast", ["total_i64"], ["total_sl"], name="metadata.total_sl", to=T.INT32),
        helper.make_node("Cast", ["position_ids"], ["float_positions"], name="decoder.positions", to=tensor_type),
    ]
    positions = "float_positions"
    if qwen:
        inits.extend([_init("axis_0", [0], np.int64), _init("mrope_weights", [[[1]], [[2]], [[4]]], dtype)])
        nodes.extend(
            [
                helper.make_node(
                    "Mul", [positions, "mrope_weights"], ["weighted_positions"], name="decoder.mrope_weights"
                ),
                helper.make_node(
                    "ReduceSum",
                    ["weighted_positions", "axis_0"],
                    ["summed_positions"],
                    name="decoder.mrope_sum",
                    keepdims=0,
                ),
            ]
        )
        positions = "summed_positions"
    nodes.extend(
        [
            helper.make_node("Unsqueeze", [positions, "axis_2"], ["positions_3d"], name="decoder.position_expand"),
            helper.make_node(
                "MatMul", ["positions_3d", "position_projection"], ["position_bias"], name="decoder.position_bias"
            ),
            helper.make_node("Add", ["inputs_embeds", "position_bias"], ["hidden"], name="decoder.hidden"),
        ]
    )
    for kind, width in (("q", HIDDEN_SIZE), ("k", NUM_KV_HEADS * HEAD_SIZE), ("v", NUM_KV_HEADS * HEAD_SIZE)):
        inits.append(_init(f"{kind}.weight", rng.normal(0, 0.09, (HIDDEN_SIZE, width)), dtype))
        nodes.append(helper.make_node("MatMul", ["hidden", f"{kind}.weight"], [f"l0.{kind}"], name=f"decoder.{kind}"))
    nodes.append(gqa_node("l0", 0))
    inits.append(_init("lm_head.weight", rng.normal(0, 0.15, (HIDDEN_SIZE, VOCAB_SIZE)), dtype))
    # Keep EOS out of deterministic stress runs, without masking the other numerical logits.
    bias = np.zeros(VOCAB_SIZE)
    bias[-1] = -100
    inits.append(_init("lm_head.bias", bias, dtype))
    nodes.extend(
        [
            helper.make_node("MatMul", ["l0.attn", "lm_head.weight"], ["raw_logits"], name="decoder.lm_head"),
            helper.make_node("Add", ["raw_logits", "lm_head.bias"], ["logits"], name="decoder.logits"),
        ]
    )
    inputs = [
        _info("inputs_embeds", tensor_type, [1, "seq", HIDDEN_SIZE]),
        _info("position_ids", T.INT64, [3, 1, "seq"] if qwen else [1, "seq"]),
        # Captured decode uses GenAI's fixed-capacity mask. Making the capacity
        # explicit lets ORT fold Shape/Gather/Cast to the host GQA length constant,
        # rather than leaving CPU shape nodes in the CUDA captured graph.
        _info("attention_mask", T.INT64, [1, 192 if capture else "total_seq"]),
    ]
    outputs = [_info("logits", tensor_type, [1, "seq", VOCAB_SIZE])]
    for kind in ("key", "value"):
        inputs.append(_info(f"past_key_values.0.{kind}", tensor_type, [1, NUM_KV_HEADS, "past_seq", HEAD_SIZE]))
        outputs.append(_info(f"present.0.{kind}", tensor_type, [1, NUM_KV_HEADS, "cache_seq", HEAD_SIZE]))
    model = helper.make_model(
        helper.make_graph(nodes, "multimodal_gqa", inputs, outputs, inits),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def create_model(output_dir, family="phi3v", *, device="cpu", dtype=None, shared=False, capture=False, profile=False):
    dtype = dtype or supported_dtypes(device)[0]
    if dtype not in supported_dtypes(device):
        raise ValueError(f"GQA fixture does not support {dtype} on {device}")
    if capture and (not shared or device not in GPU_DEVICES):
        raise ValueError("Actual capture requires CUDA/WebGPU and shared GQA")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np_dtype = np.float16 if dtype == "fp16" else np.float32
    vision, embedding = _media_models(family, np_dtype)
    for name, graph in (
        ("vision", vision),
        ("embedding", embedding),
        ("decoder", make_decoder(family, np_dtype, capture=capture)),
    ):
        onnx.checker.check_model(graph)
        onnx.save(graph, output_dir / f"{name}.onnx")
    path = output_dir / "genai_config.json"
    config = make_config(family)
    decoder = config["model"]["decoder"]
    decoder.update(
        hidden_size=HIDDEN_SIZE, head_size=HEAD_SIZE, num_attention_heads=NUM_HEADS, num_key_value_heads=NUM_KV_HEADS
    )
    for role in ("vision", "embedding", "decoder"):
        options = {}
        if device in GPU_DEVICES:
            options["device_filtering_options"] = {"hardware_device_type": "gpu"}
            options["enable_cuda_graph" if device == "cuda" else "enableGraphCapture"] = (
                "1" if capture and role == "decoder" else "0"
            )
        session = {
            "provider_options": [] if device == "cpu" else [{device: options}],
            # GQA has host-only length inputs. Python audits the exact named metadata
            # nodes; every vision/embedding numerical node must be on the requested EP.
            "session.disable_cpu_ep_fallback": "0" if device == "cpu" or role == "decoder" else "1",
        }
        if profile:
            session["enable_profiling"] = str((output_dir / f"profile_{role}").resolve())
        config["model"][role]["session_options"] = session
    config["search"]["past_present_share_buffer"] = shared
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--family", choices=("phi3v", "qwen2_5_vl", "mistral3"), default="phi3v")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16"))
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--suite", action="store_true", help="Build the native matrix beneath output_dir")
    parser.add_argument("--devices", default="cpu", help="Comma-separated native suite EPs; generation needs no GPU")
    args = parser.parse_args()
    if args.suite:
        for device in dict.fromkeys(("cpu", *args.devices.split(","))):
            for dtype in supported_dtypes(device):
                for family in ("phi3v", "qwen2_5_vl", "mistral3"):
                    create_model(
                        args.output_dir / f"{device}-{dtype}-{family}",
                        family,
                        device=device,
                        dtype=dtype,
                        profile=True,
                    )
                if device in GPU_DEVICES:
                    create_model(
                        args.output_dir / f"{device}-{dtype}-phi3v-capture",
                        device=device,
                        dtype=dtype,
                        shared=True,
                        capture=True,
                        profile=True,
                    )
    else:
        options = vars(args)
        del options["suite"], options["devices"]
        create_model(**options)


if __name__ == "__main__":
    main()
