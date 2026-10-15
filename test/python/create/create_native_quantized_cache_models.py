# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Small opt-in CUDA native-quantization/GenAI interoperability fixtures.

These arithmetic graphs test prepared BF16 media, sparse cache ownership, and
native operator loading, not the full Bolt checkpoint or vision preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx_ir as ir
from create_cache_dispatch_models import create_model
from create_multimodal_cache_model import create_auxiliary


def create_native_fixture(directory: Path, *, nvfp4: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    create_model(directory, recurrent=True, model_type="nemotron_h_omni", multimodal=True)
    for vision in (False, True):
        create_auxiliary(directory, vision=vision)
    vision = ir.load(directory / "vision.onnx")
    dense_features = vision.graph.outputs.pop()
    dense_features.name = "features_f32"
    vision_features = ir.Value(
        name="image_features", type=ir.TensorType(ir.DataType.BFLOAT16), shape=dense_features.shape
    )
    vision.graph.append(ir.Node("", "Cast", [dense_features], [ir.AttrInt64("to", 16)], outputs=[vision_features]))
    vision.graph.outputs.append(vision_features)
    ir.save(vision, directory / "vision.onnx")
    embedding = ir.load(directory / "embedding.onnx")
    ids = next(v for v in embedding.graph.inputs if v.name == "input_ids")
    ids.type = ir.TensorType(ir.DataType.INT64)
    image_token = embedding.graph.initializers["image_token"]
    image_token.type = ir.TensorType(ir.DataType.INT64)
    image_token.const_value = ir.tensor(np.array(18, dtype=np.int64), name="image_token")
    features = next(v for v in embedding.graph.inputs if v.name == "image_features")
    features.type = ir.TensorType(ir.DataType.BFLOAT16)
    converted = ir.Value(name="features_f32", type=ir.TensorType(ir.DataType.FLOAT), shape=features.shape)
    features.replace_all_uses_with(converted)
    cast = ir.Node("", "Cast", [features], [ir.AttrInt64("to", 1)], outputs=[converted])
    embedding.graph.insert_before(embedding.graph.node(0), cast)
    ir.save(embedding, directory / "embedding.onnx")

    model = ir.load(directory / "decoder.onnx")
    graph = model.graph
    old_logits = graph.outputs[0]
    old_logits.name = "original_logits"
    graph.outputs.remove(old_logits)

    def constant(name, array):
        tensor = ir.tensor(array, name=name)
        result = ir.Value(name=name, const_value=tensor, type=ir.TensorType(tensor.dtype), shape=tensor.shape)
        graph.initializers[name] = result
        return result

    def op(kind, inputs, name, dtype, shape, *, domain="", attrs=()):
        result = ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))
        graph.append(ir.Node(domain, kind, inputs, attrs, outputs=[result]))
        return result

    state = next(v for v in graph.outputs if v.name == "present.0.conv_state")
    total = op("ReduceMean", [state], "state_total", ir.DataType.FLOAT, [], attrs=[ir.AttrInt64("keepdims", 0)])
    width = 16 if nvfp4 else 32
    expanded = op(
        "Expand",
        [total, constant("operand_shape", np.array([1, 1, width], dtype=np.int64))],
        "operand_f32",
        ir.DataType.FLOAT,
        [1, 1, width],
    )
    operand = op("Cast", [expanded], "operand", ir.DataType.BFLOAT16, [1, 1, width], attrs=[ir.AttrInt64("to", 16)])
    bias = np.zeros(16, dtype=ml_dtypes.bfloat16)
    bias[:3] = [-1.25, 0.375, 0.625]
    native_bias = constant("native_bias", bias)
    if nvfp4:
        # E2M1 rows .5, 1, -1; E4M3 scale 1.625. Global .333 is applied
        # AFTER the FP32 dot, not folded into the native decoded weights.
        blocks = np.zeros((16, 8), dtype=np.uint8)
        blocks[:3] = np.array([0x11, 0x22, 0xAA], dtype=np.uint8)[:, None]
        quantized = op(
            "MatMulNVFP4A16",
            [
                operand,
                constant("packed", blocks),
                constant("local_scale", np.full((16, 1), 0x3D, dtype=np.uint8)),
                constant("global_scale", np.array([0.333], dtype=np.float32)),
            ],
            "before_bias",
            ir.DataType.BFLOAT16,
            [1, 1, 16],
            domain="com.microsoft",
        )
        quantized = op("Add", [quantized, native_bias], "native_logits", ir.DataType.BFLOAT16, [1, 1, 16])
    else:
        # FP16 d=.1 and integer q=9 distinguish BF16-decoded native weights
        # from both FP32 reconstruction and rounding the embedded scale first.
        blocks = np.zeros((16, 1, 34), dtype=np.uint8)
        blocks[:3, 0, :2] = [0x66, 0x2E]
        blocks[0, 0, 2:] = 1
        blocks[1, 0, 2:] = 9
        blocks[2, 0, 2:] = 0xF7
        quantized = op(
            "MatMulGGMLQ8_0",
            [operand, constant("packed", blocks), native_bias],
            "native_logits",
            ir.DataType.BFLOAT16,
            [1, 1, 16],
            domain="com.microsoft",
        )
    logits = op("Cast", [quantized], "logits", ir.DataType.FLOAT, [1, 1, 16], attrs=[ir.AttrInt64("to", 1)])
    graph.outputs.insert(0, logits)
    graph.outputs.append(quantized)
    graph.opset_imports[""] = 21
    graph.opset_imports["com.microsoft"] = 1
    ir.save(model, directory / "decoder.onnx")
    config_path = directory / "genai_config.json"
    config = json.loads(config_path.read_text())
    config["model"]["decoder"]["session_options"] = {
        "provider_options": [{"cuda": {"enable_cuda_graph": "0", "use_tf32": "0"}}],
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "models" / "cache-dispatch"
    for name in ("q8", "nvfp4"):
        create_native_fixture(root / f"native-{name}-bf16", nvfp4=name == "nvfp4")
