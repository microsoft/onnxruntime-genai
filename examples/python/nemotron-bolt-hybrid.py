"""Convert a packed R8 Bolt package to R8 prefill/R1 ordinary decode.

Requires onnx-ir, numpy and ml-dtypes. Large weights are hardlinked, never repacked.
Use an ORT build preserving four-part Mamba readout for multi-slot R1 replay.
The output is ordinary-only; keep the original R8 package for speculation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx_ir as ir


def attention_width(value: ir.Value | None) -> int:
    if value is not None:
        shape = value.shape
        if shape is not None and len(shape) == 3 and isinstance(shape[2], int) and shape[2] > 0:
            return shape[2]
        producer = value.producer()
        attribute = producer.attributes.get("N") if producer is not None else None
        if attribute is not None and attribute.type == ir.AttributeType.INT:
            width = attribute.as_int()
            if width > 0:
                return width
    name = value.name if value is not None else "<missing input>"
    raise ValueError(
        f"Cannot determine positive Q/K/V width for {name}: expected static rank-3 width or producer integer N"
    )


def convert(source: Path, output: Path) -> None:
    source = source.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be a fresh directory")
    output = output.resolve()
    source_entries = list(source.iterdir())
    if any(not path.is_file() for path in source_entries):
        raise ValueError("Expected a flat packed package")
    config = json.loads((source / "genai_config.json").read_text())
    if config["model"]["type"] != "nemotron_bolt":
        raise ValueError("Expected a packed nemotron_bolt package")
    filename = config["model"]["decoder"]["filename"]
    if filename != "model.onnx":
        raise ValueError("Expected the flat model.onnx packed package")
    config.setdefault("speculative", {})["max_draft_tokens"] = 0
    model = ir.load(source / filename)
    original_nodes = list(model.graph)
    initializers = dict(model.graph.initializers)
    for value in [*model.graph.inputs, *model.graph.outputs]:
        if value.name in ("input_ids", "logits", "pred_ids", "aux_hidden_states"):
            shape = list(value.shape)
            if shape[:2] != [1, 8]:
                raise ValueError(f"Expected R8 shape for {value.name}, got {shape}")
            value.shape = ir.Shape([1, "target_rows", *shape[2:]])
    query_rows = ir.Value(name="query_rows", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape(["target_rows"]))
    model.graph.inputs.append(query_rows)

    def constant(name: str, values: list[int]) -> tuple[ir.Node, ir.Value]:
        value = ir.Value(name=name, type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([len(values)]))
        node = ir.Node(
            "",
            "Constant",
            [],
            outputs=[value],
            attributes={"value": ir.AttrTensor("value", ir.tensor(np.asarray(values, dtype=np.int64)))},
            name=name + "_node",
        )
        return node, value

    definitions = [
        constant("ordinary_r1_slice_start", [0]),
        constant("ordinary_r1_slice_end", [8]),
        constant("ordinary_r1_slice_axis", [1]),
    ]
    start, end, axis = [value for _, value in definitions]
    nodes = [node for node, _ in definitions]
    zeros = {}
    attention_count = 0
    for node in original_nodes:
        if node.op_type != "GroupQueryAttention":
            nodes.append(node)
            continue
        # Keep Flash's physical R8 geometry and cache semantics for both logical row counts.
        for index in range(3):
            original = node.inputs[index]
            width = attention_width(original)
            if width not in zeros:
                data = np.zeros((1, 7, width), dtype=ml_dtypes.bfloat16)
                value = ir.Value(
                    name=f"ordinary_r1_zeros_{width}",
                    type=ir.TensorType(ir.DataType.BFLOAT16),
                    shape=ir.Shape(data.shape),
                )
                nodes.append(
                    ir.Node(
                        "",
                        "Constant",
                        [],
                        outputs=[value],
                        attributes={"value": ir.AttrTensor("value", ir.tensor(data))},
                        name=value.name + "_node",
                    )
                )
                zeros[width] = value
            padded = ir.Value(name=original.name + "_ordinary_r1_pad")
            nodes.append(
                ir.Node(
                    "",
                    "Concat",
                    [original, zeros[width]],
                    outputs=[padded],
                    attributes={"axis": ir.AttrInt64("axis", 1)},
                    name=padded.name + "_node",
                )
            )
            bounded = ir.Value(name=original.name + "_ordinary_hybrid_eight")
            nodes.append(
                ir.Node(
                    "",
                    "Slice",
                    [padded, start, end, axis],
                    outputs=[bounded],
                    name=bounded.name + "_node",
                )
            )
            node.replace_input_with(index, bounded)
        nodes.append(node)
        original_output = node.outputs[0]
        cropped = ir.Value(name=original_output.name + "_ordinary_r1_crop")
        original_output.replace_all_uses_with(cropped)
        nodes.append(
            ir.Node(
                "",
                "Gather",
                [original_output, query_rows],
                outputs=[cropped],
                attributes={"axis": ir.AttrInt64("axis", 1)},
                name=cropped.name + "_node",
            )
        )
        attention_count += 1
    if attention_count != 8:
        raise ValueError(f"Expected eight packed attention layers, got {attention_count}")
    model.graph.remove(original_nodes, safe=False)
    model.graph.extend(nodes)
    assets = [
        (path.name, path.resolve(strict=True))
        for path in source_entries
        if path.name not in (filename, "genai_config.json")
    ]
    missing_parents = []
    parent = output.parent
    while not parent.exists():
        missing_parents.append(parent)
        parent = parent.parent
    destination_device = parent.stat().st_dev
    for _, path in assets:
        if path.stat().st_size >= 64 << 20 and path.stat().st_dev != destination_device:
            raise ValueError("Place output on the same filesystem as the packed weight payload")
    created_parents = []
    try:
        for parent in reversed(missing_parents):
            parent.mkdir()
            created_parents.append(parent)
        # Publish only a fully validated package; failed nested conversions leave the source flat.
        with tempfile.TemporaryDirectory(prefix=".bolt-hybrid-", dir=output.parent) as temporary:
            staged = Path(temporary)
            for name, path in assets:
                try:
                    os.link(path, staged / name)
                except OSError as error:
                    if path.stat().st_size >= 64 << 20:
                        raise ValueError(f"Cannot hardlink large weight payload: {path.name}") from error
                    shutil.copy2(path, staged / name)
            ir.save(model, staged / filename)
            checked = ir.load(staged / filename)
            if checked.graph.initializers.keys() != initializers.keys():
                raise ValueError("Initializer inventory changed")
            for name, value in checked.graph.initializers.items():
                old, new = initializers[name].const_value, value.const_value
                if isinstance(old, ir.ExternalTensor):
                    if not isinstance(new, ir.ExternalTensor) or (
                        old.location,
                        old.offset,
                        old.length,
                        old.dtype,
                        old.shape,
                    ) != (new.location, new.offset, new.length, new.dtype, new.shape):
                        raise ValueError(f"External initializer changed: {name}")
                elif (old.dtype, old.shape, old.numpy().tobytes()) != (new.dtype, new.shape, new.numpy().tobytes()):
                    raise ValueError(f"Inline initializer changed: {name}")
            (staged / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
            if output.exists() or output.is_symlink():
                raise ValueError("Output must be a fresh directory")
            staged.rename(output)
    except BaseException:
        # Remove only empty parents created by this conversion, never another writer's contents.
        for parent in reversed(created_parents):
            with suppress(OSError):
                parent.rmdir()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original packed R8 package")
    parser.add_argument("--output", type=Path, required=True, help="Fresh ordinary-only package")
    args = parser.parse_args()
    convert(args.input, args.output)
