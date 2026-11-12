"""Convert a packed R8 Nemotron-Bolt package to dynamic-R prefill + R8 speculative verify.

Preserves all weight payloads via hardlinks (zero weight repacking) and keeps R=8 CUDA-graph
speculative verification intact while enabling large-chunk prefill with selective logit_rows
and aux_rows gathering.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def convert(source: Path, output: Path) -> None:
    source = source.resolve(strict=True)
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    config = json.loads((source / "genai_config.json").read_text())
    if config["model"]["type"] != "nemotron_bolt":
        raise ValueError("Expected a nemotron_bolt package")
    filename = config["model"]["decoder"]["filename"]
    model = onnx.load(source / filename, load_external_data=False)
    graph = model.graph

    # 1. Update input_ids, logits, pred_ids, aux_hidden_states shapes to independent symbolic row dimensions
    for vi in list(graph.input) + list(graph.output):
        if vi.name == "input_ids":
            vi.type.tensor_type.shape.dim[1].ClearField("dim_value")
            vi.type.tensor_type.shape.dim[1].dim_param = "target_rows"
        elif vi.name in ("logits", "pred_ids"):
            vi.type.tensor_type.shape.dim[1].ClearField("dim_value")
            vi.type.tensor_type.shape.dim[1].dim_param = "logit_row_count"
        elif vi.name == "aux_hidden_states":
            vi.type.tensor_type.shape.dim[1].ClearField("dim_value")
            vi.type.tensor_type.shape.dim[1].dim_param = "aux_row_count"

    # Clear stale internal value_info with fixed R=8 so ORT re-infers dynamic shapes cleanly
    del graph.value_info[:]

    # 2. Add logit_rows and aux_rows with independent symbolic row counts
    graph.input.append(helper.make_tensor_value_info("logit_rows", TensorProto.INT64, ["logit_row_count"]))
    graph.input.append(helper.make_tensor_value_info("aux_rows", TensorProto.INT64, ["aux_row_count"]))

    # Locate LM head node (MatMulPacked feeding Identity -> logits, or producing logits directly)
    logits_src = "logits"
    for n in graph.node:
        if n.op_type == "Identity" and "logits" in n.output:
            logits_src = n.input[0]
            break

    lm_head_idx = None
    for idx, n in enumerate(graph.node):
        if logits_src in n.output and n.op_type == "MatMulPacked":
            lm_head_idx = idx
            break
    if lm_head_idx is None:
        raise RuntimeError("Could not locate LM head MatMulPacked node")

    lm_head_node = graph.node[lm_head_idx]
    orig_lm_in = lm_head_node.input[0]
    gathered_lm_in = orig_lm_in + "_selected_logit_rows"
    gather_lm_node = helper.make_node(
        "Gather",
        inputs=[orig_lm_in, "logit_rows"],
        outputs=[gathered_lm_in],
        axis=1,
        name="Gather_LogitRows",
    )
    lm_head_node.input[0] = gathered_lm_in

    # Locate Aux Concat node (Concat feeding Identity -> aux_hidden_states, or producing aux_hidden_states)
    aux_src = "aux_hidden_states"
    for n in graph.node:
        if n.op_type == "Identity" and "aux_hidden_states" in n.output:
            aux_src = n.input[0]
            break

    aux_concat_idx = None
    for idx, n in enumerate(graph.node):
        if aux_src in n.output and n.op_type == "Concat":
            aux_concat_idx = idx
            break
    if aux_concat_idx is None:
        raise RuntimeError("Could not locate aux_hidden_states Concat node")

    aux_concat_node = graph.node[aux_concat_idx]
    aux_gather_nodes = []
    for tap_i, tap_name in enumerate(aux_concat_node.input):
        selected_tap = f"{tap_name}_selected_aux_rows_{tap_i}"
        aux_gather_nodes.append(
            helper.make_node(
                "Gather",
                inputs=[tap_name, "aux_rows"],
                outputs=[selected_tap],
                axis=1,
                name=f"Gather_AuxRows_{tap_i}",
            )
        )
        aux_concat_node.input[tap_i] = selected_tap

    # Rebuild node list in topological order with Gather nodes inserted immediately before consumers
    new_nodes = []
    for n in graph.node:
        if n is aux_concat_node:
            new_nodes.extend(aux_gather_nodes)
        if n is lm_head_node:
            new_nodes.append(gather_lm_node)
        new_nodes.append(n)
    del graph.node[:]
    graph.node.extend(new_nodes)

    output.mkdir(parents=True)
    for path in source.iterdir():
        if path.name == filename:
            continue
        resolved = path.resolve(strict=True)
        if resolved.stat().st_dev == output.stat().st_dev:
            os.link(resolved, output / path.name)
        elif resolved.stat().st_size < (64 << 20):
            shutil.copy2(resolved, output / path.name)
        else:
            raise ValueError(f"Large payload {path} is on a different filesystem from {output}")

    onnx.save_model(model, str(output / filename), save_as_external_data=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    convert(args.input, args.output)
