#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Prepare an ONNX model and GenAI config for TurboQuant KV cache compression.

This script:
1. Updates the ONNX model's past_key/past_value and present_key/present_value
   tensor shapes to use the compressed head dimension (dynamic or fixed).
2. Updates genai_config.json to add kv_cache_head_size and the WebGPU EP option.

Usage:
  python prepare_turbo_quant.py --model_dir C:/models/phi4-onnx

  # Dry-run (show changes without writing):
  python prepare_turbo_quant.py --model_dir C:/models/phi4-onnx --dry_run

  # Custom head_size (auto-detected from genai_config.json by default):
  python prepare_turbo_quant.py --model_dir C:/models/phi4-onnx --head_size 128
"""

import argparse
import json
import os
import sys

import onnx


def compute_compressed_dim(head_size: int) -> int:
    """Compute the TurboQuant compressed KV cache dimension for fp16.

    Layout per token (in fp16 values):
      [norm_lo, norm_hi]  (2 fp16 = 1 u32 bitcast to 2xf16)
      [packed indices]    (head_size/4 fp16 values, each pair = 1 u32 = 8 x 4-bit indices)
      [padding]           (to vec4 alignment)

    Formula: ((head_size / 4 + 2 + 3) / 4) * 4
    """
    return ((head_size // 4 + 2 + 3) // 4) * 4


def update_onnx_model(model_path: str, output_path: str, head_size: int, dry_run: bool = False) -> list:
    """Update ONNX model past/present KV tensor shapes to use dynamic last dim.

    Returns list of changes made.
    """
    model = onnx.load(model_path, load_external_data=False)
    graph = model.graph
    changes = []

    # Patterns to match for KV cache tensors
    kv_patterns = ["past_key_values.", "present."]

    def is_kv_tensor(name: str) -> bool:
        return any(p in name for p in kv_patterns) and (
            name.endswith(".key") or name.endswith(".value")
        )

    def update_shape(tensor_info, label: str):
        """Update the last dimension of a tensor to be dynamic."""
        shape = tensor_info.type.tensor_type.shape
        if shape and len(shape.dim) == 4:
            last_dim = shape.dim[3]
            old_val = last_dim.dim_value if last_dim.dim_value else last_dim.dim_param
            if last_dim.dim_value == head_size:
                last_dim.ClearField("dim_value")
                last_dim.dim_param = "kv_cache_dim"
                changes.append(f"  {label}: dim[3] {old_val} -> 'kv_cache_dim'")
            elif last_dim.dim_param:
                changes.append(f"  {label}: dim[3] already dynamic ('{last_dim.dim_param}'), skipped")
            elif last_dim.dim_value != head_size:
                changes.append(
                    f"  {label}: dim[3]={last_dim.dim_value} != head_size={head_size}, skipped"
                )

    # Update graph inputs
    for inp in graph.input:
        if is_kv_tensor(inp.name):
            update_shape(inp, f"input '{inp.name}'")

    # Update graph outputs
    for out in graph.output:
        if is_kv_tensor(out.name):
            update_shape(out, f"output '{out.name}'")

    if changes and not dry_run:
        # We loaded without external data (weights stay in the .data file),
        # so the protobuf is small enough to serialize directly.
        with open(output_path, "wb") as f:
            f.write(model.SerializeToString())

    return changes


def update_genai_config(config_path: str, compressed_dim: int, dry_run: bool = False) -> list:
    """Update genai_config.json with kv_cache_head_size and TurboQuant EP option.

    Returns list of changes made.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    changes = []
    decoder = config.get("model", {}).get("decoder", {})

    # 1. Add kv_cache_head_size
    if decoder.get("kv_cache_head_size") == compressed_dim:
        changes.append(f"  kv_cache_head_size already set to {compressed_dim}")
    else:
        old = decoder.get("kv_cache_head_size")
        decoder["kv_cache_head_size"] = compressed_dim
        if old is not None:
            changes.append(f"  kv_cache_head_size: {old} -> {compressed_dim}")
        else:
            changes.append(f"  kv_cache_head_size: added ({compressed_dim})")

    # 2. Add ep.webgpu.turbo_quant to provider_options
    session_options = decoder.get("session_options", {})
    provider_options = session_options.get("provider_options", [])

    webgpu_opts = None
    for po in provider_options:
        if isinstance(po, dict) and "webgpu" in po:
            webgpu_opts = po["webgpu"]
            break

    if webgpu_opts is None:
        # No WebGPU provider_options entry — add one
        webgpu_opts = {}
        provider_options.append({"webgpu": webgpu_opts})
        session_options["provider_options"] = provider_options
        decoder["session_options"] = session_options
        changes.append("  provider_options: added WebGPU entry")

    if webgpu_opts.get("turboQuant") == "1":
        changes.append("  turboQuant already set")
    else:
        webgpu_opts["turboQuant"] = "1"
        changes.append("  turboQuant: set to '1'")

    if changes and not dry_run:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            f.write("\n")

    return changes


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an ONNX model and GenAI config for TurboQuant."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the model directory containing model.onnx and genai_config.json",
    )
    parser.add_argument(
        "--head_size",
        type=int,
        default=None,
        help="Head size (auto-detected from genai_config.json if not specified)",
    )
    parser.add_argument(
        "--model_filename",
        default="model.onnx",
        help="ONNX model filename (default: model.onnx)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show changes without modifying files",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.model_dir, "genai_config.json")
    model_path = os.path.join(args.model_dir, args.model_filename)

    if not os.path.exists(config_path):
        print(f"ERROR: genai_config.json not found at {config_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"ERROR: ONNX model not found at {model_path}")
        sys.exit(1)

    # Read head_size from config if not specified
    with open(config_path, "r") as f:
        config = json.load(f)

    head_size = args.head_size or config["model"]["decoder"]["head_size"]
    compressed_dim = compute_compressed_dim(head_size)

    print(f"Model directory: {args.model_dir}")
    print(f"Head size: {head_size}")
    print(f"Compressed dim (fp16): {compressed_dim}")
    print(f"Compression ratio: {head_size / compressed_dim:.1f}x ({compressed_dim * 2} bytes vs {head_size * 2} bytes per token per head)")
    print()

    if args.dry_run:
        print("=== DRY RUN (no files modified) ===\n")

    # Update ONNX model
    print(f"ONNX model: {model_path}")
    model_changes = update_onnx_model(model_path, model_path, head_size, args.dry_run)
    if model_changes:
        for c in model_changes:
            print(c)
    else:
        print("  No changes needed")

    # Update genai_config.json
    print(f"\nGenAI config: {config_path}")
    config_changes = update_genai_config(config_path, compressed_dim, args.dry_run)
    if config_changes:
        for c in config_changes:
            print(c)
    else:
        print("  No changes needed")

    print()
    if args.dry_run:
        print("Re-run without --dry_run to apply changes.")
    else:
        total = len(model_changes) + len(config_changes)
        print(f"Done. {total} changes applied.")


if __name__ == "__main__":
    main()
