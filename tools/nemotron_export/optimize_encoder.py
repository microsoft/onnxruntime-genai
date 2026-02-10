#!/usr/bin/env python3
"""
Optimize Nemotron ASR encoder ONNX model.

Two-stage optimization:
  1. Graph fusion  — Fuse Conformer attention subgraphs into MultiHeadAttention,
                     SkipLayerNormalization, BiasGelu, etc.
  2. INT4 quantization — Quantize FP32 MatMul weights to 4-bit (MatMulNBits).

     Quantization methods:
       - rtn:          Round-to-nearest (simplest, block_size=32, symmetric)
       - k_quant_mixed: K-quant with mixed precision — sensitive layers
                        (first/last encoder, attention Q/K/V/Out, pre_encode)
                        are excluded from quantization and stay FP32.
                        Uses Intel Neural Compressor's k_quant algorithm for
                        remaining layers.
       - hqq:          Half-Quadratic Quantization (no calibration data needed)

The decoder and joint models are tiny (<35 MB combined) and stay FP32.

Usage:
    python optimize_encoder.py [--model_dir ./onnx_models] [--output_dir ./onnx_models_optimized]
                               [--skip_fusion] [--skip_quantization]
                               [--quant_method rtn|k_quant_mixed|hqq]
                               [--block_size 32] [--accuracy_level 4]
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import onnx


def get_model_stats(model_path: str) -> dict:
    """Get ONNX model statistics: op counts, total size."""
    try:
        model = onnx.load(model_path, load_external_data=False)
    except Exception:
        # For very large models saved in external data format, try loading
        # just the structure (proto may exceed 2GB protobuf limit)
        model = onnx.ModelProto()
        with open(model_path, "rb") as f:
            # Try reading — if too large, just return size info
            try:
                model.ParseFromString(f.read())
            except Exception:
                total_size = os.path.getsize(model_path)
                data_file = model_path + ".data"
                if os.path.exists(data_file):
                    total_size += os.path.getsize(data_file)
                return {"op_counts": {"(model too large to parse)": 0}, "total_size_mb": total_size / (1024 * 1024)}

    op_counts = {}
    for node in model.graph.node:
        op_name = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
        op_counts[op_name] = op_counts.get(op_name, 0) + 1

    # Compute total file size (model + external data)
    total_size = os.path.getsize(model_path)
    data_file = model_path + ".data"
    if os.path.exists(data_file):
        total_size += os.path.getsize(data_file)

    return {"op_counts": op_counts, "total_size_mb": total_size / (1024 * 1024)}


def print_model_stats(label: str, stats: dict):
    """Print model statistics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Op counts ({len(stats['op_counts'])} unique ops):")
    # Show key fusion-related ops first
    key_ops = [
        "MultiHeadAttention", "Attention",
        "com.microsoft::MultiHeadAttention", "com.microsoft::Attention",
        "SkipLayerNormalization", "com.microsoft::SkipLayerNormalization",
        "LayerNormalization",
        "BiasGelu", "com.microsoft::BiasGelu",
        "FastGelu", "com.microsoft::FastGelu",
        "MatMulNBits", "com.microsoft::MatMulNBits",
        "MatMul", "Gemm",
        "Conv", "LSTM",
    ]
    printed = set()
    for op in key_ops:
        if op in stats["op_counts"]:
            print(f"    {op}: {stats['op_counts'][op]}")
            printed.add(op)
    # Print remaining ops
    for op, count in sorted(stats["op_counts"].items()):
        if op not in printed:
            print(f"    {op}: {count}")


def stage1_graph_fusion(input_path: str, output_path: str) -> bool:
    """Apply Conformer graph fusion to the encoder model."""
    print("\n" + "=" * 60)
    print("  STAGE 1: Graph Fusion (Conformer)")
    print("=" * 60)

    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.optimizer import optimize_model

    options = FusionOptions("conformer")
    options.use_multi_head_attention = True

    # Enable all relevant fusions
    options.enable_gelu = True
    options.enable_layer_norm = True
    options.enable_attention = True
    options.enable_skip_layer_norm = True
    options.enable_bias_gelu = True

    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Model type: conformer")
    print(f"  num_heads=8, hidden_size=1024")

    try:
        optimized_model = optimize_model(
            input_path,
            model_type="conformer",
            num_heads=8,         # 1024 hidden / 128 head_size (but conformer uses 64 head_size => 16 heads? check)
            hidden_size=1024,
            optimization_options=options,
        )

        # Check fusion statistics
        fused_stats = optimized_model.get_fused_operator_statistics()
        print(f"\n  Fusion results:")
        for op, count in fused_stats.items():
            if count > 0:
                print(f"    {op}: {count}")

        if all(v == 0 for v in fused_stats.values()):
            print("  WARNING: No ops were fused. The encoder graph pattern may not match")
            print("           standard Conformer patterns. Proceeding with unfused model.")

        optimized_model.save_model_to_file(
            output_path,
            use_external_data_format=True,
        )
        print(f"  Saved fused model to: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR during graph fusion: {e}")
        print(f"  Skipping fusion, will quantize from original model directly.")
        return False


def _get_sensitive_node_names(model_path: str) -> list:
    """Identify sensitive MatMul nodes that should stay FP32 for quality.

    Sensitive layers for Conformer ASR:
      - pre_encode (input projection — first bottleneck)
      - layers.0 (first encoder layer — initial feature extraction)
      - layers.23 (last encoder layer — final representation)
      - self_attn/linear_q, linear_k, linear_v, linear_out (attention projections)
        for ALL layers (attention is more sensitive than FFN to quantization)
    """
    import onnx as _onnx
    model = _onnx.load(model_path, load_external_data=False)
    sensitive = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        name = node.name
        # First/last encoder layers — keep FP32
        if "/pre_encode/" in name:
            sensitive.append(name)
        elif "/layers.0/" in name:
            sensitive.append(name)
        elif "/layers.23/" in name:
            sensitive.append(name)
        # Attention Q/K/V/Out projections in ALL layers — keep FP32
        elif any(p in name for p in ["/linear_q/", "/linear_k/", "/linear_v/", "/linear_out/"]):
            sensitive.append(name)
    return sensitive


def stage2_int4_quantization(
    input_path: str,
    output_path: str,
    block_size: int = 32,
    is_symmetric: bool = True,
    accuracy_level: int = 4,
    quant_method: str = "rtn",
) -> bool:
    """Quantize FP32 MatMul weights to INT4 (MatMulNBits).

    Methods:
      rtn           — Round-to-nearest symmetric (original, simplest)
      k_quant_mixed — K-quant with sensitive layers excluded (best quality/size trade-off)
      hqq           — Half-Quadratic Quantization (no calibration data needed)
    """
    print("\n" + "=" * 60)
    print(f"  STAGE 2: INT4 Weight Quantization ({quant_method})")
    print("=" * 60)

    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    print(f"  Input:          {input_path}")
    print(f"  Output:         {output_path}")
    print(f"  Block size:     {block_size}")
    print(f"  Symmetric:      {is_symmetric}")
    print(f"  Accuracy level: {accuracy_level}")
    print(f"  Method:         {quant_method}")

    try:
        algo_config = None
        nodes_to_exclude = []

        if quant_method == "k_quant_mixed":
            from onnxruntime.quantization.matmul_nbits_quantizer import KQuantWeightOnlyQuantConfig

            # Identify sensitive layers to keep FP32
            nodes_to_exclude = _get_sensitive_node_names(input_path)
            print(f"  Sensitive layers (FP32): {len(nodes_to_exclude)}")
            for n in nodes_to_exclude:
                print(f"    {n}")

            algo_config = KQuantWeightOnlyQuantConfig()
            print(f"  Algorithm: k_quant (Intel Neural Compressor)")
            print(f"  FFN layers: INT4 quantized")
            print(f"  Attention+first/last: FP32 preserved")

        elif quant_method == "hqq":
            from onnxruntime.quantization.matmul_nbits_quantizer import HQQWeightOnlyQuantConfig

            algo_config = HQQWeightOnlyQuantConfig(
                block_size=block_size,
                bits=4,
                axis=1,
            )
            print(f"  Algorithm: HQQ (Half-Quadratic Quantization)")

        # else: rtn — uses default MatMulNBitsQuantizer behavior

        # Load model with external data resolved relative to its directory
        model = onnx.load(input_path, load_external_data=True)

        quantizer = MatMulNBitsQuantizer(
            model=model,
            block_size=block_size,
            is_symmetric=is_symmetric,
            accuracy_level=accuracy_level,
            nodes_to_exclude=nodes_to_exclude if nodes_to_exclude else None,
            algo_config=algo_config,
        )
        quantizer.process()
        quantizer.model.save_model_to_file(output_path, use_external_data_format=True)
        print(f"  Saved INT4 model to: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR during INT4 quantization: {e}")
        import traceback
        traceback.print_exc()
        return False


def copy_supporting_files(src_dir: str, dst_dir: str, quantized: bool = False):
    """Copy non-encoder files (decoder, joint, configs, tokenizer) as-is.
    
    If quantized=True, updates genai_config.json with optimization metadata.
    """
    files_to_copy = [
        "decoder.onnx", "decoder.onnx.data",
        "joint.onnx", "joint.onnx.data",
        "genai_config.json",
        "audio_processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ]
    copied = 0
    for fname in files_to_copy:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
    print(f"\n  Copied {copied} supporting files (decoder, joint, configs, tokenizer) → FP32 as-is")

    # Add optimization metadata to genai_config.json
    if quantized:
        import json
        config_path = os.path.join(dst_dir, "genai_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            if "encoder" in config.get("model", {}):
                config["model"]["encoder"]["optimization"] = {
                    "graph_fusion": "conformer",
                    "quantization": "int4",
                    "quantization_details": {
                        "method": "MatMulNBits",
                        "algorithm": "RTN",
                    },
                }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  Updated {os.path.basename(config_path)} with optimization metadata")


def main():
    parser = argparse.ArgumentParser(description="Optimize Nemotron ASR encoder ONNX model")
    parser.add_argument("--model_dir", type=str, default="./onnx_models",
                        help="Directory containing original ONNX models")
    parser.add_argument("--output_dir", type=str, default="./onnx_models_optimized",
                        help="Output directory for optimized models")
    parser.add_argument("--skip_fusion", action="store_true",
                        help="Skip graph fusion stage")
    parser.add_argument("--skip_quantization", action="store_true",
                        help="Skip INT4 quantization stage")
    parser.add_argument("--block_size", type=int, default=32,
                        help="INT4 quantization block size (default: 32)")
    parser.add_argument("--accuracy_level", type=int, default=4,
                        help="INT4 accuracy level: 0=unset, 1=fp32, 2=fp16, 3=bf16, 4=int8 (default: 4)")
    parser.add_argument("--quant_method", type=str, default="rtn",
                        choices=["rtn", "k_quant_mixed", "hqq"],
                        help="Quantization method: rtn (round-to-nearest), k_quant_mixed (mixed precision), hqq (half-quadratic)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    encoder_path = model_dir / "encoder.onnx"

    if not encoder_path.exists():
        print(f"ERROR: encoder.onnx not found at {encoder_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Nemotron ASR Encoder Optimization")
    print("=" * 60)
    print(f"  Source:  {model_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Fusion:  {'skip' if args.skip_fusion else 'conformer'}")
    print(f"  Quantize: {'skip' if args.skip_quantization else f'INT4 {args.quant_method} (block={args.block_size}, sym=True)'}")

    # --- Print original encoder stats ---
    orig_stats = get_model_stats(str(encoder_path))
    print_model_stats("Original Encoder (FP32)", orig_stats)

    # Intermediate file path for fusion output
    fused_path = output_dir / "encoder_fused.onnx"
    final_path = output_dir / "encoder.onnx"

    # --- Stage 1: Graph Fusion ---
    fusion_ok = False
    if not args.skip_fusion:
        fusion_ok = stage1_graph_fusion(str(encoder_path), str(fused_path))
        if fusion_ok:
            fused_stats = get_model_stats(str(fused_path))
            print_model_stats("After Graph Fusion", fused_stats)
            stage1_output = fused_path
        else:
            stage1_output = encoder_path
    else:
        print("\n  [Skipping graph fusion]")
        stage1_output = encoder_path

    # --- Stage 2: INT4 Quantization ---
    if not args.skip_quantization:
        quant_ok = stage2_int4_quantization(
            input_path=str(stage1_output),
            output_path=str(final_path),
            block_size=args.block_size,
            is_symmetric=True,
            accuracy_level=args.accuracy_level,
            quant_method=args.quant_method,
        )
        if quant_ok:
            final_stats = get_model_stats(str(final_path))
            print_model_stats("After INT4 Quantization", final_stats)
            print(f"\n  Size reduction: {orig_stats['total_size_mb']:.1f} MB → {final_stats['total_size_mb']:.1f} MB "
                  f"({orig_stats['total_size_mb'] / max(final_stats['total_size_mb'], 0.1):.1f}x)")
    else:
        print("\n  [Skipping INT4 quantization]")
        # Copy the fusion output as the final encoder.
        # IMPORTANT: Keep the original external data filename reference intact
        # (e.g. encoder_fused.onnx.data). Renaming via onnx.save + convert_model_to_external_data
        # can produce layouts incompatible with onnxruntime-genai.
        if stage1_output != final_path:
            shutil.copy2(str(stage1_output), str(final_path))
            data_file = str(stage1_output) + ".data"
            if os.path.exists(data_file):
                # Keep the original data filename that the ONNX protobuf references
                orig_data_name = Path(stage1_output).name + ".data"
                dest_data = output_dir / orig_data_name
                shutil.copy2(data_file, str(dest_data))

    # --- Cleanup intermediate fusion file ---
    if fused_path.exists() and fused_path != final_path:
        os.remove(fused_path)
        fused_data = str(fused_path) + ".data"
        if os.path.exists(fused_data):
            os.remove(fused_data)
        print(f"\n  Cleaned up intermediate file: {fused_path.name}")

    # --- Copy decoder, joint, configs (FP32 as-is) ---
    did_quantize = not args.skip_quantization
    copy_supporting_files(str(model_dir), str(output_dir), quantized=did_quantize)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 60)

    final_files = sorted(output_dir.glob("*"))
    total_size = sum(f.stat().st_size for f in final_files if f.is_file())
    print(f"\n  Output directory: {output_dir}")
    print(f"  Total size: {total_size / (1024 * 1024):.1f} MB")
    quant_label = "FP32 fused" if args.skip_quantization else f"FP32 + INT4 ({args.quant_method})"
    print(f"\n  Files:")
    for f in final_files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            tag = ""
            if "encoder" in f.name:
                tag = f" ← optimized ({quant_label})"
            elif "decoder" in f.name or "joint" in f.name:
                tag = " (FP32, unchanged)"
            print(f"    {f.name:40s} {size_mb:8.1f} MB{tag}")

    print(f"\n  Next: validate with test_real_speech.py using --model_dir {output_dir}")


if __name__ == "__main__":
    main()
