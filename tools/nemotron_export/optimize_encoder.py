#!/usr/bin/env python3
"""
Optimize Nemotron ASR encoder ONNX model.

Two-stage optimization:
  1. Graph fusion  — Fuse Conformer attention subgraphs into MultiHeadAttention,
                     SkipLayerNormalization, BiasGelu, etc.
  2. Dtype conversion / quantization (controlled by --dtype):
       - fp32: No conversion (fusion only)
       - fp16: Convert to mixed FP16 via OnnxModel.convert_float_to_float16,
               keeping graph I/O and certain ops (e.g. LayerNormalization) in FP32.
       - int8: Dynamic INT8 quantization via onnxruntime quantize_dynamic.
       - int4: INT4 weight quantization via MatMulNBits (RTN, k_quant_mixed, or HQQ).

The decoder and joint models are tiny (<35 MB combined) and stay FP32.

Usage:
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype fp16
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype int4
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype int8
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype fp32
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype int4 --quant_method k_quant_mixed
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
    """Apply Conformer graph fusion + graph cleanup to the encoder model.

    Steps:
      1. Conformer-specific fusions (MHA, SkipLayerNorm, BiasGelu, …)
      2. OnnxModel cleanup passes (cascaded Cast, useless Cast,
         duplicate initializers, unused constants)
      3. ORT session-level constant folding (Add+Add, Sub/Div with
         constants, identity elimination, etc.)
    """
    print("\n" + "=" * 60)
    print("  STAGE 1: Graph Fusion (Conformer) + Cleanup")
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

        # --- OnnxModel graph cleanup passes ---
        print("\n  Graph cleanup passes:")

        # Remove Cast -> Cast chains (e.g. float32->float16->float32)
        optimized_model.remove_cascaded_cast_nodes()
        print("    [OK] remove_cascaded_cast_nodes")

        # Remove Cast nodes where input and output types match
        optimized_model.remove_useless_cast_nodes()
        print("    [OK] remove_useless_cast_nodes")

        # Remove duplicate initializers (same weight referenced by different names)
        # optimized_model.remove_duplicated_initializer(cache=None)
        # print("    [OK] remove_duplicated_initializer")

        # Remove constant nodes that are no longer consumed
        optimized_model.remove_unused_constant()
        print("    [OK] remove_unused_constant")

        # Final prune: drop unreachable nodes and stale value_info entries
        optimized_model.prune_graph()
        print("    [OK] prune_graph")

        optimized_model.save_model_to_file(
            output_path,
            use_external_data_format=True,
        )
        print(f"  Saved fused + cleaned model to: {output_path}")

        # --- ORT session-level constant folding ---
        # ORT_ENABLE_BASIC applies constant folding, redundant-op elimination,
        # and other semantics-preserving rewrites (e.g. Add(c1,Add(x,c2))→Add(x,c1+c2)).
        _ort_constant_fold(output_path)

        return True

    except Exception as e:
        print(f"  ERROR during graph fusion: {e}")
        print(f"  Skipping fusion, will quantize from original model directly.")
        return False


def _ort_constant_fold(model_path: str):
    """Run ORT session-level constant folding on an ONNX model in-place.

    Creates an ORT session with ORT_ENABLE_BASIC which folds constant
    sub-expressions (Add+Add, Mul chains, identity ops, etc.) and writes
    the optimized graph back to the same path.
    """
    import tempfile
    import onnxruntime as ort

    print("\n  ORT constant-folding pass:")
    out_dir = str(Path(model_path).parent)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".onnx", dir=out_dir)
    os.close(tmp_fd)

    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = tmp_path
        # Creating the session triggers optimization and saves the result
        ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])

        # Count node reduction
        orig_model = onnx.load(model_path, load_external_data=False)
        opt_model = onnx.load(tmp_path, load_external_data=False)
        orig_nodes = len(orig_model.graph.node)
        opt_nodes = len(opt_model.graph.node)
        del orig_model, opt_model

        if opt_nodes < orig_nodes:
            # ORT embeds weights inline; re-save with external data for large models
            full_model = onnx.load(tmp_path, load_external_data=True)
            data_name = Path(model_path).name + ".data"
            data_file = Path(out_dir) / data_name
            if data_file.exists():
                data_file.unlink()
            onnx.save(
                full_model, model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_name,
                size_threshold=1024,
            )
            print(f"    [OK] Reduced {orig_nodes} → {opt_nodes} nodes "
                  f"({orig_nodes - opt_nodes} removed)")
        else:
            print(f"    [OK] No further reduction ({orig_nodes} nodes)")
    except Exception as e:
        print(f"    [SKIP] ORT constant folding failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        # ORT may create a .data sidecar for the temp file
        tmp_data = tmp_path + ".data"
        if os.path.exists(tmp_data):
            os.remove(tmp_data)


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


def stage2_int_quantization(
    input_path: str,
    output_path: str,
    bits: int = 4,
    block_size: int = 32,
    is_symmetric: bool = True,
    accuracy_level: int = 4,
    quant_method: str = "k_quant_mixed",
) -> bool:
    """Quantize FP32 MatMul weights to INT4 or INT8 (MatMulNBits).

    Args:
        bits: 4 or 8.  When 8, all MatMul nodes are set to {"bits": 8}
              via customized_weight_config so the KQuant algorithm
              quantises every weight to 8-bit.
        quant_method: rtn | k_quant_mixed | hqq (only used for bits=4).

    Methods (bits=4):
      rtn           — Round-to-nearest symmetric (simplest)
      k_quant_mixed — K-quant with sensitive layers excluded
      hqq           — Half-Quadratic Quantization (no calibration data)
    """
    label = f"INT{bits}"
    print("\n" + "=" * 60)
    print(f"  STAGE 2: {label} Weight Quantization ({quant_method})")
    print("=" * 60)

    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    print(f"  Input:          {input_path}")
    print(f"  Output:         {output_path}")
    print(f"  Bits:           {bits}")
    print(f"  Block size:     {block_size}")
    print(f"  Symmetric:      {is_symmetric}")
    print(f"  Accuracy level: {accuracy_level}")
    print(f"  Method:         {quant_method}")

    try:
        algo_config = None
        nodes_to_exclude = []

        if bits == 8:
            # INT8: set every MatMul to 8-bit via customized_weight_config
            from onnxruntime.quantization.matmul_nbits_quantizer import KQuantWeightOnlyQuantConfig

            model_meta = onnx.load(input_path, load_external_data=False)
            customized_weight_config = {
                node.name: {"bits": 8}
                for node in model_meta.graph.node
                if node.op_type == "MatMul"
            }
            del model_meta
            print(f"  MatMul nodes:   {len(customized_weight_config)} (all set to bits=8)")
            algo_config = KQuantWeightOnlyQuantConfig(customized_weight_config=customized_weight_config)

        elif quant_method == "k_quant_mixed":
            from onnxruntime.quantization.matmul_nbits_quantizer import KQuantWeightOnlyQuantConfig

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
                bits=bits,
                axis=1,
            )
            print(f"  Algorithm: HQQ (Half-Quadratic Quantization)")

        # else: rtn — uses default MatMulNBitsQuantizer behavior

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

        # Remove stale output files before saving (input may be same path)
        for f in [output_path, output_path + ".data"]:
            if os.path.exists(f):
                os.remove(f)

        quantizer.model.save_model_to_file(output_path, use_external_data_format=True)
        print(f"  Saved {label} model to: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR during {label} quantization: {e}")
        import traceback
        traceback.print_exc()
        return False


# Ops that should stay FP32 during FP16 conversion (numerically sensitive)
FP16_OP_BLOCK_LIST = [
    "LayerNormalization",
    "SkipLayerNormalization",
    "Softmax",
    "ReduceMean",
]


def stage2_fp16_conversion(input_path: str, output_path: str) -> bool:
    """Convert FP32 model to mixed FP16 precision.

    Uses OnnxModel.convert_float_to_float16 with:
      - keep_io_types=True    — graph inputs/outputs stay FP32
      - op_block_list          — LayerNorm, Softmax, ReduceMean stay FP32
    """
    print("\n" + "=" * 60)
    print("  STAGE 2: FP16 Mixed-Precision Conversion")
    print("=" * 60)

    from onnxruntime.transformers.onnx_model import OnnxModel

    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Ops kept FP32: {FP16_OP_BLOCK_LIST}")

    try:
        model = onnx.load(input_path, load_external_data=True)
        onnx_model = OnnxModel(model)
        onnx_model.convert_float_to_float16(
            use_symbolic_shape_infer=True,
            keep_io_types=True,
            op_block_list=FP16_OP_BLOCK_LIST,
        )

        # Remove stale output files before saving (input may be same path)
        for f in [output_path, output_path + ".data"]:
            if os.path.exists(f):
                os.remove(f)

        onnx_model.save_model_to_file(output_path, use_external_data_format=True)
        print(f"  Saved FP16 model to: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR during FP16 conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def copy_supporting_files(src_dir: str, dst_dir: str):
    """Copy all non-encoder files from src_dir to dst_dir as-is."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    copied = 0
    for src_file in src_path.iterdir():
        if not src_file.is_file():
            continue
        # Skip encoder files — those are produced by the optimization pipeline
        if src_file.name.startswith("encoder"):
            continue
        dst_file = dst_path / src_file.name
        if src_file.resolve() != dst_file.resolve():
            shutil.copy2(str(src_file), str(dst_file))
            copied += 1
    print(f"\n  Copied {copied} supporting files (decoder, joint, configs, tokenizer, etc.) → as-is")


def main():
    parser = argparse.ArgumentParser(description="Optimize Nemotron ASR encoder ONNX model")
    parser.add_argument("--model_dir", type=str, default="./onnx_models",
                        help="Directory containing original ONNX models")
    parser.add_argument("--output_dir", type=str, default="./onnx_models_optimized",
                        help="Output directory for optimized models")
    parser.add_argument("--skip_fusion", action="store_true",
                        help="Skip graph fusion stage")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp32", "fp16", "int8", "int4"],
                        help="Target dtype: fp32 (fusion only), fp16, int8, or int4 (default: fp32)")
    parser.add_argument("--block_size", type=int, default=32,
                        help="INT4 quantization block size (default: 32)")
    parser.add_argument("--accuracy_level", type=int, default=4,
                        help="INT4 accuracy level: 0=unset, 1=fp32, 2=fp16, 3=bf16, 4=int8 (default: 4)")
    parser.add_argument("--quant_method", type=str, default="k_quant_mixed",
                        choices=["rtn", "k_quant_mixed", "hqq"],
                        help="INT4 quantization method: rtn (round-to-nearest), k_quant_mixed (mixed precision), hqq (half-quadratic)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    encoder_path = model_dir / "encoder.onnx"

    if not encoder_path.exists():
        print(f"ERROR: encoder.onnx not found at {encoder_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    dtype_label = {
        "fp32": "FP32 (fusion only)",
        "fp16": "FP16 mixed precision",
        "int8": "Dynamic INT8",
        "int4": f"INT4 {args.quant_method} (block={args.block_size}, sym=True)",
    }[args.dtype]

    print("=" * 60)
    print("  Nemotron ASR Encoder Optimization")
    print("=" * 60)
    print(f"  Source:  {model_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Fusion:  {'skip' if args.skip_fusion else 'conformer'}")
    print(f"  Dtype:   {dtype_label}")

    # --- Print original encoder stats ---
    orig_stats = get_model_stats(str(encoder_path))
    print_model_stats("Original Encoder (FP32)", orig_stats)

    # All stages write directly to the final encoder.onnx — no intermediate files.
    final_path = output_dir / "encoder.onnx"

    # --- Stage 1: Graph Fusion ---
    fusion_ok = False
    if not args.skip_fusion:
        fusion_ok = stage1_graph_fusion(str(encoder_path), str(final_path))
        if fusion_ok:
            fused_stats = get_model_stats(str(final_path))
            print_model_stats("After Graph Fusion", fused_stats)
            stage1_output = final_path
        else:
            stage1_output = encoder_path
    else:
        print("\n  [Skipping graph fusion]")
        stage1_output = encoder_path

    # --- Stage 2: Dtype conversion / quantization ---
    if args.dtype == "fp16":
        ok = stage2_fp16_conversion(
            input_path=str(stage1_output),
            output_path=str(final_path),
        )
        if ok:
            final_stats = get_model_stats(str(final_path))
            print_model_stats("After FP16 Conversion", final_stats)
            print(f"\n  Size reduction: {orig_stats['total_size_mb']:.1f} MB → {final_stats['total_size_mb']:.1f} MB "
                  f"({orig_stats['total_size_mb'] / max(final_stats['total_size_mb'], 0.1):.1f}x)")

    elif args.dtype in ("int8", "int4"):
        bits = int(args.dtype[3:])
        ok = stage2_int_quantization(
            input_path=str(stage1_output),
            output_path=str(final_path),
            bits=bits,
            block_size=args.block_size,
            is_symmetric=True,
            accuracy_level=args.accuracy_level,
            quant_method=args.quant_method,
        )
        if ok:
            final_stats = get_model_stats(str(final_path))
            print_model_stats(f"After INT{bits} Quantization", final_stats)
            print(f"\n  Size reduction: {orig_stats['total_size_mb']:.1f} MB → {final_stats['total_size_mb']:.1f} MB "
                  f"({orig_stats['total_size_mb'] / max(final_stats['total_size_mb'], 0.1):.1f}x)")

    else:
        # fp32 — no dtype conversion
        if stage1_output != final_path:
            shutil.copy2(str(stage1_output), str(final_path))
            data_file = Path(str(stage1_output) + ".data")
            if data_file.exists():
                dest_data = output_dir / "encoder.onnx.data"
                if data_file.resolve() != dest_data.resolve():
                    shutil.copy2(str(data_file), str(dest_data))

    # --- Copy decoder, joint, configs (as-is) ---
    copy_supporting_files(str(model_dir), str(output_dir))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 60)

    final_files = sorted(output_dir.glob("*"))
    total_size = sum(f.stat().st_size for f in final_files if f.is_file())
    print(f"\n  Output directory: {output_dir}")
    print(f"  Total size: {total_size / (1024 * 1024):.1f} MB")
    print(f"\n  Files:")
    for f in final_files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            tag = ""
            if "encoder" in f.name:
                tag = f" ← optimized ({dtype_label})"
            elif "decoder" in f.name or "joint" in f.name:
                tag = " (FP32, unchanged)"
            print(f"    {f.name:40s} {size_mb:8.1f} MB{tag}")

    print(f"\n  Next: validate with test_real_speech.py using --model_dir {output_dir}")


if __name__ == "__main__":
    main()
