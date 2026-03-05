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
  3. (Optional) Conv INT8 — with --conv_int8, remaining FP32 Conv ops are
       quantized to ConvInteger (INT8) via dynamic quantization.  This reduces
       model size further but may hurt inference speed.
  4. (Optional) Decoder/Joint quantization — with --lstm_int8, both decoder
       and joint models are quantized:
       - LSTM ops → DynamicQuantizeLSTM (com.microsoft, INT8 weights)
       - MatMul ops → MatMulNBits (same --dtype/--quant_method as encoder)
  5. Constant input folding — the ``length`` input (always equal to the chunk
       size) is converted to a graph initializer so ORT can constant-fold
       downstream mask/padding ops at session creation.  Disable with
       --no_fold_constants.

The decoder and joint models are tiny (<35 MB combined) and stay FP32 unless
--lstm_int8 is used.

Usage:
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype fp16
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype int4
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype int8
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype fp32
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype int4 --quant_method k_quant_mixed
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype int4 --quant_method k_quant
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype int4 --conv_int8
    python optimize_encoder.py --model_dir ./onnx_models --output_dir ./out --dtype int4 --quant_method k_quant --lstm_int8
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
        "MatMul", "MatMulInteger", "Gemm",
        "Conv", "LSTM",
        "DynamicQuantizeLSTM", "com.microsoft::DynamicQuantizeLSTM",
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
    """Identify sensitive MatMul nodes that should use INT8 in k_quant_mixed.

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
      k_quant       — K-quant with all nodes at the specified bit width
      k_quant_mixed — K-quant with sensitive layers quantized to INT8
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

        elif quant_method == "k_quant":
            from onnxruntime.quantization.matmul_nbits_quantizer import KQuantWeightOnlyQuantConfig

            algo_config = KQuantWeightOnlyQuantConfig()
            print(f"  Algorithm: k_quant (all nodes INT{bits})")

        elif quant_method == "k_quant_mixed":
            from onnxruntime.quantization.matmul_nbits_quantizer import KQuantWeightOnlyQuantConfig

            sensitive_names = _get_sensitive_node_names(input_path)
            # Sensitive nodes get INT8 instead of INT4 (not excluded as FP32)
            customized_weight_config = {name: {"bits": 8} for name in sensitive_names}
            print(f"  Sensitive layers (INT8): {len(sensitive_names)}")
            for n in sensitive_names:
                print(f"    {n}")
            algo_config = KQuantWeightOnlyQuantConfig(customized_weight_config=customized_weight_config)
            print(f"  Algorithm: k_quant (Intel Neural Compressor)")
            print(f"  FFN layers: INT4 quantized")
            print(f"  Attention+first/last: INT8 quantized")

        elif quant_method == "hqq":
            from onnxruntime.quantization.matmul_nbits_quantizer import HQQWeightOnlyQuantConfig

            algo_config = HQQWeightOnlyQuantConfig(
                block_size=block_size,
                bits=bits,
                axis=1,
            )
            print(f"  Algorithm: HQQ (Half-Quadratic Quantization)")

        else:
            # rtn — Round-to-nearest symmetric
            from onnxruntime.quantization.matmul_nbits_quantizer import RTNWeightOnlyQuantConfig

            algo_config = RTNWeightOnlyQuantConfig()
            print(f"  Algorithm: RTN (Round-to-Nearest)")

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


def stage3_conv_int8_quantization(input_path: str, output_path: str) -> bool:
    """Quantize remaining FP32 Conv and MatMul ops to INT8 via dynamic quantization.

    This is intended to run *after* MatMulNBits quantization, which only
    converts large MatMul ops into MatMulNBits.  Any Conv or MatMul ops
    still in FP32 will be converted to ConvInteger / MatMulInteger.

    Note: ConvInteger kernels in ORT are generally slower than optimised
    FP32 Conv or MatMulNBits.  Use this option when model size reduction
    matters more than inference speed.
    """
    print("\n" + "=" * 60)
    print("  STAGE 3: Conv/MatMul INT8 Dynamic Quantization")
    print("=" * 60)

    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Target ops: Conv → ConvInteger, MatMul → MatMulInteger (dynamic INT8)")

    try:
        # Count target ops before quantization
        model_meta = onnx.load(input_path, load_external_data=False)
        conv_count = sum(1 for n in model_meta.graph.node if n.op_type == "Conv")
        matmul_count = sum(1 for n in model_meta.graph.node if n.op_type == "MatMul")
        del model_meta
        print(f"  Conv ops found:   {conv_count}")
        print(f"  MatMul ops found: {matmul_count}")

        if conv_count == 0 and matmul_count == 0:
            print("  No Conv/MatMul ops to quantize — skipping.")
            if str(Path(input_path).resolve()) != str(Path(output_path).resolve()):
                shutil.copy2(input_path, output_path)
                data_file = input_path + ".data"
                if os.path.exists(data_file):
                    shutil.copy2(data_file, output_path + ".data")
            return True

        # Load model with external data and pass as ModelProto to avoid
        # infer_shapes_path failing on large external-data models.
        # Must load before removing output files since input and output may be the same path.
        model = onnx.load(input_path, load_external_data=True)

        # Remove stale output files before saving
        for f in [output_path, output_path + ".data"]:
            if os.path.exists(f):
                os.remove(f)

        quantize_dynamic(
            model_input=model,
            model_output=output_path,
            op_types_to_quantize=["Conv", "MatMul"],
            weight_type=QuantType.QUInt8,
            use_external_data_format=True,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
        print(f"  Saved INT8 model to: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR during Conv/MatMul INT8 quantization: {e}")
        import traceback
        traceback.print_exc()
        return False


def quantize_decoder_lstm(input_path: str, output_path: str) -> bool:
    """Quantize LSTM ops in the decoder model to DynamicQuantizeLSTM (com.microsoft).

    Uses onnxruntime quantize_dynamic to convert standard LSTM ops into
    DynamicQuantizeLSTM with INT8 quantized weights (W, R) and float32
    activations.  The quantized op uses per-channel (per-gate) INT8 weights
    with scale/zero_point parameters for both W and R matrices.

    This is a CPU-only contrib op that can reduce decoder model size with
    minimal accuracy impact since LSTM weights are small.
    """
    print("\n" + "=" * 60)
    print("  Decoder LSTM → DynamicQuantizeLSTM (INT8)")
    print("=" * 60)

    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Target: LSTM → com.microsoft::DynamicQuantizeLSTM (INT8 weights)")

    try:
        model_meta = onnx.load(input_path, load_external_data=False)
        lstm_count = sum(1 for n in model_meta.graph.node if n.op_type == "LSTM")
        del model_meta
        print(f"  LSTM ops found: {lstm_count}")

        if lstm_count == 0:
            print("  No LSTM ops to quantize — copying as-is.")
            if str(Path(input_path).resolve()) != str(Path(output_path).resolve()):
                shutil.copy2(input_path, output_path)
                data_file = input_path + ".data"
                if os.path.exists(data_file):
                    shutil.copy2(data_file, output_path + ".data")
            return True

        model = onnx.load(input_path, load_external_data=True)

        for f in [output_path, output_path + ".data"]:
            if os.path.exists(f):
                os.remove(f)

        quantize_dynamic(
            model_input=model,
            model_output=output_path,
            op_types_to_quantize=["LSTM"],
            weight_type=QuantType.QInt8,
            use_external_data_format=True,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )

        # Verify the conversion
        out_model = onnx.load(output_path, load_external_data=False)
        dq_lstm_count = sum(
            1 for n in out_model.graph.node
            if n.op_type == "DynamicQuantizeLSTM"
        )
        del out_model
        print(f"  DynamicQuantizeLSTM ops created: {dq_lstm_count}")
        print(f"  Saved quantized decoder to: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR during decoder LSTM quantization: {e}")
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


def stage_fold_constant_inputs(input_path: str, output_path: str) -> bool:
    """Fold constant-valued graph inputs into initializers.

    In the streaming encoder, the ``length`` input is always equal to the
    chunk size (the second dimension of ``audio_signal``).  When that
    dimension is static (fixed integer, not symbolic), ``length`` is
    converted from a dynamic input to a constant initializer so ORT can
    constant-fold the downstream mask / padding computation chain at
    session creation time, removing unnecessary runtime ops.

    If ``audio_signal`` has a dynamic (symbolic) sequence dimension, the
    chunk size cannot be determined at export time and ``length`` is left
    as a dynamic input.

    After folding, the encoder graph has 4 inputs (audio_signal,
    cache_last_channel, cache_last_time, cache_last_channel_len).

    Returns True if the model was modified and saved, False otherwise.
    """
    import numpy as np

    try:
        model = onnx.load(input_path, load_external_data=True)
        graph = model.graph

        # Derive the chunk size from audio_signal shape [batch, chunk_frames, feat_dim].
        # Only fold when the dimension is a fixed integer (static shape).
        chunk_size = None
        for inp in graph.input:
            if inp.name == "audio_signal":
                dims = inp.type.tensor_type.shape.dim
                if len(dims) >= 2:
                    dim = dims[1]
                    if dim.dim_value > 0 and not dim.dim_param:
                        chunk_size = dim.dim_value
                    elif dim.dim_param:
                        print(f"  [SKIP] audio_signal has dynamic sequence dimension "
                              f"('{dim.dim_param}') — cannot fold length")
                        return False
                break

        if chunk_size is None:
            print("  [SKIP] Could not determine static chunk size from audio_signal — skipping length folding")
            return False

        # Check that 'length' is a graph input (not already an initializer)
        length_input = None
        for inp in graph.input:
            if inp.name == "length":
                length_input = inp
                break

        if length_input is None:
            print("  [SKIP] No 'length' input found — nothing to fold")
            return False

        init_names = {init.name for init in graph.initializer}
        if "length" in init_names:
            print("  [SKIP] 'length' is already an initializer — nothing to fold")
            return False

        # Create a constant initializer with value = chunk_size
        length_tensor = onnx.numpy_helper.from_array(
            np.array([chunk_size], dtype=np.int64), name="length"
        )
        graph.initializer.append(length_tensor)

        # Remove 'length' from graph inputs (it is now supplied by the initializer)
        inputs_to_keep = [inp for inp in graph.input if inp.name != "length"]
        del graph.input[:]
        graph.input.extend(inputs_to_keep)

        # Save
        out_dir = os.path.dirname(output_path)
        data_name = os.path.basename(output_path) + ".data"
        data_file = os.path.join(out_dir, data_name)
        if os.path.exists(data_file):
            os.remove(data_file)
        if os.path.exists(output_path) and output_path != input_path:
            os.remove(output_path)

        onnx.save(
            model, output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
        )

        print(f"  [OK] Folded 'length' input as constant (value={chunk_size})")

        # Run ORT constant-folding to propagate the constant through
        # downstream ops (Unsqueeze, Less, Cast, Add, Expand, etc.)
        _ort_constant_fold(output_path)

        print(f"  Saved to: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR during constant input folding: {e}")
        import traceback
        traceback.print_exc()
        return False


def copy_supporting_files(src_dir: str, dst_dir: str, skip_prefixes: list | None = None):
    """Copy all non-encoder files from src_dir to dst_dir as-is.

    Args:
        skip_prefixes: Additional filename prefixes to skip (e.g. ["decoder"]
            when the decoder is handled separately by LSTM quantization).
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    skip_prefixes = skip_prefixes or []
    copied = 0
    for src_file in src_path.iterdir():
        if not src_file.is_file():
            continue
        # Skip encoder files — those are produced by the optimization pipeline
        if src_file.name.startswith("encoder"):
            continue
        # Skip files handled by other stages (e.g. decoder with --lstm_int8)
        if any(src_file.name.startswith(p) for p in skip_prefixes):
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
                        choices=["rtn", "k_quant", "k_quant_mixed", "hqq"],
                        help="INT4 quantization method: rtn (round-to-nearest), k_quant (all nodes same bits), k_quant_mixed (sensitive layers INT8), hqq (half-quadratic)")
    parser.add_argument("--conv_int8", action="store_true",
                        help="After MatMulNBits quantization, also quantize remaining FP32 Conv ops to INT8 (ConvInteger) via dynamic quantization")
    parser.add_argument("--lstm_int8", action="store_true",
                        help="Quantize decoder/joint models: LSTM → DynamicQuantizeLSTM (INT8), MatMul → MatMulNBits (same --dtype/--quant_method as encoder)")
    parser.add_argument("--no_fold_constants", action="store_true",
                        help="Skip folding constant inputs (e.g. 'length') into the encoder graph")
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

            # --- Optional: quantize remaining Conv ops to INT8 ---
            if args.conv_int8:
                conv_ok = stage3_conv_int8_quantization(
                    input_path=str(final_path),
                    output_path=str(final_path),
                )
                if conv_ok:
                    final_stats = get_model_stats(str(final_path))
                    print_model_stats(f"After INT{bits} + Conv/MatMul INT8 Quantization", final_stats)
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

    # --- Fold constant inputs (e.g. 'length' → chunk_size) ---
    if not args.no_fold_constants:
        print("\n--- Folding Constant Inputs ---")
        fold_ok = stage_fold_constant_inputs(
            input_path=str(final_path),
            output_path=str(final_path),
        )
        if fold_ok:
            final_stats = get_model_stats(str(final_path))
            print_model_stats("After Constant Folding", final_stats)

    # --- Copy decoder, joint, configs (as-is or selectively) ---
    skip_prefixes = []
    if args.lstm_int8:
        skip_prefixes.extend(["decoder", "joint"])
    copy_supporting_files(str(model_dir), str(output_dir), skip_prefixes=skip_prefixes)

    # --- Optional: Quantize decoder & joint (MatMulNBits + DynamicQuantizeLSTM) ---
    if args.lstm_int8:
        quant_bits = int(args.dtype[3:]) if args.dtype in ("int4", "int8") else None

        for model_name in ["decoder", "joint"]:
            src_path = model_dir / f"{model_name}.onnx"
            if not src_path.exists():
                print(f"\n  WARNING: --lstm_int8 specified but {model_name}.onnx not found at {src_path}")
                print(f"  Skipping {model_name} quantization.")
                continue

            dst_path = output_dir / f"{model_name}.onnx"
            orig_stats = get_model_stats(str(src_path))
            print_model_stats(f"Original {model_name.capitalize()} (FP32)", orig_stats)

            # Current working path — start from the source
            current_path = str(src_path)

            # Step 1: MatMulNBits quantization (same method as encoder)
            if quant_bits is not None:
                ok = stage2_int_quantization(
                    input_path=current_path,
                    output_path=str(dst_path),
                    bits=quant_bits,
                    block_size=args.block_size,
                    is_symmetric=True,
                    accuracy_level=args.accuracy_level,
                    quant_method=args.quant_method if quant_bits == 4 else "k_quant",
                )
                if ok:
                    current_path = str(dst_path)
                    step_stats = get_model_stats(current_path)
                    print_model_stats(
                        f"{model_name.capitalize()} after MatMulNBits (INT{quant_bits})", step_stats
                    )

            # Step 2: DynamicQuantizeLSTM for any LSTM ops
            ok = quantize_decoder_lstm(
                input_path=current_path,
                output_path=str(dst_path),
            )
            if ok:
                final_stats = get_model_stats(str(dst_path))
                print_model_stats(f"{model_name.capitalize()} after Quantization", final_stats)

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
            elif ("decoder" in f.name or "joint" in f.name):
                tag = " ← quantized (MatMulNBits + DynamicQuantizeLSTM)" if args.lstm_int8 else " (FP32, unchanged)"
            print(f"    {f.name:40s} {size_mb:8.1f} MB{tag}")

    print(f"\n  Next: validate with test_real_speech.py using --model_dir {output_dir}")


if __name__ == "__main__":
    main()
