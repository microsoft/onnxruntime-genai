#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
End-to-end demo: LLM inference with INT8 quantized KV cache on CPU.

This script demonstrates the full workflow:
1. Export a model using the onnxruntime-genai model builder with quantized KV cache
2. (Optional) Calibrate KV cache scales from sample data
3. Run text generation using onnxruntime-genai with CPU execution provider
4. Compare output quality and performance vs FP32 KV cache baseline

Prerequisites:
- onnxruntime (built from main branch with quantized KV cache support)
- onnxruntime-genai (with model builder changes for quantized KV cache)
- transformers, torch, numpy

Usage:
    # Quick demo with default scale (no calibration):
    python quantized_kv_cache_cpu_demo.py --model Qwen/Qwen2.5-0.5B-Instruct

    # With calibration for better accuracy:
    python quantized_kv_cache_cpu_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --calibrate

    # Using a pre-calibrated scale file:
    python quantized_kv_cache_cpu_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --scale_file kv_scales.json

    # Compare with FP32 baseline:
    python quantized_kv_cache_cpu_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --compare
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def get_builder_path():
    """Find the model builder script."""
    # Try relative path from this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "..", "src", "python", "py", "models", "builder.py"),
        os.path.join(script_dir, "../../src/python/py/models/builder.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    # Try to find via package
    try:
        import onnxruntime_genai  # noqa: PLC0415

        pkg_dir = os.path.dirname(onnxruntime_genai.__file__)
        builder_path = os.path.join(pkg_dir, "models", "builder.py")
        if os.path.exists(builder_path):
            return builder_path
    except ImportError:
        pass

    raise FileNotFoundError(
        "Could not find builder.py. Please run this script from the onnxruntime-genai repository "
        "or install onnxruntime-genai with model builder support."
    )


def get_calibration_path():
    """Find the calibration utility script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "..", "src", "python", "py", "models", "calibrate_kv_scales.py"),
        os.path.join(script_dir, "../../src/python/py/models/calibrate_kv_scales.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError("Could not find calibrate_kv_scales.py")


def export_model(model_name, output_dir, quant_type="int8_per_tensor", scale_file=None, cache_dir=None):
    """Export model using model builder with quantized KV cache."""
    builder_path = get_builder_path()

    extra_options = [f"kv_cache_quant_type={quant_type}"]
    if scale_file:
        extra_options.append(f"kv_cache_scale_file={scale_file}")

    cmd = [
        sys.executable,
        builder_path,
        "-m",
        model_name,
        "-o",
        output_dir,
        "-p",
        "fp32",
        "-e",
        "cpu",
        "--extra_options",
        *extra_options,
    ]
    if cache_dir:
        cmd.extend(["-c", cache_dir])

    print(f"\n{'=' * 60}")
    print(f"Exporting model with {quant_type} KV cache...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, check=False, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Model export failed with return code {result.returncode}")

    print(f"\nModel exported to: {output_dir}")
    return output_dir


def export_baseline_model(model_name, output_dir, cache_dir=None):
    """Export model with FP32 KV cache (baseline for comparison)."""
    builder_path = get_builder_path()

    cmd = [
        sys.executable,
        builder_path,
        "-m",
        model_name,
        "-o",
        output_dir,
        "-p",
        "fp32",
        "-e",
        "cpu",
    ]
    if cache_dir:
        cmd.extend(["-c", cache_dir])

    print(f"\n{'=' * 60}")
    print("Exporting baseline model (FP32 KV cache)...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, check=False, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Baseline model export failed with return code {result.returncode}")

    print(f"\nBaseline model exported to: {output_dir}")
    return output_dir


def calibrate_scales(model_name, output_file, quant_type="int8_per_tensor", num_samples=32, max_length=128):
    """Run KV cache scale calibration."""
    calibration_path = get_calibration_path()

    cmd = [
        sys.executable,
        calibration_path,
        "--model_name",
        model_name,
        "--quant_type",
        quant_type,
        "--output",
        output_file,
        "--num_samples",
        str(num_samples),
        "--max_length",
        str(max_length),
    ]

    print(f"\n{'=' * 60}")
    print("Calibrating KV cache scales...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, check=False, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Calibration failed with return code {result.returncode}")

    print(f"\nScales saved to: {output_file}")
    return output_file


def run_generation(model_path, prompts, max_length=100, verbose=True):
    """Run text generation using onnxruntime directly with greedy decoding."""
    if verbose:
        print(f"\nLoading model from: {model_path}")

    # Get the original model name from config for tokenizer
    tokenizer_path = model_path  # genai builder saves tokenizer files in model dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load ONNX model
    model_file = os.path.join(model_path, "model.onnx")
    sess = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])

    # Determine model structure from session inputs
    output_names = [out.name for out in sess.get_outputs()]

    # Find number of layers from KV cache inputs
    num_layers = 0
    kv_dtype = np.int8  # default for quantized
    for inp in sess.get_inputs():
        if inp.name.startswith("past_key_values.") and inp.name.endswith(".key"):
            num_layers += 1
            if "int8" in inp.type:
                kv_dtype = np.int8
            elif "uint8" in inp.type:
                kv_dtype = np.uint8
            elif "float" in inp.type:
                kv_dtype = np.float32

    # Get KV cache shape info: [batch, num_kv_heads, seq_len, head_dim]
    kv_input = sess.get_inputs()[2]  # first past_key input
    num_kv_heads = kv_input.shape[1] if isinstance(kv_input.shape[1], int) else 2
    head_dim = kv_input.shape[3] if isinstance(kv_input.shape[3], int) else 64

    if verbose:
        print(f"Model loaded: {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")
        print(f"KV cache dtype: {kv_dtype.__name__}")

    results = []
    total_tokens = 0
    total_time = 0.0

    eos_token_id = tokenizer.eos_token_id

    for prompt in prompts:
        if verbose:
            print(f"\nPrompt: {prompt}")

        input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int64)
        seq_len = input_ids.shape[1]

        # Initialize empty past KV cache
        past_kv = {}
        for i in range(num_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=kv_dtype)
            past_kv[f"past_key_values.{i}.value"] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=kv_dtype)

        generated_tokens = []
        current_ids = input_ids
        total_seq_len = seq_len

        start_time = time.perf_counter()

        for _step in range(max_length - seq_len):
            # Build attention mask
            attention_mask = np.ones((1, total_seq_len), dtype=np.int64)

            # Build feed dict
            feeds = {
                "input_ids": current_ids,
                "attention_mask": attention_mask,
            }
            feeds.update(past_kv)

            # Run inference
            outputs = sess.run(output_names, feeds)

            # Parse outputs: logits + present KV
            logits = outputs[0]  # [batch, seq, vocab]

            # Update past KV from present outputs
            past_kv = {}
            for i in range(num_layers):
                past_kv[f"past_key_values.{i}.key"] = outputs[1 + i]
                past_kv[f"past_key_values.{i}.value"] = outputs[1 + num_layers + i]

            # Greedy: pick the last token's top logit
            next_token_id = int(np.argmax(logits[0, -1, :]))
            generated_tokens.append(next_token_id)

            # Stop on EOS
            if next_token_id == eos_token_id:
                break

            # Next step: single token input
            current_ids = np.array([[next_token_id]], dtype=np.int64)
            total_seq_len += 1

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        num_tokens = len(generated_tokens)

        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        total_tokens += num_tokens
        total_time += elapsed

        results.append(
            {
                "prompt": prompt,
                "output": output_text,
                "num_tokens": num_tokens,
                "time_s": elapsed,
                "tokens_per_sec": num_tokens / elapsed if elapsed > 0 else 0,
            }
        )

        if verbose:
            print(f"Output: {output_text}")
            print(f"  Generated {num_tokens} tokens in {elapsed:.2f}s ({num_tokens / elapsed:.1f} tok/s)")

    summary = {
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "avg_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
    }

    return results, summary


def compare_outputs(baseline_results, quantized_results):
    """Compare generation outputs between baseline and quantized models."""
    print(f"\n{'=' * 60}")
    print("COMPARISON: FP32 KV Cache vs INT8 Quantized KV Cache")
    print(f"{'=' * 60}")

    for i, (base, quant) in enumerate(zip(baseline_results, quantized_results, strict=False)):
        print(f"\n--- Prompt {i + 1}: {base['prompt'][:50]}...")
        print(f"  FP32 output:  {base['output'][:100]}...")
        print(f"  INT8 output:  {quant['output'][:100]}...")
        match = base["output"] == quant["output"]
        print(f"  Exact match:  {'Yes' if match else 'No'}")
        if not match:
            # Find first difference
            for j, (c1, c2) in enumerate(zip(base["output"], quant["output"], strict=False)):
                if c1 != c2:
                    print(f"  First diff at char {j}: FP32='{c1}' INT8='{c2}'")
                    break


def print_performance_summary(baseline_summary, quantized_summary):
    """Print performance comparison."""
    print(f"\n{'=' * 60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  FP32 KV Cache:    {baseline_summary['avg_tokens_per_sec']:.1f} tokens/sec")
    print(f"  INT8 KV Cache:    {quantized_summary['avg_tokens_per_sec']:.1f} tokens/sec")
    if baseline_summary["avg_tokens_per_sec"] > 0:
        speedup = quantized_summary["avg_tokens_per_sec"] / baseline_summary["avg_tokens_per_sec"]
        print(f"  Speedup:          {speedup:.2f}x")
    print("\n  Note: KV cache memory reduction with INT8 is ~4x vs FP32.")
    print("  Speedup is most pronounced with longer sequences where")
    print("  memory bandwidth becomes the bottleneck.")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end demo: LLM inference with INT8 quantized KV cache on CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo (default scales):
  python quantized_kv_cache_cpu_demo.py --model Qwen/Qwen2.5-0.5B-Instruct

  # With calibration:
  python quantized_kv_cache_cpu_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --calibrate

  # Compare with FP32 baseline:
  python quantized_kv_cache_cpu_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --compare
        """,
    )
    parser.add_argument(
        "--model", "-m", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./quantized_kv_demo_output",
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="int8_per_tensor",
        choices=["int8_per_tensor", "int8_per_channel", "int4_per_tensor", "int4_per_channel"],
        help="KV cache quantization type",
    )
    parser.add_argument("--calibrate", action="store_true", help="Run calibration to compute optimal KV cache scales")
    parser.add_argument("--scale_file", type=str, default=None, help="Pre-computed scale file from calibration utility")
    parser.add_argument("--compare", action="store_true", help="Also export and run FP32 baseline for comparison")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--skip_export", action="store_true", help="Skip model export (use existing exported model)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for model downloads")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt (default: use built-in test prompts)")
    args = parser.parse_args()

    # Define test prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "The capital of France is",
            "Explain quantum computing in one sentence:",
            "Write a haiku about programming:",
        ]

    # Set up directories
    quant_model_dir = os.path.join(args.output_dir, f"model_{args.quant_type}")
    baseline_model_dir = os.path.join(args.output_dir, "model_fp32_baseline")
    scale_file = args.scale_file

    # Step 1: Calibrate scales (optional)
    if args.calibrate and not scale_file:
        scale_file = os.path.join(args.output_dir, "kv_scales.json")
        os.makedirs(args.output_dir, exist_ok=True)
        calibrate_scales(args.model, scale_file, args.quant_type)

    # Step 2: Export model with quantized KV cache
    if not args.skip_export:
        os.makedirs(quant_model_dir, exist_ok=True)
        export_model(args.model, quant_model_dir, args.quant_type, scale_file, args.cache_dir)

    # Step 3: Export baseline model (if comparison requested)
    if args.compare and not args.skip_export:
        os.makedirs(baseline_model_dir, exist_ok=True)
        export_baseline_model(args.model, baseline_model_dir, args.cache_dir)

    # Step 4: Run generation with quantized KV cache
    print(f"\n{'=' * 60}")
    print("Running generation with INT8 quantized KV cache...")
    print(f"{'=' * 60}")
    quant_results, quant_summary = run_generation(quant_model_dir, prompts, args.max_length)

    # Step 5: Run baseline (if comparison requested)
    if args.compare:
        print(f"\n{'=' * 60}")
        print("Running generation with FP32 KV cache (baseline)...")
        print(f"{'=' * 60}")
        baseline_results, baseline_summary = run_generation(baseline_model_dir, prompts, args.max_length)

        # Step 6: Compare
        compare_outputs(baseline_results, quant_results)
        print_performance_summary(baseline_summary, quant_summary)
    else:
        print(f"\n{'=' * 60}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Model: {args.model}")
        print(f"  KV Cache: {args.quant_type}")
        print(f"  Total tokens: {quant_summary['total_tokens']}")
        print(f"  Total time: {quant_summary['total_time_s']:.2f}s")
        print(f"  Throughput: {quant_summary['avg_tokens_per_sec']:.1f} tokens/sec")
        print("\n  Run with --compare to see FP32 baseline comparison.")

    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
