# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# This is an end-to-end benchmarking script for any multi-modal ONNX model pipeline.
#
# Prerequisites:
# 0) Install ONNX Runtime GenAI and ONNX Runtime
#
# 1) Create or download the desired ONNX model
#
# 2) Run this script with the desired arguments. Run benchmark_multimodal.py -h for help.

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time

import numpy as np
import onnxruntime_genai as og
import pandas as pd
import psutil
from tqdm import tqdm

peak_cpu_memory = 0.0
peak_gpu_memory = 0.0
peak_memory_lock = threading.Lock()
stop_monitoring = False

try:
    subprocess.run(["nvidia-smi"], check=True)
    IS_NVIDIA_SYSTEM = True
except Exception:
    IS_NVIDIA_SYSTEM = False


# Monitor the GPU memory usage
def monitor_gpu_memory():
    global peak_gpu_memory  # noqa: PLW0603

    while not stop_monitoring:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )

        memory_usage = result.stdout.splitlines()

        if len(memory_usage) >= 1:
            gpu_memory = [float(line) for line in memory_usage]
            current_peak = round(max(gpu_memory) / 1024, 2)
            with peak_memory_lock:
                peak_gpu_memory = max(current_peak, peak_gpu_memory)
        else:
            print("No GPU Memory Info Found")
        time.sleep(0.1)


# Monitor the CPU memory usage
def monitor_cpu_memory():
    global peak_cpu_memory  # noqa: PLW0603

    while not stop_monitoring:
        current_used_memory = round(psutil.virtual_memory().used / 1024**3, 2)
        with peak_memory_lock:
            peak_cpu_memory = max(peak_cpu_memory, current_used_memory)
        time.sleep(0.1)


def save_results(results, filename, baseline_csv=None):
    df = pd.DataFrame(
        results,
        columns=[
            "Prompt Length",
            "Tokens Generated",
            "Max Length",
            "Processing Latency (ms)",
            "Prompt Latency (ms)",
            "Token Generation Throughput (tps)",
            "Token Generation Latency (ms)",
            "Sampling Throughput (tps)",
            "Sampling Latency (ms)",
            "Wall Clock Throughput (tps)",
            "Wall Clock Time (s)",
            "Memory Usage (GiB)",
            "Image Count",
            "TTFT Median (ms)",
            "TTFT P95 (ms)",
            "Processing Median (ms)",
            "Processing P95 (ms)",
            "Run SHA256",
            "Output Tokens SHA256",
        ],
    )
    if baseline_csv:
        baseline = pd.read_csv(baseline_csv)
        keys = ["Prompt Length", "Tokens Generated", "Max Length", "Image Count", "Run SHA256", "Output Tokens SHA256"]
        if not {*keys, "TTFT Median (ms)"}.issubset(baseline.columns):
            raise ValueError("Baseline must be produced by this version of benchmark_multimodal.py")
        paired = df.merge(
            baseline[[*keys, "TTFT Median (ms)"]],
            on=keys,
            how="left",
            suffixes=("", " Baseline"),
            validate="one_to_one",
        )
        if len(paired) != len(baseline) or paired["TTFT Median (ms) Baseline"].isna().any():
            raise ValueError("Baseline inputs, configuration, or generated tokens differ from this run")
        df["TTFT Median Speedup"] = paired["TTFT Median (ms) Baseline"] / paired["TTFT Median (ms)"]
    df.to_csv(filename, header=True, index=False)
    print(f"Results saved in {filename}!")


def run_benchmark_memory(args, model, processor, image, audio, generation_length, max_length):
    """
    This function is to run benchmark and print the memory usage
    """
    global stop_monitoring  # noqa: PLW0603
    global peak_gpu_memory  # noqa: PLW0603
    global peak_cpu_memory  # noqa: PLW0603

    # Reset the peak memory variables and the monitoring flag
    stop_monitoring = False
    peak_gpu_memory = 0.0
    peak_cpu_memory = 0.0

    if IS_NVIDIA_SYSTEM:
        monitor_thread = threading.Thread(target=monitor_gpu_memory)
    else:
        monitor_thread = threading.Thread(target=monitor_cpu_memory)

    monitor_thread.start()

    try:
        metrics = run_benchmark(args, model, processor, image, audio, generation_length, max_length)
    finally:
        stop_monitoring = True
        monitor_thread.join()

    if IS_NVIDIA_SYSTEM:
        metrics.insert(11, peak_gpu_memory)
    else:
        metrics.insert(11, peak_cpu_memory)

    return metrics


def image_paths(args):
    return args.image_paths or ([args.image_path] if args.image_path else [])


def build_prompt(model_type, image_count, has_audio, tokenizer=None):
    if model_type in {"gemma4", "gemma4_unified"}:
        if has_audio:
            raise ValueError("Gemma4 audio benchmarking is not supported by this image benchmark")
        content = (
            "<|image|>" * image_count + "Describe these images." if image_count else "What is the meaning of life?"
        )
        if tokenizer is not None:
            try:
                return tokenizer.apply_chat_template(
                    json.dumps([{"role": "user", "content": content}]), add_generation_prompt=True
                )
            except RuntimeError as error:
                if "Invalid or unsupported chat template" not in str(error) and str(error) != "Empty chat template.":
                    raise
        # Older tokenizer runtimes cannot render Gemma4's canonical Jinja template.
        return f"<bos><|turn>user\n{content}<turn|>\n<|turn>model\n"
    image_special = (
        "<|endoftext10|>\n" * image_count
        if model_type == "phi4mm"
        else "".join(f"<|image_{i + 1}|>\n" for i in range(image_count))
    )
    audio_special = "<|endoftext11|>\n" if model_type == "phi4mm" else "<|audio_1|>\n"
    if image_count and has_audio:
        question = "What are some of the similarities and differences between the provided inputs?"
    elif image_count:
        question = "What is shown in this image?"
    elif has_audio:
        question = "What is described in this audio?"
    else:
        question = "What is the meaning of life?"
    return f"<|user|>\n{image_special}{audio_special if has_audio else ''}{question}<|end|>\n<|assistant|>\n"


def run_benchmark(args, model, processor, image, audio, generation_length, max_length):
    # Get user arguments
    num_repetitions = args.repetitions
    temperature = 1.0

    # Get model type
    model_type = None
    if hasattr(model, "type"):
        model_type = model.type
    else:
        with open(os.path.join(args.input_folder, "genai_config.json")) as f:
            genai_config = json.load(f)
            model_type = genai_config["model"]["type"]

    # Process prompt, image, and audio
    if args.verbose:
        print("Processing inputs...")

    paths = image_paths(args)
    tokenizer = og.Tokenizer(model) if model_type in {"gemma4", "gemma4_unified"} else None
    prompt = build_prompt(model_type, len(paths), audio is not None, tokenizer)

    if model_type == "whisper":
        decoder_prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
        prompts = ["".join(decoder_prompt_tokens)]
    elif model_type in {"gemma4", "gemma4_unified"}:
        prompts = prompt
    else:
        prompts = [prompt]
    inputs = processor(prompts, images=image, audios=audio)
    shape = inputs["input_ids"].shape
    prompt_length = (shape() if callable(shape) else shape)[1]
    if generation_length <= 0 or prompt_length + generation_length > max_length:
        raise ValueError("generation_length must be positive and fit within max_length after the prompt")
    with open(os.path.join(args.input_folder, "genai_config.json"), encoding="utf-8") as file:
        manifest = json.load(file)
    contents = []
    for path in [*paths, *([args.audio_path] if args.audio_path else [])]:
        with open(path, "rb") as file:
            contents.append(hashlib.sha256(file.read()).hexdigest())
    signature = hashlib.sha256(
        json.dumps(
            [
                manifest,
                contents,
                args.repetitions,
                args.warmup,
                args.top_k,
                args.top_p,
                args.chunk_size,
                prompt,
                os.environ.get("ORTGENAI_PREPROCESSING_THREADS"),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if args.verbose:
        print(f"Prompt used: {prompt}")

    do_sample = args.top_k > 1 or (args.top_p != 1.0 and args.top_p > 0.0)
    search_options = dict(
        do_sample=do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=temperature,
        max_length=max_length,
        min_length=max_length,
    )
    if args.chunk_size is not None:
        search_options["chunk_size"] = args.chunk_size
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)

    if args.verbose:
        print("Processed inputs, running warmup runs...")
    for _ in tqdm(range(args.warmup)):
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        i = 0
        while not generator.is_done() and i < generation_length:
            generator.generate_next_token()
            generator.get_next_tokens()
            i += 1
        if args.print_model_output:
            print(processor.decode(generator.get_sequence(0)))
        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    process_times = []
    prompt_times = []
    token_gen_times = []
    sampling_times = []
    wall_clock_times = []
    sequences = []
    if args.verbose:
        print(f"Done with warmup, running benchmark for {num_repetitions} repetitions...")
    for _ in tqdm(range(num_repetitions)):
        wall_clock_start_time = time.perf_counter()

        # Measure prompt and image processing
        process_start_time = time.perf_counter()
        inputs = processor(prompts, images=image, audios=audio)
        process_end_time = time.perf_counter()
        process_times.append(process_end_time - process_start_time)

        # Prepare run
        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)

        # Measure prompt processing
        prompt_start_time = time.perf_counter()
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        sampling_start_time = time.perf_counter()
        generator.generate_next_token()
        generator.get_next_tokens()
        sampling_end_time = time.perf_counter()
        sampling_times.append(sampling_end_time - sampling_start_time)
        prompt_times.append(sampling_end_time - prompt_start_time)

        # Measure token generation
        i = 1
        while not generator.is_done() and i < generation_length:
            # Run inference
            token_gen_start_time = time.perf_counter()
            generator.generate_next_token()
            generator.get_next_tokens()
            token_gen_end_time = time.perf_counter()

            token_gen_times.append(token_gen_end_time - token_gen_start_time)
            i += 1
        wall_clock_end_time = time.perf_counter()
        wall_clock_times.append(wall_clock_end_time - wall_clock_start_time)
        sequences.append(generator.get_sequence(0).tolist())
        if args.print_model_output:
            print(processor.decode(generator.get_sequence(0)))

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    # Calculate processing metrics
    avg_processing_latency_s = sum(process_times) / len(process_times)
    avg_processing_latency_ms = avg_processing_latency_s * 1000
    print(f"Average Processing Latency: {avg_processing_latency_ms} ms")

    # Calculate Time to First Token metrics
    avg_prompt_latency_s = sum(prompt_times) / len(prompt_times)
    avg_prompt_latency_ms = avg_prompt_latency_s * 1000
    print(f"Average Time to First Token: {avg_prompt_latency_ms} ms")

    # Calculate token generation input prep metrics
    avg_token_gen_latency_s = sum(token_gen_times) / len(token_gen_times) if token_gen_times else float("nan")
    avg_token_gen_latency_ms = avg_token_gen_latency_s * 1000
    avg_token_gen_thrpt = 1 / avg_token_gen_latency_s
    print(f"Average Token Generation Latency (per token): {avg_token_gen_latency_ms} ms")
    print(f"Average Token Generation Throughput (per token): {avg_token_gen_thrpt} tps")

    # Calculate sampling metrics
    avg_sampling_latency_s = sum(sampling_times) / len(sampling_times)
    avg_sampling_latency_ms = avg_sampling_latency_s * 1000
    avg_sampling_thrpt = 1 / avg_sampling_latency_s
    print(f"Average Sampling Latency (per token): {avg_sampling_latency_ms} ms")
    print(f"Average Sampling Throughput (per token): {avg_sampling_thrpt} tps")

    # Calculate wall clock time
    avg_wall_clock_time = sum(wall_clock_times) / len(wall_clock_times)
    avg_wall_clock_thrpt = generation_length / avg_wall_clock_time
    print(f"Average Wall Clock Time: {avg_wall_clock_time} s")
    print(f"Average Wall Clock Throughput: {avg_wall_clock_thrpt} tps")

    metrics = [
        prompt_length,
        generation_length,
        max_length,
        avg_processing_latency_ms,
        avg_prompt_latency_ms,
        avg_token_gen_thrpt,
        avg_token_gen_latency_ms,
        avg_sampling_thrpt,
        avg_sampling_latency_ms,
        avg_wall_clock_thrpt,
        avg_wall_clock_time,
        len(paths),
        float(np.percentile(prompt_times, 50)) * 1000,
        float(np.percentile(prompt_times, 95)) * 1000,
        float(np.percentile(process_times, 50)) * 1000,
        float(np.percentile(process_times, 95)) * 1000,
        signature,
        hashlib.sha256(json.dumps(sequences, separators=(",", ":")).encode()).hexdigest(),
    ]
    return metrics


def main(args):
    if args.repetitions < 1 or args.warmup < 0:
        raise ValueError("repetitions must be positive and warmup must be non-negative")
    if args.max_lengths and len(args.max_lengths) != len(args.generation_lengths):
        raise ValueError("Supply one max_length per generation_length")
    if args.baseline_csv and (args.top_k != 1 or args.top_p != 1.0):
        raise ValueError("Baseline comparisons require greedy generation: --top_k 1 --top_p 1")
    all_csv_metrics = []

    # Get tokenizer, and model
    model_path = args.input_folder
    if args.verbose:
        print("Loading model... ")
    model = og.Model(f"{model_path}")
    if args.verbose:
        print("Model loaded, loading processor...")
    processor = model.create_multimodal_processor()
    if args.verbose:
        print("Processor loaded, loading image...")

    # Get image
    paths = image_paths(args)
    if paths:
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image file not found: {path}")
        image = og.Images.open(*paths)
    else:
        image = None

    # Get audio
    audio_path = args.audio_path
    if args.audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio = og.Audios.open(audio_path)
    else:
        audio = None

    if args.verbose:
        print("Image loaded, starting benchmark...")
    for g, gen_length in enumerate(args.generation_lengths):
        if args.max_lengths:
            max_length = args.max_lengths[g]
        else:
            max_length = 3072
        print(f"Args: tokens = {gen_length}, max_length = {max_length}")
        metrics = run_benchmark_memory(args, model, processor, image, audio, gen_length, max_length)
        all_csv_metrics.append(metrics)
    # Add metrics to CSV
    if args.verbose:
        print("Adding results to CSV")
    filename = args.output
    save_results(all_csv_metrics, filename, args.baseline_csv)


def str2intlist(value):
    return [int(v) for v in value.split(",")]


def str2strlist(value):
    return [str(v) for v in value.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end benchmarking for ONNX Runtime GenAI")
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="Onnx model folder path (must contain genai_config.json and ONNX models)",
    )
    parser.add_argument("-au", "--audio_path", type=str, default="", required=False, help="Path to the audio")
    images = parser.add_mutually_exclusive_group()
    images.add_argument("-im", "--image_path", type=str, default="", help="Path to one image")
    images.add_argument("--image_paths", nargs="+", default=[], help="Ordered image paths for one multi-image prompt")
    parser.add_argument(
        "-g", "--generation_lengths", type=str2intlist, default=[256], help="Number of tokens to generate after prompt"
    )
    parser.add_argument(
        "-m",
        "--max_lengths",
        type=str2intlist,
        default=[7680],
        help="Max length buffer sizes... User should supply one for every Generation length",
    )
    parser.add_argument("-r", "--repetitions", type=int, default=30, help="Number of times to repeat the benchmark")
    parser.add_argument("-w", "--warmup", type=int, default=5, help="Number of warmup runs before benchmarking")
    parser.add_argument("--chunk_size", type=int, help="Prefill chunk size required by a fixed-shape decoder export")
    parser.add_argument("-k", "--top_k", type=int, default=50, help="Top k tokens to sample from")
    parser.add_argument("-p", "--top_p", type=float, default=1.0, help="Top p probability to sample with")
    parser.add_argument(
        "-o", "--output", type=str, default="genai_e2e.csv", help="Output CSV file name or path (with .csv extension)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print extra information")
    parser.add_argument("-mo", "--print_model_output", action="store_true", help="Print model output")
    parser.add_argument("--baseline_csv", help="Compare greedy outputs and median TTFT against a reference build's CSV")
    args = parser.parse_args()
    main(args)
