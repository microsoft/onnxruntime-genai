# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# This is an end-to-end benchmarking script for any ONNX model.
#
# Prerequisites: 
# 0) Install onnxruntime-genai and onnxruntime
#
# 1) Use builder.py to build the desired ONNX model
#
# 2) Run this script with the desired arguments. Run benchmark_e2e.py -h for help.

import onnxruntime_genai as og
import time
import argparse
from tqdm import tqdm
import subprocess
import threading
import psutil
import os
import json
from metrics import BenchmarkRecord

import numpy as np

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
    global peak_gpu_memory

    while not stop_monitoring:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)

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
    global peak_cpu_memory

    while not stop_monitoring:
        current_used_memory = round(psutil.virtual_memory().used / 1024**3, 2)
        with peak_memory_lock:
            peak_cpu_memory = max(peak_cpu_memory, current_used_memory)
        time.sleep(0.1)

# Use input model to generate prompt
def generate_prompt(model, tokenizer, prompt_length, use_graph_capture) -> str:
    prompt = "a"
    tokens = tokenizer.encode(prompt)
    params=og.GeneratorParams(model)
    params.set_search_options(max_length=prompt_length, min_length=prompt_length)
    params.input_ids = tokens

    if use_graph_capture:
        params.try_graph_capture_with_max_batch_size(1)

    generator=og.Generator(model, params)
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
    return tokenizer.decode(generator.get_sequence(0))

def get_target_pip_package_version(target_pip_package_name_list):
    # get package name and version
    import pkg_resources

    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(
        [
            f"{i.key}=={i.version}"
            for i in installed_packages
            if i.key in target_pip_package_name_list
        ]
    )

    pkg_name = ""
    pkg_version = ""
    if installed_packages_list:
        pkg_name = installed_packages_list[0].split("==")[0]
        pkg_version = installed_packages_list[0].split("==")[1]
    return pkg_name, pkg_version

def get_model_info_from_genai_config(model_input_folder):
    genai_config_file_path = os.path.join(model_input_folder, "genai_config.json")
    genai_config_file = open(genai_config_file_path)
    genai_config = json.load(genai_config_file)
    model_info = {}  
    model_info["execution_provider"] = "cpu"
    provider_options = genai_config["model"]["decoder"]["session_options"]["provider_options"]
    if len(provider_options) > 0 and len(provider_options[0].keys()) > 0:
        model_info["execution_provider"] = list(genai_config["model"]["decoder"]["session_options"]["provider_options"][0].keys())[0]
    genai_config_file.close()
    return model_info

def save_results(args, results, filename, print_memory_usage=False):
    import pandas as pd

    columns=[
    "Batch Size",
    "Prompt Length",
    "Tokens Generated",
    "Max Length",
    "Tokenization Throughput (tps)",
    "Tokenization Latency (ms)",
    "Prompt Processing Throughput (tps)",
    "Prompt Processing Latency (ms)",
    "Token Generation Throughput (tps)",
    "Token Generation Latency (ms)",
    "Sampling Throughput (tps)",
    "Sampling Latency (ms)",
    "Wall Clock Throughput (tps)",
    "Wall Clock Time (s)",
    ]

    if print_memory_usage:
        if IS_NVIDIA_SYSTEM:
            columns.append("peak_gpu_memory (GiB)")
        else:
            columns.append("peak_cpu_memory (GiB)")

    df = pd.DataFrame(
        results,
        columns=columns,
    )
    # df = df.transpose()  # This line swaps the rows and columns
    
    genai_package_name, genai_package_version = get_target_pip_package_version(["onnxruntime-genai", "onnxruntime-genai-cuda", "onnxruntime-genai-directml"])
    model_info = get_model_info_from_genai_config(args.input_folder)
    
    records = []
    for _, row in df.iterrows():
        record = BenchmarkRecord(args.model_name, args.precision, "onnxruntime-genai", model_info["execution_provider"], genai_package_name, genai_package_version )
        record.config.batch_size = row["Batch Size"]
        record.config.customized["prompt_length"] = row["Prompt Length"]
        record.config.customized["tokens_generated"] = row["Tokens Generated"]
        record.config.customized["max_length"] = row["Max Length"]
        record.metrics.customized["tokenization_throughput_tps"] = row["Tokenization Throughput (tps)"]
        record.metrics.customized["tokenization_latency_ms"] = row["Tokenization Latency (ms)"]
        record.metrics.customized["prompt_processing_throughput_tps"] = row["Prompt Processing Throughput (tps)"]
        record.metrics.customized["prompt_processing_latency_ms"] = row["Prompt Processing Latency (ms)"]
        record.metrics.customized["token_generation_throughput_tps"] = row["Token Generation Throughput (tps)"]
        record.metrics.customized["token_generation_latency_ms"] = row["Token Generation Latency (ms)"]
        record.metrics.customized["sampling_throughput_tps"] = row["Sampling Throughput (tps)"]
        record.metrics.customized["sampling_latency_ms"] = row["Sampling Latency (ms)"]   
        record.metrics.customized["wall_clock_throughput_tps"] = row["Wall Clock Throughput (tps)"]
        record.metrics.customized["wall_clock_time_s"] = row["Wall Clock Time (s)"]

        if print_memory_usage:
            if IS_NVIDIA_SYSTEM:
                record.metrics.customized["peak_gpu_memory_gb"] = row["peak_gpu_memory (GiB)"]
            else:
                record.metrics.customized["peak_cpu_memory_gb"] = row["peak_cpu_memory (GiB)"]
        
        records.append(record)
        
    # df.to_csv(filename, header=True, index=False)
    BenchmarkRecord.save_as_csv(filename, records)
    BenchmarkRecord.save_as_json(filename.replace(".csv", ".json"), records)
    print(f"Results saved in {filename}!")

def run_benchmark_memory(args, batch_size, prompt_length, generation_length, max_length):
    """
    This function is to run benchmark and print the momory usage
    """
    global stop_monitoring
    global peak_gpu_memory
    global peak_cpu_memory

    # Reset the peak memory variables and the monitoring flag
    stop_monitoring = False
    peak_gpu_memory = 0.0
    peak_cpu_memory = 0.0

    if IS_NVIDIA_SYSTEM:
        monitor_thread = threading.Thread(target=monitor_gpu_memory)
    else:
        monitor_thread = threading.Thread(target=monitor_cpu_memory)
    
    monitor_thread.start()

    metrics = run_benchmark(args, batch_size, prompt_length, generation_length, max_length)

    stop_monitoring = True
    monitor_thread.join()

    if IS_NVIDIA_SYSTEM:
        metrics.append(peak_gpu_memory)
    else:
        metrics.append(peak_cpu_memory)
    
    return metrics

def run_benchmark(args, batch_size, prompt_length, generation_length, max_length):

    # Get user arguments
    num_repetitions = args.repetitions
    temperature = 1.0

    # Get tokenizer, and model
    if args.verbose: print("Loading model... ")
    model=og.Model(f'{args.input_folder}')
    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)

 
    # Generate prompt
    tokens, prompt = None, None
    if args.use_random_tokens:
        # use random tokens instead of generating a prompt using the model and then tokenizing it
        tokens = np.random.randint(100, size=(batch_size, prompt_length))
        prompt = [tokenizer.decode(tokens[0])] * batch_size
    else:
        prompt = [generate_prompt(model, tokenizer, prompt_length, args.use_graph_capture)] * batch_size
        tokens = tokenizer.encode_batch(prompt)

    params = og.GeneratorParams(model)
    params.input_ids = tokens
    do_sample = args.top_k > 1 or (args.top_p != 1.0 and args.top_p > 0.0)
    params.set_search_options(do_sample=do_sample, top_k=args.top_k, top_p=args.top_p, temperature=temperature, max_length=max_length, min_length=max_length)

    if args.use_graph_capture:
        params.try_graph_capture_with_max_batch_size(batch_size)

    if args.verbose: print("Running warmup runs...")
    for _ in tqdm(range(args.warmup)):
        generator = og.Generator(model, params)
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
        if args.print_model_output: print(tokenizer.decode(generator.get_sequence(0)))
        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    tokenize_times = []
    prompt_times = []
    token_gen_times = []
    sampling_times = []
    wall_clock_times = []
    if args.verbose: print(f"Running benchmark for batch size = {batch_size}, prompt length = {prompt_length}")
    for _ in tqdm(range(num_repetitions)):
        wall_clock_start_time = time.time()

        # Measure tokenization
        tokenize_start_time = time.perf_counter()
        tokens = tokenizer.encode_batch(prompt)
        tokenize_end_time = time.perf_counter()
        tokenize_times.append(tokenize_end_time - tokenize_start_time)

        # Prepare run
        params = og.GeneratorParams(model)
        params.input_ids = tokens
        params.set_search_options(do_sample=do_sample, top_k=args.top_k, top_p=args.top_p, temperature=temperature, max_length=max_length, min_length=max_length)

        if args.use_graph_capture:
            params.try_graph_capture_with_max_batch_size(batch_size)

        generator = og.Generator(model, params)

        # Measure prompt processing
        prompt_start_time = time.perf_counter()
        generator.compute_logits()
        prompt_end_time = time.perf_counter()
        prompt_times.append(prompt_end_time - prompt_start_time)

        sampling_start_time = time.perf_counter()
        generator.generate_next_token()
        sampling_end_time = time.perf_counter()
        sampling_times.append(sampling_end_time - sampling_start_time)

        # Measure token generation
        i = 1
        while not generator.is_done() and i < generation_length:
            # Run inference
            token_gen_start_time = time.perf_counter()
            generator.compute_logits()
            token_gen_end_time = time.perf_counter()

            sampling_start_time = time.perf_counter()
            generator.generate_next_token()
            sampling_end_time = time.perf_counter()
            
            token_gen_times.append(token_gen_end_time - token_gen_start_time)
            sampling_times.append(sampling_end_time - sampling_start_time)
            i += 1
        wall_clock_end_time = time.time()
        wall_clock_times.append(wall_clock_end_time - wall_clock_start_time)
        if args.print_model_output: print(tokenizer.decode(generator.get_sequence(0)))

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    # Calculate tokenization metrics
    avg_tokenization_latency_s = sum(tokenize_times) / len(tokenize_times)
    avg_tokenization_latency_ms = avg_tokenization_latency_s * 1000
    avg_per_token_tokenization_latency_ms = avg_tokenization_latency_ms / prompt_length
    avg_tokenization_thrpt = batch_size * (1000 / avg_per_token_tokenization_latency_ms)
    print(f"Average Tokenization Latency (per token): {avg_per_token_tokenization_latency_ms} ms")
    print(f"Average Tokenization Throughput (per token): {avg_tokenization_thrpt} tps")

    # Calculate prompt processing metrics
    avg_prompt_latency_s = sum(prompt_times) / len(prompt_times)
    avg_prompt_latency_ms = avg_prompt_latency_s * 1000
    avg_per_token_prompt_latency_ms = avg_prompt_latency_ms / prompt_length
    avg_per_token_prompt_thrpt = batch_size * (1000 / avg_per_token_prompt_latency_ms)
    print(f"Average Prompt Processing Latency (per token): {avg_per_token_prompt_latency_ms} ms")
    print(f"Average Prompt Processing Throughput (per token): {avg_per_token_prompt_thrpt} tps")

    # Calculate token generation input prep metrics
    avg_token_gen_latency_s = sum(token_gen_times) / len(token_gen_times)
    avg_token_gen_latency_ms = avg_token_gen_latency_s * 1000
    avg_token_gen_thrpt = batch_size * (1 / avg_token_gen_latency_s)
    print(f"Average Token Generation Latency (per token): {avg_token_gen_latency_ms} ms")
    print(f"Average Token Generation Throughput (per token): {avg_token_gen_thrpt} tps")
    
    # Calculate sampling metrics
    avg_sampling_latency_s = sum(sampling_times) / len(sampling_times)
    avg_sampling_latency_ms = avg_sampling_latency_s * 1000
    avg_sampling_thrpt = batch_size * (1 / avg_sampling_latency_s)
    print(f"Average Sampling Latency (per token): {avg_sampling_latency_ms} ms")
    print(f"Average Sampling Throughput (per token): {avg_sampling_thrpt} tps")

    # Calculate wall clock time
    avg_wall_clock_time = sum(wall_clock_times) / len(wall_clock_times)
    avg_wall_clock_thrpt = batch_size * (max_length / avg_wall_clock_time)
    print(f"Average Wall Clock Time: {avg_wall_clock_time} s")
    print(f"Average Wall Clock Throughput: {avg_wall_clock_thrpt} tps")

    if args.print_memory_usage:
        if IS_NVIDIA_SYSTEM:
            print(f"Peak GPU Memory Usage: {peak_gpu_memory} GiB ")
        else:
            print(f"Peak CPU Memory Usage: {peak_cpu_memory} GiB ")

    metrics = [
        batch_size, 
        prompt_length,
        generation_length,
        max_length,
        avg_tokenization_thrpt, 
        avg_tokenization_latency_ms, 
        avg_per_token_prompt_thrpt, 
        avg_per_token_prompt_latency_ms, 
        avg_token_gen_thrpt, 
        avg_token_gen_latency_ms, 
        avg_sampling_thrpt, 
        avg_sampling_latency_ms,
        avg_wall_clock_thrpt,
        avg_wall_clock_time,
    ]
    return metrics


def main(args):
    all_csv_metrics = []

    for batch_size in args.batch_sizes:
        for l, prompt_length in enumerate(args.prompt_lengths):
            for g, gen_length in enumerate(args.generation_lengths):
                if args.max_lengths:
                    m = l * len(args.generation_lengths) + g
                    max_length = args.max_lengths[0] if len(args.max_lengths) == 1 else args.max_lengths[m]
                else:
                    max_length = prompt_length + gen_length
                print(f"\nArgs: batch_size = {batch_size}, prompt_length = {prompt_length}, tokens = {gen_length}, max_length = {max_length}")
                if args.print_memory_usage:
                    metrics = run_benchmark_memory(args, batch_size, prompt_length, gen_length, max_length)
                else:
                    metrics = run_benchmark(args, batch_size, prompt_length, gen_length, max_length)
                all_csv_metrics.append(metrics)
    # Add metrics to CSV
    if args.verbose: print("Adding results to CSV")
    filename = args.output

    if args.print_memory_usage:
        save_results(args, all_csv_metrics, filename, print_memory_usage=True)
    else:
        save_results(args, all_csv_metrics, filename)

def str2intlist(value):
    return [int(v) for v in value.split(',')]

def str2strlist(value):
    return [str(v) for v in value.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end benchmarking for gen-ai")
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Onnx model folder path (must contain genai_config.json and model.onnx)')
    parser.add_argument('-b', '--batch_sizes', type=str2intlist, default=[1], help='Number of sequences to generate in parallel')
    parser.add_argument('-l', '--prompt_lengths', type=str2intlist, default=[16], help='Number of tokens for prompt')
    parser.add_argument('-g', '--generation_lengths', type=str2intlist, default=[256], help='Number of tokens to generate after prompt')
    parser.add_argument('-m', '--max_lengths', type=str2intlist, default=[], help='Max length is either a combination of prompt and generation length or one value broadcasting for all.')
    parser.add_argument('-r', '--repetitions', type=int, default=10, help='Number of times to repeat the benchmark')
    parser.add_argument('-w', '--warmup', type=int, default=5, help='Number of warmup runs before benchmarking')
    parser.add_argument('-k', '--top_k', type=int, default=50, help='Top k tokens to sample from')
    parser.add_argument('-p', '--top_p', type=float, default=1.0, help='Top p probability to sample with')
    parser.add_argument('-o', '--output', type=str, default='genai_e2e', help='Output CSV file name or path (with .csv extension)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print extra information')
    parser.add_argument('-mo', '--print_model_output', action='store_true', help='Print model output')
    parser.add_argument('-pm', '--print_memory_usage', default=False, help='Print memory footprint')
    parser.add_argument('-gc', '--use_graph_capture', action='store_true', help='Use the graph capture feature for CUDA or DML')
    parser.add_argument('-mn', '--model_name', type=str, default='model_name', help='Model name defined by users')
    parser.add_argument('-pr', '--precision', type=str, default='fp16', help='Model precision for metrics info')
    parser.add_argument('--use_random_tokens', action='store_true', help='Use random tokens instead of generating a prompt')
    args = parser.parse_args()

    # check max_lengths
    is_max_lengths_valid = not args.max_lengths or len(args.max_lengths) == 1 or len(args.max_lengths) == len(args.prompt_lengths) * len(args.generation_lengths)
    assert is_max_lengths_valid, "len(args.max_lengths) is either a combination of args.prompt_lengths and args.generation_lengths or 1 that broadcasts for all"
    main(args)