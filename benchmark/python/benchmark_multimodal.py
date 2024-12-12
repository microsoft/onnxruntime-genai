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
import os
import time
import argparse
from tqdm import tqdm


def save_results(results, filename):
    import pandas as pd
    df = pd.DataFrame(
        results,
        columns=[
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
        ],
    )
    # df = df.transpose()  # This line swaps the rows and columns
    df.to_csv(filename, header=True, index=False)
    print(f"Results saved in {filename}!")

def run_benchmark(args, model, processor, image, generation_length, max_length):
    # Get user arguments
    num_repetitions = args.repetitions
    temperature = 1.0

    # Process prompt and image
    if args.verbose: print("Processing image and prompt...")
    prompt = "<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
    inputs = processor(prompt, images=image)

    params = og.GeneratorParams(model)
    params.set_inputs(inputs)
    do_sample = args.top_k > 1 or (args.top_p != 1.0 and args.top_p > 0.0)
    params.set_search_options(do_sample=do_sample, top_k=args.top_k, top_p=args.top_p, temperature=temperature, max_length=max_length, min_length=max_length)

    if args.use_graph_capture:
        params.try_graph_capture_with_max_batch_size(1)

    if args.verbose: print("Processed image and prompt, running warmup runs...")
    for _ in tqdm(range(args.warmup)):
        generator = og.Generator(model, params)
        while not generator.is_done():
            generator.generate_next_token()
        if args.print_model_output: print(processor.decode(generator.get_sequence(0)))
        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    process_times = []
    prompt_times = []
    token_gen_times = []
    sampling_times = []
    wall_clock_times = []
    if args.verbose: print(f"Done with warmup, running benchmark for {num_repetitions} repetitions...")
    for _ in tqdm(range(num_repetitions)):
        wall_clock_start_time = time.time()

        # Measure prompt and image processing
        process_start_time = time.perf_counter()
        inputs = processor(prompt, images=image)
        process_end_time = time.perf_counter()
        process_times.append(process_end_time - process_start_time)

        # Prepare run
        params = og.GeneratorParams(model)
        params.set_inputs(inputs)
        params.set_search_options(do_sample=do_sample, top_k=args.top_k, top_p=args.top_p, temperature=temperature, max_length=max_length, min_length=max_length)

        if args.use_graph_capture:
            params.try_graph_capture_with_max_batch_size(1)


        # Measure prompt processing
        prompt_start_time = time.perf_counter()
        generator = og.Generator(model, params)
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
            generator.generate_next_token()
            token_gen_end_time = time.perf_counter()
            
            token_gen_times.append(token_gen_end_time - token_gen_start_time)
            i += 1
        wall_clock_end_time = time.time()
        wall_clock_times.append(wall_clock_end_time - wall_clock_start_time)
        if args.print_model_output: print(processor.decode(generator.get_sequence(0)))

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
    avg_token_gen_latency_s = sum(token_gen_times) / len(token_gen_times)
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
    avg_wall_clock_thrpt = max_length / avg_wall_clock_time
    print(f"Average Wall Clock Time: {avg_wall_clock_time} s")
    print(f"Average Wall Clock Throughput: {avg_wall_clock_thrpt} tps")

    metrics = [
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
    ]
    return metrics

def main(args):
    all_csv_metrics = []
    # Get tokenizer, and model
    model_path = args.input_folder
    if args.verbose: print(f"Loading model... ")
    model=og.Model(f'{model_path}')
    if args.verbose: print("Model loaded, loading processor...")
    processor = model.create_multimodal_processor()
    if args.verbose: print("Processor loaded, loading image...")
    # Get image
    image_path = args.image_path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = og.Images.open(image_path)
    if args.verbose: print("Image loaded, starting benchmark...")
    for g, gen_length in enumerate(args.generation_lengths):
        if args.max_lengths:
            max_length = args.max_lengths[g]
        else:
            max_length = 3072
        print(f"Args: tokens = {gen_length}, max_length = {max_length}")
        metrics = run_benchmark(args, model, processor, image, gen_length, max_length)
        all_csv_metrics.append(metrics)
    # Add metrics to CSV
    if args.verbose: print("Adding results to CSV")
    filename = args.output
    save_results(all_csv_metrics, filename)

def str2intlist(value):
    return [int(v) for v in value.split(',')]

def str2strlist(value):
    return [str(v) for v in value.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end benchmarking for gen-ai")
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Onnx model folder path (must contain genai_config.json and model.onnx)')
    parser.add_argument('-im', '--image_path', type=str, required=True, help='Path to the image')
    parser.add_argument('-g', '--generation_lengths', type=str2intlist, default=[256], help='Number of tokens to generate after prompt')
    parser.add_argument('-m', '--max_lengths', type=str2intlist, default=[3072], help='Max length buffer sizes... User should supply one for every Generation length')
    parser.add_argument('-r', '--repetitions', type=int, default=10, help='Number of times to repeat the benchmark')
    parser.add_argument('-w', '--warmup', type=int, default=5, help='Number of warmup runs before benchmarking')
    parser.add_argument('-k', '--top_k', type=int, default=50, help='Top k tokens to sample from')
    parser.add_argument('-p', '--top_p', type=float, default=1.0, help='Top p probability to sample with')
    parser.add_argument('-o', '--output', type=str, default='genai_e2e', help='Output CSV file name or path (with .csv extension)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print extra information')
    parser.add_argument('-mo', '--print_model_output', action='store_true', help='Print model output')
    parser.add_argument('-gc', '--use_graph_capture', action='store_true', help='Use the graph capture feature for CUDA or DML')
    args = parser.parse_args()
    main(args)
