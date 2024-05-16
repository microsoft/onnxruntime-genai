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

# Use input model to generate prompt
def generate_prompt(model, tokenizer, prompt_length, use_graph_capture) -> str:
    temperature = 1.0
    prompt = "a"
    tokens = tokenizer.encode(prompt)
    params=og.GeneratorParams(model)
    params.set_search_options(do_sample=True, top_k=5, temperature=temperature, max_length=prompt_length, min_length=prompt_length+1)
    params.input_ids = tokens

    if use_graph_capture:
        params.try_use_cuda_graph_with_max_batch_size(1)

    generator=og.Generator(model, params)
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
    return tokenizer.decode(generator.get_sequence(0))

def save_results(results, filename):
    import pandas as pd
    df = pd.DataFrame(
        results,
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
        ],
    )
    # df = df.transpose()  # This line swaps the rows and columns
    df.to_csv(filename, header=True, index=False)
    print(f"Results saved in {filename}!")

def run_benchmark(args, model, tokenizer, batch_size, prompt_length, generation_length, max_length):
    # Get user arguments
    num_repetitions = args.repetitions
    temperature = 1.0

    # Generate prompt
    prompt = [generate_prompt(model, tokenizer, prompt_length, args.use_graph_capture)] * batch_size
    tokens = tokenizer.encode_batch(prompt)

    params = og.GeneratorParams(model)
    params.input_ids = tokens
    params.set_search_options(do_sample=True, top_k=args.top_k, top_p=args.top_p, temperature=temperature, max_length=max_length, min_length=max_length)

    if args.use_graph_capture:
        params.try_use_cuda_graph_with_max_batch_size(batch_size)

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

        # Prepare run
        generator = og.Generator(model, params)

        # Measure tokenization
        tokenize_start_time = time.perf_counter()
        tokens = tokenizer.encode_batch(prompt)
        tokenize_end_time = time.perf_counter()
        tokenize_times.append(tokenize_end_time - tokenize_start_time)

        # Prepare run
        params = og.GeneratorParams(model)
        params.input_ids = tokens
        params.set_search_options(max_length=max_length, min_length=max_length)

        if args.use_graph_capture:
            params.try_use_cuda_graph_with_max_batch_size(batch_size)

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
    avg_tokenization_thrpt = batch_size * (1 / avg_tokenization_latency_s)
    print(f"Average Tokenization Latency (per token): {avg_tokenization_latency_ms} ms")
    print(f"Average Tokenization Throughput (per token): {avg_tokenization_thrpt} tps")

    # Calculate prompt processing metrics
    avg_prompt_latency_s = sum(prompt_times) / len(prompt_times)
    avg_prompt_latency_ms = avg_prompt_latency_s * 1000
    avg_per_token_prompt_latency_ms = avg_prompt_latency_ms / prompt_length
    avg_per_token_prompt_thrpt = batch_size * (1 / (avg_per_token_prompt_latency_ms / 1000))
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
    # Get tokenizer, and model
    model_path = args.input_folder
    if args.verbose: print(f"Loading model... ")
    model=og.Model(f'{model_path}')
    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    if args.verbose: print("Benchmarking " + model_path)
    for batch_size in args.batch_sizes:
        for l, prompt_length in enumerate(args.prompt_lengths):
            for g, gen_length in enumerate(args.generation_lengths):
                if args.max_lengths:
                    m = l * len(args.generation_lengths) + g
                    max_length = args.max_lengths[m]
                else:
                    max_length = prompt_length + gen_length
                print(f"Args: batch_size = {batch_size}, prompt_length = {prompt_length}, tokens = {gen_length}, max_length = {max_length}")
                metrics = run_benchmark(args, model, tokenizer, batch_size, prompt_length, gen_length, max_length)
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
    parser.add_argument('-b', '--batch_sizes', type=str2intlist, default=[1], help='Number of sequences to generate in parallel')
    parser.add_argument('-l', '--prompt_lengths', type=str2intlist, default=[16], help='Number of tokens for prompt')
    parser.add_argument('-g', '--generation_lengths', type=str2intlist, default=[256], help='Number of tokens to generate after prompt')
    parser.add_argument('-m', '--max_lengths', type=str2intlist, default=[], help='Max length buffer sizes... User should supply one for every combination of Prompt and Generation length')
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
