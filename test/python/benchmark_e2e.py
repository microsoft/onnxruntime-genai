# This is an end-to-end benchmarking script for the Phi-2 model.
#
# TODO: what is steps to run
# Prerequisites: 
# 1) Install `huggingface-cli`:
#
# $ pip install huggingface_hub
#
# 2) Install `ONNX Runtime v1.16.2 or higher`
#
# Main: install ONNX Runtime stable build
# $ pip install onnxruntime-gpu
#
# Alternative: install ONNX Runtime nightly build
# $ pip install ort-nightly-gpu
#
# Alternative: build from source (instructions available at https://onnxruntime.ai/docs/build/inferencing.html)

from typing import List
import datetime
import itertools
import json
# import numpy as np
import onnxruntime_genai as og
# import torch
import time
import argparse

# Use input model to generate prompt
def generate_prompt(model, tokenizer, prompt_length) -> str:
    prompt = "a"
    tokens = tokenizer.encode(prompt)
    params=og.GeneratorParams(model)
    params.max_length = prompt_length
    params.input_ids = tokens
    generator=og.Generator(model, params)
    # TODO: handle eos token case
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token_top_k(50, 1.0)
    return tokenizer.decode(generator.get_sequence(0).get_array())

def save_results(results, filename):
    import pandas as pd
    df = pd.DataFrame(
        results,
        columns=[
            "Batch Size",
            "Prompt Length",
            "Tokenization Throughput (tps)",
            "Tokenization Latency (ms)",
            "Prompt Processing Throughput (tps)",
            "Prompt Processing Latency (ms)",
            "Token Generation Throughput (tps)",
            "Token Generation Latency (ms)",
            "Sampling Throughput (tps)",
            "Sampling Latency (ms)",
        ],
    )
    df = df.transpose()  # This line swaps the rows and columns
    df.to_csv(filename, header=False)
    print(f"Results saved in {filename}!")

def main(args):
    # Get user arguments
    num_repititions = args.repetitions
    generation_length = args.generation_length
    batch_size, prompt_length = args.batch_size, args.prompt_length

    # Get tokenizer, and model
    if args.verbose: print(f"Loading model... ")
    model=og.Model(f'{args.input_folder}', og.DeviceType.CPU if args.device == 'cpu' else og.DeviceType.CUDA)
    if args.verbose: print("Model loaded")
    tokenizer = model.create_tokenizer()

    # Generate prompt
    prompt = [generate_prompt(model, tokenizer, prompt_length)] * batch_size
    if args.verbose: print("Running warmup runs...")
    for i in range(args.warmup):
        if args.verbose: print(f"Running warmup repetition {i+1}...")
        tokens = tokenizer.encode_batch(prompt)
        params = og.GeneratorParams(model)
        params.max_length = prompt_length + generation_length
        params.input_ids = tokens
        generator = og.Generator(model, params)
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token_top_k_top_p(args.top_k, args.top_p, 1.0)
        if args.print_model_output: print(tokenizer.decode(generator.get_sequence(0).get_array()))

    tokenize_times = []
    prompt_times = []
    token_gen_times = []
    sampling_times = []
    if args.verbose: print(f"Running benchmark for batch size = {batch_size}, prompt length = {prompt_length}")
    for i in range(num_repititions):
        if args.verbose: print(f"Running repetition {i+1}...")
        # Prepare run
        max_length = prompt_length + generation_length
        params = og.GeneratorParams(model)
        params.max_length = max_length
        params.input_ids = tokens
        generator = og.Generator(model, params)

        # Measure tokenization
        tokenize_start_time = time.time()
        tokens = tokenizer.encode_batch(prompt)
        tokenize_end_time = time.time()
        tokenize_times.append(tokenize_end_time - tokenize_start_time)

        # Measure prompt processing
        prompt_start_time = time.time()
        generator.compute_logits()
        prompt_end_time = time.time()
        prompt_times.append(prompt_end_time - prompt_start_time)

        sampling_start_time = time.time()
        generator.generate_next_token_top_k_top_p(args.top_k, args.top_p, 1.0)
        sampling_end_time = time.time()
        sampling_times.append(sampling_end_time - sampling_start_time)

        # Measure token generation
        while not generator.is_done():
            # Run inference
            token_gen_start_time = time.time()
            generator.compute_logits()
            token_gen_end_time = time.time()

            sampling_start_time = time.time()
            generator.generate_next_token_top_k_top_p(args.top_k, args.top_p, 1.0)
            sampling_end_time = time.time()
            
            token_gen_times.append(token_gen_end_time - token_gen_start_time)
            sampling_times.append(sampling_end_time - sampling_start_time)
            # TODO: might want or have to add check eos token here...
        if args.print_model_output: print(tokenizer.decode(generator.get_sequence(0).get_array()))

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

    all_csv_metrics = [[
        batch_size, 
        prompt_length, 
        avg_tokenization_thrpt, 
        avg_tokenization_latency_ms, 
        avg_per_token_prompt_thrpt, 
        avg_per_token_prompt_latency_ms, 
        avg_token_gen_thrpt, 
        avg_token_gen_latency_ms, 
        avg_sampling_thrpt, 
        avg_sampling_latency_ms,
    ]]

    # Add metrics to CSV
    if args.verbose: print("Adding results to CSV")
    filename = args.output + ".csv"
    save_results(all_csv_metrics, filename)

if __name__ == "__main__":
    # TODO: add top_k and top_p as arguments
    parser = argparse.ArgumentParser(description="End-to-end benchmarking for gen-ai")
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Number of sequences to generate in parallel')
    parser.add_argument('-l', '--prompt_length', type=int, default=16, help='Number of tokens for prompt')
    parser.add_argument('-g', '--generation_length', type=int, default=256, help='Number of tokens to generate after prompt')
    parser.add_argument('-r', '--repetitions', type=int, default=10, help='Number of times to repeat the benchmark')
    parser.add_argument('-w', '--warmup', type=int, default=5, help='Number of warmup runs before benchmarking')
    parser.add_argument('-k', '--top_k', type=int, default=50, help='Top k tokens to sample from')
    parser.add_argument('-p', '--top_p', type=float, default=1.0, help='Top p probability to sample with')
    parser.add_argument('-o', '--output', type=str, default='genai_e2e', help='Output CSV file name or path')
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Device to use, default is CPU, use CUDA for GPU')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print extra information')
    parser.add_argument('-mo', '--print_model_output', action='store_true', help='Print model output')
    args = parser.parse_args()
    main(args)
