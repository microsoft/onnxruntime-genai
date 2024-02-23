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

def save_results(results, filename):
    import pandas as pd
    df = pd.DataFrame(
        results,
        columns=[
            "Batch Size",
            "Prompt Length",
            "Sampling Latency (ms)",
            "Sampling Throughput (tps)",
            "Token Generation Input Prep Latency (ms)",
            "Token Generation Input Prep Throughput (tps)",
            "First Token Generated Latency (ms)",
            "First Token Generated Throughput (tps)",
            "First 128 Tokens Generated Avg Latency (ms)",
            "First 128 Tokens Generated Avg Throughput (tps)",
            "First 256 Tokens Generated Avg Latency (ms)",
            "First 256 Tokens Generated Avg Throughput (tps)",
            "Wall-Clock Latency (s)",
            "Wall-Clock Throughput (tps)",
        ],
    )
    df = df.transpose()  # This line swaps the rows and columns
    df.to_csv(filename, header=False)
    print(f"Results saved in {filename}!")

def main(args):
    with open("size_to_prompt.json", "r") as f:
        size_to_prompt = json.load(f)

    # Get information based on user settings
    # num_prompt_runs = 50
    generation_length = 256
    
    # batch_sizes, prompt_lengths = [1, 4, 16], [16, 64, 256, 1024, 2048, 3840]
    batch_sizes, prompt_lengths = [1, 4, 16], [16, 64, 256, 1024, 2048]

    # Get tokenizer, and model
    print(f"Loading model... ")
    model=og.Model(f'{args.input_folder}', og.DeviceType.CPU if args.device == 'cpu' else og.DeviceType.CUDA)
    print("Model loaded")
    tokenizer = model.create_tokenizer()

    all_csv_metrics = []
    for (batch_size, prompt_length) in itertools.product(batch_sizes, prompt_lengths):
        print(f"Running batch size = {batch_size}, prompt length = {prompt_length}")
        max_length = prompt_length + generation_length
        prompt = [size_to_prompt[str(prompt_length)]] * batch_size
        csv_metrics = [batch_size, prompt_length]

        # Measure prompt processing
        # print("Measuring prompt processing...")
        try:
            # Measure token generation
            print("Measuring token generation...")
            input_preparation_start_time = time.time()
            tokens = tokenizer.encode_batch(prompt)
            params=og.GeneratorParams(model)
            params.max_length = max_length
            params.input_ids = tokens
            generator=og.Generator(model, params)
            input_preparation_end_time = time.time()
            input_preparation_times = [input_preparation_end_time - input_preparation_start_time]

            accelerator_times = []  # 0th entry will have prompt accelerator time, 1st entry onwards will have token generation accelerator time
            sampling_times = []  # cost to sample after each model run
            wall_clock_start_time = time.time()
            while not generator.is_done():
                # Run inference
                accelerator_start_time = time.time()
                generator.compute_logits()
                accelerator_end_time = time.time()

                sampling_start_time = time.time()
                generator.generate_next_token_top_k(50, 1.0)
                sampling_end_time = time.time()
                
                accelerator_times.append(accelerator_end_time - accelerator_start_time)
                sampling_times.append(sampling_end_time - sampling_start_time)


            wall_clock_end_time = time.time()
            wall_clock_latency = wall_clock_end_time - wall_clock_start_time

            # Calculate sampling metrics
            avg_sampling_latency_s = sum(sampling_times) / len(sampling_times)
            avg_sampling_latency_ms = avg_sampling_latency_s * 1000
            avg_sampling_thrpt = batch_size * (1 / avg_sampling_latency_s)
            print(f"Average Sampling Latency: {avg_sampling_latency_s * 1000} ms")
            print(f"Average Sampling Throughput: {batch_size * (1 / avg_sampling_latency_s)} tps")

            # Calculate token generation input prep metrics
            print(input_preparation_times)
            avg_token_input_prep_latency_s = sum(input_preparation_times) / len(input_preparation_times)
            avg_token_input_prep_latency_ms = avg_token_input_prep_latency_s * 1000
            avg_token_input_prep_thrpt = batch_size * (1 / avg_token_input_prep_latency_s)
            print(f"Average Token Generation Input Preparation Latency: {avg_token_input_prep_latency_s * 1000} ms")
            print(f"Average Token Generation Input Preparation Throughput: {batch_size * (1 / avg_token_input_prep_latency_s)} tps")

            # Calculate first token generated metrics
            avg_accelerator_token_latency_s = accelerator_times[1] / 1
            avg_accelerator_token_latency_ms = avg_accelerator_token_latency_s * 1000
            avg_accelerator_token_thrpt = batch_size * (1 / avg_accelerator_token_latency_s)
            print(f"First Token Average Accelerator Token Generation Latency: {avg_accelerator_token_latency_s * 1000} ms")
            print(f"First Token Average Accelerator Token Generation Throughput: {batch_size * (1 / avg_accelerator_token_latency_s)} tps")

            csv_metrics.extend([avg_sampling_latency_ms, avg_sampling_thrpt, avg_token_input_prep_latency_ms, avg_token_input_prep_thrpt, avg_accelerator_token_latency_ms, avg_accelerator_token_thrpt])

            halfway_idx = 1 + (generation_length // 2)  # +1 is for prompt entry

            # Calculating average of first 128 tokens generated metrics
            avg_accelerator_token_latency_s = sum(accelerator_times[1 : halfway_idx]) / len(accelerator_times[1 : halfway_idx])
            avg_accelerator_token_latency_ms = avg_accelerator_token_latency_s * 1000
            avg_accelerator_token_thrpt = batch_size * (1 / avg_accelerator_token_latency_s)
            print(f"First 128 Tokens Average Accelerator Token Generation Latency: {avg_accelerator_token_latency_s * 1000} ms")
            print(f"First 128 Tokens Average Accelerator Token Generation Throughput: {batch_size * (1 / avg_accelerator_token_latency_s)} tps")

            csv_metrics.extend([avg_accelerator_token_latency_ms, avg_accelerator_token_thrpt])

            avg_accelerator_token_latency_s = sum(accelerator_times[1:]) / len(accelerator_times[1:])
            avg_accelerator_token_latency_ms = avg_accelerator_token_latency_s * 1000
            avg_accelerator_token_thrpt = batch_size * (1 / avg_accelerator_token_latency_s)
            print(f"First 256 Tokens Average Accelerator Token Generation Latency: {avg_accelerator_token_latency_s * 1000} ms")
            print(f"First 256 Tokens Average Accelerator Token Generation Throughput: {batch_size * (1 / avg_accelerator_token_latency_s)} tps")
            
            # Calculate wall-clock metrics
            wall_clock_thrpt = batch_size * ((prompt_length + generation_length) / wall_clock_latency)
            print(f"Wall-Clock Latency: {wall_clock_latency} s")
            print(f"Wall-Clock Throughput: {batch_size * ((prompt_length + generation_length) / wall_clock_latency)} tps")

            csv_metrics.extend([avg_accelerator_token_latency_ms, avg_accelerator_token_thrpt, wall_clock_latency, wall_clock_thrpt])

            # Add metrics to CSV
            print("Adding results to CSV")
            all_csv_metrics.append(csv_metrics)
        except Exception as e:
            print(f"Error: {e}")
            continue

    filename = args.output + ".csv"
    save_results(all_csv_metrics, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end benchmarking for gen-ai")
    parser.add_argument('-o', '--output', type=str, default='genai_e2e', help='Output csv file name')
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Model folder')
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Device to use')
    args = parser.parse_args()
    main(args)
