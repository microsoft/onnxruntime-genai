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

def main(args):
    # Get user arguments
    num_repetitions = args.repetitions
    temperature = 1.0

    # Get tokenizer, and model
    model=og.Model(f'{args.input_folder}')
    tokenizer = og.Tokenizer(model)

    # Generate prompt
    sys_prompt = "<|system|>You are a world class AI programming assistant who excels in software development.\r\nWhen asked your name, you must respond with \"GitHub Copilot\".\r\nFollow the user's requirements carefully & to the letter.\r\nThe user is a proficient software developer working in Visual Studio 2022.\r\nWhile the user may have experience in software development, you should not elude to their background, i.e. prefer general greetings like \"Hello! How can I assist you today?\" This approach respects the user's expertise without immediately categorizing their profession.\r\nFor questions not related to software development, give a reminder that you are an AI programming assistant.\r\nFollow Microsoft content policies and avoid content that violates copyrights.\r\nRespond in the following locale: en-US\r\n\r\nRespond in Markdown, for multi-line code, use language-specific markdown code fences.\r\nEnsure your response is short, impersonal, expertly written and easy to understand.\r\nBefore responding take a deep breath and then work on the user's problem step-by-step.\r\nFocus on being clear, helpful, and thorough without assuming extensive prior knowledge.\r\n\r\nGenerated code should adhere to the existing coding style in the provided context.\r\nWhen generating code prefer languages provided in context. If the coding language is unclear fallback to generating code in C#.\r\nGenerate code that can be copy & pasted without modification, i.e. preserve surrounding user code, avoid placeholder comments like \"existing code here...\" etc. \r\nAfter generating mutated code consider mentioning what specifically was changed and your reasoning if it would help the user.\r\n\r\nThe active document or selection is the source code the user is looking at right now and is what they care about.<|end|><|user|>What is 1+1?<|end|><|assistant|>"
    user_prompt = "<|user|>What are the first 7 numbers in the fibonacci sequence?<|end|>"
    sys_tokens = tokenizer.encode(sys_prompt)
    user_tokens = tokenizer.encode(user_prompt)
    sys_user_tokens = tokenizer.encode(sys_prompt + user_prompt)
    sys_length = len(sys_tokens)
    user_length = len(user_tokens)
    sys_user_length = len(sys_user_tokens)

    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, temperature=temperature)
    if args.max_length > 0: params.set_search_options(max_length=args.max_length)

    print("Warming up...")
    for _ in tqdm(range(args.warmup)):
        generator = og.Generator(model, params)
        generator.append_tokens(sys_user_tokens)
        while not generator.is_done():
            generator.generate_next_token()
        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    # Separate System and User Prompt Processing
    sys_times = []
    user_times = []
    print("Benchmarking Separate System and User Prompt Processing...")
    for _ in tqdm(range(num_repetitions)):
        # Prepare run
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, temperature=temperature)
        if args.max_length > 0: params.set_search_options(max_length=args.max_length)

        generator = og.Generator(model, params)

        # Measure system prompt processing
        sys_start_time = time.perf_counter()
        generator.append_tokens(sys_tokens)
        sys_end_time = time.perf_counter()
        sys_times.append(sys_end_time - sys_start_time)

        # Measure user prompt processing
        user_start_time = time.perf_counter()
        generator.append_tokens(user_tokens)
        user_end_time = time.perf_counter()
        user_times.append(user_end_time - user_start_time)

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    # Process System and User Prompts together
    sys_user_times = []
    for _ in tqdm(range(num_repetitions)):
        # Prepare run
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, temperature=temperature)
        if args.max_length > 0: params.set_search_options(max_length=args.max_length)

        generator = og.Generator(model, params)

        # Measure system and user prompt processing
        sys_user_start_time = time.perf_counter()
        generator.append_tokens(sys_user_tokens)
        sys_user_end_time = time.perf_counter()
        sys_user_times.append(sys_user_end_time - sys_user_start_time)

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    # Print args
    print(f"Prompt Length: {sys_length} tokens")
    print(f"User Prompt Length: {user_length} tokens")
    print(f"System + User Prompt Length: {sys_user_length} tokens")
    if args.max_length > 0: print(f"Max Generation Length: {args.max_length} tokens")
    print(f"Repetitions: {num_repetitions}")
    print(f"Warmup Runs: {args.warmup}")
    print()
    # Calculate system prompt processing metrics
    avg_sys_latency_s = sum(sys_times) / len(sys_times)
    avg_sys_latency_ms = avg_sys_latency_s * 1000
    print(f"Average System Prompt Processing Latency: {avg_sys_latency_ms} ms")
    # Calculate user prompt processing metrics
    avg_user_latency_s = sum(user_times) / len(user_times)
    avg_user_latency_ms = avg_user_latency_s * 1000
    print(f"Average User Prompt Processing Latency: {avg_user_latency_ms} ms")
    # Calculate system and user prompt processing metrics
    avg_sys_user_latency_s = sum(sys_user_times) / len(sys_user_times)
    avg_sys_user_latency_ms = avg_sys_user_latency_s * 1000
    print(f"Average (System + User) Prompt Processing Latency: {avg_sys_user_latency_ms} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end benchmarking for gen-ai")
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Onnx model folder path (must contain genai_config.json and model.onnx)')
    parser.add_argument('-m', '--max_length', type=int, default=-1, help='Max length is either a combination of prompt and generation length or one value broadcasting for all.')
    parser.add_argument('-r', '--repetitions', type=int, default=10, help='Number of times to repeat the benchmark')
    parser.add_argument('-w', '--warmup', type=int, default=5, help='Number of warmup runs before benchmarking')
    args = parser.parse_args()
    main(args)