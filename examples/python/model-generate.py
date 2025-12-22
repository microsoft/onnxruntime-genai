# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import time

import onnxruntime_genai as og
from common import get_config, get_generator_params_args, get_search_options
# og.set_log_options(enabled=True, model_input_values=inputs, model_output_values=outputs)

def main(args):
    if args.verbose:
        print("Loading model...")

    if hasattr(args, "prompts"):
        prompts = args.prompts
    else:
        if args.non_interactive:
            prompts = [
                "The first 4 digits of pi are",
                "The square root of 2 is",
                "The first 6 numbers of the Fibonacci sequence are",
            ]
        else:
            text = input("Input: ")
            prompts = [text]

    search_config = {"batch_size": len(prompts), "chunk_size": args.chunk_size, "num_beams": args.num_beams}
    config = get_config(args.model_path, args.execution_provider, ep_options={}, search_options=search_config)

    model = og.Model(config)
    if args.verbose:
        print("Model loaded")

    tokenizer = og.Tokenizer(model)
    if args.verbose:
        print("Tokenizer created")

    if args.chat_template:
        if args.chat_template.count("{") != 1 or args.chat_template.count("}") != 1:
            print(
                "Error, chat template must have exactly one pair of curly braces, e.g. '<|user|>\n{input} <|end|>\n<|assistant|>'"
            )
            exit(1)
        prompts[:] = [f"{args.chat_template.format(input=text)}" for text in prompts]

    input_tokens = tokenizer.encode_batch(prompts)
    if args.verbose:
        print(f"Prompt(s) encoded: {prompts}")

    params = og.GeneratorParams(model)
    search_options = get_search_options(args)
    params.set_search_options(**search_options)
    if args.verbose:
        print(f"GeneratorParams created: {search_options}")

    generator = og.Generator(model, params)
    if args.verbose:
        print("Generator created")

    generator.append_tokens(input_tokens)
    if args.verbose:
        print("Input tokens added")

    if args.verbose:
        print("Generating tokens ...\n")
    start_time = time.time()
    while True:
        generator.generate_next_token()
        if generator.is_done():
            break
    run_time = time.time() - start_time

    for i in range(len(prompts)):
        print(f"Prompt #{i}: {prompts[i]}")
        print()
        print(tokenizer.decode(generator.get_sequence(i)))
        print()

    print()
    total_tokens = sum(len(generator.get_sequence(i)) for i in range(len(prompts)))
    print(f"Tokens: {total_tokens} Time: {run_time:.2f} Tokens per second: {total_tokens / run_time:.2f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end token generation loop example for ORT GenAI")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="ONNX model folder path (must contain genai_config.json and model.onnx)")
    parser.add_argument("-e", "--execution_provider", type=str, required=False, default="follow_config", choices=["cpu", "cuda", "dml", "NvTensorRtRtx", "follow_config"], help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Print verbose output and timing information. Defaults to false")
    parser.add_argument("-pr", "--prompts", nargs="*", required=False, help="Input prompts to generate tokens from. Provide this parameter multiple times to batch multiple prompts")
    parser.add_argument("-ct", "--chat_template", type=str, default="", help="Chat template to use for the prompt. User input will be injected into {input}. If not set, the prompt is used as is.")
    parser.add_argument("--non-interactive", action=argparse.BooleanOptionalAction, required=False, default=False, help="Non-interactive mode, mainly for CI usage")

    get_generator_params_args(parser)

    args = parser.parse_args()
    main(args)
