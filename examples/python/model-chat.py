# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import time

import onnxruntime_genai as og
from common import apply_chat_template, get_config, get_generator_params_args, get_guidance, get_guidance_args, get_search_options, register_ep, set_logger


def main(args):
    if args.debug:
        set_logger()
    register_ep(args.execution_provider, args.ep_path, args.use_winml)

    if args.verbose:
        print("Loading model...")

    # Create model
    config = get_config(args.model_path, args.execution_provider)
    model = og.Model(config)
    if args.verbose:
        print("Model loaded")

    # Create tokenizer
    tokenizer = og.Tokenizer(model)
    stream = tokenizer.create_stream()
    if args.verbose:
        print("Tokenizer created")

    # Get and set search options for generator params
    params = og.GeneratorParams(model)
    search_options = get_search_options(args)
    params.set_search_options(**search_options)
    if args.verbose:
        print(f"GeneratorParams created: {search_options}")

    # Create system message
    message = [{"role": "system", "content": args.system_prompt}]

    # Get guidance info if requested
    guidance_type, guidance_data, tools = "", "", ""
    if args.response_format != "":
        print("Make sure your tool call start id and tool call end id are marked as special in tokenizer.json")
        guidance_type, guidance_data, tools = get_guidance(
            response_format=args.response_format,
            filepath=args.tools_file,
            text_output=args.text_output,
            tool_output=args.tool_output,
            tool_call_start=args.tool_call_start,
            tool_call_end=args.tool_call_end,
        )
        message[0]["tools"] = tools

        params.set_guidance(guidance_type, guidance_data)
        if args.verbose:
            print()
            print(f"Guidance type is: {guidance_type}")
            print(f"Guidance data is: \n{guidance_data}")
            print()

    # Create generator
    generator = og.Generator(model, params)
    if args.verbose:
        print("Generator created")

    # Apply chat template
    try:
        system_prompt = apply_chat_template(model_path=args.model_path, tokenizer=tokenizer, messages=json.dumps(message), tools=tools, add_generation_prompt=False)
    except Exception as e:
        if args.verbose:
            print(f"Exception in apply_chat_template for system_prompt: {e}")
        system_prompt = args.system_prompt
    if args.verbose:
        print(f"System prompt: {system_prompt}")

    # Encode system prompt and append tokens to model
    system_tokens = tokenizer.encode(system_prompt)
    system_prompt_length = len(system_tokens)
    generator.append_tokens(system_tokens)

    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    # Keep asking for input prompts in a loop
    while True:
        text = input("Prompt (Use quit() to exit): ")
        if not text:
            print("Error, input cannot be empty")
            continue

        if text == "quit()":
            break

        if args.timings:
            started_timestamp = time.time()

        # Create user message
        message = [{"role": "user", "content": text}]
        
        # Apply chat template
        try:
            user_prompt = apply_chat_template(model_path=args.model_path, tokenizer=tokenizer, messages=json.dumps(message), add_generation_prompt=True)
        except Exception as e:
            if args.verbose:
                print(f"Exception in apply_chat_template for user_prompt: {e}")
            user_prompt = text
        if args.verbose:
            print(f"User prompt: {user_prompt}")

        # Encode user prompt and append tokens to model
        user_tokens = tokenizer.encode(user_prompt)
        generator.append_tokens(user_tokens)

        if args.verbose:
            print("Running generation loop...")
        if args.timings:
            first = True
            new_tokens = []

        print()
        print("Output: ", end="", flush=True)

        # Run generation loop
        try:
            while not generator.is_done():
                generator.generate_next_token()
                if args.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]
                print(stream.decode(new_token), end='', flush=True)
                if args.timings: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(
                f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens) / prompt_time:.2f} tps, New tokens per second: {len(new_tokens) / run_time:.2f} tps"
            )

        # Rewind the generator to the system prompt. This will erase all the chat history with the model.
        if args.rewind:
            generator.rewind_to(system_prompt_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI chat example for ORT GenAI")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='ONNX model folder path (must contain genai_config.json and model.onnx)')
    parser.add_argument('-e', '--execution_provider', type=str, required=False, default='follow_config', choices=["cpu", "cuda", "dml", "follow_config"], help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='Dump input and output tensors with debug mode. Defaults to false')
    parser.add_argument('-g', '--timings', action='store_true', default=False, help='Print timing information for each generation step. Defaults to false')
    parser.add_argument('-sp', '--system_prompt', type=str, default='You are a helpful AI assistant.', help='System prompt to use for the model.')
    parser.add_argument('-rw', '--rewind', action='store_true', default=False, help='Rewind to the system prompt after each generation. Defaults to false')
    parser.add_argument("--ep_path", type=str, required=False, default='', help='Path to execution provider DLL/SO for plug-in providers (ex: onnxruntime_providers_cuda.dll or onnxruntime_providers_tensorrt.dll)')
    parser.add_argument("--use_winml", action=argparse.BooleanOptionalAction, required=False, default=False, help='Use WinML to register execution providers') 

    get_generator_params_args(parser)
    get_guidance_args(parser)

    args = parser.parse_args()
    main(args)
