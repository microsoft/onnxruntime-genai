# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import json
import os
import time

import onnxruntime_genai as og

# og.set_log_options(enabled=True, model_input_values=True, model_output_values=True)

def get_json_grammar(input_tools: str):
    """
    Example of input_tools format:
    '[{"name": "fn1", "description": "fn details", "parameters": {"p1": {"description": "details", "type": "string"}}}, {"name": "fn2", ...}, {"name": "fn3", ...}, ...]'
    """
    
    # Get list of tools as string for prompt registration
    # Must be without brackets and must use single quotes instead of double quotes
    prompt_tools = input_tools[1:-1].replace("\"", "'")

    # Get grammar for guidance
    # Spacing here matters
    grammar = '{ "anyOf": ' + input_tools + ' }'

    return prompt_tools, grammar

def get_lark_grammar(input_tools: str, tool_calling_token: str = "<|tool_call|>"):
    """
    Example of input_tools format:
    '[{"name": "fn1", "description": "fn details", "parameters": {"p1": {"description": "details", "type": "string"}}}, {"name": "fn2", ...}, {"name": "fn3", ...}, ...]'
    """

    # Get tools for prompt registration and inner grammar for LARK grammar
    prompt_tools, inner_grammar = get_json_grammar(input_tools)

    # Get grammar for guidance
    start_row = "start: TEXT | fun_call"
    text_row = "TEXT: /[^{](.|\\n)*/"
    func_row = f"fun_call: {tool_calling_token} %json "
    grammar = start_row + " \n" + text_row + " \n" + func_row + inner_grammar

    return prompt_tools, grammar

def get_guidance_info(guidance_info):
    """
    Returns a JSON string with guidance info
    """
    # Raise error if guidance info is not provided with guidance type
    if not guidance_info:
        raise ValueError("Guidance information is required if guidance type is provided")

    # If guidance info is provided via a JSON file
    if os.path.exists(guidance_info):
        with open(guidance_info, 'r') as f:
            guidance_data = json.load(f)               # Read JSON file into memory
            guidance_data = json.dumps(guidance_data)  # Uses double quotes and lowercases any booleans

        return guidance_data

    # If guidance info is provided as a JSON string
    try:
        tools_list = json.loads(guidance_info)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for tools list. Format must be list of dictionaries stored as a JSON string. Example of expected format: '[{\"name\": \"fn1\"},{\"name\": \"fn2\"}]'")
    if len(tools_list) == 0:
        raise ValueError("Tools list cannot be empty")        

    return guidance_info

def main(args):
    if args.verbose:
        print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            if args.verbose:
                print(f"Setting model to {args.execution_provider}")
            config.append_provider(args.execution_provider)
    model = og.Model(config)

    if args.verbose:
        print("Model loaded")

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if args.verbose:
        print("Tokenizer created")
    if args.verbose:
        print()

    search_options = {
        name: getattr(args, name)
        for name in ["do_sample", "max_length", "min_length", "top_p", "top_k", "temperature", "repetition_penalty"]
        if name in args
    }
    search_options["batch_size"] = 1

    if args.verbose:
        print(search_options)

    system_prompt = args.system_prompt
    prompt_tools = ""
    guidance_type = ""
    guidance_input = ""
    if args.guidance_type != "none":
        guidance_type = args.guidance_type
        guidance_info = get_guidance_info(args.guidance_info)
        if guidance_type == "json_schema":
            prompt_tools, guidance_input = get_json_grammar(guidance_info)
        elif guidance_type == "lark_grammar":
            prompt_tools, guidance_input = get_lark_grammar(guidance_info)
        elif guidance_type == "regex":
            guidance_input = guidance_info
        else:
            raise ValueError("Guidance Type can only be [json_schema, regex, or lark_grammar]")

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    if guidance_type:
        params.set_guidance(guidance_type, guidance_input)
        if args.verbose:
            print("Guidance type is set to:", guidance_type)
            print("Guidance input is:", guidance_input)

    generator = og.Generator(model, params)
    if args.verbose:
        print("Generator created")
    if guidance_type == "json_schema" or guidance_type == "lark_grammar":
        messages = f"""[{{"role": "system", "content": "{system_prompt}", "tools": "{prompt_tools}"}}]"""
    else:
        messages = f"""[{{"role": "system", "content": "{system_prompt}"}}]"""


    # Apply Chat Template
    final_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=False)
    if args.verbose: print(final_prompt)
    final_input = tokenizer.encode(final_prompt)
    # Ignoring the last end of text token as it is messes up the generation when grammar is enabled
    if guidance_type:
        input_tokens = input_tokens[:-1]
    system_prompt_length = len(input_tokens)
    generator.append_tokens(input_tokens)

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

        messages = f"""[{{"role": "user", "content": "{text}"}}]"""

        # Apply Chat Template
        final_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
        if args.verbose: print(final_prompt)
        final_input = tokenizer.encode(final_prompt)
        generator.append_tokens(final_input)

        if args.verbose:
            print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        print()
        print("Output: ", end="", flush=True)

        try:
            while True:
                generator.generate_next_token()
                if args.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                if generator.is_done():
                    break

                new_token = generator.get_next_tokens()[0]
                # print(new_token, end=', ', flush=True)
                print(tokenizer_stream.decode(new_token), end='', flush=True)
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

        # Rewind the generator to the system prompt, this will erase all the memory of the model.
        if args.rewind:
            generator.rewind_to(system_prompt_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Onnx model folder path (must contain genai_config.json and model.onnx)')
    parser.add_argument('-e', '--execution_provider', type=str, required=False, default='follow_config', choices=["cpu", "cuda", "dml", "follow_config"], help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead.")
    parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_sample', action='store_true', help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-re', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('-g', '--timings', action='store_true', default=False, help='Print timing information for each generation step. Defaults to false')
    parser.add_argument('-gtype', '--guidance_type', type=str, default="none", choices=["none", "json_schema", "regex", "lark_grammar"], help='Provide guidance type for the model, options are json_schema, regex, or lark_grammar.')
    parser.add_argument('-ginfo', '--guidance_info', type=str, default="", help='Provide information of the guidance type used (e.g. list of JSON tools, regex string, etc) or the path to the file containing the information. It is required if guidance_type is provided.')
    parser.add_argument('-s', '--system_prompt', type=str, default='You are a helpful AI assistant.', help='System prompt to use for the prompt.')
    parser.add_argument('-r', '--rewind', action='store_true', default=False, help='Rewind to the system prompt after each generation. Defaults to false')
    args = parser.parse_args()
    main(args)
