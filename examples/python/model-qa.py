# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnxruntime_genai as og
import argparse
import time
import json
import numpy as np

def get_tools_list(input_tools):
    # input_tools format: '[{"name": "fn1", "description": "fn details", "parameters": {"p1": {"description": "details", "type": "string"}}},
    # {"fn2": 2},{"fn3": 3}]'
    tools_list = []
    try:
        tools_list = json.loads(input_tools)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for tools list, expected format: '[{\"name\": \"fn1\"},{\"name\": \"fn2\"}]'")
    if len(tools_list) == 0:
        raise ValueError("Tools list cannot be empty")
    return tools_list

def create_prompt_tool_input(tools_list):
    tool_input = str(tools_list[0])
    for tool in tools_list[1:]:
        tool_input += ',' + str(tool)
    return tool_input

def get_json_grammar(input_tools):
    tools_list = get_tools_list(input_tools)
    prompt_tool_input = create_prompt_tool_input(tools_list)
    if len(tools_list) == 1:
        return prompt_tool_input, json.dumps(tools_list[0])
    else:
        output = '{ "anyOf": [' + json.dumps(tools_list[0])
        for tool in tools_list[1:]:
            output += ',' + json.dumps(tool)
        output += '] }'
        return prompt_tool_input, output

def get_lark_grammar(input_tools):
    tools_list = get_tools_list(input_tools)
    prompt_tool_input = create_prompt_tool_input(tools_list)
    if len(tools_list) == 1:
        # output = ("start: TEXT | fun_call\n" "TEXT: /[^{](.|\\n)*/\n" " fun_call: <|tool_call|> %json " + json.dumps(tools_list[0]))
        output = ("start: TEXT | fun_call\n" "TEXT: /[^{](.|\\n)*/\n" " fun_call: <|tool_call|> %json " + json.dumps(convert_tool_to_grammar_input(tools_list[0])))
        return prompt_tool_input, output
    else:
        return prompt_tool_input, "start: TEXT | fun_call \n TEXT: /[^{](.|\n)*/ \n fun_call: <|tool_call|> %json {\"anyOf\": [" + ','.join([json.dumps(tool) for tool in tools_list]) + "]}"

def convert_tool_to_grammar_input(tool):
    param_props = {}
    required_params = []
    for param_name, param_info in tool.get("parameters", {}).items():
        param_props[param_name] = {
            "type": param_info.get("type", "string"),
            "description": param_info.get("description", "")
        }
        required_params.append(param_name)
    output_schema = {
        "description": tool.get('description', ''),
        "type": "object",
        "required": ["name", "parameters"],
        "additionalProperties": False,
        "properties": {
            "name": { "const": tool["name"] },
            "parameters": {
                "type": "object",
                "properties": param_props,
                "required": required_params,
                "additionalProperties": False
            }
        }
    }
    if len(param_props) == 0:
        output_schema["required"] = ["name"]
    return output_schema

def main(args):
    if args.verbose: print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            if args.verbose: print(f"Setting model to {args.execution_provider}")
            config.append_provider(args.execution_provider)
    model = og.Model(config)

    if args.verbose: print("Model loaded")
    
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if args.verbose: print("Tokenizer created")
    if args.verbose: print()

    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}
    search_options['batch_size'] = 1

    if args.verbose: print(search_options)

    system_prompt = args.system_prompt
    guidance_type = ""
    prompt_tool_input = ""
    guidance_input = ""
    if args.guidance_type != "none":
        guidance_type = args.guidance_type
        if not args.guidance_info:
            raise ValueError("Guidance information is required if guidance type is provided")
        if guidance_type == "json_schema" or guidance_type == "lark_grammar":
            tools_list = args.guidance_info
            if guidance_type == "json_schema":
                prompt_tool_input, guidance_input = get_json_grammar(tools_list)
            elif guidance_type == "lark_grammar":
                prompt_tool_input, guidance_input = get_lark_grammar(tools_list)
        elif guidance_type == "regex":
            guidance_input = args.guidance_info
        else:
            raise ValueError("Guidance Type can only be [json_schema, regex, or lark_grammar]")

    # Keep asking for input prompts in a loop
    while True:
        if args.input_prompt:
            text = args.input_prompt
        else:
            text = input("Prompt (Use quit() to exit): ")
        if not text:
            print("Error, input cannot be empty")
            continue

        if text == "quit()":
            break

        if args.timings: started_timestamp = time.time()

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)

        if guidance_type:
            params.set_guidance(guidance_type, guidance_input)
            if args.verbose:
                print("Guidance type is set to:", guidance_type)
                print("Guidance input is:", guidance_input)

        generator = og.Generator(model, params)
        if args.verbose: print("Generator created")
        if guidance_type == "json_schema" or guidance_type == "lark_grammar":
            messages = f"""[{{"role": "system", "content": "{system_prompt}", "tools": "{prompt_tool_input}"}}, {{"role": "user", "content": "{text}"}}]"""
        else:
            messages = f"""[{{"role": "system", "content": "{system_prompt}"}}, {{"role": "user", "content": "{text}"}}]"""
        # Apply Chat Template
        if model.type == "marian-ssru":
            prompt = text
        else:
            prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)

        input_tokens = tokenizer.encode(prompt)
        generator.append_tokens(input_tokens)

        if args.verbose: print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        print()
        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.generate_next_token()
                if args.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)
                if args.timings: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled

        del generator

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")
        # If Input prompt is provided it will just run the model for the input prompt and exit
        if args.input_prompt:
            break

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
    parser.add_argument('-ginfo', '--guidance_info', type=str, default='', help='Provide information of the guidance type used, it could be either tools or regex string. It is required if guidance_type is provided')
    parser.add_argument('-s', '--system_prompt', type=str, default='You are a helpful AI assistant.', help='System prompt to use for the prompt.')
    parser.add_argument('-inp', '--input_prompt', type=str, default='', help='Input Prompt, if provided it will just run the prompt and exit')
    args = parser.parse_args()
    main(args)
