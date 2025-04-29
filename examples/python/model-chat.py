# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnxruntime_genai as og
import argparse
import time

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

    # Get model type
    model_type = None
    if hasattr(model, "type"):
        model_type = model.type
    else:
        import json, os

        with open(os.path.join(args.model_path, "genai_config.json"), "r") as f:
            genai_config = json.load(f)
            model_type = genai_config["model"]["type"]
    
    # Set chat template
    if args.chat_template:
        if args.chat_template.count('{') != 1 or args.chat_template.count('}') != 1:
            raise ValueError("Chat template must have exactly one pair of curly braces with input word in it, e.g. '<|user|>\n{input} <|end|>\n<|assistant|>'")
    else:
        if model_type.startswith("phi4"):
            args.chat_template = '<|im_start|>user<|im_sep|>\n{input}<|im_end|>\n<|im_start|>assistant<|im_sep|>'
        elif model_type.startswith("phi"): # For Phi2 and Phi3
            args.chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
        elif model_type.startswith("llama"):
            args.chat_template = '<|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
            print("Using Chat Template for LLAMA 3, if you are using LLAMA  2 please pass the argument --chat_template '{input} [/INST]')")
        elif model_type.startswith("qwen2"):
            args.chat_template = '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'
        elif model_type == "gemma3_text":
            args.chat_template = '<start_of_turn>user\n{system_prompt}{input}<end_of_turn>\n<start_of_turn>model\n'
        else:
            raise ValueError(f"Chat Template for model type {model_type} is not known. Please provide chat template using --chat_template")

    if args.verbose:
        print("Model type is:", model_type)
        print("Chat Template is:", args.chat_template)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)
    if args.verbose: print("Generator created")

    # Set system prompt
    if "<|" in args.system_prompt and "|>" in args.system_prompt:
        # User-provided system template already has tags
        system_prompt = args.system_prompt
    else:
        if model_type.startswith('phi4'):
            system_prompt = f"<|im_start|>system<|im_sep|>\n{args.system_prompt}<|im_end|>"
        elif model_type.startswith('phi'): # For Phi2 and Phi3
            system_prompt = f"<|system|>\n{args.system_prompt}<|end|>"
        elif model_type.startswith("llama"):
            system_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{args.system_prompt}<|eot_id|>"
            print("Using System Prompt for LLAMA 3, if you are using LLAMA  2 please pass the argument --system_prompt '<s>[INST] <<SYS>>\\n{args.system_prompt}\\n<</SYS>>')")
        elif model_type.startswith("qwen2"):
            system_prompt = f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n"
        elif model_type == "gemma3_text":
            system_prompt = f"{args.system_prompt}"
        else:
            system_prompt = args.system_prompt

    system_tokens = tokenizer.encode(system_prompt)
    generator.append_tokens(system_tokens)
    system_prompt_length = len(system_tokens)

    # Keep asking for input prompts in a loop
    while True:
        text = input("Prompt (Use quit() to exit): ")
        if not text:
            print("Error, input cannot be empty")
            continue

        if text == "quit()":
            break

        if args.timings: started_timestamp = time.time()

        prompt = f'{args.chat_template.format(system_prompt=system_prompt, input=text)}'
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

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")
        
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
    parser.add_argument('-c', '--chat_template', type=str, default='', help='Chat template to use for the prompt. User input will be injected into {input}')
    parser.add_argument('-s', '--system_prompt', type=str, default='You are a helpful assistant.', help='System prompt to use for the prompt.')
    parser.add_argument('-r', '--rewind', action='store_true', default=False, help='Rewind to the system prompt after each generation. Defaults to false')
    args = parser.parse_args()
    main(args)
