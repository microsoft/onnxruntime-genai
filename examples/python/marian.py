# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnxruntime_genai as og
import argparse
import time

og.set_log_options(enabled=True, model_input_values=True, model_output_values=True, model_logits=False)

def main(args):
    if args.verbose: print("Loading model...")
    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        if args.verbose:
            print(f"Setting model to {args.execution_provider}...")
        config.append_provider(args.execution_provider)
    model = og.Model(config)

    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if args.verbose: print("Tokenizer created")
    print("Tokenizer = ", tokenizer)

    if hasattr(args, 'prompts'):
        prompts = args.prompts
    else:
        if args.non_interactive:
            prompts = ["Hello"]
        else:
            text = input("Input: ")
            prompts = [text]

    input_tokens = tokenizer.encode_batch(prompts)
    if args.verbose: print(f'Prompt(s) encoded: {prompts}')

    params = og.GeneratorParams(model)

    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args} 
    search_options['num_beams'] = 1

    if (args.verbose): print(f'Args: {args}')
    if (args.verbose): print(f'Search options: {search_options}')

    params.set_search_options(**search_options)
    if args.verbose: print("GeneratorParams created")

    generator = og.Generator(model, params)
    if args.verbose: print("Generator created")

    print("input tokens = ", input_tokens)
    generator.append_tokens(input_tokens)
    if args.verbose: print("Input tokens added")

    if args.verbose: print("Generating tokens ...\n")
    start_time = time.time()
    all_output_ids = []
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        # print(generator.get_output("logits"))
        all_output_ids.append(new_token)
    run_time = time.time() - start_time

    for i in range(len(prompts)):
        print(f'Prompt #{i}: {prompts[i]}')
        print()
        print(tokenizer.decode(generator.get_sequence(i)))
        print()

    print()
    total_tokens = sum(len(generator.get_sequence(i)) for i in range(len(prompts)))
    print(f"Tokens: {total_tokens} Time: {run_time:.2f} Tokens per second: {total_tokens/run_time:.2f}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end token generation loop example for gen-ai")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Onnx model folder path (must contain genai_config.json and model.onnx)')
    parser.add_argument("-e", "--execution_provider", type=str, required=True, choices=["cpu", "cuda", "dml"], help="Provider to run model")
    parser.add_argument('-pr', '--prompts', nargs='*', required=False, help='Input prompts to generate tokens from. Provide this parameter multiple times to batch multiple prompts')
    parser.add_argument('-i', '--min_length', type=int, default=1, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, default=50, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_sample', action='store_true', help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('--non-interactive', action=argparse.BooleanOptionalAction, required=False, default=False, help='Non-interactive mode, mainly for CI usage')

    args = parser.parse_args()
    main(args)