import onnxruntime_genai as og
import argparse
import time

def main(args):
    if args.verbose: print("Loading model...")

    if hasattr(args, 'prompts'):
        prompts = args.prompts
    else:
        if args.non_interactive:
            prompts = ["The first 4 digits of pi are",
                       "The square root of 2 is",
                       "The first 6 numbers of the Fibonacci sequence are",]
        else:
            text = input("Input: ")
            prompts = [text]

    batch_size = len(prompts)

    config = og.Config(args.model_path)
    config.overlay(f'{{"search": {{"batch_size": {batch_size}}}}}')

    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            if args.verbose:
                print(f"Setting model to {args.execution_provider}...")
            config.append_provider(args.execution_provider)
    model = og.Model(config)

    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    if args.verbose: print("Tokenizer created")

    if args.chat_template:
        if args.chat_template.count('{') != 1 or args.chat_template.count('}') != 1:
            print("Error, chat template must have exactly one pair of curly braces, e.g. '<|user|>\n{input} <|end|>\n<|assistant|>'")
            exit(1)
        prompts[:] = [f'{args.chat_template.format(input=text)}' for text in prompts]

    input_tokens = tokenizer.encode_batch(prompts)
    if args.verbose: print(f'Prompt(s) encoded: {prompts}')

    params = og.GeneratorParams(model)

    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args} 
    search_options['num_beams'] = 3

    if (args.verbose): print(f'Args: {args}')
    if (args.verbose): print(f'Search options: {search_options}')

    params.set_search_options(**search_options)
    if args.verbose: print("GeneratorParams created")

    generator = og.Generator(model, params)
    if args.verbose: print("Generator created")
    
    generator.append_tokens(input_tokens)
    if args.verbose: print("Input tokens added")

    if args.verbose: print("Generating tokens ...\n")
    start_time = time.time()
    while not generator.is_done():
        generator.generate_next_token()
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
    parser.add_argument('-e', '--execution_provider', type=str, required=False, default='follow_config', choices=["cpu", "cuda", "dml", "follow_config"], help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead.")
    parser.add_argument('-pr', '--prompts', nargs='*', required=False, help='Input prompts to generate tokens from. Provide this parameter multiple times to batch multiple prompts')
    parser.add_argument('-i', '--min_length', type=int, default=25, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, default=50, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_sample', action='store_true', help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('-b', '--batch_size_for_cuda_graph', type=int, default=1, help='Max batch size for CUDA graph')
    parser.add_argument('-c', '--chat_template', type=str, default='', help='Chat template to use for the prompt. User input will be injected into {input}. If not set, the prompt is used as is.')
    parser.add_argument('--non-interactive', action=argparse.BooleanOptionalAction, required=False, default=False, help='Non-interactive mode, mainly for CI usage')

    args = parser.parse_args()
    main(args)
