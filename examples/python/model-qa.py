import onnxruntime_genai as og
import argparse
import time

def main(args):
    if args.verbose: print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    model = og.Model(f'{args.model}')
    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if args.verbose: print("Tokenizer created")
    if args.verbose: print()

    # Keep asking for input prompts in a loop
    while True:
        text = input("Input: ")
        if not text:
            print("Error, input cannot be empty")
            continue

        if args.timings: started_timestamp = time.time()

        input_tokens = tokenizer.encode(args.system_prompt + text)

        prompt_length = 

        params = og.GeneratorParams(model)
        params.set_search_options({"do_sample": False, "max_length": args.max_length, "min_length": args.min_length, "top_p": args.top_p, "top_k": args.top_k, "temperature": args.temperature, "repetition_penalty": args.repetition_penalty})
        params.input_ids = input_tokens
        generator = og.Generator(model, params)
        if args.verbose: print("Generator created")

        if args.verbose: print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            if args.timings:
                if first:
                    first_token_timestamp = time.time()
                    first = False

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end='', flush=True)
            if args.timings: new_tokens.append(new_token)
        print()

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {prompt_length/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end chat-bot example for gen-ai")
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-i', '--min_length', type=int, default=0, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, default=200, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-p', '--top_p', type=float, default=0.9, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, default=50, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, default=1.0, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('-s', '--system_prompt', type=str, default='', help='Prepend a system prompt to the user input prompt. Defaults to empty')
    parser.add_argument('-g', '--timings', action='store_true', help='Print timing information for each generation step. Defaults to false')
    args = parser.parse_args()
    main(args)