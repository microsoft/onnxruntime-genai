import onnxruntime_genai as og
import argparse
import time

def main(args):
    if args.verbose: print("Loading model...")
    model = og.Model(f'{args.model}')
    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    if args.verbose: print("Tokenizer created")

    if args.prompts is not None:
        prompts = args.prompts
    else:
        prompts = ["I like walking my cute dog",
                "What is the best restaurant in town?",
                "Hello, how are you today?"]
<<<<<<< Updated upstream
=======
    
    if args.chat_template:
        if args.chat_template.count('{') != 1 or args.chat_template.count('}') != 1:
            print("Error, chat template must have exactly one pair of curly braces, e.g. '<|user|>\n{input} <|end|>\n<|assistant|>'")
            exit(1)
        prompts[:] = [f'{args.chat_template.format(input=text)}' for text in prompts]
        
>>>>>>> Stashed changes
    input_tokens = tokenizer.encode_batch(prompts)
    if args.verbose: print("Prompt(s) encoded")

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=args.max_length, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty)
    if args.cuda_graph_with_max_batch_size > 0:
        params.try_use_cuda_graph_with_max_batch_size(args.cuda_graph_with_max_batch_size)
    params.input_ids = input_tokens
    if args.verbose: print("GeneratorParams created")

    if args.verbose: print("Generating tokens ...\n")
    start_time = time.time()
    output_tokens = model.generate(params)
    run_time = time.time() - start_time

    for i in range(len(prompts)):
        print(f'Prompt #{i}: {prompts[i]}')
        print()
        print(tokenizer.decode(output_tokens[i]))
        print()

    print()
    print(f"Tokens: {len(output_tokens[0])} Time: {run_time:.2f} Tokens per second: {len(output_tokens[0])/run_time:.2f}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end token generation loop example for gen-ai")
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-pr', '--prompts', nargs='*', required=False, help='Input prompts to generate tokens from')
    parser.add_argument('-l', '--max_length', type=int, default=512, help='Max number of tokens to generate after prompt')
    parser.add_argument('-p', '--top_p', type=float, default=0.9, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, default=50, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, default=1.0, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('-c', '--cuda_graph_with_max_batch_size', type=int, default=0, help='Max batch size for CUDA graph')
    args = parser.parse_args()
    main(args)