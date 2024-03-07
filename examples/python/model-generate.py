import onnxruntime_genai as og
import argparse
import time

def main(args):
    if args.verbose: print("Loading model...")
    model = og.Model(f'{args.model}', og.DeviceType.CPU if args.execution_provider == 'cpu' else og.DeviceType.CUDA)
    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    if args.verbose: print("Tokenizer created")

    if args.input is not None:
        prompts = [args.input]
    else:
        prompts = ["I like walking my cute dog",
                "What is the best restaurant in town?",
                "Hello, how are you today?"]
    input_tokens = tokenizer.encode_batch(prompts)
    if args.verbose: print("Prompt(s) encoded")

    params = og.GeneratorParams(model)
    params.set_search_options({"max_length": args.max_length, "top_p": args.top_p, "top_k": args.top_k, "temperature": args.temperature, "repetition_penalty": args.repetition_penalty})
    params.input_ids = input_tokens
    if args.verbose: print("GeneratorParams created")

    if args.verbose: print("Generating tokens ...")
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
    parser.add_argument('-ep', '--execution_provider', type=str, choices=['cpu', 'cuda'], required=True, help='Execution provider (device) to use, default is CPU, use CUDA for GPU')
    parser.add_argument('-i', '--input', type=str, required=False, help='Input prompt to generate tokens from')
    parser.add_argument('-l', '--max_length', type=int, default=512, help='Max number of tokens to generate after prompt')
    parser.add_argument('-p', '--top_p', type=float, default=0.9, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, default=50, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, default=1.0, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    main(args)