import onnxruntime_genai as og
import argparse
import time
from ...src.python.py.models.builder import create_model
from datasets import load_dataset 
import torch
import json
import tqdm
 
# def calculate_perplexity(args, output_dir):
#     print("We are now calculating perplexity")
 
#     # actual model object
#     model = og.Model(f'{output_dir}')

#     # tokenizer should already be created
#     tokenizer = og.Tokenizer(model)
#     print(tokenizer)

#     test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

#     print(test)
#     # Calculate the perplexity, understand the output of the models
#     # Understand the output from a model and logits.
#     print("Test is now loaded")

#     max_length = model.config.n_positions
#     stride = 512
#     seq_len = encodings.input_ids.size(1)

#     nlls = []
#     prev_end_loc = 0
#     for begin_loc in tqdm(range(0, seq_len, stride)):
#         end_loc = min(begin_loc + max_length, seq_len)
#         trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
#         input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#         target_ids = input_ids.clone()
#         target_ids[:, :-trg_len] = -100

#         with torch.no_grad():
#             outputs = model(input_ids, labels=target_ids)

#             # loss is calculated using CrossEntropyLoss which averages over valid labels
#             # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
#             # to the left by 1.
#             neg_log_likelihood = outputs.loss

#         nlls.append(neg_log_likelihood)

#         prev_end_loc = end_loc
#         if end_loc == seq_len:
#             break

#     ppl = torch.exp(torch.stack(nlls).mean())
#     return



def validate_model(args, output_dir):
    if args.verbose: print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    model = og.Model(f'{output_dir}')
    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if args.verbose: print("Tokenizer created")
    if args.verbose: print()
    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}
    
    # Set the max length to something sensible by default, unless it is specified by the user,
    # since otherwise it will be set to the entire context length

    search_options['max_length'] = 512

    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'


    inputs = ["Provide a detailed analysis of the causes and consequences of the French Revolution, including key events, figures, and social changes.",
              "Explain the process of photosynthesis in plants, detailing the chemical reactions involved, the role of chlorophyll, and the importance of sunlight"]   
    
    for input in inputs:

        complete_text = ''
        
        if args.timings: started_timestamp = time.time()

        # If there is a chat template, use it
        prompt = f'{chat_template.format(input=input)}'


        # Tokenizer has interesting behavior, it creates multiple inputs
        input_tokens = tokenizer.encode(prompt)


        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens

        generator = og.Generator(model, params)
        if args.verbose: print("Generator created")

        if args.verbose: print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        print()
        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                if args.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]
                
                value_to_save = tokenizer_stream.decode(new_token)

                complete_text += value_to_save

                print(tokenizer_stream.decode(new_token), end='', flush=True)

                
                if args.timings: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

                        

        with open('output.txt', 'a') as file:
            file.write(complete_text)


        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")

    parser.add_argument('-j', '--json', type=str, required=True, help='Path to the JSON file containing the arguments')

    # parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('-g', '--timings', action='store_true', default=False, help='Print timing information for each generation step. Defaults to false')
        
    args = parser.parse_args()

    with open(args.json, 'r') as file:
        data = json.load(file)
        args.model = data['models']

    '''
    1. Download the model 
    2. Convert to onnx format 
    3. Create model and tokenizer from the model 
    '''   
    model = create_model(args.model, '', './output', 'int4', 'cpu', "./cache")

    '''
    4. Iterate through input texts. In each iteration, embed the input, create generator, generate output text. 
    5. Log the input text and output text for manual evaluation. 
    '''
    validate_model(args, 'output')

    '''
    6. Automatically calculate the perplexity metrics if the model has the corresponding dataset *. 
    '''
    # calculate_perplexity(args, 'output')