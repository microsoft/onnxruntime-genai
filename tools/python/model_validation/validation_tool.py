import onnxruntime_genai as og
import argparse
from onnxruntime_genai.models.builder import create_model
import json
import os
import pandas as pd

def create_table(output):
    df = pd.DataFrame(output, columns=['Model Name', 'Validation Completed', 'Exceptions / Failures'])
    return df

def validate_model(args, model_dict, model_dir):
    if args["verbose"]: print("Loading model...")

    model = og.Model(f'{model_dir}')

    if args["verbose"]: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream() 

    if args["verbose"]: print("Tokenizer created")
    if args["verbose"]: print()  

    chat_template = model_dict["chat_template"]

    search_options = args["search_options"]

    for text in args["inputs"]:

        complete_text = ''
        
        prompt = f'{chat_template.format(input=text)}'

        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens

        generator = og.Generator(model, params)
        if args["verbose"]: print("Generator created")

        if args["verbose"]: print("Running generation loop ...")

        print()
        print("Output: ", end='', flush=True)

        generation_successful = True

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                
                value_to_save = tokenizer_stream.decode(new_token)

                complete_text += value_to_save

                print(tokenizer_stream.decode(new_token), end='', flush=True)
                
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
            generation_successful = False
        except Exception as e:
            print(f"An error occurred: {e}")
            generation_successful = False

        with open(f'{model_dir}/output.txt', 'a', encoding='utf-8') as file:
            file.write(complete_text)

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    return generation_successful

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")
    parser.add_argument('-j', '--json', type=str, required=True, help='Path to the JSON file containing the arguments')
    args = parser.parse_args()

    with open(args.json, 'r') as file:
        args = json.load(file)

    os.makedirs(args["output_directory"], exist_ok=True)
    os.makedirs(args["cache_directory"], exist_ok=True)

    output = []

    validation_complete = False
    e = None
    exception = False

    for model_dict in args["models"]:

        print(f"We are validating {model_dict['name']}")
        adjusted_model = model_dict["name"].replace("/", "_")

        output_path = args["output_directory"] + f'/{adjusted_model}'
        cache_path = args["cache_directory"] + f'/{adjusted_model}'

        try:
            create_model(model_dict["name"], '', output_path, args["precision"], args["execution_provider"], cache_path)
        except Exception as e:
            print(f'Failure after create model {e}')
            output.append([model_dict["name"], validation_complete, e])
            exception = True
            continue
        try:          
            validation_complete = validate_model(args, model_dict, output_path)
        except Exception as e:
            print(f'Failure after validation model {e}')
            exception = True
            output.append([model_dict["name"], validation_complete, e]) 
        
        if not exception:
            output.append([model_dict["name"], validation_complete, e]) 
            
    df = create_table(output)

    df.to_csv("validation_summary.csv")
