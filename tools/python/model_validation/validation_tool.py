import onnxruntime_genai as og
import argparse
from onnxruntime_genai.models.builder import create_model
import json
import os
import pandas as pd

def create_table(output):
    df = pd.DataFrame(output, columns=['Model Name', 'Validation Completed', 'Exceptions / Failures'])
    return df

def validate_model(config, model_directory):
    if config["verbose"]: print("Loading model...")

    model = og.Model(f'{model_directory}')

    if config["verbose"]: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream() 

    if config["verbose"]: print("Tokenizer created")
    if config["verbose"]: print()  

    chat_template = get_chat_template(model_directory.lower())

    search_options = config["search_options"]

    for text in config["inputs"]:

        complete_text = ''
        
        prompt = f'{chat_template.format(input=text)}'

        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens

        generator = og.Generator(model, params)
        if config["verbose"]: print("Generator created")

        if config["verbose"]: print("Running generation loop ...")

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

                # print(tokenizer_stream.decode(new_token), end='', flush=True)
                
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
            generation_successful = False
        except Exception as e:
            print(f"An error occurred: {e}")
            generation_successful = False

        with open(f'{model_directory}/output.txt', 'a', encoding='utf-8') as file:
            file.write(complete_text)

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    return generation_successful

def get_chat_template(model_name):
    if 'phi' in model_name: 
        return '<|user|>\n{input} <|end|>\n<|assistant|>'
    elif 'qwen' in model_name: 
        return '<s>\n<|user|>\n{input} <|end|>\n<|assistant|>'
    elif 'mistral' in model_name: 
        return '<|im_start|> <|user|> \n {input} <|im_end>|\n'
    elif model_name.contains("llama"): 
        return '<s>[INST]<<SYS>>\n{input}<</SYS>>[INST]'
    elif model_name.contains("gemma"): 
        return '<start_of_turn>' + 'user' '\n' + {input}  + '<end_of_turn>\n'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")

    parser.add_argument('-j', '--json', type=str, required=True, help='Path to the JSON file containing the arguments')

    args = parser.parse_args()

    with open(args.json, 'r') as file:
        config = json.load(file)

    os.makedirs(config["output_directory"], exist_ok=True)
    os.makedirs(config["cache_directory"], exist_ok=True)

    output = []

    validation_complete = False
    e = None
    exception = False

    for model in config["models"]:

        print(f"We are validating {model}")
        adjusted_model = model.replace("/", "_")
        output_path = config["output_directory"] + f'/{adjusted_model}'
        cache_path = config["cache_directory"] + f'/{adjusted_model}'
        try:
            create_model(model, '', output_path, config["precision"], config["executive_provider"], cache_path)
        except Exception as e:
            print(f'Failure after create model {e}')
            output.append([model, validation_complete, e])
            exception = True
            continue
        try:          
            validation_complete = validate_model(config, output_path)
        except Exception as e:
            print(f'Failure after validation model {e}')
            exception = True
            output.append([model, validation_complete, e]) 
        
        if not exception:
            output.append([model, validation_complete, e]) 
            
    df = create_table(output)

    df.to_csv("validation_summary.csv")
