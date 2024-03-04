﻿import onnxruntime_genai as og

print("Loading model...")

# The first argument is the name of the folder containing the model files
model=og.Model("./example-models/gemma-2b-cuda", og.DeviceType.CUDA)
print("Model loaded")
tokenizer=og.Tokenizer(model)
print("Tokenizer created")

prompts = ["I like walking my cute dog",
           "What is the best restaurant in town?",
           "Hello, how are you today?"]

input_tokens = tokenizer.encode_batch(prompts)

params=og.GeneratorParams(model)
params.set_search_options({"max_length":64})
params.input_ids=input_tokens


generator=og.Generator(model, params)

print("Generator created")

print("Running generation loop ...")

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token_top()

print("Outputs:")

for i in range(len(prompts)):
   print(tokenizer.decode(generator.get_sequence(i)))
   print()
