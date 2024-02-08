import onnxruntime_genai as og
from transformers import LlamaTokenizer

print("Loading model...")
model=og.Model("model", og.DeviceType.CPU)
print("Model loaded")
tokenizer=model.create_tokenizer()
print("Tokenizer created")

prompt = "I like walking my cute dog"

input_tokens = tokenizer.encode(prompt)

params=og.SearchParams(model)
params.max_length = 256
params.input_ids = input_tokens

generator=og.Generator(model, params)

print("Prompt:")
print(prompt, end='', flush=True)

while not generator.is_done():
    generator.compute_logits()

    # search.apply_minLength(1)
    # search.apply_repetition_penalty(1.0)

    generator.generate_next_token_top_p(0.9, 1.0)

print("Output:")

print(tokenizer.decode(generator.get_sequence(0).GetArray()))
    
print()
print()