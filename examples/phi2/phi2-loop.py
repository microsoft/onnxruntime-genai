import onnxruntime_genai as og

print("Loading model...")
model=og.Model("model", og.DeviceType.CPU)
print("Model loaded")
tokenizer=model.create_tokenizer()
print("Tokenizer created")

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

input_tokens = tokenizer.encode(prompt)

params=og.SearchParams(model)
params.max_length = 256
params.input_ids = input_tokens

generator=og.Generator(model, params)

print("Output:")

print(prompt, end='', flush=True)


while not generator.is_done():
    generator.compute_logits()

    # search.apply_minLength(1)
    # search.apply_repetition_penalty(1.0)

    generator.generate_next_token_topp(0.7, 0.6)

    print(tokenizer.decode([generator.get_next_tokens().GetArray()[0]]), ' ', end='', flush=True)

    # Print sequence all at once vs as it's decoded:
    print(tokenizer.decode(generator.get_sequence(0).GetArray()))
    
    print()
    print()
