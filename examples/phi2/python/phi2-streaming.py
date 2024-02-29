import onnxruntime_genai as og

print("Loading model...")

# The first argument is the name of the folder containing the model files
model=og.Model(f'example-models/phi2-int4-cpu', og.DeviceType.CPU)
print("Model loaded")
tokenizer=model.create_tokenizer()
print("Tokenizer created")

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

input_tokens = tokenizer.encode(prompt)

params=og.GeneratorParams(model)
params.set_search_options({"max_length":256})
params.input_ids = input_tokens

generator=og.Generator(model, params)
tokenizer_stream=tokenizer.create_stream()

print("Generator created")

print("Output:")
print(prompt, end='', flush=True)

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token_top_p(0.7, 0.6)
    print(tokenizer_stream.decode(generator.get_sequence(0).get_array()[-1]), end='', flush=True)
