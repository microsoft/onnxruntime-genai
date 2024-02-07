import argparse
import time
import onnxruntime_genai as og

print(f"Loading model... ")
model=og.Model(f'model', og.DeviceType.CPU)
print("Model loaded")

tokenizer = model.create_tokenizer()

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

tokens = tokenizer.encode(prompt)

print(tokens)

params=og.search_params(model)
params.max_length = 200
params.input_ids = tokens

start_time=time.time()
output_tokens=model.generate(params)
run_time=time.time()-start_time

print(f"Tokens: {len(output_tokens)} Time: {run_time:.2f} Tokens per second: {len(output_tokens)/run_time:.2f}")

text = tokenizer.decode(output_tokens)

print("Output:")
print(text)