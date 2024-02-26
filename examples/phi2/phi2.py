import time
import onnxruntime_genai as og

print(f"Loading model... ")

# The first argument is the name of the folder containing the model files
model=og.Model(f'example-models/phi2-int4-cpu', og.DeviceType.CPU)
print("Model loaded")

tokenizer = model.create_tokenizer()

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

tokens = tokenizer.encode(prompt)

params=og.GeneratorParams(model)
params.max_length = 200
params.input_ids = tokens

start_time=time.time()
output_tokens=model.generate(params)[0]
run_time=time.time()-start_time

print(f"Tokens: {len(output_tokens)} Time: {run_time:.2f} Tokens per second: {len(output_tokens)/run_time:.2f}")

text = tokenizer.decode(output_tokens)

print("Output:")
print(text)