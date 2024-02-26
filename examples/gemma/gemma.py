import onnxruntime_genai as og
import time

print("Loading model...")
model=og.Model("./example-models/gemma-2b-cuda", og.DeviceType.CUDA)
print("Model loaded")
tokenizer=model.create_tokenizer()
print("Tokenizer created")


# Keep asking for input prompts in an loop
while True:
    text = input("Input:")
    input_tokens = tokenizer.encode(text)

    params=og.GeneratorParams(model)
    params.max_length = 64
    params.input_ids = input_tokens

    start_time=time.time()
    output_tokens=model.generate(params)[0]
    run_time=time.time()-start_time
    print(f"Tokens: {len(output_tokens)}, Time: {run_time:.2f}, Tokens per second: {len(output_tokens)/run_time:.2f}")

    print("Output:")
    print(tokenizer.decode(output_tokens))

    print()
    print()
