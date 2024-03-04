import onnxruntime_genai as og
import time

print("Loading model...")

# The first argument is the name of the folder containing the model files
model=og.Model("example-models/llama2-7b-chat-int4-cpu", og.DeviceType.CPU)
print("Model loaded")
tokenizer=og.Tokenizer(model)
print("Tokenizer created")


# Keep asking for input prompts in an loop
while True:
    text = input("Input:")
    input_tokens = tokenizer.encode(text)

    params=og.GeneratorParams(model)
    params.set_search_options({"max_length":64})
    params.input_ids = input_tokens

    start_time=time.time()
    output_tokens=model.generate(params)[0]
    run_time=time.time()-start_time
    print(f"Tokens: {len(output_tokens)} Time: {run_time:.2f} Tokens per second: {len(output_tokens)/run_time:.2f}")

    print("Output:")
    print(tokenizer.decode(output_tokens))

    print()
    print()
