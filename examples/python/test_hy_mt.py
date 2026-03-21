import onnxruntime_genai as og
from transformers import AutoTokenizer

model_path = "onnx_HY-MT1.5-1.8B"
hf_model_path = "pytorch_HY-MT1.5-1.8B"

print("Loading model...")
model = og.Model(model_path)

# Use HF tokenizer for encoding (og.Tokenizer has issues with HY-MT special tokens)
hf_tok = AutoTokenizer.from_pretrained(hf_model_path)
tokenizer_stream = og.Tokenizer(model).create_stream()

messages = [
    {"role": "system", "content": "You are a helpful translation assistant."},
    {"role": "user", "content": "Translate to French: Hello, how are you today?"},
]
prompt = hf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_tokens = hf_tok.encode(prompt)

print(f"Prompt: {prompt!r}")
print("Response: ", end="", flush=True)

params = og.GeneratorParams(model)
params.set_search_options(max_length=200, temperature=0.7, top_k=20, top_p=0.6, repetition_penalty=1.05)

generator = og.Generator(model, params)
generator.append_tokens(input_tokens)
while not generator.is_done():
    generator.generate_next_token()
    token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(token), end="", flush=True)

print("\nDone.")
