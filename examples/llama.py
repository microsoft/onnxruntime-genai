import onnxruntime_genai as og
from transformers import LlamaTokenizer

# device_type = og.DeviceType.CPU
device_type = og.DeviceType.CUDA

# Generate input tokens from the text prompt

print("Loading model...")
# model=og.Model("../../test_models/llama2-7b-fp32-cpu", device_type)
#model=og.Llama_Model("../../test_models/llama2-7b-fp16-gpu/rank_0_Llama-2-7b-hf_decoder_merged_model_fp16.onnx", device_type)
#model=og.Llama_Model("../../test_models/llama2-7b-int4-gpu/rank_0_Llama-2-7b-hf_decoder_merged_model_int4.onnx", device_type)
model=og.Model("../test_models/llama2-7b-chat-int4-gpu", device_type)
print("Model loaded")
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
# tokenizer=model.CreateTokenizer()
print("Tokenizer created")

# Keep asking for input prompts in an loop
while True:
    text = input("Input:")
    input_tokens = tokenizer.encode(text, return_tensors='np')
    # input_tokens = tokenizer.encode(text)

    params=og.SearchParams(model)
    params.max_length = 128
    params.input_ids = input_tokens

    generator=og.Generator(model, params)

    print("Output:")

    print(text, end='', flush=True)
    while not generator.IsDone():
        generator.ComputeLogits()

        # search.Apply_MinLength(1)
        # search.Apply_RepetitionPenalty(1.0)

        generator.AppendNextToken_TopP(0.7, 0.6)

        # print(tokenizer.decode([generator.GetNextTokens().GetArray()[0]]), ' ', end='', flush=True)
        # Print each token as we compute it, we have to do some work to get newlines & spaces to appear properly:
        word=tokenizer.convert_ids_to_tokens([generator.GetNextTokens().GetArray()[0]])[0]
        if word=='<0x0A>':
          word = '\n'
        if word.startswith('▁'):
          word = ' ' + word[1:]
        print(word, end='', flush=True)

    # Print sequence all at once vs as it's decoded:
    print(tokenizer.decode(generator.GetSequence(0).GetArray()))
    print()
    print()
