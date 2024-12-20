import argparse
import os

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from onnxruntime_genai.models.builder import create_model
import onnxruntime_genai as og

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="Folder to load PyTorch model and associated files from",
    )

    parser.add_argument(
        "-q",
        "--quant_path",
        required=True,
        help="Folder to save AWQ-quantized PyTorch model and associated files in",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Folder to save AWQ-quantized ONNX model and associated files in",
    )

    parser.add_argument(
        "-e",
        "--execution_provider",
        default="cuda",
        choices=["cpu", "cuda", "dml"],
        help="Target execution provider to apply quantization (e.g. dml, cuda)",
    )

    parser.add_argument(
        "--use_qdq",
        action="store_true",
        help="Use the QDQ decomposition for quantized MatMul instead of the MatMulNBits operator",
    )

    args = parser.parse_args()
    return args

def quantize_model(args):
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Quantize model
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(args.quant_path)
    tokenizer.save_pretrained(args.quant_path)

    print(f'Model is quantized and saved at "{args.quant_path}"')

def run_model(args):
    # Load model
    print("Loading model...")
    config = og.Config(args.output_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        print(f"Setting model to {args.execution_provider}")
        config.append_provider(args.execution_provider)
    model = og.Model(config)
    print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    # Override any default search options in `genai_config.json`
    search_options = {
        'min_length': 1,
        'max_length': 2048,
    }

    # Chat template for Phi-3 (replace with the chat template for your model)
    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
    while True:
        text = input("Input: ")
        if not text:
            print("Error, input cannot be empty")
            continue
        prompt = f'{chat_template.format(input=text)}'

        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)

        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)

        print()
        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

def main():
    args = parse_args()

    # Quantize PyTorch model
    quantize_model(args)

    # Create ONNX model
    model_name = None
    input_folder = args.quant_path
    output_folder = args.output_path
    precision = "int4"
    execution_provider = args.execution_provider
    cache_dir = os.path.join(".", "cache_dir")

    extra_options = {
        "use_qdq": args.use_qdq,
    }

    create_model(model_name, input_folder, output_folder, precision, execution_provider, cache_dir, **extra_options)

    if args.execution_provider == "dml":
        if og.__id__ != "onnxruntime-genai-directml":
            raise ValueError(f"onnxruntime-genai-directml is required to be installed. Please uninstall all ORT GenAI packages with `pip uninstall -y onnxruntime-genai onnxruntime-genai-cuda onnxruntime-genai-directml` and only install the DML version with `pip install onnxruntime-genai-directml`.")
    # Run ONNX model
    run_model(args)

if __name__ == "__main__":
    main()
