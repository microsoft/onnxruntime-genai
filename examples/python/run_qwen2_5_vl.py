#!/usr/bin/env python3
"""
Example script to run Qwen2.5-VL multimodal model with ONNX Runtime GenAI
"""

import argparse
import time

import onnxruntime_genai as og

# og.set_log_options(enabled=True, model_input_values=True, model_output_values=True)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL multimodal model")
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="Path to model directory containing genai_config.json"
    )
    parser.add_argument("-i", "--image_paths", type=str, nargs="+", required=True, help="One or more image file paths")
    parser.add_argument(
        "-p", "--prompt", type=str, default="Describe this image in detail.", help="Text prompt for the model"
    )
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "dml", "rocm"],
        help="Execution provider (default: cuda)",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum generation length (default: 2048)")

    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Using execution provider: {args.execution_provider}")

    # Validate image files exist
    for img_path in args.image_paths:
        import os

        if not os.path.exists(img_path):
            print(f"Error: Image file not found: {img_path}")
            return

    # Load model
    model = og.Model(args.model_path)
    print(f"Model type: {model.type}")

    # Create processor and tokenizer
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    stream = processor.create_stream()

    # Load images
    print(f"Loading {len(args.image_paths)} image(s)...")
    try:
        images = og.Images.open(*args.image_paths)
    except Exception as e:
        print(f"Error loading images: {e}")
        print("Please ensure the image files are valid JPEG, PNG, or other supported formats.")
        del stream
        del tokenizer
        del processor
        del model
        return

    # Prepare messages in Qwen2VL format
    # For Qwen2VL, we use structured content with image/text types
    content_list = []
    for _ in args.image_paths:
        content_list.append({"type": "image"})
    content_list.append({"type": "text", "text": args.prompt})

    messages = [{"role": "user", "content": content_list}]

    # Apply chat template to get formatted prompt with vision tokens
    import json

    message_json = json.dumps(messages)
    prompt = tokenizer.apply_chat_template(message_json, add_generation_prompt=True)
    print(f"\nPrompt: {args.prompt}")

    # Process images and prompt
    # Note: The prompt already contains vision tokens from the chat template
    print("Processing images and prompt...")
    try:
        inputs = processor(prompt, images=images)
    except Exception as e:
        print(f"Error processing inputs: {e}")
        import traceback

        traceback.print_exc()
        del images
        del stream
        del tokenizer
        del processor
        del model
        return

    # Create generator parameters
    print("\nGenerating response...")
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=args.max_length)

    # Disable graph capture for Qwen2.5-VL to avoid GPU tensor caching issues
    params.try_graph_capture_with_max_batch_size(0)

    # Create generator and set inputs
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    # Generate
    start_time = time.time()

    print("\nResponse: ", end="", flush=True)
    while not generator.is_done():
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        print(stream.decode(new_token), end="", flush=True)

    print()  # newline

    elapsed_time = time.time() - start_time
    total_tokens = len(generator.get_sequence(0))

    print(f"\n{'=' * 60}")
    print("Generation completed!")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens/sec: {total_tokens / elapsed_time:.2f}")
    print(f"{'=' * 60}")

    # Proper cleanup to avoid memory leaks
    del generator
    del inputs
    del images
    del stream
    del tokenizer
    del processor
    del model


if __name__ == "__main__":
    main()
