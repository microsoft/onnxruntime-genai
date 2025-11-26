#!/usr/bin/env python3
"""
Example script to run Qwen2.5-VL multimodal model with ONNX Runtime GenAI
"""

import argparse
import json
import time
from pathlib import Path

import textwrap

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
    default_system_prompt = textwrap.dedent(
        """
        You are a web agent trying to complete user tasks on websites using function calls.

        The functions at your disposal are:
        <tools>
        {"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer based on screenshots.\n- This is an interface to a web browser. You do not have access to a terminal or applications menu, only the browser.\n- Some pages, etc. may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click a home page icon and a window doesn't change, try wait and taking another screenshot.\n- Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n- If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n- Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\n- When a separate scrollable container prominently overlays the webpage, if you want to scroll within it, you typically need to mouse_move() over it first and then scroll().\nScreen resolution: 1428x896", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `key`: Press keyboard keys, like \"Enter\", \"Alt\", \"Shift\", \"Tab\", \"Control\", \"Backspace\", \"Delete\", \"Escape\", etc. Keys are pressed down in the order given, then released in reverse order.\n* `type`: Type a string of text on the keyboard.\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button.\n* `scroll`: Performs a scroll of the mouse scroll wheel.\n* `visit_url`: Visit a specified URL.\n* `web_search`: Perform a web search with a specified query.\n* `history_back`: Go back to the previous page in the browser history.\n* `pause_and_memorize_fact`: Pause and memorize a fact for future reference.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "scroll", "visit_url", "web_search", "history_back", "pause_and_memorize_fact", "wait", "terminate"], "type": "string"}, "keys": {"description": "Keyboard keys to be pressed in order. Required only by `action=key`.", "type": "array"}, "text": {"description": "Text to type. Required only by `action=type`.", "type": "string"}, "press_enter": {"description": "Whether to press the 'Enter' key after typing. Required only by `action=type`.", "type": "boolean"}, "delete_existing_text": {"description": "Whether to delete existing text before typing. Required only by `action=type`.", "type": "boolean"}, "coordinate": {"description": "[x, y]: The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`, `action=mouse_move`, and `action=type`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}, "url": {"description": "The URL to visit. Required only by `action=visit_url`.", "type": "string"}, "query": {"description": "The query to search for. Required only by `action=web_search`.", "type": "string"}, "fact": {"description": "The fact to remember for the future. Required only by `action=pause_and_memorize_fact`.", "type": "string"}, "time": {"description": "Number of seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
        </tools>

        To make a function call, you should output a json object inside <tool_call></tool_call> XML tags. The json object must contain the function name and its arguments, like this:
        <tool_call>
        {"name": <function-name>, "arguments": <args-json-object>}
        </tool_call>
        """
    ).strip()

    parser.add_argument(
        "-s",
        "--system_prompt",
        type=str,
        default=default_system_prompt,
        help="System prompt (optional, for role-based prompting)",
    )
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        default="follow_config",
        choices=["follow_config", "cpu", "cuda", "dml", "rocm", "qnn"],
        help="Override execution provider (default: follow_config, use providers defined in genai_config.json)",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum generation length (default: 2048)")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling instead of greedy decoding")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (default: 0.7)")

    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Using execution provider: {args.execution_provider}")

    # Validate image files exist
    for img_path in args.image_paths:
        import os

        if not os.path.exists(img_path):
            print(f"Error: Image file not found: {img_path}")
            return

    # Load model via config so folder paths (containing genai_config.json) resolve correctly
    print("Loading model = ", args.model_path)
    model_dir = Path(args.model_path).expanduser().resolve()
    if not model_dir.exists():
        print(f"Error: Model directory does not exist: {model_dir}")
        return

    config_file = model_dir / "genai_config.json"
    if not config_file.exists():
        print(f"Error: genai_config.json not found at: {config_file}")
        print("Directory contents:")
        for child in sorted(model_dir.iterdir()):
            print(f"  - {child.name}")
        return

    with config_file.open("r", encoding="utf-8") as f:
        config_json = json.load(f)

    eos_token_ids = config_json.get("model", {}).get("eos_token_id", [])
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    config = og.Config(str(model_dir))
    if args.execution_provider != "follow_config":
        config.clear_providers()
        config.append_provider(args.execution_provider)
    try:
        model = og.Model(config)
    except RuntimeError as err:
        print(f"Error: Failed to create model from {model_dir}\n{err}")
        print("Directory contents:")
        for child in sorted(model_dir.iterdir()):
            print(f"  - {child.name}")
        raise
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

    # Prepare messages in Qwen2VL format (similar to Fara demo)
    messages = []
    
    # Add system prompt if provided (like Fara demo's web agent system prompt)
    if args.system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": args.system_prompt}]
        })
    
    # Build user message with images and text
    content_list = []
    for _ in args.image_paths:
        content_list.append({"type": "image"})
    content_list.append({"type": "text", "text": args.prompt})

    messages.append({"role": "user", "content": content_list})

    # Apply chat template to get formatted prompt with vision tokens
    message_json = json.dumps(messages)
    prompt = tokenizer.apply_chat_template(message_json, add_generation_prompt=True)
    
    print(f"\n=== User Query ===")
    print(args.prompt)
    if args.system_prompt:
        print(f"\nSystem Prompt: {args.system_prompt[:100]}..." if len(args.system_prompt) > 100 else args.system_prompt)

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
    params.set_search_options(
        max_length=args.max_length,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    # Disable graph capture for Qwen2.5-VL to avoid GPU tensor caching issues
    params.try_graph_capture_with_max_batch_size(0)

    # Create generator and set inputs
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    # Generate
    start_time = time.time()

    print("\n=== Answer ===")
    while not generator.is_done():
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        print(stream.decode(new_token), end="", flush=True)

        if eos_token_ids and new_token in eos_token_ids:
            break
        current_length = len(generator.get_sequence(0))
        if current_length >= args.max_length:
            break

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
