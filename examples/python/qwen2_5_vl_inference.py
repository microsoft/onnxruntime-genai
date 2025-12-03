import winml
print(winml.register_execution_providers(ort=False, ort_genai=True))


import argparse
import json
import sys
from pathlib import Path

import onnxruntime_genai as og

TOOL_CALL_SYSTEM_PROMPT = """You are a web agent trying to complete user tasks on websites using function calls.

The functions at your disposal are:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer based on screenshots.\n- This is an interface to a web browser. You do not have access to a terminal or applications menu, only the browser.\n- Some pages, etc. may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click a home page icon and a window doesn't change, try wait and taking another screenshot.\n- Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n- If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n- Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\n- When a separate scrollable container prominently overlays the webpage, if you want to scroll within it, you typically need to mouse_move() over it first and then scroll().\nScreen resolution: 1428x896", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `key`: Press keyboard keys, like \"Enter\", \"Alt\", \"Shift\", \"Tab\", \"Control\", \"Backspace\", \"Delete\", \"Escape\", etc. Keys are pressed down in the order given, then released in reverse order.\n* `type`: Type a string of text on the keyboard.\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button.\n* `scroll`: Performs a scroll of the mouse scroll wheel.\n* `visit_url`: Visit a specified URL.\n* `web_search`: Perform a web search with a specified query.\n* `history_back`: Go back to the previous page in the browser history.\n* `pause_and_memorize_fact`: Pause and memorize a fact for future reference.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "scroll", "visit_url", "web_search", "history_back", "pause_and_memorize_fact", "wait", "terminate"], "type": "string"}, "keys": {"description": "Keyboard keys to be pressed in order. Required only by `action=key`.", "type": "array"}, "text": {"description": "Text to type. Required only by `action=type`.", "type": "string"}, "press_enter": {"description": "Whether to press the 'Enter' key after typing. Required only by `action=type`.", "type": "boolean"}, "delete_existing_text": {"description": "Whether to delete existing text before typing. Required only by `action=type`.", "type": "boolean"}, "coordinate": {"description": "[x, y]: The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`, `action=mouse_move`, and `action=type`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}, "url": {"description": "The URL to visit. Required only by `action=visit_url`.", "type": "string"}, "query": {"description": "The query to search for. Required only by `action=web_search`.", "type": "string"}, "fact": {"description": "The fact to remember for the future. Required only by `action=pause_and_memorize_fact`.", "type": "string"}, "time": {"description": "Number of seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

To make a function call, you should output a json object inside <tool_call></tool_call> XML tags. The json object must contain the function name and its arguments, like this:
<tool_call>
{\"name\": <function-name>, \"arguments\": <args-json-object>}
</tool_call>
"""

def run_inference(config_dir: Path, image_path: Path, prompt_text: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float,
                  do_sample: bool = False, min_length: int = 0, repetition_penalty: float = 1.0):
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load model and create multimodal processor (uses C++ Qwen2_5VLImageProcessor)
    model = og.Model(str(config_dir))
    
    tokenizer = og.Tokenizer(model)
    
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()
    
    # Load image using GenAI's image loader (internally uses onnxruntime-extensions)
    images = og.Images.open(str(image_path))
    
    # Build conversation with prompt
    conversation = [
        {"role": "system", "content": TOOL_CALL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    
    # Apply chat template to format the conversation
    message_json = json.dumps(conversation)
    prompt = tokenizer.apply_chat_template(message_json, add_generation_prompt=True)
    
    # Process prompt and images together
    # The C++ processor will automatically:
    # 1. Preprocess images using processor_config.json pipeline
    # 2. Insert image tokens in the correct places
    # 3. Return properly formatted inputs (pixel_values, image_grid_thw, input_ids)
    inputs = processor(prompt, images=images)

    if "input_ids" in inputs:
        input_ids_tensor = inputs["input_ids"]
        input_length = input_ids_tensor.shape()[1]
    else:
        input_length = len(tokenizer.encode(prompt))

    # Setup generation parameters
    try:
        with open(config_dir / "genai_config.json", "r") as f:
            config = json.load(f)
            context_len = config.get("model", {}).get("context_length", 2048)
            eos_val = config.get("model", {}).get("eos_token_id", [])
            eos_ids = eos_val if isinstance(eos_val, list) else [eos_val] if eos_val else []
    except Exception:
        context_len = 2048
        eos_ids = []
    
    max_length = min(input_length + max_new_tokens, context_len)
    
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p,
                              do_sample=do_sample, min_length=min_length, repetition_penalty=repetition_penalty)
    
    # Generate
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    output_tokens = []
    accum_text = ""
    started_toolcall = False
    print("\n=== Generating ===")
    
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        output_tokens.append(token)
        
        if eos_ids and token in eos_ids and len(output_tokens) >= min_length:
            break
        
        decoded = tokenizer_stream.decode(token)
        accum_text += decoded
        
        if not started_toolcall and "<tool_call>" in accum_text:
            started_toolcall = True
            sys.stdout.write(accum_text[accum_text.index("<tool_call>"):])
            sys.stdout.flush()
        elif started_toolcall:
            sys.stdout.write(decoded)
            sys.stdout.flush()
            if "</tool_call>" in accum_text:
                break
    
    print("\n=== Generation Complete ===")
    full_output = processor.decode(output_tokens)
    
    if started_toolcall and "</tool_call>" not in accum_text:
        print("[WARNING] Incomplete <tool_call> structure")
    
    print("\nFINAL OUTPUT:", full_output)
    return full_output


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL inference using onnxruntime-genai")
    parser.add_argument("--config_dir", "--model_path", type=Path, required=True, help="Directory with genai_config.json")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="User text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--min_length", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()

    run_inference(
        config_dir=args.config_dir,
        image_path=args.image,
        prompt_text=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        min_length=args.min_length,
        repetition_penalty=args.repetition_penalty,
    )

if __name__ == "__main__":
    main()
