# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import glob
import json
import os
import readline
import time
from pathlib import Path

import onnxruntime_genai as og

# og.set_log_options(enabled=True, model_input_values=True, model_output_values=True)

# Tool-calling system prompt for Qwen/Fara models
FARA_SYSTEM_PROMPT = """You are a web agent trying to complete user tasks on websites using function calls.

The functions at your disposal are:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer based on screenshots.\\n- This is an interface to a web browser. You do not have access to a terminal or applications menu, only the browser.\\n- Some pages, etc. may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click a home page icon and a window doesn't change, try wait and taking another screenshot.\\n- Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n- If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n- Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\\n- When a separate scrollable container prominently overlays the webpage, if you want to scroll within it, you typically need to mouse_move() over it first and then scroll().\\nScreen resolution: 1428x896", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `key`: Press keyboard keys, like \\"Enter\\", \\"Alt\\", \\"Shift\\", \\"Tab\\", \\"Control\\", \\"Backspace\\", \\"Delete\\", \\"Escape\\", etc. Keys are pressed down in the order given, then released in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `visit_url`: Visit a specified URL.\\n* `web_search`: Perform a web search with a specified query.\\n* `history_back`: Go back to the previous page in the browser history.\\n* `pause_and_memorize_fact`: Pause and memorize a fact for future reference.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "scroll", "visit_url", "web_search", "history_back", "pause_and_memorize_fact", "wait", "terminate"], "type": "string"}, "keys": {"description": "Keyboard keys to be pressed in order. Required only by `action=key`.", "type": "array"}, "text": {"description": "Text to type. Required only by `action=type`.", "type": "string"}, "press_enter": {"description": "Whether to press the 'Enter' key after typing. Required only by `action=type`.", "type": "boolean"}, "delete_existing_text": {"description": "Whether to delete existing text before typing. Required only by `action=type`.", "type": "boolean"}, "coordinate": {"description": "[x, y]: The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`, `action=mouse_move`, and `action=type`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}, "url": {"description": "The URL to visit. Required only by `action=visit_url`.", "type": "string"}, "query": {"description": "The query to search for. Required only by `action=web_search`.", "type": "string"}, "fact": {"description": "The fact to remember for the future. Required only by `action=pause_and_memorize_fact`.", "type": "string"}, "time": {"description": "Number of seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

To make a function call, you should output a json object inside <tool_call></tool_call> XML tags. The json object must contain the function name and its arguments, like this:
<tool_call>
{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}
</tool_call>
"""


def _find_dir_contains_sub_dir(current_dir: Path, target_dir_name):
    curr_path = Path(current_dir).absolute()
    target_dir = glob.glob(target_dir_name, root_dir=curr_path)
    if target_dir:
        return Path(curr_path / target_dir[0]).absolute()
    else:
        if curr_path.parent == curr_path:
            # Root dir
            return None
        return _find_dir_contains_sub_dir(curr_path / "..", target_dir_name)


def _complete(text, state):
    return [*glob.glob(text + "*"), None][state]


def run(args: argparse.Namespace):
    if args.use_winml:
        try:
            import winml

            print(winml.register_execution_providers(ort=False, ort_genai=True))
        except ImportError:
            print("WinML not available, using default execution providers")
        except Exception as e:
            print(f"Failed to register WinML execution providers: {e}")

    print("Loading model...")
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            print(f"Setting model to {args.execution_provider}...")
            config.append_provider(args.execution_provider)
    model = og.Model(config)
    print("Model loaded")

    tokenizer = og.Tokenizer(model)
    processor = model.create_multimodal_processor()
    stream = processor.create_stream()

    interactive = not args.non_interactive

    while True:
        if interactive:
            try:
                readline.set_completer_delims(" \t\n;")
                readline.parse_and_bind("tab: complete")
                readline.set_completer(_complete)
            except ImportError:
                # Not available on some platforms. Ignore it.
                pass
            image_paths = [
                image_path.strip()
                for image_path in input("Image Path (comma separated; leave empty if no image): ").split(",")
            ]
        else:
            if args.image_paths:
                image_paths = args.image_paths
            else:
                image_paths = [
                    str(
                        _find_dir_contains_sub_dir(Path(__file__).parent, "test")
                        / "test_models"
                        / "images"
                        / "australia.jpg"
                    )
                ]

        image_paths = [image_path for image_path in image_paths if image_path]

        images = None
        if len(image_paths) == 0:
            print("No image provided")
        else:
            for _, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                print(f"Using image: {image_path}")

            images = og.Images.open(*image_paths)

        if interactive:
            text = input("Prompt: ")
        else:
            if args.prompt:
                text = args.prompt
            else:
                text = "What is shown in this image?"

        # Construct the "messages" argument passed to apply_chat_template
        messages = []
        if model.type == "phi3v":
            # Combine all image tags and text into one user message
            content = "".join([f"<|image_{i + 1}|>\n" for i in range(len(image_paths))]) + text
            messages.append({"role": "user", "content": content})
        elif model.type in ["qwen2_5_vl", "fara"]:
            messages.append({"role": "system", "content": FARA_SYSTEM_PROMPT})
            content = "".join(["<|vision_start|><|image_pad|><|vision_end|>" for _ in image_paths]) + text
            messages.append({"role": "user", "content": content})
        else:
            # Gemma3-style multimodal: structured content
            content_list = [{"type": "image"} for _ in image_paths]
            content_list.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content_list})

        # Apply the chat template using the tokenizer
        message_json = json.dumps(messages)
        prompt = tokenizer.apply_chat_template(message_json, add_generation_prompt=True)

        print("Processing images and prompt...")
        inputs = processor(prompt, images=images)

        print("Generating response...")
        params = og.GeneratorParams(model)
        max_length = args.max_length if args.max_length else 7680
        params.set_search_options(max_length=max_length)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        start_time = time.time()

        while True:
            generator.generate_next_token()
            if generator.is_done():
                break

            new_token = generator.get_next_tokens()[0]
            print(stream.decode(new_token), end="", flush=True)

        print()
        total_run_time = time.time() - start_time
        print(f"Total Time : {total_run_time:.2f}")

        for _ in range(3):
            print()

        # Delete the generator to free the captured graph before creating another one
        del generator

        if not interactive:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the folder containing the model")
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        required=False,
        default="follow_config",
        choices=["cpu", "cuda", "dml", "follow_config"],
        help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead.",
    )
    parser.add_argument(
        "--image_paths", nargs="*", type=str, required=False, help="Path to the images, mainly for CI usage"
    )
    parser.add_argument(
        "-pr", "--prompt", required=False, help="Input prompts to generate tokens from, mainly for CI usage"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=False,
        default=None,
        help="Maximum generation length. Defaults to model's context_length from config.",
    )
    parser.add_argument(
        "--non-interactive",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="Non-interactive mode, mainly for CI usage",
    )
    parser.add_argument(
        "--use-winml",
        action="store_true",
        required=False,
        help="Register WinML execution providers before loading the model",
    )
    args = parser.parse_args()
    run(args)
