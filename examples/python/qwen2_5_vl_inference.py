import argparse
import json
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer

import onnxruntime_genai as og  # Requires built/installed onnxruntime-genai Python package

# ----------------------------------------------------------------------------
# Helper: build expanded image token sequence matching vision embeddings count
# ----------------------------------------------------------------------------
IMAGE_PAD_TOKEN = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYSTEM_PROMPT = "You are a helpful assistant."

# Image preprocessing constants (from Qwen2.5-VL config)
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
PATCH_SIZE = 14
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
MAX_RATIO = 200

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS):
    """Rescale image maintaining aspect ratio within pixel bounds."""
    import math
    
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"Aspect ratio must be smaller than {MAX_RATIO}")
    
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    
    return h_bar, w_bar

def load_prepatched_embeddings(image_path: Path, resize_width=800, resize_height=480):
    """Load and preprocess image into pre-patched format for vision pipeline."""
    img = Image.open(image_path).convert("RGB")
    patch_merge_size = PATCH_SIZE * MERGE_SIZE
    
    # Two-stage resize with factor constraint
    h1, w1 = smart_resize(resize_height, resize_width, factor=patch_merge_size)
    img = img.resize((w1, h1), Image.BICUBIC)
    h2, w2 = smart_resize(h1, w1, factor=patch_merge_size)
    img = img.resize((w2, h2), Image.BICUBIC)
    
    # Normalize with ImageNet stats
    pixel_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    pixel_array = (pixel_array - mean) / std
    
    # Convert to (B, C, H, W) format
    patches = np.array([pixel_array]).transpose(0, 3, 1, 2)
    
    # Pad temporal dimension if needed
    if patches.shape[0] % TEMPORAL_PATCH_SIZE != 0:
        pad_frames = np.repeat(patches[-1:], TEMPORAL_PATCH_SIZE - 1, axis=0)
        patches = np.concatenate([patches, pad_frames], axis=0)
    
    channel, grid_t = patches.shape[1], patches.shape[0] // TEMPORAL_PATCH_SIZE
    grid_h, grid_w = h2 // PATCH_SIZE, w2 // PATCH_SIZE
    
    # Reshape and flatten patches
    patches = patches.reshape(
        grid_t,
        TEMPORAL_PATCH_SIZE,
        channel,
        grid_h // MERGE_SIZE,
        MERGE_SIZE,
        PATCH_SIZE,
        grid_w // MERGE_SIZE,
        MERGE_SIZE,
        PATCH_SIZE,
    )
    
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, 
                                     channel * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)
    
    return flatten_patches[np.newaxis, :], np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)

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

def expand_image_tokens(grid_thw, merge_size):
    """Compute number of image pad tokens after spatial merging.
    Qwen2.5-VL vision pipeline produces one embedding per merged spatial cell.
    Token count = (t * h * w) / (merge_size ** 2)."""
    t, h, w = grid_thw
    merge_area = merge_size ** 2
    if (h * w) % merge_area != 0:
        raise ValueError(f"Grid (h={h}, w={w}) not divisible by merge_size^2={merge_area}")
    return (t * h * w) // merge_area


def build_prompt(user_text, num_image_tokens, use_tool_call_prompt=False):
    # Construct minimal chat-style prompt with expanded image pad tokens.
    image_tokens = IMAGE_PAD_TOKEN * num_image_tokens
    # Wrap in vision start/end markers once (matching template semantically) but repeated pad tokens inside.
    vision_block = f"{VISION_START}{image_tokens}{VISION_END}"
    system_text = TOOL_CALL_SYSTEM_PROMPT if use_tool_call_prompt else SYSTEM_PROMPT
    prompt = (
        f"{IM_START}system\n{system_text}{IM_END}\n"
        f"{IM_START}user\n{vision_block}{user_text}{IM_END}\n"
        f"{IM_START}assistant\n"
    )
    return prompt

def build_prompt_from_sample(sample_json_path: Path, use_tool_call_prompt=False, tokenizer=None):
    """Construct base prompt (single <|image_pad|>) from sample.json conversation using apply_chat_template."""
    with open(sample_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    convo = data.get("conversation", [])
    
    # Count images in the conversation
    image_count = 0
    for msg in convo:
        if msg.get("role") == "user":
            content_list = msg.get("content", [])
            for c in content_list:
                if isinstance(c, dict) and (c.get("type") == "image" or "image" in c):
                    image_count += 1
    
    if image_count == 0:
        raise ValueError("Sample JSON contained no image entries; cannot build vision prompt.")
    
    # Use apply_chat_template to match baseline behavior exactly
    if tokenizer is not None:
        prompt = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback to manual construction (shouldn't happen if tokenizer is passed)
        system_text = ""
        user_parts = []
        for msg in convo:
            role = msg.get("role")
            content_list = msg.get("content", [])
            if role == "system":
                system_parts = [c.get("text", "") for c in content_list if isinstance(c, dict)]
                system_text = "\n".join(system_parts)
            elif role == "user":
                for c in content_list:
                    if isinstance(c, dict):
                        if c.get("type") == "image" or "image" in c:
                            user_parts.append(f"{VISION_START}{IMAGE_PAD_TOKEN}{VISION_END}")
                        elif c.get("type") == "text":
                            user_parts.append(c.get("text", ""))
        user_text = "".join(user_parts)
        prompt = (
            f"{IM_START}system\n{system_text}{IM_END}\n"
            f"{IM_START}user\n{user_text}{IM_END}\n"
            f"{IM_START}assistant\n"
        )
    
    return prompt, image_count

def expand_image_tokens_in_prompt(base_prompt: str, image_grid_thw, merge_size: int):
    """Expand single <|image_pad|> placeholder to multiple tokens."""
    t, h, w = image_grid_thw
    num_tokens = (t * h * w) // (merge_size ** 2)
    return base_prompt.replace(IMAGE_PAD_TOKEN, IMAGE_PAD_TOKEN * num_tokens, 1), num_tokens


def run_inference(config_dir: Path, image_path: Path, prompt_text: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float, sample_dir: Path | None = None,
                  enable_qnn: bool = False, qnn_backend_path: str | None = None, qnn_provider_library: str | None = None,
                  do_sample: bool = False, min_length: int = 0, repetition_penalty: float = 1.0, tool_call_prompt: bool = False):
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    pixel_values, grid_thw_array = load_prepatched_embeddings(image_path)
    grid_thw = grid_thw_array[0]
    
    # Load model with optional QNN acceleration
    if enable_qnn:
        if qnn_provider_library:
            og.register_execution_provider_library("QNN", qnn_provider_library)
        cfg = og.Config(str(config_dir / "genai_config.json" if (config_dir / "genai_config.json").is_file() else config_dir))
        cfg.append_provider("QNN")
        cfg.append_provider("CPUExecutionProvider")
        if qnn_backend_path:
            cfg.set_provider_option("QNN", "backend_path", qnn_backend_path)
        cfg.set_provider_option("QNN", "performance_mode", "3")
        model = og.Model(cfg)
    else:
        model = og.Model(str(config_dir))
    
    tokenizer_hf = AutoTokenizer.from_pretrained(str(config_dir), trust_remote_code=True)
    tokenizer_ort = og.Tokenizer(model)

    if sample_dir:
        base_prompt, _ = build_prompt_from_sample(sample_dir / "sample.json", tool_call_prompt, tokenizer_hf)
        prompt, num_image_tokens = expand_image_tokens_in_prompt(base_prompt, grid_thw, MERGE_SIZE)
    else:
        num_image_tokens = expand_image_tokens(grid_thw, MERGE_SIZE)
        prompt = build_prompt(prompt_text, num_image_tokens, tool_call_prompt)

    input_ids_np = np.array(tokenizer_hf.encode(prompt), dtype=np.int32)

    params = og.GeneratorParams(model)
    try:
        with open(config_dir / "genai_config.json", "r") as f:
            context_len = json.load(f).get("model", {}).get("context_length", input_ids_np.shape[0] + max_new_tokens)
    except Exception:
        context_len = input_ids_np.shape[0] + max_new_tokens
    
    params.set_search_options(max_length=context_len, temperature=temperature, top_k=top_k, top_p=top_p,
                              do_sample=do_sample, min_length=min_length, repetition_penalty=repetition_penalty)

    generator = og.Generator(model, params)
    generator.set_model_input("pixel_values", np.ascontiguousarray(pixel_values.astype(np.float32)))
    generator.set_model_input("image_grid_thw", np.ascontiguousarray(grid_thw_array.astype(np.int64)))
    generator.append_tokens(input_ids_np)

    stream = tokenizer_ort.create_stream()
    output_tokens = []
    print("\n=== Generating ===")
    
    try:
        with open(config_dir / "genai_config.json", "r") as f:
            eos_val = json.load(f).get("model", {}).get("eos_token_id", [])
            eos_ids = eos_val if isinstance(eos_val, list) else [eos_val] if eos_val else []
    except Exception:
        eos_ids = []
    
    accum_text, started_toolcall, closed_toolcall, step = "", False, False, 0
    
    while not generator.is_done() and step < max_new_tokens:
        generator.generate_next_token()
        new_tok = generator.get_next_tokens()[0]
        output_tokens.append(new_tok)
        
        if eos_ids and new_tok in eos_ids and step >= min_length:
            break
        
        decoded_piece = stream.decode(new_tok)
        
        if tool_call_prompt:
            accum_text += decoded_piece
            if not started_toolcall and "<tool_call>" in accum_text:
                started_toolcall = True
                sys.stdout.write(accum_text[accum_text.index("<tool_call>"):])
                sys.stdout.flush()
            elif started_toolcall:
                sys.stdout.write(decoded_piece)
                sys.stdout.flush()
                if "</tool_call>" in accum_text:
                    closed_toolcall = True
                    break
        elif decoded_piece:
            sys.stdout.write(decoded_piece)
            sys.stdout.flush()
        
        step += 1
    
    print("\n=== Generation Complete ===")
    full_output = tokenizer_hf.decode(np.array(output_tokens, dtype=np.int32))
    
    if tool_call_prompt and not closed_toolcall:
        print("\n[WARNING] Model did not emit complete <tool_call> structure.")
    
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
    parser.add_argument("--sample_dir", type=Path, help="Sample directory with sample.json")
    parser.add_argument("--enable_qnn", action="store_true")
    parser.add_argument("--qnn_backend_path", type=str, default="QnnHtp.dll")
    parser.add_argument("--qnn_provider_library", type=str)
    parser.add_argument("--tool_call_prompt", action="store_true")
    args = parser.parse_args()

    run_inference(
        config_dir=args.config_dir,
        image_path=args.image,
        prompt_text=args.prompt if not args.sample_dir else "",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        min_length=args.min_length,
        repetition_penalty=args.repetition_penalty,
        sample_dir=args.sample_dir,
        enable_qnn=args.enable_qnn,
        qnn_backend_path=args.qnn_backend_path if args.enable_qnn else None,
        qnn_provider_library=args.qnn_provider_library,
        tool_call_prompt=args.tool_call_prompt,
    )

if __name__ == "__main__":
    main()
