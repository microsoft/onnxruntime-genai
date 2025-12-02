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
MIN_PIXELS = 4 * 28 * 28
MAX_RATIO = 200

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS):
    """Baseline's smart_resize logic - rescales image maintaining aspect ratio."""
    import math
    
    def ceil_by_factor(number, factor):
        return math.ceil(number / factor) * factor
    
    def floor_by_factor(number, factor):
        return math.floor(number / factor) * factor
    
    def round_by_factor(number, factor):
        return round(number / factor) * factor
    
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"Aspect ratio must be smaller than {MAX_RATIO}")
    
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar

def load_prepatched_embeddings(image_path: Path, resize_width=800, resize_height=480):
    """Load image and convert to pre-patched embeddings format matching baseline.
    
    This matches the baseline's approach: manually patch the image in Python
    before passing to the ONNX vision pipeline. The patch_embed model expects
    pre-patched data (1, num_patches, patch_dim), NOT raw pixels (B, C, H, W).
    
    Args:
        image_path: Path to image file
        resize_width: Target width for first resize (default 800)
        resize_height: Target height for first resize (default 480)
    
    Returns:
        pixel_values: np.ndarray of shape (1, num_patches, patch_dim)
        grid_thw: (t, h, w) grid dimensions after patching
    """
    # Load and convert to RGB
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    
    # Two-stage resize matching baseline:
    # 1. First resize to target dimensions with factor=28 constraint
    patch_merge_size = PATCH_SIZE * MERGE_SIZE  # 14 * 2 = 28
    h1, w1 = smart_resize(resize_height, resize_width, factor=patch_merge_size, 
                         min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    img = img.resize((w1, h1), Image.BICUBIC)
    
    # 2. Second smart_resize with same constraints (matches baseline fetch_image_data)
    h2, w2 = smart_resize(h1, w1, factor=patch_merge_size,
                         min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    img = img.resize((w2, h2), Image.BICUBIC)
    
    print(f"[INFO] Resized image: {orig_w}x{orig_h} -> {w1}x{h1} -> {w2}x{h2}")
    
    # Convert to numpy array (H, W, C) and normalize to [0, 1]
    pixel_array = np.array(img).astype(np.float32) / 255.0
    
    # Apply ImageNet normalization (from Qwen2.5-VL processor config)
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    pixel_array = (pixel_array - mean) / std
    
    # --- Patching logic from baseline image_utils.patch_image ---
    # Start with (H, W, C) format, add batch dimension
    patches = np.array([pixel_array])  # shape: (1, H, W, C)
    
    # Convert to (B, C, H, W) format
    patches = patches.transpose(0, 3, 1, 2)  # shape: (1, C, H, W)
    
    # Handle temporal dimension (for video, but we use single frame)
    if patches.shape[0] % TEMPORAL_PATCH_SIZE != 0:
        repeats = np.repeat(patches[-1][np.newaxis], TEMPORAL_PATCH_SIZE - 1, axis=0)
        patches = np.concatenate([patches, repeats], axis=0)
    
    channel = patches.shape[1]
    grid_t = patches.shape[0] // TEMPORAL_PATCH_SIZE
    grid_h = h2 // PATCH_SIZE
    grid_w = w2 // PATCH_SIZE
    
    # Reshape into patches with spatial merging
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
    
    # Transpose to group patches spatially
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    
    # Flatten to (num_patches, patch_dim)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, 
        channel * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
    )
    
    # Add batch dimension: (1, num_patches, patch_dim)
    pixel_values = flatten_patches[np.newaxis, :]
    # Calculate grid dimensions for image_grid_thw
    grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    
    return pixel_values, grid_thw

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
    """Expand single <|image_pad|> placeholder to multiple tokens based on actual image patches.
    
    This replicates the logic from baseline's get_image_padding_from_text:
    - Calculates num_tokens = (t * h * w) / (merge_size^2)
    - Replaces first occurrence of <|image_pad|> with that many <|image_pad|> tokens
    """
    t, h, w = image_grid_thw
    merge_area = merge_size ** 2
    num_tokens = (t * h * w) // merge_area
    
    # Replace first occurrence of IMAGE_PAD_TOKEN with num_tokens copies
    # (matches baseline behavior: replace once per image)
    expanded_prompt = base_prompt.replace(IMAGE_PAD_TOKEN, IMAGE_PAD_TOKEN * num_tokens, 1)
    
    return expanded_prompt, num_tokens


def run_inference(config_dir: Path, image_path: Path, prompt_text: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float, sample_dir: Path | None = None,
                  enable_qnn: bool = False, qnn_backend_path: str | None = None, qnn_provider_library: str | None = None,
                  do_sample: bool = False, min_length: int = 0, repetition_penalty: float = 1.0, tool_call_prompt: bool = False):
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # 1. Load raw pixel values (B, C, H, W) for GenAI vision pipeline
    pixel_values, grid_thw_array = load_prepatched_embeddings(image_path)
    grid_thw = grid_thw_array[0]  # Extract (t, h, w) tuple
    merge_size = MERGE_SIZE
    
    # DEBUG: Run vision pipeline manually to compare embeddings with baseline
    # run_vision_pipeline_debug(pixel_values, config_dir)
    
    # NOTE: GenAI will automatically run the vision pipeline (patch_embed -> vision_attn -> patch_merger)
    # when pixel_values are provided. No need to run it manually.
    
    # 2. Load model & tokenizer FIRST (needed for apply_chat_template)
    # Optionally register and prioritize QNN EP for vision attention acceleration.
    if enable_qnn:
        # Dynamically register QNN EP provider library if supplied (needed when ORT not built with QNN statically)
        if qnn_provider_library:
            og.register_execution_provider_library("QNN", qnn_provider_library)
        cfg_path = (config_dir / "genai_config.json") if (config_dir / "genai_config.json").is_file() else config_dir
        cfg = og.Config(str(cfg_path))
        cfg.append_provider("QNN")
        cfg.append_provider("CPUExecutionProvider")  # fallback for non-attention stages
        if qnn_backend_path:
            cfg.set_provider_option("QNN", "backend_path", qnn_backend_path)
        # Burst performance mode (3) if not overridden
        cfg.set_provider_option("QNN", "performance_mode", "3")
        model = og.Model(cfg)
    else:
        model = og.Model(str(config_dir))  # expects genai_config.json inside
    
    # Use HuggingFace tokenizer instead of ORT tokenizer to match baseline behavior
    tokenizer_hf = AutoTokenizer.from_pretrained(str(config_dir), trust_remote_code=True)
    
    # Also create ORT tokenizer for streaming decode during generation
    tokenizer_ort = og.Tokenizer(model)

    # Build prompt AFTER tokenizer is loaded (needed for apply_chat_template)
    if sample_dir is not None:
        base_prompt, image_count_in_sample = build_prompt_from_sample(sample_dir / "sample.json", use_tool_call_prompt=tool_call_prompt, tokenizer=tokenizer_hf)
        prompt, num_image_tokens = expand_image_tokens_in_prompt(base_prompt, grid_thw, merge_size)
    else:
        num_image_tokens = expand_image_tokens(grid_thw, merge_size)
        prompt = build_prompt(prompt_text, num_image_tokens, use_tool_call_prompt=tool_call_prompt)

    # Verify image token id exists
    image_token_id = tokenizer_hf.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if image_token_id is None:
        raise RuntimeError(f"Image token {IMAGE_PAD_TOKEN} not found in tokenizer")
    
    # Encode using HuggingFace tokenizer
    input_ids_list = tokenizer_hf.encode(prompt)
    input_ids_np = np.array(input_ids_list, dtype=np.int32)

    # Sanity check: count occurrences
    occurrences = int(np.sum(input_ids_np == image_token_id))
    if occurrences != num_image_tokens:
        print(f"[WARN] Token count mismatch: expected {num_image_tokens}, tokenizer found {occurrences}")

    # 4. Prepare generation params (respect model context_length)
    params = og.GeneratorParams(model)
    # Fetch context_length from config if available; else default to max_new_tokens
    context_len = None
    try:
        cfg_file2 = config_dir / "genai_config.json" if config_dir.is_dir() else config_dir
        with open(cfg_file2, "r", encoding="utf-8") as f2:
            cfgj = json.load(f2)
            mdl = cfgj.get("model", {}) if isinstance(cfgj, dict) else {}
            if isinstance(mdl, dict) and "context_length" in mdl:
                context_len = int(mdl["context_length"])
    except Exception:
        context_len = None
    # Align with reference semantics:
    # - Use model context capacity for total max_length (prefill + generation)
    # - Cap number of generated tokens separately via loop counter
    total_capacity = int(context_len) if context_len else int(input_ids_np.shape[0] + max_new_tokens)
    gen_cap = int(max_new_tokens)
    params.set_search_options(max_length=total_capacity, temperature=temperature, top_k=top_k, top_p=top_p,
                              do_sample=bool(do_sample), min_length=int(min_length), repetition_penalty=float(repetition_penalty))

    generator = og.Generator(model, params)

    # 5. Set pixel_values as input - GenAI will automatically run vision pipeline
    # Note: pixel_values is pre-patched format (1, num_patches, patch_dim), matching baseline
    pixel_values_f32 = np.ascontiguousarray(pixel_values.astype(np.float32))
    generator.set_model_input("pixel_values", pixel_values_f32)
    generator.set_model_input("image_grid_thw", np.ascontiguousarray(grid_thw_array.astype(np.int64)))

    # 6. Append textual tokens (chunked to satisfy context model input length)
    input_ids_i32 = input_ids_np.astype(np.int32)
    # Do not truncate the prompt; the runtime will process it in windows
    # according to the configured sliding window and chunk size.
    # Log final prefill window preview
    try:
        preview_len = min(50, input_ids_i32.shape[0])
    except Exception as e:
        print(f"[DEBUG] Failed to decode preview: {e}")
    # Append the full prompt once to avoid QNN continuous decoding constraints
    generator.append_tokens(input_ids_i32)

    # 7. Stream generation
    stream = tokenizer_ort.create_stream()
    output_tokens = []
    print("\n=== Generating ===")
    # Stream generation; rely on runtime-managed sequence lengths
    step_idx = 0
    # Read EOS token id(s) from config for early stop
    eos_ids = []
    try:
        cfg_file_eos = config_dir / "genai_config.json" if config_dir.is_dir() else config_dir
        with open(cfg_file_eos, "r", encoding="utf-8") as f_eos:
            cfgj_eos = json.load(f_eos)
            mdl = cfgj_eos.get("model", {}) if isinstance(cfgj_eos, dict) else {}
            if isinstance(mdl, dict):
                eos_val = mdl.get("eos_token_id")
                if isinstance(eos_val, list):
                    eos_ids = [int(x) for x in eos_val]
                elif isinstance(eos_val, (int, float)):
                    eos_ids = [int(eos_val)]
    except Exception:
        eos_ids = []
    
    # Tool-call mode state tracking
    accum_text = ""
    started_toolcall = False
    closed_toolcall = False
    
    while not generator.is_done():
        try:
            generator.generate_next_token()
        except Exception as gen_err:
            print(f"[ERROR] generate_next_token failed at step {step_idx}: {gen_err}")
            raise
        new_tok = generator.get_next_tokens()[0]
        output_tokens.append(new_tok)
        # Stop on EOS after min_length tokens
        if eos_ids and int(new_tok) in eos_ids and step_idx >= int(min_length):
            break
        decoded_piece = stream.decode(new_tok)
        
        if tool_call_prompt:
            # In tool-call mode: buffer text and only print from <tool_call> onwards
            accum_text += decoded_piece
            if not started_toolcall:
                if "<tool_call>" in accum_text:
                    started_toolcall = True
                    idx = accum_text.index("<tool_call>")
                    sys.stdout.write(accum_text[idx:])
                    sys.stdout.flush()
            else:
                # Already started printing tool_call region
                sys.stdout.write(decoded_piece)
                sys.stdout.flush()
                if "</tool_call>" in accum_text:
                    closed_toolcall = True
                    print(f"\n[DEBUG] tool_call closed at step {step_idx}")
                    break
        else:
            # Normal mode: print everything
            if decoded_piece:
                sys.stdout.write(decoded_piece)
                sys.stdout.flush()
        
        # print(f"\n[DEBUG] Gen step {step_idx}: token_id={new_tok} decoded='{decoded_piece}'")
        step_idx += 1
        if step_idx >= gen_cap:
            break
    
    print("\n=== Generation Complete ===")

    full_output = tokenizer_hf.decode(np.array(output_tokens, dtype=np.int32))
    
    # Report whether tool_call was successfully emitted
    if tool_call_prompt and not closed_toolcall:
        print("\n[WARNING] Model did not emit complete <tool_call>...</tool_call> structure.")
        print("[WARNING] Consider adjusting prompt, temperature, or sampling parameters.")
    
    # Write to file instead of stdout to handle Unicode characters
    # with open("generation_output.txt", "w", encoding="utf-8") as f:
    #     f.write("\n[FINAL OUTPUT]\n" + full_output)
    print("\n FINAL OUTPUT: ", full_output)
    return full_output


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL inference using onnxruntime-genai pipeline")
    # Support both --config_dir (current) and legacy --model_path name.
    parser.add_argument("--config_dir", type=Path, help="Directory containing genai_config.json for qwen2_5_vl (or use --model_path)")
    parser.add_argument("--model_path", type=Path, help="Alias for --config_dir (legacy)")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, help="User text prompt; if omitted and --sample_dir provided, sample conversation is used")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling to reduce repetition")
    parser.add_argument("--min_length", type=int, default=0, help="Minimum generated tokens before allowing EOS")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help=">1.0 discourages repetition")
    parser.add_argument("--sample_dir", type=Path, help="Optional dataset sample directory (contains sample.json & image)")
    parser.add_argument("--enable_qnn", action="store_true", help="Enable QNN execution provider (vision attention acceleration)")
    parser.add_argument("--qnn_backend_path", type=str, default="QnnHtp.dll", help="Path to QNN backend (e.g., QnnHtp.dll)")
    parser.add_argument("--qnn_provider_library", type=str, help="Path to onnxruntime QNN EP shared library (e.g., onnxruntime_providers_qnn.dll)")
    parser.add_argument("--tool_call_prompt", action="store_true", help="Enable tool-call mode: use baseline tools schema and emit <tool_call> XML")
    args = parser.parse_args()

    # Resolve config directory
    config_dir = args.config_dir or args.model_path
    if not config_dir:
        parser.error("One of --config_dir or --model_path is required.")

    # Determine prompt text
    if args.prompt:
        prompt_text = args.prompt
    elif args.sample_dir is not None:
        # Will be built from sample.json
        prompt_text = ""  # placeholder; not used when sample_dir provided
    else:
        prompt_text = "Describe the image."  # default fallback

    run_inference(
        config_dir=config_dir,
        image_path=args.image,
        prompt_text=prompt_text,
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
        qnn_provider_library=args.qnn_provider_library if args.enable_qnn else None,
        tool_call_prompt=args.tool_call_prompt,
    )

if __name__ == "__main__":
    main()
