"""
VideoChat-Flash **video** inference using pure ONNX Runtime.
Handles video frame extraction, vision encoding (with temporal compression),
embedding merge, and decoder KV-cache autoregressive decoding directly.

Memory strategy: only one large model is loaded at a time.
  Phase 1 – Vision  (~1GB): load → run all segments → free
  Phase 2 – Embed   (~2GB): load → run initial prompt → extract weight → free
  Phase 3 – Decoder (~14GB): load → autoregressive decode

Models needed in --model_path:
  vcf-vision-video.onnx + vcf-vision-video.onnx.data  (InternVideo2 + mm_projector, compress=True)
  vcf-embed.onnx        + vcf-embed.onnx.data         (embedding + visual merge)
  model.onnx            + model.onnx.data              (Qwen2.5-7B decoder)

The video vision model was exported with local_num_frames=4 (T=4).
Each segment of T frames produces 16*T = 64 visual tokens via ToMe compression.
For a video sampled at N total frames → ceil(N/T) segments → ceil(N/T)*64 visual tokens.

Usage:
  python inference_ort_video.py --model_path ./vcf-oga-fp32 --video video.mp4 --prompt "Describe this video"
  python inference_ort_video.py --model_path ./vcf-oga-fp32 --video video.mp4 --num_frames 16 --prompt "What happens?"
  python inference_ort_video.py --model_path ./vcf-oga-fp32 --image cat.jpeg --prompt "Describe this image"
  python inference_ort_video.py --model_path ./vcf-oga-fp32 --prompt "Hello, who are you?"
"""

import argparse
import gc
import math
import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from PIL import Image

MODEL_ID = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"
IMAGE_PAD_ID = 151655      # <|image_pad|>
VISION_START_ID = 151652   # <|vision_start|>
VISION_END_ID = 151653     # <|vision_end|>
EOS_TOKEN_ID = 151645      # <|im_end|>
LOCAL_NUM_FRAMES = 4       # T — baked into vcf-vision-video.onnx at export time
TOKENS_PER_SEGMENT = 64    # 16 * T with compress=True
NUM_LAYERS = 28
NUM_KV_HEADS = 4
HEAD_SIZE = 128
HIDDEN_SIZE = 3584

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224


def preprocess_frame(frame_rgb):
    """Resize and normalize a single RGB frame (PIL Image or ndarray) → [3, 224, 224]."""
    if isinstance(frame_rgb, np.ndarray):
        frame_rgb = Image.fromarray(frame_rgb)
    frame_rgb = frame_rgb.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    pixels = np.array(frame_rgb, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    return pixels.transpose(2, 0, 1)  # HWC → CHW


def extract_video_frames(video_path, num_frames):
    """Extract `num_frames` uniformly-spaced RGB frames from a video file.

    Returns a list of PIL Images.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    
    breakpoint()

    sample_count = min(num_frames, total)
    indices = np.linspace(0, total - 1, sample_count, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, bgr = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    cap.release()
    print(f"  Video: {total} total frames, {fps:.1f} fps, sampled {len(frames)} frames")
    return frames


def prepare_video_segments(frames):
    """Group preprocessed frames into segments of LOCAL_NUM_FRAMES (T=4).

    If the frame count isn't divisible by T, the last segment is padded by
    repeating the final frame.

    Returns: np.ndarray [num_segments, T, 3, 224, 224]
    """
    breakpoint()
    preprocessed = np.stack([preprocess_frame(f) for f in frames])  # [N, 3, H, W]
    N = preprocessed.shape[0]

    T = LOCAL_NUM_FRAMES
    num_segments = math.ceil(N / T)
    padded_len = num_segments * T

    if padded_len > N:
        pad = np.stack([preprocessed[-1]] * (padded_len - N))
        preprocessed = np.concatenate([preprocessed, pad], axis=0)

    segments = preprocessed.reshape(num_segments, T, 3, IMAGE_SIZE, IMAGE_SIZE)
    return segments


def preprocess_image(image_path):
    """Preprocess a single image → [1, T, 3, 224, 224] (repeat to fill one segment)."""
    img = Image.open(image_path).convert("RGB")
    frame = preprocess_frame(img)
    segment = np.stack([frame] * LOCAL_NUM_FRAMES)  # [T, 3, H, W]
    return segment[np.newaxis, :, :, :, :]  # [1, T, 3, H, W]


def build_prompt(tokenizer, user_prompt, num_visual_tokens):
    """Build tokenized prompt with chat template and the correct number of pad tokens."""
    if num_visual_tokens > 0:
        image_pads = "<|image_pad|>" * num_visual_tokens
        content = f"<|vision_start|>{image_pads}<|vision_end|>\n{user_prompt}"
    else:
        content = user_prompt

    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text, return_tensors="np").astype(np.int64)
    return input_ids  # [1, seq_len]


def run_vision_segments(session, segments):
    """Run vision ONNX on each segment and concatenate visual tokens.

    segments: [num_segments, T, 3, H, W]
    Returns:  [1, num_segments * TOKENS_PER_SEGMENT, HIDDEN_SIZE]
    """
    all_tokens = []
    for i in range(segments.shape[0]):
        seg = segments[i:i+1]  # [1, T, 3, H, W]
        outputs = session.run(None, {"images": seg})
        all_tokens.append(outputs[0])  # [1, 64, 3584]

    visual_tokens = np.concatenate(all_tokens, axis=1)  # [1, total_tokens, 3584]
    return visual_tokens


def run_embedding(session, input_ids, image_features):
    """Run embedding ONNX: input_ids + image_features → inputs_embeds."""
    outputs = session.run(None, {
        "input_ids": input_ids,
        "image_features": image_features,
    })
    return outputs[0]  # [1, seq_len, 3584]


def extract_embed_weight(model_dir):
    """Extract embed_tokens.weight from the ONNX external data file.

    Loads only the lightweight graph protobuf, reads the offset/length of the
    embedding initializer, then reads just that slice from the .onnx.data file.
    """
    import onnx

    graph_path = os.path.join(model_dir, "vcf-embed.onnx")
    model_proto = onnx.load(graph_path, load_external_data=False)

    for init in model_proto.graph.initializer:
        if "embed_tokens.weight" in init.name:
            ext = {e.key: e.value for e in init.external_data}
            location = ext["location"]
            offset = int(ext.get("offset", 0))
            length = int(ext["length"])
            shape = tuple(init.dims)

            data_path = os.path.join(model_dir, location)
            with open(data_path, "rb") as f:
                f.seek(offset)
                raw = f.read(length)

            del model_proto
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()

    raise ValueError("embed_tokens.weight not found in vcf-embed.onnx")


def embed_token_np(embed_weight, token_id):
    """Numpy-based single-token embedding lookup → [1, 1, hidden_size]."""
    return embed_weight[token_id][np.newaxis, np.newaxis, :]


def greedy_decode(decoder_session, embed_weight, inputs_embeds, max_new_tokens=512):
    """Autoregressive decoding with KV cache."""
    batch = 1
    seq_len = inputs_embeds.shape[1]

    past_kv = {}
    for i in range(NUM_LAYERS):
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch, NUM_KV_HEADS, 0, HEAD_SIZE), dtype=np.float32
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch, NUM_KV_HEADS, 0, HEAD_SIZE), dtype=np.float32
        )

    attention_mask = np.ones((batch, seq_len), dtype=np.int64)
    current_embeds = inputs_embeds

    for step in range(max_new_tokens):
        feeds = {"inputs_embeds": current_embeds, "attention_mask": attention_mask}
        feeds.update(past_kv)

        outputs = decoder_session.run(None, feeds)

        logits = outputs[0]
        next_token = int(np.argmax(logits[0, -1, :]))

        if next_token == EOS_TOKEN_ID:
            break

        for i in range(NUM_LAYERS):
            past_kv[f"past_key_values.{i}.key"] = outputs[1 + i]
            past_kv[f"past_key_values.{i}.value"] = outputs[1 + NUM_LAYERS + i]

        total_len = past_kv["past_key_values.0.key"].shape[2]
        attention_mask = np.ones((batch, total_len + 1), dtype=np.int64)

        current_embeds = embed_token_np(embed_weight, next_token)

        yield next_token


def main():
    parser = argparse.ArgumentParser(
        description="VideoChat-Flash video inference (pure ONNX Runtime)"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (single-frame mode)")
    parser.add_argument("--num_frames", type=int, default=28,
                        help="Number of frames to sample from video (rounded up to multiple of T=4)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    if args.video and args.image:
        print("ERROR: specify either --video or --image, not both")
        return

    print(f"onnxruntime version: {ort.__version__}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_cpu_mem_arena = False

    model_dir = args.model_path
    has_visual = args.video is not None or args.image is not None

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # ---- Phase 1: Vision (load → run → free) ----
    visual_tokens = np.zeros((1, 0, HIDDEN_SIZE), dtype=np.float32)
    num_visual_tokens = 0

    if has_visual:
        vision_path = os.path.join(model_dir, "vcf-vision-video.onnx")
        if not os.path.exists(vision_path):
            print(f"ERROR: vcf-vision-video.onnx not found in {model_dir}")
            return

        print("\nLoading video vision model...")
        vision_session = ort.InferenceSession(vision_path, so)
        print("  Loaded: vcf-vision-video.onnx")

        if args.video:
            print(f"Extracting frames from: {args.video}")
            frames = extract_video_frames(args.video, args.num_frames)
            if len(frames) == 0:
                print("ERROR: no frames extracted from video")
                return

            print(f"Preparing segments (T={LOCAL_NUM_FRAMES})...")
            segments = prepare_video_segments(frames)
            num_segments = segments.shape[0]
            print(f"  {len(frames)} frames → {num_segments} segment(s) of {LOCAL_NUM_FRAMES} frames")
            del frames
        else:
            print(f"Preprocessing image (single-frame mode): {args.image}")
            segments = preprocess_image(args.image)
            num_segments = 1
            print(f"  1 image → 1 segment ({LOCAL_NUM_FRAMES} repeated frames)")

        print("Running video vision model...")
        visual_tokens = run_vision_segments(vision_session, segments)
        num_visual_tokens = visual_tokens.shape[1]
        print(f"  Visual tokens: {visual_tokens.shape}  ({num_segments} seg × {TOKENS_PER_SEGMENT} tok)")

        del vision_session, segments
        gc.collect()
        print("  Vision session freed.")

    # ---- Phase 2: Embedding (load → run prompt → free session) ----
    print("\nLoading embedding model...")
    embed_session = ort.InferenceSession(
        os.path.join(model_dir, "vcf-embed.onnx"), so
    )
    print("  Loaded: vcf-embed.onnx")

    print(f"\nPrompt: {args.prompt}")
    input_ids = build_prompt(tokenizer, args.prompt, num_visual_tokens)
    print(f"  Token count: {input_ids.shape[1]}  (includes {num_visual_tokens} image_pad tokens)")

    print("Running embedding (merges visual tokens into prompt)...")
    inputs_embeds = run_embedding(embed_session, input_ids, visual_tokens)
    print(f"  inputs_embeds: {inputs_embeds.shape}")

    del embed_session, visual_tokens, input_ids
    gc.collect()
    print("  Embedding session freed.")

    print("Extracting embedding weight table from ONNX data file...")
    embed_weight = extract_embed_weight(model_dir)
    print(f"  embed_weight: {embed_weight.shape} ({embed_weight.nbytes / 1e9:.2f} GB)")

    # ---- Phase 3: Decoder (load last, largest model) ----
    print("\nLoading decoder model...")
    decoder_session = ort.InferenceSession(
        os.path.join(model_dir, "model.onnx"), so
    )
    print("  Loaded: model.onnx")

    print("\nResponse: ", end="", flush=True)
    for token_id in greedy_decode(decoder_session, embed_weight, inputs_embeds, args.max_tokens):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        print(text, end="", flush=True)
    print()
    print("\nDone.")


if __name__ == "__main__":
    main()
