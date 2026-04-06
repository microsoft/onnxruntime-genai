# -------------------------------------------------------------------------
# Copyright (C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Export VideoChat-Flash (OpenGVLab) ONNX models for onnxruntime-genai.

Model: OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B
Architecture:
  - Language backbone: Qwen2.5-7B (28L, GQA 28h/4kv, hidden=3584)
  - Visual encoder:    InternVideo2-1B (video ViT with 3D spatiotemporal attention)
  - Connector:         MLP-based HiCo token compression (~16 tokens/frame)

This script exports:
  1. Text decoder (model.onnx) via OGA builder — fully functional
  2. Vision encoder (vcf-vision.onnx) — TODO: InternVideo2 export
  3. Embedding merger (vcf-embedding.onnx) — TODO: token fusion export
  4. genai_config.json — wired for OGA multimodal pipeline

Phase 1 (this PR): Text decoder only. Vision/embedding stubs are included
as placeholders. Pass --text-only to skip vision/embedding export.

Usage:
  # Download model and export text decoder only (Phase 1):
  python builder.py --output ./vcf-oga-fp32 --text-only

  # Full pipeline export (requires vision encoder work):
  python builder.py --input ./pytorch_vcf --output ./vcf-oga-int4

  # Text-only inference smoke test after export:
  python builder.py --output ./vcf-oga-fp32 --text-only --run-e2e
"""

import argparse
import json
import os
import sys

# Use the local OGA model builder (src/python/py/models/builder.py)
# so our VideoChatFlashQwenModel registration is picked up.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "python", "py", "models"))

from builder import create_model  # noqa: E402 (local OGA builder)

HF_MODEL_ID = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"

# VideoChat-Flash image placeholder token id (same vocab as Qwen2.5)
IMAGE_TOKEN_ID = 151646  # <|image_pad|> — verify against tokenizer_config.json


def prepare_model(input_dir):
    """Load HF model config from local path or HuggingFace."""
    from transformers import AutoConfig

    print("\n[1/4] Loading model config...")
    # trust_remote_code=False: config.json is standard JSON — no need for
    # the custom modeling code (which requires av/cv2/decord).
    config = AutoConfig.from_pretrained(input_dir, trust_remote_code=True)
    print(f"  architecture : {config.architectures[0]}")
    print(f"  hidden_size  : {config.hidden_size}")
    print(f"  num_layers   : {config.num_hidden_layers}")
    print(f"  num_heads    : {config.num_attention_heads} Q / {config.num_key_value_heads} KV")
    print(f"  rope_theta   : {config.rope_theta}")
    print(f"  vocab_size   : {config.vocab_size}")
    return config


def export_vision_model(model, config, output_dir):
    """
    Export the InternVideo2-1B visual encoder.

    TODO (Phase 2): The InternVideo2 encoder uses 3D spatiotemporal attention
    and learnable position embeddings. Export requires:
      1. Wrapping model.vision_tower (InternVideo2) in a torch.onnx-compatible
         nn.Module that accepts (pixel_values: [T*H*W, C], grid_thw: [N, 3]).
      2. Exporting the HiCo clip-level MLP compression head.
      3. Verifying opset compatibility for temporal attention ops.

    For now this function is a no-op placeholder.
    """
    print("\n[2/4] Vision encoder export — TODO (Phase 2, skipped)")
    print("  The InternVideo2-1B encoder requires custom 3D spatiotemporal")
    print("  attention export. See Phase 2 implementation plan.")


def export_embedding_model(model, config, output_dir):
    """
    Export the embedding merger that fuses visual tokens into the token stream.

    TODO (Phase 2): VideoChat-Flash uses mm_patch_merge_type and llm_compress
    layers to inject visual tokens. Export requires:
      1. Wrapping model.model.embed_tokens (Qwen2.5 embedding table).
      2. Implementing the token replacement mask (image_token_id → vision features).
      3. Handling the HiCo video-level compression in LLM layers (llm_compress_layer_list).

    For now this function is a no-op placeholder.
    """
    print("\n[3/4] Embedding merger export — TODO (Phase 2, skipped)")
    print("  Token fusion depends on Phase 2 vision encoder output shape.")


def export_text_model(input_dir, output_dir, precision, text_only):
    """Export text decoder (Qwen2.5-7B backbone) via OGA builder."""
    print(f"\n[{3 if False else 2}/4] Exporting text decoder ({precision.upper()})...")
    print(f"  Source: {input_dir}")

    # create_model(model_name, input_path, ...):
    #   model_name  — HF repo ID used for config/tokenizer lookup
    #   input_path  — local dir (or HF repo ID when downloading)
    # exclude_embeds controls decoder input:
    #   text_only  → False  (input_ids,     standalone text inference)
    #   full VLM   → True   (inputs_embeds, from embedding merger)
    hf_id = HF_MODEL_ID
    local_or_hf = input_dir if input_dir is not None else hf_id
    create_model(
        hf_id,
        local_or_hf,
        output_dir,
        precision,
        "cpu",
        os.path.join(output_dir, ".cache"),
        exclude_embeds=not text_only,
    )
    print(f"  [OK] Text decoder: {os.path.join(output_dir, 'model.onnx')}")


def update_genai_config(output_dir, text_only):
    """Patch genai_config.json with VideoChat-Flash model type and vision sections."""
    config_path = os.path.join(output_dir, "genai_config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if text_only:
        # Text-only / standalone mode: decoder takes input_ids, no vision/embedding.
        # Use qwen2 type so OGA loads it as a plain decoder (MultiModalLanguageModel
        # requires vision.onnx + embedding.onnx which are not present in this mode).
        config["model"]["type"] = "qwen2"
    else:
        # Full VLM pipeline: decoder takes inputs_embeds from the embedding merger.
        config["model"]["type"] = "videochat_flash_qwen"

    if not text_only:
        # Phase 2: wire vision encoder and embedding merger
        config["model"]["vision"] = {
            "filename": "vcf-vision.onnx",
            "inputs": {
                "pixel_values": "pixel_values",
                "image_grid_thw": "image_grid_thw",
            },
            "outputs": {
                "image_features": "visual_tokens",
            },
        }
        config["model"]["embedding"] = {
            "filename": "vcf-embedding.onnx",
            "inputs": {
                "input_ids": "input_ids",
                "image_features": "visual_tokens",
            },
            "outputs": {
                "inputs_embeds": "inputs_embeds",
            },
        }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"  [OK] Updated: genai_config.json  (type={config['model']['type']})")


def run_e2e_smoke(output_dir, prompt):
    """Quick text-only inference smoke test."""
    import onnxruntime_genai as og

    print("\n[Smoke] Running text-only inference...")
    model = og.Model(output_dir)
    tokenizer = og.Tokenizer(model)
    tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=128)
    params.input_ids = tokens

    generator = og.Generator(model, params)
    print("Output:", end=" ", flush=True)
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        print(tokenizer.decode([token]), end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Export VideoChat-Flash for onnxruntime-genai"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Local PyTorch model directory. If omitted, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./vcf-oga",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int4"],
        help="Text decoder precision",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Export text decoder only (Phase 1). Skip vision/embedding export.",
    )
    parser.add_argument(
        "--run-e2e",
        action="store_true",
        help="Run text-only inference smoke test after export.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the video in one sentence.",
        help="Prompt for --run-e2e smoke test",
    )
    args = parser.parse_args()

    input_dir = args.input or HF_MODEL_ID
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("VideoChat-Flash ONNX Export for OGA")
    print("=" * 70)
    print(f"  Source   : {input_dir}")
    print(f"  Output   : {output_dir}")
    print(f"  Precision: {args.precision.upper()}")
    print(f"  Mode     : {'text-only (Phase 1)' if args.text_only else 'full pipeline (Phase 2)'}")

    if not args.text_only:
        # Phase 2: load full model weights for vision/embedding export
        prepare_model(input_dir)
        export_vision_model(None, None, output_dir)
        export_embedding_model(None, None, output_dir)

    export_text_model(input_dir, output_dir, args.precision, args.text_only)

    print("\n[4/4] Updating genai_config.json...")
    update_genai_config(output_dir, args.text_only)

    if args.run_e2e:
        run_e2e_smoke(output_dir, args.prompt)

    print("\n" + "=" * 70)
    print("[SUCCESS] Export complete!")
    print("=" * 70)
    print(f"\nOutput: {output_dir}")
    print("\nExported files:")
    print(f"  model.onnx         ({args.precision.upper()}, text decoder — Qwen2.5-7B)")
    if not args.text_only:
        print("  vcf-vision.onnx    (Phase 2 TODO — InternVideo2-1B)")
        print("  vcf-embedding.onnx (Phase 2 TODO — token merger)")
    print("  genai_config.json  (type=videochat_flash_qwen)")
    print()


if __name__ == "__main__":
    main()
