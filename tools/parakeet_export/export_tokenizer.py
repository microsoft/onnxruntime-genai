#!/usr/bin/env python3
"""
Export tokenizer for Parakeet-TDT ASR model.

Downloads the NeMo model, extracts the SentencePiece vocabulary, and converts
it into HuggingFace Unigram format (tokenizer.json + tokenizer_config.json)
that ORT Extensions can load via the T5Tokenizer path.

Usage:
    python export_tokenizer.py --output_dir ./onnx_models
    python export_tokenizer.py --output_dir ./onnx_models --test
"""

import argparse
import json
import sys
from pathlib import Path


def extract_vocab(model_name: str, output_dir: Path) -> list:
    """Extract vocabulary from NeMo model and save vocab.txt."""

    print("[1/3] Loading NeMo model...")
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("Error: NeMo not installed. Run:")
        print("  pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]")
        sys.exit(1)

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)

    if not hasattr(asr_model, "tokenizer"):
        print("Error: Model does not have a tokenizer attribute")
        sys.exit(1)

    tokenizer = asr_model.tokenizer
    vocab_size = asr_model.cfg.joint.num_classes + 1  # +1 for blank

    # Build ordered token list by decoding each ID
    print(f"[2/3] Extracting {vocab_size} tokens...")
    tokens = []
    for i in range(vocab_size - 1):  # Exclude blank (last)
        try:
            decoded = tokenizer.ids_to_tokens([i])
            token = decoded[0] if isinstance(decoded, list) else decoded
            tokens.append(str(token))
        except Exception:
            tokens.append(f"<unk_{i}>")
    tokens.append("<blank>")

    # Save vocab.txt
    vocab_txt = output_dir / "vocab.txt"
    with open(vocab_txt, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(f"{token}\n")
    print(f"  Saved {vocab_txt} ({len(tokens)} tokens)")

    return tokens


def create_unigram_tokenizer(tokens: list, output_dir: Path):
    """
    Create HuggingFace Unigram tokenizer.json + tokenizer_config.json.

    ORT Extensions' Unigram path (used by T5Tokenizer) expects:
    - tokenizer.json with model.type = "Unigram" and model.vocab = [[token, score], ...]
    - tokenizer_config.json with tokenizer_class = "T5Tokenizer"
    """
    vocab_size = len(tokens)

    # Build Unigram vocab as [[token, score], ...]
    # Rank-based scores: lower index = more frequent = higher score
    unigram_vocab = []
    for i, token in enumerate(tokens):
        if token in ("<unk>", "<blank>"):
            score = 0.0
        else:
            score = -float(i) / vocab_size * 10.0
        unigram_vocab.append([token, score])

    # tokenizer.json
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "<unk>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": vocab_size - 1,
                "content": "<blank>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "add_prefix_space": True,
            "prepend_scheme": "always",
        },
        "model": {
            "type": "Unigram",
            "unk_id": 0,
            "vocab": unigram_vocab,
        },
    }

    tokenizer_json_path = output_dir / "tokenizer.json"
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved {tokenizer_json_path}")

    # tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "T5Tokenizer",
        "unk_token": "<unk>",
        "eos_token": "<blank>",
        "pad_token": "<blank>",
        "model_max_length": 8192,
        "sp_model_kwargs": {},
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            str(vocab_size - 1): {
                "content": "<blank>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
        },
    }

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print(f"  Saved {tokenizer_config_path}")


def test_tokenizer(model_dir: str):
    """Smoke-test that ORT GenAI can load and decode tokens."""
    import numpy as np

    try:
        import onnxruntime_genai as og
    except ImportError:
        print("  Skipped: onnxruntime-genai not installed")
        return

    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)

    test_ids = [5, 33, 3]
    for tid in test_ids:
        text = tokenizer.decode(np.array([tid], dtype=np.int32))
        print(f"  Token {tid} -> {text!r}")

    ids = np.array(test_ids, dtype=np.int32)
    text = tokenizer.decode(ids)
    print(f"  Batch {test_ids} -> {text!r}")
    print("  [OK] Tokenizer working")


def main():
    parser = argparse.ArgumentParser(
        description="Export Parakeet-TDT ASR tokenizer to ORT-compatible Unigram format"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="HuggingFace model name or local .nemo path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./onnx_models",
        help="Output directory for tokenizer files",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test tokenizer with ORT GenAI after export",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Parakeet-TDT ASR Tokenizer Export")
    print("=" * 60)

    tokens = extract_vocab(args.model_name, output_dir)

    print("[3/3] Creating ORT-compatible Unigram tokenizer...")
    create_unigram_tokenizer(tokens, output_dir)

    if args.test:
        print("\nTesting tokenizer with ORT GenAI...")
        test_tokenizer(str(output_dir))

    print("\n" + "=" * 60)
    print(f"Done! Tokenizer files saved to {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
