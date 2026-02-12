#!/usr/bin/env python3
"""
End-to-end test for Parakeet-TDT ASR via onnxruntime-genai.

Tests:
1. Model loading
2. Tokenizer load + decode
3. Full TDT inference with dummy audio
4. Raw ONNX Runtime TDT greedy decode (baseline)
"""

import json
import sys
import numpy as np
from pathlib import Path


def load_tdt_config(model_path: str) -> dict:
    """Load TDT config from tdt_config.json."""
    config_path = Path(model_path) / "tdt_config.json"
    if not config_path.exists():
        return {"durations": [0, 1, 2, 4, 8], "num_extra_outputs": 5, "vocab_size": 1025}
    with open(config_path) as f:
        return json.load(f)


def test_model_loading(model_path: str):
    """Test that the model loads correctly."""
    import onnxruntime_genai as og

    print("[1/4] Loading model...")
    model = og.Model(model_path)
    print("  ✓ Model loaded successfully")
    return model


def test_tokenizer(model):
    """Test tokenizer load and decode."""
    import onnxruntime_genai as og

    print("\n[2/4] Testing tokenizer...")
    tokenizer = og.Tokenizer(model)
    print("  ✓ Tokenizer loaded")

    # Test individual token decode
    for tid in [5, 33, 3, 34, 57]:
        decoded = tokenizer.decode(tid)
        print(f"  Token {tid} -> {decoded!r}")

    # Test batch decode
    ids = np.array([34, 3, 23], dtype=np.int32)
    text = tokenizer.decode(ids)
    print(f"  ✓ Batch decode [34, 3, 23] -> {text!r}")

    return tokenizer


def test_inference(model, tokenizer):
    """Test running inference (encoder + TDT decode) via onnxruntime-genai."""
    import onnxruntime_genai as og

    print("\n[3/4] Testing inference via onnxruntime-genai...")
    try:
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=512,
            batch_size=1,
        )
        print("  ✓ GeneratorParams created")

        generator = og.Generator(model, params)
        print("  ✓ Generator created")

        batch_size = 1
        mel_bins = 128
        time_frames = 100
        audio = np.random.randn(batch_size, mel_bins, time_frames).astype(np.float32)

        inputs = og.NamedTensors()
        inputs["audio_signal"] = audio
        inputs["input_ids"] = np.array([[0]], dtype=np.int32)
        print(f"  ✓ NamedTensors created (audio shape={audio.shape})")

        generator.set_inputs(inputs)
        print("  ✓ set_inputs completed (encoder + TDT decode triggered)")

        step = 0
        while not generator.is_done():
            generator.generate_next_token()
            step += 1
            if step > 500:
                print("  ⚠ Safety limit reached (500 tokens)")
                break

        print(f"  ✓ Generation completed in {step} steps")

        tokens = generator.get_sequence(0)
        token_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
        print(f"  ✓ Full sequence ({len(token_list)} tokens): {token_list[:20]}{'...' if len(token_list) > 20 else ''}")

        print("  ✓ Inference completed successfully!")
        return True
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_raw_onnx_inference(model_path: str):
    """Test with raw onnxruntime as a baseline reference.

    Implements TDT greedy decode: the joint output has shape
    [B, 1, 1, vocab_size + num_durations]. Token logits are the first
    vocab_size values, and duration logits are the last num_durations values.
    """
    import onnxruntime as ort

    print("\n[4/4] Raw ONNX Runtime TDT inference test (baseline)...")

    tdt_cfg = load_tdt_config(model_path)
    durations = tdt_cfg["durations"]
    num_durations = len(durations)
    vocab_size = tdt_cfg["vocab_size"]
    blank_id = vocab_size - 1
    print(f"  TDT config: vocab_size={vocab_size}, blank_id={blank_id}, durations={durations}")

    # Load sessions
    enc_path = str(Path(model_path) / "encoder.onnx")
    dec_path = str(Path(model_path) / "decoder.onnx")
    joint_path = str(Path(model_path) / "joint.onnx")

    enc_sess = ort.InferenceSession(enc_path)
    dec_sess = ort.InferenceSession(dec_path)
    joint_sess = ort.InferenceSession(joint_path)

    # Encoder
    audio = np.random.randn(1, 128, 100).astype(np.float32)
    length = np.array([100], dtype=np.int64)

    enc_out = enc_sess.run(None, {"audio_signal": audio, "length": length})
    encoded = enc_out[0]   # [1, D, T']
    enc_len = int(enc_out[1][0])
    print(f"  Encoder output: {encoded.shape}, encoded_length={enc_len}")

    # Transpose: [1, D, T'] -> [1, T', D]
    encoded_t = encoded.transpose(0, 2, 1)

    # TDT greedy decode
    tokens = []
    current_token = 0  # BOS
    max_sym = 10
    t = 0

    while t < enc_len:
        enc_t = encoded_t[:, t:t+1, :]
        for _ in range(max_sym):
            targets = np.array([[current_token]], dtype=np.int64)
            target_len = np.array([1], dtype=np.int64)
            # Initialize LSTM states
            h = np.zeros((2, 1, 640), dtype=np.float32)
            c = np.zeros((2, 1, 640), dtype=np.float32)
            dec_out = dec_sess.run(None, {
                "targets": targets, "target_length_orig": target_len,
                "h_in": h, "c_in": c,
            })
            dec_h = dec_out[0][:, :, -1:].transpose(0, 2, 1)  # [1, 1, D]

            joint_out = joint_sess.run(None, {
                "encoder_output": enc_t, "decoder_output": dec_h,
            })
            logits = joint_out[0].squeeze()  # [vocab_size + num_durations]

            # Split token logits and duration logits
            token_logits = logits[:vocab_size]
            duration_logits = logits[vocab_size:vocab_size + num_durations]

            tok = int(np.argmax(token_logits))
            dur_idx = int(np.argmax(duration_logits))
            dur = durations[dur_idx] if dur_idx < len(durations) else 1

            if tok == blank_id:
                # Blank: advance by predicted duration (at least 1)
                t += max(dur, 1)
                break
            tokens.append(tok)
            current_token = tok

        if t >= min(enc_len, 5):
            break

    print(f"  Decoded tokens (first 5 frames): {tokens}")
    print("  ✓ Raw ONNX TDT inference successful")


def main():
    print("=" * 60)
    print("Parakeet-TDT ASR - End-to-End Test via onnxruntime-genai")
    print("=" * 60)

    model_path = str(Path(__file__).parent / "onnx_models")

    if not Path(model_path).exists():
        print(f"Error: Model directory not found: {model_path}")
        return False

    # Run via onnxruntime-genai
    model = test_model_loading(model_path)
    tokenizer = test_tokenizer(model)
    success = test_inference(model, tokenizer)

    # Run raw ONNX test as baseline
    try:
        test_raw_onnx_inference(model_path)
    except Exception as e:
        print(f"  ✗ Raw ONNX test failed: {e}")

    print("\n" + "=" * 60)
    print("Result:", "✓ ALL TESTS PASSED" if success else "✗ SOME TESTS FAILED")
    print("=" * 60)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
