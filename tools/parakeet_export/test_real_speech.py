#!/usr/bin/env python3
"""
Test Parakeet-TDT ASR with real speech audio.
Compares raw ONNX Runtime TDT inference with onnxruntime-genai pipeline.

Key difference from Nemotron: Parakeet-TDT joint output includes duration
logits alongside token logits. The greedy decode uses predicted durations
to skip frames efficiently.
"""

import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path


def load_tdt_config(model_dir: Path) -> dict:
    """Load TDT config from tdt_config.json."""
    config_path = model_dir / "tdt_config.json"
    if not config_path.exists():
        return {"durations": [0, 1, 2, 4, 8], "num_extra_outputs": 5, "vocab_size": 1025}
    with open(config_path) as f:
        return json.load(f)


def main():
    model_dir = Path(__file__).parent / "onnx_models"
    audio_path = Path(__file__).parent.parent.parent / "test" / "test_models" / "audios" / "jfk.flac"

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return False

    # Load TDT config
    tdt_cfg = load_tdt_config(model_dir)
    durations = tdt_cfg["durations"]
    num_durations = len(durations)
    vocab_size = tdt_cfg["vocab_size"]
    blank_id = vocab_size - 1
    print(f"TDT config: vocab_size={vocab_size}, blank_id={blank_id}, durations={durations}")

    # Load audio
    waveform_np, sr = sf.read(str(audio_path), dtype="float32")
    print(f"Audio: shape={waveform_np.shape}, sr={sr}, duration={len(waveform_np)/sr:.2f}s")

    if len(waveform_np.shape) > 1:
        waveform_np = waveform_np.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import torchaudio
        waveform_t = torch.from_numpy(waveform_np).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform_t = resampler(waveform_t)
        waveform_np = waveform_t.squeeze(0).numpy()
        sr = 16000
        print(f"Resampled to 16kHz, length={len(waveform_np)}")

    # Use NeMo preprocessor for mel features
    waveform_t = torch.from_numpy(waveform_np).unsqueeze(0)
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

    preprocessor = FilterbankFeatures(
        sample_rate=16000,
        n_window_size=400,   # 25ms
        n_window_stride=160, # 10ms
        n_fft=512,
        nfilt=128,
        dither=0.0,
        pad_to=0,
        normalize="per_feature",
    )
    length = torch.tensor([waveform_t.shape[1]])
    features, feat_length = preprocessor(waveform_t, length)
    mel_np = features.numpy().astype(np.float32)
    print(f"Mel features: {mel_np.shape}, feat_length={feat_length.item()}")

    # ---- Raw ONNX Runtime TDT Greedy Decode ----
    import onnxruntime as ort

    enc = ort.InferenceSession(str(model_dir / "encoder.onnx"))
    dec = ort.InferenceSession(str(model_dir / "decoder.onnx"))
    jnt = ort.InferenceSession(str(model_dir / "joint.onnx"))

    enc_out = enc.run(
        None,
        {"audio_signal": mel_np, "length": np.array([mel_np.shape[2]], dtype=np.int64)},
    )
    encoded_t = enc_out[0].transpose(0, 2, 1)  # [1, T', D]
    enc_len = int(enc_out[1][0])
    print(f"Encoder: enc_len={enc_len}")

    # TDT greedy decode with duration-based frame skipping
    tokens_raw = []
    cur = 0  # current token (BOS)
    h = np.zeros((2, 1, 640), dtype=np.float32)
    c = np.zeros((2, 1, 640), dtype=np.float32)
    t = 0

    while t < enc_len:
        enc_t = encoded_t[:, t:t+1, :]
        for _ in range(10):  # max symbols per frame
            tgt = np.array([[cur]], dtype=np.int64)
            tl = np.array([1], dtype=np.int64)
            d = dec.run(None, {
                "targets": tgt, "target_length_orig": tl,
                "h_in": h, "c_in": c,
            })
            dh = d[0][:, :, -1:].transpose(0, 2, 1)  # [1, 1, D]
            h, c = d[2], d[3]  # update LSTM states

            j = jnt.run(None, {"encoder_output": enc_t, "decoder_output": dh})
            logits = j[0].squeeze()  # [vocab_size + num_durations]

            # Split token and duration logits
            token_logits = logits[:vocab_size]
            duration_logits = logits[vocab_size:vocab_size + num_durations]

            tok = int(np.argmax(token_logits))
            dur_idx = int(np.argmax(duration_logits))
            dur = durations[dur_idx] if dur_idx < len(durations) else 1

            if tok == blank_id:
                # Blank: advance by predicted duration (at least 1)
                t += max(dur, 1)
                break
            tokens_raw.append(tok)
            cur = tok
        else:
            # Max symbols hit, advance one frame
            t += 1

    # ---- onnxruntime-genai ----
    import onnxruntime_genai as og

    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=512, batch_size=1)
    gen = og.Generator(model, params)

    inputs = og.NamedTensors()
    inputs["audio_signal"] = mel_np
    inputs["input_ids"] = np.array([[0]], dtype=np.int32)
    gen.set_inputs(inputs)

    step = 0
    while not gen.is_done():
        gen.generate_next_token()
        step += 1
        if step > 500:
            break

    tok_list = list(gen.get_sequence(0))
    text_ids_og = [t for t in tok_list[1:] if t != blank_id]

    # Compare and decode
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Raw ONNX tokens ({len(tokens_raw)}): {tokens_raw}")
    print(f"OG tokens       ({len(text_ids_og)}): {text_ids_og}")
    print(f"Token match: {tokens_raw == text_ids_og}")

    if tokens_raw:
        text = tokenizer.decode(np.array(tokens_raw, dtype=np.int32))
        print(f'\nTranscription (raw ONNX): "{text}"')
    if text_ids_og:
        text = tokenizer.decode(np.array(text_ids_og, dtype=np.int32))
        print(f'Transcription (OG):       "{text}"')
    if not tokens_raw and not text_ids_og:
        print("Both paths: all blanks (no speech tokens)")

    result = tokens_raw == text_ids_og

    # Cleanup ORT GenAI objects to avoid leak warnings
    del gen, params, inputs, tokenizer, model

    return result


if __name__ == "__main__":
    success = main()
