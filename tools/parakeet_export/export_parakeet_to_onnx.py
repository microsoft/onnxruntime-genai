#!/usr/bin/env python3
"""
Export NVIDIA Parakeet-TDT-0.6B-v3 ASR model to ONNX format.

This script downloads the model from HuggingFace and exports it to ONNX format
suitable for inference with ONNX Runtime. The decoder is exported with explicit
LSTM state inputs/outputs for stateful RNNT decoding.

The key difference from Nemotron is that Parakeet-TDT uses a Token-and-Duration
Transducer (TDT) joint network, which outputs both token logits and duration
logits. The joint output has shape [B, T, U, V + 1 + num_durations].

Usage:
    python export_parakeet_to_onnx.py --output_dir ./onnx_models
    python export_parakeet_to_onnx.py --output_dir ./onnx_models --device cuda
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Parakeet-TDT-0.6B-v3 ASR model to ONNX"
    )
    parser.add_argument("--model_name", type=str,
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="HuggingFace model name or path to local .nemo file")
    parser.add_argument("--output_dir", type=str, default="./onnx_models",
        help="Directory to save ONNX models")
    parser.add_argument("--opset_version", type=int, default=17,
        help="ONNX opset version")
    parser.add_argument("--device", type=str, default="cpu",
        choices=["cpu", "cuda"], help="Device to use for export")
    parser.add_argument("--dynamo", action="store_true",
        help="Use torch.onnx.export with dynamo=True (TorchDynamo-based export)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------
# PyTorch 2.10 torch.onnx.export routes through the dynamo exporter by
# default and fails on NeMo typed forward() signatures with:
#   "All arguments must be passed by kwargs only for typed methods"
# These thin wrappers provide an untyped forward() so torch.onnx.export
# can trace the graph without issues.
# ---------------------------------------------------------------------------


def _make_encoder_wrapper(encoder):
    """Wrap the NeMo encoder so torch.onnx.export can trace it."""
    import torch.nn as nn

    class EncoderWrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, audio_signal, length):
            encoded, encoded_len = self.enc(
                audio_signal=audio_signal, length=length
            )
            return encoded, encoded_len

    return EncoderWrapper(encoder)


def _make_stateful_decoder_wrapper(decoder):
    """
    Wrap the NeMo decoder to expose LSTM hidden/cell states as explicit
    graph inputs and outputs for stateful RNNT decoding.
    """
    import torch.nn as nn

    class StatefulDecoderWrapper(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.decoder = dec
            self.decoder._rnnt_export = True

        def forward(self, targets, target_length, h_in, c_in):
            g, states = self.decoder.predict(
                y=targets, state=(h_in, c_in), add_sos=False
            )
            h_out, c_out = states
            g = g.transpose(1, 2)  # [B,1,D] -> [B,D,1]
            return g, target_length, h_out, c_out

    return StatefulDecoderWrapper(decoder)


def _make_joint_wrapper(joint):
    """Wrap the NeMo RNNTJoint so torch.onnx.export can trace it.

    For TDT models, the joint output includes both token logits and
    duration logits: [B, T, U, V + 1 + num_durations].
    """
    import torch.nn as nn

    class JointWrapper(nn.Module):
        def __init__(self, j):
            super().__init__()
            self.joint = j

        def forward(self, encoder_output, decoder_output):
            return self.joint.joint(encoder_output, decoder_output)

    return JointWrapper(joint)


def _verify_stateful_decoder(decoder_path):
    """ORT smoke-test: two steps, verify LSTM states evolve."""
    import numpy as np
    import onnxruntime as ort

    print("      Verifying stateful decoder with ONNX Runtime...")
    sess = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    targets = np.zeros((1, 1), dtype=np.int64)
    tlen = np.array([1], dtype=np.int64)
    h = np.zeros((2, 1, 640), dtype=np.float32)
    c = np.zeros((2, 1, 640), dtype=np.float32)

    out1 = sess.run(None, {"targets": targets, "target_length_orig": tlen, "h_in": h, "c_in": c})
    out2 = sess.run(None, {
        "targets": np.array([[5]], dtype=np.int64),
        "target_length_orig": tlen, "h_in": out1[2], "c_in": out1[3],
    })
    h_diff = float(np.abs(out2[2] - out1[2]).max())
    c_diff = float(np.abs(out2[3] - out1[3]).max())
    if h_diff > 0.001 and c_diff > 0.001:
        print(f"      [OK] LSTM states evolve correctly (h_diff={h_diff:.4f}, c_diff={c_diff:.4f})")
    else:
        print(f"      [WARNING] States may not be changing (h_diff={h_diff:.6f}, c_diff={c_diff:.6f})")


def consolidate_single_model(onnx_path):
    """Consolidate external data into one <name>.onnx.data file."""
    import onnx

    onnx_path = Path(onnx_path)
    output_dir = onnx_path.parent
    print(f"      Consolidating {onnx_path.name}...")

    try:
        model = onnx.load(str(onnx_path), load_external_data=True)
        data_filename = f"{onnx_path.stem}.onnx.data"
        consolidated_path = output_dir / f"{onnx_path.stem}_consolidated.onnx"

        onnx.save(model, str(consolidated_path),
            save_as_external_data=True, all_tensors_to_one_file=True,
            location=data_filename, size_threshold=1024)

        for pattern in ["onnx__*", "layers.*", "pre_encode.*", "Constant_*"]:
            for ef in output_dir.glob(pattern):
                if ef.exists() and ef.suffix != ".onnx":
                    ef.unlink()
        for ef in output_dir.glob("*.weight"):
            if ef.exists():
                ef.unlink()

        onnx_path.unlink()
        consolidated_path.rename(onnx_path)

        data_file = output_dir / data_filename
        if data_file.exists():
            size_mb = data_file.stat().st_size / (1024 * 1024)
            print(f"      [OK] Consolidated: {onnx_path.name} + {data_filename} ({size_mb:.1f} MB)")
        else:
            print(f"      [OK] {onnx_path.name} (weights embedded)")
        return True
    except Exception as e:
        print(f"      Warning: Could not consolidate {onnx_path.name}: {e}")
        return False


def generate_genai_config(asr_model, output_dir):
    """Generate genai_config.json from the loaded NeMo model."""
    encoder = asr_model.encoder
    decoder = asr_model.decoder
    joint = asr_model.joint

    # Extract dimensions from model
    encoder_hidden = getattr(encoder, 'd_model', 1024)
    encoder_layers = getattr(encoder, 'num_layers', 24)
    encoder_heads = getattr(encoder, 'num_attention_heads',
                   getattr(encoder, '_num_heads', 8))
    if encoder_heads == 8 and hasattr(encoder, 'layers') and len(encoder.layers) > 0:
        layer = encoder.layers[0]
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'h'):
            encoder_heads = layer.self_attn.h
    head_size = encoder_hidden // encoder_heads

    decoder_hidden = getattr(decoder, 'pred_hidden',
                    getattr(decoder, 'd_model', 640))
    decoder_layers = getattr(decoder, 'pred_rnn_layers',
                    getattr(decoder, 'num_layers', 2))

    vocab_size = joint.num_classes_with_blank
    # TDT-specific: get durations and num_extra_outputs
    num_extra_outputs = getattr(joint, '_num_extra_outputs', 0)
    # For TDT, num_classes_with_blank includes duration outputs.
    # The decoder embedding only covers the base vocab (without durations).
    base_vocab_size = vocab_size - num_extra_outputs
    blank_id = base_vocab_size - 1

    # Durations from the loss config
    durations = []
    loss_cfg = asr_model.cfg.get('loss', {})
    if hasattr(loss_cfg, 'get'):
        durations = list(loss_cfg.get('durations', []))
    if not durations:
        # Try decoding config
        dec_cfg = asr_model.cfg.get('decoding', {})
        if hasattr(dec_cfg, 'get'):
            durations = list(dec_cfg.get('durations', []))
    if not durations and num_extra_outputs > 0:
        # Default TDT durations
        durations = [0, 1, 2, 4, 8][:num_extra_outputs]

    encoder_config = {
        "filename": "encoder.onnx",
        "hidden_size": encoder_hidden,
        "num_hidden_layers": encoder_layers,
        "num_attention_heads": encoder_heads,
        "head_size": head_size,
        "inputs": {
            "audio_features": "audio_signal",
            "encoder_input_lengths": "length",
        },
        "outputs": {
            "hidden_states": "outputs",
            "encoder_output_lengths": "encoded_lengths",
        },
    }

    decoder_config = {
        "filename": "decoder.onnx",
        "hidden_size": decoder_hidden,
        "num_hidden_layers": decoder_layers,
        "inputs": {
            "input_ids": "targets",
            "input_ids_length": "target_length_orig",
            "encoder_hidden_states": "encoder_outputs",
        },
        "outputs": {
            "logits": "decoder_output",
        },
        "pipeline": {
            "joint": {
                "filename": "joint.onnx",
                "inputs": ["encoder_output", "decoder_output"],
                "outputs": ["joint_output"],
            }
        },
    }

    # Audio preprocessing config
    preprocessor_cfg = asr_model.cfg.get('preprocessor', {})
    sample_rate = preprocessor_cfg.get('sample_rate', 16000)
    n_mels = preprocessor_cfg.get('features', preprocessor_cfg.get('nfilt', 128))
    window_size = preprocessor_cfg.get('window_size', preprocessor_cfg.get('n_window_size', 0.025))
    window_stride = preprocessor_cfg.get('window_stride', preprocessor_cfg.get('n_window_stride', 0.01))
    if isinstance(window_size, float) and window_size < 1.0:
        frame_length_ms = window_size * 1000
    elif isinstance(window_size, int) and window_size > 100:
        frame_length_ms = window_size / sample_rate * 1000
    else:
        frame_length_ms = 25
    if isinstance(window_stride, float) and window_stride < 1.0:
        frame_shift_ms = window_stride * 1000
    elif isinstance(window_stride, int) and window_stride > 10:
        frame_shift_ms = window_stride / sample_rate * 1000
    else:
        frame_shift_ms = 10

    config = {
        "model": {
            "type": "nemotron_asr",
            "vocab_size": base_vocab_size,
            "context_length": 8192,
            "bos_token_id": 0,
            "eos_token_id": blank_id,
            "pad_token_id": blank_id,
            "encoder": encoder_config,
            "decoder": decoder_config,
            "speech": {
                "config_filename": "audio_processor_config.json",
                "sample_rate": sample_rate,
                "mel_bins": n_mels,
                "frame_length_ms": frame_length_ms,
                "frame_shift_ms": frame_shift_ms,
            },
        },
        "search": {
            "max_length": 8192,
            "min_length": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "do_sample": False,
            "early_stopping": True,
        },
    }

    config_path = output_dir / "genai_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"      [OK] Generated {config_path.name}")

    # Save TDT-specific config separately (not parsed by onnxruntime-genai)
    tdt_config = {
        "durations": durations,
        "num_extra_outputs": num_extra_outputs,
        "vocab_size": base_vocab_size,
        "blank_id": blank_id,
    }
    tdt_config_path = output_dir / "tdt_config.json"
    with open(tdt_config_path, "w") as f:
        json.dump(tdt_config, f, indent=2)
    print(f"      [OK] Generated {tdt_config_path.name}")
    if durations:
        print(f"      TDT durations: {durations}")


def generate_audio_processor_config(asr_model, output_dir):
    """Generate audio_processor_config.json from the loaded NeMo model."""
    preprocessor_cfg = asr_model.cfg.get('preprocessor', {})
    sample_rate = preprocessor_cfg.get('sample_rate', 16000)
    n_fft = preprocessor_cfg.get('n_fft', 512)
    n_mels = preprocessor_cfg.get('features', preprocessor_cfg.get('nfilt', 128))
    window_size = preprocessor_cfg.get('window_size', preprocessor_cfg.get('n_window_size', 0.025))
    window_stride = preprocessor_cfg.get('window_stride', preprocessor_cfg.get('n_window_stride', 0.01))

    if isinstance(window_size, float) and window_size < 1.0:
        window_length = int(window_size * sample_rate)
    elif isinstance(window_size, int):
        window_length = window_size
    else:
        window_length = 400
    if isinstance(window_stride, float) and window_stride < 1.0:
        hop_length = int(window_stride * sample_rate)
    elif isinstance(window_stride, int):
        hop_length = window_stride
    else:
        hop_length = 160

    dither = preprocessor_cfg.get('dither', 0.0)
    preemphasis = preprocessor_cfg.get('preemph', preprocessor_cfg.get('preemphasis', 0.97))
    normalize = preprocessor_cfg.get('normalize', 'none')

    audio_config = {
        "model_type": "speech_features",
        "audio_params": {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "window_length": window_length,
            "window_type": "hann",
            "fmin": 0,
            "fmax": sample_rate // 2,
            "dither": dither,
            "preemphasis": preemphasis,
            "log_zero_guard_type": "add",
            "log_zero_guard_value": 1e-10,
            "normalize": normalize,
            "center": True,
            "mag_power": 2.0,
        },
    }

    config_path = output_dir / "audio_processor_config.json"
    with open(config_path, "w") as f:
        json.dump(audio_config, f, indent=2)
    print(f"      [OK] Generated {config_path.name}")


def export_model(args):
    """Export the Parakeet-TDT ASR model components to ONNX."""
    print("=" * 60)
    print("Parakeet-TDT-0.6B-v3 ONNX Export")
    print("=" * 60)

    print("\n[1/5] Importing NeMo ASR module...")
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError:
        print("Error: NeMo not installed.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")

    print(f"\n[2/5] Loading model: {args.model_name}")
    print("      This may take a few minutes on first run...")
    try:
        if args.model_name.endswith(".nemo"):
            asr_model = nemo_asr.models.ASRModel.restore_from(args.model_name)
        else:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    device = "cpu"
    if args.device == "cuda" and torch.cuda.is_available():
        asr_model = asr_model.cuda()
        device = "cuda"
    else:
        asr_model = asr_model.cpu()
    print(f"      Model loaded on {device.upper()}")
    asr_model.eval()

    # Print TDT-specific info
    joint = asr_model.joint
    num_extra_outputs = getattr(joint, '_num_extra_outputs', 0)
    vocab_size = joint.num_classes_with_blank
    print(f"\n[3/5] Model info:")
    print(f"      Type: TDT (Token-and-Duration Transducer)")
    print(f"      Vocab size (with blank): {vocab_size}")
    print(f"      Num extra outputs (durations): {num_extra_outputs}")
    print(f"      Joint output dim: {vocab_size + num_extra_outputs}")

    print(f"\n[4/5] Exporting to ONNX (opset {args.opset_version})...")

    import torch

    batch_size = 1
    mel_features = 128
    time_steps = 100

    # ---------- Encoder ----------
    print("      Exporting encoder...")
    encoder = asr_model.encoder
    encoder.eval()

    encoder_wrapper = _make_encoder_wrapper(encoder)
    encoder_wrapper.eval()

    dummy_audio = torch.randn(batch_size, mel_features, time_steps)
    dummy_length = torch.tensor([time_steps], dtype=torch.int64)
    if device == "cuda":
        dummy_audio = dummy_audio.cuda()
        dummy_length = dummy_length.cuda()

    encoder_path = output_dir / "encoder.onnx"
    encoder_data = output_dir / "encoder.onnx.data"
    if encoder_data.exists():
        encoder_data.unlink()

    with torch.no_grad():
        torch.onnx.export(
            encoder_wrapper, (dummy_audio, dummy_length), str(encoder_path),
            export_params=True, opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=["audio_signal", "length"],
            output_names=["outputs", "encoded_lengths"],
            dynamic_axes={
                "audio_signal": {0: "batch", 2: "time"},
                "length": {0: "batch"},
                "outputs": {0: "batch", 1: "time_encoded"},
                "encoded_lengths": {0: "batch"},
            },
            dynamo=args.dynamo,
        )
    print(f"      [OK] Encoder exported to {encoder_path}")
    consolidate_single_model(encoder_path)

    # ---------- Decoder (stateful LSTM) ----------
    print("      Exporting stateful decoder...")
    decoder = asr_model.decoder
    decoder.eval()
    hidden_size = getattr(decoder, 'pred_hidden', 640)
    num_layers = getattr(decoder, 'pred_rnn_layers', 2)

    dec_wrapper = _make_stateful_decoder_wrapper(decoder)
    dec_wrapper.eval()

    dummy_targets = torch.zeros(batch_size, 1, dtype=torch.int64)
    dummy_target_length = torch.tensor([1], dtype=torch.int64)
    dummy_h = torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float32)
    dummy_c = torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float32)
    if device == "cuda":
        dummy_targets = dummy_targets.cuda()
        dummy_target_length = dummy_target_length.cuda()
        dummy_h = dummy_h.cuda()
        dummy_c = dummy_c.cuda()

    with torch.no_grad():
        out = dec_wrapper(dummy_targets, dummy_target_length, dummy_h, dummy_c)
        print(f"      Decoder forward check: output={out[0].shape}, h={out[2].shape}, c={out[3].shape}")

    decoder_path = output_dir / "decoder.onnx"
    decoder_data = output_dir / "decoder.onnx.data"
    if decoder_data.exists():
        decoder_data.unlink()

    with torch.no_grad():
        torch.onnx.export(
            dec_wrapper, (dummy_targets, dummy_target_length, dummy_h, dummy_c),
            str(decoder_path), export_params=True, opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=["targets", "target_length_orig", "h_in", "c_in"],
            output_names=["decoder_output", "target_length", "h_out", "c_out"],
            dynamic_axes={
                "targets": {0: "batch", 1: "target_len"},
                "target_length_orig": {0: "batch"},
                "h_in": {1: "batch"}, "c_in": {1: "batch"},
                "decoder_output": {0: "batch", 2: "target_len"},
                "target_length": {0: "batch"},
                "h_out": {1: "batch"}, "c_out": {1: "batch"},
            },
            dynamo=args.dynamo,
        )
    print(f"      [OK] Stateful decoder exported to {decoder_path}")
    consolidate_single_model(decoder_path)
    _verify_stateful_decoder(decoder_path)

    # ---------- Joint network (TDT) ----------
    print("      Exporting TDT joint network...")
    joint = asr_model.joint
    joint.eval()

    encoder_dim = asr_model.cfg.encoder.d_model
    decoder_dim = (
        getattr(decoder, "pred_hidden", None)
        or getattr(decoder, "d_model", None)
        or asr_model.joint.pred_hidden
    )
    print(f"      encoder_dim={encoder_dim}, decoder_dim={decoder_dim}")
    print(f"      Joint output: {vocab_size} (vocab+blank) + {num_extra_outputs} (durations) = {vocab_size + num_extra_outputs}")

    joint_wrapper = _make_joint_wrapper(joint)
    joint_wrapper.eval()

    dummy_enc_out = torch.randn(batch_size, 1, encoder_dim)
    dummy_dec_out = torch.randn(batch_size, 1, decoder_dim)
    if device == "cuda":
        dummy_enc_out = dummy_enc_out.cuda()
        dummy_dec_out = dummy_dec_out.cuda()

    joint_path = output_dir / "joint.onnx"
    joint_data = output_dir / "joint.onnx.data"
    if joint_data.exists():
        joint_data.unlink()

    with torch.no_grad():
        torch.onnx.export(
            joint_wrapper, (dummy_enc_out, dummy_dec_out), str(joint_path),
            export_params=True, opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=["encoder_output", "decoder_output"],
            output_names=["joint_output"],
            dynamic_axes={
                "encoder_output": {0: "batch", 1: "time"},
                "decoder_output": {0: "batch", 1: "target_len"},
                "joint_output": {0: "batch", 1: "time", 2: "target_len"},
            },
            dynamo=args.dynamo,
        )
    print(f"      [OK] TDT joint network exported to {joint_path}")
    consolidate_single_model(joint_path)

    # ---------- Generate config files ----------
    print("\n      Generating config files...")
    generate_genai_config(asr_model, output_dir)
    generate_audio_processor_config(asr_model, output_dir)

    # Summary
    print(f"\n[5/5] Export complete! Generated files:")
    for f in sorted(output_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / (1024 * 1024)
        data_f = output_dir / f"{f.stem}.onnx.data"
        if data_f.exists():
            dsize = data_f.stat().st_size / (1024 * 1024)
            print(f"      - {f.name} ({size_mb:.1f} MB) + {data_f.name} ({dsize:.1f} MB)")
        else:
            print(f"      - {f.name} ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"Model:        {args.model_name}")
    print(f"Type:         TDT (Token-and-Duration Transducer)")
    print(f"Output dir:   {output_dir.absolute()}")
    print(f"ONNX opset:   {args.opset_version}")
    print("=" * 60)


def verify_onnx_model(onnx_path):
    """Verify an exported ONNX model."""
    import onnx
    print(f"\nVerifying {onnx_path}...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("  [OK] ONNX model is valid")
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs: {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")


if __name__ == "__main__":
    args = parse_args()
    export_model(args)

    output_dir = Path(args.output_dir)
    for onnx_file in output_dir.glob("*.onnx"):
        try:
            verify_onnx_model(str(onnx_file))
        except Exception as e:
            print(f"Verification warning for {onnx_file}: {e}")
