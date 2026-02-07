#!/usr/bin/env python3
"""
Export NVIDIA Nemotron-Speech-Streaming-En-0.6b ASR model to ONNX format.

This script downloads the model from HuggingFace and exports it to ONNX format
suitable for inference with ONNX Runtime. The decoder is exported with explicit
LSTM state inputs/outputs for stateful RNNT decoding.

Streaming mode (--streaming):
  Exports the encoder with cache-aware streaming I/O:
    Inputs:  audio_signal[B,128,T], length[B],
             cache_last_channel[B,24,70,1024], cache_last_time[B,24,1024,8],
             cache_last_channel_len[B]
    Outputs: outputs[B,1024,T'], encoded_lengths[B],
             cache_last_channel_next[B,24,70,1024], cache_last_time_next[B,24,1024,8],
             cache_last_channel_len_next[B]

Usage:
    python export_nemotron_to_onnx.py --output_dir ./onnx_models
    python export_nemotron_to_onnx.py --output_dir ./onnx_models --streaming
    python export_nemotron_to_onnx.py --output_dir ./onnx_models --chunk_size 1.12 --streaming
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Nemotron-Speech-Streaming ASR model to ONNX"
    )
    parser.add_argument("--model_name", type=str,
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        help="HuggingFace model name or path to local .nemo file")
    parser.add_argument("--output_dir", type=str, default="./onnx_models",
        help="Directory to save ONNX models")
    parser.add_argument("--chunk_size", type=float, default=1.12,
        choices=[0.08, 0.16, 0.56, 1.12],
        help="Streaming chunk size in seconds")
    parser.add_argument("--opset_version", type=int, default=17,
        help="ONNX opset version")
    parser.add_argument("--device", type=str, default="cpu",
        choices=["cpu", "cuda"], help="Device to use for export")
    parser.add_argument("--streaming", action="store_true",
        help="Export encoder with cache-aware streaming I/O for incremental processing")
    return parser.parse_args()


def get_att_context_size(chunk_size: float) -> list:
    """Get attention context size based on chunk size."""
    return {0.08: [70, 0], 0.16: [70, 1], 0.56: [70, 6], 1.12: [70, 13]}.get(chunk_size, [70, 13])


def get_streaming_cache_shapes(encoder, att_context_size):
    """
    Get cache tensor shapes for streaming export from the NeMo encoder.

    Returns dict with: n_layers, d_model, last_channel_cache_size,
    conv_context, chunk_mel_frames, pre_encode_cache_size
    """
    n_layers = getattr(encoder, 'num_layers', 24)
    d_model = getattr(encoder, 'd_model', 1024)

    # Try to get streaming config from NeMo
    if hasattr(encoder, 'get_streaming_config'):
        cfg = encoder.get_streaming_config()
        last_channel_cache_size = cfg.get('last_channel_cache_size', att_context_size[0])
        shift_size = cfg.get('shift_size', [105, 112])
        chunk_frames = shift_size[1] if len(shift_size) > 1 else shift_size[0]
        pre_encode_cache = cfg.get('pre_encode_cache_size', [0, 9])
    else:
        last_channel_cache_size = att_context_size[0]  # 70
        # Default shift sizes for common chunk sizes
        # chunk_size=1.12s → 112 mel frames, chunk_size=0.56s → 56 frames, etc.
        chunk_frames = 112
        pre_encode_cache = [0, 9]

    # Get conv context from encoder layers
    conv_context = 8  # default: conv_kernel_size(9) - 1 = 8
    if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
        layer = encoder.layers[0]
        if hasattr(layer, 'conv') and hasattr(layer.conv, 'conv'):
            # ConvolutionModule kernel_size
            conv = layer.conv.conv
            if hasattr(conv, 'kernel_size'):
                ks = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
                conv_context = ks - 1

    return {
        'n_layers': n_layers,
        'd_model': d_model,
        'last_channel_cache_size': last_channel_cache_size,
        'conv_context': conv_context,
        'chunk_mel_frames': chunk_frames,
        'pre_encode_cache_size': pre_encode_cache,
    }


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


def _make_streaming_encoder_wrapper(encoder):
    """
    Wrap the NeMo CacheAware encoder for streaming export.

    The encoder's forward_for_export() accepts cache tensors for incremental
    chunk-by-chunk processing. Cache tensors:
      - cache_last_channel: [B, n_layers, cache_len, d_model] — MHA left-context
      - cache_last_time:    [B, n_layers, d_model, conv_ctx]  — causal conv buffer
      - cache_last_channel_len: [B] — how many cache positions are filled

    Returns updated caches alongside encoder output.
    """
    import torch
    import torch.nn as nn

    class StreamingEncoderWrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, audio_signal, length,
                    cache_last_channel, cache_last_time, cache_last_channel_len):
            # NeMo CacheAware conformer expects cache in [n_layers, B, ...] order,
            # but ONNX convention uses [B, n_layers, ...] for batch-first.
            # Transpose: [B, n_layers, cache_len, d_model] -> [n_layers, B, cache_len, d_model]
            cache_ch = cache_last_channel.transpose(0, 1).contiguous()
            cache_tm = cache_last_time.transpose(0, 1).contiguous()

            encoded, encoded_len, cache_ch_next, cache_tm_next, cache_len_next = \
                self.enc.forward_for_export(
                    audio_signal=audio_signal,
                    length=length,
                    cache_last_channel=cache_ch,
                    cache_last_time=cache_tm,
                    cache_last_channel_len=cache_last_channel_len,
                )

            # Transpose back: [n_layers, B, ...] -> [B, n_layers, ...]
            cache_ch_next = cache_ch_next.transpose(0, 1).contiguous()
            cache_tm_next = cache_tm_next.transpose(0, 1).contiguous()

            return encoded, encoded_len, cache_ch_next, cache_tm_next, cache_len_next

    return StreamingEncoderWrapper(encoder)


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
    """Wrap the NeMo RNNTJoint so torch.onnx.export can trace it."""
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


def _verify_streaming_encoder(encoder_path, batch_size, mel_features, chunk_frames,
                               n_layers, d_model, cache_len, conv_context):
    """ORT smoke-test: two chunks, verify cache states evolve."""
    import numpy as np
    import onnxruntime as ort

    print("      Verifying streaming encoder with ONNX Runtime...")
    sess = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])

    audio = np.random.randn(batch_size, mel_features, chunk_frames).astype(np.float32)
    length = np.array([chunk_frames], dtype=np.int64)
    cache_ch = np.zeros((batch_size, n_layers, cache_len, d_model), dtype=np.float32)
    cache_tm = np.zeros((batch_size, n_layers, d_model, conv_context), dtype=np.float32)
    cache_len_val = np.zeros((batch_size,), dtype=np.int64)

    # Run chunk 1
    out1 = sess.run(None, {
        "audio_signal": audio, "length": length,
        "cache_last_channel": cache_ch, "cache_last_time": cache_tm,
        "cache_last_channel_len": cache_len_val,
    })
    enc_out1, enc_len1, cache_ch1, cache_tm1, cache_len1 = out1

    # Run chunk 2 with updated caches
    audio2 = np.random.randn(batch_size, mel_features, chunk_frames).astype(np.float32)
    out2 = sess.run(None, {
        "audio_signal": audio2, "length": length,
        "cache_last_channel": cache_ch1, "cache_last_time": cache_tm1,
        "cache_last_channel_len": cache_len1,
    })

    ch_diff = float(np.abs(out2[2] - cache_ch1).max())
    tm_diff = float(np.abs(out2[3] - cache_tm1).max())
    print(f"      Encoder output shape: {enc_out1.shape}")
    print(f"      Encoded length: {enc_len1}")
    print(f"      cache_last_channel_len after chunk 1: {cache_len1}")
    if ch_diff > 0.001 or tm_diff > 0.001:
        print(f"      [OK] Encoder caches evolve (ch_diff={ch_diff:.4f}, tm_diff={tm_diff:.4f})")
    else:
        print(f"      [WARNING] Caches may not be updating (ch_diff={ch_diff:.6f}, tm_diff={tm_diff:.6f})")


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


def export_model(args):
    """Export the Nemotron ASR model components to ONNX."""
    print("=" * 60)
    print("Nemotron-Speech-Streaming ONNX Export")
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

    print(f"\n[3/5] Configuring streaming context...")
    att_context_size = get_att_context_size(args.chunk_size)
    print(f"      Chunk size: {args.chunk_size}s")
    print(f"      Attention context size: {att_context_size}")
    encoder = asr_model.encoder
    encoder.eval()
    if hasattr(encoder, "set_default_att_context_size"):
        encoder.set_default_att_context_size(att_context_size)
        print("      [OK] Set encoder attention context size")

    print(f"\n[4/5] Exporting to ONNX (opset {args.opset_version})...")

    batch_size = 1
    mel_features = 128  # Nemotron uses 128 mel bins
    time_steps = 100    # ~1 second of audio

    # ---------- Encoder ----------
    if args.streaming:
        print("      Exporting STREAMING encoder (cache-aware)...")
        streaming_wrapper = _make_streaming_encoder_wrapper(encoder)
        streaming_wrapper.eval()

        # Get streaming config from encoder to determine cache shapes
        cache_cfg = get_streaming_cache_shapes(encoder, att_context_size)
        n_layers = cache_cfg['n_layers']
        d_model = cache_cfg['d_model']
        last_channel_cache_size = cache_cfg['last_channel_cache_size']
        conv_context = cache_cfg['conv_context']
        chunk_frames = cache_cfg['chunk_mel_frames']

        print(f"      Streaming config:")
        print(f"        n_layers={n_layers}, d_model={d_model}")
        print(f"        last_channel_cache_size={last_channel_cache_size}")
        print(f"        conv_context={conv_context}")
        print(f"        chunk_mel_frames={chunk_frames}")

        dummy_audio = torch.randn(batch_size, mel_features, chunk_frames)
        dummy_length = torch.tensor([chunk_frames], dtype=torch.int64)
        dummy_cache_ch = torch.zeros(batch_size, n_layers, last_channel_cache_size, d_model)
        dummy_cache_tm = torch.zeros(batch_size, n_layers, d_model, conv_context)
        dummy_cache_len = torch.zeros(batch_size, dtype=torch.int64)

        if device == "cuda":
            dummy_audio = dummy_audio.cuda()
            dummy_length = dummy_length.cuda()
            dummy_cache_ch = dummy_cache_ch.cuda()
            dummy_cache_tm = dummy_cache_tm.cuda()
            dummy_cache_len = dummy_cache_len.cuda()

        # Verify forward pass works
        with torch.no_grad():
            test_out = streaming_wrapper(dummy_audio, dummy_length,
                                         dummy_cache_ch, dummy_cache_tm, dummy_cache_len)
            print(f"      Forward check: output={test_out[0].shape}, "
                  f"cache_ch_next={test_out[2].shape}, cache_tm_next={test_out[3].shape}")

        encoder_path = output_dir / "encoder.onnx"
        encoder_data = output_dir / "encoder.onnx.data"
        if encoder_data.exists():
            encoder_data.unlink()

        with torch.no_grad():
            torch.onnx.export(
                streaming_wrapper,
                (dummy_audio, dummy_length, dummy_cache_ch, dummy_cache_tm, dummy_cache_len),
                str(encoder_path),
                export_params=True, opset_version=args.opset_version,
                do_constant_folding=True,
                input_names=["audio_signal", "length",
                             "cache_last_channel", "cache_last_time",
                             "cache_last_channel_len"],
                output_names=["outputs", "encoded_lengths",
                              "cache_last_channel_next", "cache_last_time_next",
                              "cache_last_channel_len_next"],
                dynamic_axes={
                    "audio_signal": {0: "batch", 2: "time"},
                    "length": {0: "batch"},
                    "cache_last_channel": {0: "batch"},
                    "cache_last_time": {0: "batch"},
                    "cache_last_channel_len": {0: "batch"},
                    "outputs": {0: "batch", 1: "time_encoded"},
                    "encoded_lengths": {0: "batch"},
                    "cache_last_channel_next": {0: "batch"},
                    "cache_last_time_next": {0: "batch"},
                    "cache_last_channel_len_next": {0: "batch"},
                },
                dynamo=False,
            )
        print(f"      [OK] Streaming encoder exported to {encoder_path}")
        consolidate_single_model(encoder_path)

        # Verify streaming encoder with ORT
        _verify_streaming_encoder(encoder_path, batch_size, mel_features, chunk_frames,
                                  n_layers, d_model, last_channel_cache_size, conv_context)
    else:
        print("      Exporting encoder (non-streaming)...")
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
                dynamo=False,
            )
        print(f"      [OK] Encoder exported to {encoder_path}")
        consolidate_single_model(encoder_path)

    # ---------- Decoder (stateful LSTM) ----------
    print("      Exporting stateful decoder...")
    decoder = asr_model.decoder
    decoder.eval()
    hidden_size = 640
    num_layers = 2

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
            dynamo=False,
        )
    print(f"      [OK] Stateful decoder exported to {decoder_path}")
    consolidate_single_model(decoder_path)
    _verify_stateful_decoder(decoder_path)

    # ---------- Joint network ----------
    print("      Exporting joint network...")
    joint = asr_model.joint
    joint.eval()

    encoder_dim = asr_model.cfg.encoder.d_model
    decoder_dim = (
        getattr(decoder, "pred_hidden", None)
        or getattr(decoder, "d_model", None)
        or asr_model.joint.pred_hidden
    )
    print(f"      encoder_dim={encoder_dim}, decoder_dim={decoder_dim}")

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
            dynamo=False,
        )
    print(f"      [OK] Joint network exported to {joint_path}")
    consolidate_single_model(joint_path)

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
    print(f"Chunk size:   {args.chunk_size}s")
    print(f"Streaming:    {args.streaming}")
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
