#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Calibrate symmetric per-channel KV-cache scales for Qwen3.8-Flash-Next NVFP4 packages
(both the 12 target sparse-attention layers in dense/model.onnx and the 1 MTP head sparse-attention
layer in head_kv/model.onnx).

Reuses the corpus, tokenization, and threshold estimation from
onnxruntime_genai.models.quantization.kv_cache_calibration, while handling the Qwen3.8-Flash-Next
PLE input (`ple_embeddings.1`), `past_position_ids`, `logits_indices`, the partial-rotary
(`rotary_dim = 64` of `head_size = 256`) rotation-invariant pair envelope, and the MTP head KV model.
Embedding-only graphs use the package's CPU embedding session for the prompt and shifted head inputs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import torch
from onnxruntime_genai.models.quantization.kv_cache_calibration import (
    CORPUS,
    _get_quant_type_max,
    _subsample,
    _tokenize_corpus,
)
from transformers import AutoTokenizer

# Every run calibrates both quantizations; --out receives the one named by --quant-type.
QUANT_TYPES = ("int8_per_channel", "fp8_per_channel")
METHODS = ("minmax", "percentile")


def validate_method(method):
    if method not in METHODS:
        raise ValueError(
            "dual INT8/FP8 calibration supports minmax or percentile; MSE models integer rounding, not FP8"
        )


def custom_ops_paths(pkg, options, name, fallback=None):
    if not options.get("custom_ops_library") and not options.get("custom_ops_libraries"):
        options = fallback or {}
    libraries = ([options["custom_ops_library"]] if options.get("custom_ops_library") else []) + options.get(
        "custom_ops_libraries", []
    )
    paths = list(dict.fromkeys((pkg / lib).resolve() for lib in libraries))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{name} custom-op library not found: {path}")
    return paths


def create_calibration_sessions(ort, pkg):
    """Loads the dense/head-KV graphs with their configured custom ops on separate session options.

    An auxiliary session's explicit library list replaces the decoder's list; if it specifies neither library field,
    the decoder's libraries are used. Relative paths are resolved from the package, as in the Engine packages.
    """
    pkg = pkg.resolve()
    model = json.loads((pkg / "genai_config.json").read_text())["model"]
    parent = model.get("decoder", {}).get("session_options", {})
    prefill = model.get("qwen38_graph", {}).get("prefill", {})
    plans = []
    for name in ("dense", "head_kv"):
        graph = prefill.get(name, {})
        paths = custom_ops_paths(pkg, graph.get("session_options", {}), name, parent)
        plans.append((pkg / graph.get("filename", f"{name}/model.onnx"), paths))

    sessions = []
    providers = [("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"})]
    for filename, libraries in plans:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
        for library in libraries:
            so.register_custom_ops_library(str(library))
        print(f"Loading calibration session from {filename}...", flush=True)
        sessions.append(ort.InferenceSession(str(filename), sess_options=so, providers=providers))
    return sessions


def create_embedding_session(ort, pkg):
    pkg = pkg.resolve()
    graph = json.loads((pkg / "genai_config.json").read_text())["model"].get("embedding", {})
    if not graph.get("filename"):
        raise ValueError("inputs_embeds calibration requires model.embedding.filename")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.enable_cpu_mem_arena = False
    for library in custom_ops_paths(pkg, graph.get("session_options", {}), "embedding"):
        so.register_custom_ops_library(str(library))
    session = ort.InferenceSession(str(pkg / graph["filename"]), sess_options=so, providers=["CPUExecutionProvider"])
    return (
        session,
        graph.get("inputs", {}).get("input_ids", "input_ids"),
        graph.get("outputs", {}).get("embeddings", "inputs_embeds"),
    )


def run_embeddings(ort, session, input_name, output_name, tokens: np.ndarray, device) -> torch.Tensor:
    meta = next((meta for meta in session.get_inputs() if meta.name == input_name), None)
    if meta is None or len(meta.shape) not in (1, 2):
        raise ValueError(f"embedding input {input_name!r} must exist and have rank 1 or 2")
    ids = tokens.reshape(-1) if len(meta.shape) == 1 else tokens
    binding = session.io_binding()
    binding.bind_cpu_input(input_name, np.ascontiguousarray(ids, dtype=np.int64))
    binding.bind_output(output_name, "cpu")
    session.run_with_iobinding(binding)
    output = binding.get_outputs()[0]
    if output.data_type() == "tensor(bfloat16)":
        # CPU/DirectML wheels need not expose DLPack, and OrtValue.numpy() does not support BF16.
        # Copy the raw CPU storage so the tensor also survives the binding's lifetime.
        raw = bytearray(ctypes.string_at(output.data_ptr(), int(np.prod(output.shape())) * 2))
        embeddings = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(output.shape())
    else:
        embeddings = torch.from_numpy(output.numpy().copy())
    return embeddings.reshape(1, tokens.shape[1], -1).to(device=device).contiguous()


def ensure_distinct_outputs(output_paths: dict[str, Path]) -> None:
    """Raises ValueError when two quantizations would be written to the same file."""
    owners: dict[Path, str] = {}
    for qtype, path in output_paths.items():
        resolved = Path(os.path.normcase(path.resolve()))
        if resolved in owners:
            raise ValueError(f"{owners[resolved]} and {qtype} calibrations would both be written to {path}")
        owners[resolved] = qtype


def plan_output_paths(out_json: Path, quant_type: str) -> dict[str, Path]:
    """Maps each calibrated quantization to its own output file.

    `out_json` receives the `quant_type` calibration. The other quantization is written beside it under a name built from the
    quantization names, not from the rest of the basename: the `quant_type` token of the name is swapped for the other type
    (kv_scales_int8_per_channel.json -> kv_scales_fp8_per_channel.json), and a name without it gets the other type as an extra
    suffix (scales.json -> scales.fp8_per_channel.json). Raises ValueError, before any calibration work, for an unsupported
    `quant_type` or when two quantizations would share a file.
    """
    if quant_type not in QUANT_TYPES:
        raise ValueError(f"quant_type must be one of {', '.join(QUANT_TYPES)}, got {quant_type!r}")
    other_type = next(qtype for qtype in QUANT_TYPES if qtype != quant_type)
    if quant_type in out_json.name:
        other_name = out_json.name.replace(quant_type, other_type)
    else:
        other_name = f"{out_json.stem}.{other_type}{out_json.suffix}"
    by_type = {quant_type: out_json, other_type: out_json.with_name(other_name)}
    output_paths = {qtype: by_type[qtype] for qtype in QUANT_TYPES}
    ensure_distinct_outputs(output_paths)
    return output_paths


def pair_envelope_partial(x: np.ndarray, num_kv_heads: int, head_size: int, rotary_dim: int) -> np.ndarray:
    """Rotation-invariant envelope for partial-rotary K (rotary_dim <= head_size).

    Channels [0, rotary_dim) are rotated in pairs (d, d + rotary_dim/2); channels [rotary_dim, head_size)
    are unrotated and keep their raw magnitude.
    """
    if rotary_dim <= 0:
        return np.abs(x).astype(np.float32)
    if rotary_dim % 2 != 0 or rotary_dim > head_size:
        raise ValueError(f"Invalid rotary_dim={rotary_dim} for head_size={head_size}")
    half = rotary_dim // 2
    heads = x.reshape(x.shape[0], num_kv_heads, head_size).astype(np.float32)
    rot_norm = np.sqrt(heads[:, :, :half] ** 2 + heads[:, :, half:rotary_dim] ** 2)
    out = np.empty_like(heads, dtype=np.float32)
    out[:, :, :half] = rot_norm
    out[:, :, half:rotary_dim] = rot_norm
    if rotary_dim < head_size:
        out[:, :, rotary_dim:] = np.abs(heads[:, :, rotary_dim:])
    return out.reshape(x.shape[0], -1)


def discover_kv_outputs(output_names):
    keys, values = {}, {}
    for name in output_names:
        parts = name.split(".")
        if len(parts) == 3 and parts[0] == "present" and parts[1].isdigit():
            if parts[2] == "key":
                keys[int(parts[1])] = name
            elif parts[2] == "value":
                values[int(parts[1])] = name
    if not keys or keys.keys() != values.keys():
        raise ValueError(
            f"expected paired KV outputs: key-only layers {sorted(keys.keys() - values.keys())}, value-only layers {sorted(values.keys() - keys.keys())}"
        )
    return keys, values


def calibrate_package(
    pkg: Path,
    checkpoint: Path,
    out_json: Path,
    quant_type: str = "int8_per_channel",
    method: str = "percentile",
    percentile: float = 99.99,
    num_seqs: int = 24,
    target_seq: int = 512,
    rotary_dim: int = 64,
    sample_budget: int = 16384,
) -> dict:
    output_paths = plan_output_paths(out_json, quant_type)
    validate_method(method)

    import onnxruntime as ort  # noqa: PLC0415
    from mobius._configs import Qwen4ExpConfig  # noqa: PLC0415
    from mobius.generation._qwen4_exp_ple import Qwen4ExpCPUPLELookup  # noqa: PLC0415
    from mobius.integrations.transformers._builder import _load_transformers_config  # noqa: PLC0415

    provider = os.environ.get("QWEN_PROVIDER") or str(pkg / "provider/libonnxruntime_providers_cuda.so")
    if (
        "CUDAExecutionProvider" not in ort.get_available_providers()
        and hasattr(ort, "register_execution_provider_library")
        and Path(provider).exists()
    ):
        ort.register_execution_provider_library("CUDAExecutionProvider", provider)

    tokenizer = AutoTokenizer.from_pretrained(str(pkg))
    seqs = _tokenize_corpus(tokenizer, num_seqs, target_seq, CORPUS)

    hf_config, _ = _load_transformers_config(str(checkpoint), revision=None, trust_remote_code=False)
    config = Qwen4ExpConfig.from_transformers(hf_config)
    ple_layer = next(int(i) - 1 for i in (config.ple_layer_ids or []))
    lookup = Qwen4ExpCPUPLELookup(str(checkpoint), config, ple_layer)

    dense_sess, head_kv_sess = create_calibration_sessions(ort, pkg)
    dense_inputs = {meta.name for meta in dense_sess.get_inputs()}
    head_inputs = {meta.name for meta in head_kv_sess.get_inputs()}
    embedding = (
        create_embedding_session(ort, pkg)
        if "inputs_embeds" in dense_inputs or "inputs_embeds" in head_inputs
        else None
    )

    num_kv_heads = 2
    head_size = 256
    channels = num_kv_heads * head_size

    present_keys, present_values = discover_kv_outputs([o.name for o in dense_sess.get_outputs()])
    layer_ids = sorted(present_keys)
    num_layers = len(layer_ids)
    print(f"Target sparse attention layers ({num_layers}): {layer_ids}", flush=True)

    # Initial empty state tensors on device
    device = torch.device("cuda:0")
    init_states = {}
    for meta in dense_sess.get_inputs():
        name = meta.name
        if name in (
            "input_ids",
            "inputs_embeds",
            "attention_mask",
            "position_ids",
            "logits_indices",
        ) or name.startswith("ple_embeddings."):
            continue
        shape = [1 if not isinstance(d, int) or d <= 0 else d for d in meta.shape]
        if name == "past_position_ids":
            shape = [1, 0]
            dt = torch.int64
        elif name.endswith((".key", ".value")):
            shape[2] = 0
            dt = torch.bfloat16
        elif name.endswith(".index_key"):
            shape[1] = 0
            dt = torch.bfloat16
        elif meta.type == "tensor(float)":
            dt = torch.float32
        elif meta.type == "tensor(bfloat16)":
            dt = torch.bfloat16
        else:
            dt = torch.int64
        init_states[name] = torch.zeros(shape, dtype=dt, device=device)

    # All 12 target layers + 1 head layer (index num_layers)
    total_entries = num_layers + 1
    k_amax = np.zeros((total_entries, channels), dtype=np.float32)
    v_amax = np.zeros((total_entries, channels), dtype=np.float32)
    k_raw_buf = [[] for _ in range(total_entries)]
    v_raw_buf = [[] for _ in range(total_entries)]
    k_env_buf = [[] for _ in range(total_entries)]
    v_env_buf = [[] for _ in range(total_entries)]
    per_seq_budget = max(1, sample_budget // max(1, len(seqs)))

    for pi, seq_np in enumerate(seqs):
        seq_len = int(seq_np.shape[1])
        prompt_list = seq_np[0].tolist()
        ctx = lookup.initial_context(1)
        ple_emb, _ = lookup([prompt_list], ctx)
        ple_dev = ple_emb.to(device=device, dtype=torch.bfloat16).contiguous()
        embeddings = None
        if embedding is not None:
            embeddings = run_embeddings(ort, *embedding, seq_np, device)

        binding = dense_sess.io_binding()
        keep = []

        def bind_in(name: str, t: torch.Tensor, b=binding, keep_ref=keep):
            t = t.contiguous()
            ov = ort.OrtValue.from_dlpack(torch.utils.dlpack.to_dlpack(t))
            keep_ref.append((t, ov))
            b.bind_ortvalue_input(name, ov)

        if "input_ids" in dense_inputs:
            bind_in("input_ids", torch.from_numpy(seq_np).to(device=device, dtype=torch.int64))
        if "inputs_embeds" in dense_inputs:
            assert embeddings is not None
            bind_in("inputs_embeds", embeddings)
        bind_in("attention_mask", torch.ones((1, seq_len), dtype=torch.int64, device=device))
        bind_in("position_ids", torch.arange(seq_len, dtype=torch.int64, device=device).reshape(1, seq_len))
        bind_in("logits_indices", torch.tensor([seq_len - 1], dtype=torch.int64, device=device))
        bind_in(f"ple_embeddings.{ple_layer}", ple_dev)
        for sname, stensor in init_states.items():
            bind_in(sname, stensor)

        want_outs = (
            [present_keys[lid] for lid in layer_ids] + [present_values[lid] for lid in layer_ids] + ["hidden_states"]
        )
        for oname in want_outs:
            binding.bind_output(oname, "cuda", 0)

        dense_sess.run_with_iobinding(binding)
        ort_outs = binding.get_outputs()
        out_map = {name: torch.from_dlpack(ov) for name, ov in zip(want_outs, ort_outs, strict=True)}

        for i, lid in enumerate(layer_ids):
            kt = out_map[present_keys[lid]].float().cpu().numpy().reshape(num_kv_heads, seq_len, head_size)
            vt = out_map[present_values[lid]].float().cpu().numpy().reshape(num_kv_heads, seq_len, head_size)
            k2 = np.transpose(kt, (1, 0, 2)).reshape(seq_len, channels)
            v2 = np.transpose(vt, (1, 0, 2)).reshape(seq_len, channels)
            k_raw_buf[i].append(k2.copy())
            v_raw_buf[i].append(v2.copy())
            ke = pair_envelope_partial(k2, num_kv_heads, head_size, rotary_dim)
            ve = np.abs(v2)
            k_amax[i] = np.maximum(k_amax[i], ke.max(axis=0))
            v_amax[i] = np.maximum(v_amax[i], ve.max(axis=0))
            k_env_buf[i].append(_subsample(ke, per_seq_budget).astype(np.float16))
            v_env_buf[i].append(_subsample(v2, per_seq_budget).astype(np.float16))

        # Run head_kv on rows 1..seq_len-1
        hidden = out_map["hidden_states"][:, :-1, :].contiguous()
        m = seq_len - 1
        hb = head_kv_sess.io_binding()
        if "input_ids" in head_inputs:
            bind_in("input_ids", torch.from_numpy(seq_np[:, 1:]).to(device=device, dtype=torch.int64), hb)
        if "inputs_embeds" in head_inputs:
            assert embeddings is not None
            bind_in("inputs_embeds", embeddings[:, 1:, :], hb)
        bind_in("attention_mask", torch.ones((1, m), dtype=torch.int64, device=device), hb)
        bind_in("position_ids", torch.arange(1, seq_len, dtype=torch.int64, device=device).reshape(1, m), hb)
        bind_in("past_position_ids", torch.zeros((1, 0), dtype=torch.int64, device=device), hb)
        bind_in("past_key_values.0.index_key", torch.zeros((1, 0, 128), dtype=torch.bfloat16, device=device), hb)
        bind_in("hidden_states", hidden, hb)
        for oname in ("key", "value", "index_key"):
            hb.bind_output(oname, "cuda", 0)
        head_kv_sess.run_with_iobinding(hb)
        h_outs = {
            name: torch.from_dlpack(ov)
            for name, ov in zip(("key", "value", "index_key"), hb.get_outputs(), strict=True)
        }
        hk = h_outs["key"].float().cpu().numpy()
        if hk.ndim == 4:
            hk2 = np.transpose(hk.reshape(num_kv_heads, m, head_size), (1, 0, 2)).reshape(m, channels)
        else:
            hk2 = hk.reshape(m, channels)
        hv2 = h_outs["value"].float().cpu().numpy().reshape(m, channels)
        hi = num_layers
        k_raw_buf[hi].append(hk2.copy())
        v_raw_buf[hi].append(hv2.copy())
        hke = pair_envelope_partial(hk2, num_kv_heads, head_size, rotary_dim)
        hve = np.abs(hv2)
        k_amax[hi] = np.maximum(k_amax[hi], hke.max(axis=0))
        v_amax[hi] = np.maximum(v_amax[hi], hve.max(axis=0))
        k_env_buf[hi].append(_subsample(hke, per_seq_budget).astype(np.float16))
        v_env_buf[hi].append(_subsample(hv2, per_seq_budget).astype(np.float16))

        if (pi + 1) % 6 == 0 or pi == 0:
            print(
                f"  [{pi + 1}/{len(seqs)}] L3 k_amax={k_amax[0].max():.4f} v_amax={v_amax[0].max():.4f} | head k_amax={k_amax[hi].max():.4f} v_amax={v_amax[hi].max():.4f}",
                flush=True,
            )

    return calibrate_quant_types(
        output_paths,
        k_amax,
        v_amax,
        k_env_buf,
        v_env_buf,
        k_raw_buf,
        v_raw_buf,
        layer_ids=layer_ids,
        method=method,
        percentile=percentile,
        rotary_dim=rotary_dim,
    )


def calibrate_quant_types(
    output_paths: dict[str, Path],
    k_amax: np.ndarray,
    v_amax: np.ndarray,
    k_env_buf: list,
    v_env_buf: list,
    k_raw_buf: list,
    v_raw_buf: list,
    *,
    layer_ids: list[int],
    method: str,
    percentile: float,
    rotary_dim: int,
) -> dict:
    """Turns the captured K/V statistics into the scales of every quantization and writes each to its own file.

    The buffers have one entry per target layer followed by one for the MTP head (`len(layer_ids) + 1` in total).
    """
    validate_method(method)
    for name, arrays in (
        ("k_amax", [k_amax]),
        ("v_amax", [v_amax]),
        ("k_env", [a for entry in k_env_buf for a in entry]),
        ("v_env", [a for entry in v_env_buf for a in entry]),
        ("k_raw", [a for entry in k_raw_buf for a in entry]),
        ("v_raw", [a for entry in v_raw_buf for a in entry]),
    ):
        if any(not np.isfinite(a).all() for a in arrays):
            raise ValueError(f"non-finite calibration capture: {name}")
    num_layers = len(layer_ids)
    total_entries = num_layers + 1
    results = {}
    for qtype in output_paths:
        qmax = _get_quant_type_max(qtype)
        is_fp8 = qtype.startswith("fp8")
        k_thr = k_amax.copy()
        v_thr = v_amax.copy()
        if method == "percentile":
            for i in range(total_entries):
                ks = np.concatenate(k_env_buf[i], axis=0).astype(np.float32)
                vs = np.concatenate(v_env_buf[i], axis=0).astype(np.float32)
                k_thr[i] = np.percentile(np.abs(ks), percentile, axis=0)
                v_thr[i] = np.percentile(np.abs(vs), percentile, axis=0)
        k_scales = np.maximum(k_thr, 1e-6) / qmax
        v_scales = np.maximum(v_thr, 1e-6) / qmax
        if not np.isfinite(k_scales).all() or not np.isfinite(v_scales).all():
            raise ValueError(f"non-finite {qtype} calibration scales")

        # Compute empirical reconstruction relative L2 error across all captured tokens
        k_errs, v_errs = [], []
        for i in range(total_entries):
            kr = np.concatenate(k_raw_buf[i], axis=0)
            vr = np.concatenate(v_raw_buf[i], axis=0)
            ks_i = k_scales[i][None, :]
            vs_i = v_scales[i][None, :]
            if is_fp8:
                kq = np.clip(kr / ks_i, -448.0, 448.0).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * ks_i
                vq = np.clip(vr / vs_i, -448.0, 448.0).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * vs_i
            else:
                kq = np.clip(np.rint(kr / ks_i), -128.0, 127.0) * ks_i
                vq = np.clip(np.rint(vr / vs_i), -128.0, 127.0) * vs_i
            k_norm, v_norm = float(np.linalg.norm(kr)), float(np.linalg.norm(vr))
            k_errs.append(float(np.linalg.norm(kr - kq)) / k_norm if k_norm else 0.0)
            v_errs.append(float(np.linalg.norm(vr - vq)) / v_norm if v_norm else 0.0)
        if not np.isfinite(k_errs).all() or not np.isfinite(v_errs).all():
            raise ValueError(f"non-finite {qtype} calibration error statistics")

        payload = {
            "scales": {
                "k_scales": k_scales[:num_layers].tolist(),
                "v_scales": v_scales[:num_layers].tolist(),
                "head_k_scales": [k_scales[num_layers].tolist()],
                "head_v_scales": [v_scales[num_layers].tolist()],
            },
            "qmax": qmax,
            "quant_type": qtype,
            "method": method,
            "percentile": percentile,
            "rotary_dim": rotary_dim,
            "layer_ids": layer_ids,
            "head_layer_ids": [0],
            "stats": {
                "k_scale_min": float(k_scales.min()),
                "k_scale_max": float(k_scales.max()),
                "v_scale_min": float(v_scales.min()),
                "v_scale_max": float(v_scales.max()),
                "mean_k_rel_l2": float(np.mean(k_errs[:num_layers])),
                "mean_v_rel_l2": float(np.mean(v_errs[:num_layers])),
                "head_k_rel_l2": float(k_errs[num_layers]),
                "head_v_rel_l2": float(v_errs[num_layers]),
            },
        }
        results[qtype] = payload
    # Serialize both payloads before publishing either: invalid scales/statistics must not leave a partial calibration.
    serialized = {qtype: json.dumps(payload, indent=2, allow_nan=False) + "\n" for qtype, payload in results.items()}
    for qtype, target_path in output_paths.items():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(serialized[qtype])
        print(f"Wrote {target_path}: stats={json.dumps(results[qtype]['stats'])}", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="v12b package folder")
    parser.add_argument("--checkpoint", type=Path, required=True, help="HuggingFace checkpoint folder for PLE lookup")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output calibrated scale JSON path of --quant-type; the other quantization is written beside it "
        "(kv_scales_int8_per_channel.json -> kv_scales_fp8_per_channel.json, scales.json -> scales.fp8_per_channel.json)",
    )
    parser.add_argument("--quant-type", default="int8_per_channel", choices=QUANT_TYPES)
    parser.add_argument(
        "--method", default="percentile", choices=METHODS, help="MSE is unsupported for the FP8 companion"
    )
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument("--num-seqs", type=int, default=24)
    parser.add_argument("--target-seq", type=int, default=512)
    parser.add_argument("--rotary-dim", type=int, default=64)
    args = parser.parse_args()
    calibrate_package(
        args.package.resolve(),
        args.checkpoint.resolve(),
        args.out.resolve(),
        quant_type=args.quant_type,
        method=args.method,
        percentile=args.percentile,
        num_seqs=args.num_seqs,
        target_seq=args.target_seq,
        rotary_dim=args.rotary_dim,
    )


if __name__ == "__main__":
    main()
