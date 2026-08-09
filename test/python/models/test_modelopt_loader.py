# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Structural test for the Model Optimizer (NVFP4/FP8) loader.

Builds a tiny synthetic modelopt-style checkpoint (one linear-attention layer,
one full-attention layer, plus globals) and verifies that ModeloptModel:
  * builds the module tree the ONNX Runtime GenAI builder walks,
    * preserves FP8 and NVFP4 tensors in their original quantized formats, and
    * materializes routed experts for native QMoE preprocessing.
"""

import importlib.util
import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file


def _load_quantized_model_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "python", "py", "models", "quantized_model.py"
    )
    spec = importlib.util.spec_from_file_location("_genai_quantized_model_under_test", os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QM = _load_quantized_model_module()

_FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _make_nvfp4(n, k, rng):
    """Return (weight_u8[N,K/2], weight_scale_e4m3[N,K/16], weight_scale_2 scalar, ref_bf16[N,K])."""
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint8)  # e2m1 codes [N,K]
    low, high = codes[:, 0::2], codes[:, 1::2]
    weight_u8 = ((high << 4) | low).astype(np.uint8)  # [N,K/2]
    block = rng.integers(0, 0x40, size=(n, k // 16), dtype=np.uint8)
    ws = torch.from_numpy(block).view(torch.float8_e4m3fn)  # [N,K/16]
    g = np.float32(rng.uniform(0.02, 0.2))
    mag = _FP4_LUT[codes & 0x7]
    val = np.where((codes & 0x8) > 0, -mag, mag)
    bs = ws.to(torch.float32).numpy().repeat(16, axis=1)
    ref = torch.from_numpy((val * bs * float(g)).astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(weight_u8), ws, torch.tensor(g, dtype=torch.float32), ref


def _make_fp8(out_f, in_f, rng):
    """Return (weight_f8[out,in], weight_scale scalar, ref_bf16)."""
    raw = rng.integers(0, 0x40, size=(out_f, in_f), dtype=np.uint8)
    w = torch.from_numpy(raw).view(torch.float8_e4m3fn)
    s = torch.tensor(np.float32(rng.uniform(0.02, 0.2)))
    ref = (w.to(torch.float32) * float(s)).to(torch.bfloat16)
    return w, s, ref


def test_modelopt_nvfp4_dequant_helper_matches_reference():
    weight, block_scale, global_scale, reference = _make_nvfp4(8, 32, np.random.default_rng(0))

    actual = QM.ModeloptModel._dequant_nvfp4(weight, block_scale, global_scale, name="lm_head")

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def _build_synthetic_checkpoint(d):
    rng = np.random.default_rng(0)
    hidden, inter, vocab = 32, 16, 40
    tensors = {}
    refs = {}

    tensors["model.language_model.embed_tokens.weight"] = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    tensors["model.language_model.norm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
    w, ws, g, ref = _make_nvfp4(vocab, hidden, rng)  # lm_head NVFP4
    tensors["lm_head.weight"], tensors["lm_head.weight_scale"], tensors["lm_head.weight_scale_2"] = w, ws, g
    refs["lm_head"] = ref

    def add_nvfp4(prefix, n, k):
        w, ws, g, ref = _make_nvfp4(n, k, rng)
        tensors[f"{prefix}.weight"] = w
        tensors[f"{prefix}.weight_scale"] = ws
        tensors[f"{prefix}.weight_scale_2"] = g
        refs[prefix] = ref

    def add_fp8(prefix, out_f, in_f):
        w, s, ref = _make_fp8(out_f, in_f, rng)
        tensors[f"{prefix}.weight"] = w
        tensors[f"{prefix}.weight_scale"] = s
        refs[prefix] = ref

    for layer in (0, 1):
        p = f"model.language_model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        if layer == 0:  # linear attention
            add_fp8(f"{p}.linear_attn.in_proj_qkv", 24, hidden)
            add_fp8(f"{p}.linear_attn.in_proj_z", 12, hidden)
            add_fp8(f"{p}.linear_attn.out_proj", hidden, 12)
            tensors[f"{p}.linear_attn.in_proj_a.weight"] = torch.zeros(8, hidden, dtype=torch.bfloat16)
            tensors[f"{p}.linear_attn.in_proj_b.weight"] = torch.zeros(8, hidden, dtype=torch.bfloat16)
            tensors[f"{p}.linear_attn.conv1d.weight"] = torch.zeros(24, 1, 4, dtype=torch.bfloat16)
            tensors[f"{p}.linear_attn.A_log"] = torch.zeros(8, dtype=torch.bfloat16)
            tensors[f"{p}.linear_attn.dt_bias"] = torch.zeros(8, dtype=torch.bfloat16)
            tensors[f"{p}.linear_attn.norm.weight"] = torch.ones(12, dtype=torch.bfloat16)
        else:  # full attention
            add_fp8(f"{p}.self_attn.q_proj", hidden, hidden)
            add_fp8(f"{p}.self_attn.k_proj", hidden, hidden)
            add_fp8(f"{p}.self_attn.v_proj", hidden, hidden)
            add_fp8(f"{p}.self_attn.o_proj", hidden, hidden)
            tensors[f"{p}.self_attn.q_norm.weight"] = torch.ones(8, dtype=torch.bfloat16)
            tensors[f"{p}.self_attn.k_norm.weight"] = torch.ones(8, dtype=torch.bfloat16)
        tensors[f"{p}.mlp.gate.weight"] = torch.zeros(4, hidden, dtype=torch.bfloat16)
        add_nvfp4(f"{p}.mlp.shared_expert.gate_proj", inter, hidden)
        add_nvfp4(f"{p}.mlp.shared_expert.up_proj", inter, hidden)
        add_nvfp4(f"{p}.mlp.shared_expert.down_proj", hidden, inter)
        tensors[f"{p}.mlp.shared_expert_gate.weight"] = torch.zeros(1, hidden, dtype=torch.bfloat16)
        # One routed expert retained in native NVFP4 form.
        add_nvfp4(f"{p}.mlp.experts.0.gate_proj", inter, hidden)
        add_nvfp4(f"{p}.mlp.experts.0.up_proj", inter, hidden)
        add_nvfp4(f"{p}.mlp.experts.0.down_proj", hidden, inter)

    save_file(tensors, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": dict.fromkeys(tensors, "model.safetensors")}, f)
    cfg = {"text_config": {"num_hidden_layers": 2, "hidden_size": hidden, "num_experts": 1}}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return refs


def test_modelopt_loader_tree_preserves_quantized_tensors():
    with tempfile.TemporaryDirectory() as d:
        _build_synthetic_checkpoint(d)
        model = QM.QuantModel.from_pretrained(
            "modelopt", input_path=d, quant_attrs={}, q_size=32, kv_size=32, intermediate_size=16, num_layers=2
        )

        assert isinstance(model, QM.QuantizedModel)
        assert isinstance(model.lm_head, QM.ModeloptLinearModule)
        mods = model.modules()
        assert mods[0] is model.embedding and mods[-1] is model.lm_head
        assert len(model.layers) == 2
        assert all(m.__class__.__name__.endswith("DecoderLayer") for m in model.layers)

        l0, l1 = model.layers
        # Layer 0 is linear-attention, layer 1 is full-attention.
        assert l0.linear_attn is not None and l0.self_attn is None
        assert l1.self_attn is not None and l1.linear_attn is None

        # FP8 projections retain their weights and scales for native contrib-op export.
        assert l0.linear_attn.in_proj_qkv.weight.dtype == torch.float8_e4m3fn
        assert l0.linear_attn.in_proj_qkv.weight_scale is not None
        assert l0.linear_attn.A_log is not None and l0.linear_attn.conv1d.weight is not None

        # NVFP4 modules retain packed E2M1 weights, E4M3 block scales, and global scales.
        assert l0.mlp.shared_expert.gate_proj.weight.dtype == torch.uint8
        assert l0.mlp.shared_expert.gate_proj.weight_scale.dtype == torch.float8_e4m3fn
        assert l0.mlp.shared_expert.gate_proj.weight_scale_2.numel() == 1
        assert model.lm_head.weight.dtype == torch.uint8

        # Routed experts are materialized once by the loader for native QMoE preprocessing.
        assert len(l0.mlp.experts) == 1
        assert l0.mlp.experts[0].gate_proj.weight.dtype == torch.uint8
        # Router / shared-expert-gate are present as plain tensors.
        assert l0.mlp.gate.weight is not None and l0.mlp.shared_expert_gate.weight is not None
        # All safetensors handles are released once loading finishes.
        assert not model._open_handles
    print("OK: ModeloptModel builds the tree and preserves FP8/NVFP4 tensors.")


def _load(d):
    return QM.QuantModel.from_pretrained(
        "modelopt", input_path=d, quant_attrs={}, q_size=32, kv_size=32, intermediate_size=16, num_layers=2
    )


def test_modelopt_loader_rejects_bad_checkpoints():
    # No index file and an ambiguous number of shards -> deterministic error, not StopIteration.
    with tempfile.TemporaryDirectory() as d:
        _build_synthetic_checkpoint(d)
        os.remove(os.path.join(d, "model.safetensors.index.json"))
        shutil.copyfile(os.path.join(d, "model.safetensors"), os.path.join(d, "model-2.safetensors"))
        with pytest.raises(ValueError, match="exactly one .safetensors file"):
            _load(d)

    # NVFP4 tensor with a global scale but no FP8 block scales -> explicit error.
    with tempfile.TemporaryDirectory() as d:
        _build_synthetic_checkpoint(d)
        tensors = load_file(os.path.join(d, "model.safetensors"))
        del tensors["lm_head.weight_scale"]
        save_file(tensors, os.path.join(d, "model.safetensors"))
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": dict.fromkeys(tensors, "model.safetensors")}, f)
        with pytest.raises(ValueError, match="no 'weight_scale'"):
            _load(d)

    # A layer with neither linear_attn nor self_attn -> named error, not a later AttributeError.
    with tempfile.TemporaryDirectory() as d:
        _build_synthetic_checkpoint(d)
        tensors = load_file(os.path.join(d, "model.safetensors"))
        del tensors["model.language_model.layers.1.self_attn.q_proj.weight"]
        save_file(tensors, os.path.join(d, "model.safetensors"))
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": dict.fromkeys(tensors, "model.safetensors")}, f)
        with pytest.raises(ValueError, match="attention variant cannot be determined"):
            _load(d)
    print("OK: ModeloptModel rejects ambiguous and incomplete checkpoints.")


def test_modelopt_loader_preserves_raw_uint8_block_scales():
    """Some exporters store the E4M3 block scales as raw uint8 bytes."""
    with tempfile.TemporaryDirectory() as d:
        _build_synthetic_checkpoint(d)
        tensors = load_file(os.path.join(d, "model.safetensors"))
        tensors["lm_head.weight_scale"] = tensors["lm_head.weight_scale"].view(torch.uint8)
        save_file(tensors, os.path.join(d, "model.safetensors"))
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": dict.fromkeys(tensors, "model.safetensors")}, f)
        assert _load(d).lm_head.weight_scale.dtype == torch.uint8
    print("OK: ModeloptModel preserves uint8-stored E4M3 block scales.")


if __name__ == "__main__":
    test_modelopt_loader_tree_preserves_quantized_tensors()
    test_modelopt_loader_rejects_bad_checkpoints()
    test_modelopt_loader_preserves_raw_uint8_block_scales()
