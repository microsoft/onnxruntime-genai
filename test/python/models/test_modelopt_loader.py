# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Structural + numerical test for the Model Optimizer (NVFP4/FP8) loader.

Builds a tiny synthetic modelopt-style checkpoint (one linear-attention layer,
one full-attention layer, plus globals) and verifies that ModeloptModel:
  * builds the module tree the ONNX Runtime GenAI builder walks,
  * exposes dequantized weights as plain TensorModule.weight (no `qweight`, so
    make_matmul takes the float path),
  * dequantizes FP8 attention and NVFP4 shared-expert/lm_head correctly, and
  * skips (streams) the routed experts.
"""

import importlib.util
import json
import os
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file


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
        # A routed expert that the loader must ignore (streamed by the builder).
        add_nvfp4(f"{p}.mlp.experts.0.gate_proj", inter, hidden)

    save_file(tensors, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {name: "model.safetensors" for name in tensors}}, f)
    cfg = {"text_config": {"num_hidden_layers": 2, "hidden_size": hidden}}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return refs


def test_modelopt_loader_tree_and_dequant():
    with tempfile.TemporaryDirectory() as d:
        refs = _build_synthetic_checkpoint(d)
        model = QM.QuantModel.from_pretrained(
            "modelopt", input_path=d, quant_attrs={}, q_size=32, kv_size=32, intermediate_size=16, num_layers=2
        )

        mods = model.modules()
        assert mods[0] is model.embedding and mods[-1] is model.lm_head
        assert len(model.layers) == 2
        assert all(m.__class__.__name__.endswith("DecoderLayer") for m in model.layers)

        l0, l1 = model.layers
        # Layer 0 is linear-attention, layer 1 is full-attention.
        assert l0.linear_attn is not None and l0.self_attn is None
        assert l1.self_attn is not None and l1.linear_attn is None

        # Dequantized linear layers are plain TensorModules (no qweight -> float path).
        assert not hasattr(l0.linear_attn.in_proj_qkv, "qweight")
        assert l0.linear_attn.in_proj_qkv.weight.dtype == torch.bfloat16
        assert l0.linear_attn.A_log is not None and l0.linear_attn.conv1d.weight is not None

        # FP8 attention dequant is exact.
        torch.testing.assert_close(
            l0.linear_attn.in_proj_qkv.weight, refs["model.language_model.layers.0.linear_attn.in_proj_qkv"]
        )
        torch.testing.assert_close(
            l1.self_attn.q_proj.weight, refs["model.language_model.layers.1.self_attn.q_proj"]
        )
        # NVFP4 shared-expert + lm_head dequant is exact.
        torch.testing.assert_close(
            l0.mlp.shared_expert.gate_proj.weight, refs["model.language_model.layers.0.mlp.shared_expert.gate_proj"]
        )
        torch.testing.assert_close(model.lm_head.weight, refs["lm_head"])

        # Routed experts are NOT materialized by the loader (streamed later).
        assert l0.mlp.experts is None
        # Router / shared-expert-gate are present as plain tensors.
        assert l0.mlp.gate.weight is not None and l0.mlp.shared_expert_gate.weight is not None
    print("OK: ModeloptModel builds the tree and dequantizes FP8/NVFP4 correctly.")


if __name__ == "__main__":
    test_modelopt_loader_tree_and_dequant()
