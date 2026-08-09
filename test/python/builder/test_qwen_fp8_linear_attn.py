# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import sys
from pathlib import Path
from types import MethodType

import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py"))

from models.builders.qwen import Qwen35MoeTextModel


def _make_model(**attrs):
    """Build a minimal ``Qwen35MoeTextModel`` stub exposing only the FP8 key mapping."""
    model = object.__new__(Qwen35MoeTextModel)
    for k, v in attrs.items():
        setattr(model, k, v)
    return model


QKV = "/model/layers.3/linear_attn/in_proj_qkv/MatMul"
Z = "/model/layers.3/linear_attn/in_proj_z/MatMul"
OUT = "/model/layers.3/linear_attn/out_proj/MatMul"


def test_linear_attn_projections_map_to_checkpoint_keys():
    model = _make_model()

    assert model._fp8_weight_key_for_matmul(QKV) == "model.language_model.layers.3.linear_attn.in_proj_qkv"
    assert model._fp8_weight_key_for_matmul(Z) == "model.language_model.layers.3.linear_attn.in_proj_z"
    assert model._fp8_weight_key_for_matmul(OUT) == "model.language_model.layers.3.linear_attn.out_proj"


def test_bf16_linear_attn_projections_are_not_fp8():
    model = _make_model()

    for proj in ("in_proj_a", "in_proj_b", "conv1d"):
        assert model._fp8_weight_key_for_matmul(f"/model/layers.3/linear_attn/{proj}/MatMul") is None


def test_linear_attn_is_weight_only():
    model = _make_model()
    model._load_nvfp4_tensor = MethodType(lambda self, key: torch.tensor([0.125]), model)

    assert model._fp8_attention_input_scale(QKV) is None
    # The self-attention projections still get the calibrated static scale.
    assert model._fp8_attention_input_scale("/model/layers.3/attn/q_proj/MatMul") == 0.125
