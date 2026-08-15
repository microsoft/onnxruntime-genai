# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib.util
import sys
from pathlib import Path
from types import MethodType

import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py"))

module_path = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "quantized_model.py"
spec = importlib.util.spec_from_file_location("_modelopt_linear_attn_test", module_path)
quantized_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantized_model)
ModeloptModel = quantized_model.ModeloptModel


def _make_model(tensors):
    model = object.__new__(ModeloptModel)
    model._get = MethodType(lambda self, key: tensors.get(key), model)
    return model


LINEAR = "model.language_model.layers.3.linear_attn.in_proj_qkv"
ATTENTION = "model.language_model.layers.3.self_attn.q_proj"


def test_linear_attention_fp8_is_weight_only():
    tensors = {
        f"{LINEAR}.weight": torch.ones((4, 4), dtype=torch.float8_e4m3fn),
        f"{LINEAR}.weight_scale": torch.tensor(0.125),
        f"{LINEAR}.input_scale": torch.tensor(0.25),
    }
    module = _make_model(tensors)._linear_module(LINEAR)

    assert module.weight.dtype == torch.float8_e4m3fn
    assert module.weight_scale.item() == 0.125
    assert module.input_scale is None


def test_self_attention_fp8_keeps_calibrated_input_scale():
    tensors = {
        f"{ATTENTION}.weight": torch.ones((4, 4), dtype=torch.float8_e4m3fn),
        f"{ATTENTION}.weight_scale": torch.tensor(0.125),
        f"{ATTENTION}.input_scale": torch.tensor(0.25),
    }
    module = _make_model(tensors)._linear_module(ATTENTION)

    assert module.input_scale.item() == 0.25


def test_bf16_linear_attention_projection_stays_unquantized():
    base = "model.language_model.layers.3.linear_attn.in_proj_a"
    weight = torch.ones((4, 4), dtype=torch.bfloat16)
    module = _make_model({f"{base}.weight": weight})._linear_module(base)

    assert module.weight is weight
    assert module.weight_scale is None
