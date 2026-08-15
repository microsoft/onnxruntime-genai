# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import sys
from pathlib import Path

import onnx_ir as ir
import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py"))

from models.builders.base import Model


def _make_model():
    model = object.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    model._fp8_activation_scale_cache = {}

    initializers = []

    def record_initializer(tensor, name, **kwargs):
        initializers.append((name, tensor))
        return name

    model.make_initializer = record_initializer
    return model, initializers


def test_scale_initializer_is_shared_for_matching_qkv_scales():
    model, initializers = _make_model()

    q_name = model.make_fp8_activation_scale_initializer(0.125)
    k_name = model.make_fp8_activation_scale_initializer(0.125)
    v_name = model.make_fp8_activation_scale_initializer(0.125)

    assert k_name == q_name
    assert v_name == q_name
    assert len(initializers) == 1
    assert initializers[0][1].dtype == torch.float32
    assert initializers[0][1].numel() == 1


def test_scale_initializer_is_not_shared_when_scale_differs():
    model, initializers = _make_model()

    q_name = model.make_fp8_activation_scale_initializer(0.125)
    o_name = model.make_fp8_activation_scale_initializer(0.25)

    assert o_name != q_name
    assert len(initializers) == 2
