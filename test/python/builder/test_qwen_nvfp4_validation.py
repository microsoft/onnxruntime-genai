# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py"))

from models.builders.base import Model
from models.builders.qwen import Qwen35MoeTextModel


def test_modelopt_e4m3_bytes_accepts_float8_and_preserves_shape():
    scales = torch.ones((4, 2), dtype=torch.float8_e4m3fn)

    raw = Qwen35MoeTextModel._modelopt_e4m3_bytes(scales, "scales", (4, 2))

    assert raw.dtype == torch.uint8
    assert raw.shape == scales.shape


def test_modelopt_e4m3_bytes_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="must contain E4M3 bytes"):
        Qwen35MoeTextModel._modelopt_e4m3_bytes(torch.ones((4, 2)), "scales", (4, 2))


def test_modelopt_e4m3_bytes_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"expected \(4, 2\)"):
        Qwen35MoeTextModel._modelopt_e4m3_bytes(torch.ones((2, 4), dtype=torch.uint8), "scales", (4, 2))


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_modelopt_positive_scalar_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite and positive"):
        Qwen35MoeTextModel._modelopt_positive_scalar(torch.tensor(value), "global_scale")


def test_close_nvfp4_handles_closes_every_cached_file():
    closed = []
    model = object.__new__(Qwen35MoeTextModel)
    model._nvfp4_handles = {
        "a": SimpleNamespace(__exit__=lambda *args: closed.append("a")),
        "b": SimpleNamespace(__exit__=lambda *args: closed.append("b")),
    }
    model._nvfp4_handle_keys = {"a": {"x"}, "b": {"y"}}

    model._close_nvfp4_handles()

    assert closed == ["a", "b"]
    assert model._nvfp4_handles == {}
    assert model._nvfp4_handle_keys == {}


def test_make_model_closes_nvfp4_handles_when_graph_build_fails(monkeypatch):
    model = object.__new__(Qwen35MoeTextModel)
    closed = []
    model._close_nvfp4_handles = lambda: closed.append(True)
    monkeypatch.setattr(Model, "make_model", lambda self, input_path: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        model.make_model("checkpoint")

    assert closed == [True]


def test_nvfp4_qmoe_rejects_mismatched_gate_up_global_scales():
    model = object.__new__(Qwen35MoeTextModel)
    model.moe_attrs = {"num_experts": 1}
    prefix = "model.language_model.layers.0.mlp.experts.0"
    tensors = {
        f"{prefix}.gate_proj.weight": torch.zeros((16, 8), dtype=torch.uint8),
        f"{prefix}.up_proj.weight": torch.zeros((16, 8), dtype=torch.uint8),
        f"{prefix}.down_proj.weight": torch.zeros((16, 8), dtype=torch.uint8),
        f"{prefix}.gate_proj.weight_scale": torch.ones((16, 1), dtype=torch.float8_e4m3fn),
        f"{prefix}.up_proj.weight_scale": torch.ones((16, 1), dtype=torch.float8_e4m3fn),
        f"{prefix}.down_proj.weight_scale": torch.ones((16, 1), dtype=torch.float8_e4m3fn),
        f"{prefix}.gate_proj.weight_scale_2": torch.tensor(0.5),
        f"{prefix}.up_proj.weight_scale_2": torch.tensor(0.25),
        f"{prefix}.down_proj.weight_scale_2": torch.tensor(0.5),
    }
    model._load_nvfp4_tensor = lambda name: tensors[name]

    with pytest.raises(ValueError, match="gate/up global scales must match"):
        model.make_nvfp4_moe_initializers(0, "gw", "gs", "gg", "dw", "ds", "dg")
