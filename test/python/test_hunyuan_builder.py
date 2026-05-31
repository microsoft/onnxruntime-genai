from __future__ import annotations

import math
import importlib.util
import sys
import types
from pathlib import Path

import pytest

BUILDERS_DIR = Path(__file__).parents[2] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
hunyuan_module = _load_builder_module("hunyuan")
HunyuanDenseV1Model = hunyuan_module.HunyuanDenseV1Model
Model = base_module.Model


def _build_hunyuan_config(config, monkeypatch):
    def base_init(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.config = config

    monkeypatch.setattr(Model, "__init__", base_init)
    return HunyuanDenseV1Model(config, None, None, "cpu", None, {})


@pytest.mark.parametrize("rope_field", ["rope_scaling", "rope_parameters"])
def test_hunyuan_dynamic_alpha_rope_is_baked_into_theta(rope_field, monkeypatch):
    rope_config = {"type": "dynamic", "alpha": 1000.0, "rope_theta": 10000.0}
    config_kwargs = {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "head_dim": 128,
        "rope_theta": None if rope_field == "rope_parameters" else 10000.0,
        "rope_scaling": rope_config if rope_field == "rope_scaling" else None,
        "rope_parameters": rope_config if rope_field == "rope_parameters" else None,
    }
    config = types.SimpleNamespace(**config_kwargs)

    _build_hunyuan_config(config, monkeypatch)

    expected_theta = 10000.0 * (1000.0 ** (128 / 126))
    assert math.isclose(config.rope_theta, expected_theta)
    assert config.rope_scaling is None
    assert config.rope_parameters is None