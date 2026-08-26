from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parent))


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


@pytest.mark.parametrize("rope_field", ["rope_scaling", "rope_parameters"])
def test_hunyuan_dynamic_alpha_rope_is_baked_into_theta(rope_field):
    rope_config = {"rope_type": "dynamic", "alpha": 1000.0, "rope_theta": 10000.0}
    config_kwargs = {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "head_dim": 128,
    }
    config_kwargs[rope_field] = rope_config
    config = types.SimpleNamespace(**config_kwargs)

    model = HunyuanDenseV1Model.__new__(HunyuanDenseV1Model)
    Model.make_config_init(model, config)
    model.head_size = config.head_dim
    model.rope_attrs = {"theta": config.rope_theta}
    model.make_rope_init(config)

    expected_theta = 10000.0 * (1000.0 ** (128 / 126))
    assert math.isclose(model.rope_attrs["theta"], expected_theta)
