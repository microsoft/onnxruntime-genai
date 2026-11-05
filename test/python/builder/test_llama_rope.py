from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parent))


def _load_builder_module(module_name):
    if f"models.builders.{module_name}" in sys.modules:
        return sys.modules[f"models.builders.{module_name}"]
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
llama_module = _load_builder_module("llama")
LlamaModel = llama_module.LlamaModel
Model = base_module.Model


def _make_rope_init(rope_config=None, rope_field="rope_scaling"):
    config_kwargs = {"hidden_size": 2048, "num_attention_heads": 16, "head_dim": 128}
    if rope_config is not None:
        config_kwargs[rope_field] = rope_config
    config = types.SimpleNamespace(**config_kwargs)

    model = LlamaModel.__new__(LlamaModel)
    Model.make_config_init(model, config)
    model.rope_attrs = {"rescale_factors": 1}
    model.make_rope_init(config)
    return model


@pytest.mark.parametrize("rope_field", ["rope_scaling", "rope_parameters"])
def test_llama_linear_rope_scaling_is_applied(rope_field):
    # DeepSeek-Coder: rope_scaling {"type": "linear", "factor": 4.0} (HF adds "rope_type")
    model = _make_rope_init({"rope_type": "linear", "type": "linear", "factor": 4.0}, rope_field)
    assert model.rope_attrs["rescale_factors"] == 4.0
    assert "rescale_inv_freq" not in model.rope_attrs


def test_llama3_rope_scaling_is_unchanged():
    model = _make_rope_init(
        {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        }
    )
    assert model.rope_attrs["rescale_inv_freq"] == {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
    }
    assert model.rope_attrs["rescale_factors"] == 1


def test_llama_without_rope_scaling_is_a_noop():
    model = _make_rope_init()
    assert model.rope_attrs == {"rescale_factors": 1}
