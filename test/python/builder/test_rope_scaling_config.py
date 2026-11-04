# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""``Model.make_config_init`` must accept Transformers 4.x and 5.x RoPE configs (issue #2861)."""

from types import SimpleNamespace

import pytest
from _builder_test_utils import load_builder_module

base = load_builder_module("base")
llama = load_builder_module("llama")
phi = load_builder_module("phi")

HEAD_DIM = 64
NUM_FACTORS = int(HEAD_DIM * 0.75) // 2


def make_rope_model(model_class, config):
    model = model_class.__new__(model_class)
    base.Model.make_config_init(model, config)
    model.context_length = config.max_position_embeddings
    model.original_context_length = getattr(config, "original_max_position_embeddings", config.max_position_embeddings)
    model.rope_attrs = {"mscale_policy": ""}
    model.make_rope_init(config)
    return model


def test_phi3_without_rope_scaling():
    # Transformers 4.x Phi-3-mini-4k / phi-4 configs set rope_scaling=None and have no rope_parameters.
    config = SimpleNamespace(rope_scaling=None, rope_theta=10000.0, max_position_embeddings=4096)
    model = make_rope_model(phi.Phi3MiniModel, config)

    assert not hasattr(config, "rope_parameters")
    assert config.rope_theta == 10000.0
    assert model.rope_attrs == {"mscale_policy": ""}


def test_phi4_mini_longrope_with_legacy_type_key():
    # Transformers 4.x Phi-4-mini configs keep the legacy {"type": "longrope", ...} rope_scaling.
    short_factor, long_factor = [1.0] * NUM_FACTORS, [2.0] * NUM_FACTORS
    rope_scaling = {"type": "longrope", "short_factor": short_factor, "long_factor": long_factor}
    config = SimpleNamespace(
        rope_scaling=rope_scaling,
        rope_theta=10000.0,
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
    )
    model = make_rope_model(phi.Phi3MiniLongRoPEModel, config)

    assert config.rope_parameters["rope_type"] == "longrope"
    assert config.rope_scaling == {"type": "longrope", "short_factor": short_factor, "long_factor": long_factor}
    assert model.rope_attrs["mscale_policy"] == "longrope"
    assert model.rope_attrs["multi_cache"]["short_factor"].tolist() == short_factor
    assert model.rope_attrs["multi_cache"]["long_factor"].tolist() == long_factor
    assert model.rope_attrs["multi_cache"]["long_mscale"] > 1.0


def test_phi4_mini_longrope_with_rope_type_only():
    # Configs written for Transformers 5.x may carry only {"rope_type": "longrope", ...}, without the legacy "type" key.
    short_factor, long_factor = [1.0] * NUM_FACTORS, [2.0] * NUM_FACTORS
    rope_parameters = {
        "rope_type": "longrope",
        "rope_theta": 10000.0,
        "short_factor": short_factor,
        "long_factor": long_factor,
    }
    config = SimpleNamespace(
        rope_parameters=rope_parameters,
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
    )
    model = make_rope_model(phi.Phi3MiniLongRoPEModel, config)

    assert model.rope_attrs["mscale_policy"] == "longrope"
    assert model.rope_attrs["multi_cache"]["long_factor"].tolist() == long_factor
    assert model.rope_attrs["multi_cache"]["long_mscale"] > 1.0


def test_llama_without_rope_scaling():
    # Transformers 4.x Llama-2-style configs set rope_scaling=None and have no rope_parameters.
    config = SimpleNamespace(rope_scaling=None, rope_theta=10000.0, max_position_embeddings=4096)
    model = make_rope_model(llama.LlamaModel, config)

    assert "rescale_inv_freq" not in model.rope_attrs


def test_rope_scaling_does_not_overwrite_rope_parameters():
    # Transformers 5.x configs already carry rope_parameters (rope_scaling aliases it).
    rope_parameters = {"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 1.0}
    config = SimpleNamespace(
        rope_parameters=rope_parameters,
        rope_scaling={"type": "linear", "rope_theta": 10000.0},
        max_position_embeddings=4096,
    )
    base.Model.make_config_init(base.Model.__new__(base.Model), config)

    assert config.rope_parameters is rope_parameters
    assert rope_parameters == {
        "rope_type": "default",
        "rope_theta": 500000.0,
        "partial_rotary_factor": 1.0,
        "type": "linear",
    }
    assert config.rope_theta == 500000.0


@pytest.mark.parametrize("rope_scaling", [None, {"type": "longrope"}])
def test_phi3_config_from_installed_transformers(rope_scaling):
    transformers = pytest.importorskip("transformers")
    # _builder_test_utils installs an importable transformers stub when the package is absent.
    if not hasattr(transformers, "Phi3Config"):
        pytest.skip("transformers does not provide Phi3Config")
    kwargs = {"hidden_size": 256, "num_attention_heads": 4, "max_position_embeddings": 131072}
    if rope_scaling:
        rope_scaling = {**rope_scaling, "short_factor": [1.0] * 32, "long_factor": [2.0] * 32}
        kwargs["original_max_position_embeddings"] = 4096
    config = transformers.Phi3Config(rope_scaling=rope_scaling, **kwargs)
    model_class = phi.Phi3MiniLongRoPEModel if rope_scaling else phi.Phi3MiniModel
    model = make_rope_model(model_class, config)

    assert config.rope_theta == 10000.0
    if rope_scaling:
        assert config.rope_parameters["rope_type"] == "longrope"
        assert model.rope_attrs["multi_cache"]["short_factor"].tolist() == [1.0] * 32
    else:
        assert "multi_cache" not in model.rope_attrs
