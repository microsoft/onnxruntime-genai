# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Config-only regression tests; no model weights are loaded."""

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Phi3Config, Qwen2Config

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "python" / "py" / "models"))

from builders.base import Model
from builders.phi import Phi3MiniLongRoPEModel


@pytest.mark.parametrize("scaling", ["absent", None, {}])
@pytest.mark.parametrize("parameters", ["absent", None, {}])
def test_no_rope_scaling(scaling, parameters):
    config = SimpleNamespace(rope_theta=10000.0)
    if scaling != "absent":
        config.rope_scaling = scaling
    if parameters != "absent":
        config.rope_parameters = parameters
    model = object.__new__(Model)
    model.rope_attrs = {}

    model.make_config_init(config)
    model.make_rope_init(config)

    assert model.rope_attrs == {}
    assert config.rope_theta == 10000.0
    assert getattr(config, "rope_scaling", "absent") == scaling


@pytest.mark.parametrize("parameters", [None, {}])
@pytest.mark.parametrize("type_key", ["type", "rope_type"])
def test_legacy_scaling_with_missing_parameters(parameters, type_key):
    scaling = {type_key: "linear", "factor": 2.0, "rope_theta": 500000.0}
    original = copy.deepcopy(scaling)
    config = SimpleNamespace(rope_scaling=scaling, rope_parameters=parameters, rope_theta=10000.0)
    model = object.__new__(Model)
    model.rope_attrs = {}

    model.make_config_init(config)
    model.make_rope_init(config)

    assert config.rope_parameters["rope_type"] == "linear"
    assert config.rope_parameters["factor"] == 2.0
    assert config.rope_theta == 10000.0
    assert model.rope_attrs["rescale_factors"] == 2.0
    assert scaling == original


def test_canonical_type_wins_over_legacy_alias():
    config = SimpleNamespace(
        rope_parameters={"rope_type": "linear", "type": "yarn", "factor": 4.0},
        rope_scaling={"rope_type": "yarn", "factor": 2.0, "rope_theta": 10000.0},
    )
    model = object.__new__(Model)
    model.rope_attrs = {}

    model.make_config_init(config)
    model.make_rope_init(config)

    assert config.rope_parameters["rope_type"] == "linear"
    assert model.rope_attrs["rescale_factors"] == 4.0
    assert config.rope_theta == 10000.0


@pytest.mark.parametrize("kwargs", [{}, {"rope_scaling": None}, {"rope_parameters": None}])
def test_qwen_config_round_trip(kwargs, tmp_path):
    config = Qwen2Config(**kwargs)
    config.save_pretrained(tmp_path)
    config = Qwen2Config.from_pretrained(tmp_path, local_files_only=True)
    model = object.__new__(Model)
    model.rope_attrs = {}

    model.make_config_init(config)
    model.make_rope_init(config)

    assert model.rope_attrs == {}


@pytest.mark.parametrize("field", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize("type_key", ["type", "rope_type"])
def test_phi_longrope_parameters(field, type_key):
    params = {type_key: "longrope", "short_factor": [1.0, 1.5], "long_factor": [2.0, 3.0]}
    original = copy.deepcopy(params)
    config = SimpleNamespace(**{field: params})
    model = object.__new__(Phi3MiniLongRoPEModel)
    model.context_length = 131072
    model.original_context_length = 4096
    model.rope_attrs = {}

    model.make_config_init(config)
    model.make_rope_init(config)

    assert model.rope_attrs["mscale_policy"] == "longrope"
    cache = model.rope_attrs["multi_cache"]
    for name in ("short_factor", "long_factor"):
        torch.testing.assert_close(cache[name], torch.tensor(original[name], dtype=torch.float32))
    expected_mscale = math.sqrt(1 + math.log(32) / math.log(4096))
    assert cache["short_mscale"] == pytest.approx(expected_mscale)
    assert cache["long_mscale"] == pytest.approx(expected_mscale)
    if field == "rope_scaling":
        assert params == original


def test_phi_canonical_config_round_trip(tmp_path):
    config = Phi3Config(
        hidden_size=8,
        num_attention_heads=2,
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        rope_parameters={"rope_type": "longrope", "short_factor": [1.0, 1.5], "long_factor": [2.0, 3.0]},
    )
    config.save_pretrained(tmp_path)
    config = Phi3Config.from_pretrained(tmp_path, local_files_only=True)
    assert "type" not in config.rope_parameters
    model = object.__new__(Phi3MiniLongRoPEModel)
    model.context_length = config.max_position_embeddings
    model.original_context_length = config.original_max_position_embeddings
    model.rope_attrs = {}

    model.make_config_init(config)
    model.make_rope_init(config)

    assert model.rope_attrs["mscale_policy"] == "longrope"
    assert model.rope_attrs["multi_cache"]["long_mscale"] == pytest.approx(math.sqrt(1 + math.log(32) / math.log(4096)))
