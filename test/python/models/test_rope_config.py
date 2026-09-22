# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Config-only RoPE regressions; no model weights are loaded."""

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


@pytest.mark.parametrize("scaling", ["absent", None])
@pytest.mark.parametrize("parameters", ["absent", None, {"rope_type": "default", "rope_theta": 10000.0}])
def test_no_rope_scaling(scaling, parameters):
    config = SimpleNamespace(rope_theta=10000.0)
    if scaling != "absent":
        config.rope_scaling = scaling
    if parameters != "absent":
        config.rope_parameters = copy.deepcopy(parameters)
    original = copy.deepcopy(vars(config))
    model = object.__new__(Model)
    model.rope_attrs = {}

    model.make_config_init(config)
    model.make_rope_init(config)

    assert model.rope_attrs == {}
    assert config.rope_theta == 10000.0
    for key, value in original.items():
        assert getattr(config, key) == value
    assert getattr(config, "rope_scaling", "absent") == scaling
    assert getattr(config, "rope_parameters", "absent") == parameters


@pytest.mark.parametrize("kwargs", [{}, {"rope_scaling": None}])
def test_qwen_config_round_trip(kwargs, tmp_path):
    config = Qwen2Config(**kwargs)
    config.save_pretrained(tmp_path)
    config = Qwen2Config.from_pretrained(tmp_path, local_files_only=True)
    model = object.__new__(Model)
    model.rope_attrs = {}

    model.make_config_init(config)
    model.make_rope_init(config)

    assert model.rope_attrs == {}


@pytest.mark.parametrize("mscales", [{}, {"short_mscale": 1.25, "long_mscale": 1.5}])
def test_phi_canonical_longrope_config_round_trip(mscales, tmp_path):
    parameters = {
        "rope_type": "longrope",
        "short_factor": [1.0, 1.5],
        "long_factor": [2.0, 3.0],
        **mscales,
    }
    config = Phi3Config(
        hidden_size=8,
        num_attention_heads=2,
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        rope_parameters=copy.deepcopy(parameters),
    )
    config.save_pretrained(tmp_path)
    config = Phi3Config.from_pretrained(tmp_path, local_files_only=True)
    assert "type" not in config.rope_parameters
    original = copy.deepcopy(config.rope_parameters)
    model = object.__new__(Phi3MiniLongRoPEModel)
    model.context_length = config.max_position_embeddings
    model.original_context_length = config.original_max_position_embeddings
    model.rope_attrs = {}

    model.make_config_init(config)
    model.make_rope_init(config)

    assert model.rope_attrs["mscale_policy"] == "longrope"
    cache = model.rope_attrs["multi_cache"]
    for name in ("short_factor", "long_factor"):
        torch.testing.assert_close(cache[name], torch.tensor(parameters[name], dtype=torch.float32))
    expected_mscale = math.sqrt(1 + math.log(32) / math.log(4096))
    for name in ("short_mscale", "long_mscale"):
        assert cache[name] == pytest.approx(mscales.get(name, expected_mscale))
    assert config.rope_parameters == original
