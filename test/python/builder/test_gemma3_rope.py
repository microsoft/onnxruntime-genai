# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Gemma 3 RoPE caches under Transformers 5 (issues #2788 and #2789).

Transformers 5 stores Gemma 3's RoPE per layer type in ``rope_parameters``
(``{"full_attention": {...}, "sliding_attention": {...}}``) with no flat ``rope_theta`` /
``rope_local_base_freq``. The global (full-attention) layers take the configured scaling (linear,
factor 8 on Gemma 3 4B/12B/27B); the local (sliding) layers use default, unscaled RoPE with their
own theta.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from builder_test_utils import load_builder_module

gemma = load_builder_module("gemma")

HEAD_DIM = 16
CONTEXT_LENGTH = 64
GLOBAL_THETA, LOCAL_THETA, FACTOR = 1_000_000.0, 10_000.0, 8.0
LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]
SMALL_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": len(LAYER_TYPES),
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": HEAD_DIM,
    "query_pre_attn_scalar": HEAD_DIM,
    "sliding_window": 8,
    "max_position_embeddings": CONTEXT_LENGTH,
}
SCALED_ROPE_PARAMETERS = {
    "sliding_attention": {"rope_type": "default", "rope_theta": LOCAL_THETA},
    "full_attention": {"rope_type": "linear", "factor": FACTOR, "rope_theta": GLOBAL_THETA},
}
UNSCALED_ROPE_PARAMETERS = {
    "sliding_attention": {"rope_type": "default", "rope_theta": LOCAL_THETA},
    "full_attention": {"rope_type": "default", "rope_theta": GLOBAL_THETA},
}


def reference_cos_sin(theta, factor=1.0):
    inv_freq = 1.0 / (factor * theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.int64).float() / HEAD_DIM))
    freqs = torch.outer(torch.arange(CONTEXT_LENGTH).float(), inv_freq)
    return freqs.cos().numpy(), freqs.sin().numpy()


def make_namespace_config(rope_parameters):
    return SimpleNamespace(
        architectures=["Gemma3ForCausalLM"],
        _name_or_path="",
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        layer_types=LAYER_TYPES,
        rope_parameters=rope_parameters,
        **SMALL_CONFIG,
    )


def build(config, tmp_path):
    return gemma.Gemma3Model(config, 1, 1, "cpu", str(tmp_path), {})


def caches(model, kind):
    return (
        model.values[f"cos_cache_{kind}"].const_value.numpy(),
        model.values[f"sin_cache_{kind}"].const_value.numpy(),
    )


def assert_caches_match(model, kind, expected):
    for actual, wanted in zip(caches(model, kind), expected, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=1e-5, atol=1e-5)


def test_local_caches_drop_global_linear_scaling(tmp_path):
    # Gemma 3 4B/12B/27B: linear factor 8 on the full-attention layers only.
    model = build(make_namespace_config(SCALED_ROPE_PARAMETERS), tmp_path)

    assert_caches_match(model, "global", reference_cos_sin(GLOBAL_THETA, FACTOR))
    assert_caches_match(model, "local", reference_cos_sin(LOCAL_THETA))
    # The global RoPE state is left in place for anything built after the caches.
    assert model.rope_attrs["theta"] == GLOBAL_THETA
    assert model.rope_attrs["rescale_factors"] == FACTOR


def test_unscaled_config_keeps_default_caches(tmp_path):
    # Gemma 3 1B: no RoPE scaling on either layer type.
    model = build(make_namespace_config(UNSCALED_ROPE_PARAMETERS), tmp_path)

    assert_caches_match(model, "global", reference_cos_sin(GLOBAL_THETA))
    assert_caches_match(model, "local", reference_cos_sin(LOCAL_THETA))


def test_scaled_sliding_layers_are_rejected(tmp_path):
    rope_parameters = {
        **SCALED_ROPE_PARAMETERS,
        "sliding_attention": {"rope_type": "linear", "factor": 2.0, "rope_theta": LOCAL_THETA},
    }
    with pytest.raises(NotImplementedError, match="sliding-attention"):
        build(make_namespace_config(rope_parameters), tmp_path)


def transformers_reference(config):
    """cos/sin halves from Transformers' own Gemma 3 rotary embedding, per layer type."""
    modeling_gemma3 = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
    text_model = modeling_gemma3.Gemma3TextModel(config).eval()
    x = torch.zeros(1, CONTEXT_LENGTH, HEAD_DIM)
    position_ids = torch.arange(CONTEXT_LENGTH)[None]
    with torch.no_grad():
        outputs = {
            "global": text_model.rotary_emb(x, position_ids, "full_attention"),
            "local": text_model.rotary_emb(x, position_ids, "sliding_attention"),
        }
    return {
        kind: (cos[0, :, : HEAD_DIM // 2].numpy(), sin[0, :, : HEAD_DIM // 2].numpy())
        for kind, (cos, sin) in outputs.items()
    }


@pytest.mark.parametrize(
    "rope_config",
    [
        # Native Transformers 5 form.
        {"rope_parameters": SCALED_ROPE_PARAMETERS},
        {"rope_parameters": UNSCALED_ROPE_PARAMETERS},
        # The keys Google's Gemma 3 4B/12B/27B config.json ships with; Transformers 5 converts them
        # to the per-layer-type rope_parameters above when it loads the config.
        {
            "rope_theta": GLOBAL_THETA,
            "rope_local_base_freq": LOCAL_THETA,
            "rope_scaling": {"rope_type": "linear", "factor": FACTOR},
        },
    ],
    ids=["rope_parameters_linear", "rope_parameters_default", "hub_config_json"],
)
def test_caches_match_transformers_rotary_embedding(rope_config, tmp_path):
    transformers = pytest.importorskip("transformers")
    # builder_test_utils installs an importable transformers stub when the package is absent.
    if not hasattr(transformers, "Gemma3TextConfig"):
        pytest.skip("transformers does not provide Gemma3TextConfig")
    config = transformers.Gemma3TextConfig(**SMALL_CONFIG, **rope_config)
    reference = transformers_reference(config)
    config.architectures = ["Gemma3ForCausalLM"]
    model = build(config, tmp_path)

    assert_caches_match(model, "global", reference["global"])
    assert_caches_match(model, "local", reference["local"])
