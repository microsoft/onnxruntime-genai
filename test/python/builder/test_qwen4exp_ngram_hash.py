# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Deterministic n-gram hashing layout for Qwen-3.8 Flash Next (``Qwen4Exp``) PLE layers.

The exported ``NGramHashMapping`` initializers (multipliers, per-head vocabulary sizes and
offsets) must match the reference ``modeling_qwen4_exp.py`` bit for bit, otherwise every PLE
layer indexes a different row of the n-gram embedding table.  These tests re-derive the values
from an independent implementation rather than calling the builder's own helpers back on
themselves.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))

sys.modules.setdefault("models", types.ModuleType("models"))
_builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
_builders_package.__path__ = [str(BUILDERS_DIR)]


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


_load_builder_module("base")
_load_builder_module("quant_config")
_load_builder_module("qwen")
qwen4exp = _load_builder_module("qwen4exp")


#####################################################################################
# Independent reference implementations
#####################################################################################

_M64 = 0xFFFFFFFFFFFFFFFF


def _ref_splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & _M64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _M64
    return (value ^ (value >> 31)) & _M64


def _ref_multipliers(vocab_size, ngram_size, ple_layer_index, seed):
    bound = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
    base = seed + 10007 * ple_layer_index
    return [
        2 * (_ref_splitmix64((base + 0x9E3779B97F4A7C15 * (i + 1)) & _M64) % bound) + 1 for i in range(ngram_size)
    ]


def _ref_primes_after(start, count):
    def prime(n):
        if n < 2:
            return False
        d = 2
        while d * d <= n:
            if n % d == 0:
                return False
            d += 1
        return True

    found = []
    candidate = start
    while len(found) < count:
        candidate += 1
        if prime(candidate):
            found.append(candidate)
    return found


#####################################################################################
# splitmix64 / multipliers
#####################################################################################


def test_splitmix64_matches_reference_constants():
    # Known-answer values pinned against the reference implementation's constants.
    for value in (0, 1, 2, 12345, 2**63, _M64):
        assert qwen4exp.splitmix64(value) == _ref_splitmix64(value)
        assert 0 <= qwen4exp.splitmix64(value) <= _M64


@pytest.mark.parametrize("ple_layer_index", [0, 1, 5])
def test_layer_multipliers_match_reference(ple_layer_index):
    vocab_size, ngram_size, seed = 248320, 4, 1234
    got = qwen4exp.build_layer_multipliers(vocab_size, ngram_size, ple_layer_index, seed)
    assert got == _ref_multipliers(vocab_size, ngram_size, ple_layer_index, seed)
    assert len(got) == ngram_size


def test_layer_multipliers_are_odd_and_cannot_overflow_int64():
    vocab_size, ngram_size = 248320, 4
    multipliers = qwen4exp.build_layer_multipliers(vocab_size, ngram_size, 3, 1234)
    for multiplier in multipliers:
        assert multiplier % 2 == 1, "multipliers must be odd so the hash is a bijection mod 2**k"
        assert (vocab_size - 1) * multiplier <= (1 << 63) - 1, "token_id * multiplier must stay in int64"


def test_layer_multipliers_differ_per_ple_layer():
    a = qwen4exp.build_layer_multipliers(248320, 4, 0, 1234)
    b = qwen4exp.build_layer_multipliers(248320, 4, 1, 1234)
    assert a != b


def test_layer_multipliers_are_seed_dependent():
    assert qwen4exp.build_layer_multipliers(248320, 4, 0, 1234) != qwen4exp.build_layer_multipliers(
        248320, 4, 0, 5678
    )


#####################################################################################
# Per-head prime vocabulary layout
#####################################################################################


def test_is_prime_matches_small_primes():
    assert [n for n in range(30) if qwen4exp.is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_find_nth_prime_after_walks_forward():
    assert qwen4exp.find_nth_prime_after(10, 1) == 11
    assert qwen4exp.find_nth_prime_after(10, 2) == 13
    assert qwen4exp.find_nth_prime_after(10, 3) == 17


def test_head_vocab_layout_uses_consecutive_primes():
    base, heads = 1000, 6
    sizes, offsets, total = qwen4exp.ngram_head_vocab_layout(base, heads, ple_layer_index=0)

    assert sizes == _ref_primes_after(base - 1, heads)
    assert offsets == [sum(sizes[:i]) for i in range(heads)]
    assert total == sum(sizes)


def test_head_vocab_layout_is_disjoint_across_ple_layers():
    base, heads = 1000, 6
    layer0, _, _ = qwen4exp.ngram_head_vocab_layout(base, heads, ple_layer_index=0)
    layer1, _, _ = qwen4exp.ngram_head_vocab_layout(base, heads, ple_layer_index=1)

    # Layer l consumes primes [l * heads, (l + 1) * heads), so no prime is reused.
    assert set(layer0).isdisjoint(layer1)
    assert layer1 == _ref_primes_after(base - 1, 2 * heads)[heads:]


def test_head_offsets_partition_the_table_without_gaps():
    sizes, offsets, total = qwen4exp.ngram_head_vocab_layout(1000, 6, ple_layer_index=2)
    for head_idx, (size, offset) in enumerate(zip(sizes, offsets)):
        end = offset + size
        assert end <= total
        if head_idx + 1 < len(offsets):
            assert offsets[head_idx + 1] == end


@pytest.mark.parametrize(
    "total, divisor, expected",
    [(100, 8, 104), (104, 8, 104), (1, 128, 128), (0, 128, 0), (256, 1, 256)],
)
def test_padded_vocab_rounds_up_to_divisor(total, divisor, expected):
    assert qwen4exp.padded_ngram_vocab_size(total, divisor) == expected


#####################################################################################
# Parity against the reference implementation shipped in `transformers`
#####################################################################################


@pytest.fixture(scope="module")
def reference_module():
    """The upstream `modeling_qwen4_exp` module, when the installed transformers ships it."""
    return pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")


@pytest.mark.parametrize("value", [0, 1, 7, 12345, 2**31, 2**63, (1 << 64) - 1])
def test_splitmix64_matches_reference(reference_module, value):
    assert qwen4exp.splitmix64(value) == reference_module._splitmix64(value)


@pytest.mark.parametrize("vocab_size", [1, 1000, 248320])
@pytest.mark.parametrize("ngram_size", [2, 3, 5])
@pytest.mark.parametrize("ple_layer_index", [0, 1, 7])
def test_layer_multipliers_match_reference(reference_module, vocab_size, ngram_size, ple_layer_index):
    expected = reference_module._build_layer_multipliers(vocab_size, ngram_size, ple_layer_index, 1234)
    assert qwen4exp.build_layer_multipliers(vocab_size, ngram_size, ple_layer_index, 1234) == expected.tolist()


def test_primality_helpers_match_reference(reference_module):
    assert all(qwen4exp.is_prime(value) == reference_module._is_prime(value) for value in range(5000))
    for start in (0, 10, 999, 19999999):
        for count in (1, 2, 5, 17):
            assert qwen4exp.find_nth_prime_after(start, count) == reference_module._find_nth_prime_after(start, count)


@pytest.mark.parametrize("ple_layer_index", [0, 1])
def test_head_layout_matches_a_real_ngram_embedding_module(reference_module, ple_layer_index):
    """The builder's initializers must line up with the checkpoint's embedding table."""
    config_module = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    config = config_module.Qwen4ExpTextConfig(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention"] * 3 + ["qwen_sparse_attention"],
        hc_count=2,
        hc_lowrank=16,
        ple_layer_ids=[1, 2],
        ple_embed_dim=32,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=8,
        seed=1234,
        eos_token_id=2,
    )
    ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
    embedding = reference_module.Qwen4ExpTextNGramEmbedding(
        config, config.ple_embed_dim, layer_idx=config.ple_layer_ids[ple_layer_index] - 1, ple_layer_index=ple_layer_index
    )

    sizes, offsets, total = qwen4exp.ngram_head_vocab_layout(
        config.ngram_vocab_size_base, ngram_heads, ple_layer_index
    )
    assert sizes == embedding.ngram_heads_vocab_sizes.tolist()
    assert offsets == embedding.ngram_heads_offsets.tolist()
    assert total == embedding.total_vocab_size
    assert qwen4exp.build_layer_multipliers(
        config.vocab_size, config.ngram_size, ple_layer_index, config.seed
    ) == embedding.layer_multipliers.tolist()
    padded = qwen4exp.padded_ngram_vocab_size(total, config.make_ngram_vocab_size_divisible_by)
    assert (padded, config.ple_embed_dim // ngram_heads) == tuple(embedding.ngram_embedding.weight.shape)
