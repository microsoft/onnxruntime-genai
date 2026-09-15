# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Validation tests for the ``paged_block_size`` builder option.

The builder used to demand a multiple of 256. The ONNX Runtime PagedAttention op only
requires a power of two of at least 16 (``CheckInputs`` in ``paged_attention_helper.h``);
the multiple-of-256 rule belongs to one FlashAttention tile size, and a block below that
tile makes ORT pick a different backend rather than reject the model.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
BUILDERS_DIR = MODELS_DIR / "builders"
sys.path.insert(0, str(MODELS_DIR))


def _load_builder_entrypoint_module():
    # builder.py imports every concrete model class; stub the package so these
    # validation-only tests stay free of those dependencies.
    builders_stub = types.ModuleType("builders")

    def _stub_getattr(name):  # PEP 562: satisfies `from builders import <ModelClass>`
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

    builders_stub.__getattr__ = _stub_getattr
    builders_stub.__path__ = [str(BUILDERS_DIR)]
    sys.modules["builders"] = builders_stub

    spec = importlib.util.spec_from_file_location("models_builder_entrypoint", MODELS_DIR / "builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder_module = _load_builder_entrypoint_module()


def _check_extra_options(extra_options):
    builder_module.get_hf_details = lambda *args, **kwargs: {
        "hf_config": types.SimpleNamespace(tie_word_embeddings=True)
    }
    builder_module.check_extra_options("dummy-model", "", "", "fp16", "cuda", "", extra_options)


@pytest.mark.parametrize("block_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
def test_powers_of_two_are_accepted(block_size):
    options = {"use_paged_attention": "true", "paged_block_size": block_size}
    _check_extra_options(options)

    assert options["paged_block_size"] == block_size


def test_block_size_is_parsed_from_a_string():
    # Olive and the command line both hand extra options through as strings.
    options = {"use_paged_attention": "true", "paged_block_size": "64"}
    _check_extra_options(options)

    assert options["paged_block_size"] == 64


@pytest.mark.parametrize("block_size", [8, 15])
def test_block_sizes_below_the_minimum_are_rejected(block_size):
    with pytest.raises(ValueError, match="paged_block_size must be a power of two and at least 16"):
        _check_extra_options({"use_paged_attention": "true", "paged_block_size": block_size})


@pytest.mark.parametrize("block_size", [48, 96, 192, 768])
def test_non_powers_of_two_are_rejected(block_size):
    # 768 is a multiple of 256, so the old rule let it through and the op rejected it at load.
    with pytest.raises(ValueError, match="paged_block_size must be a power of two and at least 16"):
        _check_extra_options({"use_paged_attention": "true", "paged_block_size": block_size})


@pytest.mark.parametrize("block_size", [0, -256])
def test_non_positive_block_sizes_are_rejected(block_size):
    with pytest.raises(ValueError, match="paged_block_size must be a positive integer"):
        _check_extra_options({"use_paged_attention": "true", "paged_block_size": block_size})
