# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Utilities for generating and fixing ``genai_config.json`` files.

This module is intentionally kept dependency-free (no torch / onnx imports)
so that it can be used in lightweight tests and tooling scripts.
"""

from __future__ import annotations

# Search-section defaults for onnxruntime-genai.  In transformers >= 5,
# GenerationConfig attributes default to None instead of concrete values,
# which can produce null entries in genai_config.json that
# onnxruntime-genai does not accept.
_GENAI_SEARCH_DEFAULTS = {
    "diversity_penalty": 0.0,
    "do_sample": False,
    "early_stopping": True,
    "length_penalty": 1.0,
    "min_length": 0,
    "no_repeat_ngram_size": 0,
    "num_beams": 1,
    "num_return_sequences": 1,
    "repetition_penalty": 1.0,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
}


def fix_genai_config(genai_config):
    """Fix a genai_config dict for compatibility with onnxruntime-genai.

    In transformers >= 5, :class:`transformers.GenerationConfig` attributes
    default to ``None`` instead of concrete values.  When
    :meth:`Model.make_genai_config` reads those attributes to populate the
    ``search`` section of ``genai_config.json``, the result can contain
    ``null`` entries that onnxruntime-genai rejects.

    This function replaces every ``null`` (``None``) value in the ``search``
    section with the corresponding onnxruntime-genai default.  Values that are
    already set are left unchanged, so the function is safe to call regardless
    of the transformers version in use.

    :param genai_config: genai_config dict (modified in-place and returned).
    :return: the modified *genai_config* dict.
    """
    search = genai_config.get("search", {})
    for key, default_val in _GENAI_SEARCH_DEFAULTS.items():
        if search.get(key) is None:
            search[key] = default_val
    return genai_config
