# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Catalog of models used by the integration pipelines.

Most logical ids are top-level folders under ``foundrylocalmodels/models/``.
The paged Engine model is fetched from ``staging/paged-attention`` and
normalized to the same local layout:
``<root>/<logical_id>/onnx/<device_dir>/v<N>/``.

When the Foundry team uploads a new model, append it to ``MODELS`` (with
the device tags it ships with). Then add it to a suite below if you want
it gated on PRs or main merges.
"""

from __future__ import annotations

DEVICE_DIRNAMES: dict[str, str] = {
    "cpu": "cpu_and_mobile",
    "cuda": "cuda",
    "webgpu": "webgpu",
}


# Every model in the blob container that is text-to-text and has at least
# one device folder we currently test. Update by listing the parent
# directories of every genai_config.json in the container:
#     az storage blob list --account-name foundrylocalmodels \
#         --container-name models --auth-mode login \
#         --query "[?ends_with(name, 'genai_config.json')].name" -o tsv \
#         | sed 's:/genai_config.json$::'
MODELS: dict[str, set[str]] = {
    "Phi-3-mini-128k-instruct": {"cpu", "cuda", "webgpu"},
    "Phi-3-mini-4k-instruct": {"cpu", "cuda", "webgpu"},
    "Phi-3.5-mini-instruct": {"cpu", "cuda", "webgpu"},
    "Phi-4": {"cpu", "cuda", "webgpu"},
    "Phi-4-mini-instruct": {"cpu", "cuda", "webgpu"},
    "Phi-4-mini-reasoning": {"cpu", "cuda", "webgpu"},
    "Phi-4-reasoning": {"cpu", "cuda", "webgpu"},
    "deepseek-r1-distill-llama-8b": {"cpu", "cuda", "webgpu"},
    "deepseek-r1-distill-qwen-1.5b": {"cpu", "cuda", "webgpu"},
    "deepseek-r1-distill-qwen-14b": {"cpu", "cuda", "webgpu"},
    "deepseek-r1-distill-qwen-7b": {"cpu", "cuda", "webgpu"},
    "gpt-oss-20b": {"cpu", "cuda", "webgpu"},
    "ministral-3-3b-Instruct-2512": {"cpu", "cuda", "webgpu"},
    "mistral-nemo-12b-instruct": {"cpu", "cuda", "webgpu"},
    "mistralai-Mistral-7B-Instruct-v0-2": {"cpu", "cuda", "webgpu"},
    "olmo-3-7b-instruct": {"cpu", "cuda", "webgpu"},
    "qwen2.5-0.5b-instruct": {"cpu", "cuda", "webgpu"},
    # PagedAttention is CUDA-only.
    "qwen2.5-0.5b-instruct-paged": {"cuda"},
    "qwen2.5-1.5b-instruct": {"cpu", "cuda", "webgpu"},
    "qwen2.5-14b-instruct": {"cpu", "cuda", "webgpu"},
    "qwen2.5-3b-instruct": {"cpu", "cuda", "webgpu"},
    "qwen2.5-7b-instruct": {"cpu", "cuda", "webgpu"},
    "qwen2.5-coder-0.5b-instruct": {"cpu", "cuda", "webgpu"},
    "qwen2.5-coder-1.5b-instruct": {"cpu", "cuda", "webgpu"},
    "qwen2.5-coder-14b-instruct": {"cpu", "cuda", "webgpu"},
    "qwen2.5-coder-3b-instruct": {"cpu", "cuda", "webgpu"},
    "qwen2.5-coder-7b-instruct": {"cpu", "cuda", "webgpu"},
    "qwen3-0.6b": {"cpu", "cuda", "webgpu"},
    "qwen3-0.6b-pp-finetuned": {"cpu"},
    "qwen3-0.6b-pp-finetuned-mtt": {"cpu"},
    "qwen3-1.7b": {"cpu", "cuda", "webgpu"},
    "qwen3-14b": {"cpu", "cuda", "webgpu"},
    "qwen3-4b": {"cpu", "cuda", "webgpu"},
    "qwen3-8b": {"cpu", "cuda", "webgpu"},
    "qwen3.5-0.8b": {"cpu", "cuda", "webgpu"},
    "qwen3.5-2b": {"cpu", "cuda", "webgpu"},
    "qwen3.5-2b-text": {"cpu", "cuda", "webgpu"},
    "qwen3.5-4b": {"cpu", "cuda", "webgpu"},
    "qwen3.5-9b": {"cpu", "cuda", "webgpu"},
    "smollm3-3b": {"cpu", "cuda", "webgpu"},
}


PINNED_VERSIONS: dict[str, int] = {
    "qwen2.5-0.5b-instruct-paged": 1,
}


PINNED_IDENTITY: dict[str, dict[str, str]] = {
    "qwen2.5-0.5b-instruct-paged": {
        "genai_config.json": "2d2b99a2a2d8049fd921d4483fb05f4fad22603f058fc1dffd5f73f9a36e148c",
        "model.onnx": "0d83e2d70d043d8b1723ae1d94619bc191b14b459bada10fda20a9d314ce53c5",
        "model.onnx.data": "ad98abcb190a2085bca70110df04b04128c405fda8e0a526e2b7850d4d36a184",
    },
}


# Suites are ordered cheapest-first. The Engine suite runs in its own stage.
pr: list[str] = [
    "qwen2.5-0.5b-instruct",
    "qwen3-0.6b",
    "Phi-3.5-mini-instruct",
    "Phi-4-mini-instruct",
    "smollm3-3b",
    "ministral-3-3b-Instruct-2512",
]

all_: list[str] = [
    *pr,
    "Phi-3-mini-4k-instruct",
    "Phi-4",
    "Phi-4-mini-reasoning",
    "Phi-4-reasoning",
    "deepseek-r1-distill-qwen-1.5b",
    "olmo-3-7b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-3b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-coder-1.5b-instruct",
    "qwen3-1.7b",
    "qwen3-4b",
    "qwen3-8b",
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    "qwen3.5-4b",
]

engine: list[str] = [
    "qwen2.5-0.5b-instruct-paged",
]


SUITES: dict[str, list[str]] = {
    "pr": pr,
    "all": all_,
    "engine": engine,
}


def supports(logical_id: str, device: str) -> bool:
    return device in MODELS.get(logical_id, set())


def pinned_version(logical_id: str) -> int | None:
    """Return the pinned ``v<N>`` for ``logical_id``, or None to use newest."""
    return PINNED_VERSIONS.get(logical_id)


def pinned_identity(logical_id: str) -> dict[str, str] | None:
    """Return ``{filename: sha256}`` for ``logical_id``'s pinned artifact.

    None when no content identity is recorded (the id is not pinned by hash).
    """
    return PINNED_IDENTITY.get(logical_id)


def storage_subpath(logical_id: str, device: str) -> str:
    """Relative path under the blob container for ``(logical_id, device)``.

    The ``vN`` subdirectory is appended at runtime by the resolver, which
    picks either the version pinned in ``PINNED_VERSIONS`` or, failing that,
    the newest version present.
    """
    return f"{logical_id}/onnx/{DEVICE_DIRNAMES[device]}"
