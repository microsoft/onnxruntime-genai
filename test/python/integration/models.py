# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Catalog of models used by the integration pipelines.

Most logical ids are top-level folders under ``foundrylocalmodels/models/``.
The paged Engine model is fetched from ``staging/paged-attention`` and
the opt-in VLM from a hash-pinned public Hugging Face revision. Both use:
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


# Text models in the blob container, plus explicitly pinned Engine/public
# multimodal entries. For Foundry text models, update by listing the parent
# directories of every genai_config.json in the container:
#     az storage blob list --account-name foundrylocalmodels \
#         --container-name models --auth-mode login \
#         --query "[?ends_with(name, 'genai_config.json')].name" -o tsv \
#         | sed 's:/genai_config.json$::'
MODELS: dict[str, set[str]] = {
    "Phi-3-mini-128k-instruct": {"cpu", "cuda", "webgpu"},
    "Phi-3-mini-4k-instruct": {"cpu", "cuda", "webgpu"},
    "Phi-3.5-mini-instruct": {"cpu", "cuda", "webgpu"},
    # Public pinned VLM; deliberately excluded from the text suites.
    "Phi-3.5-vision-instruct": {"cpu", "cuda"},
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
    "Phi-3.5-vision-instruct": 1,
}

PUBLIC_ARTIFACTS = {
    "Phi-3.5-vision-instruct": {
        "repo_id": "microsoft/Phi-3.5-vision-instruct-onnx",
        "revision": "672d73375fa86f3d7787e40ac593e33a4f04a055",
        "subdirs": {
            "cpu": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
            "cuda": "gpu/gpu-int4-rtn-block-32",
        },
    },
}

# SHA-256 of every file in each pinned public export, including processor and
# tokenizer data. Keep the immutable revision and these identities together.
_PHI_COMMON_IDENTITY = {
    "genai_config.json": "aab58b61ca85a8bf2d4781fe1b4299e3d0f20a4cd902bc5c02adfe361b62aad9",
    "processor_config.json": "b6f5a3ceb16ad076b31430e358d68f0ebb53d9ddeb91270e9e5c4ff8fefed67f",
    "tokenizer.json": "478fb6fd0bf1403eb2ffda892e285f5ae968b9e0c06eb2815552ebedacb8ca93",
}
PUBLIC_IDENTITY = {
    "Phi-3.5-vision-instruct": {
        "cpu": {
            **_PHI_COMMON_IDENTITY,
            "phi-3.5-v-instruct-embedding.onnx": "83bb330b0a319812eccdf6b6bc5ce17eb6a120bcf76120f6af27c19c3d396ea1",
            "phi-3.5-v-instruct-embedding.onnx.data": "2dcbf0fcb14433ac8fa82b35c39675c01fb552512b4c0e5d54e8f247046ef393",
            "phi-3.5-v-instruct-text.onnx": "f7580bd589693e87e3825ffe8eda3f5fda4158d2d82d27803c263aa0e7454a81",
            "phi-3.5-v-instruct-text.onnx.data": "214fd2e8a09b8c7b141f9095171aed6f994768a36803b7eeb9ea3be16da99dd6",
            "phi-3.5-v-instruct-vision.onnx": "e06ccb2d2c35c2565f0abe507241f33cae9d7f63a1d79f6170b317d12f35c3bf",
            "phi-3.5-v-instruct-vision.onnx.data": "45af0e626ba0eff702d9c1873c84f9e60f1f17bb2036bf1abc045968b5e645df",
            "special_tokens_map.json": "f9535fada19315e0d8eb89521cebe111c008fee98d16e7c6744acfb50505d8c3",
            "tokenizer_config.json": "d637971d627e4f8ce0422401bd0a305d9f763657ad21e8205afe8d36b6672524",
        },
        "cuda": {
            **_PHI_COMMON_IDENTITY,
            "phi-3.5-v-instruct-embedding.onnx": "b707f7f659d8585a9253f3b95b1740522f3062a7eb4465f09d93c984743e399d",
            "phi-3.5-v-instruct-embedding.onnx.data": "e0af51a7c122ed63878262191faa905c1e317c7153185a6f7275e61d22fda63a",
            "phi-3.5-v-instruct-text.onnx": "d598bf368ed0b84b9d9db91a6a4843a2c92d071bf51df6763daadb3213922204",
            "phi-3.5-v-instruct-text.onnx.data": "de9df6b7062a3d133af7674041e583534dff5a95758572c15468eb1bf602e576",
            "phi-3.5-v-instruct-vision.onnx": "30d5d0dc644db606e42cf0d759acb2a22aef4ef572c77689e910ad9fa3b5ad1e",
            "phi-3.5-v-instruct-vision.onnx.data": "68aafcb815850a4982644fc81efec6800054865bb6dbc4ff41f275cc73533eac",
            "special_tokens_map.json": "8aa9f4bae46e5280787c8979252ffbf81f411a9aae79650f33a03d2a4b282514",
            "tokenizer_config.json": "7743c1615083d279af0170c5550dd7f40ea4de275fb3a99a5cb09bbe43e50662",
        },
    },
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

multimodal: list[str] = [
    "Phi-3.5-vision-instruct",
]

SUITES: dict[str, list[str]] = {
    "pr": pr,
    "all": all_,
    "engine": engine,
    "multimodal": multimodal,
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
    """Normalized local subpath for ``(logical_id, device)``.

    Also the Foundry blob subpath for non-public artifacts.
    The ``vN`` subdirectory is appended at runtime by the resolver, which
    picks either the version pinned in ``PINNED_VERSIONS`` or, failing that,
    the newest version present.
    """
    return f"{logical_id}/onnx/{DEVICE_DIRNAMES[device]}"
