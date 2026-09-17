# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Small offline checks for public artifact, resolver, suite and partition contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import conftest, fetch_public_models, models, resolver
from .check_models_in_sync import main as check_catalog
from .test_integration_multimodal import _assert_reference, _audit_profile, _session_overrides

_MODEL = "Phi-3.5-vision-instruct"


def test_multimodal_catalog_is_separate_and_pinned():
    assert _MODEL in models.multimodal
    assert _MODEL not in models.pr + models.all_ + models.engine
    assert models.MODELS[_MODEL] == {"cpu", "cuda"}
    artifact = models.PUBLIC_ARTIFACTS[_MODEL]
    assert artifact["revision"] == "672d73375fa86f3d7787e40ac593e33a4f04a055"
    for device in models.MODELS[_MODEL]:
        identity = models.PUBLIC_IDENTITY[_MODEL][device]
        assert len(identity) == 11
        assert len([name for name in identity if name.endswith(".onnx.data")]) == 3
        assert all(len(digest) == 64 for digest in identity.values())


@pytest.mark.parametrize("provider", ["cpu", "cuda"])
@pytest.mark.parametrize("default_kernels", [False, True])
def test_default_kernels_are_separate_from_precision_oracle(provider, default_kernels):
    roles = _session_overrides(provider, Path("build"), default_kernels=default_kernels)
    disabled = roles["decoder"]["session_options"].get("session.disable_prepacking") == "1"
    assert disabled is (provider == "cpu" and not default_kernels)
    for role in ("vision", "embedding", "decoder"):
        options = roles[role]["session_options"]
        assert options["provider_options"] == (
            [] if provider == "cpu" else [{"cuda": {"device_filtering_options": {"hardware_device_type": "gpu"}}}]
        )
        assert "config_entries" not in options
        if role != "decoder":
            assert "session.disable_prepacking" not in options


def test_catalog_checker_detects_multimodal_drift(capsys):
    base = ["--pr", ",".join(models.pr), "--all", ",".join(models.all_)]
    assert check_catalog([*base, "--multimodal", ",".join(models.multimodal)]) == 0
    assert check_catalog([*base, "--multimodal", "unverified-model"]) == 1
    assert "multimodal_models" in capsys.readouterr().err
    assert check_catalog(base) == 0


@pytest.mark.parametrize("multimodal", [False, True])
def test_default_collection_does_not_add_vision_models_to_text(multimodal):
    selected = {}
    metafunc = SimpleNamespace(
        fixturenames=["model"],
        config=SimpleNamespace(getoption=lambda option: []),
        definition=SimpleNamespace(get_closest_marker=lambda marker: multimodal or None),
        parametrize=lambda name, values: selected.update({name: values}),
    )
    conftest.pytest_generate_tests(metafunc)
    assert (_MODEL in selected["model"]) is multimodal
    if multimodal:
        assert selected["model"] == models.multimodal


def test_opt_in_markers_are_independent():
    items = [
        SimpleNamespace(keywords={marker: True}, add_marker=lambda mark, marker=marker: skipped.append(marker))
        for marker in ("engine", "multimodal", "text")
    ]
    skipped = []
    config = SimpleNamespace(getoption=lambda option: option == "--run-engine-tests")
    conftest.pytest_collection_modifyitems(config, items)
    assert skipped == ["multimodal"]


def test_required_resolver_rejects_unconfigured_and_unverified(monkeypatch):
    monkeypatch.delenv("ORTGENAI_MODEL_ROOT", raising=False)
    with pytest.raises(pytest.fail.Exception, match="Required model source missing"):
        resolver.get_path_for(_MODEL, "cpu", required=True)
    with pytest.raises(pytest.fail.Exception, match="No verified artifact"):
        resolver.get_path_for(_MODEL, "webgpu", required=True)
    with pytest.raises(pytest.skip.Exception, match="No model source"):
        resolver.get_path_for("qwen3-0.6b", "cpu")


def test_resolver_pins_normalized_public_layout(tmp_path):
    base = tmp_path / models.storage_subpath(_MODEL, "cpu")
    for version in (1, 99):
        directory = base / f"v{version}"
        directory.mkdir(parents=True)
        (directory / "genai_config.json").write_text("{}", encoding="utf-8")
    assert resolver.get_path_for(_MODEL, "cpu", model_root=str(tmp_path), required=True) == base / "v1"


def test_public_fetch_pins_revision_and_checks_all_content(tmp_path, monkeypatch):
    payload = b"public pinned fixture"
    source = tmp_path / "source"
    source.write_bytes(payload)
    identity = {"genai_config.json": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setitem(models.PUBLIC_IDENTITY, _MODEL, {"cpu": identity})
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr(fetch_public_models, "hf_hub_download", download)
    destination = fetch_public_models.fetch(_MODEL, "cpu", tmp_path)
    assert calls[0]["revision"] == models.PUBLIC_ARTIFACTS[_MODEL]["revision"]
    assert calls[0]["token"] is False
    assert calls[0]["subfolder"] == models.PUBLIC_ARTIFACTS[_MODEL]["subdirs"]["cpu"]
    assert destination == tmp_path / models.storage_subpath(_MODEL, "cpu") / "v1"
    assert (destination / "genai_config.json").stat().st_nlink == 1
    fetch_public_models.verify_artifact(destination, _MODEL, "cpu")
    (destination / "genai_config.json").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fetch_public_models.fetch(_MODEL, "cpu", tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize("fallback_type", ["float", "float16"])
def test_partition_audit_rejects_numerical_cpu_fallback(tmp_path, fallback_type):
    profile = tmp_path / "profile.json"
    gpu = {
        "cat": "Node",
        "name": "gpu_embedding_kernel_time",
        "args": {"provider": "CUDAExecutionProvider", "op_name": "Gather"},
    }
    cpu = {
        "cat": "Node",
        "name": "cpu_shape_kernel_time",
        "args": {"provider": "CPUExecutionProvider", "op_name": "Shape", "input_type_shape": [{"float16": [2, 3]}]},
    }
    profile.write_text(json.dumps([gpu, cpu]), encoding="utf-8")
    _audit_profile(profile, "embedding", "cuda")
    cpu["args"].update(op_name="Gather", input_type_shape=[{fallback_type: [2, 3]}, {"int64": [1]}])
    profile.write_text(json.dumps([gpu, cpu]), encoding="utf-8")
    with pytest.raises(AssertionError, match="unapproved fallback"):
        _audit_profile(profile, "embedding", "cuda")


def test_partition_audit_requires_actual_session_work(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text("[]", encoding="utf-8")
    with pytest.raises(AssertionError, match="No actual partition evidence"):
        _audit_profile(profile, "vision", "cuda")


def test_reference_oracle_does_not_advance_uncommitted_eos():
    generator = SimpleNamespace(is_done=lambda: True)
    with pytest.raises(AssertionError, match="before sampling EOS"):
        _assert_reference(None, generator, None, True)


@pytest.mark.parametrize("op", ["SequenceConstruct", "SplitToSequence"])
def test_partition_audit_allows_only_integer_sequence_metadata(tmp_path, op):
    profile = tmp_path / "profile.json"
    events = [
        {
            "cat": "Node",
            "name": "gpu_embedding_kernel_time",
            "args": {"provider": "CUDAExecutionProvider", "op_name": "Gather"},
        },
        {
            "cat": "Node",
            "name": "integer_sequence_kernel_time",
            "args": {
                "provider": "CPUExecutionProvider",
                "op_name": op,
                "input_type_shape": [{"int64": [2, 3]}, {"int64": []}],
                "output_type_shape": [],
            },
        },
    ]
    profile.write_text(json.dumps(events), encoding="utf-8")
    _audit_profile(profile, "embedding", "cuda")
    events[1]["args"]["input_type_shape"][0] = {"float16": [2, 3]}
    profile.write_text(json.dumps(events), encoding="utf-8")
    with pytest.raises(AssertionError, match="unapproved fallback"):
        _audit_profile(profile, "embedding", "cuda")


def test_partition_audit_inspects_control_flow_body(tmp_path):
    profile = tmp_path / "profile.json"
    events = [
        {
            "cat": "Node",
            "name": "pinned_loop_kernel_time",
            "args": {"provider": "CPUExecutionProvider", "op_name": "Loop"},
        },
        {
            "cat": "Node",
            "name": "body_matmul_kernel_time",
            "args": {"provider": "CUDAExecutionProvider", "op_name": "MatMul"},
        },
    ]
    profile.write_text(json.dumps(events), encoding="utf-8")
    _audit_profile(profile, "vision", "cuda", {"pinned_loop": "Loop"})
    with pytest.raises(AssertionError, match="unapproved fallback"):
        _audit_profile(profile, "vision", "cuda", {"different_loop": "Loop"})
    events[1]["args"]["provider"] = "CPUExecutionProvider"
    profile.write_text(json.dumps(events), encoding="utf-8")
    with pytest.raises(AssertionError, match="body_matmul"):
        _audit_profile(profile, "vision", "cuda", {"pinned_loop": "Loop"})
