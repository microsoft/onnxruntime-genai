# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/python/qwen38_flash_next/build_package_gguf.py"


@pytest.fixture
def package_builder():
    spec = importlib.util.spec_from_file_location("build_package_gguf", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_builds_variants_without_runtime_library_paths(package_builder, monkeypatch, tmp_path):
    gguf_root = tmp_path / "gguf"
    engram = gguf_root / "engram/Qwen3.8-Flash-Next-engram-IQ4_NL.gguf_file"
    engram.parent.mkdir(parents=True)
    engram.touch()
    for variant in ("UD-IQ1_S", "UD-IQ2_XS"):
        (gguf_root / variant).mkdir()
    base_pkg = tmp_path / "base"
    out_root = tmp_path / "output"
    calls = []

    def build_package(variant_name, gguf_dir, engram_gguf, base_pkg, out_dir):
        calls.append((variant_name, gguf_dir, engram_gguf, base_pkg, out_dir))

    monkeypatch.setattr(package_builder, "build_gguf_package", build_package)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--gguf-root",
            str(gguf_root),
            "--base-pkg",
            str(base_pkg),
            "--out-root",
            str(out_root),
            "--variants",
            "UD-IQ1_S, UD-IQ2_XS",
        ],
    )
    assert package_builder.main() == 0
    assert calls == [
        (
            f"qwen_flash_next_gguf_{slug}",
            gguf_root / variant,
            engram,
            base_pkg,
            out_root / f"qwen_flash_next_gguf_{slug}",
        )
        for variant, slug in (("UD-IQ1_S", "ud_iq1_s"), ("UD-IQ2_XS", "ud_iq2_xs"))
    ]


@pytest.mark.parametrize("option", ["--ort-lib-dir", "--plane-lib"])
def test_cli_rejects_removed_runtime_library_options(option):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), option, "unused"], check=False, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert f"unrecognized arguments: {option} unused" in result.stderr


@pytest.mark.parametrize(
    ("weight_bytes", "expected_mins"),
    [
        (38_285_551_616, (47_000_000_000, 48_000_000_000, 49_000_000_000, 52_000_000_000)),
        (47_738_679_296, (56_000_000_000, 57_000_000_000, 58_000_000_000, 61_000_000_000)),
        (55_544_619_008, (64_000_000_000, 65_000_000_000, 66_000_000_000, 69_000_000_000)),
    ],
)
def test_gguf_runtime_profiles_and_block_counts(package_builder, weight_bytes, expected_mins):
    assert package_builder.MIN_NUM_BLOCKS_32K == 141
    assert package_builder.NUM_BLOCKS_32K == 142
    assert package_builder.MIN_NUM_BLOCKS_64K == 269
    assert package_builder.NUM_BLOCKS_64K == 270
    assert package_builder.NUM_BLOCKS_128K == 525
    assert package_builder.NUM_BLOCKS_256K == 1037

    cfg = package_builder.default_base_genai_config()
    package_builder.configure_gguf_genai_config(cfg, total_weight_bytes=weight_bytes)
    profiles = {p["id"]: p for p in cfg["runtime_profiles"]}
    assert list(profiles) == ["gpu-32k", "gpu-64k", "gpu-128k", "gpu-256k"]
    assert profiles["gpu-32k"]["overlay"]["engine"]["dynamic_batching"]["num_blocks"] == 142
    assert profiles["gpu-64k"]["overlay"]["engine"]["dynamic_batching"]["num_blocks"] == 270
    assert profiles["gpu-128k"]["overlay"]["engine"]["dynamic_batching"]["num_blocks"] == 525
    assert profiles["gpu-256k"]["overlay"]["engine"]["dynamic_batching"]["num_blocks"] == 1037
    assert (
        profiles["gpu-32k"]["eligibility"]["minimum_total_device_memory_bytes"],
        profiles["gpu-64k"]["eligibility"]["minimum_total_device_memory_bytes"],
        profiles["gpu-128k"]["eligibility"]["minimum_total_device_memory_bytes"],
        profiles["gpu-256k"]["eligibility"]["minimum_total_device_memory_bytes"],
    ) == expected_mins
    assert "ep.cuda.qmoe_int_dequant_max_scratch_bytes" not in cfg["model"]["decoder"]["session_options"]
    assert cfg["model"]["bot_token_id"] == 248058
    assert cfg["model"]["eor_token_id"] == 248069


def test_precision_options_profiles_and_overrides():
    spec = importlib.util.spec_from_file_location(
        "precision_options",
        SCRIPT.parent / "precision_options.py",
    )
    assert spec is not None and spec.loader is not None
    po = importlib.util.module_from_spec(spec)
    sys.modules["precision_options"] = po
    spec.loader.exec_module(po)

    import argparse

    def parse(argv):
        ap = argparse.ArgumentParser()
        po.add_precision_cli_args(ap)
        return po.resolve_precision_config(ap.parse_args(argv))

    c_def = parse([])
    assert not c_def.is_non_default()
    assert c_def.kv_cache_dtype == "int8"
    assert c_def.rotary_cache_dtype == "bf16"

    c_fp32 = parse(["--precision-profile", "fp32"])
    assert c_fp32.is_non_default()
    assert c_fp32.kv_cache_dtype == "bf16"
    assert c_fp32.rotary_cache_dtype == "bf16"
    assert c_fp32.indexer_dtype == "fp32"
    assert c_fp32.gdn_gate_dtype == "fp32"
    assert c_fp32.gdn_precision == "fp32"
    assert c_fp32.ple_engram == "q8_0"
    assert c_fp32.ple_norm_dtype == "fp32"
    assert c_fp32.small_weights_source == "hf_bf16"

    c_fp16kv = parse(["--precision-profile", "fp32_fp16kv"])
    assert c_fp16kv.is_non_default()
    assert c_fp16kv.kv_cache_dtype == "fp16"
    assert c_fp16kv.rotary_cache_dtype == "fp32"
    assert c_fp16kv.qsa_qkv_dtype == "default"
    assert c_fp16kv.indexer_dtype == "fp32"
    assert c_fp16kv.gdn_gate_dtype == "fp32"
    assert c_fp16kv.gdn_precision == "fp32"
    assert c_fp16kv.ple_engram == "q8_0"

    c_v2 = parse(["--precision-profile", "fp32_fp16kv_v2"])
    assert c_v2.is_non_default()
    assert c_v2.kv_cache_dtype == "fp16"
    assert c_v2.rotary_cache_dtype == "fp32"
    assert c_v2.qsa_qkv_dtype == "v_rc5_fp16"
    assert c_v2.indexer_dtype == "fp32"
    assert c_v2.gdn_gate_dtype == "fp32"
    assert c_v2.gdn_precision == "fp32"
    assert c_v2.ple_engram == "default"

    c_custom = parse(["--precision-profile", "fp32", "--kv-cache-dtype", "fp16", "--ple-engram", "default"])
    assert c_custom.kv_cache_dtype == "fp16"
    assert c_custom.ple_engram == "default"
    assert c_custom.indexer_dtype == "fp32"
