# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests for the unified Qwen3.8 RC5 package builder (`tools/python/qwen38_flash_next/build_package.py`) and helpers.

Covers:
1. KL-similarity `(x INT4 + y INT3 + z INT2)` QMoE layer sorting, 2D dynamic programming, and predefined RC5 presets.
2. SM80 INT4/INT2 prepack/unpack roundtrips, raw INT3 pack/unpack, MSE quantization, PLE INT8 shards, and `ContribWeightWriter`.
3. Path safety (`reject_output_overlap`) across all input flags before any file deletion or write.
4. `model.qwen38_graph` config migration (`migrate_qwen38_graph_config`), INT3 scratch session options, and `runtime_profiles`.
5. RC5 graph surgery (`SpeculativePrefixCommit` 7-in/3-out, `RouterGateUpProj` fusion, `QMoE` `(x, y, z)` rewriting, context tiers).
6. Unified GPU + QMoE offload readiness (`_qwen38_qmoe_offload_contract` simulation for INT4, INT3, INT2, NVFP4, and STRP v1 plans).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import hashlib
import importlib
import json
import os
import struct
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from _qwen38_qmoe_offload_contract import (
    parse_node_overrides,
    simulate_session_creation,
    validate_popularity_file,
)
from _qwen38_tools_test_utils import (
    REPO_ROOT,
    prefill_chunk_problem,
    unknown_qwen38_graph_keys,
)
from onnx import TensorProto, helper, numpy_helper

ALIGN = 4096
STRP_MAGIC = 0x50525453
FLASH_NEXT_TOOLS_DIR = REPO_ROOT / "tools" / "python" / "qwen38_flash_next"


@pytest.fixture(scope="module")
def builders():
    sys.path.insert(0, str(FLASH_NEXT_TOOLS_DIR))
    try:
        config = importlib.import_module("package_config")
        paths = importlib.import_module("package_paths")
        gb = importlib.import_module("graph_builder")
        pkg = importlib.import_module("build_package")
        rope = importlib.import_module("rope")
        yield types.SimpleNamespace(config=config, paths=paths, gb=gb, pkg=pkg, rope=rope)
    finally:
        for name in (
            "package_config",
            "package_paths",
            "package_io",
            "precision_options",
            "graph_builder",
            "build_package",
            "rope",
        ):
            sys.modules.pop(name, None)
        if sys.path and sys.path[0] == str(FLASH_NEXT_TOOLS_DIR):
            sys.path.pop(0)


def snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): "<dir>" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
    }


def spy_on_written_files(monkeypatch, root: Path) -> list:
    handles = []
    real_open = builtins.open
    resolved_root = root.resolve()

    def open_and_record(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if (
            isinstance(file, (str, os.PathLike))
            and set(mode) & set("wa+")
            and Path(file).resolve().is_relative_to(resolved_root)
        ):
            handles.append(handle)
        return handle

    monkeypatch.setattr(builtins, "open", open_and_record)
    return handles


# ---------------------------------------------------------------------------
# 1. KL-Similarity (x INT4 + y INT3 + z INT2) Sorting & Predefined Presets
# ---------------------------------------------------------------------------


def test_predefined_core_rc5_presets_reproduce_exact_layer_assignments(builders):
    pkg = builders.pkg

    # 1. all_int4: qwen_flash_next_int4_int8_rc5_contrib_dflash2 (48 / 0 / 0)
    spec_all4 = pkg.resolve_preset_spec("qwen_flash_next_int4_int8_rc5_contrib_dflash2")
    x4, y3, z2, mode = pkg.select_qmoe_layers_by_kl(
        spec_all4["x"], spec_all4["y"], spec_all4["z"], kl_mode=spec_all4["kl_mode"]
    )
    assert (x4, y3, z2, mode) == (list(range(48)), [], [], "rank")

    # 2. top16_44_33_b64: qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top16_44_33_b64 (16 / 32 / 0)
    spec_top16 = pkg.resolve_preset_spec("top16_44_33_b64")
    x4, y3, z2, mode = pkg.select_qmoe_layers_by_kl(
        spec_top16["x"], spec_top16["y"], spec_top16["z"], kl_mode=spec_top16["kl_mode"]
    )
    assert x4 == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 26, 27, 29]
    assert len(y3) == 32 and z2 == [] and mode == "rank"

    # 3. top08_44_33_b64: qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_top08_44_33_b64 (8 / 40 / 0)
    spec_top08 = pkg.resolve_preset_spec("top08_44_33_b64")
    x4, y3, z2, mode = pkg.select_qmoe_layers_by_kl(
        spec_top08["x"], spec_top08["y"], spec_top08["z"], kl_mode=spec_top08["kl_mode"]
    )
    assert x4 == [0, 1, 2, 3, 7, 26, 27, 29]
    assert len(y3) == 40 and z2 == [] and mode == "rank"

    # 4. 2b375_x02_y14_z32_dp23_b64: qwen_flash_next_int4_int8_rc5_contrib_dflash2_qmoe_2b375_x02_y14_z32_dp23_b64 (2 / 14 / 32)
    spec_2b375 = pkg.resolve_preset_spec("2b375_x02_y14_z32_dp23_b64")
    x4, y3, z2, mode = pkg.select_qmoe_layers_by_kl(
        spec_2b375["x"], spec_2b375["y"], spec_2b375["z"], kl_mode=spec_2b375["kl_mode"]
    )
    assert x4 == [32, 40]
    assert y3 == [1, 2, 5, 11, 12, 17, 18, 29, 31, 33, 38, 39, 43, 46]
    assert len(z2) == 32 and mode == "dp23"
    assert set(x4) | set(y3) | set(z2) == set(range(48))


@pytest.mark.parametrize(
    ("preset_name", "expected_x", "expected_y_or_z", "check_z"),
    [
        ("3b0_x06_y36_z06_dp3_b64", [24, 26, 28, 32, 40, 46], [2, 16, 23, 30, 34, 37], True),
        (
            "2b8_x04_y30_z14_dp3_b64",
            [24, 26, 32, 40],
            [2, 3, 9, 10, 16, 23, 28, 30, 34, 35, 36, 37, 41, 46],
            True,
        ),
        (
            "2b5_x02_y20_z26_dp3_b64",
            [26, 32],
            [0, 1, 5, 6, 8, 11, 15, 17, 18, 20, 21, 22, 25, 27, 31, 33, 38, 39, 43, 47],
            False,
        ),
        (
            "2b375_x00_y18_z30_dp23_b64",
            [],
            [0, 1, 2, 5, 11, 12, 14, 17, 18, 29, 31, 33, 38, 39, 40, 41, 43, 46],
            False,
        ),
        (
            "2b25_x00_y12_z36_dp2_b64",
            [],
            [1, 2, 12, 18, 23, 28, 29, 31, 39, 41, 43, 46],
            False,
        ),
        ("2b125_x00_y06_z42_dp2_b64", [], [1, 2, 18, 31, 43, 46], False),
        ("2b0_x00_y00_z48_b64", [], [], False),
    ],
)
def test_sub_3bit_champion_presets_match_kl_study(builders, preset_name, expected_x, expected_y_or_z, check_z):
    pkg = builders.pkg
    spec = pkg.resolve_preset_spec(preset_name)
    x4, y3, z2, _ = pkg.select_qmoe_layers_by_kl(spec["x"], spec["y"], spec["z"], kl_mode=spec["kl_mode"])
    assert x4 == expected_x
    if check_z:
        assert z2 == expected_y_or_z
    else:
        assert y3 == expected_y_or_z


def test_auto_kl_mode_selects_regime_appropriately(builders):
    pkg = builders.pkg
    assert pkg.select_qmoe_layers_by_kl(16, 32, 0, kl_mode="auto")[3] == "rank"
    assert pkg.select_qmoe_layers_by_kl(6, 36, 6, kl_mode="auto")[3] == "dp3"
    assert pkg.select_qmoe_layers_by_kl(2, 14, 32, kl_mode="auto")[3] == "dp23"
    assert pkg.select_qmoe_layers_by_kl(0, 6, 42, kl_mode="auto")[3] == "dp2"


@pytest.mark.parametrize("kl_mode", ["rank", "dp4", "dp3", "dp23", "dp2"])
def test_all_kl_modes_partition_arbitrary_xyz(builders, kl_mode):
    pkg = builders.pkg
    x4, y3, z2, mode = pkg.select_qmoe_layers_by_kl(4, 20, 24, kl_mode=kl_mode, bs3=32, bs2=32)
    assert mode == kl_mode
    assert (len(x4), len(y3), len(z2)) == (4, 20, 24)
    assert set(x4) | set(y3) | set(z2) == set(range(48))


def test_custom_kl_study_json_is_supported(builders, tmp_path):
    pkg = builders.pkg
    study_file = tmp_path / "study.json"
    study_data = {
        "uniform_results": {
            "uniform_int3_b64": {"kl_mean": 0.15},
            "uniform_int2_b64": {"kl_mean": 0.40},
        },
        "single_from_int4": {
            k: [{"layer": i, "kl_mean": 0.01 * (i + 1)} for i in range(4)]
            for k in ("int3_b64", "int3_b32", "int2_b64", "int2_b32")
        },
        "single_from_int3_b64": {
            "to_int4_b32": [{"layer": i, "kl_mean": 0.15 - 0.01 * (i + 1)} for i in range(4)],
            "to_int2_b64": [{"layer": i, "kl_mean": 0.15 + 0.02 * (i + 1)} for i in range(4)],
            "to_int2_b32": [{"layer": i, "kl_mean": 0.15 + 0.02 * (i + 1)} for i in range(4)],
        },
        "single_from_int2_b64": {
            "to_int4_b32": [{"layer": i, "kl_mean": 0.40 - 0.03 * (i + 1)} for i in range(4)],
            "to_int3_b64": [{"layer": i, "kl_mean": 0.40 - 0.02 * (i + 1)} for i in range(4)],
        },
    }
    study_file.write_text(json.dumps(study_data))
    x4, y3, z2, _ = pkg.select_qmoe_layers_by_kl(1, 2, 1, kl_mode="dp23", num_layers=4, study_json=study_file)
    assert (x4, y3, z2) == ([3], [1, 2], [0])


def test_invalid_xyz_or_preset_raises(builders):
    pkg = builders.pkg
    with pytest.raises(ValueError, match="Unknown preset"):
        pkg.resolve_preset_spec("nonexistent_preset")
    with pytest.raises(ValueError, match="sum to 48"):
        pkg.select_qmoe_layers_by_kl(10, 10, 10)
    with pytest.raises(ValueError, match="non-negative"):
        pkg.select_qmoe_layers_by_kl(-1, 25, 24)
    with pytest.raises(ValueError, match="Unknown kl_mode"):
        pkg.select_qmoe_layers_by_kl(16, 32, 0, kl_mode="bad")


def test_compute_size_info_matches_reference_packages(builders):
    pkg = builders.pkg
    s_all4 = pkg.compute_size_info(48, 0, 0)
    assert s_all4["avg_bits"] == 4.0 and s_all4["qmoe_total_gib"] == 63.281
    s_top16 = pkg.compute_size_info(16, 32, 0)
    assert s_top16["qmoe_total_gib"] == 51.562
    s_2b375 = pkg.compute_size_info(2, 14, 32)
    assert s_2b375["avg_bits"] == 2.375 and s_2b375["qmoe_total_gib"] == 37.061


# ---------------------------------------------------------------------------
# 2. Quantization, SM80 Prepack/Unpack, PLE Shards & ContribWeightWriter
# ---------------------------------------------------------------------------


def test_sm80_int2_and_raw_int3_pack_unpack_roundtrip(builders):
    pkg = builders.pkg
    rng = torch.Generator().manual_seed(42)
    q2 = torch.randint(-2, 2, (2, 128, 128), generator=rng, dtype=torch.int8)
    packed2 = pkg.prepack_qmoe_sm80_torch(q2, bits=2)
    assert list(packed2.shape) == [2, 128, 32]
    q2_rec = pkg.unpack_sm80_int2_to_signed(packed2, 2, 128, 128)
    assert torch.equal(q2.to(torch.int16), q2_rec)

    w_bf16 = torch.randn((2, 64, 64), generator=rng, dtype=torch.bfloat16)
    p3, s3 = pkg.quantize_and_pack_raw_qmoe(w_bf16, bits=3, block_size=32, mode="mse", sm80_prepack=False)
    assert list(p3.shape) == [2, 64, 24] and list(s3.shape) == [2, 64, 2]
    q3_rec = pkg.unpack_raw_int3_to_signed(p3, 2, 64, 64)
    assert q3_rec.min().item() >= -4 and q3_rec.max().item() <= 3

    p4, s4 = pkg.quantize_and_pack_raw_qmoe(w_bf16, bits=4, block_size=32, mode="sym_rtn", sm80_prepack=True)
    assert list(p4.shape) == [2, 64, 32] and list(s4.shape) == [2, 64, 2]

    p2, s2 = pkg.quantize_and_pack_raw_qmoe(w_bf16, bits=2, block_size=64, mode="mse", sm80_prepack=True)
    assert list(p2.shape) == [2, 64, 16] and list(s2.shape) == [2, 64, 1]


def test_ple_shards_are_quantized_and_closed_on_error(builders, tmp_path, monkeypatch):
    pkg = builders.pkg
    v12d_dir = tmp_path / "v12d"
    hf_dir = tmp_path / "hf"
    v12d_dir.mkdir()
    hf_dir.mkdir()

    shard_lines = ["qwen38_ple_table 1", "scale -0.0005", "width 160"]
    raw_fp8 = bytearray()
    for i in range(128):
        rows = 2 + (i % 2)
        shard_lines.append(f"shard ple_fp8.bin {len(raw_fp8)} {rows}")
        vals = torch.full((rows * 160,), 0.25 if i % 2 == 0 else -0.25, dtype=torch.float32).to(torch.float8_e4m3fn)
        raw_fp8.extend(vals.view(torch.uint8).numpy().tobytes())
    (v12d_dir / "ple_table.txt").write_text("\n".join(shard_lines) + "\n")
    (hf_dir / "ple_fp8.bin").write_bytes(bytes(raw_fp8))

    out = tmp_path / "out_ok"
    res = pkg.build_ple_int8(v12d_dir, hf_dir, out)
    assert res["shards"] == 128 and res["files"] == 10
    assert (out / "ple_table.txt").exists()
    assert sorted(p.name for p in (out / "ple").iterdir()) == [f"ple_int8_{i:02d}.data" for i in range(10)]

    # Truncated source file closes opened handles and leaves no ple_table.txt
    (hf_dir / "ple_fp8.bin").write_bytes(bytes(raw_fp8[:320]))
    out_fail = tmp_path / "out_fail"
    handles = spy_on_written_files(monkeypatch, out_fail)
    with pytest.raises(RuntimeError, match="Short PLE read"):
        pkg.build_ple_int8(v12d_dir, hf_dir, out_fail)
    assert handles and [h.name for h in handles if not h.closed] == []
    assert not (out_fail / "ple_table.txt").exists()


def test_contrib_weight_writer_copy_on_write_and_alignment(builders, tmp_path):
    pkg = builders.pkg
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    ref_data = ref_dir / "contrib_weights.data"
    ref_data.write_bytes(b"ABCD" * 256)
    ref_hash = hashlib.sha256(ref_data.read_bytes()).hexdigest()

    # 1. Reused without new writes stays a hard link
    out1 = tmp_path / "out1"
    out1.mkdir()
    w1 = pkg.ContribWeightWriter(ref_dir, out1, reuse_data=ref_data)
    w1.close()
    assert os.path.samefile(ref_data, out1 / "contrib_weights.data")

    # 2. Reused with new writes copies-on-write and aligns to 4096 bytes
    out2 = tmp_path / "out2"
    out2.mkdir()
    w2 = pkg.ContribWeightWriter(ref_dir, out2, reuse_data=ref_data)
    meta = w2.write_bytes(b"XYZ", TensorProto.UINT8, [3])
    w2.close()
    assert meta["offset"] == ALIGN and meta["length"] == 3
    assert not os.path.samefile(ref_data, out2 / "contrib_weights.data")
    assert hashlib.sha256(ref_data.read_bytes()).hexdigest() == ref_hash
    with pytest.raises(ValueError, match="closed"):
        w2.write_bytes(b"1", TensorProto.UINT8, [1])


# ---------------------------------------------------------------------------
# 3. Output Overlap Protection & Config Surface Migration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overlap", ["equal", "child", "parent"])
@pytest.mark.parametrize("flag", ["--base-pkg", "--pool-dir", "--hf-dir", "--expert-profile", "--provider-lib"])
def test_builder_refuses_output_overlapping_any_input(builders, tmp_path, flag, overlap):
    base = tmp_path / "inputs" / "base"
    base.mkdir(parents=True)
    (base / "genai_config.json").write_text("{}")
    target = base if flag == "--base-pkg" else (tmp_path / "inputs" / flag.lstrip("-"))
    if flag in ("--expert-profile", "--provider-lib"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"data")
    else:
        target.mkdir(parents=True, exist_ok=True)

    if overlap == "equal":
        out = target
    elif overlap == "child":
        out = target / "sub"
    else:
        out = target.parent

    before = snapshot(tmp_path / "inputs")
    argv = ["--base-pkg", str(base), "--out", str(out)]
    if flag != "--base-pkg":
        argv.extend([flag, str(target)])
    with pytest.raises(SystemExit, match="must not be equal to, nested inside or contain"):
        builders.pkg.main(argv)
    assert snapshot(tmp_path / "inputs") == before


@pytest.mark.parametrize("location", ["nested", "equal", "resolved", "symlink", "presets"])
def test_rope_config_overlap_is_rejected_before_output_mutation(builders, tmp_path, location):
    pkg = builders.pkg
    reuse = tmp_path / "reuse"
    reuse.mkdir()
    out = tmp_path / "output"
    if location == "presets":
        first = out / pkg.resolve_preset_spec("all_int4")["pkg_name"]
        first.mkdir(parents=True)
        (first / "keep").write_text("unchanged")
        out /= pkg.resolve_preset_spec("top08")["pkg_name"]
    out.mkdir(parents=True)
    (out / "keep").write_text("unchanged")
    rope_path = out / "rope.json"
    rope_path.write_text('{"rope_type": "default"}\n')
    if location == "equal":
        out = rope_path
    elif location == "resolved":
        alias = tmp_path / "alias"
        alias.mkdir()
        out = alias / ".." / out.name
    elif location == "symlink":
        alias = tmp_path / "rope-alias.json"
        alias.symlink_to(rope_path)
        rope_path = alias

    argv = ["--reuse-banks-from", str(reuse), "--rope-config", str(rope_path)]
    if location == "presets":
        argv.extend(["--models-root", str(out.parent), "--presets", "all_int4,top08"])
    else:
        argv.extend(["--out", str(out)])
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit, match="contain --rope-config"):
        pkg.main(argv)
    assert snapshot(tmp_path) == before


def test_rope_config_outside_output_remains_supported(builders, tmp_path, monkeypatch):
    rope_path, reuse = tmp_path / "rope.json", tmp_path / "reuse"
    rope_path.write_text('{"rope_type": "default"}\n')
    reuse.mkdir()
    before = rope_path.read_bytes()

    def stop_before_weights(*args, **kwargs):
        raise RuntimeError("procedural export reached")

    monkeypatch.setattr(builders.pkg, "_prepare_procedural_base_assets", stop_before_weights)
    with pytest.raises(RuntimeError, match="procedural export reached"):
        builders.pkg.main(
            ["--out", str(tmp_path / "out"), "--reuse-banks-from", str(reuse), "--rope-config", str(rope_path)]
        )
    assert rope_path.read_bytes() == before


def test_config_migration_and_rc5_session_options(builders):
    pkg = builders.pkg
    legacy = {
        "model": {
            "decoder": {
                "filename": "model.onnx",
                "session_options": {
                    "provider_options": [
                        {
                            "cuda": {
                                "enable_cuda_graph": "1",
                                "min_num_runs_before_cuda_graph_capture": "4",
                                "arena_extend_strategy": "kSameAsRequested",
                            }
                        }
                    ]
                },
            },
            "qwen38_graph": {
                "kv_capacity": 33032,
                "warmup": True,
                "target_s1_graph_id": 1,
                "mtp": {"enabled": True, "prefill_head": False, "skip_refeed": False},
                "fold": {
                    "enabled": True,
                    "target_graph_id": 71,
                    "force_reject": False,
                    "mtp": {"filename": "mtp.onnx", "graph_id": 81, "session_options": {}},
                },
                "prefill": {"enabled": True, "chunk": 2048, "weights_pool": "weights_pool.txt"},
            },
        },
        "search": {"max_length": 33024},
    }

    cfg_int3 = copy.deepcopy(legacy)
    changes = pkg.configure_rc5_genai_config(cfg_int3, has_int3_layers=True)
    assert changes
    qwen = cfg_int3["model"]["qwen38_graph"]
    assert unknown_qwen38_graph_keys(qwen) == []
    assert prefill_chunk_problem(qwen) is None
    assert qwen["prefill"]["chunk"] == 8192
    dec_opts = cfg_int3["model"]["decoder"]["session_options"]
    assert dec_opts["ep.cuda.qmoe_int_dequant_max_scratch_bytes"] == "2147483648"
    assert dec_opts["ep.cuda.qmoe_row_tile_size"] == "8192"
    cuda_opts = dec_opts["provider_options"][0]["cuda"]
    assert "min_num_runs_before_cuda_graph_capture" not in cuda_opts
    assert cuda_opts == {"enable_cuda_graph": "1", "arena_extend_strategy": "kSameAsRequested"}
    assert cfg_int3["model"]["bot_token_id"] == 248058
    assert cfg_int3["model"]["eot_token_id"] == 248059
    assert cfg_int3["model"]["bor_token_id"] == 248068
    assert cfg_int3["model"]["eor_token_id"] == 248069
    assert cfg_int3["model"]["context_length"] == 262144
    assert cfg_int3["search"]["max_length"] == 262144
    assert cfg_int3["engine"] == {"dynamic_batching": {"max_batch_size": 1, "max_scheduled_tokens": 8192}}
    assert len(cfg_int3["runtime_profiles"]) == 7
    p_128k = cfg_int3["runtime_profiles"][-2]
    p_256k = cfg_int3["runtime_profiles"][-1]
    assert p_128k["id"] == "gpu-128k"
    assert p_128k["overlay"]["model"]["decoder"]["filename"] == "model.onnx"
    assert p_128k["overlay"]["engine"]["dynamic_batching"]["num_blocks"] == 525
    assert p_256k["id"] == "gpu-256k"
    assert p_256k["overlay"]["model"]["decoder"]["filename"] == "model.onnx"
    assert p_256k["overlay"]["engine"]["dynamic_batching"]["num_blocks"] == 1037

    # Per-model VRAM thresholds scale with QMoE size (2b375 < top08 < top16 < all_int4)
    v_2b375 = pkg.estimate_peak_vram_and_thresholds(2, 14, 32)
    v_top08 = pkg.estimate_peak_vram_and_thresholds(8, 40, 0)
    v_top16 = pkg.estimate_peak_vram_and_thresholds(16, 32, 0)
    v_all4 = pkg.estimate_peak_vram_and_thresholds(48, 0, 0)
    assert (
        v_2b375["min_vram_128k_bytes"]
        < v_top08["min_vram_128k_bytes"]
        < v_top16["min_vram_128k_bytes"]
        < v_all4["min_vram_128k_bytes"]
    )
    assert (
        v_2b375["min_vram_256k_bytes"]
        < v_top08["min_vram_256k_bytes"]
        < v_top16["min_vram_256k_bytes"]
        < v_all4["min_vram_256k_bytes"]
    )
    assert (v_2b375["min_vram_128k_bytes"], v_2b375["min_vram_256k_bytes"]) == (57_000_000_000, 61_000_000_000)
    assert (v_top08["min_vram_128k_bytes"], v_top08["min_vram_256k_bytes"]) == (69_000_000_000, 73_000_000_000)
    assert (v_top16["min_vram_128k_bytes"], v_top16["min_vram_256k_bytes"]) == (72_000_000_000, 76_000_000_000)
    assert (v_all4["min_vram_128k_bytes"], v_all4["min_vram_256k_bytes"]) == (83_000_000_000, 87_000_000_000)

    cfg_int4_only = copy.deepcopy(cfg_int3)
    pkg.configure_rc5_genai_config(cfg_int4_only, has_int3_layers=False, prefill_chunk=4096, stop_token_ids=[248319])
    assert "ep.cuda.qmoe_int_dequant_max_scratch_bytes" not in cfg_int4_only["model"]["decoder"]["session_options"]
    assert cfg_int4_only["model"]["qwen38_graph"]["prefill"]["chunk"] == 4096
    assert cfg_int4_only["model"]["eos_token_id"] == [248319]


def test_procedural_graph_builders_emit_complete_onnx_models_without_templates(builders):
    gb = builders.gb

    # 1. Procedural embedding.onnx
    emb_m = gb.build_embedding_model_proto()
    assert len(emb_m.graph.node) == 5
    assert [i.name for i in emb_m.graph.input] == ["input_ids"]
    assert [o.name for o in emb_m.graph.output] == ["inputs_embeds"]

    # 2. Procedural dflash2.onnx
    bk_u16 = {
        f"/dflash2/layers.{lid}/{kind}/base_kernel_all_1x4x160x16": np.zeros((1, 1, 4, 160, 16), dtype=np.uint16)
        for lid in range(5)
        for kind in ("attention_conv", "mlp_conv")
    }
    df_m = gb.build_dflash2_model_proto({}, bk_u16)
    assert len(df_m.graph.node) == 310
    assert [o.name for o in df_m.graph.output[:2]] == ["draft_candidate_ids", "draft_scores"]

    # 3. Procedural 48-layer model.onnx
    target_m = gb.build_target_model_proto({}, {})
    assert len(target_m.graph.node) == 958
    assert len(builders.pkg.qmoe_nodes(target_m)) == 48
    assert sum(n.op_type == "RouterGateUpProj" for n in target_m.graph.node) == 48
    assert any(n.op_type == "SpeculativePrefixCommit" for n in target_m.graph.node)


# ---------------------------------------------------------------------------
# 4. Synthetic RC5 Package Builds, Offload Contract & STRP Popularity Plans
# ---------------------------------------------------------------------------


def make_synthetic_rc5_base_package(root: Path, num_layers: int = 4, num_experts: int = 8) -> tuple[Path, Path]:
    """Create a tiny synthetic 4-layer base package and shared low-bit bank pool for fast GPU-free testing."""
    base_pkg = root / "base_rc5"
    pool_dir = root / "_qmoe_lowbit_shared_banks"
    (base_pkg / "ple").mkdir(parents=True)
    (base_pkg / "provider").mkdir(parents=True)
    pool_dir.mkdir(parents=True)

    hidden_size, inter_size = 64, 64
    fc1_n, fc1_k = 2 * inter_size, hidden_size
    fc2_n, fc2_k = hidden_size, inter_size

    nodes = []
    inits = [
        numpy_helper.from_array(np.array([33032], dtype=np.int32), name="contrib_const_dsa_total_seq"),
        numpy_helper.from_array(np.array([8, 2 * inter_size], dtype=np.int64), name="const_split_router"),
        numpy_helper.from_array(np.zeros((hidden_size, 8 + 2 * inter_size), dtype=np.uint16), name="router_w"),
    ]

    # Create base INT4 and low-bit INT3/INT2 bank files
    fc1_w4_sz = num_experts * fc1_k * (fc1_n // 2)
    fc2_w4_sz = num_experts * fc2_k * (fc2_n // 2)
    fc1_s32_sz = num_experts * fc1_n * (fc1_k // 32) * 2
    fc2_s32_sz = num_experts * fc2_n * (fc2_k // 32) * 2
    (base_pkg / "qmoe_int4.data").write_bytes(b"\x88" * (num_layers * (fc1_w4_sz + fc2_w4_sz)))
    (base_pkg / "qmoe_scales.data").write_bytes(b"\x01" * (num_layers * (fc1_s32_sz + fc2_s32_sz)))

    for bs in (32, 64):
        (pool_dir / f"qmoe_fc1_int3_b{bs}.data").write_bytes(
            b"\x24" * (num_layers * num_experts * fc1_n * (fc1_k * 3 // 8))
        )
        (pool_dir / f"qmoe_fc1_int3_b{bs}_scales.data").write_bytes(
            b"\x01" * (num_layers * num_experts * fc1_n * (fc1_k // bs) * 2)
        )
        (pool_dir / f"qmoe_fc2_int3_b{bs}.data").write_bytes(
            b"\x24" * (num_layers * num_experts * fc2_n * (fc2_k * 3 // 8))
        )
        (pool_dir / f"qmoe_fc2_int3_b{bs}_scales.data").write_bytes(
            b"\x01" * (num_layers * num_experts * fc2_n * (fc2_k // bs) * 2)
        )
        (pool_dir / f"qmoe_fc1_int2_b{bs}_sm80.data").write_bytes(
            b"\x55" * (num_layers * num_experts * fc1_k * (fc1_n // 4))
        )
        (pool_dir / f"qmoe_fc1_int2_b{bs}_scales.data").write_bytes(
            b"\x01" * (num_layers * num_experts * fc1_n * (fc1_k // bs) * 2)
        )
        (pool_dir / f"qmoe_fc2_int2_b{bs}_sm80.data").write_bytes(
            b"\x55" * (num_layers * num_experts * fc2_k * (fc2_n // 4))
        )
        (pool_dir / f"qmoe_fc2_int2_b{bs}_scales.data").write_bytes(
            b"\x01" * (num_layers * num_experts * fc2_n * (fc2_k // bs) * 2)
        )

    for i in range(num_layers):
        pfx = f"model.layers.{i}.mlp"
        # Unfused router + shared-expert prefix to exercise fuse_router_gate_up_proj
        nodes.extend(
            [
                helper.make_node("MatMul", [f"h{i}", "router_w"], [f"{pfx}.mm_out"], name=f"{pfx}/MatMulBf16"),
                helper.make_node(
                    "Split",
                    [f"{pfx}.mm_out", "const_split_router"],
                    [f"{pfx}.router_logits", f"{pfx}.gate_up"],
                    name=f"{pfx}/SplitRouterSwishMul/Split",
                    axis=-1,
                ),
                helper.make_node(
                    "SwiGLU", [f"{pfx}.gate_up"], [f"{pfx}.swish"], name=f"{pfx}/SwiGLU", domain="com.microsoft"
                ),
            ]
        )
        w_off = i * (fc1_w4_sz + fc2_w4_sz)
        s_off = i * (fc1_s32_sz + fc2_s32_sz)
        for tname, loc, off, ln, dt, dims in (
            (
                f"{pfx}.fc1_experts_weights",
                "qmoe_int4.data",
                w_off,
                fc1_w4_sz,
                TensorProto.UINT8,
                [num_experts, fc1_k, fc1_n // 2],
            ),
            (
                f"{pfx}.fc2_experts_weights",
                "qmoe_int4.data",
                w_off + fc1_w4_sz,
                fc2_w4_sz,
                TensorProto.UINT8,
                [num_experts, fc2_k, fc2_n // 2],
            ),
            (
                f"{pfx}.fc1_scales",
                "qmoe_scales.data",
                s_off,
                fc1_s32_sz,
                TensorProto.BFLOAT16,
                [num_experts, fc1_n, fc1_k // 32],
            ),
            (
                f"{pfx}.fc2_scales",
                "qmoe_scales.data",
                s_off + fc1_s32_sz,
                fc2_s32_sz,
                TensorProto.BFLOAT16,
                [num_experts, fc2_n, fc2_k // 32],
            ),
        ):
            t = TensorProto()
            t.name = tname
            t.data_type = dt
            t.dims.extend(dims)
            t.data_location = TensorProto.EXTERNAL
            for k, v in (("location", loc), ("offset", str(off)), ("length", str(ln))):
                e = t.external_data.add()
                e.key, e.value = k, v
            inits.append(t)

        # Include optional router_weights (slot 14) to verify builder clears it for offload compatibility
        q_inputs = [
            f"h{i}",
            f"{pfx}.router_logits",
            f"{pfx}.fc1_experts_weights",
            f"{pfx}.fc1_scales",
            "",
            f"{pfx}.fc2_experts_weights",
            f"{pfx}.fc2_scales",
            *[""] * 7,
            f"{pfx}.router_logits",
        ]
        nodes.append(
            helper.make_node(
                "QMoE",
                q_inputs,
                [f"h{i + 1}"],
                name=f"model/layers.{i}/mlp/QMoE",
                domain="com.microsoft",
                activation_type="swiglu",
                swiglu_fusion=1,
                normalize_routing_weights=1,
                k=2,
                expert_weight_bits=4,
                block_size=32,
                weights_prepacked=1,
            )
        )

    # Legacy 10-input/7-output SpeculativePrefixCommit to exercise ensure_rc5_speculative_prefix_commit
    nodes.append(
        helper.make_node(
            "SpeculativePrefixCommit",
            [
                "next_tokens",
                "input_ids",
                "validation.aggregate",
                "fold.ctl",
                "fold.ctx_commit",
                "fold.ctx_a",
                "fold.ctx_b",
                "fold.ptrs",
                "cos",
                "sin",
            ],
            ["fold.head_ids", "fold.tail_done", "s", "k", "rc", "rs", "sel"],
            name="fold/SpeculativePrefixCommit",
            domain="com.microsoft",
            head_rows=8,
        )
    )

    inputs = [
        helper.make_tensor_value_info("h0", TensorProto.BFLOAT16, [1, 8, hidden_size]),
        helper.make_tensor_value_info("past_key_values.0.key", TensorProto.INT8, [1, 2, 33032, 64]),
        helper.make_tensor_value_info("past_key_values.0.index_key", TensorProto.BFLOAT16, [1, 8259, 64]),
        helper.make_tensor_value_info("fold.ctl", TensorProto.INT64, [16]),
        helper.make_tensor_value_info("fold.ctx_commit", TensorProto.INT64, [2]),
        helper.make_tensor_value_info("fold.ctx_b", TensorProto.INT64, [8]),
    ]
    outputs = [
        helper.make_tensor_value_info(f"h{num_layers}", TensorProto.BFLOAT16, [1, 8, hidden_size]),
        helper.make_tensor_value_info("fold.tail_done", TensorProto.INT32, [1]),
    ]
    graph = helper.make_graph(nodes, "synthetic_rc5", inputs, outputs, inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)]
    )
    onnx.save_model(model, str(base_pkg / "model.onnx"), save_as_external_data=False)

    # Write STRP v1 expert-profile.bin: ranked (layer, expert) uint16 pairs, hottest first
    pairs = [(lay, exp) for exp in range(num_experts) for lay in range(num_layers)]
    header = struct.pack("<IIIIII", STRP_MAGIC, 1, num_layers, num_experts, num_layers * 2, len(pairs))
    body = b"".join(struct.pack("<HH", *p) for p in pairs)
    (base_pkg / "expert-profile.bin").write_bytes(header + body)
    (base_pkg / "weights_pool.txt").write_text("qwen38_weights_pool 1\n")
    (base_pkg / "ple" / "ple_int8_00.data").write_bytes(b"\x00" * 160)
    (base_pkg / "provider" / "libonnxruntime_providers_cuda.so").write_bytes(b"prov")
    (base_pkg / "genai_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "qwen38_flash_graph",
                    "decoder": {"filename": "model.onnx", "session_options": {}},
                    "qwen38_graph": {
                        "kv_capacity": 33032,
                        "warmup": True,
                        "mtp": {"enabled": True, "prefill_head": False},
                        "fold": {"enabled": True, "target_graph_id": 71},
                        "prefill": {"enabled": True, "chunk": 2048, "weights_pool": "weights_pool.txt"},
                    },
                }
            }
        )
    )
    return base_pkg, pool_dir


@pytest.mark.parametrize("mode", ["required", "auto"])
@pytest.mark.parametrize("placement", ["adaptive", "static"])
def test_built_mixed_432_package_passes_offload_and_gpu_contracts(builders, tmp_path, mode, placement):
    base_pkg, pool_dir = make_synthetic_rc5_base_package(tmp_path, num_layers=4, num_experts=8)
    out_pkg = tmp_path / "out_mixed"
    link_root = tmp_path / "links"

    rc = builders.pkg.main(
        [
            "--base-pkg",
            str(base_pkg),
            "--pool-dir",
            str(pool_dir),
            "--out",
            str(out_pkg),
            "--link-root",
            str(link_root),
            "--qmoe-xyz",
            "1,1,2",
            "--kl-mode",
            "dp23",
            "--context-tiers",
            "2k,8k,128k",
            "--offload-mode",
            mode,
            "--offload-placement",
            placement,
            "--total-resident-slots",
            "10",
            "--uniform-profile-slots",
        ]
    )
    assert rc == 0
    assert (link_root / "out_mixed").is_symlink()
    assert (out_pkg / "expert-profile.bin").is_file()
    validate_popularity_file(out_pkg / "expert-profile.bin")

    # Unified model.onnx uses symbolic kv_capacity / state_capacity and contrib_const_dsa_total_seq=[262152]
    m_unified = onnx.load(str(out_pkg / "model.onnx"), load_external_data=False)
    kv_in_unified = next(i for i in m_unified.graph.input if i.name == "past_key_values.0.key")
    assert kv_in_unified.type.tensor_type.shape.dim[2].dim_param == "kv_capacity"
    idx_in_unified = next(i for i in m_unified.graph.input if i.name == "past_key_values.0.index_key")
    assert idx_in_unified.type.tensor_type.shape.dim[1].dim_param == "state_capacity"

    for gname, exp_kv in (
        ("model_2k.onnx", 2312),
        ("model_8k.onnx", 8456),
        ("model_128k.onnx", 131080),
    ):
        m = onnx.load(str(out_pkg / gname), load_external_data=False)
        kv_in = next(i for i in m.graph.input if i.name == "past_key_values.0.key")
        assert kv_in.type.tensor_type.shape.dim[2].dim_value == exp_kv
        assert sum(n.op_type == "RouterGateUpProj" for n in m.graph.node) == 4
        spc = next(n for n in m.graph.node if n.op_type == "SpeculativePrefixCommit")
        assert len(spc.input) == 7 and list(spc.output) == ["fold.tail_done", "fold.ctl_out", "fold.ctx_commit_out"]
        assert "fold.ctx_b" not in {i.name for i in m.graph.input}

    m_main = onnx.load(str(out_pkg / "model.onnx"), load_external_data=False)
    cfg = json.loads((out_pkg / "genai_config.json").read_text())
    entries = cfg["model"]["decoder"]["session_options"]
    sim = simulate_session_creation(m_main, entries, model_dir=out_pkg)
    assert sim == [f"model/layers.{i}/mlp/QMoE" for i in range(4)]
    overrides = parse_node_overrides(entries["session.moe.offload.node_overrides"])
    assert sum(o["max_resident_slots"] for o in overrides) == 10
    assert sorted(o["max_resident_slots"] for o in overrides) == [2, 2, 3, 3]


def test_strp_profile_plans_and_error_handling(builders, tmp_path):
    pkg = builders.pkg
    base_pkg, _ = make_synthetic_rc5_base_package(tmp_path, num_layers=3, num_experts=8)
    prof = base_pkg / "expert-profile.bin"

    for total in range(25):
        for uniform in (False, True):
            plan = pkg.load_expert_profile_plan(prof, num_layers=3, num_experts=8, total_slots=total, uniform=uniform)
            assert sum(len(v) for v in plan.values()) == total

    with pytest.raises(ValueError, match="exceeds"):
        pkg.load_expert_profile_plan(prof, num_layers=3, num_experts=8, total_slots=25)
    with pytest.raises(ValueError, match="must not be negative"):
        pkg.load_expert_profile_plan(prof, num_layers=3, num_experts=8, total_slots=-1)


def test_strp_profile_records_are_layer_expert_pairs(builders, tmp_path):
    # The QMoE offload planner reads every ranked STRP v1 entry as two uint16 values (layer, expert).
    pkg, gb = builders.pkg, builders.gb
    base_pkg, _ = make_synthetic_rc5_base_package(tmp_path, num_layers=3, num_experts=8)
    prof = base_pkg / "expert-profile.bin"
    assert prof.stat().st_size == 24 + 3 * 8 * 4
    plan = pkg.load_expert_profile_plan(prof, num_layers=3, num_experts=8, total_slots=4, min_slots_per_layer=0)
    assert plan == {0: [0, 1], 1: [0], 2: [0]}

    gate_norms = {0: [1.0, 5.0, 3.0, 2.0], 1: [4.0, 0.5, 6.0, 0.1]}

    class GateLoader:
        def read_tensor(self, name):
            lid = int(name.split(".")[3])
            return torch.tensor(gate_norms[lid]).unsqueeze(1)

    written = tmp_path / "expert-profile-hf.bin"
    gb.build_expert_profile_from_hf(GateLoader(), written, num_layers=2, num_experts=4)
    assert written.stat().st_size == 24 + 2 * 4 * 4
    ranked = struct.unpack_from("<16H", written.read_bytes(), 24)
    assert list(zip(ranked[0::2], ranked[1::2], strict=True))[:3] == [(1, 2), (0, 1), (1, 0)]
    plan = pkg.load_expert_profile_plan(written, num_layers=2, num_experts=4, total_slots=3, min_slots_per_layer=0)
    assert plan == {0: [1], 1: [0, 2]}


def test_bank_metadata_and_weights_pool_roundtrip(builders, tmp_path):
    gb = builders.gb
    locs = {
        "model.rotary_emb.cos_cache": {
            "location": "norm_and_rope.data",
            "offset": 0,
            "length": 16777216,
            "dtype": 16,
            "dims": [262144, 32],
        },
        "model.rotary_emb.sin_cache": {
            "location": "norm_and_rope.data",
            "offset": 16777216,
            "length": 16777216,
            "dtype": 16,
            "dims": [262144, 32],
        },
        "capture.indexer_cos_cache.folded": {
            "location": "attn_consts.data",
            "offset": 0,
            "length": 33554432,
            "dtype": 16,
            "dims": [262144, 64],
        },
        "capture.indexer_sin_cache.folded": {
            "location": "attn_consts.data",
            "offset": 33554432,
            "length": 33554432,
            "dtype": 16,
            "dims": [262144, 64],
        },
        "lm_head.weight_t.qweight": {
            "location": "matmul_int8.data",
            "offset": 100,
            "length": 635699200,
            "dtype": 2,
            "dims": [248320, 80, 32],
        },
    }
    emb_w = {
        "v_model.layers.0.attn_hyper_connection.hc_norm.Add_87.folded": gb.make_bf16_tensor(
            "v_model.layers.0.attn_hyper_connection.hc_norm.Add_87.folded",
            np.ones((16,), dtype=np.uint16),
            [16],
        )
    }
    gb.save_bank_metadata_to_dir(tmp_path, locs, emb_w)
    loaded_locs, loaded_emb = gb.load_bank_metadata_from_dir(tmp_path)
    assert loaded_locs == locs
    assert set(loaded_emb) == set(emb_w)
    # Saving into a package whose bank_metadata.json is a hard link into another package must leave that package alone.
    linked = tmp_path / "linked"
    linked.mkdir()
    os.link(tmp_path / "bank_metadata.json", linked / "bank_metadata.json")
    gb.save_bank_metadata_to_dir(linked, {}, {})
    assert gb.load_bank_metadata_from_dir(linked)[0] == {}
    assert gb.load_bank_metadata_from_dir(tmp_path)[0] == locs
    gb.write_weights_pool_txt(tmp_path, locs)
    pool_text = (tmp_path / "weights_pool.txt").read_text()
    assert "qwen38_weights_pool 1" in pool_text
    assert "session dflash2 lm_head.weight_t.qweight 4" in pool_text


def test_removed_legacy_flags_and_conflicting_arguments_fail_cleanly(builders):
    pkg = builders.pkg
    with pytest.raises(SystemExit):
        pkg.main(["--migrate-shared-onnx"])
    with pytest.raises(SystemExit):
        pkg.main(["--base-pkg", "a", "--out", "b", "--prefill-chunk", "2048", "--prefill-chunk-tokens", "4096"])
    with pytest.raises(SystemExit):
        pkg.main(["--base-pkg", "a", "--out", "b", "-x", "10", "-y", "20"])


def test_precision_options_profiles_and_overrides():
    sys.path.insert(0, str(FLASH_NEXT_TOOLS_DIR))
    try:
        po = importlib.import_module("precision_options")
    finally:
        if sys.path and sys.path[0] == str(FLASH_NEXT_TOOLS_DIR):
            sys.path.pop(0)

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

    c_fp16kv = parse(["--precision-profile", "fp32_fp16kv"])
    assert c_fp16kv.is_non_default()
    assert c_fp16kv.kv_cache_dtype == "fp16"
    assert c_fp16kv.rotary_cache_dtype == "fp32"
    assert c_fp16kv.indexer_dtype == "fp32"
    assert c_fp16kv.gdn_gate_dtype == "fp32"
    assert c_fp16kv.gdn_precision == "fp32"
    assert c_fp16kv.ple_engram == "q8_0"

    c_v2 = parse(["--precision-profile", "fp32_fp16kv_v2"])
    assert c_v2.is_non_default()
    assert c_v2.kv_cache_dtype == "fp16"
    assert c_v2.qsa_qkv_dtype == "v_rc5_fp16"
    assert c_v2.ple_engram == "default"
    assert c_v2.emit_int8kv_variant is True

    c_int8kv_v2 = parse(["--precision-profile", "fp32_int8kv_v2"])
    assert c_int8kv_v2.is_non_default()
    assert c_int8kv_v2.kv_cache_dtype == "int8"
    assert c_int8kv_v2.qsa_qkv_dtype == "v_rc5_fp16"
    assert c_int8kv_v2.ple_engram == "default"
    assert c_int8kv_v2.emit_int8kv_variant is False

    c_custom = parse(["--precision-profile", "fp32", "--kv-cache-dtype", "fp16", "--ple-engram", "default"])
    assert c_custom.kv_cache_dtype == "fp16"
    assert c_custom.ple_engram == "default"
    assert c_custom.indexer_dtype == "fp32"


@pytest.fixture(scope="module")
def yarn_reference(builders):
    transformers = pytest.importorskip(
        "transformers", reason="Install qwen38_flash_next/requirements-yarn.txt for reference parity"
    )
    source = Path(transformers.__file__).parent / "modeling_rope_utils.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != builders.rope.HF_ROPE_SHA256:
        pytest.skip("Install qwen38_flash_next/requirements-yarn.txt for the exact optional reference")
    return builders.rope.yarn_parameters


@pytest.fixture(scope="module", autouse=True)
def export_test_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def test_default_rope_preserves_original_formulas(builders):
    positions = torch.tensor([0, 1, 32767, 262143, 524288, 1048591], dtype=torch.float32)
    inv = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    phase = positions[:, None] * inv[None, :]
    cos, sin = builders.rope.target_rope(positions)
    torch.testing.assert_close(cos, torch.cos(phase), rtol=0, atol=0)
    torch.testing.assert_close(sin, torch.sin(phase), rtol=0, atol=0)
    inv_d = 1.0 / (10000.0 ** (np.arange(0, 128, 2, dtype=np.float64) / 128))
    phase_d = positions.numpy().astype(np.float64)[:, None] * inv_d[None, :]
    for actual, fn in zip(builders.rope.dflash_rope(positions), (np.cos, np.sin), strict=True):
        np.testing.assert_array_equal(actual.numpy(), fn(phase_d).astype(np.float32))


@pytest.mark.parametrize("factor", [1.0, 4.0])
def test_yarn_reference_frequencies_ramp_magnitude_and_partial_rotation(builders, yarn_reference, factor):
    cfg = builders.rope.RopeConfig(rope_type="yarn", factor=factor)
    inv, magnitude = yarn_reference(cfg)
    frequencies = 10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    ramp = ((torch.arange(32, dtype=torch.float32) - 14) / (22 - 14)).clamp(0, 1)
    expected = (1.0 / (factor * frequencies)) * ramp + (1.0 / frequencies) * (1 - ramp)
    torch.testing.assert_close(inv, expected, rtol=0, atol=0)
    assert magnitude == (1.0 if factor == 1 else 1.138629436111989)
    positions = [0, 262143, 524288, 1048575, 1048583, 1048591]
    cos, sin = builders.rope.target_rope(positions, cfg)
    phase = torch.tensor(positions, dtype=torch.float32)[:, None] * inv[None, :]
    torch.testing.assert_close(cos, phase.cos() * magnitude, rtol=0, atol=0)
    torch.testing.assert_close(sin, phase.sin() * magnitude, rtol=0, atol=0)
    if factor == 1:
        torch.testing.assert_close(inv, 1.0 / frequencies, rtol=0, atol=0)
    q = torch.randn(len(positions), 256, generator=torch.Generator().manual_seed(7))
    rotated = q.clone()
    rotated[:, :32] = q[:, :32] * cos - q[:, 32:64] * sin
    rotated[:, 32:64] = q[:, 32:64] * cos + q[:, :32] * sin
    torch.testing.assert_close(rotated[:, 64:], q[:, 64:], rtol=0, atol=0)
    complex_q = torch.complex(q[:, :32], q[:, 32:64])
    expected_q = complex_q * torch.complex(cos, sin)
    torch.testing.assert_close(rotated[:, :32], expected_q.real)
    torch.testing.assert_close(rotated[:, 32:64], expected_q.imag)


@pytest.mark.parametrize(
    "values",
    [
        {"rope_type": "unknown"},
        {"rope_type": "yarn", "factor": float("nan")},
        {"rope_type": "yarn", "factor": 0.5},
        {"head_dim": 64},
        {"partial_rotary_factor": 1.0},
        {"mrope_sections": [16, 16]},
        {"factor": 4.0},
        {"unknown": 1},
    ],
)
def test_invalid_rope_configuration_is_rejected(builders, values):
    with pytest.raises(ValueError):
        builders.rope.RopeConfig.from_dict(values)


@pytest.mark.parametrize("context", [4096, 131072, 524288, 1048576])
@pytest.mark.parametrize("layout", ["unified", "fixed"])
def test_context_capacity_graph_and_selection_contract(builders, context, layout):
    cap = builders.rope.ContextCapacity(context, layout)
    assert (cap.kv_capacity, cap.state_capacity, cap.positional_rows) == (
        context + 8,
        (context + 8) // 4 + 1,
        context + 16,
    )
    gb, pkg = builders.gb, builders.pkg
    graph = gb.build_target_model_proto({}, {}, kv_capacity=cap.kv_capacity, state_capacity=cap.state_capacity)
    if layout == "unified":
        pkg.ensure_unified_dynamic_cache_model(graph)
    elif context == 1048576:
        assert cap.graph_filename == "model_1m.onnx"
    for vi in [*graph.graph.input, *graph.graph.output]:
        dims = vi.type.tensor_type.shape.dim
        if vi.name in ("past_position_ids", "present_position_ids"):
            assert dims[1].dim_value == cap.kv_capacity if layout == "fixed" else dims[1].dim_param == "kv_capacity"
        elif vi.name.endswith((".key", ".value")):
            assert dims[2].dim_value == cap.kv_capacity if layout == "fixed" else dims[2].dim_param == "kv_capacity"
        elif vi.name.endswith(".index_key"):
            assert (
                dims[1].dim_value == cap.state_capacity if layout == "fixed" else dims[1].dim_param == "state_capacity"
            )
    indexers = [n for n in graph.graph.node if n.op_type == "PackedSparseAttentionIndexer"]
    assert len(indexers) == 12
    assert all(next(a.i for a in n.attribute if a.name == "state_capacity") == cap.state_capacity for n in indexers)
    assert {p.key: p.value for p in graph.metadata_props}["experiment.fixed_kv_capacity"] == str(cap.kv_capacity)
    assert numpy_helper.to_array(
        next(t for t in graph.graph.initializer if t.name == "contrib_const_dsa_total_seq")
    ) == [cap.kv_capacity]
    assert all(
        next(a.f for a in n.attribute if a.name == "scale") == 0.0625
        for n in graph.graph.node
        if n.op_type == "DynamicSparseAttention"
    )
    tier = pkg.build_context_tier_model(graph, 4104, 1027)
    assert all(
        next(a.i for a in n.attribute if a.name == "state_capacity") == 1027
        for n in tier.graph.node
        if n.op_type == "PackedSparseAttentionIndexer"
    )
    config = pkg.default_base_genai_config()
    pkg.configure_rc5_genai_config(config, has_int3_layers=True, capacity=cap, prefill_chunk=512)
    assert config["model"]["decoder"]["num_attention_heads"] == 24
    assert config["model"]["decoder"]["filename"] == cap.graph_filename
    assert config["model"]["context_length"] == config["search"]["max_length"] == context
    assert config["model"]["qwen38_graph"]["kv_capacity"] == cap.kv_capacity
    assert config["runtime_profiles"] == []
    if layout == "fixed":
        assert "dynamic_batching" not in config["engine"]
    else:
        assert config["engine"]["dynamic_batching"]["num_blocks"] == context // 256 + pkg.DFLASH2_RESERVED_BLOCKS


def test_positional_serialization_pool_and_stale_reuse(builders, tmp_path, yarn_reference):
    gb, rope = builders.gb, builders.rope
    cfg = rope.RopeConfig(rope_type="yarn", factor=4)
    cap = rope.ContextCapacity(513, "fixed")
    locs = {}
    gb.build_reused_position_tables(tmp_path, locs, cfg, cap.positional_rows, cap.positional_rows)
    declared = rope.position_config(cfg, cap.positional_rows, cap.positional_rows, cap)
    gb.save_bank_metadata_to_dir(tmp_path, locs, {}, declared)
    assert rope.validate_reused_positions(tmp_path, locs) == declared
    mapped = {
        name: np.memmap(
            tmp_path / meta["location"], mode="r", dtype="<u2", offset=meta["offset"], shape=tuple(meta["dims"])
        )
        for name, meta in locs.items()
    }
    for source, duplicate in zip(rope.TARGET_NAMES[:2], rope.TARGET_NAMES[2:], strict=True):
        np.testing.assert_array_equal(mapped[duplicate], np.concatenate((mapped[source], mapped[source]), axis=1))
    for name in rope.DRAFT_NAMES:
        expected = rope.dflash_rope(np.arange(cap.positional_rows))[rope.DRAFT_NAMES.index(name)]
        np.testing.assert_array_equal(mapped[name], expected.to(torch.bfloat16).view(torch.uint16).numpy())
    locs["lm_head.weight_t.qweight"] = dict(location="unused.data", offset=0, length=1, dtype=2, dims=[1])
    gb.write_weights_pool_txt(tmp_path, locs)
    pool_lines = (tmp_path / "weights_pool.txt").read_text().splitlines()
    for index, name in enumerate(rope.TARGET_NAMES):
        fields = pool_lines[index + 1].split()
        meta = locs[name]
        assert fields[3:5] == [str(meta["offset"]), str(meta["length"])]
        assert fields[6:9] == [str(cap.positional_rows), str(meta["dims"][1]), meta["location"]]
    bad = copy.deepcopy(locs)
    bad[rope.TARGET_NAMES[0]]["length"] -= 2
    with pytest.raises(ValueError, match="external data"):
        rope.validate_reused_positions(tmp_path, bad)
    declared["target_rows"] -= 1
    gb.save_bank_metadata_to_dir(tmp_path, locs, {}, declared)
    with pytest.raises(ValueError, match="coverage metadata"):
        rope.validate_reused_positions(tmp_path, locs)
    declared["target_rows"] += 1
    gb.save_bank_metadata_to_dir(tmp_path, locs, {}, declared)
    del mapped
    with open(tmp_path / "position_tables.data", "r+b") as stream:
        stream.write(b"\xff\xff")
    with pytest.raises(ValueError, match="Stale or unqualified"):
        rope.validate_reused_positions(tmp_path, locs)


def test_legacy_reuse_requires_actual_standard_rope_bytes(builders, tmp_path):
    gb, rope = builders.gb, builders.rope
    locations = {}
    gb.build_reused_position_tables(tmp_path, locations, rope.RopeConfig(), 257, 261)
    assert rope.validate_reused_positions(tmp_path, locations)["target_rows"] == 257


def test_capacity_metadata_survives_tier_and_unified_conversion(builders):
    pkg, rope = builders.pkg, builders.rope
    cap = rope.ContextCapacity(1048576, "fixed")
    graph = builders.gb.build_target_model_proto({}, {}, kv_capacity=cap.kv_capacity, state_capacity=cap.state_capacity)
    props = {p.key: p.value for p in graph.metadata_props}
    props["qwen38.position_config"] = json.dumps(
        rope.position_config(rope.RopeConfig(), cap.positional_rows, cap.positional_rows, cap)
    )
    helper.set_model_props(graph, props)
    pkg.ensure_unified_dynamic_cache_model(graph)
    positions = json.loads({p.key: p.value for p in graph.metadata_props}["qwen38.position_config"])
    assert positions["capacity"] == rope.ContextCapacity(1048576, "unified").to_dict()
    tier = pkg.build_context_tier_model(graph, 4104, 1027)
    positions = json.loads({p.key: p.value for p in tier.metadata_props}["qwen38.position_config"])
    assert positions["capacity"] == rope.ContextCapacity(4096, "fixed").to_dict()
    with pytest.raises(ValueError, match="exceeds positional coverage"):
        pkg.build_context_tier_model(graph, 2097160, 524291)


@pytest.mark.parametrize("existing_output", [False, True])
def test_legacy_base_package_explicit_capacity_is_rejected_before_output_mutation(builders, tmp_path, existing_output):
    pkg, rope = builders.pkg, builders.rope
    source, pool = make_synthetic_rc5_base_package(tmp_path)
    cap = rope.ContextCapacity(1048576, "unified")
    model_path = source / "model.onnx"
    model = onnx.load(model_path, load_external_data=False)
    pkg.ensure_unified_dynamic_cache_model(model, cap.kv_capacity, cap.state_capacity)
    props = {p.key: p.value for p in model.metadata_props}
    props["qwen38.position_config"] = json.dumps(
        rope.position_config(rope.RopeConfig(rope_type="yarn", factor=4), cap.positional_rows, cap.positional_rows, cap)
    )
    helper.set_model_props(model, props)
    onnx.save_model(model, model_path, save_as_external_data=False)
    config_path = source / "genai_config.json"
    config = json.loads(config_path.read_text())
    pkg.configure_rc5_genai_config(config, has_int3_layers=True, capacity=cap, prefill_chunk=512)
    config_path.write_text(json.dumps(config))
    out = tmp_path / "output"
    if existing_output:
        out.mkdir()
        (out / "keep").write_text("unchanged")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="explicit-capacity.*--reuse-banks-from"):
        pkg.build_rc5_package(out, base_pkg=source, pool_dir=pool, x_int4=1, y_int3=1, z_int2=2, kl_mode="dp23")
    assert snapshot(tmp_path) == before


def test_cache_cannot_write_into_reuse_source(builders, tmp_path):
    source, out = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    out.mkdir()
    marker = out / "keep"
    marker.write_text("unchanged")
    with pytest.raises(ValueError, match="all model/checkpoint directories"):
        builders.pkg.build_rc5_package(
            out,
            x_int4=2,
            y_int3=14,
            z_int2=32,
            reuse_banks_from=source,
            capacity=builders.rope.ContextCapacity(1048576, "fixed"),
            tactic_cache_dir=source / "cache",
        )
    assert marker.read_text() == "unchanged"


def test_precision_overrides_fail_before_output_mutation(builders, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "keep"
    marker.write_text("unchanged")
    with pytest.raises(SystemExit) as exc:
        builders.pkg.main(
            [
                "--out",
                str(out),
                "--max-context-length",
                "1048576",
                "--cache-layout",
                "fixed",
                "--precision-profile",
                "fp32",
            ]
        )
    assert exc.value.code == 2
    assert marker.read_text() == "unchanged"
    (out / "package_manifest.json").write_text(json.dumps({"position_config": {"capacity": 1048576}}))
    po = importlib.import_module("precision_options")
    with pytest.raises(ValueError, match="not qualified"):
        po.apply_precision_config_to_package(out, po.PrecisionConfig(profile="fp32"))
    assert marker.read_text() == "unchanged"


def test_unqualified_yarn_reference_is_rejected(builders, tmp_path, monkeypatch):
    (tmp_path / "modeling_rope_utils.py").write_text("raise RuntimeError('must not execute')")
    monkeypatch.setattr(
        builders.rope.importlib.util, "find_spec", lambda _: types.SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    with pytest.raises(ValueError, match="Unqualified Transformers"):
        builders.rope.yarn_parameters(builders.rope.RopeConfig(rope_type="yarn", factor=4))


def test_actual_1m_prototype_graph_and_all_positional_rows(builders, yarn_reference):
    root = os.environ.get("QWEN38_REFERENCE_PACKAGE")
    if not root:
        pytest.skip("Set QWEN38_REFERENCE_PACKAGE to the read-only validated 1M package")
    root = Path(root)
    gb, pkg, rope = builders.gb, builders.pkg, builders.rope
    cfg = rope.RopeConfig(rope_type="yarn", factor=4)
    cap = rope.ContextCapacity(1048576, "fixed")
    locations, embedded = gb.load_bank_metadata_from_dir(root)
    original_config = json.loads((root / "genai_config.json").read_text())
    original = onnx.load(root / original_config["model"]["decoder"]["filename"], load_external_data=False)
    generated = gb.build_target_model_proto(
        locations, embedded, kv_capacity=cap.kv_capacity, state_capacity=cap.state_capacity
    )
    selection = pkg.select_qmoe_layers_by_kl(2, 14, 32, kl_mode="dp23")
    pkg.rewrite_qmoe_layers(generated, *selection[:3])

    def key(n):
        return n.domain, n.op_type, list(n.input), list(n.output), sorted(a.SerializeToString() for a in n.attribute)

    assert {n.name: key(n) for n in generated.graph.node} == {n.name: key(n) for n in original.graph.node}
    assert list(generated.graph.input) == list(original.graph.input)
    assert list(generated.graph.output) == list(original.graph.output)
    original_inits = {t.name: t for t in original.graph.initializer}
    assert set(original_inits) == {t.name for t in generated.graph.initializer}
    for tensor in generated.graph.initializer:
        expected = original_inits[tensor.name]
        if tensor.data_location == TensorProto.EXTERNAL:
            assert tensor == expected
        else:
            np.testing.assert_array_equal(numpy_helper.to_array(tensor), numpy_helper.to_array(expected))
    for names in (rope.TARGET_NAMES, rope.DRAFT_NAMES):
        for index, name in enumerate(names):
            meta = locations[name]
            assert meta["dims"][0] == cap.positional_rows
            actual = np.memmap(
                root / meta["location"], mode="r", dtype="<u2", offset=meta["offset"], shape=tuple(meta["dims"])
            )
            for start in range(0, cap.positional_rows, 4096):
                positions = np.arange(start, min(start + 4096, cap.positional_rows))
                values = rope.target_rope(positions, cfg) if names == rope.TARGET_NAMES else rope.dflash_rope(positions)
                expected = values[index % 2].to(torch.bfloat16)
                if names == rope.TARGET_NAMES and index >= 2:
                    expected = torch.cat((expected, expected), dim=-1)
                np.testing.assert_array_equal(
                    actual[start : start + len(positions)], expected.view(torch.uint16).numpy()
                )


def test_exported_package_against_1m_prototype(builders):
    reference = os.environ.get("QWEN38_REFERENCE_PACKAGE")
    exported = os.environ.get("QWEN38_EXPORTED_PACKAGE")
    if not reference or not exported:
        pytest.skip("Set QWEN38_REFERENCE_PACKAGE and QWEN38_EXPORTED_PACKAGE for complete-package qualification")
    reference, exported = Path(reference), Path(exported)
    cfg_ref = json.loads((reference / "genai_config.json").read_text())
    cfg = json.loads((exported / "genai_config.json").read_text())
    positional = set(builders.rope.TARGET_NAMES + builders.rope.DRAFT_NAMES)
    assert cfg["model"]["context_length"] == cfg["search"]["max_length"] == 1048576
    assert cfg["model"]["qwen38_graph"]["kv_capacity"] == 1048584
    assert cfg["model"]["decoder"]["filename"] == "model_1m.onnx"
    assert cfg["model"]["decoder"]["num_attention_heads"] == 24
    assert cfg["runtime_profiles"] == [] and "dynamic_batching" not in cfg["engine"]
    assert sorted(p.name for p in exported.glob("model*.onnx")) == ["model_1m.onnx"]
    all_initializers = {}
    for section in ("decoder", "dflash2", "embedding"):
        expected = onnx.load(reference / cfg_ref["model"][section]["filename"], load_external_data=False)
        actual = onnx.load(exported / cfg["model"][section]["filename"], load_external_data=False)

        def node_key(n):
            return (
                n.domain,
                n.op_type,
                list(n.input),
                list(n.output),
                sorted(a.SerializeToString() for a in n.attribute),
            )

        if section == "embedding":
            # The procedural embedding only changes names, independent Gather ordering and explicit axis=0.
            def embedding_contract(model):
                values = {t.name: ("initializer", t.name) for t in model.graph.initializer}
                values.update({v.name: ("input", v.name) for v in model.graph.input})
                for node in model.graph.node:
                    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                    if node.op_type == "Gather":
                        attrs.setdefault("axis", 0)
                    signature = (
                        node.domain,
                        node.op_type,
                        tuple(sorted(attrs.items())),
                        tuple(values[name] for name in node.input),
                    )
                    for index, name in enumerate(node.output):
                        values[name] = signature, index
                return {v.name: values[v.name] for v in model.graph.output}

            assert len(actual.graph.node) == len(expected.graph.node) == 5
            assert embedding_contract(actual) == embedding_contract(expected)
        else:
            assert {n.name: node_key(n) for n in actual.graph.node} == {
                n.name: node_key(n) for n in expected.graph.node
            }
        assert list(actual.graph.input) == list(expected.graph.input)
        assert list(actual.graph.output) == list(expected.graph.output)
        expected_inits = {t.name: t for t in expected.graph.initializer}
        assert set(expected_inits) == {t.name for t in actual.graph.initializer}
        all_initializers.update({t.name: t for t in actual.graph.initializer})
        for tensor in actual.graph.initializer:
            other = expected_inits[tensor.name]
            assert tensor.data_type == other.data_type and list(tensor.dims) == list(other.dims)
            if tensor.data_location != TensorProto.EXTERNAL:
                np.testing.assert_array_equal(numpy_helper.to_array(tensor), numpy_helper.to_array(other))
                continue
            desc = {e.key: e.value for e in tensor.external_data}
            path = exported / desc["location"]
            offset, length = int(desc["offset"]), int(desc["length"])
            assert path.resolve().is_relative_to(exported.resolve()) and not path.is_symlink()
            assert path.stat().st_nlink == 1
            assert 0 <= offset < offset + length <= path.stat().st_size
            if tensor.name not in positional:
                assert tensor == other
                continue
            desc_ref = {e.key: e.value for e in other.external_data}
            a = np.memmap(path, mode="r", dtype="<u2", offset=offset, shape=tuple(tensor.dims))
            b = np.memmap(
                reference / desc_ref["location"],
                mode="r",
                dtype="<u2",
                offset=int(desc_ref["offset"]),
                shape=tuple(other.dims),
            )
            for start in range(0, tensor.dims[0], 4096):
                np.testing.assert_array_equal(a[start : start + 4096], b[start : start + 4096])
        if section == "decoder":
            props = {p.key: p.value for p in actual.metadata_props}
            assert props["experiment.fixed_kv_capacity"] == "1048584"
            assert json.loads(props["qwen38.position_config"])["capacity"]["state_capacity"] == 262147
    pools = {}
    for line in (exported / "weights_pool.txt").read_text().splitlines()[1:]:
        fields = line.split()
        if fields[0] == "pool":
            pools[fields[1]] = fields
        else:
            tensor = all_initializers[fields[2]]
            desc = {e.key: e.value for e in tensor.external_data}
            entry = pools[fields[3]]
            assert entry[2:5] == [str(tensor.data_type), desc["offset"], desc["length"]]
            assert list(map(int, entry[6:-1])) == list(tensor.dims)
            assert entry[-1] == desc["location"]
    for section in ("decoder", "dflash2"):
        options = cfg["model"][section]["session_options"]
        assert options["ep.cuda.gemm_tactic_cache_to_model"] == "0"
        assert not Path(options["ep.cuda.gemm_tactic_cache_dir"]).resolve().is_relative_to(exported.resolve())


@pytest.fixture(params=["sendfile", "buffered"])
def file_range_copy(request, monkeypatch):
    """Runs a consolidation test with os.sendfile and with the buffered copy that Windows and macOS use."""
    if request.param == "sendfile" and not (sys.platform.startswith("linux") and hasattr(os, "sendfile")):
        pytest.skip("os.sendfile copies into regular files only on Linux")
    package_io = sys.modules[import_precision_options().append_file_range.__module__]
    monkeypatch.setattr(package_io, "USE_SENDFILE", request.param == "sendfile")
    # A small chunk makes the buffered copy loop over several reads and writes per range.
    monkeypatch.setattr(package_io, "COPY_CHUNK_BYTES", 64)
    return request.param


def test_consolidate_package_external_data_merges_and_page_aligns(tmp_path, file_range_copy):
    sys.path.insert(0, str(FLASH_NEXT_TOOLS_DIR))
    try:
        po = importlib.import_module("precision_options")
    finally:
        if sys.path and sys.path[0] == str(FLASH_NEXT_TOOLS_DIR):
            sys.path.pop(0)

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "bank_a.data").write_bytes(b"A" * 100)
    (pkg / "bank_b.data").write_bytes(b"B" * 200)

    def make_ext_init(name, loc, offset, length):
        t = onnx.TensorProto()
        t.name = name
        t.data_type = TensorProto.UINT8
        t.dims.extend([length])
        t.data_location = TensorProto.EXTERNAL
        for k, v in (("location", loc), ("offset", str(offset)), ("length", str(length))):
            kv = t.external_data.add()
            kv.key = k
            kv.value = v
        return t

    inits = [
        make_ext_init("w_a", "bank_a.data", 0, 100),
        make_ext_init("w_b", "bank_b.data", 0, 200),
        numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="node_GuardedDSA_3_k_scale"),
    ]
    g = helper.make_graph([], "test_g", [], [], initializer=inits)
    m = helper.make_model(g)
    onnx.save(m, str(pkg / "model.onnx"))
    onnx.save(m, str(pkg / "model_int8kv.onnx"))
    (pkg / "weights_pool.txt").write_text("qwen38_weights_pool 1\nsession dflash2 w_a 4\n")

    res = po.consolidate_package_external_data(pkg, target_data_name="model.onnx.data")
    assert res["consolidated"] is True and len(res["removed_files"]) == 2
    assert (pkg / "model.onnx.data").is_file()
    assert not (pkg / "bank_a.data").exists()
    assert not (pkg / "bank_b.data").exists()

    raw = (pkg / "model.onnx.data").read_bytes()
    assert raw[0:100] == b"A" * 100
    assert raw[4096:4296] == b"B" * 200

    for fname in ("model.onnx", "model_int8kv.onnx"):
        m_loaded = onnx.load(str(pkg / fname), load_external_data=False)
        ext_map = {
            init.name: {kv.key: kv.value for kv in init.external_data}
            for init in m_loaded.graph.initializer
            if init.data_location == TensorProto.EXTERNAL
        }
        assert ext_map["w_a"] == {"location": "model.onnx.data", "offset": "0", "length": "100"}
        assert ext_map["w_b"] == {"location": "model.onnx.data", "offset": "4096", "length": "200"}


def import_precision_options():
    sys.path.insert(0, str(FLASH_NEXT_TOOLS_DIR))
    try:
        return importlib.import_module("precision_options")
    finally:
        if sys.path and sys.path[0] == str(FLASH_NEXT_TOOLS_DIR):
            sys.path.pop(0)


def external_locations(path: Path) -> dict[str, dict[str, str]]:
    m = onnx.load(str(path), load_external_data=False)
    return {
        init.name: {kv.key: kv.value for kv in init.external_data}
        for init in m.graph.initializer
        if init.data_location == TensorProto.EXTERNAL
    }


def external_initializer(name: str, location: str, offset: int, length: int) -> TensorProto:
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.UINT8
    t.dims.extend([length])
    t.data_location = TensorProto.EXTERNAL
    for k, v in (("location", location), ("offset", str(offset)), ("length", str(length))):
        kv = t.external_data.add()
        kv.key, kv.value = k, v
    return t


def test_precision_options_and_package_io_do_not_import_the_builders():
    for module in ("precision_options.py", "package_io.py"):
        tree = ast.parse((FLASH_NEXT_TOOLS_DIR / module).read_text())
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not imported & {"build_package", "build_package_gguf"}, module


def test_consolidation_relocates_existing_target_data_file(tmp_path, file_range_copy):
    po = import_precision_options()
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "model.onnx.data").write_bytes(b"T" * 300)
    (pkg / "bank_a.data").write_bytes(b"A" * 100)

    inits = [
        external_initializer("w_t", "model.onnx.data", 100, 200),
        external_initializer("w_a", "bank_a.data", 0, 100),
    ]
    onnx.save(helper.make_model(helper.make_graph([], "g", [], [], initializer=inits)), str(pkg / "model.onnx"))
    tier = [external_initializer("w_a", "bank_a.data", 0, 100)]
    onnx.save(helper.make_model(helper.make_graph([], "g", [], [], initializer=tier)), str(pkg / "model_2k.onnx"))

    res = po.consolidate_package_external_data(pkg)
    assert res["consolidated"] is True and res["removed_files"] == ["bank_a.data"]
    raw = (pkg / "model.onnx.data").read_bytes()
    locs = external_locations(pkg / "model.onnx")
    for name, payload in (("w_t", b"T" * 200), ("w_a", b"A" * 100)):
        assert locs[name]["location"] == "model.onnx.data"
        off, length = int(locs[name]["offset"]), int(locs[name]["length"])
        assert raw[off : off + length] == payload
    assert external_locations(pkg / "model_2k.onnx")["w_a"] == locs["w_a"]
    assert po.consolidate_package_external_data(pkg) == {"consolidated": False}


def test_consolidation_failure_removes_partial_output_and_leaves_package_untouched(tmp_path, file_range_copy):
    po = import_precision_options()
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "bank_a.data").write_bytes(b"A" * 100)
    (pkg / "bank_b.data").write_bytes(b"B" * 50)  # the graph below claims 200 bytes
    inits = [external_initializer("w_a", "bank_a.data", 0, 100), external_initializer("w_b", "bank_b.data", 0, 200)]
    onnx.save(helper.make_model(helper.make_graph([], "g", [], [], initializer=inits)), str(pkg / "model.onnx"))
    before = snapshot(pkg)

    with pytest.raises(RuntimeError, match=r"bank_b\.data ends at byte 50, 150 bytes short"):
        po.consolidate_package_external_data(pkg)
    assert snapshot(pkg) == before


def test_consolidated_precision_build_rewrites_every_graph_and_keeps_base_package_intact(builders, tmp_path):
    base_pkg, pool_dir = make_synthetic_rc5_base_package(tmp_path, num_layers=4, num_experts=8)
    shared = TensorProto()
    shared.name = "dflash2.shared_scales"
    shared.data_type = TensorProto.UINT8
    shared.dims.extend([16])
    shared.data_location = TensorProto.EXTERNAL
    for k, v in (("location", "qmoe_scales.data"), ("offset", "0"), ("length", "16")):
        kv = shared.external_data.add()
        kv.key, kv.value = k, v
    dflash = helper.make_model(helper.make_graph([], "dflash2", [], [], initializer=[shared]))
    onnx.save(dflash, str(base_pkg / "dflash2.onnx"))
    before = snapshot(base_pkg)

    out_pkg = tmp_path / "out_consolidated"
    rc = builders.pkg.main(
        [
            "--base-pkg",
            str(base_pkg),
            "--pool-dir",
            str(pool_dir),
            "--out",
            str(out_pkg),
            "--qmoe-xyz",
            "1,1,2",
            "--kl-mode",
            "dp23",
            "--context-tiers",
            "2k,8k",
            "--consolidate-data-files",
        ]
    )
    assert rc == 0
    assert snapshot(base_pkg) == before
    assert not (out_pkg / "qmoe_scales.data").exists()
    for gname in ("model.onnx", "model_2k.onnx", "model_8k.onnx", "dflash2.onnx"):
        locs = external_locations(out_pkg / gname)
        assert locs and {e["location"] for e in locs.values()} == {"model.onnx.data"}, gname
    assert external_locations(out_pkg / "model_2k.onnx") == external_locations(out_pkg / "model.onnx")
    assert external_locations(base_pkg / "dflash2.onnx")["dflash2.shared_scales"]["location"] == "qmoe_scales.data"


def test_precision_build_from_base_package_keeps_its_bank_metadata_intact(builders, tmp_path):
    gb, rope_mod = builders.gb, builders.rope
    base_pkg, pool_dir = make_synthetic_rc5_base_package(tmp_path, num_layers=4, num_experts=8)
    rows = 16
    locs: dict[str, dict] = {}
    gb.build_reused_position_tables(base_pkg, locs, rope_mod.DEFAULT_ROPE, rows, rows)
    positions = rope_mod.position_config(rope_mod.DEFAULT_ROPE, rows, rows, None)
    gb.save_bank_metadata_to_dir(base_pkg, locs, {}, positions)
    model = onnx.load(str(base_pkg / "model.onnx"))
    for name in ("model.rotary_emb.cos_cache", "model.rotary_emb.sin_cache"):
        meta = locs[name]
        model.graph.initializer.append(
            gb.make_ext_init(name, meta["location"], meta["offset"], meta["length"], meta["dtype"], meta["dims"])
        )
        model.graph.node.append(helper.make_node("Identity", [name], [f"{name}.out"], name=f"use_{name}"))
    onnx.save(model, str(base_pkg / "model.onnx"))
    before = snapshot(base_pkg)

    # Without --consolidate-data-files, the conversion re-synchronizes the output's bank_metadata.json, which starts out
    # as a hard link of the base package's file.
    out_pkg = tmp_path / "out_fp32_rotary"
    rc = builders.pkg.main(
        [
            "--base-pkg",
            str(base_pkg),
            "--pool-dir",
            str(pool_dir),
            "--out",
            str(out_pkg),
            "--qmoe-xyz",
            "1,1,2",
            "--kl-mode",
            "dp23",
            "--rotary-cache-dtype",
            "fp32",
        ]
    )
    assert rc == 0
    assert snapshot(base_pkg) == before
    out_locs, _ = gb.load_bank_metadata_from_dir(out_pkg)
    assert out_locs["model.rotary_emb.cos_cache"]["location"] == "attn_consts.data"
    assert out_locs["model.rotary_emb.cos_cache"]["dtype"] == TensorProto.FLOAT


def test_fp16_qsa_projections_require_fp32_indexer(builders):
    po = import_precision_options()
    ap = argparse.ArgumentParser()
    po.add_precision_cli_args(ap)
    for qsa in ("v_rc5_fp16", "kv_fp16", "fp16"):
        with pytest.raises(ValueError, match="requires --indexer-dtype fp32"):
            po.resolve_precision_config(ap.parse_args(["--qsa-qkv-dtype", qsa]))
        with pytest.raises(ValueError, match="requires --indexer-dtype fp32"):
            po.resolve_precision_config(
                ap.parse_args(
                    ["--precision-profile", "fp32_fp16kv_v2", "--qsa-qkv-dtype", qsa, "--indexer-dtype", "bf16"]
                )
            )
        assert po.resolve_precision_config(ap.parse_args(["--qsa-qkv-dtype", qsa, "--indexer-dtype", "fp32"]))
    with pytest.raises(SystemExit):
        builders.pkg.main(["--base-pkg", "a", "--out", "b", "--qsa-qkv-dtype", "fp16"])


def test_aligned_blob_writer_page_aligns_and_replaces_existing_file(tmp_path):
    import_precision_options()
    package_io = sys.modules["package_io"]
    (tmp_path / "attn_consts.data").write_bytes(b"stale")
    writer = package_io.AlignedBlobWriter(tmp_path, "attn_consts.data")
    assert writer.write_raw(b"a" * 10, TensorProto.UINT8, [10]) == ("attn_consts.data", 0, 10, TensorProto.UINT8, [10])
    assert writer.write_raw(b"b" * 3, TensorProto.UINT8, [3])[1] == 4096
    assert (tmp_path / "attn_consts.data").read_bytes() == b"a" * 10 + b"\x00" * 4086 + b"b" * 3


def make_engram_gguf(path: Path, ggml_type: int, payload: bytes) -> int:
    """Writes a GGUF file holding only per_layer_token_embd.weight and returns its data offset."""
    name = b"per_layer_token_embd.weight"
    header = struct.pack("<4sIQQ", b"GGUF", 3, 1, 1)
    header += struct.pack("<Q", 17) + b"general.alignment" + struct.pack("<II", 4, 64)
    header += struct.pack("<Q", len(name)) + name + struct.pack("<I", 2) + struct.pack("<QQ", 160, 4)
    header += struct.pack("<IQ", ggml_type, 0)
    data_offset = (len(header) + 63) // 64 * 64
    path.write_bytes(header + b"\x00" * (data_offset - len(header)) + payload)
    return data_offset


@pytest.mark.parametrize(("engram_kind", "ggml_type"), [("q8_0", 8), ("iq4_nl", 20)])
def test_gguf_ple_table_names_its_format_instead_of_a_sentinel_scale(tmp_path, engram_kind, ggml_type):
    po = import_precision_options()
    src = tmp_path / "engram.gguf_file"
    make_engram_gguf(src, ggml_type, b"\x11" * 64)
    out = tmp_path / "out"
    out.mkdir()
    po.write_gguf_ple_table(out, engram_kind, src)
    keys = [ln.split()[0] for ln in (out / "ple_table.txt").read_text().splitlines()]
    assert f"format {engram_kind}" in (out / "ple_table.txt").read_text().splitlines()
    assert keys.count("format") == 1
    assert "scale" not in keys


def test_gguf_ple_table_is_copied_into_output_when_hard_link_fails(tmp_path, monkeypatch):
    po = import_precision_options()
    src = tmp_path / "src" / "engram.gguf_file"
    src.parent.mkdir()
    data_offset = make_engram_gguf(src, 8, b"\x11" * 680)
    out = tmp_path / "out"
    out.mkdir()

    def no_link(*_args):
        raise OSError("cross-device link")

    monkeypatch.setattr(po.os, "link", no_link)
    dst_name = po.write_gguf_ple_table(out, "q8_0", src)
    assert dst_name == "Qwen3.8-Flash-Next-engram-Q8_0.gguf"
    assert (out / dst_name).read_bytes() == src.read_bytes()
    shards = [ln.split() for ln in (out / "ple_table.txt").read_text().splitlines() if ln.startswith("shard ")]
    assert shards[0] == ["shard", dst_name, str(data_offset), "2500012"]
    assert sum(int(s[3]) for s in shards) == 300001275 + 20000171


def test_rc6_fp32_gdn_kl_calibration_and_presets(builders):
    pkg = builders.pkg
    gb = builders.gb
    assert gb.DEFAULT_KV_SCALE_FILE.is_file()
    raw_kv = json.loads(gb.DEFAULT_KV_SCALE_FILE.read_text(encoding="utf-8"))
    assert raw_kv["layer_ids"] == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
    assert raw_kv["qmax"] == 128.0
    assert len(raw_kv["scales"]["k_scales"]) == 12 and len(raw_kv["scales"]["v_scales"]) == 12
    kv_scales = gb.load_calibrated_kv_scales()
    assert len(kv_scales) == 24
    assert all(arr.shape == (512,) and arr.dtype == np.float32 for arr in kv_scales.values())

    l4, l3, l2, mode = pkg.select_qmoe_layers_by_kl(2, 14, 32, kl_mode="dp23_fp32")
    assert mode == "dp23_fp32"
    assert l4 == [28, 32]
    assert l3 == [1, 4, 14, 20, 21, 22, 31, 33, 35, 38, 39, 43, 45, 46]
    assert len(l2) == 32

    top16_4, top16_3, top16_2, top16_mode = pkg.select_qmoe_layers_by_kl(16, 32, 0, kl_mode="rank_fp32")
    assert top16_mode == "rank_fp32"
    assert top16_4 == [0, 1, 2, 3, 4, 5, 6, 7, 18, 24, 26, 27, 28, 29, 32, 33]
    assert len(top16_3) == 32 and top16_2 == []

    spec = pkg.resolve_preset_spec("rc6_2b375")
    assert spec["pkg_name"] == "qwen_flash_next_rc6_2b375_fp32"
    assert (spec["x"], spec["y"], spec["z"]) == (2, 14, 32)
    assert spec["kl_mode"] == "dp23_fp32"
    assert spec["precision_profile"] == "fp32_fp16kv_v2"
    assert spec["consolidate_data_files"] is True

    spec_top16 = pkg.resolve_preset_spec("rc6_top16")
    assert spec_top16["pkg_name"] == "qwen_flash_next_rc6_top16_fp32"
    assert (spec_top16["x"], spec_top16["y"], spec_top16["z"]) == (16, 32, 0)
    assert spec_top16["kl_mode"] == "rank"

    spec_all4 = pkg.resolve_preset_spec("rc6_all_int4")
    assert spec_all4["pkg_name"] == "qwen_flash_next_rc6_all_int4_fp32"
    assert (spec_all4["x"], spec_all4["y"], spec_all4["z"]) == (48, 0, 0)
    assert spec_all4["kl_mode"] == "rank"


def _write_safetensors_file(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    hdr: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, (dtype_str, shape, raw) in tensors.items():
        hdr[name] = {"dtype": dtype_str, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(hdr).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    if pad:
        hdr_bytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.writelines(chunks)


def _make_mini_qmoe_hf_checkpoint(
    ckpt_dir: Path,
    *,
    num_layers: int = 2,
    num_experts: int = 2,
    hidden_size: int = 64,
    inter_size: int = 64,
    use_language_model_prefix: bool = True,
    single_file: bool = True,
    fill_val: float = 0.25,
    padding_bytes: tuple[int, int] = (0, 0),
) -> Path:
    """Write the expert weights of a tiny QMoE checkpoint, optionally between unrelated filler tensors of the given sizes."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    lead, trail = padding_bytes
    if lead:
        tensors["model.language_model.embed_tokens.weight"] = ("U8", [lead], bytes(lead))
    prefix = "model.language_model.layers" if use_language_model_prefix else "model.layers"
    for lid in range(num_layers):
        gu_numel = num_experts * (2 * inter_size) * hidden_size
        dn_numel = num_experts * hidden_size * inter_size
        gu = (
            torch.sin(torch.arange(gu_numel, dtype=torch.float32) * fill_val + lid)
            .reshape(num_experts, 2 * inter_size, hidden_size)
            .to(torch.bfloat16)
        )
        dn = (
            torch.cos(torch.arange(dn_numel, dtype=torch.float32) * fill_val + lid)
            .reshape(num_experts, hidden_size, inter_size)
            .to(torch.bfloat16)
        )
        tensors[f"{prefix}.{lid}.mlp.experts.gate_up_proj"] = (
            "BF16",
            list(gu.shape),
            gu.view(torch.uint16).numpy().tobytes(),
        )
        tensors[f"{prefix}.{lid}.mlp.experts.down_proj"] = (
            "BF16",
            list(dn.shape),
            dn.view(torch.uint16).numpy().tobytes(),
        )
    if trail:
        tensors["lm_head.weight"] = ("U8", [trail], bytes(trail))
    if single_file:
        _write_safetensors_file(ckpt_dir / "model.safetensors", tensors)
    else:
        shard_name = "model-00001-of-00001.safetensors"
        _write_safetensors_file(ckpt_dir / shard_name, tensors)
        weight_map = dict.fromkeys(tensors, shard_name)
        (ckpt_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}, indent=2) + "\n", encoding="utf-8"
        )
    return ckpt_dir


def test_unified_checkpoint_reader_single_and_sharded_language_model_keys(builders, tmp_path, monkeypatch):
    pkg = builders.pkg
    gb = builders.gb
    for single_file in (True, False):
        ckpt = _make_mini_qmoe_hf_checkpoint(
            tmp_path / ("single" if single_file else "sharded"),
            num_layers=2,
            num_experts=2,
            hidden_size=64,
            inter_size=64,
            use_language_model_prefix=True,
            single_file=single_file,
        )
        loader = gb.CheckpointTensorLoader(ckpt)
        t_gu = loader.read_bf16_tensor("model.layers.0.mlp.experts.gate_up_proj")
        assert t_gu.shape == (2, 128, 64) and t_gu.dtype == torch.bfloat16

        pool = tmp_path / f"pool_{'single' if single_file else 'sharded'}"
        pkg.ensure_qmoe_int4_bank(pool, hf_dir=ckpt, num_layers=2, num_experts=2, hidden_size=64, inter_size=64, bs4=32)
        assert (pool / "qmoe_int4.data").is_file()
        assert (pool / "qmoe_scales.data").is_file()
        assert (pool / "qmoe_bank_metadata.json").is_file()

        specs = [
            (
                "fc1",
                3,
                64,
                "mse",
                False,
                "qmoe_fc1_int3_b64.data",
                "qmoe_fc1_int3_b64_scales.data",
                [2, 128, 64 * 3 // 8],
                [2, 128, 1],
            ),
            (
                "fc2",
                2,
                64,
                "mse",
                True,
                "qmoe_fc2_int2_b64_sm80.data",
                "qmoe_fc2_int2_b64_scales.data",
                [2, 64, 64 // 4],
                [2, 64, 1],
            ),
        ]
        pkg.ensure_qmoe_banks(pool, specs, hf_dir=ckpt, num_layers=2)
        assert (pool / "qmoe_fc1_int3_b64.data").is_file()
        assert (pool / "qmoe_fc2_int2_b64_sm80.data").is_file()

    # Failure mid-quantization leaves no partial bank or .tmp files behind
    fail_pool = tmp_path / "fail_pool"
    ckpt_ok = _make_mini_qmoe_hf_checkpoint(tmp_path / "ckpt_fail", num_layers=2, single_file=True)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated quantization failure")

    monkeypatch.setattr(pkg, "fast_quant_prepack_batch", boom)
    with pytest.raises(RuntimeError, match="simulated quantization failure"):
        pkg.ensure_qmoe_int4_bank(
            fail_pool, hf_dir=ckpt_ok, num_layers=2, num_experts=2, hidden_size=64, inter_size=64, bs4=32
        )
    assert not (fail_pool / "qmoe_int4.data").exists()
    assert not (fail_pool / "qmoe_scales.data").exists()
    assert list(fail_pool.glob("*.tmp.*")) == []


def test_qmoe_bank_validation_rejects_truncated_files_and_stale_checkpoint_pools(builders, tmp_path):
    pkg = builders.pkg
    base_pkg, pool_dir = make_synthetic_rc5_base_package(tmp_path / "trunc", num_layers=4, num_experts=8)
    # Truncate qmoe_int4.data and qmoe_scales.data in base_pkg to 1 byte
    (base_pkg / "qmoe_int4.data").write_bytes(b"\x00")
    (base_pkg / "qmoe_scales.data").write_bytes(b"\x00")

    out_fail = tmp_path / "out_trunc_fail"
    with pytest.raises(FileNotFoundError, match="qmoe_int4.data"):
        pkg.build_rc5_package(out_fail, base_pkg=base_pkg, pool_dir=pool_dir, x_int4=4, y_int3=0, z_int2=0)

    # With a valid checkpoint, truncated base_pkg banks are ignored and regenerated into pool_dir
    ckpt_a = _make_mini_qmoe_hf_checkpoint(
        tmp_path / "ckpt_a", num_layers=4, num_experts=8, hidden_size=64, inter_size=64, fill_val=0.15
    )
    out_regen = tmp_path / "out_regen"
    pkg.build_rc5_package(out_regen, base_pkg=base_pkg, pool_dir=pool_dir, hf_dir=ckpt_a, x_int4=4, y_int3=0, z_int2=0)
    int4_bytes_a = (out_regen / "qmoe_int4.data").read_bytes()
    assert len(int4_bytes_a) > 1

    # Building from a different checkpoint B with the same shared pool_dir detects the fingerprint change and regenerates
    ckpt_b = _make_mini_qmoe_hf_checkpoint(
        tmp_path / "ckpt_b", num_layers=4, num_experts=8, hidden_size=64, inter_size=64, fill_val=0.85
    )
    out_regen_b = tmp_path / "out_regen_b"
    pkg.build_rc5_package(
        out_regen_b, base_pkg=base_pkg, pool_dir=pool_dir, hf_dir=ckpt_b, x_int4=4, y_int3=0, z_int2=0
    )
    int4_bytes_b = (out_regen_b / "qmoe_int4.data").read_bytes()
    assert int4_bytes_b != int4_bytes_a


def test_fast_quant_prepack_batch_preserves_non_default_device(builders):
    pkg = builders.pkg

    class FakeDeviceTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, data: torch.Tensor, fake_dev: torch.device):
            obj = torch.Tensor._make_subclass(cls, data, data.requires_grad)
            obj._fake_dev = fake_dev
            return obj

        @property
        def is_cuda(self):  # type: ignore[override]
            return True

        @property
        def device(self):  # type: ignore[override]
            return self._fake_dev

        def to(self, *args, **kwargs):  # type: ignore[override]
            target_dev = kwargs.get("device", args[0] if args else None)
            FakeDeviceTensor.recorded_devices.append(str(target_dev))
            return torch.Tensor(self).to(dtype=kwargs.get("dtype", self.dtype))

    FakeDeviceTensor.recorded_devices = []
    raw = torch.randn(2, 64, 64, dtype=torch.bfloat16)
    wrapped = FakeDeviceTensor(raw, torch.device("cuda:1"))
    pkg.fast_quant_prepack_batch(wrapped, bits=4, block_size=32, is_qmoe=True)
    assert FakeDeviceTensor.recorded_devices[0] == "cuda:1"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fast_quant_prepack_batch_allocates_only_on_the_input_gpu(builders):
    pkg = builders.pkg
    weights = torch.randn(2, 64, 64, dtype=torch.bfloat16, device="cuda:1")
    explicit = torch.randn(2, 64, 64, dtype=torch.bfloat16)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    default_gpu_before = torch.cuda.memory_allocated(0)

    pkg.fast_quant_prepack_batch(weights, bits=4, block_size=32, is_qmoe=True)
    assert torch.cuda.max_memory_allocated(0) == default_gpu_before
    assert torch.cuda.max_memory_allocated(1) > weights.numel() * weights.element_size()

    # An explicit device also keeps a CPU-resident input off the thread's default GPU.
    pkg.fast_quant_prepack_batch(explicit, bits=4, block_size=32, is_qmoe=True, device="cuda:1")
    assert torch.cuda.max_memory_allocated(0) == default_gpu_before


def test_lowbit_qmoe_bank_reuse_requires_matching_settings_and_publishes_atomically(builders, tmp_path, monkeypatch):
    pkg = builders.pkg

    def spec(mode: str):
        return (
            "fc1",
            3,
            64,
            mode,
            False,
            "qmoe_fc1_int3_b64.data",
            "qmoe_fc1_int3_b64_scales.data",
            [2, 128, 64 * 3 // 8],
            [2, 128, 1],
        )

    ckpt = _make_mini_qmoe_hf_checkpoint(tmp_path / "ckpt", num_layers=2)
    pool = tmp_path / "pool"
    weights, scales = pool / "qmoe_fc1_int3_b64.data", pool / "qmoe_fc1_int3_b64_scales.data"

    pkg.ensure_qmoe_banks(pool, [spec("mse")], hf_dir=ckpt, num_layers=2)
    mse_weights, mse_scales = weights.read_bytes(), scales.read_bytes()
    published = weights.stat().st_mtime_ns
    bank_meta = json.loads((pool / "qmoe_bank_metadata.json").read_text(encoding="utf-8"))["banks"][weights.name]
    assert bank_meta["mode"] == "mse" and bank_meta["bits"] == 3 and bank_meta["checkpoint_fingerprint"]

    # The same request reuses the published bank; the pool needs no checkpoint for that.
    pkg.ensure_qmoe_banks(pool, [spec("mse")], hf_dir=ckpt, num_layers=2)
    pkg.ensure_qmoe_banks(pool, [spec("mse")], hf_dir=None, num_layers=2)
    assert weights.stat().st_mtime_ns == published

    # Same file names and byte sizes but another quantization mode is a different bank. If rebuilding it fails, the
    # previously published bank must survive intact and no temporary files may remain for a retry to trust.
    def fail_to_read(self, name, device="cpu"):
        raise RuntimeError("simulated checkpoint read failure")

    with monkeypatch.context() as patch:
        patch.setattr(pkg.HFCheckpointReader, "read_bf16_tensor", fail_to_read)
        with pytest.raises(RuntimeError, match="simulated checkpoint read failure"):
            pkg.ensure_qmoe_banks(pool, [spec("sym_rtn")], hf_dir=ckpt, num_layers=2)
    assert (weights.read_bytes(), scales.read_bytes()) == (mse_weights, mse_scales)
    assert [p.name for p in pool.iterdir() if ".tmp." in p.name] == []

    pkg.ensure_qmoe_banks(pool, [spec("sym_rtn")], hf_dir=ckpt, num_layers=2)
    assert weights.read_bytes() != mse_weights
    rebuilt_meta = json.loads((pool / "qmoe_bank_metadata.json").read_text(encoding="utf-8"))["banks"][weights.name]
    assert rebuilt_meta["mode"] == "sym_rtn"


def test_checkpoint_fingerprint_covers_every_byte_and_rejects_stale_banks(builders, tmp_path, monkeypatch):
    pkg = builders.pkg
    # Cache the digest of the freshly written shard too, so the edit below must also invalidate a cached digest.
    monkeypatch.setattr(importlib.import_module("package_io"), "RACY_CHANGE_WINDOW_NS", -1)
    spec = (
        "fc1",
        3,
        64,
        "mse",
        False,
        "qmoe_fc1_int3_b64.data",
        "qmoe_fc1_int3_b64_scales.data",
        [2, 128, 64 * 3 // 8],
        [2, 128, 1],
    )
    # The experts sit 8 MiB into an ~80 MiB shard, away from its first, middle and last 4 MiB.
    ckpt = _make_mini_qmoe_hf_checkpoint(tmp_path / "ckpt", num_layers=2, padding_bytes=(8 << 20, 72 << 20))
    pool = tmp_path / "pool"
    weights = pool / spec[5]
    pkg.ensure_qmoe_banks(pool, [spec], hf_dir=ckpt, num_layers=2)
    stale_bank = weights.read_bytes()
    bank_meta = json.loads((pool / "qmoe_bank_metadata.json").read_text(encoding="utf-8"))["banks"][weights.name]
    assert bank_meta["checkpoint_fingerprint"] == pkg.checkpoint_fingerprint(ckpt)

    shard = ckpt / "model.safetensors"
    with open(shard, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    start, end = header["model.language_model.layers.0.mlp.experts.gate_up_proj"]["data_offsets"]
    begin, window = 8 + header_len + start, 4 << 20
    assert window <= begin and begin + (end - start) <= shard.stat().st_size // 2
    with open(shard, "r+b") as f:
        f.seek(begin)
        negated = np.frombuffer(f.read(end - start), dtype=np.uint16) ^ np.uint16(0x8000)
        f.seek(begin)
        f.write(negated.tobytes())

    assert pkg.checkpoint_fingerprint(ckpt) != bank_meta["checkpoint_fingerprint"]
    pkg.ensure_qmoe_banks(pool, [spec], hf_dir=ckpt, num_layers=2)
    fresh_pool = tmp_path / "fresh_pool"
    pkg.ensure_qmoe_banks(fresh_pool, [spec], hf_dir=ckpt, num_layers=2)
    assert weights.read_bytes() != stale_bank
    assert weights.read_bytes() == (fresh_pool / spec[5]).read_bytes()


def test_precision_conversion_updates_bank_metadata_and_supports_fp32_position_reuse(builders, tmp_path):
    po = import_precision_options()
    gb = builders.gb
    rope_mod = builders.rope
    pkg_dir = tmp_path / "pkg_fp32_pos"
    pkg_dir.mkdir(parents=True)

    rows = 16
    rope_cfg = rope_mod.DEFAULT_ROPE
    positions = rope_mod.position_config(rope_cfg, rows, rows, None)
    tensor_locations: dict[str, dict] = {}
    gb.build_reused_position_tables(pkg_dir, tensor_locations, rope_cfg, rows, rows)

    norm_raw = torch.ones(128, dtype=torch.bfloat16).view(torch.uint16).numpy().tobytes()
    norm_t = gb.make_bf16_tensor(
        "v_model.layers.3.self_attn.indexer.q_layernorm.Add_494.folded.bf16",
        np.frombuffer(norm_raw, dtype=np.uint16),
        [128],
    )
    embedded_weights = {norm_t.name: norm_t}
    gb.save_bank_metadata_to_dir(pkg_dir, tensor_locations, embedded_weights, positions)

    inits = [norm_t]
    for name in (*rope_mod.TARGET_NAMES,):
        meta = tensor_locations[name]
        inits.append(
            gb.make_ext_init(name, meta["location"], meta["offset"], meta["length"], meta["dtype"], meta["dims"])
        )
    nodes = [helper.make_node("Identity", [t.name], [f"{t.name}_out"], name=f"id_{idx}") for idx, t in enumerate(inits)]
    model = helper.make_model(helper.make_graph(nodes, "g", [], [], initializer=inits))
    onnx.save(model, str(pkg_dir / "model.onnx"))

    dflash_inits = [
        gb.make_ext_init(
            name,
            tensor_locations[name]["location"],
            tensor_locations[name]["offset"],
            tensor_locations[name]["length"],
            tensor_locations[name]["dtype"],
            tensor_locations[name]["dims"],
        )
        for name in rope_mod.DRAFT_NAMES
    ]
    dflash_nodes = [
        helper.make_node("Identity", [t.name], [f"{t.name}_out"], name=f"df_{idx}")
        for idx, t in enumerate(dflash_inits)
    ]
    dflash = helper.make_model(helper.make_graph(dflash_nodes, "df", [], [], initializer=dflash_inits))
    onnx.save(dflash, str(pkg_dir / "dflash2.onnx"))

    # Create a tiny checkpoint with the 12 QSA layers' indexer projections (layers 3, 7, ..., 47)
    hf_dir = tmp_path / "hf_idx"
    hf_dir.mkdir()
    idx_tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for lid in range(3, 48, 4):
        qk = torch.zeros((640, 2560), dtype=torch.bfloat16)
        ln = torch.zeros((128,), dtype=torch.bfloat16)
        idx_tensors[f"model.language_model.layers.{lid}.self_attn.indexer.index_qk_proj.weight"] = (
            "BF16",
            [640, 2560],
            qk.view(torch.uint16).numpy().tobytes(),
        )
        for kind in ("q", "k"):
            idx_tensors[f"model.language_model.layers.{lid}.self_attn.indexer.{kind}_layernorm.weight"] = (
                "BF16",
                [128],
                ln.view(torch.uint16).numpy().tobytes(),
            )
    _write_safetensors_file(hf_dir / "model.safetensors", idx_tensors)

    pcfg = po.PrecisionConfig(indexer_dtype="fp32", rotary_cache_dtype="fp32")
    po.apply_precision_config_to_package(pkg_dir, pcfg, hf_dir=hf_dir)

    reloaded_locs, reloaded_emb = gb.load_bank_metadata_from_dir(pkg_dir)
    for name in rope_mod.TARGET_NAMES:
        assert reloaded_locs[name]["location"] == "attn_consts.data"
        assert reloaded_locs[name]["dtype"] == TensorProto.FLOAT
        assert reloaded_locs[name]["dims"][0] == rows
    assert reloaded_emb[norm_t.name].data_type == TensorProto.FLOAT

    # validate_reused_positions accepts the FP32 target RoPE & indexer tables in attn_consts.data
    validated = rope_mod.validate_reused_positions(pkg_dir, reloaded_locs)
    assert validated["target_rows"] == rows


def test_explicit_device_and_host_expert_budget_overrides_without_offload_mode(builders, tmp_path):
    pkg = builders.pkg
    base_pkg, pool_dir = make_synthetic_rc5_base_package(tmp_path / "budget", num_layers=4, num_experts=8)

    # Case 1: base package already has offload enabled; overriding budgets without --offload-mode updates them
    gcfg_path = base_pkg / "genai_config.json"
    gcfg = json.loads(gcfg_path.read_text(encoding="utf-8"))
    gcfg["model"]["decoder"]["session_options"]["session.moe.offload.mode"] = "required"
    gcfg["model"]["decoder"]["session_options"]["session.moe.offload.device_expert_bytes"] = str(4 * 1024**3)
    gcfg["model"]["decoder"]["session_options"]["session.moe.offload.host_expert_bytes"] = str(16 * 1024**3)
    gcfg_path.write_text(json.dumps(gcfg, indent=2) + "\n", encoding="utf-8")

    out_with_offload = tmp_path / "out_with_offload"
    pkg.build_rc5_package(
        out_with_offload,
        base_pkg=base_pkg,
        pool_dir=pool_dir,
        x_int4=4,
        y_int3=0,
        z_int2=0,
        device_expert_bytes=str(8 * 1024**3),
        host_expert_bytes=str(24 * 1024**3),
    )
    out_cfg = json.loads((out_with_offload / "genai_config.json").read_text(encoding="utf-8"))
    sopts = out_cfg["model"]["decoder"]["session_options"]
    assert sopts["session.moe.offload.mode"] == "required"
    assert sopts["session.moe.offload.device_expert_bytes"] == str(8 * 1024**3)
    assert sopts["session.moe.offload.host_expert_bytes"] == str(24 * 1024**3)

    # Case 2: base package has offload disabled; budget override updates the budget key without enabling offload.mode
    del gcfg["model"]["decoder"]["session_options"]["session.moe.offload.mode"]
    gcfg_path.write_text(json.dumps(gcfg, indent=2) + "\n", encoding="utf-8")
    out_no_offload = tmp_path / "out_no_offload"
    pkg.build_rc5_package(
        out_no_offload,
        base_pkg=base_pkg,
        pool_dir=pool_dir,
        x_int4=4,
        y_int3=0,
        z_int2=0,
        device_expert_bytes=str(6 * 1024**3),
    )
    out_cfg2 = json.loads((out_no_offload / "genai_config.json").read_text(encoding="utf-8"))
    sopts2 = out_cfg2["model"]["decoder"]["session_options"]
    assert "session.moe.offload.mode" not in sopts2
    assert sopts2["session.moe.offload.device_expert_bytes"] == str(6 * 1024**3)
