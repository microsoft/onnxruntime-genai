# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests for the unified Qwen3.8 RC5 package builder (`tools/qwen38_graph/build_package.py`) and helpers.

Covers:
1. KL-similarity `(x INT4 + y INT3 + z INT2)` QMoE layer sorting, 2D dynamic programming, and predefined RC5 presets.
2. SM80 INT4/INT2 prepack/unpack roundtrips, raw INT3 pack/unpack, MSE quantization, PLE INT8 shards, and `ContribWeightWriter`.
3. Path safety (`reject_output_overlap`) across all input flags before any file deletion or write.
4. `model.qwen38_graph` config migration (`migrate_qwen38_graph_config`), INT3 scratch session options, and `runtime_profiles`.
5. RC5 graph surgery (`SpeculativePrefixCommit` 7-in/3-out, `RouterGateUpProj` fusion, `QMoE` `(x, y, z)` rewriting, context tiers).
6. Unified GPU + QMoE offload readiness (`_qwen38_qmoe_offload_contract` simulation for INT4, INT3, INT2, NVFP4, and STRP v1 plans).
"""

from __future__ import annotations

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
        yield types.SimpleNamespace(config=config, paths=paths, gb=gb, pkg=pkg)
    finally:
        for name in ("package_config", "package_paths", "graph_builder", "build_package"):
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

    # Write STRP v1 expert-profile.bin
    pairs = [(lay, exp, 100 - lay * 10 - exp, 0) for exp in range(num_experts) for lay in range(num_layers)]
    header = struct.pack("<IIIIII", STRP_MAGIC, 1, num_layers, num_experts, num_layers * 2, len(pairs))
    body = b"".join(struct.pack("<HHHH", *p) for p in pairs)
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
    import argparse

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


def test_consolidate_package_external_data_merges_and_page_aligns(tmp_path):
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


def test_rc6_fp32_gdn_kl_calibration_and_presets(builders):
    pkg = builders.pkg
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
    assert spec_top16["kl_mode"] == "rank_fp32"

    spec_all4 = pkg.resolve_preset_spec("rc6_all_int4")
    assert spec_all4["pkg_name"] == "qwen_flash_next_rc6_all_int4_fp32"
    assert (spec_all4["x"], spec_all4["y"], spec_all4["z"]) == (48, 0, 0)




