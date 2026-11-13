# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper

SCRIPT = Path(__file__).resolve().parents[3] / "tools/python/qwen38_flash_next/build_package_gguf.py"
BUILDER_TEST_DIR = Path(__file__).resolve().parent
if str(BUILDER_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(BUILDER_TEST_DIR))

from qwen38_tools_test_utils import prefill_chunk_problem, unknown_qwen38_graph_keys  # noqa: E402


@pytest.fixture
def package_builder():
    spec = importlib.util.spec_from_file_location("build_package_gguf", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def precision_options_mod(package_builder):
    _ = package_builder
    spec = importlib.util.spec_from_file_location("precision_options", SCRIPT.parent / "precision_options.py")
    assert spec is not None and spec.loader is not None
    po = importlib.util.module_from_spec(spec)
    sys.modules["precision_options"] = po
    spec.loader.exec_module(po)
    return po


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
    assert profiles["gpu-32k"]["overlay"]["search"]["chunk_size"] == 4096
    assert profiles["gpu-64k"]["overlay"]["engine"]["dynamic_batching"]["num_blocks"] == 270
    assert profiles["gpu-128k"]["overlay"]["engine"]["dynamic_batching"]["num_blocks"] == 525
    assert profiles["gpu-256k"]["overlay"]["engine"]["dynamic_batching"]["num_blocks"] == 1037
    assert (
        profiles["gpu-32k"]["eligibility"]["minimum_total_device_memory_bytes"],
        profiles["gpu-64k"]["eligibility"]["minimum_total_device_memory_bytes"],
        profiles["gpu-128k"]["eligibility"]["minimum_total_device_memory_bytes"],
        profiles["gpu-256k"]["eligibility"]["minimum_total_device_memory_bytes"],
    ) == expected_mins
    assert cfg["model"]["qwen38_graph"]["plane_library"] == ""
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


def _snapshot_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): "<dir>" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
    }


def _pack_gguf_kv(key: str, vtype: int, val: object) -> bytes:
    kb = key.encode("utf-8")
    out = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vtype)
    if vtype == 4:
        out += struct.pack("<I", int(val))  # type: ignore[arg-type]
    elif vtype == 8:
        sb = str(val).encode("utf-8")
        out += struct.pack("<Q", len(sb)) + sb
    elif vtype == 9:
        atype, items = val  # type: ignore[misc]
        out += struct.pack("<IQ", atype, len(items))
        if atype == 8:
            parts = []
            for s in items:
                sb = s if isinstance(s, bytes) else str(s).encode("utf-8")
                parts.append(struct.pack("<Q", len(sb)) + sb)
            out += b"".join(parts)
        else:
            raise ValueError(f"Unsupported test array type {atype}")
    else:
        raise ValueError(f"Unsupported test kv type {vtype}")
    return out


def _write_gguf_file(
    path: Path,
    tensors: list[tuple[str, list[int], int, bytes]],
    *,
    extra_kvs: list[tuple[str, int, object]] | None = None,
    alignment: int = 32,
    truncate_payload_to: int | None = None,
    leading_pad: int = 0,
) -> int:
    """Write a GGUF file and return its data section offset.

    `leading_pad` zero bytes (a multiple of `alignment`) precede the first tensor, so it starts at a nonzero relative offset.
    """
    assert leading_pad % alignment == 0
    kvs: list[tuple[str, int, object]] = [("general.alignment", 4, alignment)]
    if extra_kvs:
        kvs.extend(extra_kvs)
    parts = [struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(kvs))]
    for k, vt, v in kvs:
        parts.append(_pack_gguf_kv(k, vt, v))

    rel_off = leading_pad
    payloads: list[bytes] = [b"\x00" * leading_pad]
    for name, dims, ggml_type, raw in tensors:
        nb = name.encode("utf-8")
        parts.append(struct.pack("<Q", len(nb)) + nb)
        parts.append(struct.pack("<I", len(dims)))
        for d in dims:
            parts.append(struct.pack("<Q", int(d)))
        parts.append(struct.pack("<IQ", int(ggml_type), rel_off))
        padded = raw + b"\x00" * ((alignment - (len(raw) % alignment)) % alignment)
        payloads.append(padded)
        rel_off += len(padded)

    header = b"".join(parts)
    data_offset = (len(header) + alignment - 1) // alignment * alignment
    pad = b"\x00" * (data_offset - len(header))
    body = b"".join(payloads)
    if truncate_payload_to is not None:
        body = body[:truncate_payload_to]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + pad + body)
    return data_offset


def _encode_q8_0(arr_f32: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(arr_f32, dtype=np.float32).reshape(-1, 32)
    amax = np.max(np.abs(flat), axis=1, keepdims=True)
    d_f16 = np.where(amax > 0, amax / 127.0, np.float32(1.0 / 127.0)).astype(np.float16)
    qs = np.clip(np.round(flat / d_f16.astype(np.float32)), -127, 127).astype(np.int8)
    out = np.empty((flat.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d_f16.view(np.uint8)
    out[:, 2:] = qs.view(np.uint8)
    return out.tobytes()


def _f32_to_bf16_bytes(arr_f32: np.ndarray) -> bytes:
    u32 = np.ascontiguousarray(arr_f32, dtype=np.float32).view(np.uint32)
    u16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
    return u16.tobytes()


def _write_safetensors(path: Path, tensors_f32: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    pos = 0
    for name, arr in sorted(tensors_f32.items()):
        raw = _f32_to_bf16_bytes(arr)
        header[name] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [pos, pos + len(raw)]}
        blobs.append(raw)
        pos += len(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.writelines(blobs)


def _make_synthetic_48_layer_fixture(root: Path, package_builder) -> dict[str, object]:
    """Create a complete miniature 48-layer RC5 base package, 2-shard GGUF variant, IQ4/Q8 engrams, and HF checkpoint."""
    rng = np.random.RandomState(42)
    hidden = 64
    vocab = 96
    streams_dim = 4 * hidden
    hc_rank = 32
    num_experts = 4
    inter = 32
    router_dim = num_experts
    num_v_heads = 6
    num_k_heads = 2
    head_v_dim = 32
    gdn_v_dim = num_v_heads * head_v_dim
    gdn_qk_dim = 64
    gdn_qkv_dim = gdn_qk_dim + gdn_v_dim
    qsa_q_dim = 128
    qsa_kv_dim = 64
    iqk_q_dim = 32
    iqk_k_dim = 32
    rotary_rows = 32
    ple_rows_per_head = 4
    ple_heads = 16
    ple_head_dim = 160
    ple_total_rows = ple_heads * ple_rows_per_head

    base_pkg = root / "base_pkg"
    base_pkg.mkdir(parents=True)

    ple_vocab = [ple_rows_per_head] * ple_heads
    ple_offsets = [i * ple_rows_per_head for i in range(ple_heads)]
    (base_pkg / "ple_table.txt").write_text(
        "\n".join(
            [
                "qwen38_ple_table 1",
                f"n_heads {ple_heads}",
                f"head_dim {ple_head_dim}",
                "heads_per_ngram 8",
                "eos 10",
                "scale -0.001",
                "mult 23703573157769 20109073645365 8052911324071 0",
                "vocab " + " ".join(str(x) for x in ple_vocab),
                "offset " + " ".join(str(x) for x in ple_offsets),
                *(
                    f"shard ple/s.bin {i * ple_rows_per_head * ple_head_dim} {ple_rows_per_head}"
                    for i in range(ple_heads)
                ),
            ]
        )
        + "\n"
    )
    for asset_name, text in (
        ("chat_template.jinja", "{{ messages }}"),
        ("generation_config.json", '{"eos_token_id": [10]}'),
        ("tokenizer.json", '{"version": "1.0"}'),
        ("tokenizer_config.json", '{"model_max_length": 262144}'),
    ):
        (base_pkg / asset_name).write_text(text)
    (base_pkg / "expert-profile.bin").write_bytes(b"STRP" + b"\x00" * 20)

    cos_arr = rng.randn(rotary_rows, 32).astype(np.float32) * 0.5
    sin_arr = rng.randn(rotary_rows, 32).astype(np.float32) * 0.5
    icos_arr = rng.randn(rotary_rows, 64).astype(np.float32) * 0.5
    isin_arr = rng.randn(rotary_rows, 64).astype(np.float32) * 0.5

    rope_writer = package_builder.SmallWeightBlobWriter(base_pkg, "rope.data")
    cos_info = rope_writer.write_bf16(cos_arr)
    sin_info = rope_writer.write_bf16(sin_arr)
    rope_writer.close()

    ac_writer = package_builder.SmallWeightBlobWriter(base_pkg, "attn_consts.data")
    icos_info = ac_writer.write_bf16(icos_arr)
    isin_info = ac_writer.write_bf16(isin_arr)
    ac_writer.close()

    qw_writer = package_builder.SmallWeightBlobWriter(base_pkg, "base_qweights.data")
    dummy_qw_info = qw_writer.write_raw(b"\x11" * 128, TensorProto.UINT8, [128])
    qw_writer.close()

    def ext_tp(name: str, info: tuple[str, int, int, int, list[int]]) -> TensorProto:
        return package_builder.make_ext_init(name, *info)

    inits: list[TensorProto] = [
        numpy_helper.from_array(np.array([33032], dtype=np.int32), name="contrib_const_dsa_total_seq"),
        numpy_helper.from_array(np.array([0, 0, gdn_v_dim], dtype=np.int64), name="const_gdn_flat"),
        numpy_helper.from_array(np.array([0, 0, 2, 32], dtype=np.int64), name="contrib_const_q_4d"),
        numpy_helper.from_array(np.array([16, 16], dtype=np.int64), name="const_split_qgate"),
        numpy_helper.from_array(np.array([0, 0, 2, 32], dtype=np.int64), name="contrib_const_k_4d"),
        numpy_helper.from_array(
            np.zeros((streams_dim,), dtype=np.uint16), name="v_model.layers.1.ple.norm_key.Add_200.folded"
        ),
        numpy_helper.from_array(
            np.zeros((streams_dim,), dtype=np.uint16), name="v_model.layers.1.ple.norm_query.Add_204.folded"
        ),
        numpy_helper.from_array(
            np.zeros((streams_dim,), dtype=np.uint16), name="v_model.layers.1.ple.norm_conv.Add_224.folded"
        ),
        numpy_helper.from_array(
            np.zeros((streams_dim,), dtype=np.uint16), name="v_model.hyper_connection_mixer.hc_norm.Add_5815.folded"
        ),
        ext_tp("model.rotary_emb.cos_cache", cos_info),
        ext_tp("model.rotary_emb.sin_cache", sin_info),
        ext_tp("capture.indexer_cos_cache.folded", icos_info),
        ext_tp("capture.indexer_sin_cache.folded", isin_info),
        ext_tp("merged_proj.2.weight_t.qweight", dummy_qw_info),
        ext_tp("lm_head.weight_t.qweight", dummy_qw_info),
        ext_tp("model.layers.1.ple.conv1d.weight", dummy_qw_info),
        ext_tp("model.hyper_connection_mixer.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16", dummy_qw_info),
        ext_tp("model.hyper_connection_mixer.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16", dummy_qw_info),
    ]
    for bf_idx in (5, 6, 7, 8):
        inits[bf_idx].data_type = TensorProto.BFLOAT16

    nodes: list[onnx.NodeProto] = []
    inputs: list[onnx.ValueInfoProto] = [
        helper.make_tensor_value_info("inputs_embeds", TensorProto.BFLOAT16, [1, "seq", hidden]),
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, [1, "seq"]),
        helper.make_tensor_value_info("past_position_ids", TensorProto.INT64, [1, 33032]),
        helper.make_tensor_value_info("cumulative_sequence_lengths", TensorProto.INT32, [2]),
        helper.make_tensor_value_info("committed_length", TensorProto.INT32, [1]),
        helper.make_tensor_value_info("capture.seqlens_k", TensorProto.INT32, [1]),
        helper.make_tensor_value_info("state_update_capture_count", TensorProto.INT32, [1]),
    ]
    outputs: list[onnx.ValueInfoProto] = [
        helper.make_tensor_value_info("logits", TensorProto.BFLOAT16, [1, "seq", vocab]),
        helper.make_tensor_value_info("present_position_ids", TensorProto.INT64, [1, 33032]),
    ]

    cur = "inputs_embeds"
    dsa_idx = 4
    for lid in range(48):
        if lid == 1:
            nodes.extend(
                [
                    helper.make_node(
                        "MatMulNBits",
                        [cur, "merged_proj.2.weight_t.qweight"],
                        ["ple.merged.out"],
                        name="skinny_gemm/ple/MatMulNBits",
                        domain="com.microsoft",
                    ),
                    helper.make_node(
                        "Split", ["ple.merged.out"], ["ple.k", "ple.v"], name="skinny_gemm/ple/Split", axis=-1
                    ),
                    helper.make_node(
                        "BranchwiseRMSNorm",
                        ["ple.k", "v_model.layers.1.ple.norm_key.Add_200.folded"],
                        ["ple.nk"],
                        name="ple/norm_key",
                        domain="com.microsoft",
                    ),
                    helper.make_node(
                        "BranchwiseRMSNorm",
                        ["ple.nk", "v_model.layers.1.ple.norm_query.Add_204.folded"],
                        ["ple.nq"],
                        name="ple/norm_query",
                        domain="com.microsoft",
                    ),
                    helper.make_node(
                        "BranchwiseRMSNorm",
                        ["ple.nq", "v_model.layers.1.ple.norm_conv.Add_224.folded"],
                        ["ple.nc"],
                        name="ple/norm_conv",
                        domain="com.microsoft",
                    ),
                    helper.make_node(
                        "CausalConvWithState",
                        ["ple.nc", "model.layers.1.ple.conv1d.weight", "ple.v"],
                        ["ple.conv_out"],
                        name="ple/conv",
                        domain="com.microsoft",
                    ),
                ]
            )
            cur = "ple.conv_out"

        hc_attn_norm = f"v_model.layers.{lid}.attn_hyper_connection.hc_norm.Add_0.folded"
        hc_mlp_norm = f"v_model.layers.{lid}.mlp_hyper_connection.hc_norm.Add_1.folded"
        for nm in (hc_attn_norm, hc_mlp_norm):
            tp = numpy_helper.from_array(np.zeros((streams_dim,), dtype=np.uint16), name=nm)
            tp.data_type = TensorProto.BFLOAT16
            inits.append(tp)

        for kind_onnx in ("attn_hyper_connection", "mlp_hyper_connection"):
            inits.append(
                ext_tp(
                    f"model.layers.{lid}.{kind_onnx}.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
                    dummy_qw_info,
                )
            )
            inits.append(
                ext_tp(
                    f"model.layers.{lid}.{kind_onnx}.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
                    dummy_qw_info,
                )
            )
            inits.append(ext_tp(f"model.layers.{lid}.{kind_onnx}.block_inject_weight.weight_t", dummy_qw_info))

        a_in = f"v_model.layers.{lid}.attn_pre"
        nodes.append(
            helper.make_node(
                "HyperConnectionLowRankMix",
                [
                    cur,
                    hc_attn_norm,
                    f"model.layers.{lid}.attn_hyper_connection.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
                    f"model.layers.{lid}.attn_hyper_connection.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
                    f"model.layers.{lid}.attn_hyper_connection.block_inject_weight.weight_t",
                ],
                [a_in],
                name=f"model/layers.{lid}/attn_hc",
                domain="com.microsoft",
            )
        )

        mp_in_name = f"merged_proj.in.{lid}.weight_t.qweight"
        inits.append(ext_tp(mp_in_name, dummy_qw_info))

        if lid % 4 != 3:
            inits.append(
                numpy_helper.from_array(
                    np.zeros((num_v_heads,), dtype=np.float32), name=f"model.layers.{lid}.linear_attn.A_log.f32"
                )
            )
            inits.append(
                numpy_helper.from_array(
                    np.zeros((num_v_heads,), dtype=np.float32), name=f"model.layers.{lid}.linear_attn.dt_bias.f32"
                )
            )
            inits.append(ext_tp(f"model.layers.{lid}.linear_attn.conv1d.weight", dummy_qw_info))
            inits.append(ext_tp(f"model.layers.{lid}.linear_attn.norm.weight", dummy_qw_info))
            inits.append(ext_tp(f"model.layers.{lid}.linear_attn.out_proj.weight_t.qweight", dummy_qw_info))

            qkv_o = f"l{lid}.gdn.qkv"
            z_o = f"l{lid}.gdn.z"
            b_o = f"l{lid}.gdn.b"
            a_o = f"l{lid}.gdn.a"
            gdn_raw = f"l{lid}.gdn.out"
            gdn_flat = f"l{lid}.gdn.flat"
            attn_out = f"l{lid}.gdn.proj"
            nodes.extend(
                [
                    helper.make_node(
                        "MatMulNBits",
                        [a_in, mp_in_name],
                        [f"l{lid}.mp_out"],
                        name=f"skinny_gemm/l{lid}/MatMulNBits",
                        domain="com.microsoft",
                    ),
                    helper.make_node(
                        "Split",
                        [f"l{lid}.mp_out"],
                        [qkv_o, z_o, b_o, a_o],
                        name=f"skinny_gemm/l{lid}/MatMulNBits/SplitGdn",
                        axis=-1,
                    ),
                    helper.make_node(
                        "CausalConvWithState",
                        [qkv_o, f"model.layers.{lid}.linear_attn.conv1d.weight"],
                        [f"l{lid}.conv"],
                        name=f"l{lid}/conv",
                        domain="com.microsoft",
                    ),
                    helper.make_node(
                        "GatedDeltaNet",
                        [
                            f"l{lid}.conv",
                            a_o,
                            b_o,
                            f"model.layers.{lid}.linear_attn.A_log.f32",
                            f"model.layers.{lid}.linear_attn.dt_bias.f32",
                        ],
                        [gdn_raw],
                        name=f"l{lid}/GatedDeltaNet",
                        domain="com.microsoft",
                    ),
                    helper.make_node("Reshape", [gdn_raw, "const_gdn_flat"], [gdn_flat], name=f"l{lid}/ReshapeGdn"),
                    helper.make_node(
                        "GatedRMSNorm",
                        [gdn_flat, f"model.layers.{lid}.linear_attn.norm.weight", z_o],
                        [f"l{lid}.gnorm"],
                        name=f"l{lid}/GatedRMSNorm",
                        domain="com.microsoft",
                    ),
                    helper.make_node(
                        "MatMulNBits",
                        [f"l{lid}.gnorm", f"model.layers.{lid}.linear_attn.out_proj.weight_t.qweight"],
                        [attn_out],
                        name=f"skinny_gemm/l{lid}_out/MatMulNBits",
                        domain="com.microsoft",
                    ),
                ]
            )
        else:
            qn = f"v_model.layers.{lid}.self_attn.q_norm.Add_0.folded"
            kn = f"v_model.layers.{lid}.self_attn.k_norm.Add_1.folded"
            iqn = f"v_model.layers.{lid}.self_attn.indexer.q_layernorm.Add_2.folded.bf16"
            ikn = f"v_model.layers.{lid}.self_attn.indexer.k_layernorm.Add_3.folded.bf16"
            inits.append(numpy_helper.from_array(np.ones((32,), dtype=np.float32), name=qn))
            inits.append(numpy_helper.from_array(np.ones((32,), dtype=np.float32), name=kn))
            for nm, dlen in ((iqn, iqk_q_dim), (ikn, iqk_k_dim)):
                tp = numpy_helper.from_array(np.zeros((dlen,), dtype=np.uint16), name=nm)
                tp.data_type = TensorProto.BFLOAT16
                inits.append(tp)
            inits.append(
                numpy_helper.from_array(
                    np.ones((qsa_kv_dim,), dtype=np.float32) * 0.01, name=f"node_GuardedDSA_{dsa_idx}_k_scale"
                )
            )
            inits.append(
                numpy_helper.from_array(
                    np.ones((qsa_kv_dim,), dtype=np.float32) * 0.01, name=f"node_GuardedDSA_{dsa_idx}_v_scale"
                )
            )
            inits.append(ext_tp(f"model.layers.{lid}.self_attn.o_proj.weight_t.qweight", dummy_qw_info))

            inputs.extend(
                [
                    helper.make_tensor_value_info(f"past_key_values.{lid}.key", TensorProto.INT8, [1, 2, 33032, 32]),
                    helper.make_tensor_value_info(f"past_key_values.{lid}.value", TensorProto.INT8, [1, 2, 33032, 32]),
                    helper.make_tensor_value_info(
                        f"past_key_values.{lid}.index_key", TensorProto.BFLOAT16, [1, 8259, 32]
                    ),
                    helper.make_tensor_value_info(
                        f"past_key_values.{lid}.index_buffer", TensorProto.BFLOAT16, [1, 4, 32]
                    ),
                    helper.make_tensor_value_info(f"past_key_values.{lid}.index_lengths", TensorProto.INT32, [1]),
                ]
            )
            outputs.extend(
                [
                    helper.make_tensor_value_info(f"present.{lid}.key", TensorProto.INT8, [1, 2, 33032, 32]),
                    helper.make_tensor_value_info(f"present.{lid}.value", TensorProto.INT8, [1, 2, 33032, 32]),
                    helper.make_tensor_value_info(f"present.{lid}.index_key", TensorProto.BFLOAT16, [1, 8259, 32]),
                    helper.make_tensor_value_info(f"present.{lid}.index_buffer", TensorProto.BFLOAT16, [1, 4, 32]),
                    helper.make_tensor_value_info(
                        f"state_update.{lid}.indexer_value", TensorProto.BFLOAT16, [1, 4, 32]
                    ),
                ]
            )

            iqk_o = f"l{lid}.qsa.iqk"
            qg_o = f"l{lid}.qsa.q_gate"
            k_o = f"l{lid}.qsa.k"
            v_o = f"l{lid}.qsa.v"
            attn_out = f"l{lid}.qsa.proj"
            dsa_inputs = [
                f"l{lid}.q_rot",
                f"l{lid}.k_rot",
                v_o,
                f"past_key_values.{lid}.key",
                f"past_key_values.{lid}.value",
                "",
                "",
                f"l{lid}.idx_sel",
                f"l{lid}.idx_cnt",
                "capture.seqlens_k",
                "contrib_const_dsa_total_seq",
                "",
                "",
                "",
                "",
                "",
                "",
                "committed_length",
                f"node_GuardedDSA_{dsa_idx}_k_scale",
                f"node_GuardedDSA_{dsa_idx}_v_scale",
            ]
            nodes.extend(
                [
                    helper.make_node(
                        "MatMulNBits",
                        [a_in, mp_in_name],
                        [f"l{lid}.mp_qsa"],
                        name=f"skinny_gemm/l{lid}/MatMulNBits",
                        domain="com.microsoft",
                    ),
                    helper.make_node(
                        "Split",
                        [f"l{lid}.mp_qsa"],
                        [iqk_o, qg_o, k_o, v_o],
                        name=f"skinny_gemm/l{lid}/MatMulNBits/SplitQsa",
                        axis=-1,
                    ),
                    helper.make_node("Reshape", [qg_o, "contrib_const_q_4d"], [f"l{lid}.q4d"], name=f"l{lid}/ReshapeQ"),
                    helper.make_node(
                        "Split",
                        [f"l{lid}.q4d", "const_split_qgate"],
                        [f"l{lid}.q_only", f"l{lid}.gate_only"],
                        name=f"l{lid}/SplitQG",
                        axis=-1,
                    ),
                    helper.make_node(
                        "RMSNormRotary",
                        [
                            f"l{lid}.q_only",
                            qn,
                            "position_ids",
                            "model.rotary_emb.cos_cache",
                            "model.rotary_emb.sin_cache",
                        ],
                        [f"l{lid}.q_rot"],
                        name=f"l{lid}/RMSNormRotaryQ",
                        domain="com.microsoft",
                        use_fp32_arithmetic=1,
                    ),
                    helper.make_node("Reshape", [k_o, "contrib_const_k_4d"], [f"l{lid}.k4d"], name=f"l{lid}/ReshapeK"),
                    helper.make_node(
                        "RMSNormRotary",
                        [f"l{lid}.k4d", kn, "position_ids", "model.rotary_emb.cos_cache", "model.rotary_emb.sin_cache"],
                        [f"l{lid}.k_rot"],
                        name=f"l{lid}/RMSNormRotaryK",
                        domain="com.microsoft",
                        use_fp32_arithmetic=1,
                    ),
                    helper.make_node(
                        "PackedSparseAttentionIndexer",
                        [
                            iqk_o,
                            "",
                            iqn,
                            ikn,
                            "capture.indexer_cos_cache.folded",
                            "capture.indexer_sin_cache.folded",
                            "cumulative_sequence_lengths",
                            "committed_length",
                            "",
                            "",
                            "",
                            "",
                            f"past_key_values.{lid}.index_key",
                            f"past_key_values.{lid}.index_buffer",
                            "",
                            f"past_key_values.{lid}.index_lengths",
                            "state_update_capture_count",
                        ],
                        [
                            f"l{lid}.idx_sel",
                            f"l{lid}.idx_cnt",
                            f"present.{lid}.index_key",
                            f"present.{lid}.index_buffer",
                            "",
                            f"l{lid}.idx_len_out",
                            f"state_update.{lid}.indexer_value",
                        ],
                        name=f"l{lid}/PackedQSAIndexer",
                        domain="com.microsoft",
                        state_capacity=8259,
                    ),
                    helper.make_node(
                        "DynamicSparseAttention",
                        dsa_inputs,
                        [f"l{lid}.dsa_out", f"present.{lid}.key", f"present.{lid}.value"],
                        name=f"node_GuardedDSA_{dsa_idx}",
                        domain="com.microsoft",
                        kv_num_heads=2,
                        num_heads=4,
                        scale=0.0625,
                    ),
                    helper.make_node("Sigmoid", [f"l{lid}.gate_only"], [f"l{lid}.gsig"], name=f"l{lid}/Sigmoid"),
                    helper.make_node("Mul", [f"l{lid}.dsa_out", f"l{lid}.gsig"], [f"l{lid}.mul"], name=f"l{lid}/Mul"),
                    helper.make_node(
                        "MatMulNBits",
                        [f"l{lid}.mul", f"model.layers.{lid}.self_attn.o_proj.weight_t.qweight"],
                        [attn_out],
                        name=f"skinny_gemm/l{lid}_qsa_out/MatMulNBits",
                        domain="com.microsoft",
                    ),
                ]
            )
            dsa_idx += 1

        rgup_idx = lid * 2 + 1 if lid == 0 else lid * 2 + 2
        inits.append(ext_tp(f"merged_proj.{rgup_idx}.weight_t", dummy_qw_info))
        inits.append(ext_tp(f"model.layers.{lid}.mlp.shared_expert.down_proj.weight_t", dummy_qw_info))
        inits.append(ext_tp(f"model.layers.{lid}.mlp.shared_expert_gate.weight_t", dummy_qw_info))
        inits.append(ext_tp(f"model.layers.{lid}.mlp.fc1_experts_weights", dummy_qw_info))

        mlp_in = f"v_model.layers.{lid}.mlp_pre"
        cur_next = f"v_model.layers.{lid}.out"
        nodes.extend(
            [
                helper.make_node(
                    "HyperConnectionLowRankMix",
                    [
                        attn_out,
                        hc_mlp_norm,
                        f"model.layers.{lid}.mlp_hyper_connection.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
                        f"model.layers.{lid}.mlp_hyper_connection.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
                        f"model.layers.{lid}.mlp_hyper_connection.block_inject_weight.weight_t",
                    ],
                    [mlp_in],
                    name=f"model/layers.{lid}/mlp_hc",
                    domain="com.microsoft",
                ),
                helper.make_node(
                    "RouterGateUpProj",
                    [
                        mlp_in,
                        f"merged_proj.{rgup_idx}.weight_t",
                        f"model.layers.{lid}.mlp.shared_expert.down_proj.weight_t",
                        f"model.layers.{lid}.mlp.shared_expert_gate.weight_t",
                    ],
                    [f"l{lid}.router_logits"],
                    name=f"l{lid}/RouterGateUpProj",
                    domain="com.microsoft",
                ),
                helper.make_node(
                    "QMoE",
                    [mlp_in, f"l{lid}.router_logits", f"model.layers.{lid}.mlp.fc1_experts_weights"],
                    [cur_next],
                    name=f"model/layers.{lid}/mlp/QMoE",
                    domain="com.microsoft",
                    k=2,
                ),
            ]
        )
        cur = cur_next

    nodes.extend(
        [
            helper.make_node(
                "HyperConnectionLowRankMix",
                [
                    cur,
                    "v_model.hyper_connection_mixer.hc_norm.Add_5815.folded",
                    "model.hyper_connection_mixer.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
                    "model.hyper_connection_mixer.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
                ],
                ["final_h"],
                name="final_hc",
                domain="com.microsoft",
            ),
            helper.make_node(
                "MatMulNBits",
                ["final_h", "lm_head.weight_t.qweight"],
                ["logits"],
                name="lm_head/MatMulNBits",
                domain="com.microsoft",
            ),
            helper.make_node("Identity", ["past_position_ids"], ["present_position_ids"], name="id_pos"),
        ]
    )
    base_graph = helper.make_graph(nodes, "target_rc5", inputs, outputs, initializer=inits)
    base_onnx = helper.make_model(
        base_graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    )
    onnx.save(base_onnx, str(base_pkg / "model.onnx"))

    df_writer = package_builder.SmallWeightBlobWriter(base_pkg, "dflash2.onnx.data")
    df_proj_arr = (rng.randn(hidden, hidden) * 0.1).astype(np.float32)
    df_proj_info = df_writer.write_bf16(df_proj_arr)
    df_writer.close()

    df_inits = [
        ext_tp("model.embed_tokens.weight", dummy_qw_info),
        ext_tp("model.embed_tokens.weight.row_scales", dummy_qw_info),
        ext_tp("dflash2.proj.weight", df_proj_info),
        ext_tp("lm_head.weight_t.qweight", dummy_qw_info),
    ]
    df_nodes = [
        helper.make_node(
            "Gather", ["model.embed_tokens.weight", "input_ids"], ["e0"], name="/dflash2/embed_tokens/Gather_i8"
        ),
        helper.make_node(
            "Gather",
            ["model.embed_tokens.weight.row_scales", "input_ids"],
            ["s0"],
            name="/dflash2/embed_tokens/Gather_scale",
        ),
        helper.make_node("Cast", ["e0"], ["e1"], name="/dflash2/embed_tokens/Cast_f32", to=TensorProto.FLOAT),
        helper.make_node("Mul", ["e1", "s0"], ["e2"], name="/dflash2/embed_tokens/Mul_dequant"),
        helper.make_node(
            "Cast",
            ["e2"],
            ["/dflash2/embed_tokens/Cast/output_0"],
            name="/dflash2/embed_tokens/Cast",
            to=TensorProto.BFLOAT16,
        ),
        helper.make_node(
            "MatMul", ["/dflash2/embed_tokens/Cast/output_0", "dflash2.proj.weight"], ["df_h"], name="/dflash2/proj"
        ),
        helper.make_node(
            "MatMulNBits",
            ["df_h", "lm_head.weight_t.qweight"],
            ["draft_logits"],
            name="/lm_head/MatMul",
            domain="com.microsoft",
        ),
    ]
    df_graph = helper.make_graph(
        df_nodes,
        "dflash2",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["n"])],
        [helper.make_tensor_value_info("draft_logits", TensorProto.BFLOAT16, ["n", vocab])],
        initializer=df_inits,
    )
    onnx.save(
        helper.make_model(
            df_graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
        ),
        str(base_pkg / "dflash2.onnx"),
    )
    (base_pkg / "genai_config.json").write_text(
        json.dumps(package_builder.default_base_genai_config(), indent=2) + "\n"
    )

    gguf_root = root / "gguf_root"
    variant_dir = gguf_root / "UD-IQ1_S"
    engram_dir = gguf_root / "engram"
    hf_dir = root / "hf_ckpt"

    emb_f32 = (rng.randn(vocab, hidden) * 0.25).astype(np.float32)
    lm_f32 = (rng.randn(vocab, hidden) * 0.25).astype(np.float32)
    shard1_tensors: list[tuple[str, list[int], int, bytes]] = [
        ("token_embd.weight", [hidden, vocab], 8, _encode_q8_0(emb_f32)),
        ("output.weight", [hidden, vocab], 8, _encode_q8_0(lm_f32)),
        ("blk.1.ple_norm_key.weight", [streams_dim], 0, np.ones((streams_dim,), dtype=np.float32).tobytes()),
        ("blk.1.ple_norm_query.weight", [streams_dim], 0, np.ones((streams_dim,), dtype=np.float32).tobytes()),
        ("blk.1.ple_norm_conv.weight", [streams_dim], 0, np.ones((streams_dim,), dtype=np.float32).tobytes()),
        (
            "blk.1.ple_conv1d.weight",
            [4, streams_dim],
            0,
            (rng.randn(streams_dim, 4) * 0.05).astype(np.float32).tobytes(),
        ),
        (
            "blk.1.ple_key.weight",
            [hidden, streams_dim],
            8,
            _encode_q8_0(rng.randn(streams_dim, hidden).astype(np.float32)),
        ),
        ("blk.1.ple_value.weight", [hidden, hidden], 8, _encode_q8_0(rng.randn(hidden, hidden).astype(np.float32))),
        ("output_hc_norm.weight", [streams_dim], 30, _f32_to_bf16_bytes(np.ones((streams_dim,), dtype=np.float32))),
        (
            "output_hc_down.weight",
            [streams_dim, hc_rank],
            8,
            _encode_q8_0(rng.randn(hc_rank, streams_dim).astype(np.float32)),
        ),
        ("output_hc_up.weight", [256, streams_dim], 14, bytes([1]) * (streams_dim * 210)),
    ]
    shard2_tensors: list[tuple[str, list[int], int, bytes]] = []
    hf_tensors: dict[str, np.ndarray] = {
        "model.language_model.layers.1.ple.norm_key.weight": np.zeros((streams_dim,), dtype=np.float32),
        "model.language_model.layers.1.ple.norm_query.weight": np.zeros((streams_dim,), dtype=np.float32),
        "model.language_model.layers.1.ple.norm_conv.weight": np.zeros((streams_dim,), dtype=np.float32),
    }

    for lid in range(48):
        target_list = shard1_tensors if lid < 24 else shard2_tensors
        lpfx = f"model.language_model.layers.{lid}"
        target_list.extend(
            [
                (
                    f"blk.{lid}.hc_attn_norm.weight",
                    [streams_dim],
                    0,
                    np.ones((streams_dim,), dtype=np.float32).tobytes(),
                ),
                (
                    f"blk.{lid}.hc_ffn_norm.weight",
                    [streams_dim],
                    0,
                    np.ones((streams_dim,), dtype=np.float32).tobytes(),
                ),
                (
                    f"blk.{lid}.ffn_gate_exps.weight",
                    [hidden, inter, num_experts],
                    8,
                    _encode_q8_0(np.zeros((num_experts, inter, hidden), dtype=np.float32)),
                ),
                (
                    f"blk.{lid}.ffn_up_exps.weight",
                    [hidden, inter, num_experts],
                    8,
                    _encode_q8_0(np.zeros((num_experts, inter, hidden), dtype=np.float32)),
                ),
                (
                    f"blk.{lid}.ffn_down_exps.weight",
                    [inter, hidden, num_experts],
                    8,
                    _encode_q8_0(np.zeros((num_experts, hidden, inter), dtype=np.float32)),
                ),
            ]
        )
        for kind_onnx, kind_gguf in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
            wd = (rng.randn(hc_rank, streams_dim) * 0.05).astype(np.float32)
            wu = (rng.randn(streams_dim, hc_rank) * 0.05).astype(np.float32)
            wi = (rng.randn(4, streams_dim) * 0.05).astype(np.float32)
            target_list.extend(
                [
                    (f"blk.{lid}.{kind_gguf}_down.weight", [streams_dim, hc_rank], 8, _encode_q8_0(wd)),
                    (f"blk.{lid}.{kind_gguf}_up.weight", [hc_rank, streams_dim], 8, _encode_q8_0(wu)),
                    (f"blk.{lid}.{kind_gguf}_inject.weight", [streams_dim, 4], 30, _f32_to_bf16_bytes(wi)),
                ]
            )
            hf_tensors[f"{lpfx}.{kind_onnx}.input_mix_weight_down.weight"] = wd
            hf_tensors[f"{lpfx}.{kind_onnx}.input_mix_weight_up.weight"] = wu
            hf_tensors[f"{lpfx}.{kind_onnx}.block_inject_weight.weight"] = wi

        wr = (rng.randn(router_dim, hidden) * 0.05).astype(np.float32)
        wsg = (rng.randn(inter, hidden) * 0.05).astype(np.float32)
        wsu = (rng.randn(inter, hidden) * 0.05).astype(np.float32)
        wsd = (rng.randn(hidden, inter) * 0.05).astype(np.float32)
        wsgate = (rng.randn(1, hidden) * 0.05).astype(np.float32)
        target_list.extend(
            [
                (f"blk.{lid}.ffn_gate_inp.weight", [hidden, router_dim], 0, wr.tobytes()),
                (f"blk.{lid}.ffn_gate_shexp.weight", [hidden, inter], 8, _encode_q8_0(wsg)),
                (f"blk.{lid}.ffn_up_shexp.weight", [hidden, inter], 8, _encode_q8_0(wsu)),
                (f"blk.{lid}.ffn_down_shexp.weight", [inter, hidden], 8, _encode_q8_0(wsd)),
                (f"blk.{lid}.ffn_gate_inp_shexp.weight", [hidden], 0, wsgate.tobytes()),
            ]
        )
        hf_tensors[f"{lpfx}.mlp.gate.weight"] = wr
        hf_tensors[f"{lpfx}.mlp.shared_expert.gate_proj.weight"] = wsg
        hf_tensors[f"{lpfx}.mlp.shared_expert.up_proj.weight"] = wsu
        hf_tensors[f"{lpfx}.mlp.shared_expert.down_proj.weight"] = wsd
        hf_tensors[f"{lpfx}.mlp.shared_expert_gate.weight"] = wsgate

        if lid % 4 != 3:
            wa = (rng.randn(num_v_heads, hidden) * 0.05).astype(np.float32)
            wb = (rng.randn(num_v_heads, hidden) * 0.05).astype(np.float32)
            target_list.extend(
                [
                    (f"blk.{lid}.ssm_a", [num_v_heads], 0, (-np.ones((num_v_heads,), dtype=np.float32)).tobytes()),
                    (f"blk.{lid}.ssm_dt.bias", [num_v_heads], 0, np.zeros((num_v_heads,), dtype=np.float32).tobytes()),
                    (
                        f"blk.{lid}.ssm_conv1d.weight",
                        [4, gdn_qkv_dim],
                        0,
                        (rng.randn(gdn_qkv_dim, 4) * 0.05).astype(np.float32).tobytes(),
                    ),
                    (f"blk.{lid}.ssm_norm.weight", [head_v_dim], 0, np.ones((head_v_dim,), dtype=np.float32).tobytes()),
                    (f"blk.{lid}.ssm_alpha.weight", [hidden, num_v_heads], 8, _encode_q8_0(wa)),
                    (f"blk.{lid}.ssm_beta.weight", [hidden, num_v_heads], 8, _encode_q8_0(wb)),
                    (
                        f"blk.{lid}.attn_qkv.weight",
                        [hidden, gdn_qkv_dim],
                        8,
                        _encode_q8_0((rng.randn(gdn_qkv_dim, hidden) * 0.1).astype(np.float32)),
                    ),
                    (
                        f"blk.{lid}.attn_gate.weight",
                        [hidden, gdn_v_dim],
                        8,
                        _encode_q8_0((rng.randn(gdn_v_dim, hidden) * 0.1).astype(np.float32)),
                    ),
                    (
                        f"blk.{lid}.ssm_out.weight",
                        [gdn_v_dim, hidden],
                        8,
                        _encode_q8_0((rng.randn(hidden, gdn_v_dim) * 0.1).astype(np.float32)),
                    ),
                ]
            )
            hf_tensors[f"{lpfx}.linear_attn.in_proj_a.weight"] = wa
            hf_tensors[f"{lpfx}.linear_attn.in_proj_b.weight"] = wb
        else:
            wiq = (rng.randn(iqk_q_dim, hidden) * 0.05).astype(np.float32)
            wik = (rng.randn(iqk_k_dim, hidden) * 0.05).astype(np.float32)
            wq = (rng.randn(qsa_q_dim, hidden) * 0.05).astype(np.float32)
            wk = (rng.randn(qsa_kv_dim, hidden) * 0.05).astype(np.float32)
            wv = (rng.randn(qsa_kv_dim, hidden) * 0.1 + 0.02).astype(np.float32)
            wo = (rng.randn(hidden, gdn_v_dim) * 0.05).astype(np.float32)
            target_list.extend(
                [
                    (f"blk.{lid}.attn_q_norm.weight", [32], 0, np.ones((32,), dtype=np.float32).tobytes()),
                    (f"blk.{lid}.attn_k_norm.weight", [32], 0, np.ones((32,), dtype=np.float32).tobytes()),
                    (
                        f"blk.{lid}.indexer.q_norm.weight",
                        [iqk_q_dim],
                        0,
                        np.ones((iqk_q_dim,), dtype=np.float32).tobytes(),
                    ),
                    (
                        f"blk.{lid}.indexer.k_norm.weight",
                        [iqk_k_dim],
                        0,
                        np.ones((iqk_k_dim,), dtype=np.float32).tobytes(),
                    ),
                    (f"blk.{lid}.indexer.q_proj.weight", [hidden, iqk_q_dim], 8, _encode_q8_0(wiq)),
                    (f"blk.{lid}.indexer.k_proj.weight", [hidden, iqk_k_dim], 8, _encode_q8_0(wik)),
                    (f"blk.{lid}.attn_q.weight", [hidden, qsa_q_dim], 8, _encode_q8_0(wq)),
                    (f"blk.{lid}.attn_k.weight", [hidden, qsa_kv_dim], 8, _encode_q8_0(wk)),
                    (f"blk.{lid}.attn_v.weight", [hidden, qsa_kv_dim], 8, _encode_q8_0(wv)),
                    (f"blk.{lid}.attn_output.weight", [gdn_v_dim, hidden], 8, _encode_q8_0(wo)),
                ]
            )
            hf_tensors[f"{lpfx}.self_attn.indexer.index_qk_proj.weight"] = np.concatenate([wiq, wik], axis=0)
            hf_tensors[f"{lpfx}.self_attn.indexer.q_layernorm.weight"] = np.zeros((iqk_q_dim,), dtype=np.float32)
            hf_tensors[f"{lpfx}.self_attn.indexer.k_layernorm.weight"] = np.zeros((iqk_k_dim,), dtype=np.float32)
            hf_tensors[f"{lpfx}.self_attn.q_proj.weight"] = wq
            hf_tensors[f"{lpfx}.self_attn.k_proj.weight"] = wk
            hf_tensors[f"{lpfx}.self_attn.v_proj.weight"] = wv
            hf_tensors[f"{lpfx}.self_attn.o_proj.weight"] = wo

    _write_gguf_file(variant_dir / "model-00001-of-00002.gguf", shard1_tensors)
    _write_gguf_file(variant_dir / "model-00002-of-00002.gguf", shard2_tensors)

    engram_iq4 = engram_dir / "Qwen3.8-Flash-Next-engram-IQ4_NL.gguf_file"
    engram_q8 = engram_dir / "Qwen3.8-Flash-Next-engram-Q8_0.gguf_file"
    _write_gguf_file(
        engram_iq4,
        [
            (
                "per_layer_token_embd.weight",
                [ple_head_dim, ple_total_rows],
                20,
                b"\x33" * (ple_total_rows * (ple_head_dim // 32 * 18)),
            )
        ],
    )
    _write_gguf_file(
        engram_q8,
        [
            (
                "per_layer_token_embd.weight",
                [ple_head_dim, ple_total_rows],
                8,
                b"\x44" * (ple_total_rows * (ple_head_dim // 32 * 34)),
            )
        ],
    )
    _write_gguf_file(
        variant_dir / "engram-ignored.gguf",
        [
            (
                "per_layer_token_embd.weight",
                [ple_head_dim, ple_total_rows],
                20,
                b"\x33" * (ple_total_rows * (ple_head_dim // 32 * 18)),
            )
        ],
    )
    _write_safetensors(hf_dir / "model.safetensors", hf_tensors)

    return {
        "base_pkg": base_pkg,
        "gguf_root": gguf_root,
        "variant_dir": variant_dir,
        "engram_iq4": engram_iq4,
        "engram_q8": engram_q8,
        "hf_dir": hf_dir,
        "hidden": hidden,
        "vocab": vocab,
        "num_k_heads": num_k_heads,
        "head_v_dim": head_v_dim,
        "gdn_qk_dim": gdn_qk_dim,
        "gdn_v_dim": gdn_v_dim,
        "ple_total_rows": ple_total_rows,
        "emb_f32": emb_f32,
        "hf_tensors": hf_tensors,
    }


def _load_onnx_materializing_external_data(onnx_path: Path) -> onnx.ModelProto:
    """Load `onnx_path` and materialize external data (using ONNX's loader for `.data` blobs and direct reads for hard-linked `.gguf`)."""
    base_dir = onnx_path.parent
    proto = onnx.load(str(onnx_path), load_external_data=False)
    for init in proto.graph.initializer:
        if init.data_location != TensorProto.EXTERNAL:
            continue
        ext = {e.key: e.value for e in init.external_data}
        loc = ext["location"]
        if loc.endswith(".gguf"):
            off = int(ext.get("offset", 0))
            ln = int(ext["length"])
            with open(base_dir / loc, "rb") as f:
                f.seek(off)
                raw = f.read(ln)
            assert len(raw) == ln
            del init.external_data[:]
            init.data_location = TensorProto.DEFAULT
            init.raw_data = raw
        else:
            # Non-GGUF `.data` files must be output-owned (st_nlink == 1), verified by ONNX's external data loader
            external_data_helper.load_external_data_for_tensor(init, str(base_dir))
            del init.external_data[:]
    return proto


def _run_ort_subgraph_in_pkg(
    ort, pkg_dir: Path, graph: onnx.GraphProto, feeds: dict[str, np.ndarray]
) -> list[np.ndarray]:
    """Save `graph` inside `pkg_dir` so ORT C++ resolves external `.data` slices directly from `pkg_dir`."""
    sub_path = pkg_dir / "_ort_subgraph_verify.onnx"
    try:
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)]), str(sub_path))
        sess = ort.InferenceSession(str(sub_path), providers=["CPUExecutionProvider"])
        return sess.run(None, feeds)
    finally:
        if sub_path.exists():
            sub_path.unlink()


def _assert_valid_exported_package(out_dir: Path, *, expected_ple_format: str = "iq4_nl") -> dict[str, onnx.ModelProto]:
    """Verify ONNX external files, weights_pool.txt, ple_table.txt, and genai_config.json on an exported package."""
    onnx_names = ["model.onnx", "dflash2.onnx", "embedding.onnx"]
    if (out_dir / "model_int8kv.onnx").exists():
        onnx_names.append("model_int8kv.onnx")

    models: dict[str, onnx.ModelProto] = {}
    resolved_out = out_dir.resolve()
    for name in onnx_names:
        onnx_path = out_dir / name
        assert onnx_path.is_file(), f"Missing {onnx_path}"
        proto = onnx.load(str(onnx_path), load_external_data=False)
        for init in proto.graph.initializer:
            if init.data_location == TensorProto.EXTERNAL:
                ext = {e.key: e.value for e in init.external_data}
                loc = ext["location"]
                assert not Path(loc).is_absolute() and ".." not in Path(loc).parts, f"Unsafe location {loc}"
                blob = (out_dir / loc).resolve()
                assert blob.is_relative_to(resolved_out) and blob.is_file(), (
                    f"Missing external file {loc} for {name}:{init.name}"
                )
                off = int(ext.get("offset", 0))
                ln = int(ext["length"])
                assert off >= 0 and ln > 0 and off + ln <= blob.stat().st_size, (
                    f"Out-of-bounds external slice {name}:{init.name} [{off}, {off + ln}) > {blob.stat().st_size}"
                )
        loaded = _load_onnx_materializing_external_data(onnx_path)
        assert len(loaded.graph.initializer) == len(proto.graph.initializer)
        models[name] = proto

    pool_lines = (out_dir / "weights_pool.txt").read_text().splitlines()
    assert pool_lines[0] == "qwen38_weights_pool 1"
    pools: dict[int, dict[str, object]] = {}
    sessions: list[tuple[str, str, int]] = []
    for ln in pool_lines[1:]:
        parts = ln.split()
        if not parts:
            continue
        if parts[0] == "pool":
            pid = int(parts[1])
            assert pid == len(pools), f"Non-consecutive pool id {pid}"
            dtype = int(parts[2])
            off = int(parts[3])
            length = int(parts[4])
            rank = int(parts[5])
            dims = [int(x) for x in parts[6 : 6 + rank]]
            fname = parts[6 + rank]
            fblob = out_dir / fname
            assert fblob.is_file() and off >= 0 and off + length <= fblob.stat().st_size
            pools[pid] = {"dtype": dtype, "offset": off, "length": length, "dims": dims, "location": fname}
        elif parts[0] == "session":
            sessions.append((parts[1], parts[2], int(parts[3])))

    assert len(pools) == 6 and len(sessions) == 8
    session_models = {"target": models["model.onnx"], "dflash2": models["dflash2.onnx"]}
    for sess_name, init_name, pid in sessions:
        m_inits = {t.name: t for t in session_models[sess_name].graph.initializer}
        assert init_name in m_inits, f"Pool session {sess_name} references missing initializer {init_name}"
        tp = m_inits[init_name]
        ext = {e.key: e.value for e in tp.external_data}
        assert pools[pid] == {
            "dtype": int(tp.data_type),
            "offset": int(ext.get("offset", 0)),
            "length": int(ext["length"]),
            "dims": [int(d) for d in tp.dims],
            "location": ext["location"],
        }

    ple_lines = (out_dir / "ple_table.txt").read_text().splitlines()
    assert ple_lines[0] == "qwen38_ple_table 1"
    assert f"format {expected_ple_format}" in ple_lines
    assert not any(ln.startswith("scale ") for ln in ple_lines)
    row_bytes = (160 // 32) * (18 if expected_ple_format == "iq4_nl" else 34)
    shard_rows = 0
    for ln in ple_lines[1:]:
        if ln.startswith("shard "):
            _, sfile, soff_s, srows_s = ln.split()
            soff, srows = int(soff_s), int(srows_s)
            sp = out_dir / sfile
            assert sp.is_file() and soff >= 0 and srows > 0
            assert soff + srows * row_bytes <= sp.stat().st_size
            shard_rows += srows
    assert shard_rows > 0

    cfg = json.loads((out_dir / "genai_config.json").read_text())
    qwen_sec = cfg["model"]["qwen38_graph"]
    assert unknown_qwen38_graph_keys(qwen_sec) == []
    assert prefill_chunk_problem(qwen_sec) is None
    assert qwen_sec["plane_library"] == ""
    profiles = cfg["runtime_profiles"]
    assert [p["id"] for p in profiles] == ["gpu-32k", "gpu-64k", "gpu-128k", "gpu-256k"]
    assert profiles[0]["overlay"]["search"]["chunk_size"] == 4096
    for prev_p, next_p in pairwise(profiles):
        assert (
            prev_p["eligibility"]["maximum_total_device_memory_bytes"] + 1
            == next_p["eligibility"]["minimum_total_device_memory_bytes"]
        )
    return models


def _run_representative_cpu_inference(out_dir: Path, fixture_info: dict[str, object], package_builder) -> None:
    """Execute exported package subgraphs through onnxruntime CPUExecutionProvider using actual exported external blobs."""
    ort = pytest.importorskip("onnxruntime")

    emb_proto = onnx.load(str(out_dir / "embedding.onnx"), load_external_data=False)
    emb_init = emb_proto.graph.initializer[0]
    emb_ext = {e.key: e.value for e in emb_init.external_data}
    tinfo = {
        "name": "token_embd.weight",
        "dims": [int(fixture_info["hidden"]), int(fixture_info["vocab"])],
        "ggml_type": 8,
        "data_offset": int(emb_ext["offset"]),
        "n_bytes": int(emb_ext["length"]),
    }
    deq_emb = package_builder.dequant_gguf_tensor_f32(out_dir / emb_ext["location"], tinfo)
    bf16_u16 = package_builder.f32_to_bf16_u16(deq_emb)
    bf16_tp = TensorProto()
    bf16_tp.name = "emb_bf16"
    bf16_tp.data_type = TensorProto.BFLOAT16
    bf16_tp.dims.extend(bf16_u16.shape)
    bf16_tp.raw_data = bf16_u16.tobytes()

    emb_exec_graph = helper.make_graph(
        [
            helper.make_node("Gather", ["emb_bf16", "input_ids"], ["emb_b16"], axis=0),
            helper.make_node("Cast", ["emb_b16"], ["inputs_embeds_f32"], to=TensorProto.FLOAT),
        ],
        "emb_cpu_exec",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"])],
        [
            helper.make_tensor_value_info(
                "inputs_embeds_f32", TensorProto.FLOAT, ["num_tokens", int(fixture_info["hidden"])]
            )
        ],
        initializer=[bf16_tp],
    )
    emb_sess = ort.InferenceSession(
        helper.make_model(emb_exec_graph, opset_imports=[helper.make_opsetid("", 18)]).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    sample_ids = np.array([0, 3, 7, 15], dtype=np.int64)
    got_emb = emb_sess.run(None, {"input_ids": sample_ids})[0]
    np.testing.assert_allclose(got_emb, fixture_info["emb_f32"][sample_ids], atol=0.02)

    ext_model = onnx.load(str(out_dir / "model.onnx"), load_external_data=False)
    inits_map = {t.name: t for t in ext_model.graph.initializer}
    hidden = int(fixture_info["hidden"])
    qk_dim = int(fixture_info["gdn_qk_dim"])
    v_dim = int(fixture_info["gdn_v_dim"])
    num_v_heads = 3 * int(fixture_info["num_k_heads"])

    gdn_sub_nodes = [
        n
        for n in ext_model.graph.node
        if n.name.startswith("skinny_gemm/l0/MatMulNBits/qkv/")
        and n.op_type in ("Split", "Reshape", "Transpose", "Concat")
    ]
    alpha_nodes = [n for n in ext_model.graph.node if n.name.startswith("skinny_gemm/l0/MatMulNBits/alpha/")]
    assert len(gdn_sub_nodes) == 5 and alpha_nodes

    alpha_mm = next(
        n
        for n in ext_model.graph.node
        if n.name.startswith("skinny_gemm/l0/MatMulNBits/alpha/") and n.op_type == "MatMul"
    )
    alpha_w_name = alpha_mm.input[1]
    qkv_raw_name = gdn_sub_nodes[0].input[0]
    qkv_out_name = gdn_sub_nodes[-1].output[0]

    exec_nodes = [
        helper.make_node("Cast", ["qkv_raw_f32"], [qkv_raw_name], to=TensorProto.BFLOAT16),
        *gdn_sub_nodes,
        helper.make_node("Cast", [qkv_out_name], ["qkv_out_f32"], to=TensorProto.FLOAT),
        helper.make_node("Cast", [alpha_w_name], ["alpha_w_f32"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["a_f32", "alpha_w_f32"], ["alpha_out_f32"]),
    ]
    needed = {inp for n in exec_nodes for inp in n.input if inp in inits_map}
    sub_graph = helper.make_graph(
        exec_nodes,
        "gdn_perm_exec",
        [
            helper.make_tensor_value_info("a_f32", TensorProto.FLOAT, [1, 2, hidden]),
            helper.make_tensor_value_info("qkv_raw_f32", TensorProto.FLOAT, [1, 2, qk_dim + v_dim]),
        ],
        [
            helper.make_tensor_value_info("qkv_out_f32", TensorProto.FLOAT, [1, 2, qk_dim + v_dim]),
            helper.make_tensor_value_info("alpha_out_f32", TensorProto.FLOAT, [1, 2, num_v_heads]),
        ],
        initializer=[inits_map[k] for k in sorted(needed)],
    )
    a_val = np.ones((1, 2, hidden), dtype=np.float32) * 0.5
    qkv_val = np.arange(2 * (qk_dim + v_dim), dtype=np.float32).reshape(1, 2, qk_dim + v_dim) / 100.0
    qkv_res, alpha_res = _run_ort_subgraph_in_pkg(ort, out_dir, sub_graph, {"a_f32": a_val, "qkv_raw_f32": qkv_val})
    assert qkv_res.shape == (1, 2, qk_dim + v_dim) and np.all(np.isfinite(qkv_res))
    assert alpha_res.shape == (1, 2, num_v_heads) and np.all(np.isfinite(alpha_res))
    expected_v = (
        qkv_val[..., qk_dim:]
        .reshape(1, 2, 3, int(fixture_info["num_k_heads"]), int(fixture_info["head_v_dim"]))
        .transpose(0, 1, 3, 2, 4)
        .reshape(1, 2, v_dim)
    )
    np.testing.assert_allclose(qkv_res[..., qk_dim:], expected_v, rtol=1e-2, atol=1e-2)

    # Also execute dflash2.onnx's retained external weight directly from out_dir/dflash2.onnx.data
    df_ext = onnx.load(str(out_dir / "dflash2.onnx"), load_external_data=False)
    df_inits = {t.name: t for t in df_ext.graph.initializer}
    df_sub = helper.make_graph(
        [
            helper.make_node("Cast", ["dflash2.proj.weight"], ["df_w_f32"], to=TensorProto.FLOAT),
            helper.make_node("MatMul", ["x", "df_w_f32"], ["y"]),
        ],
        "dflash_proj_exec",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, hidden])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, hidden])],
        initializer=[df_inits["dflash2.proj.weight"]],
    )
    df_y = _run_ort_subgraph_in_pkg(ort, out_dir, df_sub, {"x": np.ones((2, hidden), dtype=np.float32)})[0]
    assert df_y.shape == (2, hidden) and np.all(np.isfinite(df_y))


def test_parse_gguf_header_supports_metadata_exceeding_16_mib(package_builder, tmp_path):
    """Regression test: GGUF headers larger than 16 MiB (e.g., 248,320 vocab strings) must parse cleanly."""
    token_entry = b"x" * 64
    num_tokens = 248_320  # 248,320 * (8 + 64) = 17.88 MB > 16 MiB
    gguf_path = tmp_path / "large_header.gguf"
    payload = _encode_q8_0(np.ones((2, 64), dtype=np.float32))
    expected_doff = _write_gguf_file(
        gguf_path,
        [("token_embd.weight", [64, 2], 8, payload)],
        extra_kvs=[("tokenizer.ggml.tokens", 9, (8, [token_entry] * num_tokens))],
        alignment=64,
    )
    assert gguf_path.stat().st_size > 17 * 1024 * 1024

    kvs, tensors, doff = package_builder.parse_gguf_header(gguf_path)
    assert kvs["tokenizer.ggml.tokens"] == f"arr[{num_tokens}]"
    assert doff == expected_doff
    assert len(tensors) == 1
    assert tensors[0]["name"] == "token_embd.weight"
    assert tensors[0]["data_offset"] == expected_doff
    assert tensors[0]["n_bytes"] == len(payload)


def _ple_shard_lines(ple_table_path: Path) -> list[tuple[str, int, int]]:
    shards = []
    for line in ple_table_path.read_text().splitlines():
        if line.startswith("shard "):
            _, name, off, rows = line.split()
            shards.append((name, int(off), int(rows)))
    return shards


@pytest.mark.parametrize(
    ("engram_kind", "ggml_type", "row_bytes", "fill"),
    [("iq4_nl", 20, 5 * 18, b"\x33"), ("q8_0", 8, 5 * 34, b"\x44")],
)
def test_ple_shards_start_at_engram_tensor_offset_not_data_section_start(
    package_builder, precision_options_mod, tmp_path, engram_kind, ggml_type, row_bytes, fill
):
    """Regression test: a valid engram tensor may start past the GGUF data section start; PLE shards must follow it."""
    fx = _make_synthetic_48_layer_fixture(tmp_path / "fx", package_builder)
    total_rows = 64
    payload = fill * (total_rows * row_bytes)
    leading_pad = 96
    engram = tmp_path / "engram_nonzero_rel_offset.gguf"
    data_section = _write_gguf_file(
        engram,
        [("per_layer_token_embd.weight", [160, total_rows], ggml_type, payload)],
        leading_pad=leading_pad,
    )
    tensor_start = data_section + leading_pad
    ple_spec = package_builder.read_ple_table_spec(fx["base_pkg"] / "ple_table.txt")

    dst_name, ple_format, ple_row_bytes, tensor_offset = package_builder.validate_engram_gguf(engram, ple_spec)
    assert (ple_format, ple_row_bytes) == (engram_kind, row_bytes)
    assert tensor_offset == tensor_start != data_section

    def check_shards(ple_table_path: Path, engram_path: Path) -> None:
        shards = _ple_shard_lines(ple_table_path)
        assert shards and sum(rows for _, _, rows in shards) == total_rows
        assert shards[0][1] == tensor_start
        raw = engram_path.read_bytes()
        row = 0
        for shard_name, shard_off, shard_rows in shards:
            assert shard_name == dst_name
            assert shard_off == tensor_start + row * row_bytes
            expected_rows = payload[row * row_bytes : (row + shard_rows) * row_bytes]
            assert raw[shard_off : shard_off + shard_rows * row_bytes] == expected_rows
            row += shard_rows

    out_dir = tmp_path / "built"
    package_builder.build_gguf_package("nonzero_rel_offset", fx["variant_dir"], engram, fx["base_pkg"], out_dir)
    check_shards(out_dir / "ple_table.txt", out_dir / dst_name)

    for validate_engram in (True, False):
        table_dir = tmp_path / f"table_validate_{validate_engram}"
        table_dir.mkdir()
        (table_dir / "ple_table.txt").write_bytes((fx["base_pkg"] / "ple_table.txt").read_bytes())
        written_name = precision_options_mod.write_gguf_ple_table(
            table_dir, engram_kind, engram, validate_engram=validate_engram
        )
        assert written_name == dst_name
        check_shards(table_dir / "ple_table.txt", table_dir / written_name)


def test_truncated_or_wrong_shape_gguf_rejected_before_replacing_output(package_builder, tmp_path):
    """Regression test: header-only/truncated engram or shard GGUFs and wrong-shape engrams fail before touching out_dir."""
    fx = _make_synthetic_48_layer_fixture(tmp_path / "fx", package_builder)
    out_dir = tmp_path / "existing_output"
    out_dir.mkdir()
    (out_dir / "keep_me.txt").write_text("do not delete")
    before = _snapshot_tree(out_dir)

    real_base = tmp_path / "real_ple_base"
    real_base.mkdir()
    for p in ("model.onnx", "dflash2.onnx", "rope.data", "attn_consts.data", "base_qweights.data", "dflash2.onnx.data"):
        (real_base / p).write_bytes((fx["base_pkg"] / p).read_bytes())
    hdr_only_engram = tmp_path / "header_only_engram.gguf"
    _write_gguf_file(
        hdr_only_engram,
        [("per_layer_token_embd.weight", [160, 320001446], 20, b"")],
        truncate_payload_to=0,
    )
    assert hdr_only_engram.stat().st_size <= 128
    with pytest.raises(ValueError, match="Truncated GGUF"):
        package_builder.build_gguf_package("test_trunc", fx["variant_dir"], hdr_only_engram, real_base, out_dir)
    assert _snapshot_tree(out_dir) == before

    wrong_shape_engram = tmp_path / "wrong_shape_engram.gguf"
    _write_gguf_file(
        wrong_shape_engram,
        [("per_layer_token_embd.weight", [160, 4], 20, b"\x11" * (4 * 90))],
    )
    with pytest.raises(ValueError, match="Invalid engram tensor shape"):
        package_builder.build_gguf_package("test_shape", fx["variant_dir"], wrong_shape_engram, fx["base_pkg"], out_dir)
    assert _snapshot_tree(out_dir) == before

    bad_variant = tmp_path / "bad_variant"
    bad_variant.mkdir()
    _write_gguf_file(
        bad_variant / "shard.gguf",
        [("token_embd.weight", [64, 96], 8, b"\x00" * 100)],
        truncate_payload_to=32,
    )
    with pytest.raises(ValueError, match="Truncated GGUF"):
        package_builder.build_gguf_package("test_bad_shard", bad_variant, fx["engram_iq4"], fx["base_pkg"], out_dir)
    assert _snapshot_tree(out_dir) == before


def test_overlapping_inputs_including_hf_dir_rejected_without_touching_files(package_builder, tmp_path):
    """Regression test: overlapping out_dir and hf_dir (or other inputs) must be rejected before rmtree."""
    fx = _make_synthetic_48_layer_fixture(tmp_path / "fx", package_builder)

    parent_out = tmp_path / "out_parent"
    nested_hf = parent_out / "nested_hf_ckpt"
    nested_hf.mkdir(parents=True)
    (nested_hf / "model.safetensors").write_bytes((fx["hf_dir"] / "model.safetensors").read_bytes())
    (parent_out / "sentinel.txt").write_text("keep")
    snap_before = _snapshot_tree(parent_out)

    with pytest.raises(SystemExit, match="--hf-dir"):
        package_builder.build_gguf_package(
            "overlap_direct",
            fx["variant_dir"],
            fx["engram_iq4"],
            fx["base_pkg"],
            parent_out,
            small_weights_source="hf_bf16",
            hf_dir=nested_hf,
        )
    assert _snapshot_tree(parent_out) == snap_before

    out_root = tmp_path / "cli_out_root"
    target_pkg_out = out_root / "qwen_flash_next_gguf_ud_iq1_s"
    stage2_hf = target_pkg_out / "hf_inside_out"
    stage2_hf.mkdir(parents=True)
    (stage2_hf / "model.safetensors").write_bytes((fx["hf_dir"] / "model.safetensors").read_bytes())
    snap_stage2 = _snapshot_tree(target_pkg_out)

    with pytest.raises(SystemExit, match="--hf-dir"):
        package_builder.main(
            [
                "--gguf-root",
                str(fx["gguf_root"]),
                "--base-pkg",
                str(fx["base_pkg"]),
                "--hf-dir",
                str(stage2_hf),
                "--out-root",
                str(out_root),
                "--variants",
                "UD-IQ1_S",
                "--gdn-gate-dtype",
                "fp32",
                "--small-weights-source",
                "gguf",
            ]
        )
    assert _snapshot_tree(target_pkg_out) == snap_stage2


def test_cli_expands_home_defaults_and_tilde_arguments(package_builder, monkeypatch, tmp_path):
    """Regression test: omitted CLI paths resolve under Path.home()/models and ~ arguments expand before validation."""
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    work_dir = tmp_path / "cwd"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    ap = package_builder.build_arg_parser()
    defaults = ap.parse_args([])
    assert defaults.gguf_root == fake_home / "models" / "qwen_flash_next_ggufs"
    assert defaults.base_pkg == fake_home / "models" / "qwen_flash_next_int4_int8_rc5_contrib_dflash2"
    assert defaults.hf_dir == fake_home / "models" / "Qwen3.8-Flash-Next"
    assert defaults.out_root == fake_home / "models"

    tilde_gguf = fake_home / "custom_gguf"
    (tilde_gguf / "UD-IQ1_S").mkdir(parents=True)
    engram = tilde_gguf / "engram" / "Qwen3.8-Flash-Next-engram-IQ4_NL.gguf_file"
    engram.parent.mkdir(parents=True)
    engram.touch()
    tilde_base = fake_home / "custom_base"

    recorded = []
    monkeypatch.setattr(
        package_builder,
        "build_gguf_package",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )
    assert (
        package_builder.main(
            [
                "--gguf-root",
                "~/custom_gguf",
                "--base-pkg",
                "~/custom_base",
                "--variants",
                "UD-IQ1_S",
            ]
        )
        == 0
    )
    assert not (work_dir / "~").exists()
    assert len(recorded) == 1
    pos_args, _ = recorded[0]
    assert pos_args == (
        "qwen_flash_next_gguf_ud_iq1_s",
        tilde_gguf / "UD-IQ1_S",
        engram,
        tilde_base,
        fake_home / "models" / "qwen_flash_next_gguf_ud_iq1_s",
    )


def test_non_default_gguf_root_resolves_iq4_and_q8_engrams_and_excludes_provider_libs(package_builder, tmp_path):
    """Regression test: IQ4_NL and Q8_0 engrams resolve under --gguf-root/engram, and provider binaries are excluded."""
    fx = _make_synthetic_48_layer_fixture(tmp_path / "fx", package_builder)
    out_root = tmp_path / "out"

    prov_dir = fx["base_pkg"] / "provider"
    prov_dir.mkdir()
    real_ort = prov_dir / "libonnxruntime.so.1.32.0"
    real_ort.write_bytes(b"ORT_BINARY" * 100)
    (prov_dir / "libonnxruntime.so.1").symlink_to("libonnxruntime.so.1.32.0")
    (prov_dir / "libonnxruntime.so").symlink_to("libonnxruntime.so.1")
    (prov_dir / "libonnxruntime_providers_cuda.so").write_bytes(b"CUDA_EP" * 100)
    (prov_dir / "libqwen38_plane.so").write_bytes(b"PLANE_LIB" * 100)

    assert (
        package_builder.main(
            [
                "--gguf-root",
                str(fx["gguf_root"]),
                "--base-pkg",
                str(fx["base_pkg"]),
                "--hf-dir",
                str(fx["hf_dir"]),
                "--out-root",
                str(out_root),
                "--variants",
                "UD-IQ1_S",
            ]
        )
        == 0
    )
    iq4_pkg = out_root / "qwen_flash_next_gguf_ud_iq1_s"
    assert not (iq4_pkg / "provider").exists()
    assert not list(iq4_pkg.rglob("*.so*"))
    assert not list(iq4_pkg.rglob("*.dll"))
    _assert_valid_exported_package(iq4_pkg, expected_ple_format="iq4_nl")
    _run_representative_cpu_inference(iq4_pkg, fx, package_builder)

    assert (
        package_builder.main(
            [
                "--gguf-root",
                str(fx["gguf_root"]),
                "--base-pkg",
                str(fx["base_pkg"]),
                "--hf-dir",
                str(fx["hf_dir"]),
                "--out-root",
                str(out_root),
                "--variants",
                "UD-IQ1_S",
                "--ple-engram",
                "q8_0",
                "--pkg-suffix",
                "_q8",
            ]
        )
        == 0
    )
    q8_pkg = out_root / "qwen_flash_next_gguf_ud_iq1_s_q8"
    _assert_valid_exported_package(q8_pkg, expected_ple_format="q8_0")
    assert (q8_pkg / "Qwen3.8-Flash-Next-engram-Q8_0.gguf").is_file()

    custom_engram = tmp_path / "explicit_engram_q8.gguf_file"
    custom_engram.write_bytes(fx["engram_q8"].read_bytes())
    assert (
        package_builder.main(
            [
                "--gguf-root",
                str(fx["gguf_root"]),
                "--base-pkg",
                str(fx["base_pkg"]),
                "--hf-dir",
                str(fx["hf_dir"]),
                "--engram-gguf",
                str(custom_engram),
                "--out-root",
                str(out_root),
                "--variants",
                "UD-IQ1_S",
                "--pkg-suffix",
                "_explicit_engram",
            ]
        )
        == 0
    )
    exp_pkg = out_root / "qwen_flash_next_gguf_ud_iq1_s_explicit_engram"
    _assert_valid_exported_package(exp_pkg, expected_ple_format="q8_0")


def test_cross_device_link_failure_falls_back_to_copy(package_builder, monkeypatch, tmp_path):
    """Regression test: EXDEV cross-filesystem hard-link failures must fall back to copying shards and engram."""
    fx = _make_synthetic_48_layer_fixture(tmp_path / "fx", package_builder)
    out_dir = tmp_path / "cross_mount_pkg"

    def fail_exdev(src, dst, *args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", fail_exdev)
    package_builder.build_gguf_package("cross_dev", fx["variant_dir"], fx["engram_iq4"], fx["base_pkg"], out_dir)
    _assert_valid_exported_package(out_dir, expected_ple_format="iq4_nl")
    for shard_name in ("model-00001-of-00002.gguf", "model-00002-of-00002.gguf"):
        assert (out_dir / shard_name).stat().st_ino != (fx["variant_dir"] / shard_name).stat().st_ino


def test_consolidated_base_package_exports_and_loads_cleanly(package_builder, precision_options_mod, tmp_path):
    """Regression test: base package consolidated with consolidate_package_external_data exports and loads cleanly."""
    fx = _make_synthetic_48_layer_fixture(tmp_path / "fx", package_builder)
    base_pkg = fx["base_pkg"]

    df_proto = onnx.load(str(base_pkg / "dflash2.onnx"), load_external_data=False)
    extra_w = _f32_to_bf16_bytes(np.eye(int(fx["hidden"]), dtype=np.float32))
    (base_pkg / "dflash_shared.data").write_bytes(extra_w)
    for idx, t in enumerate(df_proto.graph.initializer):
        if t.name == "dflash2.proj.weight":
            df_proto.graph.initializer[idx].CopyFrom(
                package_builder.make_ext_init(
                    "dflash2.proj.weight",
                    "dflash_shared.data",
                    0,
                    len(extra_w),
                    TensorProto.BFLOAT16,
                    [int(fx["hidden"]), int(fx["hidden"])],
                )
            )
    m_proto = onnx.load(str(base_pkg / "model.onnx"), load_external_data=False)
    m_proto.graph.initializer.append(
        package_builder.make_ext_init(
            "unused_consolidated_ref",
            "dflash_shared.data",
            0,
            len(extra_w),
            TensorProto.BFLOAT16,
            [int(fx["hidden"]), int(fx["hidden"])],
        )
    )
    onnx.save(m_proto, str(base_pkg / "model.onnx"))
    onnx.save(df_proto, str(base_pkg / "dflash2.onnx"))

    cons = precision_options_mod.consolidate_package_external_data(base_pkg)
    assert cons["consolidated"] is True
    assert not (base_pkg / "attn_consts.data").exists()
    assert not (base_pkg / "dflash_shared.data").exists()
    assert (base_pkg / "model.onnx.data").is_file()

    out_dir = tmp_path / "out_from_consolidated_base"
    package_builder.build_gguf_package("from_cons_base", fx["variant_dir"], fx["engram_iq4"], base_pkg, out_dir)
    assert not (out_dir / "model.onnx.data").exists()
    assert (out_dir / "attn_consts.data").is_file()
    assert (out_dir / "dflash2.onnx.data").is_file()
    _assert_valid_exported_package(out_dir, expected_ple_format="iq4_nl")
    _run_representative_cpu_inference(out_dir, fx, package_builder)


@pytest.mark.parametrize("qsa_mode", ["v_rc5_fp16", "kv_fp16", "fp16"])
def test_packed_qsa_precision_rewrites_and_cpu_execution(package_builder, precision_options_mod, tmp_path, qsa_mode):
    """Regression test: v_rc5_fp16, kv_fp16, and fp16 rewrite packed QSA projections and execute accurately on CPU ORT."""
    ort = pytest.importorskip("onnxruntime")
    fx = _make_synthetic_48_layer_fixture(tmp_path / f"fx_{qsa_mode}", package_builder)
    out_root = tmp_path / f"out_{qsa_mode}"

    argv = [
        "--gguf-root",
        str(fx["gguf_root"]),
        "--base-pkg",
        str(fx["base_pkg"]),
        "--hf-dir",
        str(fx["hf_dir"]),
        "--out-root",
        str(out_root),
        "--variants",
        "UD-IQ1_S",
        "--kv-cache-dtype",
        "fp16",
        "--indexer-dtype",
        "fp32",
        "--gdn-gate-dtype",
        "fp32",
        "--qsa-qkv-dtype",
        qsa_mode,
        "--emit-int8kv-variant",
        "--consolidate-data-files",
    ]
    assert package_builder.main(argv) == 0

    pkg_dir = out_root / "qwen_flash_next_gguf_ud_iq1_s"
    models = _assert_valid_exported_package(pkg_dir, expected_ple_format="iq4_nl")
    assert (pkg_dir / "model.onnx.data").is_file()
    assert not (pkg_dir / "gguf_small_weights.data").exists()
    assert not (pkg_dir / "attn_consts.data").exists()
    assert (pkg_dir / "model_int8kv.onnx").is_file()

    model = models["model.onnx"]
    inits = {t.name: t for t in model.graph.initializer}
    packed_weights = {n.input[1] for n in model.graph.node if n.op_type == "MatMulPacked"}

    for n in model.graph.node:
        if n.op_type == "RMSNormRotary":
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            assert attrs["output_fp16"] == 1
        elif n.op_type == "DynamicSparseAttention":
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            assert attrs["output_bf16"] == 1
            assert len(n.input) == 18

    m_int8 = models["model_int8kv.onnx"]
    for n in m_int8.graph.node:
        if n.op_type == "DynamicSparseAttention":
            assert len(n.input) == 20 and n.input[18] and n.input[19]
    for vi in m_int8.graph.input:
        if re.match(r"^past_key_values\.\d+\.(key|value)$", vi.name):
            assert vi.type.tensor_type.elem_type == TensorProto.INT8

    hidden = int(fx["hidden"])
    x_f32 = np.linspace(-0.5, 0.5, 2 * hidden, dtype=np.float32).reshape(1, 2, hidden)

    if qsa_mode == "v_rc5_fp16":
        assert "model.layers.3.self_attn.v.gguf_weight" not in packed_weights
        assert "model.layers.3.self_attn.v.gguf_weight" not in inits
        assert "model.layers.3.self_attn.q.gguf_weight" in packed_weights
        assert "model.layers.3.self_attn.k.gguf_weight" in packed_weights
        v_tp = inits["model.layers.3.self_attn.v.rc5_deq.fp16"]
        assert v_tp.data_type == TensorProto.FLOAT16

        v_nodes = [n for n in model.graph.node if n.name.startswith("skinny_gemm/l3/MatMulNBits/v_proj/MatMulPacked/")]
        assert [n.op_type for n in v_nodes] == ["Cast", "MatMul"]
        sub_g = helper.make_graph(
            [helper.make_node("Cast", ["x"], [v_nodes[0].input[0]], to=TensorProto.BFLOAT16), *v_nodes],
            "v_rc5_exec",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, hidden])],
            [helper.make_tensor_value_info(v_nodes[-1].output[0], TensorProto.FLOAT16, [1, 2, 64])],
            initializer=[v_tp],
        )
        v_out = _run_ort_subgraph_in_pkg(ort, pkg_dir, sub_g, {"x": x_f32})[0]
        ref_v = fx["hf_tensors"]["model.language_model.layers.3.self_attn.v_proj.weight"].T
        np.testing.assert_allclose(v_out.astype(np.float32), x_f32 @ ref_v, rtol=2e-2, atol=2e-2)

    elif qsa_mode == "kv_fp16":
        assert "model.layers.3.self_attn.k.gguf_weight" not in packed_weights
        assert "model.layers.3.self_attn.v.gguf_weight" not in packed_weights
        assert "model.layers.3.self_attn.q.gguf_weight" in packed_weights
        kv_tp = inits["model.layers.3.self_attn.kv.weight_t.fp16"]
        assert kv_tp.data_type == TensorProto.FLOAT16

        kv_nodes = [n for n in model.graph.node if n.name.startswith("skinny_gemm/l3/MatMulNBits/k_proj/MatMulPacked/")]
        assert [n.op_type for n in kv_nodes] == ["Cast", "MatMul", "Split"]
        sub_g = helper.make_graph(
            [helper.make_node("Cast", ["x"], [kv_nodes[0].input[0]], to=TensorProto.BFLOAT16), *kv_nodes],
            "kv_fp16_exec",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, hidden])],
            [
                helper.make_tensor_value_info(kv_nodes[-1].output[0], TensorProto.FLOAT16, [1, 2, 64]),
                helper.make_tensor_value_info(kv_nodes[-1].output[1], TensorProto.FLOAT16, [1, 2, 64]),
            ],
            initializer=[kv_tp, inits["const_split_qsa_kv_512_512"]],
        )
        k_out, v_out = _run_ort_subgraph_in_pkg(ort, pkg_dir, sub_g, {"x": x_f32})
        np.testing.assert_allclose(
            k_out.astype(np.float32),
            x_f32 @ fx["hf_tensors"]["model.language_model.layers.3.self_attn.k_proj.weight"].T,
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            v_out.astype(np.float32),
            x_f32 @ fx["hf_tensors"]["model.language_model.layers.3.self_attn.v_proj.weight"].T,
            rtol=2e-2,
            atol=2e-2,
        )

    else:
        for dropped in ("q", "k", "v", "o_proj"):
            assert f"model.layers.3.self_attn.{dropped}.gguf_weight" not in packed_weights
            assert f"model.layers.3.self_attn.{dropped}.gguf_weight" not in inits
        qkv_tp = inits["model.layers.3.self_attn.qkv.weight_t.fp16"]
        o_tp = inits["model.layers.3.self_attn.o_proj.weight_t.fp16"]
        assert qkv_tp.data_type == TensorProto.FLOAT16 and o_tp.data_type == TensorProto.FLOAT16

        qkv_nodes = [
            n for n in model.graph.node if n.name.startswith("skinny_gemm/l3/MatMulNBits/q_proj/MatMulPacked/")
        ]
        o_nodes = [n for n in model.graph.node if n.name.startswith("skinny_gemm/l3_qsa_out/MatMulPacked/o_proj/")]
        assert [n.op_type for n in qkv_nodes] == ["Cast", "MatMul", "Split"]
        assert [n.op_type for n in o_nodes] == ["MatMul", "Cast"]
        sub_g = helper.make_graph(
            [
                helper.make_node("Cast", ["x"], [qkv_nodes[0].input[0]], to=TensorProto.BFLOAT16),
                *qkv_nodes,
                *o_nodes,
                helper.make_node("Cast", [o_nodes[-1].output[0]], ["o_f32"], to=TensorProto.FLOAT),
            ],
            "qkv_o_fp16_exec",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, hidden]),
                helper.make_tensor_value_info(o_nodes[0].input[0], TensorProto.FLOAT16, [1, 2, int(fx["gdn_v_dim"])]),
            ],
            [
                helper.make_tensor_value_info(qkv_nodes[-1].output[0], TensorProto.FLOAT16, [1, 2, 128]),
                helper.make_tensor_value_info("o_f32", TensorProto.FLOAT, [1, 2, hidden]),
            ],
            initializer=[qkv_tp, o_tp, inits["const_split_qsa_qkv_12288_512_512"]],
        )
        o_in = np.ones((1, 2, int(fx["gdn_v_dim"])), dtype=np.float16) * np.float16(0.25)
        q_out, o_out = _run_ort_subgraph_in_pkg(ort, pkg_dir, sub_g, {"x": x_f32, o_nodes[0].input[0]: o_in})
        np.testing.assert_allclose(
            q_out.astype(np.float32),
            x_f32 @ fx["hf_tensors"]["model.language_model.layers.3.self_attn.q_proj.weight"].T,
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            o_out,
            o_in.astype(np.float32) @ fx["hf_tensors"]["model.language_model.layers.3.self_attn.o_proj.weight"].T,
            rtol=2e-2,
            atol=2e-2,
        )


@pytest.mark.parametrize(
    "bad_flag",
    [
        ["--ple-engram", "int8"],
        ["--indexer-dtype", "int8"],
        ["--gdn-gate-dtype", "int8"],
    ],
)
def test_unsupported_gguf_int8_precision_overrides_rejected_before_build(package_builder, tmp_path, bad_flag):
    """Regression test: RC5-only INT8 component overrides are rejected before modifying any output directory."""
    fx = _make_synthetic_48_layer_fixture(tmp_path / "fx", package_builder)
    out_root = tmp_path / "out"
    target = out_root / "qwen_flash_next_gguf_ud_iq1_s"
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("intact")
    before = _snapshot_tree(target)

    with pytest.raises(ValueError, match="not supported for GGUF packages"):
        package_builder.main(
            [
                "--gguf-root",
                str(fx["gguf_root"]),
                "--base-pkg",
                str(fx["base_pkg"]),
                "--hf-dir",
                str(fx["hf_dir"]),
                "--out-root",
                str(out_root),
                "--variants",
                "UD-IQ1_S",
                *bad_flag,
            ]
        )
    assert _snapshot_tree(target) == before


def test_precision_overrides_recalculate_gguf_vram_and_runtime_profiles(
    package_builder, precision_options_mod, tmp_path
):
    """Regression test: FP16/BF16 KV cache and FP32 overrides recalculate VRAM estimates and runtime_profiles thresholds."""
    for weight_bytes, default_256k_min in (
        (38_285_551_616, 52_000_000_000),
        (47_738_679_296, 61_000_000_000),
        (55_544_619_008, 69_000_000_000),
    ):
        est_i8 = package_builder.estimate_gguf_peak_vram_and_thresholds(weight_bytes, kv_cache_dtype="int8")
        est_f16 = package_builder.estimate_gguf_peak_vram_and_thresholds(weight_bytes, kv_cache_dtype="fp16")
        assert est_f16["peak_256k_bytes"] - est_i8["peak_256k_bytes"] == 262_152 * 12_288
        assert est_f16["min_vram_256k_bytes"] > default_256k_min
        for tier in ("32k", "64k", "128k", "256k"):
            assert est_f16[f"min_vram_{tier}_bytes"] >= est_f16[f"peak_{tier}_bytes"] + 2_000_000_000

    fx = _make_synthetic_48_layer_fixture(tmp_path / "fx", package_builder)
    out_root = tmp_path / "out"
    assert (
        package_builder.main(
            [
                "--gguf-root",
                str(fx["gguf_root"]),
                "--base-pkg",
                str(fx["base_pkg"]),
                "--hf-dir",
                str(fx["hf_dir"]),
                "--out-root",
                str(out_root),
                "--variants",
                "UD-IQ1_S",
            ]
        )
        == 0
    )
    base_out = out_root / "qwen_flash_next_gguf_ud_iq1_s"
    base_manifest = json.loads((base_out / "package_manifest.json").read_text())
    base_cfg = json.loads((base_out / "genai_config.json").read_text())

    assert (
        package_builder.main(
            [
                "--gguf-root",
                str(fx["gguf_root"]),
                "--base-pkg",
                str(fx["base_pkg"]),
                "--hf-dir",
                str(fx["hf_dir"]),
                "--out-root",
                str(out_root),
                "--variants",
                "UD-IQ1_S",
                "--precision-profile",
                "fp32_fp16kv_v2",
                "--pkg-suffix",
                "_v2",
            ]
        )
        == 0
    )
    v2_out = out_root / "qwen_flash_next_gguf_ud_iq1_s_v2"
    v2_manifest = json.loads((v2_out / "package_manifest.json").read_text())
    v2_cfg = json.loads((v2_out / "genai_config.json").read_text())

    assert v2_manifest["vram_and_blocks"]["kv_cache_dtype"] == "fp16"
    assert (
        v2_manifest["vram_and_blocks"]["peak_256k_bytes"]
        > base_manifest["vram_and_blocks"]["peak_256k_bytes"] + 3 * 2**30
    )
    base_p256 = next(p for p in base_cfg["runtime_profiles"] if p["id"] == "gpu-256k")
    v2_p256 = next(p for p in v2_cfg["runtime_profiles"] if p["id"] == "gpu-256k")
    assert (
        v2_p256["eligibility"]["minimum_total_device_memory_bytes"]
        > base_p256["eligibility"]["minimum_total_device_memory_bytes"]
    )
    assert (
        v2_p256["eligibility"]["minimum_total_device_memory_bytes"] > v2_manifest["vram_and_blocks"]["peak_256k_bytes"]
    )
    assert v2_manifest["ops"]["MatMulPacked"] == 159 - 12
    assert v2_manifest["external_data_files"] == ["model.onnx.data", "dflash2.onnx.data"]
