# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests of the Qwen3.8 package builders in ``tools/qwen38_graph``.

The builders run on tiny synthetic checkpoints and packages, so no GPU, checkpoint or custom-op library is needed. The tests cover
what the builders do to files (handles that stay open on failure, an output folder that overlaps an input, the reused
``contrib_weights.data``), not the numerics of the converted graphs.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import json
import os
import re
import sys
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from _qwen38_tools_test_utils import load_tools
from onnx import TensorProto, helper, numpy_helper

ALIGN = 4096


@pytest.fixture(scope="module")
def builders():
    with load_tools("package_paths", "build_package_v12d_int4_int8", "build_package_contrib_ops") as (
        paths,
        v12d,
        contrib,
    ):
        yield types.SimpleNamespace(paths=paths, v12d=v12d, contrib=contrib)


class ReachedError(Exception):
    """Raised by the stand-ins that show how far a builder got."""


def snapshot(root):
    """Every path below ``root`` with the SHA-256 of its content (``<dir>`` for folders)."""
    return {
        path.relative_to(root).as_posix(): "<dir>" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
    }


def fingerprint(path):
    stat = path.stat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), stat.st_ino, stat.st_size, stat.st_mtime_ns


def run_main(monkeypatch, module, *argv):
    monkeypatch.setattr(sys, "argv", [f"{module.__name__}.py", *map(str, argv)])
    return module.main()


def spy_on_written_files(monkeypatch, root):
    """Returns the list that receives every file object opened for writing or appending below ``root``."""
    handles = []
    real_open = builtins.open
    root = root.resolve()

    def open_and_record(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if (
            isinstance(file, (str, os.PathLike))
            and set(mode) & set("wa+")
            and Path(file).resolve().is_relative_to(root)
        ):
            handles.append(handle)
        return handle

    monkeypatch.setattr(builtins, "open", open_and_record)
    return handles


# ---------------------------------------------------------------------------
# build_package_v12d_int4_int8.py: PLE shards
# ---------------------------------------------------------------------------

PLE_SCALE = 0.000703105776
PLE_SHARD_NAME = re.compile(
    r"model\.language_model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight"
)
PLE_FILE_SHARDS = [13] * 8 + [12] * 2


def ple_rows(shard):
    return 2 + shard % 3


def ple_codes(shard):
    # Codes beyond the INT8 range check the clamp: a stored 130 (or -130) has to become 127 (or -127).
    return np.random.default_rng(shard).integers(-130, 131, size=(ple_rows(shard), 160))


class DeviceCopy:
    """What ``read_bf16_tensor`` returns: ``.to(device="cuda", ...)`` stays on the CPU here."""

    def __init__(self, tensor):
        self.tensor = tensor

    def to(self, device=None, dtype=None):
        return self.tensor


class FakeCheckpoint:
    def __init__(self, fail_at=None):
        self.fail_at = fail_at

    def read_bf16_tensor(self, name):
        shard = int(PLE_SHARD_NAME.fullmatch(name).group(1))
        if shard == self.fail_at:
            raise RuntimeError(f"cannot read shard {shard}")
        return DeviceCopy((torch.from_numpy(ple_codes(shard)).float() * PLE_SCALE).to(torch.bfloat16))


def make_ple_inputs(root):
    v12d_dir = root / "v12d"
    v12d_dir.mkdir()
    (v12d_dir / "ple_table.txt").write_text(
        "qwen38_ple_table 1\nscale -0.0005\nwidth 160\nshard ple/stale_00.data 0 7\nshard ple/stale_00.data 1120 7\n"
    )
    return v12d_dir, root / "out"


def test_ple_shards_are_quantized_into_ten_files(builders, tmp_path):
    v12d_dir, out = make_ple_inputs(tmp_path)
    builders.v12d.build_ple_int8(FakeCheckpoint(), v12d_dir, out)

    expected_table = ["qwen38_ple_table 1", "scale -0.000703105776", "width 160"]
    shard = 0
    for index, count in enumerate(PLE_FILE_SHARDS):
        rel = f"ple/ple_int8_{index:02d}.data"
        content = b""
        for _ in range(count):
            expected_table.append(f"shard {rel} {len(content)} {ple_rows(shard)}")
            content += np.clip(ple_codes(shard), -127, 127).astype(np.int8).tobytes()
            shard += 1
        assert (out / rel).read_bytes() == content
    assert shard == 128
    assert (out / "ple_table.txt").read_text().splitlines() == expected_table
    assert sorted(p.name for p in (out / "ple").iterdir()) == [f"ple_int8_{i:02d}.data" for i in range(10)]


@pytest.mark.parametrize("fail_at", [0, 5, 13, 127], ids=["first-shard", "mid-file", "second-file", "last-shard"])
def test_ple_files_are_closed_when_a_shard_cannot_be_read(builders, tmp_path, monkeypatch, fail_at):
    v12d_dir, out = make_ple_inputs(tmp_path)
    handles = spy_on_written_files(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match=f"cannot read shard {fail_at}$") as excinfo:
        builders.v12d.build_ple_int8(FakeCheckpoint(fail_at=fail_at), v12d_dir, out)
    # The traceback keeps the frame, and with it any handle the builder still owns, alive like a caller that stores the exception.
    assert excinfo.traceback
    assert handles
    assert [handle.name for handle in handles if not handle.closed] == []
    assert not (out / "ple_table.txt").exists()


# ---------------------------------------------------------------------------
# package_paths.py and the output checks of both builders
# ---------------------------------------------------------------------------

V12D_INPUTS = {
    "--hf-dir": "dir",
    "--v12b-dir": "dir",
    "--v12d-dir": "dir",
    "--ops-dir": "dir",
    "--reuse-ple-from": "dir",
    "--reuse-qmoe-from": "dir",
    "--reuse-embed-from": "dir",
    "--plane-lib": "file",
    "--provider-lib": "file",
}
CONTRIB_INPUTS = {"--src": "dir", "--reuse-hc-bf16-from": "dir", "--plane-lib": "file", "--provider-lib": "file"}
OVERLAPS = ["equal", "child", "parent"]


def make_inputs(root, inputs, flag, overlap):
    """Creates every input of ``inputs`` (flag -> "dir" | "file") and returns ``(paths, out)``.

    The input named ``flag`` overlaps ``out`` the way ``overlap`` says: "equal", "child" (``out`` inside the input), "parent" (the
    input inside ``out``) or "sibling" (no overlap, the two names only share a prefix). A folder input holds the files a builder
    would delete or rewrite, so damage to it shows up in ``snapshot``.
    """
    paths = {name: root / "inputs" / name.lstrip("-") for name in inputs}
    out = root / "out"
    if overlap == "equal":
        out = paths[flag]
    elif overlap == "child":
        out = paths[flag] / "out"
    elif overlap == "parent":
        paths[flag] = out / "nested" / "input"
    else:
        paths[flag], out = root / "pkg", root / "pkg2"
    for name, kind in inputs.items():
        if kind == "file":
            paths[name].parent.mkdir(parents=True, exist_ok=True)
            paths[name].write_bytes(name.encode())
            continue
        for rel in ("weights.bin", "libs/lib.so", "provider/lib.so", "ple/ple_int8_00.data", "model.onnx", "mtp.onnx"):
            (paths[name] / rel).parent.mkdir(parents=True, exist_ok=True)
            (paths[name] / rel).write_bytes(f"{name} {rel}".encode())
    return paths, out


def command_line(paths, out):
    return [arg for flag, path in paths.items() for arg in (flag, str(path))] + ["--out", str(out)]


def overlap_message(flag):
    return f"nested inside or contain {re.escape(flag)} "


def stop_at_checkpoint_reader(monkeypatch, module):
    def reader(*_args, **_kwargs):
        raise ReachedError("HFCheckpointReader")

    monkeypatch.setattr(module, "HFCheckpointReader", reader)


def stop_at_weight_writer(monkeypatch, module):
    def writer(*_args, **_kwargs):
        raise ReachedError("ContribWeightWriter")

    monkeypatch.setattr(module, "ContribWeightWriter", writer)


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("flag", V12D_INPUTS)
def test_v12d_builder_refuses_an_output_overlapping_an_input(builders, tmp_path, monkeypatch, flag, overlap):
    # ("equal", "--reuse-embed-from") is the clean-flat case whose cleanup used to delete the embedding package it reads from.
    paths, out = make_inputs(tmp_path, V12D_INPUTS, flag, overlap)
    stop_at_checkpoint_reader(monkeypatch, builders.v12d)
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit, match=overlap_message(flag)):
        run_main(monkeypatch, builders.v12d, *command_line(paths, out), "--clean-flat-package")
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("flag", V12D_INPUTS)
def test_v12d_builder_accepts_an_output_that_only_shares_a_name_prefix_with_an_input(
    builders, tmp_path, monkeypatch, flag
):
    paths, out = make_inputs(tmp_path, V12D_INPUTS, flag, "sibling")
    stop_at_checkpoint_reader(monkeypatch, builders.v12d)
    with pytest.raises(ReachedError):
        run_main(monkeypatch, builders.v12d, *command_line(paths, out))
    assert (out / "libs").is_dir()


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("flag", CONTRIB_INPUTS)
def test_contrib_builder_refuses_an_output_overlapping_an_input(builders, tmp_path, monkeypatch, flag, overlap):
    paths, out = make_inputs(tmp_path, CONTRIB_INPUTS, flag, overlap)
    stop_at_weight_writer(monkeypatch, builders.contrib)
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit, match=overlap_message(flag)):
        run_main(monkeypatch, builders.contrib, *command_line(paths, out))
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("flag", CONTRIB_INPUTS)
def test_contrib_builder_accepts_an_output_that_only_shares_a_name_prefix_with_an_input(
    builders, tmp_path, monkeypatch, flag
):
    paths, out = make_inputs(tmp_path, CONTRIB_INPUTS, flag, "sibling")
    stop_at_weight_writer(monkeypatch, builders.contrib)
    with pytest.raises(ReachedError):
        run_main(monkeypatch, builders.contrib, *command_line(paths, out))


def test_contrib_builder_checks_the_reference_before_it_deletes_the_output(builders, tmp_path, monkeypatch):
    paths, out = make_inputs(tmp_path, CONTRIB_INPUTS, "--src", "sibling")
    paths["--reuse-hc-bf16-from"] = tmp_path / "missing"
    out.mkdir()
    (out / "previous.bin").write_bytes(b"previous")
    with pytest.raises(FileNotFoundError):
        run_main(monkeypatch, builders.contrib, *command_line(paths, out))
    assert (out / "previous.bin").read_bytes() == b"previous"


def test_overlap_check_compares_resolved_paths(builders, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are not available")
    for out in (alias, alias / "out", tmp_path / "other" / ".." / "source"):
        with pytest.raises(SystemExit, match=overlap_message("--src")):
            builders.paths.reject_output_overlap(out, {"--src": source})


def test_overlap_check_returns_the_resolved_output_and_skips_unset_inputs(builders, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pkg").mkdir()
    out = builders.paths.reject_output_overlap(Path("pkg2"), {"--src": Path("pkg"), "--reuse-embed-from": None})
    assert out == (tmp_path / "pkg2").resolve()
    assert out.is_absolute()


# ---------------------------------------------------------------------------
# build_package_contrib_ops.py: ContribWeightWriter
# ---------------------------------------------------------------------------


def make_writer_dirs(root):
    src, out = root / "src", root / "out"
    src.mkdir()
    out.mkdir()
    return src, out


def make_reference_data(root, size=ALIGN + 1024):
    """A ``contrib_weights.data`` whose end is off the 4096-byte grid."""
    root.mkdir()
    path = root / "contrib_weights.data"
    path.write_bytes(np.random.default_rng(size).bytes(size))
    return path


def test_reused_data_is_extended_in_a_private_copy(builders, tmp_path):
    src, out = make_writer_dirs(tmp_path)
    reference = make_reference_data(tmp_path / "reference")
    reused = reference.read_bytes()
    before = fingerprint(reference)

    writer = builders.contrib.ContribWeightWriter(src, out, reuse_data=reference)
    assert os.path.samefile(reference, out / "contrib_weights.data")
    first = writer.write_bytes(b"\xaa" * 10, TensorProto.UINT8, [10])
    second = writer.write_bytes(b"\xbb" * 20, TensorProto.UINT8, [20])
    writer.close()

    first_offset = len(reused) + (-len(reused) % ALIGN)
    assert first == {
        "location": "contrib_weights.data",
        "offset": first_offset,
        "length": 10,
        "dtype": TensorProto.UINT8,
        "dims": [10],
    }
    assert second["offset"] == first_offset + ALIGN
    expected = reused + bytes(first_offset - len(reused)) + b"\xaa" * 10 + bytes(ALIGN - 10) + b"\xbb" * 20
    assert (out / "contrib_weights.data").read_bytes() == expected
    assert fingerprint(reference) == before
    assert not os.path.samefile(reference, out / "contrib_weights.data")


def test_reused_data_stays_linked_while_nothing_is_appended(builders, tmp_path):
    src, out = make_writer_dirs(tmp_path)
    reference = make_reference_data(tmp_path / "reference")
    before = fingerprint(reference)
    writer = builders.contrib.ContribWeightWriter(src, out, reuse_data=reference)
    writer.close()
    writer.close()
    assert os.path.samefile(reference, out / "contrib_weights.data")
    assert fingerprint(reference) == before


def test_writing_after_close_is_an_error(builders, tmp_path):
    src, out = make_writer_dirs(tmp_path)
    reference = make_reference_data(tmp_path / "reference")
    for reuse_data in (None, reference):
        writer = builders.contrib.ContribWeightWriter(src, out, reuse_data=reuse_data)
        writer.close()
        with pytest.raises(ValueError, match="closed"):
            writer.write_bytes(b"late", TensorProto.UINT8, [4])


def test_a_new_data_file_is_not_written_through_a_hard_link_to_the_source(builders, tmp_path):
    src, out = make_writer_dirs(tmp_path)
    (src / "contrib_weights.data").write_bytes(b"source package weights")
    os.link(src / "contrib_weights.data", out / "contrib_weights.data")
    before = fingerprint(src / "contrib_weights.data")

    writer = builders.contrib.ContribWeightWriter(src, out)
    info = writer.write_bytes(b"new", TensorProto.UINT8, [3])
    writer.close()

    assert info["offset"] == 0
    assert (out / "contrib_weights.data").read_bytes() == b"new"
    assert fingerprint(src / "contrib_weights.data") == before


# ---------------------------------------------------------------------------
# build_package_contrib_ops.py: conversions that append weights, with and without --reuse-hc-bf16-from
# ---------------------------------------------------------------------------

HC = "l0.hc"
K, N = 64, 32


def append_external(path, raw):
    """Appends ``raw`` to ``path`` at the next 4096-byte aligned offset and returns that offset."""
    size = path.stat().st_size if path.exists() else 0
    offset = -(-size // ALIGN) * ALIGN
    with open(path, "ab") as f:
        f.write(bytes(offset - size) + raw)
    return offset


def external_init(name, dtype, dims, location, offset, length):
    tensor = TensorProto(name=name, data_type=dtype, dims=dims, data_location=TensorProto.EXTERNAL)
    for key, value in (("location", location), ("offset", str(offset)), ("length", str(length))):
        tensor.external_data.add(key=key, value=value)
    return tensor


def external_ranges(model_path):
    """Maps every external initializer of the model to ``(location, offset, length, data type, dims)``."""
    ranges = {}
    for tensor in onnx.load(str(model_path), load_external_data=False).graph.initializer:
        external = {entry.key: entry.value for entry in tensor.external_data}
        if external:
            ranges[tensor.name] = (
                external["location"],
                int(external["offset"]),
                int(external["length"]),
                tensor.data_type,
                list(tensor.dims),
            )
    return ranges


def save_model(path, nodes, outputs, initializers, domains=()):
    graph = helper.make_graph(
        nodes,
        "graph",
        [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [1, K])],
        [helper.make_tensor_value_info(name, TensorProto.BFLOAT16, None) for name in outputs],
        initializers,
    )
    opsets = [helper.make_opsetid("", 17), *(helper.make_opsetid(domain, 1) for domain in domains)]
    onnx.save_model(helper.make_model(graph, opset_imports=opsets), path)


def bf16_values(raw):
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16)


def fp8_values(raw):
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).view(torch.float8_e4m3fn)


@dataclasses.dataclass(frozen=True)
class Variant:
    """A graph whose conversion has to write weights, and what the converted package must contain."""

    nodes: list
    initializers: list
    outputs: list
    domains: tuple
    flags: tuple
    expected: dict  # initializer added by the conversion -> (data type, dims)
    verify: Callable  # called with the bytes of those initializers by name


def quantized_weight(name):
    return {
        f"{name}.fp8_nk": (TensorProto.FLOAT8E4M3FN, [N, K]),
        f"{name}.scale_n1": (TensorProto.FLOAT, [N, 1]),
    }


def bf16_weight(seed):
    weight = torch.randn(K, N, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)
    return weight, weight.view(torch.uint16).numpy().tobytes()


def verify_quantized(name, weight):
    def verify(new):
        quantized = fp8_values(new[f"{name}.fp8_nk"]).float().reshape(N, K)
        scale = torch.from_numpy(np.frombuffer(new[f"{name}.scale_n1"], dtype=np.float32).copy()).reshape(N, 1)
        torch.testing.assert_close(quantized * scale, weight.float().T, rtol=0.07, atol=1e-4)

    return verify


def fp8_skinny_gemm(root):
    """The NVFP4 + FP8 package: an FP8 weight that is transposed to [N, K] with its scales."""
    weight = np.random.default_rng(1).integers(0, 0x7F, size=(K, N), dtype=np.uint8)
    scales = (1 + np.arange(N) / 8).astype(np.float32)
    offset = append_external(root / "weights.data", weight.tobytes())

    def verify(new):
        assert new["w.fp8_nk"] == weight.T.tobytes()
        assert new["w.scale_n1"] == scales.tobytes()

    return Variant(
        nodes=[
            helper.make_node(
                "SkinnyGemm", ["x", "w"], ["y"], name="gemm", domain="qwen.fused_gemm", w_scale=scales.tolist()
            )
        ],
        initializers=[external_init("w", TensorProto.FLOAT8E4M3FN, [K, N], "weights.data", offset, weight.nbytes)],
        outputs=["y"],
        domains=("qwen.fused_gemm",),
        flags=(),
        expected=quantized_weight("w"),
        verify=verify,
    )


def bf16_skinny_gemm(root):
    """The INT4 + INT8 package with ``--fp8-mlp-shared``: a BF16 weight that is quantized to FP8."""
    weight, raw = bf16_weight(2)
    offset = append_external(root / "weights.data", raw)
    return Variant(
        nodes=[helper.make_node("SkinnyGemm", ["x", "w"], ["y"], name="gemm", domain="qwen.fused_gemm")],
        initializers=[external_init("w", TensorProto.BFLOAT16, [K, N], "weights.data", offset, len(raw))],
        outputs=["y"],
        domains=("qwen.fused_gemm",),
        flags=("--fp8-mlp-shared",),
        expected=quantized_weight("w"),
        verify=verify_quantized("w", weight),
    )


def bf16_shared_expert_down(root):
    """The INT4 + INT8 package with ``--fp8-shared-down-only``: the BF16 shared-expert down weight becomes FP8."""
    weight, raw = bf16_weight(3)
    offset = append_external(root / "weights.data", raw)
    return Variant(
        nodes=[
            helper.make_node(
                "SharedExpertDown", ["x", "x", "down", "x", "x", "x"], ["y"], name="down", domain="qwen.fused_gemm"
            )
        ],
        initializers=[external_init("down", TensorProto.BFLOAT16, [K, N], "weights.data", offset, len(raw))],
        outputs=["y"],
        domains=("qwen.fused_gemm",),
        flags=("--fp8-shared-down-only",),
        expected=quantized_weight("down"),
        verify=verify_quantized("down", weight),
    )


def fp8_hyper_connection(root):
    """The NVFP4 + FP8 package: tiled FP8 hyper-connection weights that are untiled to BF16."""
    rng = np.random.default_rng(4)
    weights = {name: rng.integers(0, 0x7F, size=(160, 20480), dtype=np.uint8) for name in ("hc.wd", "hc.wu")}
    initializers = [numpy_helper.from_array(np.ones(4, dtype=np.float32), "hc.norm")]
    initializers.append(helper.make_tensor("hc.wi", TensorProto.BFLOAT16, [4, 4], bytes(32), raw=True))
    for name, weight in weights.items():
        offset = append_external(root / "weights.data", weight.tobytes())
        initializers.append(
            external_init(name, TensorProto.FLOAT8E4M3FN, [160, 20480], "weights.data", offset, weight.nbytes)
        )

    def verify(new):
        # The scales are 1, so the untiled weight holds the same values in another order; the sums are exact in float64.
        for name, weight in weights.items():
            untiled = bf16_values(new[f"{name}.untiled_bf16"]).double().sum().item()
            assert untiled == fp8_values(weight.tobytes()).double().sum().item()

    return Variant(
        nodes=[
            helper.make_node(
                "HcBlock",
                ["x", "hc.norm", "hc.wd", "hc.wi", "hc.wu"],
                ["hc_pre", "hc_inject"],
                name="hc",
                domain="qwen.fused_hc",
                wd_scale=[1.0] * 320,
                wu_scale=[1.0] * 10240,
            )
        ],
        initializers=initializers,
        outputs=["hc_pre", "hc_inject"],
        domains=("qwen.fused_hc",),
        flags=(),
        expected={
            "hc.wd.untiled_bf16": (TensorProto.BFLOAT16, [10240, 320]),
            "hc.wu.untiled_bf16": (TensorProto.BFLOAT16, [320, 10240]),
        },
        verify=verify,
    )


VARIANTS = {
    "fp8-skinny-gemm": fp8_skinny_gemm,
    "bf16-skinny-gemm-fp8-mlp-shared": bf16_skinny_gemm,
    "bf16-shared-expert-down-fp8-down-only": bf16_shared_expert_down,
    "fp8-hyper-connection": fp8_hyper_connection,
}


def hyper_connection_chain():
    """The INT8 hyper-connection chain that ``--reuse-hc-bf16-from`` fuses into ``HyperConnectionLowRankMix``."""
    nodes = [
        helper.make_node("BranchwiseRMSNorm", ["x", f"{HC}.norm"], [f"{HC}.normed"], name=f"{HC}.rms"),
        helper.make_node("MatMulNBits", [f"{HC}.normed", f"{HC}.wd.qweight"], [f"{HC}.down"], name=f"{HC}.down"),
        helper.make_node("ScaledSiLU", [f"{HC}.down"], [f"{HC}.act"], name=f"{HC}.act"),
        helper.make_node("MatMulNBits", [f"{HC}.act", f"{HC}.wu.qweight"], [f"{HC}.up"], name=f"{HC}.up"),
        helper.make_node("Sigmoid", [f"{HC}.up"], [f"{HC}.gate"], name=f"{HC}.sigmoid"),
        helper.make_node("HyperConnectionPreMix", ["x", f"{HC}.gate"], [f"{HC}.pre"], name=f"{HC}.pre"),
        helper.make_node("HcInjectGate", [f"{HC}.pre", f"{HC}.inject_w"], [f"{HC}.inject"], name=f"{HC}.inject"),
    ]
    initializers = [
        numpy_helper.from_array(np.ones(4, dtype=np.float32), f"{HC}.norm"),
        helper.make_tensor(f"{HC}.inject_w", TensorProto.BFLOAT16, [4, 4], bytes(32), raw=True),
    ]
    return nodes, initializers, [f"{HC}.pre", f"{HC}.inject"]


def make_reference_package(root):
    """A converted package whose ``contrib_weights.data`` ends off the 4096-byte grid and holds the BF16 weights of ``HC``."""
    root.mkdir()
    initializers = []
    for suffix, dims in (("wd", [4, 8]), ("wu", [5, 10])):
        raw = np.random.default_rng(len(initializers)).bytes(2 * dims[0] * dims[1])
        offset = append_external(root / "contrib_weights.data", raw)
        initializers.append(
            external_init(
                f"{HC}.{suffix}.untiled_bf16", TensorProto.BFLOAT16, dims, "contrib_weights.data", offset, len(raw)
            )
        )
    for name in ("model.onnx", "mtp.onnx"):
        save_model(root / name, [helper.make_node("Identity", ["x"], ["y"])], ["y"], initializers)
    return root


def package_fingerprint(root):
    return {path.name: fingerprint(path) for path in sorted(root.iterdir())}


def convert(builders, monkeypatch, root, make_variant, reference=None, flags=None, extra_nodes=()):
    """Builds a synthetic source package below ``root`` and runs ``build_package_contrib_ops.main()`` on it.

    The names of the source, the reference and the output share a prefix, so a path check that compares strings would refuse them.
    """
    src = root / "qwen38"
    src.mkdir()
    variant = make_variant(src)
    nodes, initializers, outputs = list(variant.nodes), list(variant.initializers), list(variant.outputs)
    domains = variant.domains
    if reference is not None:
        chain_nodes, chain_initializers, chain_outputs = hyper_connection_chain()
        nodes, initializers, outputs = chain_nodes + nodes, initializers + chain_initializers, outputs + chain_outputs
    if extra_nodes:
        nodes += extra_nodes
        domains += ("qwen.unsupported",)
        outputs.append("unsupported")
    save_model(src / "model.onnx", nodes, outputs, initializers, domains)
    save_model(src / "mtp.onnx", [helper.make_node("Identity", ["x"], ["y"])], ["y"], [])
    config = {"model": {"decoder": {"session_options": {"custom_ops_libraries": ["libs/libqwen_fold.so"]}}}}
    (src / "genai_config.json").write_text(json.dumps(config))

    libs = root / "libs"
    libs.mkdir()
    (libs / "plane.so").write_bytes(b"plane")
    (libs / "provider.so").write_bytes(b"provider")
    out = root / "qwen38_contrib"
    argv = ["--src", src, "--out", out, "--plane-lib", libs / "plane.so", "--provider-lib", libs / "provider.so"]
    argv += variant.flags if flags is None else flags
    if reference is not None:
        argv += ["--reuse-hc-bf16-from", reference]
    status = run_main(monkeypatch, builders.contrib, *argv)
    return types.SimpleNamespace(src=src, out=out, variant=variant, status=status)


@pytest.mark.parametrize("reuse", [True, False], ids=["reuse-hc-bf16", "fresh"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_conversion_appends_new_weights_after_any_reused_data(builders, tmp_path, monkeypatch, variant, reuse):
    reference = make_reference_package(tmp_path / "qwen38_reference") if reuse else None
    reused = (reference / "contrib_weights.data").read_bytes() if reuse else b""
    reference_before = package_fingerprint(reference) if reuse else None

    run = convert(builders, monkeypatch, tmp_path, VARIANTS[variant], reference)

    assert run.status == 0
    assert json.loads((run.out / "package_manifest.json").read_text())["contrib_ops_only"] is True
    data = (run.out / "contrib_weights.data").read_bytes()
    assert data[: len(reused)] == reused
    ranges = external_ranges(run.out / "model.onnx")
    assert len(data) == max(offset + length for _, offset, length, _, _ in ranges.values())

    new = {}
    for name, (dtype, dims) in run.variant.expected.items():
        location, offset, length, got_dtype, got_dims = ranges[name]
        assert (location, got_dtype, got_dims) == ("contrib_weights.data", dtype, dims)
        assert offset % ALIGN == 0
        assert offset >= len(reused)
        new[name] = data[offset : offset + length]
    run.variant.verify(new)

    if reuse:
        # The fused chain still reads the weights of the reference at their original offsets.
        for name, (location, offset, length, _, _) in external_ranges(reference / "model.onnx").items():
            assert ranges[name][:3] == (location, offset, length)
        assert package_fingerprint(reference) == reference_before
        reference_data, converted_data = reference / "contrib_weights.data", run.out / "contrib_weights.data"
        assert not os.path.samefile(reference_data, converted_data)
        assert reference_data.stat().st_nlink == converted_data.stat().st_nlink == 1


def test_reuse_without_new_weights_links_the_reference_file(builders, tmp_path, monkeypatch):
    reference = make_reference_package(tmp_path / "qwen38_reference")
    reference_before = package_fingerprint(reference)

    # Without --fp8-mlp-shared the BF16 SkinnyGemm stays a MatMul, so nothing is appended.
    run = convert(builders, monkeypatch, tmp_path, bf16_skinny_gemm, reference, flags=())

    assert run.status == 0
    assert os.path.samefile(reference / "contrib_weights.data", run.out / "contrib_weights.data")
    assert package_fingerprint(reference) == reference_before


@pytest.mark.parametrize("reuse", [True, False], ids=["reuse-hc-bf16", "fresh"])
def test_failed_conversion_releases_the_data_file_and_keeps_the_reference(builders, tmp_path, monkeypatch, reuse):
    reference = make_reference_package(tmp_path / "qwen38_reference") if reuse else None
    reference_before = package_fingerprint(reference) if reuse else None
    handles = spy_on_written_files(monkeypatch, tmp_path)

    unsupported = [helper.make_node("Unsupported", ["x"], ["unsupported"], domain="qwen.unsupported")]
    with pytest.raises(RuntimeError, match="Remaining custom ops") as excinfo:
        convert(builders, monkeypatch, tmp_path, fp8_skinny_gemm, reference, extra_nodes=unsupported)

    assert excinfo.traceback
    assert handles
    assert [handle.name for handle in handles if not handle.closed] == []
    if reuse:
        assert package_fingerprint(reference) == reference_before
