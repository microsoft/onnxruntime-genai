# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import base64
import csv
import hashlib
import importlib.util
import io
import struct
import subprocess
import sys
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.utils import parse_wheel_filename
from wheel.wheelfile import WheelFile

REPO_ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("generate_core_wheel", REPO_ROOT / "tools/python/generate_core_wheel.py")
core_wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core_wheel)
spec = importlib.util.spec_from_file_location("verify_core_wheel", REPO_ROOT / "tools/python/verify_core_wheel.py")
verify_core_wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_core_wheel)


def make_core(core_dir, library_name, use_cuda=True):
    library = core_dir / "lib" / library_name
    library.parent.mkdir(parents=True)
    library.write_bytes(b"core library payload")
    library.with_name(library_name.replace("genai", "genai-cuda")).write_bytes(b"CUDA add-on payload")
    config = core_dir / "lib/cmake/onnxruntime-genai/onnxruntime-genaiConfig.cmake"
    config.parent.mkdir(parents=True)
    config.write_text(f"set(onnxruntime-genai_USE_CUDA {int(use_cuda)})\n", encoding="utf-8")
    (core_dir / "lib/onnxruntime.dll").write_bytes(b"ORT must not be packaged")
    (core_dir / "lib/onnxruntime-genai.pdb").write_bytes(b"symbols must not be packaged")
    (core_dir / "lib/onnxruntime-genai.lib").write_bytes(b"import library must not be packaged")
    return library


@pytest.fixture(params=[False, True], ids=["single-tag", "compound-tags"])
def auditwheel_repair(monkeypatch, request):
    commands = []

    def run(command, check):
        commands.append((command, check))
        source = Path(command[-1])
        output_dir = Path(command[command.index("--wheel-dir") + 1])
        platform_tag = command[command.index("--plat") + 1]
        platforms = [platform_tag]
        if request.param:
            platforms += (
                ["manylinux1_x86_64", "manylinux_2_5_x86_64"]
                if platform_tag.endswith("x86_64")
                else [
                    "manylinux2014_aarch64",
                    "manylinux_2_17_aarch64",
                ]
            )
        _, version, _, _ = parse_wheel_filename(source.name)
        repaired = output_dir / f"onnxruntime_genai_core-{version}-py3-none-{'.'.join(platforms)}.whl"
        with ZipFile(source) as original, WheelFile(str(repaired), "w") as wheel:
            for path in original.namelist():
                if path.endswith("/RECORD"):
                    continue
                content = original.read(path)
                if path.endswith("/WHEEL"):
                    old_tag = Parser().parsestr(content.decode("utf-8"))["Tag"]
                    content = content.replace(
                        f"Tag: {old_tag}".encode(),
                        "\n".join(f"Tag: py3-none-{platform}" for platform in platforms).encode(),
                    )
                wheel.writestr(path, content)
            wheel.writestr("onnxruntime_genai_core.libs/libsupport-123.so", b"vendored supporting library")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(core_wheel.subprocess, "run", run)
    return commands


@pytest.mark.parametrize(
    "rid,platform_tag,library_name",
    [
        ("win-x64", "win_amd64", "onnxruntime-genai.dll"),
        ("win-arm64", "win_arm64", "onnxruntime-genai.dll"),
        ("osx-arm64", "macosx_12_0_arm64", "libonnxruntime-genai.dylib"),
        ("linux-x64", "linux_x86_64", "libonnxruntime-genai.so"),
        ("linux-arm64", "linux_aarch64", "libonnxruntime-genai.so"),
    ],
)
def test_core_wheel_contents_and_metadata(tmp_path, rid, platform_tag, library_name):
    core_dir = tmp_path / "core"
    library = make_core(core_dir, library_name, use_cuda=rid != "osx-arm64")
    if rid == "osx-arm64":
        library.with_name(library_name.replace("genai", "genai-cuda")).unlink()
    wheel_path = core_wheel.generate_wheel(core_dir, rid, "0.15.0-dev1001393122", tmp_path / "wheels")
    name, version, _, tags = parse_wheel_filename(wheel_path.name)
    assert name == "onnxruntime-genai-core"
    assert str(version) == "0.15.0.dev1001393122"
    assert {str(tag) for tag in tags} == {f"py3-none-{platform_tag}"}

    dist_info = "onnxruntime_genai_core-0.15.0.dev1001393122.dist-info"
    with ZipFile(wheel_path) as wheel:
        assert set(wheel.namelist()) == {
            "onnxruntime_genai_core/__init__.py",
            f"onnxruntime_genai_core/bin/{library_name}",
            f"{dist_info}/METADATA",
            f"{dist_info}/WHEEL",
            f"{dist_info}/LICENSE",
            f"{dist_info}/ThirdPartyNotices.txt",
            f"{dist_info}/RECORD",
        }
        assert wheel.read(f"onnxruntime_genai_core/bin/{library_name}") == library.read_bytes()
        assert wheel.read("onnxruntime_genai_core/__init__.py") == b""
        metadata = Parser().parsestr(wheel.read(f"{dist_info}/METADATA").decode("utf-8"))
        assert metadata["Name"] == "onnxruntime-genai-core"
        assert metadata["Version"] == str(version)
        assert metadata["Requires-Dist"] is None
        assert metadata["Author"] == "Microsoft Corporation"
        assert "Applications must supply ONNX Runtime" in metadata.get_payload()
        wheel_metadata = Parser().parsestr(wheel.read(f"{dist_info}/WHEEL").decode("utf-8"))
        assert wheel_metadata["Root-Is-Purelib"] == "false"
        assert wheel_metadata["Tag"] == f"py3-none-{platform_tag}"
        records = list(csv.reader(io.StringIO(wheel.read(f"{dist_info}/RECORD").decode("utf-8"))))
        assert {record[0] for record in records} == set(wheel.namelist())
        for path, digest, size in records:
            if path.endswith("/RECORD"):
                assert (digest, size) == ("", "")
            else:
                content = wheel.read(path)
                expected_hash = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
                assert digest == f"sha256={expected_hash}"
                assert int(size) == len(content)


@pytest.mark.parametrize("rid", ["win-x64", "win-arm64", "linux-x64", "linux-arm64"])
def test_core_wheel_rejects_cpu_core_even_with_addon(tmp_path, rid):
    core_dir = tmp_path / "core"
    make_core(core_dir, core_wheel.PLATFORMS[rid][1], use_cuda=False)
    with pytest.raises(ValueError, match="CUDA-built GenAI core"):
        core_wheel.generate_wheel(core_dir, rid, "0.15.0", tmp_path / "wheels")
    assert not (tmp_path / "wheels").exists()


def test_core_wheel_requires_source_addon(tmp_path):
    core_dir = tmp_path / "core"
    library = make_core(core_dir, "libonnxruntime-genai.so")
    library.with_name("libonnxruntime-genai-cuda.so").unlink()
    with pytest.raises(FileNotFoundError, match="Missing CUDA add-on"):
        core_wheel.generate_wheel(core_dir, "linux-arm64", "0.15.0", tmp_path / "wheels")


def test_core_wheel_requires_core_binary(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing GenAI core library"):
        core_wheel.generate_wheel(tmp_path / "missing", "linux-arm64", "0.15.0", tmp_path / "wheels")


def test_core_wheel_cli_packages_stable_version(tmp_path, monkeypatch, capsys, auditwheel_repair):
    core_dir = tmp_path / "core"
    make_core(core_dir, "libonnxruntime-genai.so")
    output_dir = tmp_path / "wheels"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_core_wheel.py",
            "--core_dir",
            str(core_dir),
            "--rid",
            "linux-arm64",
            "--package_version",
            "0.15.0",
            "--output_dir",
            str(output_dir),
        ],
    )
    core_wheel.main()
    wheel = next(output_dir.glob("*.whl"))
    assert wheel.is_file()
    assert capsys.readouterr().out.strip() == str(wheel)
    assert len(auditwheel_repair) == 1
    assert {path.name for path in output_dir.iterdir()} == {wheel.name}


@pytest.mark.parametrize(
    "rid,platform_tag",
    [("linux-x64", "manylinux_2_28_x86_64"), ("linux-arm64", "manylinux_2_34_aarch64")],
)
def test_manylinux_repair_enforces_policy_and_preserves_support_libraries(
    tmp_path, rid, platform_tag, auditwheel_repair
):
    core_dir = tmp_path / "core"
    library = make_core(core_dir, "libonnxruntime-genai.so")
    raw = core_wheel.generate_wheel(core_dir, rid, "0.15.0-dev1001393122", tmp_path / "raw")
    output_dir = tmp_path / "wheels"
    repaired = core_wheel.repair_manylinux_wheel(raw, rid, output_dir)
    command, check = auditwheel_repair[0]
    assert command[:4] == [sys.executable, "-m", "auditwheel", "repair"]
    assert command[command.index("--plat") + 1] == platform_tag
    assert [command[i + 1] for i, arg in enumerate(command) if arg == "--exclude"] == [
        "libonnxruntime.so*",
        "libonnxruntime_providers_*.so*",
        "libonnxruntime-genai-cuda.so",
    ]
    assert command[-1] == str(raw)
    assert check is True
    assert not Path(command[command.index("--wheel-dir") + 1]).exists()
    assert {path.name for path in output_dir.iterdir()} == {repaired.name}
    _, version, _, tags = parse_wheel_filename(repaired.name)
    assert str(version) == "0.15.0.dev1001393122"
    assert any(str(tag) == f"py3-none-{platform_tag}" for tag in tags)
    assert all(tag.interpreter == "py3" and tag.abi == "none" for tag in tags)
    with ZipFile(repaired) as wheel:
        assert wheel.read("onnxruntime_genai_core/bin/libonnxruntime-genai.so") == library.read_bytes()
        assert wheel.read("onnxruntime_genai_core.libs/libsupport-123.so") == b"vendored supporting library"
        wheel_metadata = Parser().parsestr(
            wheel.read(f"onnxruntime_genai_core-{version}.dist-info/WHEEL").decode("utf-8")
        )
        assert set(wheel_metadata.get_all("Tag")) == {str(tag) for tag in tags}


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_wheel",
        "multiple_wheels",
        "tag",
        "version",
        "missing_core",
        "libonnxruntime-123abc.so.1",
        "libonnxruntime-genai-cuda-123abc.so",
    ],
)
def test_manylinux_repair_rejects_invalid_output(tmp_path, monkeypatch, invalid):
    core_dir = tmp_path / "core"
    make_core(core_dir, "libonnxruntime-genai.so")
    raw = core_wheel.generate_wheel(core_dir, "linux-arm64", "0.15.0", tmp_path / "raw")

    def run(command, check):
        output = Path(command[command.index("--wheel-dir") + 1])
        if invalid == "missing_wheel":
            return
        tag = "linux_aarch64" if invalid == "tag" else "manylinux_2_34_aarch64"
        version = "0.16.0" if invalid == "version" else "0.15.0"
        with WheelFile(str(output / f"onnxruntime_genai_core-{version}-py3-none-{tag}.whl"), "w") as wheel:
            if invalid != "missing_core":
                wheel.writestr("onnxruntime_genai_core/bin/libonnxruntime-genai.so", b"core")
            if invalid.startswith("libonnxruntime"):
                wheel.writestr(f"onnxruntime_genai_core.libs/{invalid}", b"runtime-managed library")
        if invalid == "multiple_wheels":
            with WheelFile(str(output / "onnxruntime_genai_core-0.15.0-py3-none-linux_aarch64.whl"), "w") as wheel:
                wheel.writestr("onnxruntime_genai_core/bin/libonnxruntime-genai.so", b"unexpected extra wheel")

    monkeypatch.setattr(core_wheel.subprocess, "run", run)
    output_dir = tmp_path / "wheels"
    with pytest.raises(RuntimeError):
        core_wheel.repair_manylinux_wheel(raw, "linux-arm64", output_dir)
    assert not output_dir.exists()


def test_core_wheel_cli_does_not_publish_when_audit_fails(tmp_path, monkeypatch):
    core_dir = tmp_path / "core"
    make_core(core_dir, "libonnxruntime-genai.so")
    output_dir = tmp_path / "wheels"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_core_wheel.py",
            "--core_dir",
            str(core_dir),
            "--rid",
            "linux-arm64",
            "--package_version",
            "0.15.0",
            "--output_dir",
            str(output_dir),
        ],
    )

    def fail(command, check):
        raise subprocess.CalledProcessError(1, command, stderr="too-recent versioned symbols")

    monkeypatch.setattr(core_wheel.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        core_wheel.main()
    assert not output_dir.exists()


def make_elf(machine=62, needed=()):
    strings = b"\0"
    offsets = []
    for name in needed:
        offsets.append(len(strings))
        strings += name.encode("ascii") + b"\0"
    dynamic = b"".join(struct.pack("<qQ", 1, offset) for offset in offsets) + struct.pack("<qQ", 0, 0)
    names = b"\0.dynamic\0.dynstr\0.shstrtab\0"
    section_offset = 64 + len(dynamic) + len(strings) + len(names)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + b"\0" * 9,
        3,
        machine,
        1,
        0,
        0,
        section_offset,
        0,
        64,
        0,
        0,
        64,
        4,
        3,
    )
    sections = bytes(64)
    sections += struct.pack("<IIQQQQIIQQ", names.index(b".dynamic"), 6, 2, 0, 64, len(dynamic), 2, 0, 8, 16)
    sections += struct.pack(
        "<IIQQQQIIQQ", names.index(b".dynstr"), 3, 2, 0, 64 + len(dynamic), len(strings), 0, 0, 1, 0
    )
    sections += struct.pack(
        "<IIQQQQIIQQ",
        names.index(b".shstrtab"),
        3,
        0,
        0,
        64 + len(dynamic) + len(strings),
        len(names),
        0,
        0,
        1,
        0,
    )
    return header + dynamic + strings + names + sections


@pytest.mark.parametrize("arch,machine", [("x64", 62), ("arm64", 183)])
def test_verify_linux_elf_payload(tmp_path, arch, machine):
    wheel_path = tmp_path / "core.whl"
    with ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(verify_core_wheel.CORE_PATH, make_elf(machine, ("libc.so.6",)))
        wheel.writestr("onnxruntime_genai_core.libs/libcurl-123abc.so.4", make_elf(machine, ("libc.so.6",)))
    verify_core_wheel.verify_elf_payload(wheel_path, arch)


@pytest.mark.parametrize("path", [verify_core_wheel.CORE_PATH, "onnxruntime_genai_core.libs/libsupport-123abc.so.1"])
def test_verify_linux_elf_rejects_wrong_architecture(tmp_path, path):
    wheel_path = tmp_path / "core.whl"
    with ZipFile(wheel_path, "w") as wheel:
        if path != verify_core_wheel.CORE_PATH:
            wheel.writestr(verify_core_wheel.CORE_PATH, make_elf(183))
        wheel.writestr(path, make_elf(62))
    with pytest.raises(ValueError, match="Unexpected ELF architecture"):
        verify_core_wheel.verify_elf_payload(wheel_path, "arm64")


@pytest.mark.parametrize(
    "dependency",
    [
        "libonnxruntime.so.1",
        "libonnxruntime-genai-cuda.so",
        "libonnxruntime_providers_cuda.so",
        "libcuda.so.1",
        "libcudart.so.13",
        "libcudart-123abc.so.13",
        "libcublas.so.13",
        "libnvrtc.so.13",
    ],
)
def test_verify_linux_elf_rejects_runtime_dependencies(tmp_path, dependency):
    wheel_path = tmp_path / "core.whl"
    with ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(verify_core_wheel.CORE_PATH, make_elf(183, (dependency,)))
    with pytest.raises(ValueError, match="Runtime-managed DT_NEEDED"):
        verify_core_wheel.verify_elf_payload(wheel_path, "arm64")


def test_verify_linux_elf_rejects_bundled_cuda_library(tmp_path):
    wheel_path = tmp_path / "core.whl"
    with ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(verify_core_wheel.CORE_PATH, make_elf())
        wheel.writestr("onnxruntime_genai_core.libs/libcudart-123abc.so.13", make_elf())
    with pytest.raises(ValueError, match="bundles a runtime-managed library"):
        verify_core_wheel.verify_elf_payload(wheel_path, "x64")


def test_verify_native_loading_uses_installed_wheel_and_explicit_ort(tmp_path, monkeypatch):
    ort_home = tmp_path / "ort"
    (ort_home / "lib").mkdir(parents=True)
    runtime = ort_home / "lib/libonnxruntime.so.1"
    runtime.write_bytes(b"runtime")
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))

    monkeypatch.setattr(verify_core_wheel.subprocess, "run", run)
    wheel_path = tmp_path / "core.whl"
    verify_core_wheel.verify_native_loading(wheel_path, ort_home)
    install, load = commands
    assert install[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert "--no-index" in install[0] and "--no-deps" in install[0]
    assert install[0][-1] == str(wheel_path)
    assert install[1]["check"] is True
    assert load[0][:3] == [sys.executable, "-I", "-c"]
    assert load[1]["env"]["ORT_LIB_PATH"] == str(runtime.resolve())
    assert load[1]["check"] is True
    assert not Path(load[0][-1]).exists()


def test_verify_native_loading_propagates_load_failure(tmp_path, monkeypatch):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/libonnxruntime.so").write_bytes(b"runtime")

    def run(command, **kwargs):
        if "-I" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(verify_core_wheel.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        verify_core_wheel.verify_native_loading(tmp_path / "core.whl", tmp_path)


@pytest.mark.parametrize("valid", [False, True])
def test_verify_cli_checks_elf_before_loading(tmp_path, monkeypatch, valid):
    wheel_path = tmp_path / "onnxruntime_genai_core-0.15.0-py3-none-manylinux_2_34_aarch64.whl"
    with ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(verify_core_wheel.CORE_PATH, make_elf(183 if valid else 62))
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_core_wheel.py", "--wheel_dir", str(tmp_path), "--arch", "arm64", "--ort_home", str(tmp_path / "ort")],
    )
    loads = []
    monkeypatch.setattr(verify_core_wheel, "verify_native_loading", lambda *args: loads.append(args))
    if valid:
        verify_core_wheel.main()
        assert loads == [(wheel_path, tmp_path / "ort")]
    else:
        with pytest.raises(ValueError, match="Unexpected ELF architecture"):
            verify_core_wheel.main()
        assert loads == []
