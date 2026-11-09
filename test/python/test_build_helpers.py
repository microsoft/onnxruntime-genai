# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib.util
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("strip_install", [False, True])
def test_install_core_stripping_is_opt_in(monkeypatch, tmp_path, strip_install):
    spec = importlib.util.spec_from_file_location("genai_install_review", REPO_ROOT / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    command = ["build.py", "--install_dir", str(tmp_path / "install"), "--build_dir", str(tmp_path / "build")]
    if strip_install:
        command.append("--strip_install")
    monkeypatch.setattr(sys, "argv", command)
    args = build._parse_args()
    commands = []
    env = {"PACKAGING_TEST": "1"}
    monkeypatch.setattr(build.util, "run", lambda command, **kwargs: commands.append((command, kwargs)))
    build.install_core(args, env)
    expected = [
        str(args.cmake_path),
        "--install",
        str(args.build_dir),
        "--config",
        args.config,
        "--prefix",
        str(args.install_dir),
    ]
    if strip_install:
        expected.append("--strip")
    assert commands == [(expected, {"env": env})]


@pytest.fixture
def android_stripping_tools(tmp_path):
    compiler = shutil.which("cc")
    strip = shutil.which("llvm-strip")
    if sys.platform != "linux" or compiler is None or strip is None:
        pytest.skip("Linux compiler and LLVM tools are needed to test ELF stripping")
    strip = Path(strip)
    if not strip.with_name("llvm-readelf").is_file():
        pytest.skip("llvm-readelf is needed to test ELF stripping")
    spec = importlib.util.spec_from_file_location(
        "genai_android_stripping_review", REPO_ROOT / "tools/ci_build/github/android/build_aar_package.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    def make_library(name, alignment):
        library = tmp_path / name
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                "-g",
                f"-Wl,-z,max-page-size={alignment}",
                "-x",
                "c",
                "-",
                "-o",
                str(library),
            ],
            input="int packaging_test(void) { return 0; }\n",
            text=True,
            capture_output=True,
            check=True,
        )
        return library

    return builder, strip, make_library


@pytest.mark.parametrize("alignment", [4096, 16384])
def test_android_stripping_preserves_load_segments_and_originals(android_stripping_tools, tmp_path, alignment):
    builder, strip, make_library = android_stripping_tools
    source = make_library("source.so", alignment)
    original = source.read_bytes()
    destination = tmp_path / "staged.so"
    builder._strip_library(source, destination, strip)
    assert source.read_bytes() == original
    assert destination.stat().st_size < source.stat().st_size


def test_android_stripping_rejects_changed_load_segments(android_stripping_tools, monkeypatch, tmp_path):
    builder, strip, make_library = android_stripping_tools
    source = make_library("source.so", 16384)
    replacement = make_library("misaligned.so", 4096)
    original = source.read_bytes()
    run = subprocess.run

    def change_alignment(command, **kwargs):
        if Path(command[0]) == strip:
            shutil.copy2(replacement, command[1])
            return subprocess.CompletedProcess(command, 0)
        return run(command, **kwargs)

    monkeypatch.setattr(builder.subprocess, "run", change_alignment)
    with pytest.raises(RuntimeError, match="Stripping changed LOAD segments"):
        builder._strip_library(source, tmp_path / "staged.so", strip)
    assert source.read_bytes() == original


@pytest.mark.parametrize("use_cuda", [False, True])
@pytest.mark.parametrize(
    "definitions,enabled",
    [
        ([], True),
        (["ENABLE_MSVC_SPECTRE=OFF"], False),
        (["-DENABLE_MSVC_SPECTRE:BOOL=FALSE"], False),
        (["ENABLE_MSVC_SPECTRE=0"], False),
        (["ENABLE_MSVC_SPECTRE=OFF", "ENABLE_MSVC_SPECTRE=ON"], True),
    ],
)
def test_packaged_windows_flags_honor_the_spectre_option(use_cuda, definitions, enabled):
    spec = importlib.util.spec_from_file_location("genai_packaged_build_review", REPO_ROOT / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    args = SimpleNamespace(config="Release", use_cuda=use_cuda, cmake_extra_defines=definitions)
    flags = build._get_windows_build_args(args)
    compile_flags = [
        flag
        for flag in flags
        if flag.startswith(("-DCMAKE_C_FLAGS_INIT=", "-DCMAKE_CXX_FLAGS_INIT=", "-DCMAKE_CUDA_FLAGS_INIT="))
    ]
    assert len(compile_flags) == 2 + use_cuda
    for flag in compile_flags:
        assert ("/Qspectre" in flag) is enabled
        assert "/guard:cf" in flag


def test_examples_rebuild_uses_python311_compatible_cleanup(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("genai_build_review", REPO_ROOT / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    monkeypatch.setattr(build, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build.util, "is_windows", lambda: False)
    commands = []
    monkeypatch.setattr(build.util, "run", lambda command, **kwargs: commands.append(command))
    original_rmtree = shutil.rmtree

    def python311_rmtree(path, ignore_errors=False, onerror=None):
        return original_rmtree(path, ignore_errors=ignore_errors, onerror=onerror)

    monkeypatch.setattr(build.shutil, "rmtree", python311_rmtree)
    directory = tmp_path / "examples" / "c" / "build"
    directory.mkdir(parents=True)
    old_file = directory / "old_output"
    old_file.write_text("old")
    old_file.chmod(stat.S_IREAD)
    args = SimpleNamespace(
        cmake_path=Path("cmake"),
        cmake_generator="Unix Makefiles",
        config="Debug",
        build_dir=tmp_path / "native",
        ort_home=None,
        cmake_extra_defines=[],
    )
    build.build_examples(args, {})
    build.build_examples(args, {})
    assert directory.is_dir()
    assert not old_file.exists()
    assert len(commands) == 4


@pytest.mark.parametrize("override", [None, "OFF"])
def test_standalone_msvc_security_policy_defaults_to_spectre_mitigation(tmp_path, override):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is needed to evaluate the standalone SDK security policy")
    policy = (REPO_ROOT / "cmake" / "msvc_security.cmake").as_posix()
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(SecurityPolicyProbe LANGUAGES NONE)\n"
        "set(MSVC TRUE)\n"
        f'include("{policy}")\n'
        "get_directory_property(options COMPILE_OPTIONS)\n"
        'if(EXPECT_SPECTRE AND NOT options MATCHES "/Qspectre")\n'
        '  message(FATAL_ERROR "Standalone SDK lost /Qspectre")\n'
        'elseif(NOT EXPECT_SPECTRE AND options MATCHES "/Qspectre")\n'
        '  message(FATAL_ERROR "Explicit Spectre opt-out was ignored")\n'
        "endif()\n"
    )
    command = [
        cmake,
        "-S",
        str(tmp_path),
        "-B",
        str(tmp_path / "build"),
        f"-DEXPECT_SPECTRE={'ON' if override is None else 'OFF'}",
    ]
    if override is not None:
        command.append(f"-DENABLE_MSVC_SPECTRE={override}")
    subprocess.run(command, check=True, capture_output=True, text=True)
