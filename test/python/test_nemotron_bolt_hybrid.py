# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import json
import shutil
from pathlib import Path

import onnx_ir as ir
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "test" / "models" / "engine" / "synthetic-bolt-hybrid" / "model.onnx"


@pytest.fixture
def converter():
    spec = importlib.util.spec_from_file_location(
        "nemotron_bolt_hybrid", ROOT / "examples" / "python" / "nemotron-bolt-hybrid.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def packed_package(tmp_path):
    source = tmp_path / "packed-r8"
    source.mkdir()
    shutil.copy2(FIXTURE, source / "model.onnx")
    (source / "weights.bin").write_bytes(bytes(range(128)))
    (source / "tokenizer.json").write_text("{}")
    config = {
        "model": {"type": "nemotron_bolt", "decoder": {"filename": "model.onnx"}},
        "speculative": {"max_draft_tokens": 7},
    }
    (source / "genai_config.json").write_text(json.dumps(config))
    return source


@pytest.mark.parametrize("destination", ["sibling", "nested", "deeply_nested"])
def test_fresh_output_preserves_package(converter, packed_package, destination):
    outputs = {
        "sibling": packed_package.parent / "hybrid",
        "nested": packed_package / "hybrid",
        "deeply_nested": packed_package / "outputs" / "hybrid",
    }
    output = outputs[destination]
    original_config = (packed_package / "genai_config.json").read_bytes()
    original_model = (packed_package / "model.onnx").read_bytes()

    converter.convert(packed_package, output)

    assert {path.name for path in output.iterdir()} == {
        "genai_config.json",
        "model.onnx",
        "weights.bin",
        "tokenizer.json",
    }
    assert (output / "weights.bin").samefile(packed_package / "weights.bin")
    assert (output / "weights.bin").read_bytes() == bytes(range(128))
    assert (output / "tokenizer.json").read_bytes() == (packed_package / "tokenizer.json").read_bytes()
    assert json.loads((output / "genai_config.json").read_text())["speculative"]["max_draft_tokens"] == 0
    assert (packed_package / "genai_config.json").read_bytes() == original_config
    assert (packed_package / "model.onnx").read_bytes() == original_model

    model = ir.load(output / "model.onnx")
    assert list(model.graph.inputs[-1].shape) == ["target_rows"]
    for value in [*model.graph.inputs, *model.graph.outputs]:
        if value.name in ("input_ids", "logits", "pred_ids", "aux_hidden_states"):
            assert list(value.shape)[:2] == [1, "target_rows"]
    operations = [node.op_type for node in model.graph]
    assert operations.count("GroupQueryAttention") == 8
    assert operations.count("Concat") == 24
    assert operations.count("Slice") == 24
    assert operations.count("Gather") == 8
    assert isinstance(model.graph.initializers["weight"].const_value, ir.ExternalTensor)


def test_existing_output_is_rejected(converter, packed_package):
    output = packed_package.parent / "hybrid"
    output.mkdir()
    with pytest.raises(ValueError, match="Output must be a fresh directory"):
        converter.convert(packed_package, output)


def test_nonflat_input_is_rejected_before_creating_output(converter, packed_package):
    (packed_package / "existing").mkdir()
    output = packed_package.parent / "hybrid"
    with pytest.raises(ValueError, match="Expected a flat packed package"):
        converter.convert(packed_package, output)
    assert not output.exists()


def test_missing_speculative_and_small_file_copy_fallback(converter, packed_package, monkeypatch):
    config_path = packed_package / "genai_config.json"
    config = json.loads(config_path.read_text())
    del config["speculative"]
    config_path.write_text(json.dumps(config))

    def no_hardlinks(*args):
        raise OSError("hardlinks unavailable")

    monkeypatch.setattr(converter.os, "link", no_hardlinks)
    output = packed_package / "outputs" / "hybrid"
    converter.convert(packed_package, output)
    assert (output / "weights.bin").read_bytes() == (packed_package / "weights.bin").read_bytes()
    assert not (output / "weights.bin").samefile(packed_package / "weights.bin")
    assert json.loads((output / "genai_config.json").read_text())["speculative"]["max_draft_tokens"] == 0


def test_static_width_through_identity_precedes_producer_attribute(converter, packed_package):
    path = packed_package / "model.onnx"
    model = ir.load(path)
    attention = next(node for node in model.graph if node.op_type == "GroupQueryAttention")
    for index in range(3):
        original = attention.inputs[index]
        width = original.producer().attributes["N"].as_int()
        value = ir.Value(name=f"static_{index}", type=original.type, shape=ir.Shape([1, 8, width]))
        identity = ir.Node("", "Identity", [original], outputs=[value])
        model.graph.insert_before(attention, identity)
        attention.replace_input_with(index, value)
    ir.save(model, path)
    output = packed_package / "hybrid"
    converter.convert(packed_package, output)
    assert (output / "model.onnx").is_file()


@pytest.mark.parametrize("attribute", [None, 0, -1, "not-an-integer", "no-producer"])
def test_unknown_width_has_descriptive_error(converter, packed_package, attribute):
    path = packed_package / "model.onnx"
    model = ir.load(path)
    attention = next(node for node in model.graph if node.op_type == "GroupQueryAttention")
    original = attention.inputs[0]
    original.shape = ir.Shape([1, 8, "unknown_width"])
    producer = original.producer()
    if attribute == "no-producer":
        value = ir.Value(name="unknown_input", type=original.type, shape=original.shape)
        model.graph.inputs.append(value)
        attention.replace_input_with(0, value)
    elif attribute is None:
        del producer.attributes["N"]
    elif isinstance(attribute, int):
        producer.attributes["N"] = ir.AttrInt64("N", attribute)
    else:
        producer.attributes["N"] = ir.AttrString("N", attribute)
    ir.save(model, path)
    output = packed_package / "outputs" / "hybrid"
    with pytest.raises(ValueError, match="positive Q/K/V width"):
        converter.convert(packed_package, output)
    assert not output.parent.exists()


@pytest.mark.parametrize("failure", ["large_link", "save", "verify", "config"])
def test_failed_nested_conversion_is_clean_and_retryable(converter, packed_package, monkeypatch, failure):
    output = packed_package / "outputs" / "hybrid"
    original_entries = {path.name for path in packed_package.iterdir()}
    original_model = (packed_package / "model.onnx").read_bytes()
    original_config = (packed_package / "genai_config.json").read_bytes()
    save, load, write = converter.ir.save, converter.ir.load, Path.write_text
    with monkeypatch.context() as patch:
        if failure == "large_link":
            with (packed_package / "weights.bin").open("r+b") as weights:
                weights.truncate(64 << 20)

            def no_hardlinks(*args):
                raise OSError("injected link failure")

            def no_large_copies(source, destination):
                assert source.stat().st_size < 64 << 20
                return shutil.copyfile(source, destination)

            patch.setattr(converter.os, "link", no_hardlinks)
            patch.setattr(converter.shutil, "copy2", no_large_copies)
        elif failure == "save":

            def fail_save(*args):
                save(*args)
                raise RuntimeError("injected save failure")

            patch.setattr(converter.ir, "save", fail_save)
        elif failure == "verify":

            def fail_verification(path):
                if path.parent != packed_package:
                    raise RuntimeError("injected verification failure")
                return load(path)

            patch.setattr(converter.ir, "load", fail_verification)
        else:

            def fail_config(path, *args, **kwargs):
                if path.name == "genai_config.json":
                    raise RuntimeError("injected config failure")
                return write(path, *args, **kwargs)

            patch.setattr(Path, "write_text", fail_config)
        with pytest.raises((ValueError, RuntimeError), match="(injected|Cannot hardlink)"):
            converter.convert(packed_package, output)
    assert not output.parent.exists()
    assert {path.name for path in packed_package.iterdir()} == original_entries
    assert (packed_package / "model.onnx").read_bytes() == original_model
    assert (packed_package / "genai_config.json").read_bytes() == original_config
    converter.convert(packed_package, output)
    assert (output / "weights.bin").samefile(packed_package / "weights.bin")


def test_failure_preserves_existing_parent_and_competing_output(converter, packed_package, monkeypatch):
    parent = packed_package.parent / "existing"
    parent.mkdir()
    sentinel = parent / "keep.txt"
    sentinel.write_text("unrelated")
    output = parent / "hybrid"
    save = converter.ir.save

    def competing_writer(*args):
        save(*args)
        output.mkdir()
        (output / "owned-by-other-writer").write_text("keep")

    monkeypatch.setattr(converter.ir, "save", competing_writer)
    with pytest.raises(ValueError, match="Output must be a fresh directory"):
        converter.convert(packed_package, output)
    assert sentinel.read_text() == "unrelated"
    assert {path.name for path in parent.iterdir()} == {"keep.txt", "hybrid"}
    assert {path.name for path in output.iterdir()} == {"owned-by-other-writer"}
    assert (output / "owned-by-other-writer").read_text() == "keep"


def test_asset_symlink_keeps_packaged_filename(converter, packed_package):
    weights = packed_package / "weights.bin"
    original = packed_package.parent / "external-weight-payload"
    weights.rename(original)
    try:
        weights.symlink_to(original)
    except OSError:
        pytest.skip("File symlinks are unavailable")
    output = packed_package / "hybrid"
    converter.convert(packed_package, output)
    assert (output / "weights.bin").samefile(original)
    assert not (output / original.name).exists()
