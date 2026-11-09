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
