# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests of the Qwen3.8 QMoE offload package builder (``tools/qwen38_graph/build_package_offload_v3.py``).

The builder runs on tiny synthetic packages, so no GPU, checkpoint or custom-op library is needed. Released ONNX Runtime has no
QMoE offload, so the operator contract is checked against ``_qwen38_qmoe_offload_contract``, a model of the runtime of
microsoft/onnxruntime#33622, and by creating sessions in the installed onnxruntime where the emitted graph can be loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import struct
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import pytest
from _qwen38_qmoe_offload_contract import (
    QMOE_ATTRIBUTES,
    ContractError,
    extract_layer_index,
    node_is_eligible,
    parse_node_overrides,
    parse_session_entries,
    simulate_session_creation,
    validate_popularity_file,
)
from _qwen38_tools_test_utils import (
    REMOVED_QWEN38_GRAPH_KEYS,
    load_tools,
    prefill_chunk_problem,
    unknown_qwen38_graph_keys,
)
from onnx import TensorProto, helper, numpy_helper

LAYERS = 3
EXPERTS = 8
NODE_NAME = "model/layers.{0}/mlp/QMoE_node_{0}"
STRP_MAGIC = 0x50525453
PREFIX = "session.moe.offload."


@pytest.fixture(scope="module")
def offload():
    with load_tools("package_paths", "build_package_offload_v3") as (_paths, module):
        yield module


# ---------------------------------------------------------------------------
# Synthetic graphs and packages
# ---------------------------------------------------------------------------

NVFP4_ATTRIBUTES = {
    "activation_type": "swiglu",
    "swiglu_fusion": 1,
    "k": 2,
    "expert_weight_bits": 4,
    "block_size": 16,
    "quant_type": "nvfp4",
    "weights_prepacked": 1,
    "normalize_routing_weights": 1,
}
SLOT_NAMES = {
    2: "fc1_experts_weights",
    3: "fc1_scales",
    5: "fc2_experts_weights",
    6: "fc2_scales",
    15: "fc1_global_scales",
    16: "fc2_global_scales",
}


def qmoe_node(layer=0, *, name=None, attributes=None, slots=None):
    """A standard NVFP4 ``com.microsoft::QMoE`` node of ``layer``.

    ``slots`` replaces input names by slot number and ``attributes`` replaces attributes (``None`` removes one).
    """
    names = {0: f"h{layer}", 1: f"router_probs_{layer}", 14: f"router_weights_{layer}"}
    names.update({slot: f"model.layers.{layer}.mlp.{suffix}" for slot, suffix in SLOT_NAMES.items()})
    names.update(slots or {})
    inputs = [names.get(slot, "") for slot in range(max(names) + 1)]
    merged = {key: value for key, value in {**NVFP4_ATTRIBUTES, **(attributes or {})}.items() if value is not None}
    return helper.make_node(
        "QMoE",
        inputs,
        [f"h{layer + 1}"],
        name=NODE_NAME.format(layer) if name is None else name,
        domain="com.microsoft",
        **merged,
    )


def external_tensor(name, dims, dtype=TensorProto.UINT8, location="weights.data", offset=0):
    """An initializer whose bytes are in an external file that the tests never create."""
    length = int(np.prod(dims)) * (4 if dtype == TensorProto.FLOAT else 1)
    tensor = TensorProto(name=name, data_type=dtype, dims=dims, data_location=TensorProto.EXTERNAL)
    for key, value in (("location", location), ("offset", str(offset)), ("length", str(length))):
        tensor.external_data.add(key=key, value=value)
    return tensor


def qmoe_initializers(layer=0, dims=None, skip=()):
    """The external expert tensors of a node made by ``qmoe_node``; ``dims`` replaces shapes by slot number."""
    tensors = []
    for slot, suffix in SLOT_NAMES.items():
        if slot in skip:
            continue
        shape = {2: [EXPERTS, 4, 4], 5: [EXPERTS, 4, 4], 3: [EXPERTS, 4, 2], 6: [EXPERTS, 4, 2]}.get(slot, [EXPERTS])
        dtype = TensorProto.FLOAT if slot in (15, 16) else TensorProto.UINT8
        tensors.append(
            external_tensor(
                f"model.layers.{layer}.mlp.{suffix}",
                (dims or {}).get(slot, shape),
                dtype,
                offset=4096 * (layer * 20 + slot),
            )
        )
    return tensors


def make_model(nodes, initializers, ir_version=9, overridable=()):
    """A graph of ``nodes``; every input that nothing defines is a graph input, and so is each ``overridable`` initializer."""
    known = {tensor.name for tensor in initializers}
    graph_inputs = []
    for node in nodes:
        for name in node.input:
            if name and name not in known:
                known.add(name)
                graph_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT16, ["T", 4]))
        known.update(node.output)
    graph_inputs += [helper.make_tensor_value_info(name, TensorProto.UINT8, [EXPERTS, 4, 4]) for name in overridable]
    graph = helper.make_graph(
        nodes,
        "decoder",
        graph_inputs,
        [helper.make_tensor_value_info(nodes[-1].output[0], TensorProto.FLOAT16, ["T", 4])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = ir_version
    return model


def decoder_model(layers=LAYERS, **node_arguments):
    nodes = [qmoe_node(layer, **node_arguments) for layer in range(layers)]
    return make_model(nodes, [tensor for layer in range(layers) for tensor in qmoe_initializers(layer)])


def one_node_model(*, attributes=None, slots=None, dims=None, skip=(), ir_version=9, overridable=()):
    return make_model(
        [qmoe_node(0, attributes=attributes, slots=slots)],
        qmoe_initializers(0, dims, skip),
        ir_version=ir_version,
        overridable=overridable,
    )


def dense_model():
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "dense",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 4])],
        initializer=[external_tensor("dense.layers.0.weight", [4, 4], location="source_weights/dense.data")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(path), save_as_external_data=False)


def genai_config(layered):
    return {
        "model": {
            "eos_token_id": [7],
            "decoder": {
                "filename": "target/model.onnx" if layered else "model.onnx",
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [{"cuda": {"enable_cuda_graph": "0"}}],
                },
            },
            "qwen38_graph": {
                "kv_capacity": 4096,
                "fold": {"enabled": True, "head_r2": {"filename": "head_r2/model.onnx"}},
                "prefill": {
                    "enabled": True,
                    "chunk": 1024,
                    "dense": {"filename": "dense/model.onnx", "session_options": {"provider_options": [{"cuda": {}}]}},
                },
            },
        }
    }


DEFAULT_POOL = """\
qwen38_weights_pool 1
pool 0 1 0 64 2 4 4 weights.data
pool 1 16 64 64 2 4 4 weights.data
session target model.embed_tokens.weight 0
session target model.norm.weight 1
"""


def make_package(tmp_path, *, model=None, layered=False, pool_text=DEFAULT_POOL, dense=True, config=None):
    """A source package under ``tmp_path/pkg`` (flat, or with the decoder graph in ``target/``) and the two libraries."""
    root = tmp_path / "pkg"
    decoder_dir = root / "target" if layered else root
    save_model(decoder_model() if model is None else model, decoder_dir / "model.onnx")
    (root / "genai_config.json").write_text(json.dumps(genai_config(layered) if config is None else config))
    (root / "weights_pool.txt").write_text(pool_text)
    if dense:
        save_model(dense_model(), root / "dense" / "model.onnx")
    libs = tmp_path / "libs"
    libs.mkdir(exist_ok=True)
    (libs / "libqwen38_plane.so").write_bytes(b"plane")
    (libs / "libonnxruntime_providers_cuda.so").write_bytes(b"provider")
    return types.SimpleNamespace(
        root=root,
        decoder_dir=decoder_dir,
        plane=libs / "libqwen38_plane.so",
        provider=libs / "libonnxruntime_providers_cuda.so",
    )


def write_profile(path, pairs, layers=LAYERS, experts=EXPERTS, magic=STRP_MAGIC, version=1, ranked=None):
    """An STRP v1 expert-popularity file ranking the (layer, expert) ``pairs`` from the hottest down."""
    header = struct.pack(
        "<6I", magic, version, layers, experts, layers * experts, len(pairs) if ranked is None else ranked
    )
    path.write_bytes(header + b"".join(struct.pack("<2H", layer, expert) for layer, expert in pairs))
    return path


def exhaustive_pairs(layers=LAYERS, experts=EXPERTS, seed=0):
    """Every (layer, expert) pair in a seeded random popularity order."""
    pairs = [(layer, expert) for layer in range(layers) for expert in range(experts)]
    random.Random(seed).shuffle(pairs)
    return pairs


def run_main(monkeypatch, module, *argv):
    monkeypatch.setattr(sys, "argv", [f"{module.__name__}.py", *map(str, argv)])
    return module.main()


def run_builder(monkeypatch, offload, package, out, *extra):
    return run_main(
        monkeypatch,
        offload,
        "--from-package",
        package.root,
        "--out",
        out,
        "--plane-lib",
        package.plane,
        "--provider-lib",
        package.provider,
        *extra,
    )


@pytest.fixture
def multidraft_calls(offload, monkeypatch):
    """Replaces the multi-draft graph rewrite (which needs the real folded graph) with a copy, and records its calls."""
    calls = []

    def copy_graph(in_onnx, out_onnx, verify_rows=3):
        calls.append((Path(in_onnx).name, Path(out_onnx).name, verify_rows))
        shutil.copyfile(in_onnx, out_onnx)
        return {"verify_rows": verify_rows}

    monkeypatch.setattr(offload, "build_multidraft_target_onnx", copy_graph)
    return calls


def snapshot(root):
    """Every path below ``root`` with the SHA-256 of its content (``<dir>`` for folders)."""
    return {
        path.relative_to(root).as_posix(): "<dir>" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
    }


def offload_entries(config):
    return {
        key: value for key, value in config["model"]["decoder"]["session_options"].items() if key.startswith(PREFIX)
    }


def read_output(out, layered):
    """The output config, the offload entries of its decoder, and the decoder graph with its directory."""
    config = json.loads((out / "genai_config.json").read_text())
    decoder_dir = out / "target" if layered else out
    return (
        config,
        offload_entries(config),
        onnx.load(str(decoder_dir / "model.onnx"), load_external_data=False),
        decoder_dir,
    )


def qmoe_nodes_of(model):
    return [node for node in model.graph.node if node.op_type == "QMoE"]


# ---------------------------------------------------------------------------
# The operator contract: standard QMoE and session.moe.offload.* entries
# ---------------------------------------------------------------------------


def test_contract_model_rejects_what_the_reference_test_rejects():
    """The failures of test_discarded_prototype_features_rejected in onnxruntime#33622 (test_qmoe_offload_v4.py)."""
    model = decoder_model(2)
    with pytest.raises(ContractError, match="Unknown session.moe.offload"):
        simulate_session_creation(model, {PREFIX + "unknown_knob": "1"})
    with pytest.raises(ContractError, match="discarded and not supported"):
        simulate_session_creation(model, {"moe.cpu_offload.experts": "4"})
    with pytest.raises(ContractError, match="expected 'static' or 'adaptive'"):
        simulate_session_creation(model, {PREFIX + "mode": "required", PREFIX + "placement": "basic"})
    legacy = decoder_model(2)
    legacy.graph.node[0].op_type = "QMoEOffload"
    with pytest.raises(ContractError, match="QMoEOffload"):
        simulate_session_creation(legacy, {})
    prototype = decoder_model(2)
    prototype.graph.node[0].attribute.append(helper.make_attribute("layer_index", 0))
    with pytest.raises(ContractError, match="Unrecognized attribute: layer_index"):
        simulate_session_creation(prototype, {})


def test_contract_model_follows_the_rest_of_the_reference_test(offload, tmp_path):
    model = decoder_model(2)
    offload.prepare_qmoe_for_offload(model)
    required = {PREFIX + "mode": "required"}
    assert simulate_session_creation(model, required, tmp_path) == [NODE_NAME.format(0), NODE_NAME.format(1)]
    with pytest.raises(ContractError, match="popularity_file"):
        simulate_session_creation(model, {**required, PREFIX + "popularity_file": "missing.bin"}, tmp_path)
    with pytest.raises(ContractError, match="node_overrides"):
        simulate_session_creation(model, {**required, PREFIX + "node_overrides": "no_colon"}, tmp_path)
    override = {**required, PREFIX + "node_overrides": "model/layers.0/mlp/QMoE_node_0:max_resident_slots=6"}
    assert simulate_session_creation(model, override, tmp_path)
    unknown = {**required, PREFIX + "node_overrides": "no_such_node:max_resident_slots=6"}
    with pytest.raises(ContractError, match="unknown node 'no_such_node'"):
        simulate_session_creation(model, unknown, tmp_path)
    model.graph.node[0].attribute[[a.name for a in model.graph.node[0].attribute].index("swiglu_fusion")].i = 2
    with pytest.raises(ContractError, match="rejected node"):
        simulate_session_creation(model, required, tmp_path)
    assert simulate_session_creation(model, {PREFIX + "mode": "auto"}, tmp_path) == [NODE_NAME.format(1)]


# (id, offloadable by the runtime, one_node_model arguments)
INT_ATTRIBUTES = {"quant_type": "int", "block_size": 32, "weights_prepacked": None}
NO_GLOBAL_SCALES = {"skip": (15, 16), "slots": {15: "", 16: ""}}
ELIGIBILITY = [
    ("nvfp4", True, {}),
    ("nvfp4 default block_size", True, {"attributes": {"block_size": None}}),
    ("nvfp4 swiglu_fusion 0", True, {"attributes": {"swiglu_fusion": 0}}),
    ("nvfp4 swiglu_fusion 2", False, {"attributes": {"swiglu_fusion": 2}}),
    ("nvfp4 weights_prepacked 0", False, {"attributes": {"weights_prepacked": 0}}),
    ("nvfp4 default weights_prepacked", False, {"attributes": {"weights_prepacked": None}}),
    ("nvfp4 relu", False, {"attributes": {"activation_type": "relu"}}),
    ("nvfp4 default activation", False, {"attributes": {"activation_type": None}}),
    ("nvfp4 sparse mixer", False, {"attributes": {"use_sparse_mixer": 1}}),
    ("nvfp4 8-bit", False, {"attributes": {"expert_weight_bits": 8}}),
    ("nvfp4 fc3 weights", False, {"slots": {8: "fc3_experts_weights"}}),
    ("nvfp4 fc1 zero points", False, {"slots": {11: "fc1_zero_points"}}),
    ("nvfp4 activation scales", False, {"slots": {17: "fc1_activation_scale"}}),
    ("nvfp4 no global scales", False, NO_GLOBAL_SCALES),
    ("nvfp4 one global scale", False, {"skip": (16,), "slots": {16: ""}}),
    ("nvfp4 fc1 weights are an input", False, {"slots": {2: "fc1_weights_input"}}),
    ("nvfp4 no fc2 scales", False, {"skip": (6,), "slots": {6: ""}}),
    ("nvfp4 fc1 weights rank 2", False, {"dims": {2: [EXPERTS, 4]}}),
    ("nvfp4 fc2 weights rank 4", False, {"dims": {5: [EXPERTS, 4, 4, 1]}}),
    ("nvfp4 overridable fc1 weights", False, {"overridable": ["model.layers.0.mlp.fc1_experts_weights"]}),
    (
        "nvfp4 overridable fc1 weights, IR 3",
        True,
        {"overridable": ["model.layers.0.mlp.fc1_experts_weights"], "ir_version": 3},
    ),
    ("int4 block 32", True, {"attributes": INT_ATTRIBUTES, **NO_GLOBAL_SCALES}),
    ("int4 block 32 prepacked 1", True, {"attributes": {**INT_ATTRIBUTES, "weights_prepacked": 1}, **NO_GLOBAL_SCALES}),
    ("int4 block 32 with unused global scales", True, {"attributes": INT_ATTRIBUTES}),
    ("int4 block 64", False, {"attributes": {**INT_ATTRIBUTES, "block_size": 64}, **NO_GLOBAL_SCALES}),
    ("int4 default block_size", False, {"attributes": {**INT_ATTRIBUTES, "block_size": None}, **NO_GLOBAL_SCALES}),
    ("int4 weights_prepacked 0", False, {"attributes": {**INT_ATTRIBUTES, "weights_prepacked": 0}, **NO_GLOBAL_SCALES}),
    ("int8", False, {"attributes": {**INT_ATTRIBUTES, "expert_weight_bits": 8}, **NO_GLOBAL_SCALES}),
    (
        "int3 raw block 64",
        True,
        {
            "attributes": {**INT_ATTRIBUTES, "expert_weight_bits": 3, "block_size": 64, "weights_prepacked": 0},
            **NO_GLOBAL_SCALES,
        },
    ),
    (
        "int3 prepacked",
        False,
        {"attributes": {**INT_ATTRIBUTES, "expert_weight_bits": 3, "block_size": 64}, **NO_GLOBAL_SCALES},
    ),
    ("mxfp4", False, {"attributes": {"quant_type": "fp4", "block_size": 32}, **NO_GLOBAL_SCALES}),
]


@pytest.mark.parametrize(("expected", "case"), [pytest.param(*case[1:], id=case[0]) for case in ELIGIBILITY])
def test_qmoe_eligibility_matches_the_runtime(offload, expected, case):
    model = one_node_model(**case)
    report = offload.prepare_qmoe_for_offload(model)
    assert (not report["rejected"]) == expected
    assert node_is_eligible(model, model.graph.node[0]) == expected


def test_nvfp4_block_size_the_runtime_refuses_is_reported(offload):
    model = one_node_model(attributes={"block_size": 32})
    with pytest.raises(ContractError, match="requires block_size=16"):
        node_is_eligible(model, model.graph.node[0])
    reasons = offload.prepare_qmoe_for_offload(model)["rejected"][0]
    assert any("block_size=32" in reason for reason in reasons)


def test_router_weights_make_a_node_ineligible_until_the_builder_clears_them(offload):
    model = one_node_model()
    assert not node_is_eligible(model, model.graph.node[0])
    report = offload.prepare_qmoe_for_offload(model)
    assert report["router_weights_cleared"] == 1
    assert not report["rejected"]
    assert node_is_eligible(model, model.graph.node[0])


def test_rejection_reasons_name_the_defect(offload):
    for case, fragment in (
        ({"slots": {8: "fc3_experts_weights", 17: "scale"}}, "inputs [8, 17]"),
        ({"attributes": {"activation_type": "relu"}}, "activation_type='relu'"),
        ({"dims": {2: [EXPERTS, 4]}}, "input 2 has rank 2"),
        (NO_GLOBAL_SCALES, "global scales [15, 16]"),
        ({"slots": {3: "fc1_scales_input"}}, "inputs [3]"),
        ({"attributes": {"weights_prepacked": 0}}, "weights_prepacked=0"),
    ):
        rejected = offload.prepare_qmoe_for_offload(one_node_model(**case))["rejected"]
        assert any(fragment in reason for reason in rejected[0]), (case, rejected)


# ---------------------------------------------------------------------------
# prepare_qmoe_for_offload, node overrides, session entries
# ---------------------------------------------------------------------------


def test_prepare_keeps_standard_qmoe_and_clears_only_router_weights(offload):
    model = decoder_model()
    before = onnx.ModelProto()
    before.CopyFrom(model)
    report = offload.prepare_qmoe_for_offload(model)
    assert report == {
        "nodes": [NODE_NAME.format(layer) for layer in range(LAYERS)],
        "router_weights_cleared": LAYERS,
        "rejected": {},
    }
    for old, new in zip(before.graph.node, model.graph.node, strict=True):
        assert (new.op_type, new.domain, new.name) == ("QMoE", "com.microsoft", old.name)
        assert [a.SerializeToString() for a in new.attribute] == [a.SerializeToString() for a in old.attribute]
        assert {a.name for a in new.attribute} <= QMOE_ATTRIBUTES
        assert list(new.input) == [*old.input[:14], "", *old.input[15:]]
        assert list(new.output) == list(old.output)
    assert list(model.graph.initializer) == list(before.graph.initializer)


def test_prepare_handles_graphs_without_router_weights_or_qmoe(offload):
    model = make_model([qmoe_node(0, slots={14: ""})], qmoe_initializers(0))
    assert offload.prepare_qmoe_for_offload(model)["router_weights_cleared"] == 0
    identity = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([identity], "g", [], [], initializer=[])
    report = offload.prepare_qmoe_for_offload(helper.make_model(graph))
    assert report == {"nodes": [], "router_weights_cleared": 0, "rejected": {}}


def test_prepare_reports_unnamed_and_rejected_nodes_by_position(offload):
    nodes = [qmoe_node(0), qmoe_node(1, name="", attributes={"swiglu_fusion": 2})]
    report = offload.prepare_qmoe_for_offload(make_model(nodes, qmoe_initializers(0) + qmoe_initializers(1)))
    assert report["nodes"] == [NODE_NAME.format(0), ""]
    assert list(report["rejected"]) == [1]
    assert offload.node_label("", 1) == "#1 (unnamed)"
    assert offload.node_label("a b", 1) == "'a b'"


@pytest.mark.parametrize(
    "name",
    [
        "model/layers.0/mlp/QMoE_node_0",
        "model.layers.12.mlp",
        "has space",
        "equals=sign",
        "comma,comma",
        "{braces}",
        "{}",
        "ünïcode/日本",
        "\u00a0nbsp",
        "0",
        "a" * 500,
    ],
)
def test_node_overrides_round_trip_through_the_runtime_parser(offload, name):
    value = offload.format_node_overrides({name: 5, "other": 0})
    parsed = parse_node_overrides(value)
    assert [(o["name"], o["max_resident_slots"], o["device_expert_bytes"]) for o in parsed] == [
        (name, 5, None),
        ("other", 0, None),
    ]


def test_node_overrides_round_trip_for_random_names(offload):
    rng = random.Random(7)
    alphabet = "ab/._-= ,{}()[]éß日09\t"
    for _ in range(300):
        name = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 14)))
        if name != name.strip(" \t\n\v\f\r"):
            continue
        cap = rng.choice([0, 1, 479, 2**31 - 1])
        assert [
            (o["name"], o["max_resident_slots"])
            for o in parse_node_overrides(offload.format_node_overrides({name: cap}))
        ] == [(name, cap)]


@pytest.mark.parametrize("name", ["", " leading", "trailing ", "\ttab", "semi;colon", "co:lon"])
def test_node_overrides_refuse_names_the_runtime_would_split(offload, name):
    with pytest.raises(ValueError, match="cannot appear in session.moe.offload.node_overrides"):
        offload.format_node_overrides({name: 4})


@pytest.mark.parametrize("cap", [-1, 2**31, 2**40])
def test_node_overrides_refuse_caps_outside_the_runtime_range(offload, cap):
    with pytest.raises(ValueError, match="outside 0"):
        offload.format_node_overrides({"node": cap})


def test_node_overrides_of_no_caps_is_empty_and_valid(offload):
    assert offload.format_node_overrides({}) == ""
    assert parse_node_overrides("") == []


@pytest.mark.parametrize("mode", ["required", "auto"])
@pytest.mark.parametrize("placement", ["adaptive", "static"])
def test_session_options_are_accepted_by_the_runtime_parser(offload, tmp_path, mode, placement):
    minimal = offload.offload_session_options(mode, placement)
    assert minimal == {
        PREFIX + "mode": mode,
        PREFIX + "placement": placement,
        PREFIX + "cpu_execution": "auto",
    }
    entries = offload.offload_session_options(
        mode, placement, "auto", "123456", "expert-profile.bin", "node:max_resident_slots=4"
    )
    assert all(key.startswith(PREFIX) and isinstance(value, str) for key, value in entries.items())
    for options in (minimal, entries):
        policy = parse_session_entries(options, tmp_path)
        assert (policy["mode"], policy["placement"], policy["cpu_execution"]) == (mode, placement, "auto")
    assert policy["device_expert_bytes"] is None
    assert policy["host_expert_bytes"] == 123456
    assert policy["popularity_file"] == tmp_path / "expert-profile.bin"
    assert [o["name"] for o in policy["node_overrides"]] == ["node"]


@pytest.mark.parametrize(("mode", "placement"), [("disabled", "adaptive"), ("Required", "static"), ("auto", "basic")])
def test_session_options_refuse_what_the_builder_does_not_support(offload, mode, placement):
    with pytest.raises(ValueError, match="offload (mode|placement)"):
        offload.offload_session_options(mode, placement)


def test_budget_bytes_accepts_what_the_runtime_parses(offload):
    for text, expected in (
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("0", "0"),
        ("007", "7"),
        (str(2**64 - 1), str(2**64 - 1)),
    ):
        assert offload.budget_bytes(text) == expected
        assert parse_session_entries({PREFIX + "device_expert_bytes": expected})
    for text in ("", " 5", "-1", "1.5", "0x10", "1e3", "١٢٣", str(2**64), "autos"):
        with pytest.raises(argparse.ArgumentTypeError):
            offload.budget_bytes(text)
    with pytest.raises(ContractError):
        parse_session_entries({PREFIX + "device_expert_bytes": str(2**64)})


# ---------------------------------------------------------------------------
# Layer numbers
# ---------------------------------------------------------------------------

LAYER_NAMES = [
    ("model.layers.5.mlp", 5),
    ("model/layers.12/mlp/QMoE_node_12", 12),
    ("layers.7", 7),
    ("layers.7a", None),
    ("layers.7a.layers.8", 8),
    ("layer.3.x", 3),
    ("blocks.9.", 9),
    ("layers.", None),
    ("layers.2147483647", 2147483647),
    ("layers.2147483648", None),
    ("layers.99999999999999999999", None),
    ("xlayers.3", 3),
    ("layers.1layers.2", 2),
    ("layers.4é", 4),
    ("blocks.3.layers.4", 4),
    ("layer.3.layers.4", 4),
    ("QMoE_node_3", None),
    ("", None),
]


@pytest.mark.parametrize(("name", "expected"), LAYER_NAMES)
def test_layer_numbers_are_read_like_the_runtime_reads_them(offload, name, expected):
    assert offload.parse_layer_index(name) == expected
    assert extract_layer_index(name) == expected


def test_layer_numbers_agree_with_the_runtime_for_random_names(offload):
    rng = random.Random(11)
    tokens = ["layers.", "layer.", "blocks.", "7", "12", "a", ".", "/", "_", "é", "99999999999", "0"]
    for _ in range(2000):
        name = "".join(rng.choice(tokens) for _ in range(rng.randint(1, 9)))
        assert offload.parse_layer_index(name) == extract_layer_index(name), name


def numbered_model(names, fc1_names=None):
    nodes = [
        qmoe_node(layer, name=name, slots={2: (fc1_names or {}).get(layer, f"w{layer}")})
        for layer, name in enumerate(names)
    ]
    return make_model(nodes, [])


def test_layer_numbering_accepts_names_that_match_the_graph_order(offload):
    offload.check_layer_numbering(numbered_model([f"model/layers.{i}/mlp" for i in range(3)]))
    offload.check_layer_numbering(numbered_model(["a", "b", "c"]))
    offload.check_layer_numbering(numbered_model(["", "", ""]))
    offload.check_layer_numbering(
        numbered_model(["a", "b"], {0: "model.layers.0.mlp.fc1", 1: "model.layers.1.mlp.fc1"})
    )


@pytest.mark.parametrize(
    ("names", "fc1_names", "not_offloaded", "message"),
    [
        (["model/layers.1/mlp", "model/layers.0/mlp"], None, frozenset(), "is number 0 in the graph but .* layer 1"),
        (["model/layers.48/mlp"], None, frozenset(), "numbers it layer 48"),
        (["a", "b"], {1: "layers.0.fc1"}, frozenset(), "'b' is number 1 .* layer 0"),
        (["a", "b"], None, frozenset({0}), "'b' is number 1 .* layer 0"),
        (["", ""], None, frozenset({0}), "#1 \\(unnamed\\) is number 1 .* layer 0"),
    ],
)
def test_layer_numbering_refuses_nodes_the_runtime_would_number_differently(
    offload, names, fc1_names, not_offloaded, message
):
    with pytest.raises(ValueError, match=message):
        offload.check_layer_numbering(numbered_model(names, fc1_names), not_offloaded)


# ---------------------------------------------------------------------------
# Resident-slot plans: every requested total is met or refused
# ---------------------------------------------------------------------------


def ranking_by_layer(pairs):
    """{layer: experts in popularity order}, the first mention of a pair counting."""
    ranked = {}
    for layer, expert in pairs:
        if expert not in ranked.setdefault(layer, []):
            ranked[layer].append(expert)
    return ranked


def check_plan(plan, pairs, total):
    """The plan has ``total`` slots and every layer keeps a prefix of its own popularity ranking."""
    ranked = ranking_by_layer(pairs)
    assert sum(len(experts) for experts in plan.values()) == total
    for layer, experts in plan.items():
        assert experts == ranked.get(layer, [])[: len(experts)], layer


@pytest.mark.parametrize("uniform", [False, True])
def test_exhaustive_profile_plans_meet_the_requested_total(offload, tmp_path, uniform):
    pairs = exhaustive_pairs(48, 512)
    profile = write_profile(tmp_path / "expert-profile.bin", pairs, 48, 512)
    for total in (961, 47, 960, 959, 1, 0, 100, 1000, 48 * 512 - 1, 48 * 512):
        plan = offload.load_expert_profile_plan(profile, 48, 512, total, uniform=uniform)
        check_plan(plan, pairs, total)
        if uniform:
            sizes = [len(plan[layer]) for layer in range(48)]
            assert max(sizes) - min(sizes) <= 1, total


def test_uniform_plan_of_961_slots_gives_one_layer_the_extra_slot(offload, tmp_path):
    pairs = exhaustive_pairs(48, 512)
    profile = write_profile(tmp_path / "expert-profile.bin", pairs, 48, 512)
    plan = offload.load_expert_profile_plan(profile, 48, 512, 961, uniform=True)
    sizes = sorted(len(experts) for experts in plan.values())
    assert sizes == [20] * 47 + [21]


def test_uniform_plan_of_47_slots_leaves_the_coldest_layer_empty(offload, tmp_path):
    pairs = exhaustive_pairs(48, 512)
    profile = write_profile(tmp_path / "expert-profile.bin", pairs, 48, 512)
    plan = offload.load_expert_profile_plan(profile, 48, 512, 47, uniform=True)
    assert sorted(len(experts) for experts in plan.values()) == [0] + [1] * 47
    empty = next(layer for layer, experts in plan.items() if not experts)
    ranked = ranking_by_layer(pairs)
    rank_of_best = {layer: pairs.index((layer, ranked[layer][0])) for layer in range(48)}
    assert empty == max(rank_of_best, key=rank_of_best.get)


def test_uniform_remainder_goes_to_the_layers_whose_next_expert_ranks_best(offload, tmp_path):
    hottest = [(0, 0), (1, 0), (2, 0), (2, 1), (0, 1), (1, 1)]
    pairs = hottest + [(layer, expert) for layer in range(3) for expert in range(2, 4)]
    profile = write_profile(tmp_path / "expert-profile.bin", pairs, 3, 4)
    plan = offload.load_expert_profile_plan(profile, 3, 4, 5, uniform=True)
    assert plan == {0: [0, 1], 1: [0], 2: [0, 1]}


@pytest.mark.parametrize("uniform", [False, True])
def test_every_feasible_total_is_met_exactly(offload, tmp_path, uniform):
    pairs = exhaustive_pairs(3, 8, seed=3)
    profile = write_profile(tmp_path / "expert-profile.bin", pairs, 3, 8)
    for total in range(3 * 8 + 1):
        check_plan(offload.load_expert_profile_plan(profile, 3, 8, total, uniform=uniform), pairs, total)


def test_partial_profiles_are_filled_from_the_layers_that_rank_more(offload, tmp_path):
    pairs = [(0, e) for e in range(6)] + [(1, e) for e in range(5)] + [(2, e) for e in range(2)]
    random.Random(1).shuffle(pairs)
    profile = write_profile(tmp_path / "expert-profile.bin", pairs, 3, 8)
    for total in range(len(pairs) + 1):
        check_plan(offload.load_expert_profile_plan(profile, 3, 8, total, min_slots_per_layer=4), pairs, total)
    check_plan(offload.load_expert_profile_plan(profile, 3, 8, 8, uniform=True), pairs, 8)


@pytest.mark.parametrize(
    ("ranks", "total", "message"),
    [
        ([8, 8, 8], 25, "25 resident slots requested but .* ranks only 24"),
        ([6, 5, 2], 14, "14 resident slots requested but .* ranks only 13"),
        ([8, 8, 2], 9, r"3 slots per layer requested but layers \[2\] rank fewer"),
        ([5, 2, 2], 8, "leave 2 extra slots, but only 1 layers rank more than 2"),
    ],
)
def test_a_total_the_profile_cannot_fill_is_refused(offload, tmp_path, ranks, total, message):
    pairs = [(layer, expert) for layer, count in enumerate(ranks) for expert in range(count)]
    profile = write_profile(tmp_path / "expert-profile.bin", pairs, 3, 8)
    with pytest.raises(ValueError, match=message):
        offload.load_expert_profile_plan(profile, 3, 8, total, uniform=True)


def test_a_total_beyond_the_ranked_experts_is_refused_without_uniform_too(offload, tmp_path):
    pairs = exhaustive_pairs(3, 8)
    profile = write_profile(tmp_path / "expert-profile.bin", pairs, 3, 8)
    with pytest.raises(ValueError, match="ranks only 24"):
        offload.load_expert_profile_plan(profile, 3, 8, 25)


def test_duplicate_ranked_pairs_count_once(offload, tmp_path):
    pairs = exhaustive_pairs(3, 8)
    profile = write_profile(tmp_path / "expert-profile.bin", pairs + pairs[:10], 3, 8)
    check_plan(offload.load_expert_profile_plan(profile, 3, 8, 24), pairs, 24)
    with pytest.raises(ValueError, match="ranks only 24"):
        offload.load_expert_profile_plan(profile, 3, 8, 25)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"STRP", "shorter than the 24-byte STRP header"),
        (struct.pack("<6I", 0x12345678, 1, 3, 8, 24, 0), "Incompatible expert profile header"),
        (struct.pack("<6I", STRP_MAGIC, 2, 3, 8, 24, 0), "Incompatible expert profile header"),
        (struct.pack("<6I", STRP_MAGIC, 1, 4, 8, 24, 0), "Incompatible expert profile header"),
        (struct.pack("<6I", STRP_MAGIC, 1, 3, 16, 24, 0), "Incompatible expert profile header"),
        (struct.pack("<6I", STRP_MAGIC, 1, 3, 8, 24, 5) + struct.pack("<2H", 0, 0), "ranks 5 experts but holds only 1"),
        (struct.pack("<6I", STRP_MAGIC, 1, 3, 8, 24, 1) + struct.pack("<2H", 3, 0), "outside 3 layers x 8 experts"),
        (struct.pack("<6I", STRP_MAGIC, 1, 3, 8, 24, 1) + struct.pack("<2H", 0, 8), "outside 3 layers x 8 experts"),
    ],
)
def test_a_damaged_profile_is_refused(offload, tmp_path, content, message):
    profile = tmp_path / "expert-profile.bin"
    profile.write_bytes(content)
    with pytest.raises(ValueError, match=message):
        offload.load_expert_profile_plan(profile, 3, 8, 0)


@pytest.mark.parametrize(
    "arguments",
    [{"total_slots": -1}, {"min_slots_per_layer": -1}, {"num_layers": 0}, {"num_experts": 0}],
)
def test_nonsensical_plan_arguments_are_refused(offload, tmp_path, arguments):
    profile = write_profile(tmp_path / "expert-profile.bin", exhaustive_pairs(3, 8), 3, 8)
    with pytest.raises(ValueError):
        offload.load_expert_profile_plan(profile, **{"num_layers": 3, "num_experts": 8, "total_slots": 3, **arguments})


def caps_args(**overrides):
    values = {
        "expert_profile": None,
        "total_resident_slots": None,
        "target_resident_slots": None,
        "min_slots_per_layer": None,
        "uniform_profile_slots": False,
    }
    return argparse.Namespace(**{**values, **overrides})


def test_caps_without_a_profile_apply_to_every_layer(offload):
    model = decoder_model()
    assert offload.plan_resident_caps(caps_args(), model, LAYERS) == {}
    assert offload.plan_resident_caps(caps_args(target_resident_slots=5), model, LAYERS) == dict.fromkeys(
        range(LAYERS), 5
    )


def test_a_profile_without_a_slot_count_sets_no_caps(offload, tmp_path):
    profile = write_profile(tmp_path / "expert-profile.bin", exhaustive_pairs())
    assert offload.plan_resident_caps(caps_args(expert_profile=profile), decoder_model(), LAYERS) == {}


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize(("total", "target"), [(10, None), (None, 3), (23, None), (24, None)])
def test_caps_from_a_profile_add_up_to_the_requested_total(offload, tmp_path, uniform, total, target):
    pairs = exhaustive_pairs()
    profile = write_profile(tmp_path / "expert-profile.bin", pairs)
    args = caps_args(
        expert_profile=profile, total_resident_slots=total, target_resident_slots=target, uniform_profile_slots=uniform
    )
    caps = offload.plan_resident_caps(args, decoder_model(), LAYERS)
    assert sorted(caps) == list(range(LAYERS))
    assert sum(caps.values()) == (total if total is not None else target * LAYERS)


def test_caps_from_a_profile_refuse_what_it_cannot_provide(offload, tmp_path):
    profile = write_profile(tmp_path / "expert-profile.bin", exhaustive_pairs())
    args = caps_args(expert_profile=profile, total_resident_slots=25)
    with pytest.raises(ValueError, match="--expert-profile: 25 resident slots requested"):
        offload.plan_resident_caps(args, decoder_model(), LAYERS)
    other_layers = write_profile(tmp_path / "other.bin", exhaustive_pairs(4, EXPERTS), layers=4)
    with pytest.raises(ValueError, match="--expert-profile: Incompatible expert profile header"):
        offload.plan_resident_caps(
            caps_args(expert_profile=other_layers, total_resident_slots=4), decoder_model(), LAYERS
        )


def test_caps_from_a_profile_need_one_expert_count_and_matching_layer_numbers(offload, tmp_path):
    profile = write_profile(tmp_path / "expert-profile.bin", exhaustive_pairs())
    args = caps_args(expert_profile=profile, total_resident_slots=4)
    mixed = make_model(
        [qmoe_node(0), qmoe_node(1)], qmoe_initializers(0) + qmoe_initializers(1, dims={2: [4, 4, 4], 5: [4, 4, 4]})
    )
    with pytest.raises(ValueError, match=r"one expert count, the graph has \[4, 8\]"):
        offload.plan_resident_caps(args, mixed, 2)
    shifted = make_model([qmoe_node(1), qmoe_node(0)], qmoe_initializers(0) + qmoe_initializers(1))
    with pytest.raises(ValueError, match="is number 0 in the graph but the runtime numbers it layer 1"):
        offload.plan_resident_caps(args, shifted, 2)


@pytest.mark.parametrize(
    "argv",
    [
        ["--total-resident-slots", "5"],
        ["--min-slots-per-layer", "2"],
        ["--uniform-profile-slots"],
        ["--target-resident-slots", "-1"],
        ["--expert-profile", "profile.bin", "--total-resident-slots", "-3"],
        ["--expert-profile", "profile.bin", "--uniform-profile-slots"],
        ["--expert-profile", "profile.bin", "--min-slots-per-layer", "2"],
        [
            "--expert-profile",
            "profile.bin",
            "--total-resident-slots",
            "9",
            "--uniform-profile-slots",
            "--min-slots-per-layer",
            "2",
        ],
    ],
)
def test_contradicting_slot_arguments_are_refused(offload, argv):
    parser = offload.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        offload.check_slot_arguments(parser, parser.parse_args(argv))
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--target-resident-slots", "4"],
        ["--target-resident-slots", "0"],
        ["--expert-profile", "profile.bin"],
        ["--expert-profile", "profile.bin", "--total-resident-slots", "9"],
        ["--expert-profile", "profile.bin", "--target-resident-slots", "3", "--uniform-profile-slots"],
        ["--expert-profile", "profile.bin", "--total-resident-slots", "9", "--min-slots-per-layer", "2"],
    ],
)
def test_consistent_slot_arguments_are_accepted(offload, argv):
    parser = offload.build_parser()
    offload.check_slot_arguments(parser, parser.parse_args(argv))


# ---------------------------------------------------------------------------
# weights_pool.txt: placements, and the MTP expert pools that need an aligned copy
# ---------------------------------------------------------------------------

MTP_FILE = "target/head_data/source_weights/mtp_moe.bin"
ALIGNED = "target/head_data/source_weights/mtp_moe_aligned.bin"
requires_copy_file_range = pytest.mark.skipif(not hasattr(os, "copy_file_range"), reason="needs os.copy_file_range")


def pool_text(records, bindings):
    return "\n".join(["qwen38_weights_pool 1", *records, *bindings]) + "\n"


def read_pool(path):
    """The pool records ({id: tokens}) and the session bindings of a weights_pool.txt; pool ids must be unique."""
    lines = [line.split() for line in path.read_text().splitlines()]
    assert lines[0] == ["qwen38_weights_pool", "1"]
    records = [parts for parts in lines[1:] if parts[0] in ("pool", "host_pool")]
    assert len({parts[1] for parts in records}) == len(records), "duplicate pool ids"
    bindings = [(parts[1], parts[2], int(parts[3])) for parts in lines[1:] if parts[0] == "session"]
    assert len(records) + len(bindings) == len(lines) - 1
    return {int(parts[1]): parts for parts in records}, bindings


def mtp_pool_package(tmp_path, ranges, keyword="pool", records=(), bindings=()):
    """A layered package whose MTP expert pools are ``ranges`` = [(pool id, offset, length)] of one source file.

    ``records`` and ``bindings`` add other pool and session lines. Returns the package and the bytes of the source file.
    """
    data = random.Random(3).randbytes(max(offset + length for _pid, offset, length in ranges) + 1000)
    mtp_records = [f"{keyword} {pid} 1 {offset} {length} 1 {length} {MTP_FILE}" for pid, offset, length in ranges]
    mtp_bindings = [f"session mtp model.mtp.layers.0.mlp.experts.w{pid} {pid}" for pid, _offset, _length in ranges]
    package = make_package(
        tmp_path, layered=True, pool_text=pool_text(mtp_records + list(records), mtp_bindings + list(bindings))
    )
    (package.root / MTP_FILE).parent.mkdir(parents=True, exist_ok=True)
    (package.root / MTP_FILE).write_bytes(data)
    return package, data


def build_pool(offload, package, tmp_path, **placements):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return out, offload.build_offload_weights_pool(package.root, out, **placements)


def aligned_caches(tmp_path):
    return sorted(tmp_path.glob("pkg_mtp_moe_aligned-*"))


def check_bound(records, bindings):
    assert {pid for _sess, _name, pid in bindings} <= set(records)


def check_aligned_copy(out, data, ranges, records, expected_offsets, placement="host_mapped"):
    """Each unaligned range is in the aligned file at its planned offset, byte for byte, with only zeros around."""
    aligned = np.frombuffer((out / ALIGNED).read_bytes(), dtype=np.uint8)
    gaps = np.ones(len(aligned), dtype=bool)
    end = 0
    for pid, offset, length in ranges:
        record = records[pid]
        if offset % 16 == 0:
            assert record[2:4] == ["1", str(offset)] and record[-2:] == [MTP_FILE, placement]
            continue
        destination = expected_offsets[pid]
        assert destination % 4096 == 0
        assert record == ["pool", str(pid), "1", str(destination), str(length), "1", str(length), ALIGNED, placement]
        assert aligned[destination : destination + length].tobytes() == data[offset : offset + length]
        gaps[destination : destination + length] = False
        end = max(end, destination + length)
    assert len(aligned) == end
    assert not aligned[gaps].any()


# (id, [(pool id, offset in the source file, length)], planned offsets of the unaligned pools in the aligned file)
MTP_LAYOUTS = [
    ("ordinary ids 1 and 2", [(1, 4097, 100), (2, 9001, 5000)], {1: 0, 2: 4096}),
    ("ids 607 and 608", [(607, 12296, 3000), (608, 20488, 1500)], {607: 0, 608: 4096}),
    ("ids out of file order", [(9, 20008, 700), (4, 8, 5000), (7, 12008, 300)], {4: 0, 7: 8192, 9: 12288}),
    ("sparse ids", [(40, 5001, 10), (3, 6003, 20), (12000, 7007, 4096)], {3: 0, 40: 4096, 12000: 8192}),
    ("one pool", [(3, 17, 10)], {3: 0}),
    ("aligned pools stay", [(5, 4096, 200), (6, 4105, 300), (7, 8200, 4000), (8, 16384, 64)], {6: 0, 7: 4096}),
]


@requires_copy_file_range
@pytest.mark.parametrize("keyword", ["pool", "host_pool"])
@pytest.mark.parametrize(("ranges", "offsets"), [pytest.param(*case[1:], id=case[0]) for case in MTP_LAYOUTS])
def test_every_unaligned_mtp_pool_is_moved_to_the_aligned_file(offload, tmp_path, ranges, offsets, keyword):
    package, data = mtp_pool_package(tmp_path, ranges, keyword)
    out, stats = build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    records, bindings = read_pool(out / "weights_pool.txt")
    for pid, _offset, _length in ranges:
        assert pid in records, f"pool {pid} was dropped"
    check_bound(records, bindings)
    check_aligned_copy(out, data, ranges, records, offsets)
    (cache,) = aligned_caches(tmp_path)
    assert os.path.samefile(cache, out / ALIGNED)
    assert not list(tmp_path.rglob("*.tmp"))
    assert stats["mtp_moe_placement"] == "host_mapped"
    assert stats["total_pool_entries"] == len(records)


@requires_copy_file_range
def test_mtp_pools_stay_where_they_are_when_they_are_not_host_mapped(offload, tmp_path):
    ranges = [(1, 4097, 100), (2, 9001, 5000)]
    package, _data = mtp_pool_package(tmp_path, ranges)
    out, _stats = build_pool(offload, package, tmp_path, mtp_moe_placement="device")
    records, bindings = read_pool(out / "weights_pool.txt")
    check_bound(records, bindings)
    for pid, offset, length in ranges:
        assert records[pid] == ["pool", str(pid), "1", str(offset), str(length), "1", str(length), MTP_FILE]
    assert not (out / ALIGNED).exists()
    assert not aligned_caches(tmp_path)


def test_the_aligned_file_name_is_part_of_the_package_layout(offload):
    assert offload.ALIGNED_MTP_RELATIVE == ALIGNED


@requires_copy_file_range
def test_a_changed_layout_or_source_gets_its_own_aligned_file(offload, tmp_path, monkeypatch):
    ranges = [(1, 4097, 100), (2, 9001, 5000)]
    package, data = mtp_pool_package(tmp_path, ranges)
    out, _ = build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    (first,) = aligned_caches(tmp_path)

    def must_not_copy(*_args, **_kwargs):
        raise AssertionError("an unchanged layout must reuse its aligned file")

    with monkeypatch.context() as patch:
        patch.setattr(os, "copy_file_range", must_not_copy)
        build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    assert aligned_caches(tmp_path) == [first]

    # The same pool ids and sizes read from other offsets of the file.
    shifted = [(1, 4113, 100), (2, 9017, 5000)]
    shutil.rmtree(package.root)
    package, data = mtp_pool_package(tmp_path, shifted)
    out, _ = build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    records, _bindings = read_pool(out / "weights_pool.txt")
    check_aligned_copy(out, data, shifted, records, {1: 0, 2: 4096})
    assert len(aligned_caches(tmp_path)) == 2
    assert os.path.samefile(next(path for path in aligned_caches(tmp_path) if path != first), out / ALIGNED)

    # Edited source bytes of the same size and layout.
    source = package.root / MTP_FILE
    stat = source.stat()
    source.write_bytes(bytes(reversed(data)))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    out, _ = build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    records, _bindings = read_pool(out / "weights_pool.txt")
    check_aligned_copy(out, bytes(reversed(data)), shifted, records, {1: 0, 2: 4096})
    assert len(aligned_caches(tmp_path)) == 3


@requires_copy_file_range
def test_a_cached_aligned_file_of_the_wrong_size_is_rebuilt(offload, tmp_path):
    ranges = [(1, 4097, 100), (2, 9001, 5000)]
    package, data = mtp_pool_package(tmp_path, ranges)
    build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    (cache,) = aligned_caches(tmp_path)
    cache.write_bytes(b"truncated")
    out, _ = build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    records, _bindings = read_pool(out / "weights_pool.txt")
    check_aligned_copy(out, data, ranges, records, {1: 0, 2: 4096})
    assert aligned_caches(tmp_path) == [cache]


@requires_copy_file_range
def test_a_failed_copy_leaves_no_partial_aligned_file(offload, tmp_path, monkeypatch):
    package, _data = mtp_pool_package(tmp_path, [(1, 4097, 100), (2, 9001, 5000)])
    for failure in (OSError("disk full"), KeyboardInterrupt()):

        def fail(*_args, failure=failure, **_kwargs):
            raise failure

        with monkeypatch.context() as patch:
            patch.setattr(os, "copy_file_range", fail)
            with pytest.raises(type(failure)):
                build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
        assert not aligned_caches(tmp_path)
        assert not (tmp_path / "out" / "weights_pool.txt").exists()


@requires_copy_file_range
def test_a_copy_that_stops_early_is_an_error_and_leaves_nothing(offload, tmp_path, monkeypatch):
    package, _data = mtp_pool_package(tmp_path, [(1, 4097, 100), (2, 9001, 5000)])
    monkeypatch.setattr(os, "copy_file_range", lambda *_args, **_kwargs: 0)
    with pytest.raises(RuntimeError, match="stopped early"):
        build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    assert not aligned_caches(tmp_path)


@requires_copy_file_range
def test_partial_copies_are_continued_until_the_range_is_complete(offload, tmp_path, monkeypatch):
    ranges = [(1, 4097, 100), (2, 9001, 5000)]
    package, data = mtp_pool_package(tmp_path, ranges)
    real = os.copy_file_range

    def seven_bytes(src, dst, count, offset_src=None, offset_dst=None):
        return real(src, dst, min(count, 7), offset_src=offset_src, offset_dst=offset_dst)

    monkeypatch.setattr(os, "copy_file_range", seven_bytes)
    out, _ = build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    records, _bindings = read_pool(out / "weights_pool.txt")
    check_aligned_copy(out, data, ranges, records, {1: 0, 2: 4096})


@requires_copy_file_range
@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ([(1, 4097, 100), (2, 10**6 + 1, 5000)], r"pool 2: bytes 1000001\.\.1005001 lie outside .*mtp_moe\.bin"),
        ([(1, 4097, 10**9)], r"pool 1: bytes 4097\.\.1000004097 lie outside"),
    ],
)
def test_a_pool_outside_its_file_is_refused_by_id_before_anything_is_written(offload, tmp_path, ranges, message):
    package, _data = mtp_pool_package(tmp_path, [(1, 4097, 100), (2, 9001, 5000)])
    (package.root / "weights_pool.txt").write_text(
        pool_text(
            [f"pool {pid} 1 {offset} {length} 1 {length} {MTP_FILE}" for pid, offset, length in ranges],
            [f"session mtp model.mtp.layers.0.mlp.experts.w{pid} {pid}" for pid, _offset, _length in ranges],
        )
    )
    with pytest.raises(ValueError, match=message):
        build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    assert not aligned_caches(tmp_path)
    assert not list(tmp_path.rglob("*.tmp"))
    assert not (tmp_path / "out" / "weights_pool.txt").exists()


@requires_copy_file_range
def test_a_missing_source_file_is_refused_by_pool_id(offload, tmp_path):
    package, _data = mtp_pool_package(tmp_path, [(1, 4097, 100)])
    (package.root / MTP_FILE).unlink()
    with pytest.raises(ValueError, match=r"pool 1: source file .*mtp_moe\.bin does not exist"):
        build_pool(offload, package, tmp_path, mtp_moe_placement="host_mapped")
    assert not aligned_caches(tmp_path)


@pytest.mark.parametrize(
    ("records", "bindings", "message"),
    [
        (
            ["pool 1 1 0 64 1 64 w.bin"],
            ["session mtp model.mtp.layers.0.mlp.experts.w 7"],
            "1 session bindings name a pool",
        ),
        (
            ["pool 1 1 0 64 1 64 w.bin", "host_pool 1 1 64 64 1 64 w.bin"],
            [],
            r"pool ids \[1\] are defined more than once",
        ),
    ],
)
def test_a_source_pool_file_with_dangling_or_duplicate_pools_is_refused(offload, tmp_path, records, bindings, message):
    package = make_package(tmp_path, pool_text=pool_text(records, bindings))
    with pytest.raises(ValueError, match=f"the source weights_pool.txt: .*{message}"):
        build_pool(offload, package, tmp_path)
    assert not (tmp_path / "out" / "weights_pool.txt").exists()


def test_pool_binding_checks_report_what_is_wrong(offload):
    offload.check_pool_bindings([1, 2], [("a", "x", 1), ("a", "y", 2), ("b", "x", 2)], "where")
    with pytest.raises(ValueError, match=r"where: pool ids \[2, 9\] are defined more than once"):
        offload.check_pool_bindings([9, 2, 2, 9, 1], [], "where")
    with pytest.raises(ValueError, match=r"where: 2 session bindings .* e\.g\. session a x 5, session b y 6"):
        offload.check_pool_bindings([1], [("a", "x", 5), ("b", "y", 6), ("c", "z", 1)], "where")


def test_aligned_copy_plan_packs_ranges_on_page_boundaries(offload):
    # The two MTP expert banks of the real package.
    assert offload.plan_aligned_copy([(607, 3355443200), (608, 1677721600)]) == (
        {607: 0, 608: 3355443200},
        5033164800,
    )
    assert offload.plan_aligned_copy([(1, 10), (2, 5000), (3, 1)]) == ({1: 0, 2: 4096, 3: 12288}, 12289)
    assert offload.plan_aligned_copy([(1, 4096), (2, 0), (3, 4097)]) == ({1: 0, 2: 4096, 3: 4096}, 8193)
    assert offload.plan_aligned_copy([]) == ({}, 0)


@requires_copy_file_range
def test_copy_file_ranges_writes_each_range_at_its_offset(offload, tmp_path):
    source = tmp_path / "source.bin"
    data = random.Random(1).randbytes(300)
    source.write_bytes(data)
    destination = tmp_path / "packed.bin"
    offload.copy_file_ranges(destination, [(1, source, 250, 0, 50), (2, source, 3, 128, 100)])
    packed = destination.read_bytes()
    assert len(packed) == 228
    assert packed[:50] == data[250:300] and packed[128:228] == data[3:103]
    assert not any(packed[50:128])
    offload.copy_file_ranges(destination, [])
    assert destination.read_bytes() == b""


@requires_copy_file_range
def test_copy_file_ranges_checks_every_range_before_creating_the_file(offload, tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(100))
    destination = tmp_path / "packed.bin"
    for ranges, message in (
        ([(1, source, 0, 0, 10), (5, source, 95, 10, 10)], "pool 5: bytes 95..105 lie outside"),
        ([(6, source, -1, 0, 10)], "pool 6: bytes -1..9 lie outside"),
        ([(7, source, 0, 0, -1)], "pool 7: bytes 0..-1 lie outside"),
        ([(8, tmp_path / "absent.bin", 0, 0, 1)], "pool 8: source file .* does not exist"),
    ):
        with pytest.raises(ValueError, match=message):
            offload.copy_file_ranges(destination, ranges)
        assert not destination.exists()


def test_other_pools_get_the_requested_placements(offload, tmp_path):
    records = [
        "pool 0 1 0 64 2 4 4 weights.data",
        "pool 1 1 64 64 2 4 4 weights.data",
        "host_pool 2 1 128 64 2 4 4 weights.data",
        "pool 3 1 192 64 2 4 4 weights.data",
        "pool 4 1 256 64 2 4 4 weights.data device",
        "pool 5 1 320 64 2 4 4 weights.data cpu",
        "pool 6 1 384 64 2 4 4 weights.data host_mapped",
    ]
    bindings = [
        "session target model.embed_tokens.weight 0",
        "session target model.layers.0.mlp.fc1_experts_weights 1",
        "session target model.norm.weight 2",
        "session target model.a 3",
        "session target model.b 4",
        "session target model.c 5",
        "session target model.d 6",
    ]
    package = make_package(tmp_path, pool_text=pool_text(records, bindings), dense=False)
    out, _stats = build_pool(offload, package, tmp_path)
    placements = {pid: parts[7 + int(parts[5]) :] for pid, parts in read_pool(out / "weights_pool.txt")[0].items()}
    assert [placements[pid] for pid in range(7)] == [
        ["host_mapped"],
        ["cpu"],
        ["host_mapped"],
        [],
        [],
        ["cpu"],
        ["host_mapped"],
    ]
    out, _stats = build_pool(offload, package, tmp_path, embed_placement="device")
    assert read_pool(out / "weights_pool.txt")[0][0][7 + 2 :] == []


@pytest.mark.parametrize("layered", [False, True])
def test_unbound_expert_tensors_get_new_cpu_pools_with_fresh_ids(offload, tmp_path, layered):
    pool = pool_text(
        ["pool 5 1 0 64 2 4 4 weights.data", "pool 40 1 64 64 2 4 4 weights.data"],
        ["session target model.embed_tokens.weight 5", "session target model.norm.weight 40"],
    )
    package = make_package(tmp_path, pool_text=pool, layered=layered, dense=False)
    out, stats = build_pool(offload, package, tmp_path)
    records, bindings = read_pool(out / "weights_pool.txt")
    check_bound(records, bindings)
    fresh = sorted(set(records) - {5, 40})
    assert fresh == list(range(41, 41 + 6 * LAYERS))
    location = "target/weights.data" if layered else "weights.data"
    assert {tuple(records[pid][-2:]) for pid in fresh} == {(location, "cpu")}
    assert {name for sess, name, pid in bindings if pid in fresh} == {
        init.name for layer in range(LAYERS) for init in qmoe_initializers(layer)
    }
    assert stats["cpu_qmoe_entries"] == 6 * LAYERS
    assert stats["total_pool_entries"] == len(records)


def test_dense_weights_are_pooled_only_when_asked(offload, tmp_path):
    package = make_package(tmp_path)
    out, stats = build_pool(offload, package, tmp_path, dense_placement="device")
    assert stats["added_dense_entries"] == 0
    assert not [binding for binding in read_pool(out / "weights_pool.txt")[1] if binding[0] == "dense"]
    for placement in ("host_mapped", "cpu"):
        out, stats = build_pool(offload, package, tmp_path, dense_placement=placement)
        records, bindings = read_pool(out / "weights_pool.txt")
        (dense,) = [binding for binding in bindings if binding[0] == "dense"]
        assert stats["added_dense_entries"] == 1
        assert records[dense[2]][2:] == ["2", "0", "16", "2", "4", "4", "target/source_weights/dense.data", placement]


# ---------------------------------------------------------------------------
# main(): the package a user gets
# ---------------------------------------------------------------------------

NAMES = [NODE_NAME.format(layer) for layer in range(LAYERS)]
REMOVED_FLAGS = [
    "--migrate-shared-onnx",
    "--out-onnx",
    "--strip-phase",
    "--strip-stats",
    "--target-spare-slots",
    "--dense-offload-qmoe",
    "--dense-stream-min-rows",
    "--dense-stream-wave",
]


def caps_of(entries):
    """{node name: max_resident_slots} of the node_overrides entry."""
    overrides = parse_node_overrides(entries.get(PREFIX + "node_overrides", ""))
    return {override["name"]: override["max_resident_slots"] for override in overrides}


def build(monkeypatch, offload, package, tmp_path, *extra, layered=False):
    out = tmp_path / "out"
    run_builder(monkeypatch, offload, package, out, *extra)
    return (out, *read_output(out, layered))


@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("mode", ["required", "auto"])
@pytest.mark.parametrize("placement", ["adaptive", "static"])
def test_the_built_decoder_is_standard_qmoe_that_the_runtime_offloads(
    offload, tmp_path, monkeypatch, multidraft_calls, layered, mode, placement
):
    package = make_package(tmp_path, layered=layered)
    before = snapshot(package.root)
    out, config, entries, model, decoder_dir = build(
        monkeypatch,
        offload,
        package,
        tmp_path,
        "--offload-mode",
        mode,
        "--offload-placement",
        placement,
        layered=layered,
    )

    assert entries == {PREFIX + "mode": mode, PREFIX + "placement": placement, PREFIX + "cpu_execution": "auto"}
    assert {(node.op_type, node.domain) for node in model.graph.node} == {("QMoE", "com.microsoft")}
    assert all({attribute.name for attribute in node.attribute} <= QMOE_ATTRIBUTES for node in model.graph.node)
    source = onnx.load(str(package.decoder_dir / "model.onnx"), load_external_data=False)
    for old, new in zip(source.graph.node, model.graph.node, strict=True):
        assert (new.name, list(new.output)) == (old.name, list(old.output))
        assert list(new.input) == [*old.input[:14], "", *old.input[15:]]
        assert [a.SerializeToString() for a in new.attribute] == [a.SerializeToString() for a in old.attribute]
    assert list(model.graph.initializer) == list(source.graph.initializer)

    # The runtime offloads every node of the built graph. It would not offload the nodes of the source graph, whose
    # router_weights input is not supported by the offload backend.
    assert simulate_session_creation(model, entries, decoder_dir) == NAMES
    if mode == "required":
        with pytest.raises(ContractError, match="rejected node"):
            simulate_session_creation(source, entries, package.decoder_dir)
    else:
        assert simulate_session_creation(source, entries, package.decoder_dir) == []

    assert multidraft_calls == [("model_s2.onnx", "model_s3.onnx", 3), ("model_s2.onnx", "model_s4.onnx", 4)]
    assert {path.name for path in decoder_dir.glob("model*.onnx")} == {
        "model.onnx",
        "model_s2.onnx",
        "model_s3.onnx",
        "model_s4.onnx",
    }
    assert snapshot(package.root) == before
    manifest = json.loads((out / "package_manifest.json").read_text())
    assert manifest["stack"] == "v12b-engine-qmoe-offload-v4"
    assert manifest["target_qmoe_nodes"] == manifest["router_weights_cleared"] == LAYERS
    assert manifest["qmoe_nodes_not_offloaded"] == []
    assert manifest["offload_session_options"] == entries


def test_budgets_and_the_other_options_reach_the_config(offload, tmp_path, monkeypatch, multidraft_calls):
    package = make_package(tmp_path)
    out, config, entries, model, decoder_dir = build(
        monkeypatch,
        offload,
        package,
        tmp_path,
        "--device-expert-bytes",
        "auto",
        "--host-expert-bytes",
        str(40 << 30),
        "--stop-token-id",
        "9",
        "--vram-limit-gib",
        "24",
        "--prefill-chunk",
        "256",
    )
    assert entries == {
        PREFIX + "mode": "required",
        PREFIX + "placement": "adaptive",
        PREFIX + "cpu_execution": "auto",
        PREFIX + "device_expert_bytes": "auto",
        PREFIX + "host_expert_bytes": str(40 << 30),
    }
    assert simulate_session_creation(model, entries, decoder_dir) == NAMES
    decoder = config["model"]["decoder"]["session_options"]
    assert decoder["log_id"] == "onnxruntime-genai"
    assert decoder["provider_options"][0]["cuda"] == {
        "enable_cuda_graph": "1",
        "arena_extend_strategy": "kSameAsRequested",
        "arena.extend_strategy": "1",
        "gpu_device_mem_limit": str(24 << 30),
    }
    assert config["model"]["eos_token_id"] == [9]
    prefill = config["model"]["qwen38_graph"]["prefill"]
    assert (prefill["chunk"], prefill["warmup_lengths"]) == (256, [256, 128, 37])
    assert (
        config["model"]["qwen38_graph"]["fold"]["head_r2"]["session_options"][
            "session.use_device_allocator_for_initializers"
        ]
        == "1"
    )
    assert not any(
        key.startswith(PREFIX) for key in config["model"]["qwen38_graph"]["fold"]["head_r2"]["session_options"]
    )


def test_a_package_without_a_dense_graph_builds(offload, tmp_path, monkeypatch, multidraft_calls):
    package = make_package(tmp_path, dense=False)
    out, _config, entries, model, decoder_dir = build(monkeypatch, offload, package, tmp_path)
    assert simulate_session_creation(model, entries, decoder_dir) == NAMES
    assert not (out / "dense").exists()


def test_copy_makes_independent_files_and_the_default_links_them(offload, tmp_path, monkeypatch, multidraft_calls):
    package = make_package(tmp_path)
    (package.root / "weights.data").write_bytes(b"expert bytes")
    out, *_ = build(monkeypatch, offload, package, tmp_path)
    assert os.path.samefile(package.root / "weights.data", out / "weights.data")
    shutil.rmtree(out)
    out, *_ = build(monkeypatch, offload, package, tmp_path, "--copy")
    assert (out / "weights.data").read_bytes() == b"expert bytes"
    assert not os.path.samefile(package.root / "weights.data", out / "weights.data")


def test_ops_libraries_are_copied_into_the_package(offload, tmp_path, monkeypatch, multidraft_calls):
    package = make_package(tmp_path)
    ops = tmp_path / "ops"
    ops.mkdir()
    (ops / "libqwen_a.so").write_bytes(b"a")
    (ops / "other.so").write_bytes(b"other")
    out, *_ = build(monkeypatch, offload, package, tmp_path, "--ops-dir", ops)
    assert sorted(path.name for path in (out / "libs").iterdir()) == ["libqwen38_plane.so", "libqwen_a.so"]
    assert not os.path.samefile(ops / "libqwen_a.so", out / "libs/libqwen_a.so")


def test_target_resident_slots_caps_every_node_by_name(offload, tmp_path, monkeypatch, multidraft_calls):
    package = make_package(tmp_path)
    _out, _config, entries, model, decoder_dir = build(
        monkeypatch, offload, package, tmp_path, "--target-resident-slots", "5"
    )
    assert caps_of(entries) == dict.fromkeys(NAMES, 5)
    assert PREFIX + "popularity_file" not in entries
    assert simulate_session_creation(model, entries, decoder_dir) == NAMES


def test_an_expert_profile_alone_sets_no_caps(offload, tmp_path, monkeypatch, multidraft_calls):
    package = make_package(tmp_path)
    profile = write_profile(tmp_path / "profile.bin", exhaustive_pairs())
    out, _config, entries, model, decoder_dir = build(
        monkeypatch, offload, package, tmp_path, "--expert-profile", profile
    )
    assert entries[PREFIX + "popularity_file"] == "expert-profile.bin"
    assert PREFIX + "node_overrides" not in entries
    assert (decoder_dir / "expert-profile.bin").read_bytes() == profile.read_bytes()
    assert simulate_session_creation(model, entries, decoder_dir) == NAMES
    validate_popularity_file(decoder_dir / "expert-profile.bin")


@pytest.mark.parametrize("layered", [False, True])
def test_the_profile_is_copied_next_to_the_decoder_without_touching_the_source(
    offload, tmp_path, monkeypatch, multidraft_calls, layered
):
    package = make_package(tmp_path, layered=layered)
    (package.decoder_dir / "expert-profile.bin").write_bytes(b"stale profile of the source package")
    profile = write_profile(tmp_path / "profile.bin", exhaustive_pairs())
    out, _config, _entries, _model, decoder_dir = build(
        monkeypatch, offload, package, tmp_path, "--expert-profile", profile, layered=layered
    )
    assert (package.decoder_dir / "expert-profile.bin").read_bytes() == b"stale profile of the source package"
    assert (decoder_dir / "expert-profile.bin").read_bytes() == profile.read_bytes()
    assert not os.path.samefile(decoder_dir / "expert-profile.bin", profile)
    assert not os.path.samefile(decoder_dir / "expert-profile.bin", package.decoder_dir / "expert-profile.bin")


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("total", [10, 23, 24])
def test_a_profile_and_total_spread_exactly_that_many_slots(
    offload, tmp_path, monkeypatch, multidraft_calls, uniform, total
):
    package = make_package(tmp_path)
    profile = write_profile(tmp_path / "profile.bin", exhaustive_pairs())
    extra = ["--uniform-profile-slots"] if uniform else []
    _out, _config, entries, model, decoder_dir = build(
        monkeypatch, offload, package, tmp_path, "--expert-profile", profile, "--total-resident-slots", total, *extra
    )
    caps = caps_of(entries)
    assert list(caps) == NAMES
    assert sum(caps.values()) == total
    if uniform:
        assert max(caps.values()) - min(caps.values()) <= 1
    assert all(0 <= cap <= EXPERTS for cap in caps.values())
    assert simulate_session_creation(model, entries, decoder_dir) == NAMES


def test_target_slots_with_a_profile_are_a_per_layer_average(offload, tmp_path, monkeypatch, multidraft_calls):
    package = make_package(tmp_path)
    profile = write_profile(tmp_path / "profile.bin", exhaustive_pairs())
    _out, _config, entries, _model, _decoder_dir = build(
        monkeypatch, offload, package, tmp_path, "--expert-profile", profile, "--target-resident-slots", 5
    )
    assert sum(caps_of(entries).values()) == 5 * LAYERS


def test_a_total_the_profile_cannot_fill_stops_the_build_before_it_writes(
    offload, tmp_path, monkeypatch, multidraft_calls
):
    package = make_package(tmp_path)
    profile = write_profile(tmp_path / "profile.bin", exhaustive_pairs())
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit, match="error: --expert-profile: 25 resident slots requested but .* ranks only 24"):
        run_builder(
            monkeypatch, offload, package, tmp_path / "out", "--expert-profile", profile, "--total-resident-slots", 25
        )
    assert snapshot(tmp_path) == before


def ineligible_model(layer=1):
    model = decoder_model()
    attributes = {attribute.name: attribute for attribute in model.graph.node[layer].attribute}
    attributes["swiglu_fusion"].i = 2
    return model


def test_required_mode_refuses_nodes_the_runtime_cannot_offload(offload, tmp_path, monkeypatch, multidraft_calls):
    package = make_package(tmp_path, model=ineligible_model())
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        run_builder(monkeypatch, offload, package, tmp_path / "out")
    message = str(excinfo.value)
    assert message.startswith("1 of 3 QMoE nodes cannot be offloaded by the runtime; first, ")
    assert repr(NAMES[1]) in message and "swiglu_fusion=2" in message and "--offload-mode auto" in message
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("caps", [[], ["--target-resident-slots", "4"]])
def test_auto_mode_keeps_the_nodes_the_runtime_cannot_offload_on_the_gpu(
    offload, tmp_path, monkeypatch, multidraft_calls, capsys, caps
):
    package = make_package(tmp_path, model=ineligible_model())
    _out, _config, entries, model, decoder_dir = build(
        monkeypatch, offload, package, tmp_path, "--offload-mode", "auto", *caps
    )
    assert "warning: 1 of 3 QMoE nodes cannot be offloaded by the runtime" in capsys.readouterr().out
    assert simulate_session_creation(model, entries, decoder_dir) == [NAMES[0], NAMES[2]]
    if caps:
        assert caps_of(entries) == {NAMES[0]: 4, NAMES[2]: 4}
    manifest = json.loads((tmp_path / "out/package_manifest.json").read_text())
    assert manifest["qmoe_nodes_not_offloaded"] == [repr(NAMES[1])]
    assert manifest["router_weights_cleared"] == LAYERS


def test_auto_mode_with_a_profile_keeps_the_layer_numbers_of_the_other_nodes(
    offload, tmp_path, monkeypatch, multidraft_calls
):
    package = make_package(tmp_path, model=ineligible_model(layer=0))
    profile = write_profile(tmp_path / "profile.bin", exhaustive_pairs())
    _out, _config, entries, model, decoder_dir = build(
        monkeypatch,
        offload,
        package,
        tmp_path,
        "--offload-mode",
        "auto",
        "--expert-profile",
        profile,
        "--total-resident-slots",
        12,
    )
    assert list(caps_of(entries)) == [NAMES[1], NAMES[2]]
    assert simulate_session_creation(model, entries, decoder_dir) == [NAMES[1], NAMES[2]]


def test_unnamed_nodes_cannot_get_caps_but_build_without_them(offload, tmp_path, monkeypatch, multidraft_calls):
    unnamed = decoder_model(name="")
    package = make_package(tmp_path, model=unnamed)
    before = snapshot(tmp_path)
    with pytest.raises(
        SystemExit, match=r"error: per-node resident-slot caps need uniquely named QMoE nodes, but #0 \(unnamed\)"
    ):
        run_builder(monkeypatch, offload, package, tmp_path / "out", "--target-resident-slots", "4")
    assert snapshot(tmp_path) == before
    _out, _config, entries, model, decoder_dir = build(monkeypatch, offload, package, tmp_path)
    assert simulate_session_creation(model, entries, decoder_dir) == ["", "", ""]


def test_nodes_sharing_a_name_cannot_get_caps(offload, tmp_path, monkeypatch, multidraft_calls):
    package = make_package(tmp_path, model=decoder_model(name="same"))
    with pytest.raises(
        SystemExit, match=r"error: per-node resident-slot caps need uniquely named QMoE nodes, but 'same' "
    ):
        run_builder(monkeypatch, offload, package, tmp_path / "out", "--target-resident-slots", "4")
    assert not (tmp_path / "out").exists()


def test_a_legacy_qmoe_offload_graph_is_refused(offload, tmp_path, monkeypatch, multidraft_calls):
    legacy = decoder_model()
    legacy.graph.node[1].op_type = "QMoEOffload"
    package = make_package(tmp_path, model=legacy)
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit, match=r"model\.onnx has 1 QMoEOffload nodes\. The runtime has no such operator"):
        run_builder(monkeypatch, offload, package, tmp_path / "out")
    assert snapshot(tmp_path) == before


def test_a_legacy_qmoe_offload_dense_graph_is_refused(offload, tmp_path, monkeypatch, multidraft_calls):
    package = make_package(tmp_path)
    dense = dense_model()
    dense.graph.node[0].op_type = "QMoEOffload"
    save_model(dense, package.root / "dense/model.onnx")
    with pytest.raises(SystemExit, match=r"dense/model\.onnx has 1 QMoEOffload nodes"):
        run_builder(monkeypatch, offload, package, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_graph_without_qmoe_is_refused(offload, tmp_path, monkeypatch, multidraft_calls):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    package = make_package(tmp_path, model=helper.make_model(graph))
    with pytest.raises(SystemExit, match="has no com.microsoft::QMoE node to offload"):
        run_builder(monkeypatch, offload, package, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("value", [None, "1"])
@pytest.mark.parametrize("flag", REMOVED_FLAGS)
def test_removed_flags_fail_instead_of_being_ignored(offload, tmp_path, monkeypatch, capsys, flag, value):
    package = make_package(tmp_path)
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        run_builder(monkeypatch, offload, package, tmp_path / "out", flag, *([value] if value else []))
    assert excinfo.value.code == 2
    assert f"{flag} was removed" in capsys.readouterr().err
    assert snapshot(tmp_path) == before


def test_every_removed_flag_is_tested(offload):
    assert sorted(offload.REMOVED_OPTIONS) == sorted(REMOVED_FLAGS)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--offload-mode", "disabled"),
        ("--offload-placement", "basic"),
        ("--device-expert-bytes", "-1"),
        ("--host-expert-bytes", "12.5"),
        ("--target-resident-slots", "-2"),
        ("--mtp-moe-placement", "cpu"),
    ],
)
def test_option_values_the_runtime_would_reject_are_refused_by_the_parser(offload, tmp_path, monkeypatch, flag, value):
    package = make_package(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        run_builder(monkeypatch, offload, package, tmp_path / "out", flag, value)
    assert excinfo.value.code == 2
    assert not (tmp_path / "out").exists()


def test_missing_arguments_and_files_are_named(offload, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="--from-package, --out, --plane-lib, and --provider-lib are required"):
        run_main(monkeypatch, offload, "--out", tmp_path / "out")
    for missing, label in (
        ("genai_config.json", "genai_config.json of --from-package"),
        ("weights_pool.txt", "weights_pool.txt of --from-package"),
        ("model.onnx", "the target graph of --from-package"),
    ):
        package = make_package(tmp_path / missing)
        (package.root / missing).unlink()
        with pytest.raises(SystemExit, match=f"{label} not found"):
            run_builder(monkeypatch, offload, package, tmp_path / missing / "out")
    package = make_package(tmp_path / "missing-libs")
    out = tmp_path / "missing-libs" / "out"
    for kind, label in (("plane", "--plane-lib"), ("provider", "--provider-lib")):
        library = getattr(package, kind)
        library.unlink()
        with pytest.raises(SystemExit, match=f"{label} not found"):
            run_builder(monkeypatch, offload, package, out)
        library.write_bytes(b"restored")
    with pytest.raises(SystemExit, match="--expert-profile not found"):
        run_builder(monkeypatch, offload, package, out, "--expert-profile", tmp_path / "absent.bin")
    assert not out.exists()


OVERLAPS = [
    ("--from-package", "equal"),
    ("--from-package", "inside"),
    ("--from-package", "contains"),
    ("--plane-lib", "equal"),
    ("--plane-lib", "contains"),
    ("--provider-lib", "contains"),
    ("--ops-dir", "equal"),
    ("--ops-dir", "inside"),
    ("--ops-dir", "contains"),
    ("--expert-profile", "equal"),
    ("--expert-profile", "contains"),
]


@pytest.mark.parametrize(("flag", "relation"), OVERLAPS)
def test_an_output_overlapping_an_input_is_refused_before_anything_changes(
    offload, tmp_path, monkeypatch, multidraft_calls, flag, relation
):
    package = make_package(tmp_path)
    ops = tmp_path / "ops"
    ops.mkdir()
    (ops / "libqwen_a.so").write_bytes(b"a")
    (tmp_path / "profile").mkdir()
    profile = write_profile(tmp_path / "profile" / "expert-profile.bin", exhaustive_pairs())
    inputs = {
        "--from-package": package.root,
        "--plane-lib": package.plane,
        "--provider-lib": package.provider,
        "--ops-dir": ops,
        "--expert-profile": profile,
    }
    path = inputs[flag]
    out = {"equal": path, "inside": path / "out", "contains": path.parent}[relation]
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit, match="must not be equal to, nested inside or contain"):
        run_builder(monkeypatch, offload, package, out, "--ops-dir", ops, "--expert-profile", profile)
    assert snapshot(tmp_path) == before


@requires_copy_file_range
def test_host_mapped_mtp_pools_of_any_id_are_kept_end_to_end(offload, tmp_path, monkeypatch, multidraft_calls):
    ranges = [(1, 4097, 100), (2, 9001, 5000)]
    package, data = mtp_pool_package(tmp_path, ranges)
    out, _config, _entries, _model, _decoder_dir = build(
        monkeypatch, offload, package, tmp_path, "--mtp-moe-placement", "host_mapped", layered=True
    )
    records, bindings = read_pool(out / "weights_pool.txt")
    check_bound(records, bindings)
    check_aligned_copy(out, data, ranges, records, {1: 0, 2: 4096})
    assert {pid for _sess, _name, pid in bindings} >= {1, 2}
    manifest = json.loads((out / "package_manifest.json").read_text())
    assert manifest["pool_stats"]["mtp_moe_placement"] == "host_mapped"
    assert manifest["pool_stats"]["total_pool_entries"] == len(records)


def test_the_default_mtp_placement_leaves_the_pools_alone(offload, tmp_path, monkeypatch, multidraft_calls):
    ranges = [(1, 4097, 100), (2, 9001, 5000)]
    package, _data = mtp_pool_package(tmp_path, ranges)
    out, *_ = build(monkeypatch, offload, package, tmp_path, layered=True)
    records, bindings = read_pool(out / "weights_pool.txt")
    check_bound(records, bindings)
    assert records[1][3:5] == ["4097", "100"] and records[2][3:5] == ["9001", "5000"]
    assert not (out / ALIGNED).exists()


# ---------------------------------------------------------------------------
# genai_config.json: only the model.qwen38_graph keys the engine accepts
# ---------------------------------------------------------------------------

HEAD_SESSION_OPTIONS = {
    "log_id": "onnxruntime-genai",
    "graph_optimization_level": "ORT_DISABLE_ALL",
    "provider_options": [
        {
            "cuda": {
                "enable_cuda_graph": "1",
                "min_num_runs_before_cuda_graph_capture": "4",
                "arena_extend_strategy": "kSameAsRequested",
                "arena.extend_strategy": "1",
            }
        }
    ],
    "session.use_device_allocator_for_initializers": "1",
}


def config_with_removed_key(key, layered=False):
    """The source config of a package built for an earlier config surface: ``key`` is a path below model.qwen38_graph."""
    config = genai_config(layered)
    qwen = config["model"]["qwen38_graph"]
    if key in ("fold.mtp", "fold.drafter"):
        qwen["fold"][key.removeprefix("fold.")] = qwen["fold"].pop("head_r2")
        return config
    *parents, leaf = key.split(".")
    section = qwen
    for part in parents:
        section = section.setdefault(part, {})
    section[leaf] = 1
    return config


def printed_config_changes(capsys):
    return sorted(line for line in capsys.readouterr().out.splitlines() if line.startswith("genai_config:"))


@pytest.mark.parametrize("layered", [False, True])
def test_the_built_config_has_only_keys_the_engine_accepts(offload, tmp_path, monkeypatch, multidraft_calls, layered):
    package = make_package(tmp_path, layered=layered)
    _out, config, *_ = build(monkeypatch, offload, package, tmp_path, layered=layered)
    qwen = config["model"]["qwen38_graph"]
    assert unknown_qwen38_graph_keys(qwen) == []
    assert prefill_chunk_problem(qwen) is None
    assert qwen["fold"] == {
        "enabled": True,
        "head_r2": {"filename": "head_r2/model.onnx", "session_options": HEAD_SESSION_OPTIONS},
    }
    assert qwen["prefill"]["chunk"] == 512


@pytest.mark.parametrize("key", REMOVED_QWEN38_GRAPH_KEYS)
def test_a_removed_key_of_the_source_config_is_not_carried_into_the_package(
    offload, tmp_path, monkeypatch, multidraft_calls, key
):
    assert unknown_qwen38_graph_keys(config_with_removed_key(key)["model"]["qwen38_graph"]) != []
    package = make_package(tmp_path, config=config_with_removed_key(key))
    _out, config, *_ = build(monkeypatch, offload, package, tmp_path)
    qwen = config["model"]["qwen38_graph"]
    assert unknown_qwen38_graph_keys(qwen) == []
    assert qwen["fold"]["head_r2"] == {"filename": "head_r2/model.onnx", "session_options": HEAD_SESSION_OPTIONS}


@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("alias", ["mtp", "drafter"])
def test_a_source_of_the_earlier_config_surface_is_migrated(
    offload, tmp_path, monkeypatch, multidraft_calls, capsys, layered, alias
):
    config = genai_config(layered)
    qwen = config["model"]["qwen38_graph"]
    qwen.update(
        target_s1_graph_id=1, target_s2_graph_id=2, head_s1_graph_id=3, head_s2_graph_id=4, prefill_chunk_tokens=2048
    )
    qwen["mtp"] = {"enabled": True, "prefill_head": False, "skip_refeed": True}
    head = {
        **qwen["fold"].pop("head_r2"),
        "graph_id": 9,
        "session_options": {"custom_ops_libraries": ["libs/earlier.so"]},
    }
    qwen["fold"].update({alias: head, "force_reject": False, "target_graph_id": 8})
    package = make_package(tmp_path, layered=layered, config=config)
    before = snapshot(package.root)

    _out, built, *_ = build(monkeypatch, offload, package, tmp_path, layered=layered)

    qwen = built["model"]["qwen38_graph"]
    assert unknown_qwen38_graph_keys(qwen) == []
    assert qwen["mtp"] == {"enabled": True, "prefill_head": False}
    assert qwen["fold"] == {
        "enabled": True,
        "head_r2": {"filename": "head_r2/model.onnx", "session_options": HEAD_SESSION_OPTIONS},
    }
    assert printed_config_changes(capsys) == sorted(
        f"genai_config: {change}"
        for change in (
            "dropped target_s1_graph_id",
            "dropped target_s2_graph_id",
            "dropped head_s1_graph_id",
            "dropped head_s2_graph_id",
            "dropped mtp.skip_refeed",
            "dropped fold.force_reject",
            "dropped fold.target_graph_id",
            f"renamed fold.{alias} to fold.head_r2",
            "dropped fold.head_r2.graph_id",
            "dropped prefill_chunk_tokens (prefill.chunk is the one chunk setting)",
        )
    )
    assert snapshot(package.root) == before


def test_a_source_that_is_already_on_the_current_surface_is_not_reported_as_changed(
    offload, tmp_path, monkeypatch, multidraft_calls, capsys
):
    package = make_package(tmp_path)
    build(monkeypatch, offload, package, tmp_path)
    assert printed_config_changes(capsys) == []


def test_prefill_chunk_tokens_of_a_source_without_the_dense_prefill_becomes_the_chunk_the_option_replaces(
    offload, tmp_path, monkeypatch, multidraft_calls, capsys
):
    config = genai_config(False)
    qwen = config["model"]["qwen38_graph"]
    qwen["prefill"]["enabled"] = False
    qwen["prefill_chunk_tokens"] = 4096
    package = make_package(tmp_path, config=config)
    _out, built, *_ = build(monkeypatch, offload, package, tmp_path, "--prefill-chunk", "256")
    assert printed_config_changes(capsys) == ["genai_config: renamed prefill_chunk_tokens to prefill.chunk"]
    qwen = built["model"]["qwen38_graph"]
    assert "prefill_chunk_tokens" not in qwen
    assert (qwen["prefill"]["chunk"], prefill_chunk_problem(qwen)) == (256, None)


def test_the_session_labels_of_the_weights_pool_are_not_config_keys_and_stay(
    offload, tmp_path, monkeypatch, multidraft_calls
):
    extra = [("mtp", "model.mtp.norm.weight", 0), ("drafter", "model.drafter.norm.weight", 1)]
    extra.append(("head_r2", "model.head.norm.weight", 0))
    pool = DEFAULT_POOL + "".join(f"session {label} {name} {pid}\n" for label, name, pid in extra)
    package = make_package(tmp_path, pool_text=pool)
    out, *_ = build(monkeypatch, offload, package, tmp_path)
    _records, bindings = read_pool(out / "weights_pool.txt")
    assert [binding for binding in bindings if binding[0] in {label for label, _name, _pid in extra}] == extra


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda model: model.pop("qwen38_graph"), "has no model.qwen38_graph section"),
        (lambda model: model["qwen38_graph"].pop("fold"), r"has no model.qwen38_graph.fold.head_r2 section"),
        (lambda model: model["qwen38_graph"]["fold"].pop("head_r2"), r"has no model.qwen38_graph.fold.head_r2 section"),
        (lambda model: model["qwen38_graph"].pop("prefill"), "has no model.qwen38_graph.prefill section"),
    ],
    ids=["no qwen38_graph", "no fold", "no head_r2", "no prefill"],
)
def test_a_source_config_without_what_the_builder_rewrites_is_refused_before_anything_is_written(
    offload, tmp_path, monkeypatch, multidraft_calls, edit, message
):
    config = genai_config(False)
    edit(config["model"])
    package = make_package(tmp_path, config=config)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=message):
        run_builder(monkeypatch, offload, package, out)
    assert not out.exists()


@pytest.mark.parametrize(
    ("enabled", "chunk", "message"),
    [
        (True, 0, "--prefill-chunk must be positive"),
        (True, -2, "--prefill-chunk must be positive"),
        (True, 4098, r"--prefill-chunk \(4098\) must not exceed model.qwen38_graph.kv_capacity \(4096\)"),
        (False, 0, "--prefill-chunk must be positive"),
        (False, 513, r"--prefill-chunk \(513\) must be even"),
    ],
)
def test_a_prefill_chunk_the_engine_rejects_is_refused_before_anything_is_written(
    offload, tmp_path, monkeypatch, multidraft_calls, enabled, chunk, message
):
    config = genai_config(False)
    config["model"]["qwen38_graph"]["prefill"]["enabled"] = enabled
    package = make_package(tmp_path, config=config)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=message):
        run_builder(monkeypatch, offload, package, out, "--prefill-chunk", chunk)
    assert not out.exists()


@pytest.mark.parametrize(("enabled", "chunk"), [(True, 4096), (True, 513), (True, 1), (False, 2), (False, 4098)])
def test_a_prefill_chunk_the_engine_accepts_is_written(
    offload, tmp_path, monkeypatch, multidraft_calls, enabled, chunk
):
    config = genai_config(False)
    config["model"]["qwen38_graph"]["prefill"]["enabled"] = enabled
    package = make_package(tmp_path, config=config)
    _out, built, *_ = build(monkeypatch, offload, package, tmp_path, "--prefill-chunk", chunk)
    qwen = built["model"]["qwen38_graph"]
    assert qwen["prefill"]["chunk"] == chunk
    assert prefill_chunk_problem(qwen) is None


# ---------------------------------------------------------------------------
# Sessions in the installed runtime
# ---------------------------------------------------------------------------


def stock_decoder(layers=2, op_type="QMoE", extra_attributes=None):
    """A tiny INT4 (block 32) swiglu QMoE decoder with embedded weights that the released CPU runtime can load."""
    experts, hidden, inter, block = 4, 64, 64, 32
    rng = np.random.default_rng(0)
    nodes, initializers = [], []
    graph_inputs = [helper.make_tensor_value_info("x0", TensorProto.FLOAT16, [3, hidden])]
    for layer in range(layers):
        for suffix, array in (
            ("fc1_w", rng.integers(0, 255, (experts, 2 * inter, hidden // 2), dtype=np.uint8)),
            ("fc1_s", rng.random((experts, 2 * inter, hidden // block), dtype=np.float32).astype(np.float16)),
            ("fc2_w", rng.integers(0, 255, (experts, hidden, inter // 2), dtype=np.uint8)),
            ("fc2_s", rng.random((experts, hidden, inter // block), dtype=np.float32).astype(np.float16)),
        ):
            initializers.append(numpy_helper.from_array(array, f"l{layer}_{suffix}"))
        graph_inputs.append(helper.make_tensor_value_info(f"router{layer}", TensorProto.FLOAT16, [3, experts]))
        nodes.append(
            helper.make_node(
                op_type,
                [
                    f"x{layer}",
                    f"router{layer}",
                    f"l{layer}_fc1_w",
                    f"l{layer}_fc1_s",
                    "",
                    f"l{layer}_fc2_w",
                    f"l{layer}_fc2_s",
                ],
                [f"x{layer + 1}"],
                name=f"model.layers.{layer}.mlp",
                domain="com.microsoft",
                activation_type="swiglu",
                swiglu_fusion=1,
                k=2,
                expert_weight_bits=4,
                block_size=block,
                normalize_routing_weights=1,
                **(extra_attributes or {}),
            )
        )
    graph = helper.make_graph(
        nodes,
        "decoder",
        graph_inputs,
        [helper.make_tensor_value_info(f"x{layers}", TensorProto.FLOAT16, [3, hidden])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 9
    return model


def create_session(ort, model, entries=None):
    options = ort.SessionOptions()
    for key, value in (entries or {}).items():
        options.add_session_config_entry(key, value)
    return ort.InferenceSession(
        model.SerializeToString() if isinstance(model, onnx.ModelProto) else str(model),
        options,
        providers=["CPUExecutionProvider"],
    )


@pytest.fixture(scope="module")
def ort():
    runtime = pytest.importorskip("onnxruntime")
    try:
        create_session(runtime, stock_decoder())
    except Exception as error:
        pytest.skip(f"the installed onnxruntime cannot load a tiny QMoE graph: {error}")
    return runtime


@pytest.mark.parametrize("mode", ["required", "auto"])
@pytest.mark.parametrize("placement", ["adaptive", "static"])
def test_the_built_package_creates_a_session_in_the_installed_runtime(
    ort, offload, tmp_path, monkeypatch, multidraft_calls, mode, placement
):
    package = make_package(tmp_path, model=stock_decoder())
    _out, _config, entries, model, decoder_dir = build(
        monkeypatch,
        offload,
        package,
        tmp_path,
        "--offload-mode",
        mode,
        "--offload-placement",
        placement,
        "--target-resident-slots",
        "2",
    )
    assert set(entries) == {PREFIX + key for key in ("mode", "placement", "cpu_execution", "node_overrides")}
    # The released runtime ignores session.moe.offload.*; a build with QMoE offload validates every entry here.
    for graph in ("model.onnx", "model_s2.onnx", "model_s3.onnx", "model_s4.onnx"):
        create_session(ort, decoder_dir / graph, entries)
    assert simulate_session_creation(model, entries, decoder_dir) == [f"model.layers.{layer}.mlp" for layer in range(2)]


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (lambda: stock_decoder(op_type="QMoEOffload"), "QMoEOffload"),
        (lambda: stock_decoder(extra_attributes={"layer_index": 0}), "layer_index"),
        (lambda: stock_decoder(extra_attributes={"resident_slots": 2}), "resident_slots"),
    ],
    ids=["QMoEOffload operator", "layer_index attribute", "resident_slots attribute"],
)
def test_what_the_previous_builder_emitted_does_not_load(ort, graph, message):
    with pytest.raises(Exception, match=message) as excinfo:
        create_session(ort, graph())
    assert type(excinfo.value).__name__ == "InvalidGraph"
