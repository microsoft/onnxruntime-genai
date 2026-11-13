# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import json
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def prefill_converter():
    spec = importlib.util.spec_from_file_location(
        "nemotron_bolt_prefill", ROOT / "examples" / "python" / "nemotron-bolt-prefill.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic_r8_package(tmp_path):
    source = tmp_path / "packed-r8"
    source.mkdir()
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["hidden"], to=int(TensorProto.BFLOAT16)),
        helper.make_node("Reshape", ["hidden", "shape_3d"], ["hidden_3d"]),
        helper.make_node(
            "GroupQueryAttention",
            ["hidden_3d", "kv_3d", "kv_3d", "past_key.7", "past_value.7", "seqlens_k", "total_sequence_length"],
            ["attn_3d", "pk7_tmp", "pv7_tmp"],
            domain="com.microsoft",
            num_heads=4,
            kv_num_heads=2,
            do_rotary=0,
        ),
        helper.make_node("Identity", ["pk7_tmp"], ["present_key.7"]),
        helper.make_node("Identity", ["pv7_tmp"], ["present_value.7"]),
        helper.make_node("Identity", ["attn_3d"], ["tap0"]),
        helper.make_node("Identity", ["attn_3d"], ["tap1"]),
        helper.make_node("Concat", ["tap0", "tap1"], ["aux_cat"], axis=-1),
        helper.make_node("Identity", ["aux_cat"], ["aux_hidden_states"]),
        helper.make_node(
            "MatMulPacked",
            ["attn_3d", "lm_w"],
            ["lm_out"],
            domain="com.microsoft",
            K=32,
            N=64,
            packed_type=8,
        ),
        helper.make_node("Identity", ["lm_out"], ["logits"]),
        helper.make_node("Cast", ["lm_out"], ["lm_f32"], to=int(TensorProto.FLOAT)),
        helper.make_node("ArgMax", ["lm_f32"], ["pred_ids"], axis=-1, keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic_bolt_r8",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 8]),
            helper.make_tensor_value_info("kv_3d", TensorProto.BFLOAT16, [1, 8, 16]),
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, []),
            helper.make_tensor_value_info("past_key.7", TensorProto.BFLOAT16, [1, 2, 1024, 8]),
            helper.make_tensor_value_info("past_value.7", TensorProto.BFLOAT16, [1, 2, 1024, 8]),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.BFLOAT16, [1, 8, 64]),
            helper.make_tensor_value_info("pred_ids", TensorProto.INT64, [1, 8]),
            helper.make_tensor_value_info("aux_hidden_states", TensorProto.BFLOAT16, [1, 8, 64]),
            helper.make_tensor_value_info("present_key.7", TensorProto.BFLOAT16, [1, 2, 1024, 8]),
            helper.make_tensor_value_info("present_value.7", TensorProto.BFLOAT16, [1, 2, 1024, 8]),
        ],
        initializer=[
            helper.make_tensor("shape_3d", TensorProto.INT64, [3], [1, -1, 32]),
            helper.make_tensor("lm_w", TensorProto.UINT8, [64, 34], [0] * (64 * 34)),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)]
    )
    onnx.save(model, str(source / "model.onnx"))
    (source / "weights.bin").write_bytes(bytes(range(128)))
    (source / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 12, "content": "<think>"},
                    {"id": 13, "content": "</think>"},
                    {"id": 14, "content": "<tool_call>"},
                    {"id": 15, "content": "</tool_call>"},
                ]
            }
        )
    )
    config = {
        "model": {
            "type": "nemotron_bolt",
            "context_length": 1016,
            "decoder": {
                "filename": "model.onnx",
                "num_key_value_heads": 2,
                "head_size": 8,
                "session_options": {"provider_options": [{"cuda": {"enable_cuda_graph": "1"}}]},
                "inputs": {
                    "past_sequence_length": "seqlens_k",
                    "total_sequence_length": "total_sequence_length",
                    "past_key_names": "past_key.%d",
                    "past_value_names": "past_value.%d",
                },
                "outputs": {
                    "present_key_names": "present_key.%d",
                    "present_value_names": "present_value.%d",
                },
            },
        },
        "speculative": {"max_draft_tokens": 7},
    }
    (source / "genai_config.json").write_text(json.dumps(config))
    return source


def test_prefill_converter_migrates_gqa_to_paged_attention_and_configures_engine(
    prefill_converter, synthetic_r8_package
):
    output = synthetic_r8_package.parent / "paged-pkg"
    prefill_converter.convert(synthetic_r8_package, output)

    assert (output / "weights.bin").samefile(synthetic_r8_package / "weights.bin")
    assert not (output / "genai_config.json").samefile(synthetic_r8_package / "genai_config.json")
    assert not (output / "tokenizer.json").samefile(synthetic_r8_package / "tokenizer.json")
    out_cfg = json.loads((output / "genai_config.json").read_text())
    assert out_cfg["speculative"]["max_draft_tokens"] == 7
    assert out_cfg["model"]["bor_token_id"] == 12
    assert out_cfg["model"]["eor_token_id"] == 13
    assert out_cfg["model"]["bot_token_id"] == 14
    assert out_cfg["model"]["eot_token_id"] == 15
    assert out_cfg["model"]["decoder"]["state_groups"] == [{"kind": "paged_kv", "layer_ids": [7]}]
    assert out_cfg["model"]["decoder"]["inputs"]["past_key_names"] == "past_key_values.%d.key"
    assert out_cfg["model"]["decoder"]["outputs"]["present_key_names"] == "present.%d.key"
    assert out_cfg["engine"]["dynamic_batching"]["max_batch_size"] == 1
    assert out_cfg["engine"]["dynamic_batching"]["block_size"] == 256
    assert len(out_cfg["runtime_profiles"]) == 5

    model = onnx.load(str(output / "model.onnx"))
    in_shapes = {
        vi.name: [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in model.graph.input
    }
    out_shapes = {
        vi.name: [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in model.graph.output
    }
    assert in_shapes["input_ids"] == [1, "target_rows"]
    assert in_shapes["logit_rows"] == ["logit_row_count"]
    assert in_shapes["aux_rows"] == ["aux_row_count"]
    assert "seqlens_k" not in in_shapes
    assert "total_sequence_length" not in in_shapes
    assert "past_key.7" not in in_shapes
    assert in_shapes["block_table"] == ["batch_size", "max_num_blocks"]
    assert in_shapes["cumulative_sequence_lengths"] == ["batch_size + 1"]
    assert in_shapes["past_sequence_lengths"] == ["batch_size"]
    assert in_shapes["attention_metadata"] == [3]
    assert in_shapes["past_key_values.7.key"] == ["num_blocks", 256, 2, 8]
    assert out_shapes["present.7.key"] == ["num_blocks", 256, 2, 8]
    assert out_shapes["logits"] == [1, "logit_row_count", 64]
    assert out_shapes["pred_ids"] == [1, "logit_row_count"]
    assert out_shapes["aux_hidden_states"] == [1, "aux_row_count", 64]

    op_types = [n.op_type for n in model.graph.node]
    assert "GroupQueryAttention" not in op_types
    assert "PagedAttention" in op_types
    gather_names = [n.name for n in model.graph.node if n.op_type == "Gather"]
    assert gather_names == ["Gather_AuxRows_0", "Gather_AuxRows_1", "Gather_LogitRows"]


def test_prefill_converter_keep_gqa_preserves_dense_kv_contract(prefill_converter, synthetic_r8_package):
    output = synthetic_r8_package.parent / "gqa-pkg"
    prefill_converter.convert(synthetic_r8_package, output, paged_attention=False)

    assert not (output / "genai_config.json").samefile(synthetic_r8_package / "genai_config.json")
    model = onnx.load(str(output / "model.onnx"))
    op_types = [n.op_type for n in model.graph.node]
    assert "GroupQueryAttention" in op_types
    assert "PagedAttention" not in op_types


def test_prefill_converter_injects_num_blocks_into_existing_runtime_profiles(prefill_converter, synthetic_r8_package):
    cfg_path = synthetic_r8_package / "genai_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["model"]["context_length"] = 135160
    cfg["runtime_profiles"] = [
        {
            "id": "consumer-gpu-30g",
            "eligibility": {
                "minimum_total_device_memory_bytes": 26843545601,
                "maximum_total_device_memory_bytes": 32212254720,
            },
            "overlay": {"model": {"decoder": {"session_options": {"session.moe.offload.mode": "auto"}}}},
        },
        {
            "id": "consumer-gpu-32g",
            "eligibility": {
                "minimum_total_device_memory_bytes": 32212254721,
                "maximum_total_device_memory_bytes": 53687091200,
            },
            "overlay": {"model": {"decoder": {"session_options": {"session.moe.offload.mode": "disabled"}}}},
        },
        {
            "id": "gpu-132k",
            "eligibility": {"minimum_total_device_memory_bytes": 53687091201},
            "overlay": {"model": {"decoder": {"session_options": {"session.moe.offload.mode": "disabled"}}}},
        },
    ]
    cfg_path.write_text(json.dumps(cfg))

    output = synthetic_r8_package.parent / "paged-existing-profiles"
    prefill_converter.convert(synthetic_r8_package, output)
    out_cfg = json.loads((output / "genai_config.json").read_text())
    by_id = {p["id"]: p["overlay"]["engine"]["dynamic_batching"]["num_blocks"] for p in out_cfg["runtime_profiles"]}
    assert by_id["consumer-gpu-30g"] < by_id["consumer-gpu-32g"] < by_id["gpu-132k"]


def test_prefill_converter_banks_mamba2_verify_and_enables_continuous_batching(prefill_converter, synthetic_r8_package):
    model = onnx.load(str(synthetic_r8_package / "model.onnx"))
    graph = model.graph
    graph.input.extend(
        [
            helper.make_tensor_value_info("state_row", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("past_conv.0", TensorProto.BFLOAT16, [8, 10240, 3]),
            helper.make_tensor_value_info("past_ssm.0", TensorProto.FLOAT, [8, 128, 128, 64]),
        ]
    )
    graph.output.extend(
        [
            helper.make_tensor_value_info("present_conv.0", TensorProto.BFLOAT16, [8, 10240, 3]),
            helper.make_tensor_value_info("present_ssm.0", TensorProto.FLOAT, [8, 128, 128, 64]),
        ]
    )
    graph.initializer.extend(
        [
            helper.make_tensor("m_proj", TensorProto.BFLOAT16, [8, 18560], [0] * (8 * 18560)),
            helper.make_tensor("m_cw", TensorProto.BFLOAT16, [10240, 4], [0] * (10240 * 4)),
            helper.make_tensor("m_cb", TensorProto.BFLOAT16, [10240], [0] * 10240),
            helper.make_tensor("m_dtb", TensorProto.FLOAT, [128], [0.0] * 128),
            helper.make_tensor("m_aneg", TensorProto.FLOAT, [128], [-0.1] * 128),
            helper.make_tensor("m_d", TensorProto.FLOAT, [128], [1.0] * 128),
            helper.make_tensor("m_nw", TensorProto.FLOAT, [8192], [1.0] * 8192),
        ]
    )
    graph.node.insert(
        0,
        helper.make_node(
            "Mamba2Verify",
            [
                "m_proj",
                "past_conv.0",
                "past_ssm.0",
                "m_cw",
                "m_cb",
                "m_dtb",
                "m_aneg",
                "m_d",
                "m_nw",
                "state_row",
            ],
            ["m_y", "present_conv.0", "present_ssm.0"],
            domain="com.microsoft",
            name="Mamba2Verify_0",
            heads=128,
            n_groups=8,
        ),
    )
    onnx.save(model, str(synthetic_r8_package / "model.onnx"))

    output = synthetic_r8_package.parent / "paged-cb-pkg"
    prefill_converter.convert(synthetic_r8_package, output)

    out_cfg = json.loads((output / "genai_config.json").read_text())
    assert out_cfg["engine"]["dynamic_batching"]["max_batch_size"] == 4
    profiles = {p["id"]: p["overlay"]["engine"]["dynamic_batching"] for p in out_cfg["runtime_profiles"]}
    assert profiles["consumer-gpu-30g"]["max_batch_size"] == 1
    assert profiles["consumer-gpu-32g"]["max_batch_size"] == 1
    assert "max_batch_size" not in profiles["gpu-132k"]

    out_model = onnx.load(str(output / "model.onnx"))
    in_shapes = {
        vi.name: [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in out_model.graph.input
    }
    out_shapes = {
        vi.name: [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in out_model.graph.output
    }
    assert in_shapes["state_row"] == ["batch_size"]
    assert in_shapes["state_slots"] == ["batch_size"]
    assert in_shapes["past_conv.0"] == ["num_state_banks", 8, 10240, 3]
    assert in_shapes["past_ssm.0"] == ["num_state_banks", 8, 128, 128, 64]
    assert out_shapes["present_conv.0"] == ["num_state_banks", 8, 10240, 3]
    assert out_shapes["present_ssm.0"] == ["num_state_banks", 8, 128, 128, 64]
    mamba_node = next(n for n in out_model.graph.node if n.op_type == "Mamba2Verify")
    assert list(mamba_node.input[9:12]) == ["state_row", "cumulative_sequence_lengths", "state_slots"]
