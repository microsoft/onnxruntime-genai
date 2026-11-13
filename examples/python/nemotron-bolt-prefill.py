"""Convert a packed R8 Nemotron-Bolt package for dynamic large-chunk prefill and GenAI Engine.

By default, converts the eight GroupQueryAttention layers to com.microsoft:PagedAttention,
adds selective logit_rows and aux_rows gathering for fast large-chunk prefill, hardlinks weight
payloads without repacking, and writes Engine + runtime_profiles configuration to genai_config.json.
Pass --keep-gqa to retain dense GroupQueryAttention caches instead of PagedAttention.
Temporary conversion helper until the Olive recipe emits dynamic-prefill and PagedAttention
packages directly (remove before merging into main).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper

PAGED_BLOCK_SIZE = 256
PREFILL_CHUNK_TOKENS = 4096
DEFAULT_MAX_BATCH_SIZE = 4
MUTABLE_METADATA_SUFFIXES = {".json", ".jinja", ".txt", ".md", ".yaml", ".yml"}


def _get_int_attr(node: onnx.NodeProto, name: str, default: int = 0) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _extract_protocol_tokens(source: Path, model_cfg: dict) -> None:
    tok_path = source / "tokenizer.json"
    if not tok_path.is_file():
        return
    tok_data = json.loads(tok_path.read_text())
    token_map = {entry.get("content"): int(entry["id"]) for entry in tok_data.get("added_tokens", []) if "id" in entry}
    for token_str, cfg_key in (
        ("<think>", "bor_token_id"),
        ("</think>", "eor_token_id"),
        ("<tool_call>", "bot_token_id"),
        ("</tool_call>", "eot_token_id"),
    ):
        if cfg_key not in model_cfg and token_str in token_map:
            model_cfg[cfg_key] = token_map[token_str]


def _ensure_cuda_arena_extend_strategy(session_options: dict) -> None:
    for entry in session_options.get("provider_options", []):
        if "cuda" in entry and isinstance(entry["cuda"], dict):
            entry["cuda"].setdefault("arena_extend_strategy", "kSameAsRequested")


def _default_bolt_runtime_profiles(blocks_32k: int, blocks_64k: int, blocks_full: int) -> list[dict]:
    return [
        {
            "id": "consumer-gpu-24g",
            "eligibility": {
                "minimum_total_device_memory_bytes": 21474836480,
                "maximum_total_device_memory_bytes": 26843545600,
                "minimum_effective_host_memory_bytes": 68719476736,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": "model.onnx",
                        "session_options": {
                            "session.moe.offload.mode": "auto",
                            "session.moe.offload.placement": "adaptive",
                            "session.moe.offload.cpu_execution": "auto",
                            "session.moe.offload.device_expert_bytes": "9663676416",
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": blocks_32k,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": PREFILL_CHUNK_TOKENS,
                    }
                },
            },
        },
        {
            "id": "consumer-gpu-24g-host-64g",
            "eligibility": {
                "minimum_total_device_memory_bytes": 21474836480,
                "maximum_total_device_memory_bytes": 26843545600,
                "maximum_effective_host_memory_bytes": 68719476735,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": "model.onnx",
                        "session_options": {
                            "session.moe.offload.mode": "auto",
                            "session.moe.offload.placement": "adaptive",
                            "session.moe.offload.cpu_execution": "auto",
                            "session.moe.offload.device_expert_bytes": "9663676416",
                            "session.moe.offload.host_expert_bytes": "17179869184",
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": blocks_32k,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": PREFILL_CHUNK_TOKENS,
                    }
                },
            },
        },
        {
            "id": "consumer-gpu-30g",
            "eligibility": {
                "minimum_total_device_memory_bytes": 26843545601,
                "maximum_total_device_memory_bytes": 32212254720,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": "model.onnx",
                        "session_options": {
                            "session.moe.offload.mode": "auto",
                            "session.moe.offload.placement": "adaptive",
                            "session.moe.offload.cpu_execution": "auto",
                            "session.moe.offload.device_expert_bytes": "13958643712",
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": blocks_32k,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": PREFILL_CHUNK_TOKENS,
                    }
                },
            },
        },
        {
            "id": "consumer-gpu-32g",
            "eligibility": {
                "minimum_total_device_memory_bytes": 32212254721,
                "maximum_total_device_memory_bytes": 53687091200,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": "model.onnx",
                        "session_options": {
                            "session.moe.offload.mode": "disabled",
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": blocks_64k,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": PREFILL_CHUNK_TOKENS,
                    }
                },
            },
        },
        {
            "id": "gpu-132k",
            "eligibility": {
                "minimum_total_device_memory_bytes": 53687091201,
                "is_integrated": False,
            },
            "overlay": {
                "model": {
                    "decoder": {
                        "filename": "model.onnx",
                        "session_options": {
                            "session.moe.offload.mode": "disabled",
                        },
                    }
                },
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": blocks_full,
                        "max_scheduled_tokens": PREFILL_CHUNK_TOKENS,
                    }
                },
            },
        },
    ]


def _convert_gqa_to_paged_attention(graph: onnx.GraphProto) -> list[int]:
    gqa_nodes = [n for n in graph.node if n.op_type == "GroupQueryAttention"]
    if not gqa_nodes:
        return []

    in_info = {vi.name: vi for vi in graph.input}
    first_gqa = gqa_nodes[0]
    num_heads = _get_int_attr(first_gqa, "num_heads", 32)
    kv_num_heads = _get_int_attr(first_gqa, "kv_num_heads", 2)
    do_rotary = _get_int_attr(first_gqa, "do_rotary", 0)
    first_pk = in_info[first_gqa.input[3]]
    head_size = int(first_pk.type.tensor_type.shape.dim[3].dim_value or 128)
    q_hidden = num_heads * head_size
    kv_hidden = kv_num_heads * head_size

    graph.initializer.extend(
        [
            helper.make_tensor("bolt_paged_q_shape_2d", TensorProto.INT64, [2], [-1, q_hidden]),
            helper.make_tensor("bolt_paged_kv_shape_2d", TensorProto.INT64, [2], [-1, kv_hidden]),
            helper.make_tensor("bolt_paged_out_shape_3d", TensorProto.INT64, [3], [1, -1, q_hidden]),
        ]
    )

    layer_ids: list[int] = []
    removed_inputs = {"seqlens_k", "total_sequence_length"}
    removed_outputs: set[str] = set()
    dead_present_identities: set[str] = set()

    for n in gqa_nodes:
        pk_name = n.input[3]
        pv_name = n.input[4]
        layer_id = int(pk_name.split(".")[1])
        layer_ids.append(layer_id)
        removed_inputs.add(pk_name)
        removed_inputs.add(pv_name)
        removed_outputs.add(f"present_key.{layer_id}")
        removed_outputs.add(f"present_value.{layer_id}")
        if len(n.output) > 1 and n.output[1]:
            dead_present_identities.add(n.output[1])
        if len(n.output) > 2 and n.output[2]:
            dead_present_identities.add(n.output[2])

    rebuilt_nodes: list[onnx.NodeProto] = []
    for n in graph.node:
        if n.op_type == "Identity" and n.input and n.input[0] in dead_present_identities:
            continue
        if n.op_type != "GroupQueryAttention":
            rebuilt_nodes.append(n)
            continue
        layer_id = int(n.input[3].split(".")[1])
        q_2d = f"{n.input[0]}_paged_q_2d"
        k_2d = f"{n.input[1]}_paged_k_2d"
        v_2d = f"{n.input[2]}_paged_v_2d"
        attn_2d = f"{n.output[0]}_paged_out_2d"
        rebuilt_nodes.extend(
            [
                helper.make_node(
                    "Reshape",
                    [n.input[0], "bolt_paged_q_shape_2d"],
                    [q_2d],
                    name=f"Reshape_PagedQ_{layer_id}",
                ),
                helper.make_node(
                    "Reshape",
                    [n.input[1], "bolt_paged_kv_shape_2d"],
                    [k_2d],
                    name=f"Reshape_PagedK_{layer_id}",
                ),
                helper.make_node(
                    "Reshape",
                    [n.input[2], "bolt_paged_kv_shape_2d"],
                    [v_2d],
                    name=f"Reshape_PagedV_{layer_id}",
                ),
                helper.make_node(
                    "PagedAttention",
                    [
                        q_2d,
                        k_2d,
                        v_2d,
                        f"past_key_values.{layer_id}.key",
                        f"past_key_values.{layer_id}.value",
                        "cumulative_sequence_lengths",
                        "past_sequence_lengths",
                        "block_table",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "attention_metadata",
                    ],
                    [
                        attn_2d,
                        f"present.{layer_id}.key",
                        f"present.{layer_id}.value",
                    ],
                    domain="com.microsoft",
                    name=f"PagedAttention_{layer_id}",
                    num_heads=num_heads,
                    kv_num_heads=kv_num_heads,
                    do_rotary=do_rotary,
                    is_causal=1,
                ),
                helper.make_node(
                    "Reshape",
                    [attn_2d, "bolt_paged_out_shape_3d"],
                    [n.output[0]],
                    name=f"Reshape_PagedOut_{layer_id}",
                ),
            ]
        )

    del graph.node[:]
    graph.node.extend(rebuilt_nodes)

    kept_inputs = [vi for vi in graph.input if vi.name not in removed_inputs]
    kept_inputs.extend(
        [
            helper.make_tensor_value_info("cumulative_sequence_lengths", TensorProto.INT32, ["batch_size + 1"]),
            helper.make_tensor_value_info("past_sequence_lengths", TensorProto.INT32, ["batch_size"]),
            helper.make_tensor_value_info("block_table", TensorProto.INT32, ["batch_size", "max_num_blocks"]),
            helper.make_tensor_value_info("attention_metadata", TensorProto.INT32, [3]),
        ]
    )
    cache_shape = ["num_blocks", PAGED_BLOCK_SIZE, kv_num_heads, head_size]
    for layer_id in layer_ids:
        kept_inputs.append(
            helper.make_tensor_value_info(f"past_key_values.{layer_id}.key", TensorProto.BFLOAT16, cache_shape)
        )
        kept_inputs.append(
            helper.make_tensor_value_info(f"past_key_values.{layer_id}.value", TensorProto.BFLOAT16, cache_shape)
        )
    del graph.input[:]
    graph.input.extend(kept_inputs)

    kept_outputs = [vi for vi in graph.output if vi.name not in removed_outputs]
    for layer_id in layer_ids:
        kept_outputs.append(helper.make_tensor_value_info(f"present.{layer_id}.key", TensorProto.BFLOAT16, cache_shape))
        kept_outputs.append(
            helper.make_tensor_value_info(f"present.{layer_id}.value", TensorProto.BFLOAT16, cache_shape)
        )
    del graph.output[:]
    graph.output.extend(kept_outputs)

    return layer_ids


def _convert_mamba2_verify_to_banked_batch(graph: onnx.GraphProto) -> bool:
    mamba_nodes = [n for n in graph.node if n.op_type == "Mamba2Verify"]
    if not mamba_nodes:
        return False

    for n in mamba_nodes:
        if len(n.input) == 10:
            n.input.extend(["cumulative_sequence_lengths", "state_slots"])
        elif len(n.input) >= 12:
            n.input[10] = "cumulative_sequence_lengths"
            n.input[11] = "state_slots"
        else:
            raise RuntimeError(f"Unexpected Mamba2Verify input count ({len(n.input)}) on node {n.name}")

    def _prepend_bank_dim(vi: onnx.ValueInfoProto, expected_rank: int) -> None:
        shape = vi.type.tensor_type.shape
        if len(shape.dim) == expected_rank:
            dims = list(shape.dim)
            del shape.dim[:]
            bank_dim = shape.dim.add()
            bank_dim.dim_param = "num_state_banks"
            shape.dim.extend(dims)
        elif len(shape.dim) == expected_rank + 1:
            shape.dim[0].ClearField("dim_value")
            shape.dim[0].dim_param = "num_state_banks"
        else:
            raise RuntimeError(f"Unexpected recurrent tensor rank ({len(shape.dim)}) for {vi.name}")

    has_state_slots = False
    for vi in graph.input:
        if vi.name == "state_row":
            del vi.type.tensor_type.shape.dim[:]
            dim = vi.type.tensor_type.shape.dim.add()
            dim.dim_param = "batch_size"
        elif vi.name == "state_slots":
            has_state_slots = True
            del vi.type.tensor_type.shape.dim[:]
            dim = vi.type.tensor_type.shape.dim.add()
            dim.dim_param = "batch_size"
        elif vi.name.startswith("past_conv."):
            _prepend_bank_dim(vi, 3)
        elif vi.name.startswith("past_ssm."):
            _prepend_bank_dim(vi, 4)

    if not has_state_slots:
        graph.input.append(helper.make_tensor_value_info("state_slots", TensorProto.INT32, ["batch_size"]))

    for vi in graph.output:
        if vi.name.startswith("present_conv."):
            _prepend_bank_dim(vi, 3)
        elif vi.name.startswith("present_ssm."):
            _prepend_bank_dim(vi, 4)

    return True


def convert(
    source: Path,
    output: Path,
    *,
    paged_attention: bool = True,
    max_batch_size: int | None = None,
) -> None:
    source = source.resolve(strict=True)
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    config = copy.deepcopy(json.loads((source / "genai_config.json").read_text()))
    if config["model"]["type"] != "nemotron_bolt":
        raise ValueError("Expected a nemotron_bolt package")
    filename = config["model"]["decoder"]["filename"]
    model = onnx.load(source / filename, load_external_data=False)
    graph = model.graph

    # 1. Update input_ids, logits, pred_ids, aux_hidden_states shapes to independent symbolic row dimensions
    for vi in list(graph.input) + list(graph.output):
        if vi.name == "input_ids":
            vi.type.tensor_type.shape.dim[1].ClearField("dim_value")
            vi.type.tensor_type.shape.dim[1].dim_param = "target_rows"
        elif vi.name in ("logits", "pred_ids"):
            vi.type.tensor_type.shape.dim[1].ClearField("dim_value")
            vi.type.tensor_type.shape.dim[1].dim_param = "logit_row_count"
        elif vi.name == "aux_hidden_states":
            vi.type.tensor_type.shape.dim[1].ClearField("dim_value")
            vi.type.tensor_type.shape.dim[1].dim_param = "aux_row_count"

    # Clear stale internal value_info with fixed R=8 so ORT re-infers dynamic shapes cleanly
    del graph.value_info[:]

    # 2. Add logit_rows and aux_rows with independent symbolic row counts
    existing_inputs = {vi.name for vi in graph.input}
    if "logit_rows" not in existing_inputs:
        graph.input.append(helper.make_tensor_value_info("logit_rows", TensorProto.INT64, ["logit_row_count"]))
    if "aux_rows" not in existing_inputs:
        graph.input.append(helper.make_tensor_value_info("aux_rows", TensorProto.INT64, ["aux_row_count"]))

    has_gather_lm = any(n.name == "Gather_LogitRows" for n in graph.node)
    if not has_gather_lm:
        logits_src = "logits"
        for n in graph.node:
            if n.op_type == "Identity" and "logits" in n.output:
                logits_src = n.input[0]
                break

        lm_head_idx = None
        for idx, n in enumerate(graph.node):
            if logits_src in n.output and n.op_type == "MatMulPacked":
                lm_head_idx = idx
                break
        if lm_head_idx is None:
            raise RuntimeError("Could not locate LM head MatMulPacked node")

        lm_head_node = graph.node[lm_head_idx]
        orig_lm_in = lm_head_node.input[0]
        gathered_lm_in = orig_lm_in + "_selected_logit_rows"
        gather_lm_node = helper.make_node(
            "Gather",
            inputs=[orig_lm_in, "logit_rows"],
            outputs=[gathered_lm_in],
            axis=1,
            name="Gather_LogitRows",
        )
        lm_head_node.input[0] = gathered_lm_in

        aux_src = "aux_hidden_states"
        for n in graph.node:
            if n.op_type == "Identity" and "aux_hidden_states" in n.output:
                aux_src = n.input[0]
                break

        aux_concat_idx = None
        for idx, n in enumerate(graph.node):
            if aux_src in n.output and n.op_type == "Concat":
                aux_concat_idx = idx
                break
        if aux_concat_idx is None:
            raise RuntimeError("Could not locate aux_hidden_states Concat node")

        aux_concat_node = graph.node[aux_concat_idx]
        aux_gather_nodes = []
        for tap_i, tap_name in enumerate(aux_concat_node.input):
            selected_tap = f"{tap_name}_selected_aux_rows_{tap_i}"
            aux_gather_nodes.append(
                helper.make_node(
                    "Gather",
                    inputs=[tap_name, "aux_rows"],
                    outputs=[selected_tap],
                    axis=1,
                    name=f"Gather_AuxRows_{tap_i}",
                )
            )
            aux_concat_node.input[tap_i] = selected_tap

        new_nodes = []
        for n in graph.node:
            if n is aux_concat_node:
                new_nodes.extend(aux_gather_nodes)
            if n is lm_head_node:
                new_nodes.append(gather_lm_node)
            new_nodes.append(n)
        del graph.node[:]
        graph.node.extend(new_nodes)

    # 3. Optionally migrate GroupQueryAttention layers to PagedAttention and Mamba2Verify to banked recurrent state
    paged_layer_ids = _convert_gqa_to_paged_attention(graph) if paged_attention else []
    banked_mamba = _convert_mamba2_verify_to_banked_batch(graph) if paged_layer_ids else False
    effective_max_batch_size = (
        int(max_batch_size) if max_batch_size is not None else (DEFAULT_MAX_BATCH_SIZE if banked_mamba else 1)
    )
    if effective_max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")
    if effective_max_batch_size > 1 and not banked_mamba:
        raise ValueError("max_batch_size > 1 requires PagedAttention and Mamba2Verify nodes")

    model_cfg = config["model"]
    _extract_protocol_tokens(source, model_cfg)
    decoder_cfg = model_cfg["decoder"]
    decoder_so = decoder_cfg.setdefault("session_options", {})
    _ensure_cuda_arena_extend_strategy(decoder_so)
    if "dflash2" in model_cfg and isinstance(model_cfg["dflash2"], dict):
        _ensure_cuda_arena_extend_strategy(model_cfg["dflash2"].setdefault("session_options", {}))

    if paged_layer_ids:
        decoder_so["session.paged_attention.skip_input_sanitizer"] = "1"
        dec_inputs = decoder_cfg.setdefault("inputs", {})
        dec_inputs.pop("past_sequence_length", None)
        dec_inputs.pop("total_sequence_length", None)
        dec_inputs["input_ids"] = "input_ids"
        dec_inputs["cumulative_sequence_lengths"] = "cumulative_sequence_lengths"
        dec_inputs["past_sequence_lengths"] = "past_sequence_lengths"
        dec_inputs["block_table"] = "block_table"
        dec_inputs["attention_metadata"] = "attention_metadata"
        dec_inputs["past_key_names"] = "past_key_values.%d.key"
        dec_inputs["past_value_names"] = "past_key_values.%d.value"
        dec_outputs = decoder_cfg.setdefault("outputs", {})
        dec_outputs["present_key_names"] = "present.%d.key"
        dec_outputs["present_value_names"] = "present.%d.value"
        if int(decoder_cfg.get("num_hidden_layers", 0)) <= max(paged_layer_ids):
            decoder_cfg["num_hidden_layers"] = max(paged_layer_ids) + 1
        decoder_cfg["state_groups"] = [{"kind": "paged_kv", "layer_ids": paged_layer_ids}]

        context_len = int(model_cfg.get("context_length", 135160))
        kv_heads = int(decoder_cfg.get("num_key_value_heads", 2))
        head_size = int(decoder_cfg.get("head_size", 128))
        target_block_bytes = len(paged_layer_ids) * 2 * PAGED_BLOCK_SIZE * kv_heads * head_size * 2
        aux_reserved_1 = 0
        aux_reserved_full = 0
        if "dflash2" in model_cfg and isinstance(model_cfg["dflash2"], dict):
            dflash = model_cfg["dflash2"]
            df_layers = int(dflash.get("num_hidden_layers", 5))
            df_kv_heads = int(dflash.get("num_key_value_heads", 2))
            df_head_size = int(dflash.get("head_size", 128))
            df_window = int(dflash.get("sliding_window", 1024))
            df_block = int(dflash.get("block_size", 8))
            df_pages = math.ceil((max(df_window, 0) + 2 * df_block) / PAGED_BLOCK_SIZE) + 1
            per_req_df_bytes = df_pages * df_layers * 2 * PAGED_BLOCK_SIZE * df_kv_heads * df_head_size * 2 + 196608
            aux_reserved_1 = math.ceil(per_req_df_bytes / max(target_block_bytes, 1))
            aux_reserved_full = math.ceil((effective_max_batch_size * per_req_df_bytes) / max(target_block_bytes, 1))

        full_target_blocks = math.ceil((context_len + 8) / PAGED_BLOCK_SIZE)
        blocks_full = full_target_blocks + aux_reserved_full
        blocks_64k = min(full_target_blocks, 257) + aux_reserved_1
        blocks_32k = min(full_target_blocks, 129) + aux_reserved_1
        config["engine"] = {
            "dynamic_batching": {
                "block_size": PAGED_BLOCK_SIZE,
                "num_blocks": blocks_full,
                "max_batch_size": effective_max_batch_size,
                "max_scheduled_tokens": PREFILL_CHUNK_TOKENS,
            }
        }
        if "runtime_profiles" not in config:
            config["runtime_profiles"] = _default_bolt_runtime_profiles(blocks_32k, blocks_64k, blocks_full)
        else:
            for prof in config["runtime_profiles"]:
                pid = str(prof.get("id", ""))
                overlay = prof.setdefault("overlay", {})
                eng_db = overlay.setdefault("engine", {}).setdefault("dynamic_batching", {})
                is_consumer_tier = any(tag in pid for tag in ("24g", "30g", "32g", "64k"))
                if "num_blocks" not in eng_db:
                    if "24g" in pid or "30g" in pid:
                        eng_db["num_blocks"] = blocks_32k
                    elif "32g" in pid or "64k" in pid:
                        eng_db["num_blocks"] = blocks_64k
                    else:
                        eng_db["num_blocks"] = blocks_full
                if is_consumer_tier:
                    eng_db.setdefault("max_batch_size", 1)
                else:
                    eng_db.pop("max_batch_size", None)
                eng_db.setdefault("max_scheduled_tokens", PREFILL_CHUNK_TOKENS)
                if overlay.get("speculative", {}).get("max_draft_tokens") == config.get("speculative", {}).get(
                    "max_draft_tokens"
                ):
                    overlay.pop("speculative", None)

    output.mkdir(parents=True)
    for path in source.iterdir():
        if path.name in (filename, "genai_config.json"):
            continue
        resolved = path.resolve(strict=True)
        is_mutable_metadata = path.suffix.lower() in MUTABLE_METADATA_SUFFIXES and resolved.stat().st_size < (64 << 20)
        if not is_mutable_metadata and resolved.stat().st_dev == output.stat().st_dev:
            os.link(resolved, output / path.name)
        elif resolved.stat().st_size < (64 << 20):
            shutil.copy2(resolved, output / path.name)
        else:
            raise ValueError(f"Large payload {path} is on a different filesystem from {output}")

    onnx.save_model(model, str(output / filename), save_as_external_data=False)
    (output / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--keep-gqa",
        action="store_true",
        help="Retain dense GroupQueryAttention instead of converting to PagedAttention",
    )
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Default Engine max_batch_size for banked PagedAttention packages (default: 4 when Mamba2Verify is present)",
    )
    args = ap.parse_args()
    convert(args.input, args.output, paged_attention=not args.keep_gqa, max_batch_size=args.max_batch_size)
