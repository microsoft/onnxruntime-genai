# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Procedural exporter used by the Qwen 3.8 Flash model builder."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import onnx

from .calibration import calibrate_target
from .graph import validate_config
from .target_export import WEIGHT_RECIPE, export_target, finalize_target


def package_config(target_config, draft_config, dflash_section=None):
    validate_config(target_config)
    text = target_config["text_config"]
    draft = draft_config.get("dflash_config", {})
    if (
        draft_config.get("num_hidden_layers") != 5
        or draft_config.get("hidden_size") != 2560
        or draft_config.get("num_key_value_heads") != 8
        or draft_config.get("head_dim") != 128
    ):
        raise ValueError("Expected the five-layer Qwen Flash DFlash2 raw checkpoint")
    if draft.get("target_layer_ids", [5, 15, 25, 35, 45]) != [5, 15, 25, 35, 45]:
        raise ValueError("DFlash2 auxiliary capture layers must match the procedural target")
    eos = text["eos_token_id"]
    config = {
        "model": {
            "type": "qwen38_flash_graph",
            "bos_token_id": text.get("bos_token_id", eos),
            "pad_token_id": eos,
            "eos_token_id": [248046, eos],
            "vocab_size": text["vocab_size"],
            "context_length": 262144,
            "decoder": {
                "filename": "model.onnx",
                "head_size": 256,
                "hidden_size": 2560,
                "num_attention_heads": 24,
                "num_hidden_layers": 48,
                "num_key_value_heads": 2,
                "session_options": {
                    "graph_optimization_level": "ORT_DISABLE_ALL",
                    "session.use_device_allocator_for_initializers": "1",
                    "provider_options": [
                        {
                            "cuda": {
                                "enable_cuda_graph": "1",
                                "arena_extend_strategy": "kSameAsRequested",
                                "min_num_runs_before_cuda_graph_capture": "4",
                            }
                        }
                    ],
                },
            },
            "qwen38_graph": {
                "ple_table": "ple_table.txt",
                "kv_capacity": 262148,
                "warmup": False,
                "mtp": {"enabled": True, "prefill_head": False},
                "fold": {"enabled": True, "target_graph_id": 71},
                "prefill": {
                    "weights_pool": "weights_pool.txt",
                    "enabled": True,
                    "chunk": 8192,
                    "warmup_lengths": [8192, 512, 128, 37],
                    "unified": True,
                },
                "prefill_chunk_tokens": 8192,
            },
        },
        "search": {
            "do_sample": False,
            "top_k": 1,
            "temperature": 1.0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_length": 262144,
            "min_length": 0,
            "past_present_share_buffer": False,
        },
    }
    if dflash_section is not None:
        expected = {
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "head_size": 128,
            "block_size": 4,
            "num_draft_tokens": 3,
        }
        if any(dflash_section.get(name) != value for name, value in expected.items()):
            raise ValueError("Exported DFlash metadata does not match the native three-draft contract")
        section = dict(dflash_section)
        section["session_options"] = dict(section.get("session_options", {}))
        section["session_options"]["provider_options"] = [{"cuda": {"arena_extend_strategy": "kSameAsRequested"}}]
        config["model"]["dflash2"] = section
    return config


def runtime_environment(ort_home):
    environment = dict(os.environ)
    core = (Path(ort_home) / "lib/libonnxruntime.so").resolve(strict=True)
    environment["ORT_LIB_PATH"] = str(core)
    environment["LD_LIBRARY_PATH"] = str(core.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
    for key, value in {
        "ORT_DSA_INT8_GROUPED_MAX_ROWS": "4",
        "ORT_QMOE_INT2_DECODE_MAX_EXPANDED_ROWS": "40",
        "ORT_HC_DECODE_ROWS": "1",
        "ORT_INT2_S4_CTA_N": "2",
        "ORT_INT2_S4_FUSED_FINALIZE": "1",
    }.items():
        environment.setdefault(key, value)
    return environment


def _build_model(
    target_checkpoint,
    draft_checkpoint,
    output,
    prompts,
    provider,
    ort_home,
    *,
    workers=8,
    threads=1,
    calibration_capacity=4096,
    resume=False,
):
    target_checkpoint, draft_checkpoint, prompts, provider, ort_home = (
        Path(value).resolve(strict=True) for value in (target_checkpoint, draft_checkpoint, prompts, provider, ort_home)
    )
    output = Path(output).resolve()
    package = Path(__file__).parent
    models_root = package.parents[1]
    environment = runtime_environment(ort_home)
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(models_root) + (os.pathsep + python_path if python_path else "")
    environment.setdefault("OMP_NUM_THREADS", str(threads))
    environment.setdefault("MKL_NUM_THREADS", str(threads))
    config = json.loads((target_checkpoint / "config.json").read_text())
    draft_config = json.loads((draft_checkpoint / "config.json").read_text())
    package_config(config, draft_config)
    identity = {
        "target_checkpoint": str(target_checkpoint),
        "draft_checkpoint": str(draft_checkpoint),
        "target_config_sha256": hashlib.sha256((target_checkpoint / "config.json").read_bytes()).hexdigest(),
        "draft_config_sha256": hashlib.sha256((draft_checkpoint / "config.json").read_bytes()).hexdigest(),
        "calibration_prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
    }
    if output.exists():
        if not resume:
            if output.is_dir() and not any(output.iterdir()):
                output.rmdir()
            else:
                raise FileExistsError(output)
    if output.exists():
        report = json.loads((output / "target_export.json").read_text())
        if report.get("checkpoint") != str(target_checkpoint) or not report.get("complete"):
            raise ValueError("Resume requires a completed raw-target export from the same checkpoint")
        if report.get("weight_recipe") != WEIGHT_RECIPE:
            raise ValueError("Resume requires target weights converted with the current procedural weight recipe")
        state_path = output / "model_build.json"
        if state_path.exists() and json.loads(state_path.read_text()).get("inputs") != identity:
            raise ValueError("Build inputs changed; use a new output directory")
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Target checkpoint configuration changed since raw export")
    else:
        export_target(
            target_checkpoint, output, workers=workers, threads=threads, calibration_capacity=calibration_capacity
        )
    state = {
        "format": "qwen38-procedural-build-v1",
        "inputs": identity,
        "stage": "calibration",
        "package_ready": False,
        "runtime_smoke_passed": False,
    }

    def record(stage):
        state["stage"] = stage
        (output / "model_build.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps({"stage": stage, "output": str(output)}), flush=True)

    record("calibration")
    profile_path = output / "kv_calibration.json"
    if not profile_path.exists():
        calibrate_target(output, target_checkpoint, prompts, provider, ort_home)
    else:
        profile = json.loads(profile_path.read_text())
        if (
            profile.get("prompts_sha256") != identity["calibration_prompts_sha256"]
            or profile.get("weight_recipe") != WEIGHT_RECIPE
            or profile.get("graph_sha256")
            != hashlib.sha256((output / "target.calibration.onnx").read_bytes()).hexdigest()
        ):
            raise ValueError("Resume calibration prompts, graph or weight recipe do not match the fresh profile")
    record("finalizing_target")
    if not (output / "model.onnx").exists():
        finalize_target(output)
    record("exporting_dflash")
    draft_output = output / ".draft_export"
    if not (output / "dflash2.onnx").exists():
        if draft_output.exists():
            raise FileExistsError(f"Incomplete DFlash export must be inspected before retrying: {draft_output}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "builders.qwen3_8_flash.dflash_export",
                "--checkpoint",
                str(draft_checkpoint),
                "--target",
                str(output / "model.onnx"),
                "--output",
                str(draft_output),
                "--precision",
                "fp8",
                "--drafts",
                "3",
            ],
            env=environment,
            check=True,
        )
        for name in ("dflash2.onnx", "dflash2.onnx.data", "dflash2_config.json"):
            (draft_output / name).replace(output / name)
        draft_output.rmdir()
    record("assembling_package")
    assembled_config = package_config(config, draft_config, json.loads((output / "dflash2_config.json").read_text()))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "chat_template.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    ):
        source = target_checkpoint / name
        if source.is_file():
            shutil.copy2(source, output / name)
    if not (output / "tokenizer.json").is_file():
        raise FileNotFoundError("Raw target checkpoint must contain tokenizer.json")
    (output / "genai_config.json").write_text(json.dumps(assembled_config, indent=2) + "\n")
    onnx.checker.check_model(str(output / "model.onnx"))
    # DFlash uses ORT's legacy default-domain SimplifiedLayerNormalization.
    # Its schema and execution are checked by the matching native runtime below.
    record("runtime_smoke")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "builders.qwen3_8_flash.verification",
            "--model",
            str(output),
            "--provider",
            str(provider),
            "--report",
            str(output / "build_verification.json"),
        ],
        env=environment,
        check=True,
    )
    state.update(package_ready=True, runtime_smoke_passed=True)
    record("complete")
    return state


class Qwen38FlashBuilder:
    """Build the specialized raw-checkpoint Qwen 3.8 Flash package."""

    @staticmethod
    def build(
        target_checkpoint,
        draft_checkpoint,
        output,
        prompts,
        provider,
        ort_home,
        *,
        workers=8,
        threads=1,
        calibration_capacity=4096,
        resume=False,
    ):
        return _build_model(
            target_checkpoint,
            draft_checkpoint,
            output,
            prompts,
            provider,
            ort_home,
            workers=workers,
            threads=threads,
            calibration_capacity=calibration_capacity,
            resume=resume,
        )
