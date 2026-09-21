# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build and validate paged Qwen3.8-27B through GenAI Engine on WebGPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import register_webgpu_plugin


MODEL_ID = "Qwen/Qwen3.8-27B"
PROMPT = "Reply with only the city name: What is the capital of France?"
BUILD_METADATA = "qwen38_webgpu_build.json"
REQUIRED_WEBGPU_OPS = {
    "GatedDeltaNet",
    "MatMulNBits",
    "PagedAttention",
    "VarlenCausalConvWithState",
}
BUILD_OPTIONS = {
    "exclude_mtp": True,
    "hf_token": False,
    "max_batch_size": 1,
    "max_scheduled_tokens": 512,
    "num_blocks": 64,
    "paged_block_size": 256,
    "paged_chunk_size": 512,
    "state_update_capacity": 0,
    "use_paged_attention": True,
}


def hash_model_builder_sources(repo_root: Path) -> str:
    digest = hashlib.sha256()
    model_builder_root = repo_root / "src" / "python" / "py" / "models"
    for path in sorted(model_builder_root.rglob("*.py")):
        digest.update(path.relative_to(repo_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_revision(cache_dir: Path) -> str | None:
    revision_path = cache_dir / "models--Qwen--Qwen3.8-27B" / "refs" / "main"
    return revision_path.read_text(encoding="utf-8").strip() if revision_path.is_file() else None


def build_identity(repo_root: Path, cache_dir: Path) -> dict:
    return {
        "model_id": MODEL_ID,
        "model_revision": source_revision(cache_dir),
        "precision": "int4",
        "execution_provider": "webgpu",
        "options": BUILD_OPTIONS,
        "model_builder_sha256": hash_model_builder_sources(repo_root),
        "runtime_versions": {
            package: importlib.metadata.version(package)
            for package in ("onnxruntime", "onnxruntime-genai", "onnxruntime-ep-webgpu")
        },
    }


def build_model(model_path: Path, cache_dir: Path, repo_root: Path) -> None:
    identity = build_identity(repo_root, cache_dir)
    metadata_path = model_path / BUILD_METADATA
    config_path = model_path / "genai_config.json"
    if config_path.is_file() and metadata_path.is_file():
        existing_identity = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_identity == identity:
            print(f"Reusing compatible cached model: {model_path}", flush=True)
            return

    if model_path.exists():
        print(f"Removing incompatible or incomplete cached model: {model_path}", flush=True)
        shutil.rmtree(model_path)
    model_path.mkdir(parents=True)

    builder = repo_root / "src" / "python" / "py" / "models" / "builder.py"
    extra_options = [f"{key}={str(value).lower()}" for key, value in BUILD_OPTIONS.items()]
    command = [
        sys.executable,
        str(builder),
        "-m",
        MODEL_ID,
        "-o",
        str(model_path),
        "-p",
        "int4",
        "-e",
        "webgpu",
        "-c",
        str(cache_dir),
        "--extra_options",
        *extra_options,
    ]
    print("Exporting paged Qwen3.8-27B INT4 for WebGPU", flush=True)
    subprocess.run(command, check=True)
    metadata_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_config(model_path: Path, provider: str, profile_prefix: Path | None = None) -> og.Config:
    config = og.Config(str(model_path))
    config.clear_providers()
    if provider != "cpu":
        config.append_provider(provider)
    if profile_prefix is not None:
        config.overlay(
            json.dumps(
                {
                    "model": {
                        "decoder": {
                            "session_options": {
                                "enable_profiling": str(profile_prefix),
                                "log_severity_level": 2,
                                "log_verbosity_level": 0,
                            }
                        }
                    }
                }
            )
        )
    return config


def prompt_tokens(tokenizer: og.Tokenizer) -> np.ndarray:
    messages = json.dumps([{"role": "user", "content": PROMPT}])
    prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
    return np.asarray(tokenizer.encode(prompt), dtype=np.int32)


def run_engine(model: og.Model, tokens: np.ndarray, max_new_tokens: int) -> tuple[list[int], float]:
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(int(tokens.size) + max_new_tokens)
    request = engine.create_request(options=request_options)
    turn_options = og.TurnOptions(request)
    turn_options.set_do_sample(False)
    turn_options.set_max_generated_tokens(max_new_tokens)
    request.begin_turn(tokens, turn_options)

    output = []
    event_buffer = engine.create_event_buffer(8)
    started = time.perf_counter()
    try:
        while engine.has_pending_requests():
            for event in engine.run(event_buffer):
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(f"Engine request failed with error_code={event.error_code}")
                if event.request is not request:
                    raise RuntimeError("Engine returned an event for an unknown request")
                if event.flags & og.EngineEventFlags.TOKEN:
                    output.append(int(event.token))
    finally:
        request.close()
    elapsed = time.perf_counter() - started
    del engine
    gc.collect()
    return output, elapsed


def read_profile(artifacts_dir: Path) -> tuple[Counter, Counter]:
    profiles = sorted(artifacts_dir.glob("webgpu-profile*.json"))
    if not profiles:
        raise RuntimeError(f"ONNX Runtime did not write a profile under {artifacts_dir}")

    webgpu_ops = Counter()
    cpu_ops = Counter()
    for profile_path in profiles:
        events = json.loads(profile_path.read_text(encoding="utf-8"))
        for event in events:
            args = event.get("args", {})
            provider = str(args.get("provider", "")).lower()
            op_name = args.get("op_name")
            if not provider or not op_name:
                continue
            if "webgpu" in provider:
                webgpu_ops[op_name] += 1
            elif "cpu" in provider:
                cpu_ops[op_name] += 1
    return webgpu_ops, cpu_ops


def validate_export(config_data: dict) -> None:
    dynamic_batching = config_data.get("engine", {}).get("dynamic_batching")
    if not dynamic_batching:
        raise RuntimeError("Export did not enable Engine dynamic batching")
    config_options = {
        "num_blocks": "num_blocks",
        "max_batch_size": "max_batch_size",
        "max_scheduled_tokens": "max_scheduled_tokens",
        "block_size": "paged_block_size",
    }
    for config_key, option_key in config_options.items():
        if dynamic_batching.get(config_key) != BUILD_OPTIONS[option_key]:
            raise RuntimeError(
                f"Exported dynamic batching option {config_key}={dynamic_batching.get(config_key)!r}, "
                f"expected {BUILD_OPTIONS[option_key]!r}"
            )

    decoder = config_data["model"]["decoder"]
    groups = decoder.get("state_groups", [])
    kinds = {group.get("kind") for group in groups}
    required_groups = {"paged_kv", "fixed_conv", "fixed_recurrent"}
    if not required_groups.issubset(kinds):
        raise RuntimeError(f"Exported state groups {kinds} do not include {required_groups}")
    if any("state_update" in group for group in groups):
        raise RuntimeError("WebGPU validation must not enable unsupported compact state updates")
    if config_data.get("model", {}).get("mtp", {}).get("enabled"):
        raise RuntimeError("The exported package unexpectedly enabled MTP speculative decoding")


def validate_profile(webgpu_ops: Counter, cpu_ops: Counter) -> None:
    missing = REQUIRED_WEBGPU_OPS.difference(webgpu_ops)
    if missing:
        raise RuntimeError(
            f"Required Qwen3.8 kernels were not assigned to WebGPU: {sorted(missing)}; "
            f"WebGPU ops={dict(webgpu_ops)}; CPU ops={dict(cpu_ops)}"
        )
    required_cpu_fallback = REQUIRED_WEBGPU_OPS.intersection(cpu_ops)
    if required_cpu_fallback:
        raise RuntimeError(f"Required Qwen3.8 kernels fell back to CPU: {sorted(required_cpu_fallback)}")
    if webgpu_ops["GatedDeltaNet"] < 2 * 48 or webgpu_ops["PagedAttention"] < 2 * 16:
        raise RuntimeError(
            "The profile did not contain both prefill and decode forwards for every Qwen3.8 state group"
        )


def run_backend(
    model_path: Path,
    provider: str,
    max_new_tokens: int,
    profile_prefix: Path | None = None,
) -> tuple[list[int], str, float]:
    model = og.Model(make_config(model_path, provider, profile_prefix))
    tokenizer = og.Tokenizer(model)
    tokens = prompt_tokens(tokenizer)
    output_tokens, elapsed = run_engine(model, tokens, max_new_tokens)
    output_text = tokenizer.decode(np.asarray(output_tokens, dtype=np.int32))
    del tokenizer
    del model
    gc.collect()
    return output_tokens, output_text, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/qwen3.8-27b-int4-webgpu-paged"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/root/.cache/huggingface/hub"),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/tmp/qwen38-paged-webgpu-engine"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if not register_webgpu_plugin():
        raise RuntimeError("onnxruntime-ep-webgpu is not installed")

    repo_root = Path(__file__).resolve().parents[3]
    build_model(args.model_path, args.cache_dir, repo_root)
    config_data = json.loads((args.model_path / "genai_config.json").read_text(encoding="utf-8"))
    validate_export(config_data)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    for profile_path in args.artifacts_dir.glob("webgpu-profile*.json"):
        profile_path.unlink()
    profile_prefix = args.artifacts_dir / "webgpu-profile"

    webgpu_tokens, webgpu_text, webgpu_elapsed = run_backend(
        args.model_path,
        "webgpu",
        args.max_new_tokens,
        profile_prefix,
    )
    webgpu_ops, cpu_ops = read_profile(args.artifacts_dir)
    validate_profile(webgpu_ops, cpu_ops)
    if "paris" not in webgpu_text.lower():
        raise RuntimeError(f"WebGPU output did not contain the expected answer 'Paris': {webgpu_text!r}")

    cpu_result = None
    cpu_reference_error = None
    try:
        cpu_tokens, cpu_text, cpu_elapsed = run_backend(
            args.model_path,
            "cpu",
            args.max_new_tokens,
        )
        cpu_result = {
            "elapsed_seconds": cpu_elapsed,
            "output": cpu_text,
            "tokens": cpu_tokens,
        }
    except RuntimeError as error:
        cpu_reference_error = str(error)
    if cpu_result and "paris" not in cpu_result["output"].lower():
        raise RuntimeError(f"CPU output did not contain the expected answer 'Paris': {cpu_result['output']!r}")

    result = {
        "prompt": PROMPT,
        "webgpu": {
            "elapsed_seconds": webgpu_elapsed,
            "output": webgpu_text,
            "tokens": webgpu_tokens,
        },
        "cpu": cpu_result,
        "cpu_reference_error": cpu_reference_error,
        "webgpu_operator_counts": dict(webgpu_ops),
        "cpu_fallback_operator_counts": dict(cpu_ops),
    }
    (args.artifacts_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Prompt: {PROMPT}")
    print(f"WebGPU Engine output ({webgpu_elapsed:.2f}s): {webgpu_text}")
    if cpu_result:
        print(f"CPU Engine output ({cpu_result['elapsed_seconds']:.2f}s): {cpu_result['output']}")
    else:
        print(f"CPU reference unavailable: {cpu_reference_error}")
    print(f"WebGPU operator counts: {dict(webgpu_ops)}")
    print(f"CPU fallback operator counts: {dict(cpu_ops)}")
    print(f"Artifacts: {args.artifacts_dir}")
    print("Paged Qwen3.8 Engine execution completed correctly on WebGPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
