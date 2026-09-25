# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build and validate paged Qwen3.8-27B through GenAI Engine on WebGPU.

With --dflash2, require accepted drafts across multiple rounds for each fixed greedy
prompt. This checks real Engine activity; floating-point output coverage lives in
test_dflash2_webgpu_precision.py.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import register_webgpu_plugin

MODEL_ID = "Qwen/Qwen3.8-27B"
DFLASH2_CHECKPOINT = "z-lab/Qwen3.8-27B-DFlash2"
PROMPTS = (
    ("Reply with only the city name: What is the capital of France?", r"\bParis\b"),
    ("What is 17 + 25? Reply with the number.", r"\b42\b"),
    (
        "List the integers from 1 through 20 in order, separated by commas. Do not skip any numbers.",
        r"\b" + r",\s*".join(str(i) for i in range(1, 21)) + r"\b",
    ),
)
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
# DFlash2 is a block drafter: it supersedes exclude_mtp/MTP entirely (the Engine drives one
# drafter per model) and needs the target's aux hidden states, which are derived from the
# draft checkpoint's own `dflash_config.target_layer_ids` (see build_dflash2_options()).
DFLASH2_PRECISION = "int4"


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


def download_dflash2_checkpoint(cache_dir: Path) -> Path:
    """Fetch the DFlash 2 block-drafter checkpoint for Qwen3.8-27B.

    ``dflash2_path`` must be a local directory (the builder reads its ``config.json`` and
    weights directly), so the HF repo is snapshotted onto disk rather than passed as a repo id.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    print(f"Fetching DFlash 2 draft checkpoint {DFLASH2_CHECKPOINT}", flush=True)
    return Path(snapshot_download(repo_id=DFLASH2_CHECKPOINT, cache_dir=str(cache_dir)))


def download_target_snapshot(cache_dir: Path) -> Path:
    """Resolve Qwen3.8-27B to a local directory.

    DFlash2's builder reads the target's `embed_tokens.weight`/`lm_head.weight` straight out of
    `*.safetensors` shards on disk (see DFlash2Builder.load_weights); it does not go through the
    `transformers`-based loader that the plain `-m <hub_id> -c <cache_dir>` path uses for the
    target. Every DFlash2 README example therefore passes `-i <local_dir>` instead of `-m`. The
    non-DFlash2 build path is left untouched and keeps using `-m`/`-c`.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(snapshot_download(repo_id=MODEL_ID, cache_dir=str(cache_dir)))


def dflash2_aux_hidden_state_layers(draft_dir: Path) -> list[int]:
    """SpecForge's ``target_layer_ids`` name layer *outputs*; the target's
    ``aux_hidden_state_layers`` names residual streams *entering* a layer, one higher."""
    draft_config = json.loads((draft_dir / "config.json").read_text(encoding="utf-8"))
    target_layer_ids = draft_config["dflash_config"]["target_layer_ids"]
    return [layer_id + 1 for layer_id in target_layer_ids]


def build_dflash2_options(draft_dir: Path) -> dict:
    return {
        "dflash2_path": str(draft_dir),
        "dflash2_precision": DFLASH2_PRECISION,
        "aux_hidden_state_layers": dflash2_aux_hidden_state_layers(draft_dir),
        "state_update_capacity": 7,
    }


def format_extra_option(key: str, value: object) -> str:
    # Booleans/numbers are written lowercase to match the builder's true/false parsing; strings
    # (e.g. dflash2_path, a filesystem path) must be passed through verbatim -- lowercasing them
    # would corrupt any mixed-case path on a case-sensitive filesystem.
    if isinstance(value, (bool, int, float)):
        return f"{key}={str(value).lower()}"
    if isinstance(value, (list, tuple)):
        return f"{key}=" + ",".join(str(item) for item in value)
    return f"{key}={value}"


def build_identity(repo_root: Path, cache_dir: Path, build_options: dict, dflash2_draft_dir: Path | None) -> dict:
    return {
        "model_id": MODEL_ID,
        "model_revision": source_revision(cache_dir),
        "precision": "int4",
        "execution_provider": "webgpu",
        "options": build_options,
        "dflash2_checkpoint": DFLASH2_CHECKPOINT if dflash2_draft_dir is not None else None,
        "dflash2_checkpoint_revision": dflash2_draft_dir.name if dflash2_draft_dir is not None else None,
        "model_builder_sha256": hash_model_builder_sources(repo_root),
        "runtime_versions": {
            "onnxruntime": ort.__version__,
            "onnxruntime-genai": importlib.metadata.version("onnxruntime-genai"),
            "webgpu": (
                importlib.metadata.version("onnxruntime-webgpu")
                if "WebGpuExecutionProvider" in ort.get_available_providers()
                else importlib.metadata.version("onnxruntime-ep-webgpu")
            ),
        },
    }


def build_model(
    model_path: Path,
    cache_dir: Path,
    repo_root: Path,
    dflash2_draft_dir: Path | None = None,
) -> None:
    build_options = dict(BUILD_OPTIONS)
    if dflash2_draft_dir is not None:
        # DFlash 2 supersedes the MTP head outright, so exclude_mtp is redundant but harmless;
        # leave it set for clarity and drop it in favor of the drafter's own options.
        build_options.update(build_dflash2_options(dflash2_draft_dir))

    identity = build_identity(repo_root, cache_dir, build_options, dflash2_draft_dir)
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
    extra_options = [format_extra_option(key, value) for key, value in build_options.items()]
    if dflash2_draft_dir is not None:
        # DFlash2 needs a local target directory (see download_target_snapshot()); -m alone
        # would leave DFlash2Builder globbing a HF hub id as if it were a filesystem path.
        target_dir = download_target_snapshot(cache_dir)
        source_flags = ["-i", str(target_dir)]
    else:
        source_flags = ["-m", MODEL_ID]
    command = [
        sys.executable,
        str(builder),
        *source_flags,
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
    label = "DFlash2-enabled" if dflash2_draft_dir is not None else "paged"
    print(f"Exporting {label} Qwen3.8-27B INT4 for WebGPU", flush=True)
    print("Model Builder command:", " ".join(command), flush=True)
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


def prompt_tokens(tokenizer: og.Tokenizer, prompt: str) -> np.ndarray:
    messages = json.dumps([{"role": "user", "content": prompt}])
    prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
    return np.asarray(tokenizer.encode(prompt), dtype=np.int32)


def run_engine(model: og.Model, tokens: np.ndarray, max_new_tokens: int) -> tuple[list[int], float, dict]:
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
    speculative_stats = engine.get_speculative_stats()
    del engine
    gc.collect()
    return output, elapsed, speculative_stats


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


def validate_export(config_data: dict, build_options: dict, expect_dflash2: bool) -> None:
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
        if dynamic_batching.get(config_key) != build_options[option_key]:
            raise RuntimeError(
                f"Exported dynamic batching option {config_key}={dynamic_batching.get(config_key)!r}, "
                f"expected {build_options[option_key]!r}"
            )

    decoder = config_data["model"]["decoder"]
    groups = decoder.get("state_groups", [])
    kinds = {group.get("kind") for group in groups}
    required_groups = {"paged_kv", "fixed_conv", "fixed_recurrent"}
    if not required_groups.issubset(kinds):
        raise RuntimeError(f"Exported state groups {kinds} do not include {required_groups}")
    capacity = build_options["state_update_capacity"]
    if capacity:
        if decoder.get("state_update_capacity") != capacity:
            raise RuntimeError(
                f"Exported state_update_capacity={decoder.get('state_update_capacity')!r}, expected {capacity}"
            )
        for group in groups:
            if group["kind"] in {"fixed_conv", "fixed_recurrent"}:
                if group.get("state_update", {}).get("capacity") != capacity:
                    raise RuntimeError(f"Exported {group['kind']} state_update capacity is not {capacity}")
    elif any("state_update" in group for group in groups):
        raise RuntimeError("Export enabled compact state updates despite state_update_capacity=0")
    if config_data.get("model", {}).get("mtp", {}).get("enabled"):
        raise RuntimeError("The exported package unexpectedly enabled MTP speculative decoding")

    dflash2_section = config_data.get("model", {}).get("dflash2")
    if expect_dflash2:
        if not dflash2_section:
            raise RuntimeError("Export did not add a 'model.dflash2' section despite --dflash2 being requested")
        if dflash2_section.get("filename") != "dflash2.onnx":
            raise RuntimeError(f"Unexpected DFlash2 drafter filename: {dflash2_section.get('filename')!r}")
        expected_layers = build_options["aux_hidden_state_layers"]
        actual_layers = dflash2_section.get("aux_hidden_state_layers")
        if actual_layers != expected_layers:
            raise RuntimeError(
                f"Exported dflash2.aux_hidden_state_layers={actual_layers!r}, expected {expected_layers!r}"
            )
        if decoder.get("outputs", {}).get("aux_hidden_states") != "aux_hidden_states":
            raise RuntimeError("Export did not wire the target's aux_hidden_states output for DFlash2")
    elif dflash2_section:
        raise RuntimeError("The exported package unexpectedly enabled DFlash2 speculative decoding")


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
        raise RuntimeError("The profile did not contain both prefill and decode forwards for every Qwen3.8 state group")


def run_backend(
    model_path: Path,
    provider: str,
    max_new_tokens: int,
    prompts: tuple,
    profile_prefix: Path | None = None,
) -> list[dict]:
    model = og.Model(make_config(model_path, provider, profile_prefix))
    tokenizer = og.Tokenizer(model)
    results = []
    for prompt, expected in prompts:
        tokens = prompt_tokens(tokenizer, prompt)
        output_tokens, elapsed, speculative_stats = run_engine(model, tokens, max_new_tokens)
        output_text = tokenizer.decode(np.asarray(output_tokens, dtype=np.int32))
        results.append(
            {
                "prompt": prompt,
                "expected_pattern": expected,
                "elapsed_seconds": elapsed,
                "output": output_text,
                "tokens": output_tokens,
                "speculative_stats": speculative_stats,
            }
        )
        activity = {
            key: speculative_stats[key]
            for key in (
                "completed_rounds",
                "draft_tokens_proposed",
                "draft_tokens_evaluated",
                "draft_tokens_accepted",
                "acceptance_rate",
                "dflash2_failures",
                "standard_fallback_steps",
            )
        }
        print(f"{provider} prompt: {prompt}\nOutput: {output_text}\nSpeculative stats: {activity}", flush=True)
    del tokenizer
    del model
    gc.collect()
    return results


def validate_dflash2_activity(speculative_stats: dict) -> None:
    """Confirm DFlash2 actually drafted/verified tokens rather than silently falling back."""
    if speculative_stats.get("rounds", 0) <= 0:
        raise RuntimeError("DFlash2 was configured but the Engine ran zero speculative rounds")
    if speculative_stats.get("completed_rounds", 0) < 2:
        raise RuntimeError("DFlash2 validation requires at least two completed speculative rounds per prompt")
    if speculative_stats.get("draft_tokens_proposed", 0) <= 0:
        raise RuntimeError("DFlash2 was configured but no draft tokens were ever proposed")
    if speculative_stats.get("draft_tokens_accepted", 0) <= 0:
        raise RuntimeError("DFlash2 was configured but zero draft tokens were ever accepted")
    if speculative_stats["dflash2_disables"] != 0:
        raise RuntimeError(
            f"DFlash2 was disabled mid-run ({speculative_stats['dflash2_disables']} time(s)); "
            "the Engine silently fell back to ordinary decoding"
        )
    if speculative_stats["dflash2_failures"] != 0:
        raise RuntimeError(f"DFlash2 recorded {speculative_stats['dflash2_failures']} drafter failure(s)")
    if speculative_stats["standard_fallback_steps"] != 0:
        raise RuntimeError(f"DFlash2 used {speculative_stats['standard_fallback_steps']} standard fallback step(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Defaults to /workspace/models/qwen3.8-27b-int4-webgpu-paged, or "
        ".../qwen3.8-27b-int4-webgpu-paged-dflash2 when --dflash2 is set. An explicit override "
        "always wins, so it must not be reused between the two modes: their genai_config.json "
        "are incompatible (one carries a 'model.dflash2' section, the other must not).",
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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--dflash2",
        action="store_true",
        help="Build (if needed) and run with DFlash2 block-drafter speculative decoding enabled.",
    )
    parser.add_argument(
        "--skip-cpu-reference",
        action="store_true",
        help="Skip the CPU Engine reference run (useful for a 27B model where CPU decoding is slow).",
    )
    args = parser.parse_args()
    if args.model_path is None:
        suffix = "-dflash2" if args.dflash2 else ""
        args.model_path = Path(f"/workspace/models/qwen3.8-27b-int4-webgpu-paged{suffix}")

    if "WebGpuExecutionProvider" not in ort.get_available_providers() and not register_webgpu_plugin():
        raise RuntimeError("WebGPU EP is not available in ONNX Runtime or as a plugin")

    repo_root = Path(__file__).resolve().parents[3]
    dflash2_draft_dir = download_dflash2_checkpoint(args.cache_dir) if args.dflash2 else None
    build_model(args.model_path, args.cache_dir, repo_root, dflash2_draft_dir)
    config_data = json.loads((args.model_path / "genai_config.json").read_text(encoding="utf-8"))
    build_options = dict(BUILD_OPTIONS)
    if dflash2_draft_dir is not None:
        build_options.update(build_dflash2_options(dflash2_draft_dir))
    validate_export(config_data, build_options, expect_dflash2=args.dflash2)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    for profile_path in args.artifacts_dir.glob("webgpu-profile*.json"):
        profile_path.unlink()
    profile_prefix = args.artifacts_dir / "webgpu-profile"
    prompts = PROMPTS if args.dflash2 else PROMPTS[:1]
    webgpu_results = run_backend(
        args.model_path,
        "webgpu",
        args.max_new_tokens,
        prompts,
        profile_prefix,
    )
    webgpu_ops, cpu_ops = read_profile(args.artifacts_dir)
    validate_profile(webgpu_ops, cpu_ops)
    cpu_result = None
    cpu_reference_error = None
    if not args.skip_cpu_reference:
        try:
            cpu_result = run_backend(
                args.model_path,
                "cpu",
                args.max_new_tokens,
                prompts,
            )
        except RuntimeError as error:
            cpu_reference_error = str(error)
    result = {
        "dflash2": args.dflash2,
        "webgpu": webgpu_results,
        "cpu": cpu_result,
        "cpu_reference_error": cpu_reference_error,
        "webgpu_operator_counts": dict(webgpu_ops),
        "cpu_fallback_operator_counts": dict(cpu_ops),
    }
    (args.artifacts_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for item in webgpu_results:
        if not re.search(item["expected_pattern"], item["output"], re.IGNORECASE):
            raise RuntimeError(f"WebGPU output did not contain the expected answer: {item}")
        if args.dflash2:
            validate_dflash2_activity(item["speculative_stats"])
    for item in cpu_result or []:
        if not re.search(item["expected_pattern"], item["output"], re.IGNORECASE):
            raise RuntimeError(f"CPU output did not contain the expected answer: {item}")
    if cpu_reference_error:
        print(f"CPU reference unavailable: {cpu_reference_error}")
    print(f"WebGPU operator counts: {dict(webgpu_ops)}")
    print(f"CPU fallback operator counts: {dict(cpu_ops)}")
    print(f"Artifacts: {args.artifacts_dir}")
    print("Paged Qwen3.8 Engine execution completed correctly on WebGPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
