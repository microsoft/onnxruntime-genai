# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
BENCHMARK = ROOT / "benchmark_qwen_context.py"
CONTEXTS = (8192, 16384, 32768, 65536, 131072)
WIDTHS = (1, 3, 5, 7)


def output(command):
    return subprocess.check_output(command, text=True).strip()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_table(rows, destination, prefix_cache=False):
    lines = [
        "# Optimized asonawane/qwen-38-flash-100-e2e context sweep",
        "",
        "CUDA graphs: enabled. Prompt task: source_code_review; format: chat.",
        "Three measured 512-token requests per configuration, one excluded warmup.",
        "Acceptance target: 80% +/- 5%; no samples filtered.",
        "VRAM: device-wide sampled peak including load and warmup (nominal 20 ms interval).",
        "Runtime: active Python environment; package versions and loaded library hashes recorded in manifest.",
        "Prompt: newly generated repeated Python worker modules; not the unavailable original example prompt.",
        "KV-cache budget: 0.05 for 8K-64K, 0.12 for 128K, matching the completed baseline.",
        "Graph variant is determined by --model; use the same model for paired cached/uncached comparisons.",
        "All packed NVFP4, fused SwiGLU, routing reuse, tiled down output, tensor QK/PV and bounded replay flags enabled.",
        "Repeated-prefix cached workload; cold warmup excluded. 8K may have no reusable checkpoint."
        if prefix_cache else "Uncached workload: fresh Engine for every request.",
        "",
        "| Requested prompt | Prompt length | MTP | TTFT (ms) | Decode TPS | E2E TPS | MTP acceptance | First-draft acceptance | Proposal yield | Within target | Sampled peak VRAM (MiB) | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---|",
    ]
    for row in rows:
        if row["status"] != "completed":
            lines.append(f"| {row['requested_prompt']} | - | {row['mtp']} | - | - | - | - | - | - | - | {row['peak_mib']:.0f} | failed; see log |")
            continue
        lines.append(
            f"| {row['requested_prompt']} | {row['prompt_length']} | {row['mtp']} "
            f"| {row['ttft_ms']:.2f} | {row['decode_tps']:.2f} | {row['e2e_tps']:.2f} "
            f"| {row['mtp_acceptance']:.2f}% | {row['first_draft_acceptance']:.2f}% "
            f"| {row['proposal_yield']:.2f}% | {'yes' if row['within_target'] else 'no'} "
            f"| {row['peak_mib']:.0f} | completed |"
        )
    (destination / "TABLE.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run the optimized Qwen 8K-128K context/MTP sweep")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New results directory; must not exist")
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index from nvidia-smi")
    parser.add_argument("--indexer-warp-heads", action="store_true", help="Enable experimental warp-local indexer scoring")
    parser.add_argument("--prefix-cache", action="store_true", help="Enable experimental repeated-prefix MTP workload")
    args = parser.parse_args()
    model = args.model.resolve()
    destination = args.output_dir.resolve()
    if not model.is_dir() or not (model / "genai_config.json").is_file():
        parser.error("--model must contain genai_config.json")
    if args.gpu < 0 or destination.exists():
        parser.error("--gpu must be nonnegative and --output-dir must not exist")
    if destination.is_relative_to(model):
        parser.error("--output-dir must not be inside the model directory")
    destination.mkdir(parents=True)
    environment = {
        "HOME": os.environ["HOME"],
        "PATH": os.environ["PATH"],
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        "ORT_QMOE_NVFP4_PACKED_PREFILL": "1",
        "ORT_QMOE_NVFP4_FUSED_SWIGLU": "1",
        "ORT_QMOE_NVFP4_REUSE_ROUTING_SCRATCH": "1",
        "ORT_QMOE_NVFP4_TILED_DOWN_OUTPUT": "1",
        "ORT_SPARSE_PREFILL_TENSOR_CORE_QK": "1",
        "ORT_SPARSE_PREFILL_TENSOR_CORE_PV": "1",
        "ORT_GENAI_CUDA_BOUNDED_INDEXER_REPLAY": "1",
        "ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS": "0",
        "ORT_PACKED_SPARSE_INDEXER_WARP_HEADS": "1" if args.indexer_warp_heads else "0",
    }
    for name in ("LD_LIBRARY_PATH", "PYTHONPATH"):
        if name in os.environ:
            environment[name] = os.environ[name]
    manifest = {
        "reference_ort_commit": "16289abceb0eb1d2330bca6e8aa1de708d227f86",
        "reference_genai_runtime_commit": "37e2b4773655427afa493098c7f44cb593cff0fa",
        "python": str(PYTHON),
        "packages": output([str(PYTHON), "-m", "pip", "freeze"]).splitlines(),
        "environment": environment,
        "benchmark_sha256": file_hash(BENCHMARK),
        "model": str(model),
        "model_config_sha256": file_hash(model / "genai_config.json"),
        "repetitions": 3,
        "warmup": 1,
        "output_tokens": 512,
        "prompt_task": "source_code_review",
        "prompt_format": "chat",
        "acceptance_target_percent": [75, 85],
        "memory_caveat": "Device-wide sampling during timing, load and warmup; GPU occupancy checked before each configuration, not continuously.",
        "rows": [],
        "prefix_cache": args.prefix_cache,
    }
    manifest["loaded_libraries_sha256"] = {}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    gpu_uuid = output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"])
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    nvml.nvmlErrorString.restype = ctypes.c_char_p

    def check(status):
        if status:
            raise RuntimeError(nvml.nvmlErrorString(status).decode())

    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

    check(nvml.nvmlInit_v2())
    process = None
    try:
        device = ctypes.c_void_p()
        check(nvml.nvmlDeviceGetHandleByIndex_v2(args.gpu, ctypes.byref(device)))
        memory = Memory()
        for context in CONTEXTS:
            for width in WIDTHS:
                processes = output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
                if any(line.split(",")[0].strip() == gpu_uuid for line in processes.splitlines()):
                    raise RuntimeError(f"GPU {args.gpu} occupied; refusing configuration: {processes}")
                prefix = destination / f"context{context}_mtp{width}"
                overlay = json.loads((ROOT / "width2_overlay.json").read_text())
                overlay["speculative"]["max_draft_tokens"] = width
                if args.prefix_cache:
                    overlay["engine"]["dynamic_batching"]["prefix_caching"] = True
                if context == 131072:
                    overlay["engine"]["dynamic_batching"]["gpu_utilization_factor"] = 0.12
                overlay_path = prefix.with_suffix(".overlay.json")
                overlay_path.write_text(json.dumps(overlay, indent=2) + "\n")
                result_path = prefix.with_suffix(".json")
                command = [
                    str(PYTHON), str(BENCHMARK), "--model", str(model), "--mode", "mtp",
                    "--overlay", str(overlay_path), "--prompt-length", str(context),
                    "--warmup", "1", "--repetitions", "3", "--output", str(result_path),
                ]
                if args.prefix_cache:
                    command.append("--prefix-cache")
                samples = []
                started = time.monotonic()
                print(f"Starting context={context} MTP={width}", flush=True)
                with prefix.with_suffix(".log").open("x") as log:
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        check(nvml.nvmlDeviceGetMemoryInfo(device, ctypes.byref(memory)))
                        samples.append([time.monotonic() - started, memory.used])
                        time.sleep(0.02)
                prefix.with_suffix(".memory.json").write_text(json.dumps(samples) + "\n")
                row = {
                    "requested_prompt": context,
                    "mtp": width,
                    "gpu_utilization_factor": overlay["engine"]["dynamic_batching"]["gpu_utilization_factor"],
                    "command": command,
                    "result_path": str(result_path),
                    "log_path": str(prefix.with_suffix(".log")),
                    "peak_mib": max(sample[1] for sample in samples) / 2**20,
                    "max_sampling_gap_seconds": max(
                        (b[0] - a[0] for a, b in zip(samples, samples[1:])), default=None
                    ),
                    "returncode": process.returncode,
                }
                if process.returncode != 0:
                    row["status"] = "failed"
                    print(prefix.with_suffix(".log").read_text()[-5000:], flush=True)
                else:
                    result = json.loads(result_path.read_text())
                    assert result["requested_prompt_tokens"] == context
                    assert context - 16 <= result["prompt_tokens"] <= context
                    assert result["max_draft_tokens"] == width
                    assert result["overlay"] == overlay
                    assert all(value == "1" for value in result["moe_prefill_flags"].values())
                    assert all(value == "1" for value in result["attention_tensor_core_flags"].values())
                    assert result["measured_runs"] == 3 and len(result["runs"]) == 4
                    assert result["loaded_runtime_libraries"]
                    for library in result["loaded_runtime_libraries"]:
                        path = Path(library).resolve()
                        digest = file_hash(path)
                        previous = manifest["loaded_libraries_sha256"].setdefault(str(path), digest)
                        assert previous == digest
                    measured = [run for run in result["runs"] if not run["warmup"]]
                    assert len(measured) == 3
                    if args.prefix_cache:
                        assert all(run["matches_cold_output"] for run in result["runs"])
                        prefix_stats = [run["prefix_cache_stats"] for run in measured]
                        if context > 8192:
                            assert all(stats["hits"] > 0 and stats["matched_tokens"] > 0 for stats in prefix_stats)
                        row["prefix_cache_stats"] = prefix_stats
                        row["cold_ttft_ms"] = result["runs"][0]["ttft_seconds"] * 1000
                    for run in result["runs"]:
                        assert run["output_tokens"] == 512
                        assert run["engine_max_request_length"] >= result["prompt_tokens"] + 512
                        assert run["speculative_stats"]["mtp_failures"] == 0
                        assert run["speculative_stats"]["draft_tokens_proposed"] > 0
                    counters = {
                        key: sum(run["speculative_stats"][key] for run in measured)
                        for key in ("draft_tokens_accepted", "draft_tokens_evaluated",
                                    "draft_tokens_proposed", "completed_rounds",
                                    "partial_accept_rounds", "full_accept_rounds", "zero_accept_rounds")
                    }
                    assert counters["completed_rounds"] == sum(
                        counters[key] for key in ("partial_accept_rounds", "full_accept_rounds", "zero_accept_rounds")
                    )
                    acceptance = 100 * counters["draft_tokens_accepted"] / counters["draft_tokens_evaluated"]
                    row.update({
                        "status": "completed",
                        "prompt_length": result["prompt_tokens"],
                        "prompt_sha256": result["prompt_sha256"],
                        "source_prompt_sha256": result["source_prompt_sha256"],
                        "ttft_ms": result["summary"]["mean_ttft_seconds"] * 1000,
                        "decode_tps": result["summary"]["decode_tps"],
                        "e2e_tps": result["summary"]["end_to_end_tps"],
                        "mtp_acceptance": acceptance,
                        "first_draft_acceptance": 100 * (
                            counters["partial_accept_rounds"] + counters["full_accept_rounds"]
                        ) / counters["completed_rounds"],
                        "proposal_yield": 100 * counters["draft_tokens_accepted"] / counters["draft_tokens_proposed"],
                        "within_target": 75 <= acceptance <= 85,
                        "pooled_counters": counters,
                        "loaded_runtime_libraries": result["loaded_runtime_libraries"],
                    })
                manifest["rows"].append(row)
                (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                save_table(manifest["rows"], destination, args.prefix_cache)
                print(json.dumps(row), flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        check(nvml.nvmlShutdown())
    if any(row["status"] != "completed" for row in manifest["rows"]):
        raise RuntimeError("Sweep completed with failed configurations; see persisted manifest and logs")


if __name__ == "__main__":
    main()
