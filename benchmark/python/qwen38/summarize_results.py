# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate and pool exact Qwen 8K/512 benchmark runs.")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--require-100", action="store_true")
    args = parser.parse_args()
    overlay = json.loads(Path(__file__).with_name("width2_overlay.json").read_text())
    rows = []
    for path in args.results:
        data = json.loads(path.read_text())
        if (data["prompt_tokens"] != 8192 or data["output_tokens"] != 512
                or data["prompt_sha256"] != "91620c2b4341bcf39f996f412f5428de25b97f24fac17cc0d72018f4cbad0236"
                or data["batch_size"] != 1 or data["do_sample"]
                or data["mode"] != "mtp" or data["max_draft_tokens"] != 2
                or data["overlay"] != overlay or data["warmup_runs"] < 1
                or data["attention_tensor_core_flags"] != {
                    "ORT_SPARSE_PREFILL_TENSOR_CORE_QK": "1",
                    "ORT_SPARSE_PREFILL_TENSOR_CORE_PV": "1",
                }):
            raise ValueError(f"{path}: workload or configuration differs from the measured policy")
        if len(data["runs"]) != data["warmup_runs"] + data["measured_runs"]:
            raise ValueError(f"{path}: unexpected run count")
        for index, row in enumerate(data["runs"]):
            stats = row["speculative_stats"]
            if (row["output_tokens"] != 512 or stats["mtp_failures"] != 0
                    or stats["draft_tokens_proposed"] <= 0
                    or not 0 <= 2 * stats["rounds"] - stats["draft_tokens_proposed"] <= 1
                    or row["warmup"] != (index < data["warmup_runs"])):
                raise ValueError(f"{path}: invalid request")
            if not row["warmup"]:
                rows.append(row)
        print(f"{path}: loaded libraries = {data['loaded_runtime_libraries']}")
    if not rows:
        raise ValueError("No measured requests")
    summary = {
        "measured_requests": len(rows),
        "end_to_end_tps": 512 * len(rows) / sum(row["end_to_end_seconds"] for row in rows),
        "decode_tps": 511 * len(rows) / sum(row["decode_seconds"] for row in rows),
        "mean_ttft_seconds": sum(row["ttft_seconds"] for row in rows) / len(rows),
        "mean_request_seconds": sum(row["end_to_end_seconds"] for row in rows) / len(rows),
        "request_e2e_tps_min_max": [
            min(row["end_to_end_tps"] for row in rows),
            max(row["end_to_end_tps"] for row in rows),
        ],
    }
    print(json.dumps(summary, indent=2))
    if args.require_100 and summary["end_to_end_tps"] < 100:
        raise SystemExit("Measured pooled E2E throughput is below 100 TPS; all samples retained.")


if __name__ == "__main__":
    main()
