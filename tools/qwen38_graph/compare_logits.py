#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Noise-floor comparison of the last-prompt-token logits: python PrefillEngine (ref_v12.py --save-logits) vs the GenAI fast prefill
(dump_logits_genai.py). The dense prefill is not bit-reproducible between runs, so the GenAI port is judged against the run-to-run
spread of the python reference itself: python-vs-python, genai-vs-genai and genai-vs-python pairs of the same prompt.

usage: compare_logits.py <python npz> <genai npz> [--out json]
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def metrics(a, b):
    diff = np.abs(a - b)
    top = np.argsort(-a)[:5]
    margin = float(a[top[0]] - a[top[1]])
    return {"rel_l2": float(np.linalg.norm(a - b) / np.linalg.norm(a)), "max_abs": float(diff.max()),
            "top1_equal": int(np.argmax(a)) == int(np.argmax(b)), "top5_overlap": len(set(top.tolist()) & set(np.argsort(-b)[:5].tolist())),
            "top1_margin": margin, "bitwise_equal": bool(np.array_equal(a, b))}


def load(path):
    data = np.load(path)
    out = defaultdict(dict)
    for key in data.files:
        spec, rep = key.rsplit("__", 1)
        out[spec][int(rep)] = data[key].astype(np.float32).reshape(-1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("python_npz", type=Path)
    parser.add_argument("genai_npz", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    py, gn = load(args.python_npz), load(args.genai_npz)
    report = {}
    for spec in sorted(set(py) & set(gn)):
        rows = {"py_py": [], "gn_gn": [], "gn_py": []}
        for (i, a), (j, b) in itertools.combinations(sorted(py[spec].items()), 2):
            rows["py_py"].append(metrics(a, b))
        for (i, a), (j, b) in itertools.combinations(sorted(gn[spec].items()), 2):
            rows["gn_gn"].append(metrics(a, b))
        for a in gn[spec].values():
            for b in py[spec].values():
                rows["gn_py"].append(metrics(b, a))
        summary = {}
        for kind, items in rows.items():
            if items:
                summary[kind] = {"pairs": len(items), "rel_l2_mean": float(np.mean([m["rel_l2"] for m in items])),
                                 "rel_l2_max": float(np.max([m["rel_l2"] for m in items])),
                                 "max_abs_max": float(np.max([m["max_abs"] for m in items])),
                                 "top1_equal": int(sum(m["top1_equal"] for m in items)), "bitwise": int(sum(m["bitwise_equal"] for m in items)),
                                 "top1_margin": items[0]["top1_margin"]}
        report[spec] = summary
        print(spec, json.dumps(summary), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
