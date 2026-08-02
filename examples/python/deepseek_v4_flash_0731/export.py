"""Export DeepSeek-V4-Flash-0731 to ONNX, one graph per tensor-parallel rank.

The builder streams every initializer into an external-data blob as it goes and
pulls the checkpoint in one tensor at a time, so peak host memory is a few GiB
regardless of the 311 GiB on disk.  Each rank gets its own graph: attention and
the shared expert are tensor-parallel, the routed experts are expert-parallel,
and two `AllReduce` nodes per layer stitch them back together.

    # all 8 ranks in parallel, one GPU each
    python export.py --ckpt /path/to/DeepSeek-V4-Flash-0731 --out ~/dsv4_onnx

    # 4 layers on one GPU, for a quick smoke test
    python export.py --ckpt ... --out /tmp/dsv4_w1 --world 1 --layers 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
import types

import onnx_ir as ir

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "src", "python", "py", "models")
sys.path.insert(0, os.path.abspath(_MODELS))

from builders.deepseek_v4 import DeepSeekV4FlashModel  # noqa: E402


def load_config(path):
    with open(os.path.join(path, "config.json")) as f:
        cfg = types.SimpleNamespace(**json.load(f))
    # The builder's rope helper keys off `rope_type`, which this checkpoint spells `type`.
    cfg.rope_scaling = dict(cfg.rope_scaling)
    cfg.rope_scaling.setdefault("rope_type", cfg.rope_scaling.get("type", "yarn"))
    cfg.tie_word_embeddings = getattr(cfg, "tie_word_embeddings", False)
    cfg._name_or_path = "deepseek-ai/DeepSeek-V4-Flash-0731"
    return cfg


def export_rank(a):
    out = os.path.join(a.out, f"rank_{a.rank}")
    cache = os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)

    opts = {
        "filename": "model.onnx",
        "dsv4_checkpoint": a.ckpt,
        "dsv4_max_seq_len": a.max_seq_len,
        "dsv4_tp_world": a.world,
        "dsv4_tp_rank": a.rank,
        "dsv4_repack_device": a.repack_device,
    }
    if a.layers:
        opts["dsv4_num_layers"] = a.layers

    t0 = time.time()
    model = DeepSeekV4FlashModel(load_config(a.ckpt), ir.DataType.BFLOAT16,
                                 ir.DataType.BFLOAT16, "cuda", cache, opts)
    model.make_model(None)
    print(f"[rank {a.rank}] graph built in {time.time() - t0:.0f}s", flush=True)
    model.save_model(out)
    blob = os.path.join(out, "model.onnx.data")
    print(f"[rank {a.rank}] done in {time.time() - t0:.0f}s, "
          f"{os.path.getsize(blob) / 2**30:.1f} GiB of weights", flush=True)
    return 0


def export_all(a, ranks):
    procs = []
    for r in ranks:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(r))
        log = open(os.path.join(a.out, f"export_rank{r}.log"), "w")
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--ckpt", a.ckpt, "--out", a.out,
             "--world", str(a.world), "--rank", str(r), "--layers", str(a.layers),
             "--max-seq-len", str(a.max_seq_len), "--repack-device", a.repack_device],
            env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    for r, c in zip(ranks, codes):
        tail = open(os.path.join(a.out, f"export_rank{r}.log")).read().strip().splitlines()
        print(f"rank {r} exit={c}: {tail[-1] if tail else ''}")
    return 0 if all(c == 0 for c in codes) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to the safetensors checkpoint")
    ap.add_argument("--out", required=True, help="directory that receives rank_<r>/")
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--rank", type=int, default=None,
                    help="export only this rank; default is every rank in parallel")
    ap.add_argument("--layers", type=int, default=0, help="0 = all")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--repack-device", default="cuda",
                    help="device used to repack the fp4 experts")
    a = ap.parse_args()

    a.ckpt = os.path.abspath(os.path.expanduser(a.ckpt))
    a.out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(a.out, exist_ok=True)
    if a.rank is not None:
        return export_rank(a)
    return export_all(a, range(a.world))


if __name__ == "__main__":
    raise SystemExit(main())
