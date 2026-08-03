"""Prefill/decode throughput and TTFT benchmark for the exported DeepSeek-V4-Flash-0731.

Metric definitions follow `benchmark/python/benchmark_e2e.py` so the numbers are
comparable with the rest of onnxruntime-genai:

  TTFT        = prompt processing + first-token sampling
  Prefill TPS = prompt_length / TTFT
  Decode TPS  = batch / mean(per-step latency), with the first token excluded

A decode step costs the same at any batch size -- it is ~18k kernel launches whatever
the rows contain -- so `--batch` trades a little step latency for a lot of aggregate
throughput.  Prefill stays serial across rows, so TTFT grows roughly linearly with
`--batch` and only the decode columns should be read as a batching result.

Prefill and decode are timed inside each rank (`prefill_s` / `decode_s`), so those
columns are engine compute and exclude the driver's ~0.1 ms per-token broadcast;
the wall-clock column is what a caller actually sees.

One `DSV4Engine` is reused for every configuration - the ~80 s/rank session load is
paid once - and EOS is disabled so each repetition decodes exactly `--gen-length`
tokens.  The first requests are 15-30x slower while cuBLAS/cuDNN warm up, hence
`--warmup`.

Example:
  python dsv4_benchmark.py --model ~/dsv4_onnx \
      --prompt-lengths 128,512,1024,2048,3072 --gen-length 128 --reps 5
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsv4_engine import DSV4Engine  # noqa: E402


def run_config(engine, prompt_len, gen_len, reps, warmup, vocab, rng, batch):
    """Time one (prompt_len, gen_len) point.

    Prompts are random token ids so that the length is exact; the generated text is
    meaningless, which is fine for timing but does produce occasional exact top-2
    ties that the engine reports as rank argmax disagreements.
    """
    rows = []
    for i in range(warmup + reps):
        prompts = [[int(rng.randrange(1000, min(vocab, 100000))) for _ in range(prompt_len)]
                   for _ in range(batch)]
        t0 = time.perf_counter()
        out = engine.generate_batch(prompts, max_new_tokens=gen_len,
                                    eos_token_ids=(), timeout=1800)
        wall = time.perf_counter() - t0
        n = min(len(t) for t in out["tokens"])
        if i < warmup:
            continue
        if n < 2:
            raise RuntimeError(f"only {n} token(s) generated; cannot time decode")
        rows.append({
            "ttft_s": out["prefill_s"],
            "decode_s": out["decode_s"],
            "n_decode": n - 1,          # the first token is attributed to TTFT
            "wall_s": wall,
            "n_total": n,
            "disagreements": len(out.get("rank_disagreements", [])),
        })
    return rows


def summarize(prompt_len, gen_len, rows, batch):
    ttft = [r["ttft_s"] for r in rows]
    dec_lat = [r["decode_s"] / r["n_decode"] for r in rows]
    mean_ttft = statistics.mean(ttft)
    mean_dec = statistics.mean(dec_lat)
    wall = statistics.mean(r["wall_s"] for r in rows)
    total_tok = rows[0]["n_total"]
    return {
        "prompt_length": prompt_len,
        "gen_length": gen_len,
        "batch": batch,
        "ttft_ms": mean_ttft * 1000,
        "ttft_std_ms": (statistics.stdev(ttft) * 1000) if len(ttft) > 1 else 0.0,
        "prefill_tps": prompt_len * batch / mean_ttft,
        "decode_latency_ms": mean_dec * 1000,
        "decode_tps": batch / mean_dec,
        "wall_s": wall,
        "wall_tps": batch * (prompt_len + total_tok) / wall,
        "disagreements": sum(r["disagreements"] for r in rows),
        "reps": len(rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="~/dsv4_onnx",
                    help="directory holding rank_0 .. rank_{world-1}")
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--prompt-lengths", default="128,512,1024,2048,3072")
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prefill-chunk", type=int, default=512,
                    help="paged exports only; set it above the longest prompt to"
                         " prefill in one pass, as the dense export does")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture the decode step as a CUDA graph and replay it")
    ap.add_argument("--batches", default="1",
                    help="comma-separated batch sizes to sweep; paged exports only")
    ap.add_argument("--out", default=None, help="write the raw measurements as JSON")
    a = ap.parse_args()

    lengths = [int(x) for x in a.prompt_lengths.split(",")]
    batches = [int(x) for x in a.batches.split(",")]
    rng = random.Random(a.seed)

    results = []
    load_s = 0.0
    # The batch size is fixed for a worker process -- a captured graph has fixed
    # shapes -- so each batch point needs its own engine.
    for batch in batches:
        t0 = time.time()
        engine = DSV4Engine(a.model, world=a.world, prefill_chunk=a.prefill_chunk,
                            cuda_graph=a.cuda_graph, batch=batch)
        load_s += time.time() - t0
        print(f"engine up in {time.time() - t0:.0f}s  (batch={batch}, "
              f"max_seq_len={engine.max_seq_len}, vocab={engine.vocab})\n", flush=True)
        try:
            for pl in lengths:
                if engine.max_seq_len and pl + a.gen_length >= engine.max_seq_len:
                    print(f"skip prompt_len={pl}: {pl}+{a.gen_length} exceeds the export's "
                          f"max_seq_len of {engine.max_seq_len}")
                    continue
                print(f"--- batch={batch} prompt_len={pl} gen={a.gen_length} "
                      f"({a.warmup} warmup + {a.reps} reps) ---", flush=True)
                rows = run_config(engine, pl, a.gen_length, a.reps, a.warmup,
                                  engine.vocab or 129280, rng, batch)
                s = summarize(pl, a.gen_length, rows, batch)
                results.append(s)
                print(f"  TTFT {s['ttft_ms']:8.1f} +/- {s['ttft_std_ms']:.1f} ms | "
                      f"prefill {s['prefill_tps']:8.1f} tps | "
                      f"decode {s['decode_tps']:6.2f} tps "
                      f"({s['decode_latency_ms']:.1f} ms/step)", flush=True)
        finally:
            engine.close()

    if not results:
        return

    print("\n" + "=" * 100)
    print(f"{'batch':>6} {'prompt':>7} {'gen':>5} {'TTFT ms':>10} {'+/-':>7} "
          f"{'prefill tps':>12} {'decode tps':>11} {'ms/step':>8} {'wall tps':>9}")
    print("-" * 100)
    for s in results:
        print(f"{s['batch']:>6} {s['prompt_length']:>7} {s['gen_length']:>5} "
              f"{s['ttft_ms']:>10.1f} "
              f"{s['ttft_std_ms']:>7.1f} {s['prefill_tps']:>12.1f} "
              f"{s['decode_tps']:>11.2f} {s['decode_latency_ms']:>8.1f} "
              f"{s['wall_tps']:>9.1f}")
    print("=" * 100)
    print(f"engine load: {load_s:.0f} s")
    print(f"rank argmax disagreements: {sum(s['disagreements'] for s in results)}")

    if a.out:
        out = os.path.abspath(os.path.expanduser(a.out))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"results": results, "load_s": load_s, "world": a.world}, f, indent=2)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
