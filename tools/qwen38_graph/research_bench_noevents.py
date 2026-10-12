"""Copy of CG/mtp_runner/bench_mtp.py (unchanged except: HERE points at the research runner, S2Verifier use_events=False) so the
research loop can be timed without its per-cycle CUDA-event synchronisation.

Benchmark harness matching the vLLM protocol (qwen38_vllm_20261011_01/bench_stream.py):
exact prompt lengths (2k/8k/32k), 256 new tokens, ignore_eos (fixed_length), 1 warmup + 3 measured runs (distinct prompts,
seeds as in bench_stream.py), TPS = (n-1 tokens) / decode time, natural text (--kind text == TEXT_PROMPTS=1) or random tokens.
Decode time starts when the first token is available (end of prefill) and ends at the last token.

Prints a per-cycle cost breakdown (verify GPU span by CUDA events, verify host overhead, head, other host) and a projection
table of TPS vs verify-S=2 time and acceptance.

usage (via run_mtp.sh): bench_mtp.py --out <json> [--contexts 2048,8192,32768] [--kind text|random]
        [--verifier ref|fast_ring|fast_copy] [--head eager|null|oracle] [--new 256] [--reps 3]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path("/home/tianlei/git/eval_runs/qwen38_cuda_graph_20261010_01/mtp_runner")  # research runner modules (read-only)
sys.path.insert(0, str(HERE))
import graph_mtp as gm  # noqa: E402
import prompts  # noqa: E402


class NullHead:
    """Zero-cost head: constant (wrong) draft -> every cycle rejects; isolates verify + host overhead."""

    def reset(self):
        pass

    def draft(self, token, hidden, position):
        return 0

    def refeed(self, token, hidden, position):
        pass


class OracleHead:
    """Zero-cost head proposing the true next token (needs the no-MTP reference tokens): every cycle accepts."""

    def __init__(self, ref, l0):
        self.ref, self.l0 = ref, l0

    def reset(self):
        pass

    def draft(self, token, hidden, position):
        return self.ref[position - self.l0 + 1]

    def refeed(self, token, hidden, position):
        pass


def projection(accept_rates, verify_ms, head_ms=0.6, host_ms=0.3, refeed_ms=0.0):
    """TPS = (1 + a) / (V + H + a*Hrefeed + O); a = draft acceptance, V = S=2 verify (+commit) ms."""
    rows = []
    for v in verify_ms:
        rows.append({"verify_ms": v, **{f"a={a:.2f}": round((1 + a) / ((v + head_ms + a * refeed_ms + host_ms) * 1e-3), 1)
                                        for a in accept_rates}})
    return rows


def required_verify_ms(target_tps, accept_rates, head_ms=0.6, host_ms=0.3, refeed_ms=0.0):
    return {f"a={a:.2f}": round((1 + a) / target_tps * 1e3 - head_ms - host_ms - a * refeed_ms, 2) for a in accept_rates}


def decode_once(v, head, gen_cls, pred, hidden, new, eos, *, mode, reset_head=True):
    gen = gen_cls(v, head, blind=True)
    if mode == "plain":
        t = time.perf_counter()
        toks = gen.decode_plain(pred, new, eos, fixed_length=True)
        return toks, time.perf_counter() - t, None
    toks, st = gen.decode(pred, hidden, new, eos, fixed_length=True, reset_head=reset_head)
    return toks, st.decode_s, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--contexts", default="2048,8192,32768")
    ap.add_argument("--kind", default="text", choices=("text", "random", "prose"))
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--verifier", default="fast_ring", choices=("ref", "fast_ring", "fast_copy", "s2"))
    ap.add_argument("--head", default="eager", choices=("eager", "graph", "device", "null", "oracle"))
    ap.add_argument("--reuse-warmup", action="store_true",
                    help="warmup decode on the first measured prompt's snapshot (saves one prefill)")
    ap.add_argument("--head-prefill", action="store_true",
                    help="prefill the head with the prompt (target hidden per position), as vLLM does")
    ap.add_argument("--skip-refeed", action="store_true")
    ap.add_argument("--plain", action="store_true", help="also time the no-MTP loop (device plane step / chain)")
    ap.add_argument("--models", type=Path, default=Path(os.environ["MODELS"]))
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    gm.init_runtime(os.environ["MTP_PROVIDER"])
    if args.verifier == "ref":
        v = gm.RingGraphVerifier(args.models)
    elif args.verifier == "s2":
        import s2_verifier

        v = s2_verifier.S2Verifier(args.models, use_events=False)  # no CUDA-event instrumentation (it synchronises per cycle)
    else:
        import fast_verifier as fv

        v = fv.FastRingVerifier(args.models, commit=args.verifier.split("_")[1], use_events=True)
    if args.head == "eager":
        head = gm.EagerHead()
    elif args.head == "graph":
        import graph_head

        head = graph_head.GraphHead()
    elif args.head == "device":
        import fast_head

        head = fast_head.DeviceHead(v)
        head.skip_refeed = args.skip_refeed
    else:
        head = NullHead()
    eos = prompts.EOS_IDS
    result = {"args": {k: str(x) for k, x in vars(args).items()}, "contexts": {}}
    for n in [int(x) for x in args.contexts.split(",")]:
        seeds = [1000 + n] + [i * 7 + n for i in range(args.reps)]
        runs = []
        snap0 = None
        for idx, seed in enumerate(seeds):
            warm = idx == 0
            if warm and args.reuse_warmup:
                continue
            prompt = prompts.make(args.kind, n, seed)
            t = time.perf_counter()
            pred, hidden = v.prefill(prompt, keep_hidden=args.head_prefill) if args.verifier != "ref" else v.prefill(prompt)
            prefill_s = time.perf_counter() - t
            hsnap = None
            if args.head_prefill:
                head.prefill(prompt, v.prompt_hidden)
                hsnap = head.snapshot()
                v.prompt_hidden = None
            snap = v.snapshot()
            l0 = v.length
            if args.head == "oracle":
                ref = gm.GraphMtpGenerator(v, None).decode_plain(pred, args.new + 4, eos, fixed_length=True)
                v.restore(snap)
                head = OracleHead(ref, l0)
            if hasattr(v, "host_stats"):
                for k in v.host_stats:
                    v.host_stats[k].clear()
                v.gpu_ms.clear()
            reps_here = [(f"warmup" if warm else f"run{idx}", warm)]
            if args.reuse_warmup and idx == 1:
                reps_here = [("warmup", True), ("run1", False)]
            for label, is_warm in reps_here:
                v.restore(snap)
                if hsnap is not None:
                    head.restore(hsnap)
                if hasattr(head, "collect_ms"):
                    head.timing = True
                    head.pairs = []
                toks, dec_s, st = decode_once(v, head, gm.GraphMtpGenerator, pred, hidden, args.new, eos, mode="mtp",
                                              reset_head=hsnap is None)
                tps = (len(toks) - 1) / dec_s
                rec = {"label": label, "seed": seed, "prefill_s": prefill_s, "decode_s": dec_s, "tokens": len(toks),
                       "tps": tps, "hash": gm.token_hash(toks)}
                if hasattr(head, "collect_ms"):
                    hm = head.collect_ms()
                    allm = hm[1] + hm[2]
                    rec_extra = {"head_gpu_ms_s1": statistics.median(hm[1]) if hm[1] else None,
                                 "head_gpu_ms_s2": statistics.median(hm[2]) if hm[2] else None,
                                 "head_gpu_ms_per_cycle": sum(allm) / max(1, st.cycles) if st else None}
                else:
                    rec_extra = {}
                if st is not None:
                    c = st.cycles
                    rec.update({
                        "cycles": c, "accept_rate": st.accepted / c, "tokens_per_cycle": st.tokens / c,
                        "ms_per_cycle": dec_s / c * 1e3, "head_draft_ms": st.head_s / c * 1e3,
                        "refeed_ms_per_cycle": st.refeed_s / c * 1e3, "verify_wall_ms": st.verify_s / c * 1e3,
                        "commit_ms": st.commit_s / c * 1e3,
                    })
                    if hasattr(v, "host_stats") and v.gpu_ms:
                        g = v.gpu_ms[-c:]
                        rec["verify_gpu_ms_median"] = statistics.median(g)
                        rec["verify_gpu_ms_mean"] = statistics.mean(g)
                        rec["verify_host_overhead_ms"] = rec["verify_wall_ms"] - statistics.mean(g)
                        rec["enqueue_ms_median"] = statistics.median(v.host_stats["enqueue_s"][-c:]) * 1e3
                        rec["wait_ms_median"] = statistics.median(v.host_stats["wait_s"][-c:]) * 1e3
                    rec["other_host_ms"] = rec["ms_per_cycle"] - rec["verify_wall_ms"] - rec["head_draft_ms"] \
                        - rec["refeed_ms_per_cycle"] - rec["commit_ms"]
                rec.update(rec_extra)
                runs.append(rec)
                print(json.dumps({"ctx": n, **rec}), flush=True)
                if args.plain and not is_warm and label == "run1":
                    v.restore(snap)
                    toks_p, dec_p, _ = decode_once(v, head, gm.GraphMtpGenerator, pred, hidden, args.new, eos,
                                                   mode="plain")
                    rec_p = {"plain_step_tps": (len(toks_p) - 1) / dec_p, "plain_ms_per_token": dec_p / (len(toks_p) - 1) * 1e3,
                             "same_tokens_as_mtp": toks_p == toks}
                    if hasattr(v, "step_chain") and getattr(v, "commit_mode", "") == "ring":
                        v.restore(snap)
                        t = time.perf_counter()
                        out = []
                        pend = pred
                        while len(out) < args.new - 1:
                            chunk = v.step_chain(pend, min(32, args.new - 1 - len(out)))
                            out.extend(chunk)
                            pend = chunk[-1]
                        dt = time.perf_counter() - t
                        rec_p["chain_ms_per_token"] = dt / len(out) * 1e3
                        rec_p["chain_tps"] = len(out) / dt
                        rec_p["chain_same_tokens"] = [pred] + out == toks_p[:len(out) + 1]
                    if getattr(v, "step_stats", None) and v.step_stats["gpu_ms"]:
                        nn = len(toks_p) - 1
                        rec_p["step_gpu_ms_median"] = statistics.median(v.step_stats["gpu_ms"][-nn:])
                        rec_p["step_wall_ms_median"] = statistics.median(v.step_stats["wall_ms"][-nn:])
                        rec_p["step_enqueue_ms_median"] = statistics.median(v.step_stats["enqueue_ms"][-nn:])
                    runs[-1]["plain"] = rec_p
                    print(json.dumps({"ctx": n, **rec_p}), flush=True)
        measured = [r for r in runs if r["label"] != "warmup"]
        measured.sort(key=lambda r: r["tps"])
        result["contexts"][n] = {"median": measured[len(measured) // 2], "runs": runs}
        m = result["contexts"][n]["median"]
        print(f"ctx {n}: decode {m['tps']:.2f} tps (median of {len(measured)}), accept {m.get('accept_rate')}", flush=True)
        args.out.write_text(json.dumps(result, indent=1))
    result["projection"] = {
        "assumptions": "TPS = (1+a)/(V + head 0.6 + host 0.3 ms); V = S=2 verify (incl. commit) ms",
        "table": projection([0.6, 0.7, 0.77, 0.8, 0.85],
                            [7.8, 8.2, 8.5, 8.7, 9.0, 9.5, 10.0, 11.0, 12.0, 14.0, 15.6]),
        "required_verify_ms_for_200": required_verify_ms(200, [0.6, 0.7, 0.77, 0.8, 0.85]),
        "required_verify_ms_for_220": required_verify_ms(220, [0.6, 0.7, 0.77, 0.8, 0.85]),
    }
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
