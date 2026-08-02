"""Run the exported DeepSeek-V4-Flash-0731, one process per tensor-parallel rank.

Each process holds its own graph; the ranks meet in the two
`com.microsoft.AllReduce` nodes per layer.  ORT's NCCL bootstrap only binds its
rendezvous socket deep inside session creation and gives the peers a 40 s
window, so rank 0 is started alone and the rest wait until the port is actually
listening.  `RANK0_IP` has to be a literal v4 address: rank 0 binds `0.0.0.0`
while the peers resolve the name with `AF_UNSPEC` and settle on the first family
they can open a socket for.

    # 8 GPUs
    python run.py --model ~/dsv4_onnx --tokenizer /path/to/DeepSeek-V4-Flash-0731 \
        --prompt "The capital city of France is called"

    # single-rank smoke test of a `--world 1` export
    python run.py --model /tmp/dsv4_w1 --world 1
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

# onnxruntime's ONNX_TENSOR_ELEMENT_DATA_TYPE values, for io_binding.
_ELEM = {torch.float32: 1, torch.int64: 7, torch.bool: 9, torch.float16: 10,
         torch.bfloat16: 16}
_DTYPE = {"tensor(float)": torch.float32, "tensor(bfloat16)": torch.bfloat16,
          "tensor(float16)": torch.float16, "tensor(int64)": torch.int64,
          "tensor(bool)": torch.bool}

DEFAULT_PROMPT_IDS = "[0, 671, 6102, 4593, 294, 8760, 344, 3252]"


def wait_for_listen(port, proc, timeout=1800):
    """Block until ``port`` reaches LISTEN state, or ``proc`` dies.

    Probed by reading /proc rather than connecting: a real connect would be
    accepted as one of the peers and corrupt the handshake.
    """
    want = f"{port:04X}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                with open(path) as f:
                    next(f)
                    for line in f:
                        cols = line.split()
                        if cols[3] == "0A" and cols[1].rsplit(":", 1)[1] == want:
                            return True
            except OSError:
                pass
        if proc.poll() is not None:
            return False
        time.sleep(1)
    return False


def child(a):
    import onnxruntime as ort

    t0 = time.time()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    path = os.path.join(a.model, f"rank_{a.rank}", "model.onnx")
    sess = ort.InferenceSession(path, so, providers=["CUDAExecutionProvider"])
    print(f"[rank {a.rank}] session ready in {time.time() - t0:.0f}s", flush=True)

    ins, outs_spec = sess.get_inputs(), sess.get_outputs()
    in_names = [i.name for i in ins]
    out_names = [o.name for o in outs_spec]
    vocab = outs_spec[0].shape[-1]

    def zeros(spec):
        shape = [1 if isinstance(d, str) or d is None else d for d in spec.shape]
        return torch.zeros(shape, dtype=_DTYPE[spec.type], device="cuda")

    cache = [zeros(i) for i in ins[2:]]
    ids = torch.tensor([json.loads(a.prompt)], dtype=torch.int64, device="cuda")

    past, produced = 0, []
    for step in range(a.max_new_tokens):
        io = sess.io_binding()
        past_t = torch.full((), past, dtype=torch.int64, device="cuda")
        for name, t in zip(in_names, [ids, past_t] + cache):
            t = t.contiguous()
            io.bind_input(name, "cuda", 0, _ELEM[t.dtype], tuple(t.shape), t.data_ptr())
        got = [torch.empty(1, ids.shape[1], vocab, device="cuda", dtype=torch.float32)]
        got += [torch.empty_like(c) for c in cache]
        for name, o in zip(out_names, got):
            io.bind_output(name, "cuda", 0, _ELEM[o.dtype], tuple(o.shape), o.data_ptr())
        sess.run_with_iobinding(io)
        cache = got[1:]
        past += ids.shape[1]

        logits = got[0][0, -1]
        nxt = int(logits.argmax())
        if a.rank == 0:
            print(f"[rank 0] step {step}: S={ids.shape[1]} "
                  f"finite={bool(torch.isfinite(logits).all())} "
                  f"max={logits.max().item():.3f} min={logits.min().item():.3f} "
                  f"argmax={nxt}", flush=True)
        produced.append(nxt)
        ids = torch.tensor([[nxt]], dtype=torch.int64, device="cuda")

    if a.rank == 0:
        print("tokens:", produced, flush=True)
    return 0


def spawn(a, rank):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), LOCAL_RANK=str(rank),
               LOCAL_WORLD_SIZE=str(a.world), RANK0_IP="127.0.0.1",
               RANK0_PORT=str(a.port))
    log = open(os.path.join(a.log_dir, f"run_rank{rank}.log"), "w")
    return subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--child", "--rank", str(rank),
         "--world", str(a.world), "--model", a.model, "--prompt", a.prompt,
         "--max-new-tokens", str(a.max_new_tokens)],
        env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="directory holding rank_<r>/")
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--port", type=int, default=19555, help="NCCL rendezvous port")
    ap.add_argument("--tokenizer", help="checkpoint dir; enables text prompts and decoding")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT_IDS,
                    help="prompt text, or a JSON list of token ids")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--log-dir", default="/tmp")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rank", type=int, default=0)
    a = ap.parse_args()

    a.model = os.path.abspath(os.path.expanduser(a.model))
    if a.child:
        return child(a)

    tokenizer = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(a.tokenizer))
    if not a.prompt.lstrip().startswith("["):
        if tokenizer is None:
            raise SystemExit("--tokenizer is required for a text prompt")
        a.prompt = json.dumps([tokenizer.bos_token_id or 0] + tokenizer(a.prompt)["input_ids"])

    procs = [spawn(a, 0)]
    if a.world > 1:
        if not wait_for_listen(a.port, procs[0]):
            procs[0].kill()
            raise SystemExit(f"rank 0 never listened on port {a.port}; see "
                             f"{a.log_dir}/run_rank0.log")
        print("rank 0 is listening; starting the remaining ranks", flush=True)
        procs += [spawn(a, r) for r in range(1, a.world)]
    codes = [p.wait() for p in procs]

    log = open(os.path.join(a.log_dir, "run_rank0.log")).read()
    print(log)
    print("ranks exited with", codes)
    if tokenizer is not None and "tokens: " in log:
        ids = json.loads(log.rsplit("tokens: ", 1)[1].splitlines()[0])
        print("text:", repr(tokenizer.decode(ids)))
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
