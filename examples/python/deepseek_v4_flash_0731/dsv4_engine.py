"""Serving layer for the 8-rank DeepSeek-V4-Flash-0731 ONNX export.

The model is tensor/expert parallel: one `InferenceSession` per GPU, all eight
meeting in the two `com.microsoft.AllReduce` nodes of every layer.  A rank
cannot make progress alone, so a request is a *collective*: the engine hands the
same prompt to all eight workers and they walk the decode loop in lockstep.

**KV cache.**  Every rank owns its slice of the cache as plain CUDA memory that
this process allocates once (a `torch` tensor per cache tensor, in two copies)
and binds into the session with `IOBinding`.  Nothing is copied to the host and
nothing is reallocated per token: a step binds the *current* set as the `past_*`
inputs and the *other* set as the `present_*` outputs, then swaps.  The cache is
zeroed at the start of a request; within a request only `past_lens` advances.

**Sampling.**  Greedy, and decided centrally.  Every rank computes its own
argmax, the engine takes rank 0's and broadcasts it back before the next step.
`lm_head` is replicated, so the ranks' logits agree bit for bit almost always -
but not quite always, and a single disagreement would desynchronize the group
permanently, because the losing rank would feed a different token into the next
`AllReduce`.  Disagreements are counted and reported, not tolerated silently.

Usage::

    from dsv4_engine import DSV4Engine

    with DSV4Engine("~/dsv4_onnx", world=8) as engine:
        out = engine.generate(prompt_ids, max_new_tokens=64, eos_token_ids=[1])
        print(out["tokens"], out["stop_reason"], out["decode_tok_s"])
"""

import argparse
import json
import os
import select
import socket
import subprocess
import sys
import time

# onnxruntime's ONNX_TENSOR_ELEMENT_DATA_TYPE values, for io_binding.
_ELEM = {"torch.float32": 1, "torch.int64": 7, "torch.bool": 9,
         "torch.float16": 10, "torch.bfloat16": 16}
_DTYPE = {"tensor(float)": "float32", "tensor(bfloat16)": "bfloat16",
          "tensor(float16)": "float16", "tensor(int64)": "int64",
          "tensor(bool)": "bool"}

DEFAULT_PORT = 19555
# A worker blocks indefinitely between requests; only the engine bounds its waits.
_FOREVER = 10 ** 9
# Compressed-cache capacity is `max_seq_len // ratio + 2`; the ratio-128 layers
# give the smallest one.  Used to recover the export's context bound.
_MIN_COMPRESS_RATIO = 4
_MAX_COMPRESS_RATIO = 128


class ProtocolError(RuntimeError):
    pass


class _Chan:
    """Newline-delimited JSON over a socketpair, with timeouts."""

    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def send(self, obj):
        self.sock.sendall(json.dumps(obj).encode() + b"\n")

    def recv(self, timeout):
        deadline = time.time() + timeout
        while b"\n" not in self.buf:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("no reply within %.0fs" % timeout)
            if not select.select([self.sock], [], [], min(remaining, 1.0))[0]:
                continue
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ProtocolError("worker closed the connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #


class _Worker:
    """One rank: owns a session, its KV cache, and the decode loop."""

    def __init__(self, model_dir, rank):
        import onnxruntime as ort
        import torch

        self.torch = torch
        t0 = time.time()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        path = os.path.join(model_dir, f"rank_{rank}", "model.onnx")
        self.sess = ort.InferenceSession(path, so, providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in self.sess.get_providers():
            # Without it the session silently falls back to CPU, which has no
            # bfloat16 kernels, and the first failing node is blamed instead.
            raise ProtocolError("CUDA EP unavailable; put cuDNN 9 and CUDA 13 on "
                                "LD_LIBRARY_PATH (see the export doc)")
        self.load_s = time.time() - t0
        self.rank = rank

        ins, outs = self.sess.get_inputs(), self.sess.get_outputs()
        if [i.name for i in ins[:2]] != ["input_ids", "past_lens"]:
            raise ProtocolError("unexpected graph inputs: %s" % [i.name for i in ins[:3]])
        self.cache_in = [i.name for i in ins[2:]]
        self.cache_out = [o.name for o in outs[1:]]
        if len(self.cache_in) != len(self.cache_out):
            raise ProtocolError("past/present count mismatch")
        self.vocab = int(outs[0].shape[-1])

        # Two full copies of the cache, allocated once and ping-ponged.
        self.cache = [[self._zeros(i) for i in ins[2:]] for _ in range(2)]
        self.cur = 0
        self.bytes = sum(t.numel() * t.element_size() for t in self.cache[0]) * 2

        caps = [int(i.shape[1]) for i in ins[2:] if i.name.startswith("past_comp_")]
        self.max_seq_len = (min(caps) - 2) * _MAX_COMPRESS_RATIO if caps else 0
        if caps and (max(caps) - 2) * _MIN_COMPRESS_RATIO != self.max_seq_len:
            raise ProtocolError("inconsistent compressed-cache capacities %s" % sorted(set(caps)))

        self._logits = {}

    def _zeros(self, spec):
        shape = [1 if isinstance(d, str) or d is None else d for d in spec.shape]
        return self.torch.zeros(shape, dtype=getattr(self.torch, _DTYPE[spec.type]),
                                device="cuda")

    def _logits_buf(self, seq_len):
        """Reuse one buffer per sequence length; decode only ever needs S=1."""
        buf = self._logits.get(seq_len)
        if buf is None:
            buf = self.torch.empty((1, seq_len, self.vocab), dtype=self.torch.float32,
                                   device="cuda")
            # Prefill buffers are large (S x 129280 x 4B) and each prompt has its own
            # length, so keep only the decode buffer and the newest prefill buffer.
            self._logits = {k: v for k, v in self._logits.items() if k == 1}
            self._logits[seq_len] = buf
        return buf

    def _step(self, ids, past):
        """One session run.  Returns the logits of the final position."""
        torch = self.torch
        src, dst = self.cache[self.cur], self.cache[1 - self.cur]
        seq_len = ids.shape[1]
        logits = self._logits_buf(seq_len)
        # past_lens carries one prior length per batch row; this engine runs a
        # single sequence at a time, so it is a one-element vector.
        past_t = torch.full((ids.shape[0],), past, dtype=torch.int64, device="cuda")

        io = self.sess.io_binding()
        for name, t in [("input_ids", ids), ("past_lens", past_t)] + list(zip(self.cache_in, src)):
            io.bind_input(name, "cuda", 0, _ELEM[str(t.dtype)], tuple(t.shape), t.data_ptr())
        for name, t in [("logits", logits)] + list(zip(self.cache_out, dst)):
            io.bind_output(name, "cuda", 0, _ELEM[str(t.dtype)], tuple(t.shape), t.data_ptr())
        self.sess.run_with_iobinding(io)

        self.cur = 1 - self.cur
        return logits[0, -1]

    def generate(self, prompt, max_new_tokens, eos_token_ids, chan):
        """Decode under the engine's direction.

        After every step the rank reports its own argmax and waits to be told
        which token to append: the ranks' logits are *nearly* but not always
        bit-identical, and a single disagreement would desynchronize the group
        for good.
        """
        torch = self.torch
        for buf in self.cache:
            for t in buf:
                t.zero_()
        self.cur = 0

        eos = set(eos_token_ids)
        budget = self.max_seq_len - len(prompt) if self.max_seq_len else max_new_tokens
        limit = max(0, min(max_new_tokens, budget))

        ids = torch.tensor([prompt], dtype=torch.int64, device="cuda")
        past, produced, stop, finite = 0, [], None, True
        t_prefill = t_decode = 0.0
        for step in range(limit):
            t0 = time.time()
            logits = self._step(ids, past)
            top = torch.topk(logits, 2)
            local = int(top.indices[0])  # the D2H read also synchronizes the stream
            gap = float(top.values[0] - top.values[1])
            dt = time.time() - t0
            if step == 0:
                t_prefill = dt
                finite = bool(torch.isfinite(logits).all())
            else:
                t_decode += dt

            chan.send({"event": "token", "rank": self.rank, "step": step,
                       "tok": local, "gap": gap})
            reply = chan.recv(_FOREVER)
            if reply.get("cmd") == "abort":
                stop = "abort"
                break
            nxt = reply["tok"]

            past += ids.shape[1]
            produced.append(nxt)
            if nxt in eos:
                stop = "eos"
                break
            ids = torch.tensor([[nxt]], dtype=torch.int64, device="cuda")

        if stop is None:
            stop = "length" if limit < max_new_tokens else "max_new_tokens"
        n = len(produced)
        return {
            "tokens": produced,
            "stop_reason": stop,
            "finite": finite,
            "prompt_len": len(prompt),
            "prefill_s": t_prefill,
            "decode_s": t_decode,
            "decode_tok_s": (n - 1) / t_decode if n > 1 and t_decode > 0 else 0.0,
        }


def _worker_main(a):
    chan = _Chan(socket.socket(fileno=int(os.environ["DSV4_FD"])))
    try:
        w = _Worker(a.model, a.rank)
    except Exception as e:  # report the failure instead of hanging the launcher
        import traceback
        traceback.print_exc()
        chan.send({"event": "error", "error": f"{type(e).__name__}: {e}"})
        return 1
    print(f"[rank {a.rank}] session ready in {w.load_s:.0f}s, "
          f"kv cache {w.bytes / 2**20:.1f} MiB, max_seq_len {w.max_seq_len}",
          file=sys.stderr, flush=True)
    chan.send({"event": "ready", "load_s": w.load_s, "max_seq_len": w.max_seq_len,
               "vocab": w.vocab, "kv_bytes": w.bytes})

    while True:
        try:
            req = chan.recv(_FOREVER)
        except (ProtocolError, TimeoutError):
            return 0
        if req.get("cmd") == "quit":
            return 0
        if req.get("cmd") != "generate":
            chan.send({"event": "generated", "error": "unknown command %r" % req.get("cmd")})
            continue
        try:
            reply = w.generate(req["prompt"], req["max_new_tokens"],
                               req.get("eos_token_ids") or [], chan)
        except Exception as e:
            import traceback
            traceback.print_exc()
            reply = {"error": f"{type(e).__name__}: {e}"}
        reply["event"] = "generated"
        chan.send(reply)


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #


def wait_for_listen(port, proc, timeout=1800):
    """Block until ``port`` reaches LISTEN state, or ``proc`` dies.

    Probed by reading /proc rather than connecting: a real connect would be
    accepted as one of the peers and corrupt the NCCL handshake.
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


class DSV4Engine:
    """Drives the eight rank processes as a single greedy-decoding model."""

    def __init__(self, model_dir, world=8, port=DEFAULT_PORT, log_dir="/tmp",
                 startup_timeout=1800, quiet=False):
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        self.world = world
        self.port = port
        self.log_dir = log_dir
        self.procs, self.chans = [], []
        self.max_seq_len = 0
        self.vocab = 0
        self._quiet = quiet
        self._start(startup_timeout)

    # -- lifecycle ---------------------------------------------------------- #

    def _log(self, msg):
        if not self._quiet:
            print(msg, flush=True)

    def _spawn(self, rank):
        parent_sock, child_sock = socket.socketpair()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), LOCAL_RANK=str(rank),
                   LOCAL_WORLD_SIZE=str(self.world), RANK0_IP="127.0.0.1",
                   RANK0_PORT=str(self.port), DSV4_FD=str(child_sock.fileno()))
        log = open(os.path.join(self.log_dir, f"dsv4_rank{rank}.log"), "w")
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker",
             "--rank", str(rank), "--model", self.model_dir],
            env=env, stdout=log, stderr=subprocess.STDOUT,
            pass_fds=(child_sock.fileno(),))
        child_sock.close()
        self.procs.append(proc)
        self.chans.append(_Chan(parent_sock))

    def _start(self, timeout):
        t0 = time.time()
        self._spawn(0)
        if self.world > 1:
            # Rank 0 binds the NCCL rendezvous socket deep inside session creation
            # and the peers only wait 40 s, so they must not start before it listens.
            if not wait_for_listen(self.port, self.procs[0], timeout):
                self.close()
                raise ProtocolError(
                    f"rank 0 never listened on port {self.port}; see "
                    f"{self.log_dir}/dsv4_rank0.log")
            self._log("rank 0 is listening; starting the remaining ranks")
            for r in range(1, self.world):
                self._spawn(r)

        for rank, chan in enumerate(self.chans):
            msg = chan.recv(timeout)
            if msg.get("event") != "ready":
                self.close()
                raise ProtocolError(f"rank {rank} failed to start: {msg.get('error')}")
            self.max_seq_len = msg["max_seq_len"]
            self.vocab = msg["vocab"]
            self._kv_bytes = msg["kv_bytes"]
        self._log(f"all {self.world} ranks ready in {time.time() - t0:.0f}s "
                  f"(max_seq_len={self.max_seq_len}, "
                  f"kv cache {self._kv_bytes / 2**20:.1f} MiB/rank)")

    def close(self):
        for chan in self.chans:
            try:
                chan.send({"cmd": "quit"})
            except OSError:
                pass
        for proc in self.procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        for chan in self.chans:
            chan.close()
        self.procs, self.chans = [], []

    def _kill(self):
        for proc in self.procs:
            proc.kill()
        for chan in self.chans:
            chan.close()
        self.procs, self.chans = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- inference ---------------------------------------------------------- #

    def generate(self, prompt_ids, max_new_tokens=64, eos_token_ids=(1,), timeout=600):
        """Greedily continue ``prompt_ids``.  Returns rank 0's reply dict.

        The engine drives the loop token by token: every rank reports its own
        argmax, rank 0's wins, and the decision is broadcast back.  The ranks'
        logits agree to the last bit almost always but not quite always, and one
        divergence would leave the group permanently out of step - the losing
        rank would feed a different token into the next `AllReduce`.  The count
        of disagreements is returned as `rank_disagreements`.
        """
        if not self.procs:
            raise ProtocolError("engine is closed")
        prompt_ids = [int(t) for t in prompt_ids]
        if self.max_seq_len and len(prompt_ids) >= self.max_seq_len:
            raise ValueError(f"prompt of {len(prompt_ids)} tokens exceeds the export's "
                             f"max_seq_len of {self.max_seq_len}")
        for chan in self.chans:
            chan.send({"cmd": "generate", "prompt": prompt_ids,
                       "max_new_tokens": max_new_tokens,
                       "eos_token_ids": list(eos_token_ids)})

        disagreements = []
        while True:
            msgs = [self._recv(rank, chan, timeout) for rank, chan in enumerate(self.chans)]
            kinds = {m.get("event") for m in msgs}
            if kinds == {"token"}:
                toks = [m["tok"] for m in msgs]
                if len(set(toks)) > 1:
                    disagreements.append({"step": msgs[0]["step"], "tokens": toks,
                                          "gap": min(m["gap"] for m in msgs)})
                for chan in self.chans:
                    chan.send({"tok": toks[0]})
                continue
            if kinds == {"generated"}:
                break
            self._kill()
            raise ProtocolError(f"ranks reported mixed events {kinds}; "
                                f"see {self.log_dir}/dsv4_rank*.log")

        for rank, msg in enumerate(msgs):
            if msg.get("error"):
                self._kill()
                raise ProtocolError(f"rank {rank}: {msg['error']}")
        out = dict(msgs[0])
        out["rank_disagreements"] = disagreements
        if disagreements:
            self._log(f"note: {len(disagreements)} step(s) where the ranks' argmax "
                      f"differed (smallest top-2 gap "
                      f"{min(d['gap'] for d in disagreements):.2e}); rank 0 won")
        return out

    def _recv(self, rank, chan, timeout):
        # A rank that dies mid-collective leaves the others blocked in NCCL, so the
        # wait is bounded and a miss tears the whole group down.
        try:
            return chan.recv(timeout)
        except (TimeoutError, ProtocolError) as e:
            self._kill()
            raise ProtocolError(
                f"rank {rank} did not answer ({e}); see {self.log_dir}/dsv4_rank*.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.worker:
        ap.error("dsv4_engine.py is a library; use run.py or import DSV4Engine")
    return _worker_main(a)


if __name__ == "__main__":
    raise SystemExit(main())
