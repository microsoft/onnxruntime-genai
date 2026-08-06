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
argmax and rank 0's wins.  `lm_head` is replicated, so the ranks' logits agree
bit for bit almost always - but not quite always, and a single disagreement
would desynchronize the group permanently, because the losing rank would feed a
different token into the next `AllReduce`.  Disagreements are counted and
reported, not tolerated silently.

The agreement itself happens *on the device*.  The ranks hold a second NCCL
communicator (a `torch.distributed` group, rendezvous handed out by the engine)
and each step is one `all_gather` of a 2*batch+1 float vector - every rank's
argmax, its top-2 gap, and its abort flag.  Rank 0's slot of the result is
written straight into the next step's `input_ids` without the host ever seeing
it, so the next graph replay is launched immediately.  A non-blocking D2H of the
agreed token runs alongside the replay and the host does its EOS / length
bookkeeping *one step late*, which costs at most one extra decode step per
request, whose logits are then discarded.  `DSV4_HOST_TOKEN_SYNC=1` restores the
older path, where every rank round-tripped its argmax through the engine process
once per token.

Usage::

    from dsv4_engine import DSV4Engine

    with DSV4Engine("~/dsv4_onnx", world=8) as engine:
        out = engine.generate(prompt_ids, max_new_tokens=64, eos_token_ids=[1])
        print(out["tokens"], out["stop_reason"], out["decode_tok_s"])
"""

import argparse
import atexit
import json
import os
import select
import socket
import subprocess
import sys
import time

# onnxruntime's ONNX_TENSOR_ELEMENT_DATA_TYPE values, for io_binding.
_ELEM = {"torch.float32": 1, "torch.int32": 6, "torch.int64": 7, "torch.bool": 9,
         "torch.float16": 10, "torch.bfloat16": 16}
_DTYPE = {"tensor(float)": "float32", "tensor(bfloat16)": "bfloat16",
          "tensor(float16)": "float16", "tensor(int32)": "int32",
          "tensor(int64)": "int64", "tensor(bool)": "bool"}
# Rows of `main_hidden` handed to the drafter per run.  Pinned rather than sized to
# the accepted run: ORT only captures a graph when every node is on CUDA, and the
# drafter's shape arithmetic is CPU-placed for as long as `main_len` is symbolic.
# Must be >= the most tokens one verify can commit, and divide the ring window.
_MH_PAD = 8

DEFAULT_PORT = 19555

# Single output carrying max |x| for every intermediate, added by probe_ranges.py.
PROBE_OUTPUT = "probe__all"
# A worker blocks indefinitely between requests; only the engine bounds its waits.
_FOREVER = 10 ** 9
# Compressed-cache capacity is `max_seq_len // ratio + 2`; the ratio-128 layers
# give the smallest one.  Used to recover the export's context bound.
_MIN_COMPRESS_RATIO = 4
_MAX_COMPRESS_RATIO = 128

# Uncaptured decode runs per parity before the graph is recorded.
_GRAPH_WARMUP = 3

# Decode steps between the worker's liveness pings.  Nothing is expected back;
# they exist so the engine's bounded `recv` still notices a dead rank now that a
# step no longer talks to it.
_PROGRESS_EVERY = 16


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

    def __init__(self, model_dir, rank, prefill_chunk=512, batch=1, cuda_graph=False):
        import onnxruntime as ort
        import torch

        self.torch = torch
        self.ort = ort
        t0 = time.time()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if os.environ.get("DSV4_PROFILE"):
            so.enable_profiling = True
            so.profile_file_prefix = f"/tmp/dsv4_prof_rank{rank}"
        path = os.path.join(model_dir, f"rank_{rank}", "model.onnx")
        # Speculation runs the target over a block of candidates, so the decode shape
        # changes from step to step and the session must not be in CUDA graph mode at
        # all -- with `enable_cuda_graph` on, a run that is not given an explicit
        # graph id is captured and then replayed against whatever it recorded.
        # Capture is worth nothing here anyway: the step is GPU-bound, not
        # launch-bound (12.163 ms captured vs 12.156 ms not).
        mtp_path = os.path.join(model_dir, f"rank_{rank}", "mtp.onnx")
        self.has_mtp = (os.path.exists(mtp_path)
                        and os.environ.get("DSV4_NO_MTP") != "1")
        cuda_graph = cuda_graph and not self.has_mtp
        # A decode step launches ~23k kernels but keeps the GPU busy only ~28% of the
        # step, so the wall clock is set by launch dispatch rather than by any kernel.
        # Capturing the step collapses those launches into one graph replay.
        #
        # A replay re-runs the addresses it recorded, so every activation has to keep
        # the address it had at capture time.  ORT has no allocator support for that:
        # it simply relies on a run allocating the same sizes in the same order and so
        # getting the same addresses back.  Leaving the memory pattern on is what makes
        # that hold here -- a decode step then takes one pooled block of a fixed size
        # instead of thousands of individual allocations whose placement depends on
        # whatever the interleaved prefill runs did to the arena.  `kSameAsRequested`
        # stops the arena doubling, which would move the block outright.
        provider = (("CUDAExecutionProvider",
                     {"enable_cuda_graph": "1", "arena_extend_strategy": "kSameAsRequested"})
                    if cuda_graph else "CUDAExecutionProvider")
        self.cuda_graph = cuda_graph
        self.sess = ort.InferenceSession(path, so, providers=[provider])
        if "CUDAExecutionProvider" not in self.sess.get_providers():
            # Without it the session silently falls back to CPU, which has no
            # bfloat16 kernels, and the first failing node is blamed instead.
            raise ProtocolError("CUDA EP unavailable; put cuDNN 9 and CUDA 13 on "
                                "LD_LIBRARY_PATH (see the export doc)")
        self.load_s = time.time() - t0
        self.rank = rank

        # The DSpark drafter, if this export carries one.  It is a second session over
        # the same devices; ORT's NCCL communicator is a function-local static
        # (nccl_kernels.cc:255), so both graphs share one communicator and there is no
        # second rendezvous to collide with.
        self.mtp = None
        if self.has_mtp:
            mso = so
            if os.environ.get("DSV4_MTP_PROFILE") and rank == 0:
                mso = ort.SessionOptions()
                mso.log_severity_level = 3
                mso.enable_profiling = True
                mso.profile_file_prefix = "/tmp/mtp_prof"
            # The drafter's shapes are fixed per accepted-run length, so unlike the
            # target it can be captured. Graph replay removes host dispatch overhead;
            # the remaining time is dominated by the small-token QMoE kernels.
            self.mtp_graph = os.environ.get("DSV4_NO_MTP_GRAPH") != "1"
            if self.mtp_graph:
                # Folds the shape arithmetic to constants, which is what moves the last
                # nine nodes off the CPU and lets the EP capture at all.
                mso = ort.SessionOptions() if mso is so else mso
                mso.log_severity_level = int(os.environ.get("DSV4_MTP_LOG", "3"))
                mso.add_free_dimension_override_by_name("batch_size", batch)
                mso.add_free_dimension_override_by_name("main_len", _MH_PAD)
            mprov = (("CUDAExecutionProvider",
                      {"enable_cuda_graph": "1",
                       "arena_extend_strategy": "kSameAsRequested"})
                     if self.mtp_graph else "CUDAExecutionProvider")
            self.mtp = ort.InferenceSession(mtp_path, mso, providers=[mprov])
            if os.environ.get("DSV4_MTP_PLACEMENT") and rank == 0:
                # A single CPU-placed node in the drafter serialises the whole graph,
                # so confirm the placement before blaming the kernels.
                pso = ort.SessionOptions()
                pso.log_severity_level = 0
                pso.optimized_model_filepath = "/tmp/mtp_opt.onnx"
                ort.InferenceSession(mtp_path, pso, providers=["CUDAExecutionProvider"])
            mm = self.mtp.get_modelmeta().custom_metadata_map
            self.mtp_block = int(mm["dsv4_mtp_block_size"])
            self.mtp_window = int(mm["dsv4_window"])
            self.mtp_cache_in = [i.name for i in self.mtp.get_inputs()
                                 if i.name.startswith("past_kv_")]
            self.mtp_cache_out = ["present" + n[4:] for n in self.mtp_cache_in]
            self.mtp_cache = None

        ins, outs = self.sess.get_inputs(), self.sess.get_outputs()
        if [i.name for i in ins[:2]] != ["input_ids", "past_lens"]:
            raise ProtocolError("unexpected graph inputs: %s" % [i.name for i in ins[:3]])
        self.vocab = int(outs[0].shape[-1])

        # `main_hidden` is the DSpark drafting contract, not a cache: it is produced
        # fresh every step and never fed back in.
        out_names = [o.name for o in outs]
        self.mtp_dim = (int(outs[out_names.index("main_hidden")].shape[-1])
                        if "main_hidden" in out_names else 0)
        self._mh = {}

        meta = self.sess.get_modelmeta().custom_metadata_map
        self.paged = meta.get("dsv4_paged") == "1"
        self.prefill_chunk = prefill_chunk
        self.batch = batch
        if batch != 1 and not self.paged:
            raise ProtocolError("batch > 1 needs a paged export")

        # Block tables are engine-owned constants, not caches; they are bound as plain
        # inputs and never appear as outputs.
        self.const_in = {}
        if self.paged:
            self._init_paged(ins, meta)
        cache_ins = [i for i in ins[2:] if i.name not in self.const_in]

        self.cache_in = [i.name for i in cache_ins]
        self.cache_out = [o.name for o in outs[1:]
                          if o.name not in ("main_hidden", PROBE_OUTPUT)]
        if len(self.cache_in) != len(self.cache_out):
            raise ProtocolError("past/present count mismatch")
        self._init_probe(outs)
        # Pooled entries are addressed by the block table, so their first dimension is
        # blocks rather than rows and they must never be sliced per batch row.
        self._pooled = [n in getattr(self, "pool_blocks", {}) for n in self.cache_in]
        # The compressor and indexer window states are the one kind of cache that is
        # NOT addressed by absolute position: each run re-cuts them to the last L raw
        # projections, so a run advances them by however many positions it consumed.
        # Speculation accepts fewer than it runs, so they have to be re-cut by hand.
        self._slide = ["cstate" in n for n in self.cache_in]

        # Two full copies of the cache, allocated once and ping-ponged.  The paged
        # kv cache is the exception: PagedAttention updates it in place and ORT
        # requires the input and output buffers to be the same, so both copies of
        # those entries are one tensor.
        self.cache = [[self._zeros(i) for i in cache_ins] for _ in range(2)]
        if self.paged:
            for j, i in enumerate(cache_ins):
                if i.name.startswith("past_kv_"):
                    self.cache[1][j] = self.cache[0][j]
        self.cur = 0
        seen = set()
        self.bytes = 0
        for buf in self.cache:
            for t in buf:
                if t.data_ptr() not in seen:
                    seen.add(t.data_ptr())
                    self.bytes += t.numel() * t.element_size()
        self.bytes += sum(t.numel() * t.element_size() for t in self.const_in.values())

        if self.paged:
            self.max_seq_len = int(meta["dsv4_max_seq_len"])
        else:
            caps = [int(i.shape[1]) for i in ins[2:] if i.name.startswith("past_comp_")]
            self.max_seq_len = (min(caps) - 2) * _MAX_COMPRESS_RATIO if caps else 0
            if caps and (max(caps) - 2) * _MIN_COMPRESS_RATIO != self.max_seq_len:
                raise ProtocolError("inconsistent compressed-cache capacities %s"
                                    % sorted(set(caps)))

        self._logits = {}
        self._graph_io = None
        self._ro_skip = None
        # Second communicator, established later by `init_dist`; until then the
        # rank falls back to routing every token through the engine process.
        self.dist = None
        self.dist_ready = False
        self.world = 1
        self._abort = False
        if cuda_graph:
            self._init_graph_capture()

    # -- token agreement communicator -------------------------------------- #

    def init_dist(self, rendezvous, world):
        """Join the ranks' own NCCL group, used only to agree on the next token.

        ORT already owns a communicator for the in-graph `AllReduce`, and two
        communicators in one process are safe only while every rank issues their
        collectives in the same order.  That holds here for two reasons.  ORT
        synchronizes its stream at the end of `Run`, so a step is strictly
        replay -> agree -> replay with nothing in flight across the boundary;
        and the agreement is a single unconditional `all_gather` per decode
        iteration whose result is the *only* thing the loop branches on, so no
        rank can reach a different iteration count from another.

        The rendezvous is a file rather than a socket on purpose: rank 0 already
        binds the ORT NCCL rendezvous port inside session creation and the
        engine probes that port by reading /proc precisely because a stray
        connection corrupts the handshake.  A `FileStore` adds no listener to
        collide with, and NCCL's own bootstrap picks its ports itself.
        """
        import torch.distributed as dist
        torch = self.torch
        if self.vocab >= 2 ** 24:
            # Votes ride in a float32 payload; ids have to survive it exactly.
            raise ProtocolError("vocab %d is too large for the float32 vote"
                                % self.vocab)
        torch.cuda.set_device(0)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method=rendezvous,
                                    world_size=world, rank=self.rank)
        self.dist = dist
        self.world = world
        # Build the communicator here, at a point every rank reaches with no ORT
        # work in flight, rather than lazily inside the first decode step.
        buf = torch.zeros((world, 2 * self.batch + 1), dtype=torch.float32, device="cuda")
        dist.all_gather_into_tensor(buf, buf[0].clone())
        torch.cuda.synchronize()
        self.dist_ready = True

    # -- cuda graph -------------------------------------------------------- #

    def _init_graph_capture(self):
        """Pin the decode step's buffers and pre-build one binding per cache parity.

        A captured graph replays the addresses it was captured with, so every tensor
        the decode step touches has to live at a fixed address.  The caches are
        already allocated once, but they are *ping-ponged*: step N reads copy `cur`
        and writes copy `1 - cur`.  Rather than give that up, capture both parities
        as two separate graphs and select one per step with ORT's `gpu_graph_id`.
        The paged kv entries are the same tensor in both copies, so only the
        compressor and indexer caches actually alternate.

        `input_ids` and `past_lens` become persistent buffers written in place; a
        fresh `torch.full` every step would hand ORT a new address each time.
        """
        torch, ort = self.torch, self.ort
        self.g_ids = torch.zeros((self.batch, 1), dtype=torch.int64, device="cuda")
        self.g_past = torch.zeros((self.batch,), dtype=torch.int64, device="cuda")
        self.g_logits = self._logits_buf(1, self.batch)
        if self.mtp_dim:
            self.g_mh = self._mh_buf(1, self.batch)

        # Prefill is not capturable -- every chunk is a different shape -- so it runs
        # under the reserved id that tells the EP to skip capture entirely.
        self._ro_skip = ort.RunOptions()
        self._ro_skip.add_run_config_entry("gpu_graph_id", "-1")
        self._ro = []
        self._graph_io = []
        for parity in range(2):
            ro = ort.RunOptions()
            ro.add_run_config_entry("gpu_graph_id", str(parity + 1))
            self._ro.append(ro)
            io = self.sess.io_binding()
            src, dst = self.cache[parity], self.cache[1 - parity]
            for name, t in ([("input_ids", self.g_ids), ("past_lens", self.g_past)]
                            + list(self.const_in.items()) + list(zip(self.cache_in, src))):
                io.bind_input(name, "cuda", 0, _ELEM[str(t.dtype)], tuple(t.shape), t.data_ptr())
            outs = [("logits", self.g_logits)] + list(zip(self.cache_out, dst))
            if self.mtp_dim:
                outs.append(("main_hidden", self.g_mh))
            for name, t in outs:
                io.bind_output(name, "cuda", 0, _ELEM[str(t.dtype)], tuple(t.shape), t.data_ptr())
            self._graph_io.append(io)

        # Capture records allocations as-is, so anything the first runs do lazily --
        # arena growth, cuBLAS handles, the MoE kernels' tactic search -- has to be
        # done before the recording starts, or the graph replays against pointers
        # that were never really reserved.  These runs use the skip id, so the EP
        # executes them normally and captures nothing.
        for parity in range(2):
            for _ in range(_GRAPH_WARMUP):
                self.sess.run_with_iobinding(self._graph_io[parity], self._ro_skip)

        # Then record both graphs, so that the first real decode step replays rather
        # than paying for a capture.  The EP wants `min_num_runs_before_cuda_graph
        # _capture_` ordinary runs per id before it records, so drive each id until
        # it stops executing and starts replaying.
        for parity in range(2):
            for _ in range(_GRAPH_WARMUP):
                self.sess.run_with_iobinding(self._graph_io[parity], self._ro[parity])
        self.torch.cuda.synchronize()

    # -- paged cache ------------------------------------------------------- #

    def _init_paged(self, ins, meta):
        """Size the block pools and build the (static) block tables.

        Logical position space per ratio group is ``[0, max_seq_len)`` for tokens with
        the compressor's rows stacked directly above.  The compressed rows must all be
        retained, but a query only ever reads the last ``window`` token positions, so
        the token region maps onto a small ring of physical blocks::

            physical(logical_block) = base + (lb % ring)          lb <  max_seq_len/bs
                                    = base + ring + lb - L/bs     otherwise

        The op stores every row of a step *before* it attends, so the ring must hold
        the whole span a step can touch -- ``window + S - 1`` positions, where ``S`` is
        the largest chunk `generate` will submit.  That is why prefill is chunked: a
        one-shot prefill of a 1M prompt would need a ring as big as the prompt.
        """
        torch = self.torch
        bs = int(meta["dsv4_block_size"])
        L = int(meta["dsv4_max_seq_len"])
        window = int(meta["dsv4_window"])
        if bs <= 0 or L % bs:
            raise ProtocolError("max_seq_len %d is not a multiple of block_size %d" % (L, bs))
        self.block_size, self.window = bs, window
        tok_blocks = L // bs
        ring = -(-(window + self.prefill_chunk - 1) // bs)
        if ring > tok_blocks:
            ring = tok_blocks
        self.ring_blocks = ring

        tables = {i.name: i for i in ins if i.name.startswith("block_table_")}
        if not tables:
            raise ProtocolError("dsv4_paged=1 but the graph has no block_table_* input")
        lb = torch.arange(max(int(i.shape[1]) for i in tables.values()), dtype=torch.int64)
        self.group_seq_blocks = {}
        for name, spec in tables.items():
            r = int(name.rsplit("_", 1)[1])
            gb = int(spec.shape[1])
            comp_blocks = gb - tok_blocks
            if comp_blocks < 0 or (r == 0 and comp_blocks):
                raise ProtocolError("unexpected block-table width %d for ratio %d" % (gb, r))
            per_seq = ring + comp_blocks
            self.group_seq_blocks[r] = per_seq
            l = lb[:gb]
            phys = torch.where(l < tok_blocks, l % ring, ring + (l - tok_blocks))
            # Sequence b owns the contiguous physical run [b * per_seq, (b+1) * per_seq).
            base = torch.arange(self.batch, dtype=torch.int64).unsqueeze(1) * per_seq
            self.const_in[name] = (phys.unsqueeze(0) + base).to(torch.int32).contiguous().cuda()

        # `num_blocks_<ratio>` is the only place the layer -> group map survives.
        self.pool_blocks = {}
        for i in ins:
            if i.name.startswith("past_kv_") and isinstance(i.shape[0], str):
                r = int(i.shape[0].rsplit("_", 1)[1])
                self.pool_blocks[i.name] = self.batch * self.group_seq_blocks[r]

    def _zeros(self, spec):
        shape = [1 if isinstance(d, str) or d is None else d for d in spec.shape]
        if spec.name in getattr(self, "pool_blocks", {}):
            shape[0] = self.pool_blocks[spec.name]
        else:
            shape[0] = self.batch
        return self.torch.zeros(shape, dtype=getattr(self.torch, _DTYPE[spec.type]),
                                device="cuda")

    def _logits_buf(self, seq_len, rows):
        """Reuse one buffer per (rows, sequence length); decode only ever needs S=1."""
        buf = self._logits.get((rows, seq_len))
        if buf is None:
            buf = self.torch.empty((rows, seq_len, self.vocab),
                                   dtype=self.torch.float32, device="cuda")
            # Prefill buffers are large (S x 129280 x 4B) and each prompt has its own
            # length, so keep only the decode buffer and the newest prefill buffer.
            self._logits = {k: v for k, v in self._logits.items() if k[1] == 1}
            self._logits[(rows, seq_len)] = buf
        return buf

    def _mh_buf(self, seq_len, rows):
        """The main-hidden output, sized like the logits buffer and kept per shape."""
        buf = self._mh.get((rows, seq_len))
        if buf is None:
            buf = self.torch.empty((rows, seq_len, self.mtp_dim),
                                   dtype=self.torch.bfloat16, device="cuda")
            self._mh = {k: v for k, v in self._mh.items() if k[1] <= 8}
            self._mh[(rows, seq_len)] = buf
        return buf

    def _step(self, ids, past_t, row=None, full=False):
        """One session run.  Returns the final position's logits, one row per sequence.

        ``row`` restricts the run to a single batch row, which is how prefill works:
        prompts have different lengths, so each sequence is filled independently.
        Pooled kv entries are shared and reached through the block table, so only the
        row-shaped caches and block tables get sliced.  ``full`` returns every
        position's logits, which is what verifying a speculated block needs.
        """
        torch = self.torch
        if row is None and self._graph_io is not None and ids.shape[1] == 1 and not full:
            return self._step_captured(ids, past_t)
        src, dst = self.cache[self.cur], self.cache[1 - self.cur]
        sl = slice(None) if row is None else slice(row, row + 1)
        rows = ids.shape[0]
        logits = self._logits_buf(ids.shape[1], rows)

        io = self.sess.io_binding()
        inputs = ([("input_ids", ids), ("past_lens", past_t)]
                  + [(n, t[sl]) for n, t in self.const_in.items()]
                  + [(n, t if p else t[sl])
                     for n, t, p in zip(self.cache_in, src, self._pooled)])
        for name, t in inputs:
            io.bind_input(name, "cuda", 0, _ELEM[str(t.dtype)], tuple(t.shape), t.data_ptr())
        outputs = ([("logits", logits)]
                   + [(n, t if p else t[sl])
                      for n, t, p in zip(self.cache_out, dst, self._pooled)])
        if self.mtp_dim:
            self.mh_last = self._mh_buf(ids.shape[1], rows)
            outputs.append(("main_hidden", self.mh_last))
        for name, t in outputs:
            io.bind_output(name, "cuda", 0, _ELEM[str(t.dtype)], tuple(t.shape), t.data_ptr())
        if self._probe is not None:
            b = self._probe
            io.bind_output(PROBE_OUTPUT, "cuda", 0, _ELEM[str(b.dtype)], tuple(b.shape),
                           b.data_ptr())
        self.sess.run_with_iobinding(io, self._ro_skip)
        if self._probe is not None:
            torch.maximum(self._probe_max, self._probe, out=self._probe_max)

        self.cur = 1 - self.cur
        return logits if full else logits[:, -1]

    def _init_probe(self, outs):
        """Wire up the range probes that probe_ranges.py adds to a model, if present.

        A probed export carries one extra output holding max |x| for every intermediate
        tensor; this keeps a running maximum over every step and writes it out at exit.
        Absent that output the whole facility costs one None check per step.
        """
        self._probe = self._probe_max = None
        path = os.environ.get("DSV4_PROBE_OUT")
        probe = next((o for o in outs if o.name == PROBE_OUTPUT), None)
        if path is None or probe is None:
            return
        n = int(probe.shape[0])
        self._probe = self.torch.zeros(n, dtype=self.torch.float32, device="cuda")
        self._probe_max = self.torch.zeros(n, dtype=self.torch.float32, device="cuda")
        atexit.register(lambda: json.dump(self._probe_max.cpu().tolist(),
                                          open(f"{path}.rank{self.rank}.json", "w")))

    def _prefill_row(self, b, prompt):
        """Chunked prefill for one batch row.  Returns that row's final logits.

        Rows run a different number of chunks and so end on different parities. The
        batched decode loop reads one parity for the whole batch, so each row's fresh
        state is mirrored into the other copy before the next row starts.
        """
        torch = self.torch
        ids = torch.tensor([prompt], dtype=torch.int64, device="cuda")
        chunk = self.prefill_chunk if self.paged else len(prompt)
        past, logits = 0, None
        while ids.shape[1] > 0:
            take = min(chunk, ids.shape[1])
            past_t = torch.full((1,), past, dtype=torch.int64, device="cuda")
            logits = self._step(ids[:, :take], past_t, row=b)
            if self.mtp is not None:
                self._push_main(self.mh_last)
            past += take
            ids = ids[:, take:]
        for j, pooled in enumerate(self._pooled):
            if not pooled:
                self.cache[1 - self.cur][j][b].copy_(self.cache[self.cur][j][b])
        return logits[0].clone()

    def _step_captured(self, ids, past_t):
        """One decode step through the captured graph for the current parity."""
        torch = self.torch
        parity = self.cur
        self.g_ids.copy_(ids)
        self.g_past.copy_(past_t)
        # The two writes above are queued on torch's stream, the replay runs on ORT's.
        # Nothing orders them, so publish them before handing over.
        torch.cuda.current_stream().synchronize()
        self.sess.run_with_iobinding(self._graph_io[parity], self._ro[parity])
        self.cur = 1 - parity
        return self.g_logits[:, -1]

    def _prefill(self, prompts, max_new_tokens):
        """Zero the caches, fill every row, and size the decode budget."""
        torch = self.torch
        if len(prompts) > self.batch:
            raise ProtocolError("%d prompts for a batch-%d worker" % (len(prompts), self.batch))
        for buf in self.cache:
            for t in buf:
                t.zero_()
        self.cur = 0
        self._abort = False
        if self.mtp is not None:
            self._init_mtp_state()

        lens = [len(p) for p in prompts]
        limits = [max(0, min(max_new_tokens, self.max_seq_len - n if self.max_seq_len
                             else max_new_tokens)) for n in lens]

        t0 = time.time()
        rows = [self._prefill_row(b, p) for b, p in enumerate(prompts)]
        # Unused rows still run; leave them empty so they only ever attend to their
        # own current token and cannot produce NaNs out of the zeroed cache.
        while len(rows) < self.batch:
            rows.append(torch.zeros_like(rows[0]))
            lens.append(0)
            limits.append(0)
        logits = torch.stack(rows)
        return (logits, lens, limits, time.time() - t0,
                bool(torch.isfinite(logits[:len(prompts)]).all()))

    @staticmethod
    def _reply(produced, stop, finite, lens, n, t_prefill, t_decode, steps):
        return {
            "tokens": produced,
            "stop_reason": stop,
            "finite": finite,
            "prompt_lens": lens[:n],
            "prefill_s": t_prefill,
            "decode_s": t_decode,
            "decode_step_s": t_decode / steps if steps else 0.0,
            "decode_tok_s": (steps * n) / t_decode if steps and t_decode else 0.0,
        }

    def generate(self, prompts, max_new_tokens, eos_token_ids, chan):
        """Decode a batch of prompts.  The ranks agree on every token they emit.

        Every row steps on every iteration, finished ones included. The step costs
        the same either way -- it is ~18k kernel launches whatever the rows contain --
        and holding the batch shape fixed is what lets the step stay a replayable
        graph.
        """
        if self.mtp is not None and self.dist_ready and self.batch == 1:
            return self._generate_spec(prompts, max_new_tokens, eos_token_ids, chan)
        if self.dist_ready and os.environ.get("DSV4_HOST_TOKEN_SYNC") != "1":
            return self._generate_device(prompts, max_new_tokens, eos_token_ids, chan)
        return self._generate_host(prompts, max_new_tokens, eos_token_ids, chan)

    # -- decode: agreement on the device ------------------------------------ #

    def _abort_pending(self, chan):
        """Drain the engine channel without blocking.  Rank-local on purpose: the
        flag only takes effect after the all-gather has made it unanimous."""
        if self._abort:
            return True
        try:
            while b"\n" in chan.buf or select.select([chan.sock], [], [], 0)[0]:
                if chan.recv(1.0).get("cmd") == "abort":
                    self._abort = True
                    break
        except (TimeoutError, ProtocolError, OSError):
            pass
        return self._abort

    def _disagreements(self, hist, n_steps):
        """Every (step, row) where the ranks' argmax differed, with all the votes.

        Reconstructed at the end from the gathered history rather than reported
        per step, so the decode loop never has to move a vote to the host.
        """
        B = self.batch
        h = hist[:n_steps].cpu()
        votes = h[:, :, :B]
        out = []
        for s, b in (votes != votes[:, :1, :]).any(dim=1).nonzero().tolist():
            out.append({"step": s, "row": b,
                        "tokens": [int(v) for v in votes[s, :, b].tolist()],
                        "gap": float(h[s, :, B + b].min())})
        return out

    def _generate_device(self, prompts, max_new_tokens, eos_token_ids, chan):
        """Decode with the token decided by an on-device all-gather.

        The host never sees a token before the step that consumes it is launched.
        It reads step ``s``'s token while step ``s+1`` is already running, so a row
        that hits EOS is noticed one iteration late: the replay for the step after
        the last kept token has already gone out, and its logits are dropped.  The
        tokens themselves are exactly the ones the host-synchronous path returns,
        because the same number of argmaxes is taken and the same prefix of them
        is committed.  Stopping by length needs no token at all, so that case runs
        exactly as many replays as before.
        """
        torch, dist = self.torch, self.dist
        logits, lens, limits, t_prefill, finite = self._prefill(prompts, max_new_tokens)
        n, B, W = len(prompts), self.batch, self.world
        eos = set(eos_token_ids)
        max_steps = max(limits) if limits else 0

        past_t = torch.tensor(lens, dtype=torch.int64, device="cuda")
        # `agreed` is the step's decision: B token ids plus the abort flag.  The
        # ids double as the next `input_ids`, so the same buffer is both the
        # all-gather's landing pad and the graph's input -- nothing copies between.
        agreed = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
        step_ids = agreed[:B].view(B, 1)
        produced = [[] for _ in prompts]
        done = [limits[b] == 0 for b in range(n)]
        stop = [("length" if limits[b] < max_new_tokens else "max_new_tokens")
                for b in range(n)]
        t_decode, steps, last = 0.0, 0, -1

        if max_steps:
            # [my argmax | my top-2 gap | my abort flag], gathered into `hist` so
            # the whole vote history is on the device for the post-mortem.
            send = torch.zeros(2 * B + 1, dtype=torch.float32, device="cuda")
            hist = torch.zeros((max_steps, W, 2 * B + 1), dtype=torch.float32,
                               device="cuda")
            hpin = torch.zeros((max_steps, B + 1), dtype=torch.int64).pin_memory()
            poll = self.rank == 0        # only rank 0's abort slot is ever read
            uncaptured = self._graph_io is None

            def commit(s):
                """Apply step ``s``'s agreed token.  True if the loop must stop."""
                nonlocal last
                last = s
                row = hpin[s].tolist()
                if row[B]:
                    for b in range(n):
                        if not done[b]:
                            stop[b] = "abort"
                    return True
                for b in range(n):
                    if done[b]:
                        continue
                    produced[b].append(row[b])
                    if row[b] in eos:
                        done[b], stop[b] = True, "eos"
                    elif len(produced[b]) >= limits[b]:
                        done[b] = True
                return all(done)

            stopped = False
            for step in range(max_steps):
                # Step `step - 1`'s token, copied out while `step - 1` was still
                # running: `_step` synchronizes torch's stream on its way into the
                # replay, which publishes the copy issued just before it.  This sits
                # ahead of the sampling so that a stop costs exactly one extra
                # replay -- the one already launched -- and no more.
                if step:
                    if commit(step - 1):
                        stopped = True
                        break
                if step % _PROGRESS_EVERY == 0:
                    chan.send({"event": "progress", "rank": self.rank, "step": step})
                top = torch.topk(logits, 2, dim=-1)
                send[:B].copy_(top.indices[:, 0])
                torch.sub(top.values[:, 0], top.values[:, 1], out=send[B:2 * B])
                if poll and self._abort_pending(chan):
                    send[2 * B] = 1.0
                # The one collective of the step.  Unconditional on every rank, and
                # everything the loop branches on below is derived from its result,
                # which is identical on all ranks -- so the ranks cannot diverge.
                dist.all_gather_into_tensor(hist[step], send)
                g = hist[step]
                agreed[:B].copy_(g[0, :B])      # rank 0's argmax wins, and is also
                agreed[B].copy_(g[0, 2 * B])    # already the next step's input_ids
                hpin[step].copy_(agreed, non_blocking=True)

                # Launched without the host having seen the token it consumes.
                # The last iteration is skipped: no token can outlive it.
                if step + 1 < max_steps:
                    if uncaptured:
                        # The captured path synchronizes torch's stream itself; the
                        # uncaptured one binds `step_ids` straight into the session,
                        # so the writes above have to be published by hand.
                        torch.cuda.current_stream().synchronize()
                    t1 = time.time()
                    logits = self._step(step_ids, past_t)
                    t_decode += time.time() - t1
                    steps += 1
                    past_t += 1
            if not stopped:
                # Nothing replayed after the last iteration, so publish its copy.
                torch.cuda.current_stream().synchronize()
                commit(max_steps - 1)

        out = self._reply(produced, stop, finite, lens, n, t_prefill, t_decode, steps)
        # Only rank 0's copy is read, and it is the same on every rank anyway.
        if self.rank == 0:
            out["disagreements"] = (self._disagreements(hist, last + 1)
                                    if max_steps and last >= 0 else [])
        return out

    # -- decode: DSpark speculation ---------------------------------------- #

    def _init_mtp_state(self):
        """Paged ring caches for the drafter, plus the rolling window of main states.

        The drafter's attention reads a `window`-row ring of main states, one row per
        position the target has committed.  Priming it needs the last `window` rows of
        the prompt's `main_hidden`, so prefill keeps them in `mh_hist` as it goes.
        """
        torch = self.torch
        W, B = self.mtp_window, self.batch
        if self.mtp_cache is None:
            shapes = [tuple(int(d) for d in i.shape) for i in self.mtp.get_inputs()
                      if i.name.startswith("past_kv_")]
            # PagedAttention updates its cache in place. Both captured graph ids bind the
            # same stable buffers so every draft sees all preceding ring updates.
            cache = [torch.zeros(s, dtype=torch.bfloat16, device="cuda") for s in shapes]
            self.mtp_cache = [cache, cache]
            self.mh_hist = torch.zeros((B, W, self.mtp_dim), dtype=torch.bfloat16,
                                       device="cuda")
            self._init_mtp_graphs()
        else:
            for copy in self.mtp_cache:
                for t in copy:
                    t.zero_()
            self.mh_hist.zero_()
        self.mtp_cur = 0
        self.mh_valid = 0

    def _init_mtp_graphs(self):
        """Capture two drafter graph ids over the stable in-place paged caches.

        `main_len` is pinned to `_MH_PAD`, so the only thing that alternates between
        runs is the graph id used by ORT.
        """
        torch, ort = self.torch, self.ort
        B, K = self.batch, self.mtp_block
        self.d_tok = torch.zeros((B, 1), dtype=torch.int64, device="cuda")
        self.d_past = torch.zeros((B,), dtype=torch.int64, device="cuda")
        self.d_out = torch.zeros((B, K + 1), dtype=torch.int64, device="cuda")
        self.d_conf = torch.zeros((B, K), dtype=torch.float32, device="cuda")
        self.d_mh = torch.zeros((B, _MH_PAD, self.mtp_dim), dtype=torch.bfloat16,
                                device="cuda")
        self._mtp_skip = ort.RunOptions()
        self._mtp_skip.add_run_config_entry("gpu_graph_id", "-1")
        self._mtp_ro, self._mtp_io = [], []
        for parity in range(2):
            ro = ort.RunOptions()
            ro.add_run_config_entry("gpu_graph_id", str(parity + 1))
            io = self.mtp.io_binding()
            src = dst = self.mtp_cache[parity]
            for name, t in ([("main_hidden", self.d_mh), ("input_ids", self.d_tok),
                             ("past_lens", self.d_past)]
                            + list(zip(self.mtp_cache_in, src))):
                io.bind_input(name, "cuda", 0, _ELEM[str(t.dtype)], tuple(t.shape),
                              t.data_ptr())
            for name, t in ([("output_ids", self.d_out), ("confidence", self.d_conf)]
                            + list(zip(self.mtp_cache_out, dst))):
                io.bind_output(name, "cuda", 0, _ELEM[str(t.dtype)], tuple(t.shape),
                               t.data_ptr())
            self._mtp_ro.append(ro)
            self._mtp_io.append(io)
        if not self.mtp_graph:
            self._mtp_ro = [self._mtp_skip, self._mtp_skip]
            torch.cuda.synchronize()
            self._bench_mtp()
            return

        # Same two phases as the target: run every id normally so nothing is left to
        # allocate lazily, then drive each id until the EP stops executing and starts
        # replaying.  Both phases write garbage into the ring, which the priming pass
        # after prefill overwrites in full.
        for capture in (False, True):
            for parity in range(2):
                for _ in range(_GRAPH_WARMUP):
                    self.mtp.run_with_iobinding(
                        self._mtp_io[parity],
                        self._mtp_ro[parity] if capture else self._mtp_skip)
        torch.cuda.synchronize()
        self._bench_mtp()

    def _bench_mtp(self):
        if not os.environ.get("DSV4_MTP_BENCH"):
            return
        torch = self.torch
        # Runs back to back, so what it reports is the drafter on its own, without
        # the target's collectives or any rank skew mixed in.
        n = int(os.environ["DSV4_MTP_BENCH"])
        t0 = time.time()
        for i in range(n):
            self.mtp.run_with_iobinding(self._mtp_io[i & 1], self._mtp_ro[i & 1])
        torch.cuda.synchronize()
        if self.rank == 0:
            print(f"[mtp] isolated {1000 * (time.time() - t0) / n:.2f} ms/run",
                  file=sys.stderr, flush=True)

    def _prime_ring(self, past):
        """Fill every ring row before the first draft.

        The drafter writes one ring row per row of `main_hidden`, and `main_len` is
        pinned so the graph can be captured, so priming the window takes
        `window / _MH_PAD` runs rather than one wide one.
        """
        W = self.mtp_window
        for off in range(min(self.mh_valid, W) // _MH_PAD * _MH_PAD, 0, -_MH_PAD):
            self._run_draft(self.mh_hist[:, W - off:W - off + _MH_PAD], past - off)

    def _push_main(self, mh):
        """Append this run's main states to the rolling window, newest last."""
        W = self.mtp_window
        k = min(mh.shape[1], W)
        if k < W:
            self.mh_hist[:, :W - k] = self.mh_hist[:, k:].clone()
        self.mh_hist[:, W - k:] = mh[:, -k:]
        self.mh_valid = min(self.mh_valid + mh.shape[1], W)

    def _run_draft(self, mh, past_lens):
        """Copy the drafter's inputs into the pinned buffers and replay one parity."""
        torch = self.torch
        parity = self.mtp_cur
        self.d_mh.copy_(mh)
        self.d_past.fill_(max(past_lens, 0))
        # The writes above are queued on torch's stream, the replay runs on ORT's.
        torch.cuda.current_stream().synchronize()
        self.mtp.run_with_iobinding(self._mtp_io[parity], self._mtp_ro[parity])
        self.mtp_cur = 1 - parity

    def _draft(self, tok, past):
        """One drafter run: `block` candidate ids continuing `tok` at position `past`.

        The ring is addressed by absolute position, so replaying a pinned `_MH_PAD`
        rows rather than just the ones the last verify committed only rewrites rows
        that are already there with the values they already hold.
        """
        self.d_tok.copy_(tok)
        self._run_draft(self.mh_hist[:, self.mtp_window - _MH_PAD:], past - _MH_PAD)
        self.mh_valid = 0
        return self.d_out, self.d_conf

    def _reslide(self, keep, ran):
        """Re-cut the compressor windows from `ran` positions consumed to `keep` kept.

        Both ends are already on hand: the run's input window is the other ping-pong
        copy, and its own new rows are the tail of its output, so the window that
        would have followed a `keep`-token run is just a different cut of the two.
        """
        if keep == ran:
            return
        torch = self.torch
        post, pre = self.cache[self.cur], self.cache[1 - self.cur]
        for i, slide in enumerate(self._slide):
            if not slide:
                continue
            L = post[i].shape[1]
            if ran > L:
                raise ProtocolError(f"verify of {ran} exceeds the {L}-row window state")
            post[i].copy_(torch.cat([pre[i][:, keep:],
                                     post[i][:, L - ran:L - ran + keep]], dim=1))

    def _generate_spec(self, prompts, max_new_tokens, eos_token_ids, chan):
        """Draft a block with DSpark, verify it with the target, keep the prefix.

        The target's own prediction at the first rejected position is always kept, so
        a step commits at least one token and the emitted text is exactly what plain
        decoding would have produced -- the drafter can only change the speed.

        Nothing rolls the caches back on a partial accept.  The paged kv is addressed
        by absolute position, and the compressor and indexer states are rolling buffers
        of raw projections addressed the same way (deepseek_v4.py:960), so re-running
        from the accepted position overwrites exactly the rejected entries.
        """
        torch, dist = self.torch, self.dist
        if self.batch != 1:
            raise ProtocolError("speculative decode is batch 1 only")
        logits, lens, limits, t_prefill, finite = self._prefill(prompts, max_new_tokens)
        n, K = len(prompts), self.mtp_block
        eos, limit = set(eos_token_ids), limits[0]
        produced, stop = [[]], ["max_new_tokens" if limit == max_new_tokens else "length"]
        t_decode, steps, drafted, accepted = 0.0, 0, 0, 0
        t_draft = t_verify = 0.0
        pos_hit, pos_n = torch.zeros(K, device="cuda"), 0
        debug = int(os.environ.get("DSV4_SPEC_DEBUG", "0"))

        past = lens[0]
        tok = torch.argmax(logits[0]).view(1, 1)
        produced[0].append(int(tok))
        if produced[0][0] in eos:
            stop[0] = "eos"
        vote = torch.zeros(K + 3, dtype=torch.float32, device="cuda")
        gathered = torch.zeros((self.world, K + 3), dtype=torch.float32, device="cuda")
        self._prime_ring(past)

        while len(produced[0]) < limit and stop[0] not in ("eos", "abort"):
            t1 = time.time()
            cand, conf = self._draft(tok, past)
            t_draft += time.time() - t1
            past_t = torch.full((1,), past, dtype=torch.int64, device="cuda")
            if debug and steps < debug:
                # `_reslide(0, ...)` is a full rewind, so the block can be re-run from
                # the same state twice: that isolates whether everything outside the
                # window states really is position-addressed.  Then rewind to one
                # accepted token and run the rest from one position later -- same
                # positions, same prefix, so the predictions have to line up.
                a = self._step(cand, past_t, full=True).clone()
                self._reslide(0, K + 1)
                a2 = self._step(cand, past_t, full=True).clone()
                self._reslide(0, K + 1)
                # Same block, but every position after the first carries a different
                # token.  Position 0 precedes all of them, so a causal path can only
                # be reading ahead if its logits move.
                alt = cand.clone()
                alt[:, 1:] = 12345
                c = self._step(alt, past_t, full=True).clone()
                self._reslide(0, K + 1)
                # The same probe at S=1: whatever this shows is the model's own
                # run-to-run noise and has nothing to do with the block path.
                o1 = self._step(tok, past_t).clone()
                self._reslide(0, 1)
                o2 = self._step(tok, past_t).clone()
                self._reslide(0, 1)
                self._reslide(1, K + 1)
                b = self._step(cand[:, 1:], past_t + 1, full=True).clone()
                rep = [round(v, 3) for v in (a - a2).abs().amax(-1)[0].tolist()]
                d = [round(v, 3) for v in (a[0, 1:] - b[0]).abs().amax(-1).tolist()]
                if self.rank == 0:
                    print(f"[spec {steps}] past={past} s1_repeat={float((o1 - o2).abs().max()):.4g} "
                          f"s1_vs_block={float((o1[0] - a[0, 0]).abs().max()):.4g} "
                          f"future_leak={float((a[0, 0] - c[0, 0]).abs().max()):.4g} "
                          f"repeat={rep} shift={d} a={a[0].argmax(-1).tolist()} "
                          f"b={b[0].argmax(-1).tolist()}", file=sys.stderr, flush=True)
            t2 = time.time()
            # The target sees the accepted token followed by the block, so one run
            # checks every draft and predicts one more beyond the last of them.
            full = self._step(cand, past_t, full=True)
            preds = torch.argmax(full[0], dim=-1)
            t_verify += time.time() - t2
            if debug and self.rank == 0 and steps < debug:
                print(f"[spec {steps}] draft={cand[0].tolist()} target={preds.tolist()}",
                      file=sys.stderr, flush=True)
            match = (cand[0, 1:] == preds[:K])
            j = int((~match).float().argmax()) if not bool(match.all()) else K
            new = torch.cat([cand[0, 1:j + 1], preds[j:j + 1]])
            # Per-slot hit rate, independent of the prefix rule: it separates "the
            # drafter is weak everywhere" from "it is fine but dies with depth".
            pos_hit += match.float()
            pos_n += 1

            # One collective per step, so the ranks cannot pick different lengths.
            vote.zero_()
            vote[0] = j
            vote[1] = 1.0 if (self.rank == 0 and self._abort_pending(chan)) else 0.0
            vote[2:2 + j + 1] = new.float()
            dist.all_gather_into_tensor(gathered, vote)
            g = gathered[0]
            if bool(g[1]):
                stop[0] = "abort"
                break
            j = int(g[0])
            new = g[2:2 + j + 1].to(torch.int64)

            self._reslide(j + 1, K + 1)
            self._push_main(self.mh_last[:, :j + 1])
            past += j + 1
            for t in new.tolist():
                produced[0].append(int(t))
                if int(t) in eos:
                    stop[0] = "eos"
                    break
                if len(produced[0]) >= limit:
                    break
            tok = new[-1].view(1, 1)
            drafted += K
            accepted += j
            steps += 1
            t_decode += time.time() - t1

        out = self._reply(produced, stop, finite, lens, n, t_prefill, t_decode, steps)
        out["accept_rate"] = accepted / drafted if drafted else 0.0
        out["pos_hit"] = [round(v, 3) for v in (pos_hit / max(pos_n, 1)).tolist()]
        if self.rank == 0:
            print(f"[spec] pos_hit={out['pos_hit']} steps={pos_n} "
                  f"draft_ms={1000 * t_draft / max(steps, 1):.2f} "
                  f"verify_ms={1000 * t_verify / max(steps, 1):.2f}",
                  file=sys.stderr, flush=True)
        out["draft_ms"] = 1000 * t_draft / steps if steps else 0.0
        out["verify_ms"] = 1000 * t_verify / steps if steps else 0.0
        out["tokens_per_step"] = (len(produced[0]) - 1) / steps if steps else 0.0
        # A step emits a variable number of tokens, so the generic rate is wrong.
        out["decode_tok_s"] = (len(produced[0]) - 1) / t_decode if t_decode else 0.0
        return out

    # -- decode: agreement through the engine process ----------------------- #

    def _generate_host(self, prompts, max_new_tokens, eos_token_ids, chan):
        """The original path: every rank ships its argmax to the engine and waits.

        Kept behind `DSV4_HOST_TOKEN_SYNC=1` as the reference for A/B and as the
        fallback when the second communicator could not be established.
        """
        torch = self.torch
        logits, lens, limits, t_prefill, finite = self._prefill(prompts, max_new_tokens)

        eos = set(eos_token_ids)
        past_t = torch.tensor(lens, dtype=torch.int64, device="cuda")
        step_ids = torch.zeros((self.batch, 1), dtype=torch.int64, device="cuda")
        produced = [[] for _ in prompts]
        done = [limits[b] == 0 for b in range(len(prompts))]
        stop = [("length" if limits[b] < max_new_tokens else "max_new_tokens")
                for b in range(len(prompts))]
        t_decode, steps = 0.0, 0

        for step in range(max(limits) if limits else 0):
            top = torch.topk(logits, 2, dim=-1)
            local = top.indices[:, 0].tolist()  # the D2H read also synchronizes
            gaps = (top.values[:, 0] - top.values[:, 1]).tolist()
            chan.send({"event": "token", "rank": self.rank, "step": step,
                       "toks": local, "gaps": gaps})
            reply = chan.recv(_FOREVER)
            if reply.get("cmd") == "abort":
                for b in range(len(prompts)):
                    if not done[b]:
                        stop[b] = "abort"
                break
            nxt = reply["toks"]

            for b in range(len(prompts)):
                if done[b]:
                    continue
                produced[b].append(nxt[b])
                if nxt[b] in eos:
                    done[b], stop[b] = True, "eos"
                elif len(produced[b]) >= limits[b]:
                    done[b] = True
            if all(done):
                break

            step_ids[:, 0] = torch.tensor(nxt, dtype=torch.int64, device="cuda")
            t1 = time.time()
            logits = self._step(step_ids, past_t)
            t_decode += time.time() - t1
            steps += 1
            past_t += 1

        return self._reply(produced, stop, finite, lens, len(prompts),
                           t_prefill, t_decode, steps)

    # -- speculative-decoding feasibility ----------------------------------- #

    def spec_probe(self, prompts, ks, reps):
        """Time one decode step over K token positions.

        Verifying K speculative candidates costs exactly one session run over K
        positions, so this measures the cost side of speculative decoding without
        any of the machinery that would produce the candidates.  Speculation pays
        off when the mean number of accepted tokens exceeds the ratio reported
        here of a K-position step to the captured one-position step.

        `past_lens` is deliberately not advanced between runs: every repetition
        rewrites the same cache slots, so the kv length -- and therefore the
        attention cost -- is identical across all K, and the only variable is how
        many positions the step carries.
        """
        torch = self.torch
        _, lens, _, _, _ = self._prefill(prompts, max_new_tokens=1)
        past_t = torch.tensor(lens, dtype=torch.int64, device="cuda")
        out = {}

        def timed(fn):
            for _ in range(3):  # warm up: buffers, and the first run of a shape
                fn()
            torch.cuda.synchronize()
            best = []
            for _ in range(reps):
                torch.cuda.synchronize()
                t0 = time.time()
                fn()
                torch.cuda.synchronize()
                best.append((time.time() - t0) * 1000.0)
            best.sort()
            return best[len(best) // 2]

        if self._graph_io is not None:
            ids1 = torch.zeros((self.batch, 1), dtype=torch.int64, device="cuda")
            out["captured_1"] = timed(lambda: self._step_captured(ids1, past_t))

        for k in ks:
            ids = torch.zeros((self.batch, k), dtype=torch.int64, device="cuda")
            # row=0 would restrict the run to one row; passing row=None with S>1
            # takes the uncaptured path, which is what a verify step would use.
            out[f"uncaptured_{k}"] = timed(lambda ids=ids: self._step(ids, past_t))
        return {"ms": out, "batch": self.batch, "past": int(lens[0])}


def _worker_main(a):
    chan = _Chan(socket.socket(fileno=int(os.environ["DSV4_FD"])))
    try:
        w = _Worker(a.model, a.rank, prefill_chunk=a.prefill_chunk,
                    batch=a.batch, cuda_graph=a.cuda_graph)
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
            break
        if req.get("cmd") == "init_dist":
            try:
                w.init_dist(req["rendezvous"], req["world"])
                chan.send({"event": "dist_ready"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                chan.send({"event": "dist_ready", "error": f"{type(e).__name__}: {e}"})
            continue
        if req.get("cmd") == "host_sync":
            # Some rank could not join the group; nobody may use it.
            w.dist_ready = False
            continue
        if req.get("cmd") == "spec_probe":
            try:
                reply = w.spec_probe(req["prompts"], req["ks"], req["reps"])
            except Exception as e:
                import traceback
                traceback.print_exc()
                reply = {"error": f"{type(e).__name__}: {e}"}
            reply["event"] = "probed"
            chan.send(reply)
            continue
        if req.get("cmd") != "generate":
            chan.send({"event": "generated", "error": "unknown command %r" % req.get("cmd")})
            continue
        try:
            reply = w.generate(req["prompts"], req["max_new_tokens"],
                               req.get("eos_token_ids") or [], chan)
        except Exception as e:
            import traceback
            traceback.print_exc()
            reply = {"error": f"{type(e).__name__}: {e}"}
        reply["event"] = "generated"
        chan.send(reply)

    if w.dist is not None and w.dist.is_initialized():
        w.dist.destroy_process_group()
    return 0


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
                 startup_timeout=1800, quiet=False, prefill_chunk=512, cuda_graph=False,
                 batch=1, devices=None):
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        self.world = world
        self.port = port
        self.log_dir = log_dir
        self.prefill_chunk = prefill_chunk
        self.cuda_graph = cuda_graph
        self.batch = batch
        # Which physical GPUs the ranks land on. Naming them lets a second engine
        # take the other half of the node; give it its own `port` and `log_dir` too.
        if devices is None:
            env_devices = os.environ.get("DSV4_DEVICES")
            devices = ([int(d) for d in env_devices.split(",") if d != ""]
                       if env_devices else list(range(world)))
        if len(devices) < world:
            raise ValueError(f"{len(devices)} devices for a world of {world}")
        self.devices = devices[:world]
        self.procs, self.chans = [], []
        self.max_seq_len = 0
        self.vocab = 0
        self._quiet = quiet
        # Rendezvous for the ranks' own NCCL group.  A file, not a port: rank 0
        # already owns the ORT NCCL rendezvous port and anything else listening
        # near it is a hazard (see `wait_for_listen`).
        self._pg_file = os.path.join(log_dir, f"dsv4_pg_{os.getpid()}_{port}")
        self.device_token_sync = False
        self._start(startup_timeout)

    # -- lifecycle ---------------------------------------------------------- #

    def _log(self, msg):
        if not self._quiet:
            print(msg, flush=True)

    def _spawn(self, rank):
        parent_sock, child_sock = socket.socketpair()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.devices[rank]),
                   LOCAL_RANK=str(rank),
                   LOCAL_WORLD_SIZE=str(self.world), RANK0_IP="127.0.0.1",
                   RANK0_PORT=str(self.port), DSV4_FD=str(child_sock.fileno()))
        log = open(os.path.join(self.log_dir, f"dsv4_rank{rank}.log"), "w")
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker",
             "--rank", str(rank), "--model", self.model_dir,
             "--prefill-chunk", str(self.prefill_chunk),
             "--batch", str(self.batch)]
            + (["--cuda-graph"] if self.cuda_graph else []),
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
        self._init_dist(timeout)

    def _init_dist(self, timeout):
        """Give the ranks a rendezvous so they can agree on tokens themselves.

        Done after every rank has answered `ready`: rank 0 blocks inside session
        creation until the peers join ORT's own communicator, so a collective
        issued any earlier would deadlock against a rank that has not spawned.
        """
        if os.environ.get("DSV4_HOST_TOKEN_SYNC") == "1":
            self._log("DSV4_HOST_TOKEN_SYNC=1: tokens round-trip through the engine")
            return
        try:
            os.unlink(self._pg_file)
        except OSError:
            pass
        for chan in self.chans:
            chan.send({"cmd": "init_dist", "rendezvous": "file://" + self._pg_file,
                       "world": self.world})
        errors = []
        for rank, chan in enumerate(self.chans):
            msg = self._recv(rank, chan, timeout)
            if msg.get("event") != "dist_ready" or msg.get("error"):
                errors.append(f"rank {rank}: {msg.get('error')}")
        if errors:
            # Half the ranks in a group they cannot all use is worse than none.
            for chan in self.chans:
                chan.send({"cmd": "host_sync"})
            self._log("note: on-device token agreement unavailable (%s); falling "
                      "back to the host round-trip" % "; ".join(errors))
            return
        self.device_token_sync = True

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
        try:
            os.unlink(self._pg_file)
        except OSError:
            pass

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
        """Greedily continue one prompt.  Returns rank 0's reply dict."""
        out = self.generate_batch([prompt_ids], max_new_tokens, eos_token_ids, timeout)
        out["tokens"] = out["tokens"][0]
        out["stop_reason"] = out["stop_reason"][0]
        out["prompt_len"] = out["prompt_lens"][0]
        return out

    def spec_probe(self, prompts, ks=(1, 2, 4, 6, 8), reps=20, timeout=1800):
        """Time a decode step over K positions, for each K in `ks`.

        This answers the one question that decides whether speculative decoding is
        worth building here: a verify step carries K candidate positions instead of
        one, and speculation only pays if the mean accepted length beats the cost
        ratio of a K-position step to today's captured one-position step.

        Returns rank 0's timings in milliseconds.
        """
        if not self.procs:
            raise ProtocolError("engine is closed")
        prompts = [[int(t) for t in p] for p in prompts]
        for chan in self.chans:
            chan.send({"cmd": "spec_probe", "prompts": prompts,
                       "ks": list(ks), "reps": reps})
        msgs = [self._recv(rank, chan, timeout) for rank, chan in enumerate(self.chans)]
        for m in msgs:
            if m.get("error"):
                raise ProtocolError(m["error"])
        return msgs[0]

    def generate_batch(self, prompts, max_new_tokens=64, eos_token_ids=(1,), timeout=600):
        """Greedily continue a batch of prompts.  Returns rank 0's reply dict.

        The ranks agree on every token they emit, because their logits agree to
        the last bit almost always but not quite always, and one divergence would
        leave the group permanently out of step - the losing rank would feed a
        different token into the next `AllReduce`.  Normally that agreement is an
        all-gather on the ranks' own communicator and this loop only drains
        liveness pings; under `DSV4_HOST_TOKEN_SYNC=1` the ranks report their
        argmax here every step and are told rank 0's.  Either way the count of
        disagreements is returned as `rank_disagreements`.
        """
        if not self.procs:
            raise ProtocolError("engine is closed")
        if len(prompts) > self.batch:
            raise ValueError(f"{len(prompts)} prompts for an engine built with "
                             f"batch={self.batch}")
        prompts = [[int(t) for t in p] for p in prompts]
        for p in prompts:
            if self.max_seq_len and len(p) >= self.max_seq_len:
                raise ValueError(f"prompt of {len(p)} tokens exceeds the export's "
                                 f"max_seq_len of {self.max_seq_len}")
        for chan in self.chans:
            chan.send({"cmd": "generate", "prompts": prompts,
                       "max_new_tokens": max_new_tokens,
                       "eos_token_ids": list(eos_token_ids)})

        disagreements = []
        while True:
            msgs = [self._recv(rank, chan, timeout) for rank, chan in enumerate(self.chans)]
            kinds = {m.get("event") for m in msgs}
            if kinds == {"progress"}:
                continue
            if kinds == {"token"}:
                per_rank = [m["toks"] for m in msgs]
                for row, votes in enumerate(zip(*per_rank)):
                    if len(set(votes)) > 1:
                        disagreements.append({"step": msgs[0]["step"], "row": row,
                                              "tokens": list(votes),
                                              "gap": min(m["gaps"][row] for m in msgs)})
                for chan in self.chans:
                    chan.send({"toks": per_rank[0]})
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
        # With on-device agreement the votes never reach this process, so rank 0
        # reconstructs them from the gathered history and reports them at the end.
        worker_dis = out.pop("disagreements", None)
        out["rank_disagreements"] = disagreements if worker_dis is None else worker_dis
        if out["rank_disagreements"]:
            d = out["rank_disagreements"]
            self._log(f"note: {len(d)} step(s) where the ranks' argmax "
                      f"differed (smallest top-2 gap "
                      f"{min(x['gap'] for x in d):.2e}); rank 0 won")
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
    ap.add_argument("--prefill-chunk", type=int, default=512,
                    help="paged mode only: prompt tokens per forward pass.  The window"
                         " ring is sized for this, so it bounds the ring's memory")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture the decode step as a CUDA graph and replay it")
    ap.add_argument("--batch", type=int, default=1,
                    help="sequences decoded in lockstep; fixed for the process because"
                         " a captured graph has fixed shapes")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.worker:
        ap.error("dsv4_engine.py is a library; use run.py or import DSV4Engine")
    return _worker_main(a)


if __name__ == "__main__":
    raise SystemExit(main())
