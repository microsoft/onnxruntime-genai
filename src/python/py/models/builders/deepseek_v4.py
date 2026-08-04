# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""ONNX graph builder for DeepSeek-V4-Flash.

The architecture is composed node-by-node (no tracing/dynamo) so that fused
contrib ops can be emitted directly instead of being pattern-matched out of a
decomposed subgraph later.

Novel pieces relative to a standard decoder, and how each is emitted:

* Hyper-Connections (``hc_mult`` parallel residual streams mixed by a Sinkhorn
  normalised routing matrix) -- plain ONNX ops, unrolled.
* KV compressor (gated pooling of ``ratio`` consecutive tokens into one latent
  row, with overlapping windows when ``ratio == 4``) -- plain ONNX ops.
* Sliding-window MQA over a latent row with a learned per-head softmax sink and
  *inverse* RoPE applied to the attention output.
* MoE with ``sqrtsoftplus`` scoring, ``noaux_tc`` selection bias and hash routing
  on the first ``num_hash_layers`` layers. Routing is computed in the graph and
  handed to ``com.microsoft.QMoE`` as log-domain router logits, so the stock
  kernel's softmax/top-k reproduces the reference weights exactly without needing
  a new scoring mode in the kernel.

Caches are fixed capacity: the sliding-window KV cache is a ring of exactly
``sliding_window`` rows and the compressed cache is indexed directly by slot, so
every cache tensor has a static shape.
"""

from __future__ import annotations

import math
import os

import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import to_torch_dtype

from .base import Model
from .safetensors_store import ExternalDataWriter

NEG_INF = -1e30
FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def mxfp4_quantize(w: torch.Tensor):
    """[E, N, K] float -> (blocks [E, N, K/32, 16] uint8, ue8m0 scales [E, N, K/32] uint8).

    Each byte holds two adjacent K codes, even K in the low nibble.
    """
    e, n, k = w.shape
    wb = w.float().reshape(e, n, k // 32, 32)
    amax = wb.abs().amax(-1, keepdim=True)
    exp = torch.ceil(torch.log2((amax / 6.0).clamp_min(1e-30)))
    exp = torch.where(amax > 0, exp, torch.zeros_like(exp)).clamp(-127, 127)
    q = wb / torch.exp2(exp)
    code = (q.abs().unsqueeze(-1) - FP4_LUT.to(w.device)).abs().argmin(-1).to(torch.uint8)
    code = code | ((q < 0).to(torch.uint8) << 3)
    blocks = (code[..., 0::2] | (code[..., 1::2] << 4)).reshape(e, n, k // 32, 16)
    return blocks, (exp.squeeze(-1) + 127).to(torch.uint8)


def mxfp4_dequantize(blocks, scales):
    """Inverse of :func:`mxfp4_quantize` -> [E, N, K] float."""
    e, n, kb, _ = blocks.shape
    code = torch.stack([blocks & 0x0F, blocks >> 4], dim=-1).reshape(e, n, kb, 32)
    mag = FP4_LUT.to(blocks.device)[(code & 0x07).long()]
    val = torch.where((code & 0x08) > 0, -mag, mag)
    return (val * torch.exp2(scales.float() - 127).unsqueeze(-1)).reshape(e, n, kb * 32)


def pack_for_qmoe(blocks):
    """[E, N, K/32, 16] -> QMoE's [E, K, N/2], even N in the low nibble."""
    even_n, odd_n = blocks[:, 0::2], blocks[:, 1::2]
    packed = torch.stack((((odd_n & 0x0F) << 4) | (even_n & 0x0F),
                          ((odd_n >> 4) << 4) | (even_n >> 4)), dim=-1)
    return packed.permute(0, 2, 3, 4, 1).reshape(
        blocks.shape[0], blocks.shape[2] * 32, blocks.shape[1] // 2).contiguous()
FP8_MAX = 448.0
FP4_MAX = 6.0
FP8_BLOCK = 128
INT64_MAX = 9223372036854775807

B = "batch_size"
S = "sequence_length"


class DeepSeekV4FlashModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # `intermediate_size` is absent from the checkpoint config; the experts use
        # `moe_intermediate_size` and there is no dense FFN.
        if not hasattr(config, "intermediate_size"):
            config.intermediate_size = config.moe_intermediate_size
        # Smoke-testing the real checkpoint is much cheaper with a prefix of the
        # layers; the graph is otherwise identical.
        if extra_options.get("dsv4_num_layers"):
            config.num_hidden_layers = int(extra_options["dsv4_num_layers"])

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        c = config
        self.dim = c.hidden_size
        self.n_heads = c.num_attention_heads
        self.head_dim = c.head_dim
        self.rope_head_dim = c.qk_rope_head_dim
        self.nope_dim = self.head_dim - self.rope_head_dim
        self.q_lora_rank = c.q_lora_rank
        self.o_groups = c.o_groups
        self.o_lora_rank = c.o_lora_rank
        self.window = c.sliding_window
        self.eps = c.rms_norm_eps
        self.softmax_scale = self.head_dim ** -0.5

        self.hc = c.hc_mult
        self.hc_mix_dim = (2 + self.hc) * self.hc
        self.hc_iters = c.hc_sinkhorn_iters
        self.hc_eps = c.hc_eps

        self.n_experts = c.n_routed_experts
        self.topk = c.num_experts_per_tok
        self.moe_inter = c.moe_intermediate_size
        self.route_scale = c.routed_scaling_factor
        self.swiglu_limit = c.swiglu_limit
        self.n_hash_layers = c.num_hash_layers

        self.compress_ratios = list(c.compress_ratios)[: self.num_layers]
        self.rope_theta = c.rope_theta
        self.compress_rope_theta = c.compress_rope_theta
        rs = c.rope_scaling
        self.rope_factor = rs["factor"]
        self.beta_fast = rs["beta_fast"]
        self.beta_slow = rs["beta_slow"]

        self.max_seq_len = int(extra_options.get("dsv4_max_seq_len", 4096))
        self.moe_impl = extra_options.get("dsv4_moe_impl", "qmoe")

        # Opt-in: replace the dense window-plus-compressed attention with one
        # `com.microsoft.PagedAttention` node per layer (kv_cache_layout='LATENT').  One flat
        # paged cache per layer then holds both streams, updated in place, and the O(context)
        # score tensor disappears.
        self.paged = extra_options.get("dsv4_paged_attention", "0") not in ("0", 0, False)
        self.block_size = int(extra_options.get("dsv4_block_size", 64))

        # Opt-in: the Lightning Indexer. Without it the paged path lets every query see every
        # valid compressed row, so `kv_indices` is window + comp_capacity wide -- 128 + 262146
        # at 1M context, which throws away the flat-latency property the paged op exists for.
        # With it the width is window + index_topk. Only ratio-4 layers have one; ratio-128
        # layers attend to all their (far fewer) rows by design.
        self.indexer = extra_options.get("dsv4_indexer", "0") not in ("0", 0, False)
        self.index_n_heads = getattr(c, "index_n_heads", 64)
        self.index_head_dim = getattr(c, "index_head_dim", 128)
        self.index_topk = getattr(c, "index_topk", 512)

        # A block's hyper-connection post mix is always followed by the next block's pre mix,
        # and the post output is also the next residual, so one operator covers both plus the
        # layer norm that reads its result: ~44 nodes per site down to 1, 85 sites.
        self.hc_fused = extra_options.get("dsv4_hc_fused", "1") not in ("0", 0, False)

        # The compressor is ~100 nodes of window arithmetic, a masked softmax and two simulated
        # quantisation grids, all of it producing two rows per sequence at decode: 62 sites,
        # ~5,900 nodes, and almost no arithmetic. One operator covers everything after the two
        # raw projections.
        self.comp_fused = extra_options.get("dsv4_comp_fused", "1") not in ("0", 0, False)

        # Likewise for what is left of the indexer once its compressor is fused: a rotation, a
        # cache refresh, a scoring einsum over every cached row and a top-k, ~60 nodes on each
        # of the 21 ratio-4 layers. One operator covers everything after its two projections.
        self.index_fused = extra_options.get("dsv4_index_fused", "1") not in ("0", 0, False)

        # Hybrid parallelism, matching what vLLM runs for this checkpoint:
        # tensor-parallel for attention and the shared expert, expert-parallel
        # for the routed experts.  One graph is emitted per rank; the ranks are
        # stitched together at run time by two `com.microsoft.AllReduce` nodes
        # per layer (after the output projection, and after the FFN).
        self.world = int(extra_options.get("dsv4_tp_world", 1))
        self.rank = int(extra_options.get("dsv4_tp_rank", 0))
        if self.world > 1:
            for what, n in (("o_groups", self.o_groups), ("heads", self.n_heads),
                            ("experts", self.n_experts), ("moe_inter", self.moe_inter)):
                if n % self.world:
                    raise ValueError(f"dsv4_tp_world={self.world} does not divide {what}={n}")
        self.n_heads_local = self.n_heads // self.world
        self.o_groups_local = self.o_groups // self.world
        self.n_experts_local = self.n_experts // self.world
        self.moe_inter_local = self.moe_inter // self.world
        self.expert_lo = self.rank * self.n_experts_local

        self.sd: dict[str, torch.Tensor] = {}

        # Streaming export: every initializer is appended to the external-data
        # blob as soon as it is built and the checkpoint is read one tensor at a
        # time, so a 150 GiB model can be emitted layer by layer without ever
        # being resident.  The blob is written into the cache dir and moved next
        # to the model by `save_model`; `location` is a bare basename so the
        # references survive the move.
        self.stream = extra_options.get("dsv4_stream_weights", "1") not in ("0", 0, False)
        self.writer = (ExternalDataWriter(os.path.join(cache_dir, self.filename + ".data"))
                       if self.stream else None)

        ckpt = extra_options.get("dsv4_checkpoint")
        if ckpt:
            # Serves the builder's key names over the raw checkpoint and repacks
            # the fp4 experts, reading only this rank's slice of them.
            from .deepseek_v4_weights import DSV4Weights

            self.sd = DSV4Weights(
                ckpt,
                expert_range=(self.expert_lo, self.expert_lo + self.n_experts_local),
                device=extra_options.get("dsv4_repack_device", "cpu"),
            )

        self._make_dsv4_io()

    def shard(self, t, axis):
        """Take this rank's contiguous slice of ``t`` along ``axis``."""
        if self.world == 1:
            return t
        n = t.shape[axis] // self.world
        return t.narrow(axis, self.rank * n, n).contiguous()

    def eshard(self, t, axis):
        """Like :meth:`shard`, but a no-op when the store already narrowed the
        experts to this rank (the real checkpoint reads only its own slice)."""
        return t if getattr(self.sd, "pre_sharded", False) else self.shard(t, axis)

    def all_reduce(self, name, x, dtype, shape):
        if self.world == 1:
            return x
        self.make_node("AllReduce", inputs=[x], outputs=[name], name=name, domain="com.microsoft")
        self.make_value(name, dtype, shape=shape)
        return name

    # ------------------------------------------------------------------ #
    # graph I/O
    # ------------------------------------------------------------------ #

    def coff(self, ratio):
        return 2 if ratio == 4 else 1

    def state_len(self, ratio):
        """Rolling raw-projection buffer; overlapping windows need two windows."""
        return self.coff(ratio) * ratio

    def comp_capacity(self, ratio):
        return self.max_seq_len // ratio + 2

    def has_indexer(self, ratio):
        return self.indexer and ratio == 4

    def index_width(self, ratio):
        """How many compressed rows a query may look at in a ratio group."""
        C = self.comp_capacity(ratio)
        return min(self.index_topk, C) if self.has_indexer(ratio) else C

    def group_blocks(self, ratio):
        """Block-table width for a cache group: the window region uses absolute positions, so it
        spans max_seq_len, and the compressor's rows sit directly above it."""
        span = self.max_seq_len + (self.comp_capacity(ratio) if ratio else 0)
        return (span + self.block_size - 1) // self.block_size

    def _make_dsv4_io(self):
        # The serving layer has to size and recycle the block pools, and none of this
        # geometry is recoverable from the graph's shapes alone.
        self.model.metadata_props.update({
            "dsv4_max_seq_len": str(self.max_seq_len),
            "dsv4_window": str(self.window),
            "dsv4_compress_ratios": ",".join(str(r) for r in self.compress_ratios),
            "dsv4_paged": "1" if self.paged else "0",
            "dsv4_block_size": str(self.block_size) if self.paged else "0",
            "dsv4_indexer": "1" if self.indexer else "0",
            "dsv4_index_topk": str(self.index_topk),
        })

        self.input_names = {"input_ids": "input_ids", "past_lens": "past_lens"}
        self.input_types = {"input_ids": ir.DataType.INT64, "past_lens": ir.DataType.INT64}
        self.input_shapes = {"input_ids": [B, S], "past_lens": [B]}
        self.output_names = {"logits": "logits"}
        self.output_types = {"logits": ir.DataType.FLOAT}
        self.output_shapes = {"logits": [B, S, self.vocab_size]}

        if self.paged:
            # One block table per ratio group: layers do not share a logical position space, and
            # sizing a single pool for the ratio-4 layers would cost more than the dense export.
            for r in sorted(set(self.compress_ratios)):
                n = f"block_table_{r}"
                self.input_names[n] = n
                self.input_types[n] = ir.DataType.INT32
                self.input_shapes[n] = [B, self.group_blocks(r)]

        self.cache_spec = []  # (layer_id, key)
        for i, r in enumerate(self.compress_ratios):
            if self.paged:
                entries = [("kv", self.io_dtype,
                            [f"num_blocks_{r}", self.block_size, 1, self.head_dim])]
            else:
                entries = [("kv", self.io_dtype, [B, self.window, self.head_dim])]
            if r:
                st = [B, self.state_len(r), self.coff(r) * self.head_dim]
                if not self.paged:
                    # Paged mode keeps the compressed rows in the same flat cache as the window.
                    entries.append(("comp", self.io_dtype, [B, self.comp_capacity(r), self.head_dim]))
                entries += [
                    ("cstate_kv", ir.DataType.FLOAT, st),
                    ("cstate_score", ir.DataType.FLOAT, st),
                ]
                if self.has_indexer(r):
                    # The indexer's compressed cache is read densely by the scoring einsum, so
                    # unlike the attention cache it gains nothing from being paged.
                    ist = [B, self.state_len(r), self.coff(r) * self.index_head_dim]
                    entries += [
                        ("icache", self.io_dtype,
                         [B, self.comp_capacity(r), self.index_head_dim]),
                        ("icstate_kv", ir.DataType.FLOAT, ist),
                        ("icstate_score", ir.DataType.FLOAT, ist),
                    ]
            for k, dt, shape in entries:
                self.cache_spec.append((i, k))
                self.input_names[f"past_{k}_{i}"] = f"past_{k}_{i}"
                self.input_types[f"past_{k}_{i}"] = dt
                self.input_shapes[f"past_{k}_{i}"] = shape
                self.output_names[f"present_{k}_{i}"] = f"present_{k}_{i}"
                self.output_types[f"present_{k}_{i}"] = dt
                self.output_shapes[f"present_{k}_{i}"] = shape

    def make_key_value_cache_shape(self, layer_id, shape):
        return shape

    # ------------------------------------------------------------------ #
    # emission helpers
    # ------------------------------------------------------------------ #

    def op(self, op_type, inputs, name, dtype=None, shape=None, domain="", **attrs):
        out = f"{name}/output_0"
        self.make_node(op_type, inputs=inputs, outputs=[out], name=name, domain=domain, **attrs)
        self.make_value(out, dtype, shape=shape)
        return out

    def const(self, dtype: str, value):
        return f"/model/constants/{dtype}/{value!r}"

    def init(self, tensor, name, to=None):
        if name not in self.values:
            if isinstance(tensor, ir.ExternalTensor):
                value = self.make_value(name, tensor.dtype, tensor.shape)
                value.const_value = tensor
                self.model.graph.register_initializer(value)
            elif self.writer is not None and isinstance(tensor, torch.Tensor):
                tensor = tensor.detach()
                if to is not None:
                    tensor = tensor.to(to_torch_dtype(to))
                self.init(self.writer.add(tensor.contiguous(), name), name)
            else:
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.detach().contiguous()
                self.make_initializer(tensor, name, to=to)
        return name

    def init_w(self, key, to=None):
        """Register checkpoint tensor ``key`` under its own name."""
        return self.init(self.sd[key], key, to=to)

    def proj(self, name, x, key, shape, shard_axis=None):
        """``x @ W.T`` for a checkpoint weight stored as ``[N, K]``.

        The checkpoint keeps these projections in block-scaled fp8, which
        ``MatMulBlockQuantizedFp8Weight`` consumes in that exact layout -- no
        transpose and no dequantization.  Unquantized checkpoints (the parity
        test model) fall back to a transposed initializer and a plain MatMul.
        """
        io = self.io_dtype
        q = self.sd.qweight(key) if hasattr(self.sd, "qweight") else None
        if q is None:
            w = self.sd[key]
            if shard_axis is not None:
                w = self.shard(w, shard_axis)
            return self.op("MatMul", [x, self.init(w.T, f"{key}/T", to=io)], name, io, shape)
        w, scale = q
        if shard_axis is not None:
            w = self.shard(w, shard_axis)
            # Row scales follow N; along K they follow the 128-wide blocks.
            scale = self.shard(scale, shard_axis)
        return self.op("MatMulBlockQuantizedFp8Weight",
                       [x, self.init(w, f"{key}/q"), self.init(scale, f"{key}/s")],
                       name, io, shape, domain="com.microsoft", block_size=FP8_BLOCK)

    def cast(self, name, x, dtype, shape=None):
        return self.op("Cast", [x], name, dtype, shape, to=dtype)

    def reshape(self, name, x, dims, dtype, shape):
        """dims is baked into a constant (0 = copy input dim, -1 = infer).

        Shape operands must be Constant nodes rather than initializers: everything
        is written to external data, and ORT's shape inference refuses to read
        external tensors.
        """
        return self.op("Reshape", [x, self.const("INT64", list(dims))], name, dtype, shape)

    def dyn_reshape(self, name, x, prefix_value, tail_dims, dtype, shape):
        """Reshape to ``concat(prefix_value, tail_dims)`` for [batch, seq, ...] targets."""
        t = self.const("INT64", list(tail_dims))
        full = self.op("Concat", [prefix_value, t], f"{name}/cat", ir.DataType.INT64,
                       [2 + len(tail_dims)], axis=0)
        return self.op("Reshape", [x, full], name, dtype, shape)

    def slice1(self, name, x, start, end, dtype, shape, axis=-1):
        return self.op("Slice", [x, self.const("INT64", [start]), self.const("INT64", [end]),
                                 self.const("INT64", [axis])], name, dtype, shape)

    def batched_gather(self, name, data, indices, rows, dtype, shape):
        """Gather along axis 1 with a different index vector per batch row."""
        idx = self.unsq(f"{name}/idx", indices, [2], ir.DataType.INT64, [B, rows, 1])
        return self.op("GatherND", [data, idx], name, dtype, shape, batch_dims=1)

    def floordiv(self, name, num, den, shape):
        """Floor division for possibly negative int64 numerators (ONNX Div truncates)."""
        n = self.op("Cast", [num], f"{name}/nd", ir.DataType.DOUBLE, shape, to=ir.DataType.DOUBLE)
        q = self.op("Div", [n, self.const("DOUBLE", float(den))], f"{name}/div",
                    ir.DataType.DOUBLE, shape)
        f = self.op("Floor", [q], f"{name}/floor", ir.DataType.DOUBLE, shape)
        return self.op("Cast", [f], name, ir.DataType.INT64, shape, to=ir.DataType.INT64)

    def scalar_init(self, value, name, dtype=torch.float32):
        return self.init(torch.tensor(value, dtype=dtype), name)

    def unsq(self, name, x, axes, dtype, shape):
        return self.op("Unsqueeze", [x, self.const("INT64", axes)], name, dtype, shape)

    # ------------------------------------------------------------------ #
    # rotary
    # ------------------------------------------------------------------ #

    def _rope_tables(self, base, original_seq_len):
        """cos/sin already expanded to the full rotary width.

        Storing ``repeat_interleave(cos, 2)`` lets the interleaved-pair rotation be
        written as ``x * cos + (x @ R) * sin`` with a constant signed permutation R,
        which avoids reshaping a dynamically shaped tensor.
        """
        dim = self.rope_head_dim
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if original_seq_len > 0:
            def corr_dim(nrot):
                return dim * math.log(original_seq_len / (nrot * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(corr_dim(self.beta_fast)), 0)
            high = min(math.ceil(corr_dim(self.beta_slow)), dim - 1)
            if low == high:
                high += 0.001
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
            smooth = 1 - ramp
            freqs = freqs / self.rope_factor * (1 - smooth) + freqs * smooth
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        ang = torch.outer(t, freqs)
        return ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)

    def make_rope_tables(self):
        for idx, (base, orig) in enumerate(
            [(self.rope_theta, 0), (self.compress_rope_theta, self.original_context_length)]
        ):
            cos, sin = self._rope_tables(base, orig)
            self.init(cos, f"rope/cos_{idx}")
            self.init(sin, f"rope/sin_{idx}")

        d = self.rope_head_dim
        r = torch.zeros(d, d, dtype=torch.float32)
        for k in range(d // 2):
            r[2 * k + 1, 2 * k] = -1.0
            r[2 * k, 2 * k + 1] = 1.0
        self.init(r, "rope/R")
        self.init(-r, "rope/R_inv")

    def make_rope(self, name, x, cos, sin, shape, inverse=False):
        """x is FLOAT with last dim == rope_head_dim; cos/sin broadcast against it."""
        rot = self.op("MatMul", [x, "rope/R_inv" if inverse else "rope/R"],
                      f"{name}/rot", ir.DataType.FLOAT, shape)
        a = self.op("Mul", [x, cos], f"{name}/a", ir.DataType.FLOAT, shape)
        b = self.op("Mul", [rot, sin], f"{name}/b", ir.DataType.FLOAT, shape)
        return self.op("Add", [a, b], name, ir.DataType.FLOAT, shape)

    # ------------------------------------------------------------------ #
    # norms
    # ------------------------------------------------------------------ #

    def make_rmsnorm(self, name, x, weight, shape):
        """Reference computes in fp32 with an fp32 weight; returns FLOAT."""
        xf = self.cast(f"{name}/castf", x, ir.DataType.FLOAT, shape)
        out = f"{name}/output_0"
        self.make_node("SimplifiedLayerNormalization", inputs=[xf, weight], outputs=[out],
                       name=name, axis=-1, epsilon=self.eps, stash_type=1)
        self.make_value(out, ir.DataType.FLOAT, shape=shape)
        return out

    def _rsqrt_mean_sq(self, name, xf, shape):
        red = shape[:-1] + [1]
        sq = self.op("Mul", [xf, xf], f"{name}/sq", ir.DataType.FLOAT, shape)
        mean = self.op("ReduceMean", [sq, self.const("INT64", [-1])], f"{name}/mean",
                       ir.DataType.FLOAT, red, keepdims=1)
        add = self.op("Add", [mean, self.const("FLOAT", self.eps)], f"{name}/eps",
                      ir.DataType.FLOAT, red)
        s = self.op("Sqrt", [add], f"{name}/sqrt", ir.DataType.FLOAT, red)
        return self.op("Reciprocal", [s], name, ir.DataType.FLOAT, red)

    def make_weightless_rmsnorm(self, name, x, shape, dtype):
        """``x * rsqrt(mean(x^2) + eps)`` with the scale rounded to ``dtype`` first."""
        red = shape[:-1] + [1]
        xf = self.cast(f"{name}/castf", x, ir.DataType.FLOAT, shape)
        rs = self._rsqrt_mean_sq(f"{name}/rs", xf, shape)
        if dtype != ir.DataType.FLOAT:
            rs = self.cast(f"{name}/castb", rs, dtype, red)
        return self.op("Mul", [x, rs], name, dtype, shape)

    # ------------------------------------------------------------------ #
    # simulated FP8 activation quantisation (QAT numerics that must be kept)
    # ------------------------------------------------------------------ #

    def make_act_quant(self, name, x, shape, block=64):
        """Per-block FP8-E4M3 round trip with a UE8M0 (power-of-two) scale."""
        blocked = shape[:-1] + [shape[-1] // block, block]
        red = blocked[:-1] + [1]
        xf = self.reshape(f"{name}/rs", x, [0, 0, -1, block], ir.DataType.FLOAT, blocked)
        a = self.op("Abs", [xf], f"{name}/abs", ir.DataType.FLOAT, blocked)
        amax = self.op("ReduceMax", [a, self.const("INT64", [-1])], f"{name}/amax",
                       ir.DataType.FLOAT, red, keepdims=1)
        r = self.op("Div", [amax, self.const("FLOAT", FP8_MAX)], f"{name}/norm",
                    ir.DataType.FLOAT, red)
        r = self.op("Clip", [r, self.const("FLOAT", 1e-30)], f"{name}/clipmin",
                    ir.DataType.FLOAT, red)
        lg = self.op("Log", [r], f"{name}/log", ir.DataType.FLOAT, red)
        lg = self.op("Div", [lg, self.const("FLOAT", math.log(2.0))], f"{name}/log2",
                     ir.DataType.FLOAT, red)
        e = self.op("Ceil", [lg], f"{name}/ceil", ir.DataType.FLOAT, red)
        scale = self.op("Pow", [self.const("FLOAT", 2.0), e], f"{name}/scale",
                        ir.DataType.FLOAT, red)
        pos = self.op("Greater", [amax, self.const("FLOAT", 0.0)], f"{name}/pos",
                      ir.DataType.BOOL, red)
        scale = self.op("Where", [pos, scale, self.const("FLOAT", 1.0)], f"{name}/scale1",
                        ir.DataType.FLOAT, red)
        q = self.op("Div", [xf, scale], f"{name}/div", ir.DataType.FLOAT, blocked)
        q = self.op("Clip", [q, self.const("FLOAT", -FP8_MAX), self.const("FLOAT", FP8_MAX)],
                    f"{name}/clip", ir.DataType.FLOAT, blocked)
        q = self.op("Cast", [q], f"{name}/tofp8", ir.DataType.FLOAT8E4M3FN, blocked,
                    to=ir.DataType.FLOAT8E4M3FN)
        q = self.op("Cast", [q], f"{name}/back", ir.DataType.FLOAT, blocked, to=ir.DataType.FLOAT)
        q = self.op("Mul", [q, scale], f"{name}/mul", ir.DataType.FLOAT, blocked)
        return self.reshape(name, q, [0, 0, -1], ir.DataType.FLOAT, shape)

    def make_rotate_fp4(self, name, x, shape, block=32):
        """Hadamard rotation followed by a per-block FP4-E2M1 round trip.

        The indexer applies this to *both* operands of its scoring einsum. The rotation on its
        own is orthogonal and would cancel out; it is the FP4 rounding after it that moves the
        ranking, so the two only make sense together. Dropping them fails parity against the
        reference, which is why this is not optional.

        ``x`` is FLOAT of any rank; only the last dimension (a power of two) is transformed.
        """
        F, BOOL = ir.DataType.FLOAT, ir.DataType.BOOL
        n = shape[-1]
        blocked = shape[:-1] + [n // block, block]
        red = blocked[:-1] + [1]

        h = torch.ones(1, 1, dtype=torch.float32)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        x = self.op("MatMul", [x, self.init(h * (n ** -0.5), f"hadamard_{n}")],
                    f"{name}/had", F, shape)
        # The reference rounds the rotated tensor to the activation dtype before quantising.
        x = self.cast(f"{name}/hadb", x, self.io_dtype, shape)
        x = self.cast(f"{name}/hadf", x, F, shape)

        xf = self.reshape(f"{name}/rs", x, [0] * (len(shape) - 1) + [-1, block], F, blocked)
        amax = self.op("ReduceMax", [self.op("Abs", [xf], f"{name}/abs", F, blocked),
                                     self.const("INT64", [-1])],
                       f"{name}/amax", F, red, keepdims=1)
        r = self.op("Clip", [self.op("Div", [amax, self.const("FLOAT", FP4_MAX)],
                                     f"{name}/norm", F, red),
                             self.const("FLOAT", 1e-38)], f"{name}/clipmin", F, red)
        lg = self.op("Div", [self.op("Log", [r], f"{name}/log", F, red),
                             self.const("FLOAT", math.log(2.0))], f"{name}/log2", F, red)
        scale = self.op("Pow", [self.const("FLOAT", 2.0),
                                self.op("Ceil", [lg], f"{name}/ceil", F, red)],
                        f"{name}/scale", F, red)
        scale = self.op("Where", [self.op("Greater", [amax, self.const("FLOAT", 0.0)],
                                          f"{name}/pos", BOOL, red),
                                  scale, self.const("FLOAT", 1.0)], f"{name}/scale1", F, red)
        v = self.op("Clip", [self.op("Div", [xf, scale], f"{name}/div", F, blocked),
                             self.const("FLOAT", -FP4_MAX), self.const("FLOAT", FP4_MAX)],
                    f"{name}/clip", F, blocked)

        # Round onto the E2M1 grid {0,.5,1,1.5,2,3,4,6}: the step is 0.5 below 2, 1 below 4 and
        # 2 above, and ties go toward zero -- which is `ceil(t - 0.5)`.
        u = self.op("Abs", [v], f"{name}/u", F, blocked)
        step = self.op("Where", [self.op("Less", [u, self.const("FLOAT", 2.0)],
                                         f"{name}/lt2", BOOL, blocked),
                                 self.const("FLOAT", 0.5),
                                 self.op("Where", [self.op("Less", [u, self.const("FLOAT", 4.0)],
                                                           f"{name}/lt4", BOOL, blocked),
                                                   self.const("FLOAT", 1.0),
                                                   self.const("FLOAT", 2.0)],
                                         f"{name}/step2", F, blocked)],
                       f"{name}/step", F, blocked)
        t = self.op("Sub", [self.op("Div", [u, step], f"{name}/t", F, blocked),
                            self.const("FLOAT", 0.5)], f"{name}/th", F, blocked)
        q = self.op("Mul", [step, self.op("Ceil", [t], f"{name}/qceil", F, blocked)],
                    f"{name}/q", F, blocked)
        q = self.op("Mul", [self.op("Sign", [v], f"{name}/sign", F, blocked), q],
                    f"{name}/qs", F, blocked)
        q = self.op("Mul", [q, scale], f"{name}/mul", F, blocked)
        return self.reshape(name, q, [0] * (len(shape) - 1) + [-1], F, shape)

    # ------------------------------------------------------------------ #
    # hyper-connections
    # ------------------------------------------------------------------ #

    def _hc_mixes(self, name, xflat, prefix, flat_shape):
        mix_shape = flat_shape[:-1] + [self.hc_mix_dim]
        fn = self.init(self.sd[f"{prefix}_fn"].T, f"{prefix}_fn/T", to=ir.DataType.FLOAT)
        mixes = self.op("MatMul", [xflat, fn], f"{name}/lin", ir.DataType.FLOAT, mix_shape)
        rs = self._rsqrt_mean_sq(f"{name}/rs", xflat, flat_shape)
        return self.op("Mul", [mixes, rs], name, ir.DataType.FLOAT, mix_shape)

    def _hc_affine(self, name, mixes, prefix, lo, hi, sc_i, bs_shape, out_shape):
        m = self.slice1(f"{name}/slice", mixes, lo, hi, ir.DataType.FLOAT, bs_shape + [hi - lo])
        sc = self.scalar_init(float(self.sd[f"{prefix}_scale"][sc_i]), f"{prefix}_scale/e{sc_i}")
        s = self.op("Mul", [m, sc], f"{name}/scale", ir.DataType.FLOAT, bs_shape + [hi - lo])
        b = self.init(self.sd[f"{prefix}_base"][lo:hi], f"{prefix}_base/s{lo}_{hi}")
        return self.op("Add", [s, b], name, ir.DataType.FLOAT, out_shape)

    def make_hc_pre(self, name, x, prefix, bs_shape):
        """Mix the ``hc`` residual streams into one and produce the re-expansion weights."""
        hc, dim = self.hc, self.dim
        flat_shape = bs_shape + [hc * dim]
        xf = self.cast(f"{name}/castf", x, ir.DataType.FLOAT, bs_shape + [hc, dim])
        xflat = self.reshape(f"{name}/flat", xf, [0, 0, hc * dim], ir.DataType.FLOAT, flat_shape)
        mixes = self._hc_mixes(f"{name}/mixes", xflat, prefix, flat_shape)

        pre_shape = bs_shape + [hc]
        pre = self._hc_affine(f"{name}/pre", mixes, prefix, 0, hc, 0, bs_shape, pre_shape)
        pre = self.op("Sigmoid", [pre], f"{name}/pre/sig", ir.DataType.FLOAT, pre_shape)
        pre = self.op("Add", [pre, self.const("FLOAT", self.hc_eps)], f"{name}/pre/eps",
                      ir.DataType.FLOAT, pre_shape)

        post = self._hc_affine(f"{name}/post", mixes, prefix, hc, 2 * hc, 1, bs_shape, pre_shape)
        post = self.op("Sigmoid", [post], f"{name}/post/sig", ir.DataType.FLOAT, pre_shape)
        post = self.op("Mul", [post, self.const("FLOAT", 2.0)], f"{name}/post/x2",
                       ir.DataType.FLOAT, pre_shape)

        comb_shape = bs_shape + [hc, hc]
        comb = self._hc_affine(f"{name}/comb", mixes, prefix, 2 * hc, (2 + hc) * hc, 2,
                               bs_shape, bs_shape + [hc * hc])
        comb = self.reshape(f"{name}/comb/rs", comb, [0, 0, hc, hc], ir.DataType.FLOAT, comb_shape)
        comb = self.op("Softmax", [comb], f"{name}/comb/sm", ir.DataType.FLOAT, comb_shape, axis=-1)
        comb = self.op("Add", [comb, self.const("FLOAT", self.hc_eps)], f"{name}/comb/eps",
                       ir.DataType.FLOAT, comb_shape)
        # 39 alternating row/column normalizations of a tiny hc*hc matrix: 117 nodes unrolled,
        # so it is fused into one node to keep it off the dispatch path.
        comb = self.op("SinkhornNormalize", [comb], f"{name}/comb/sk", ir.DataType.FLOAT,
                       comb_shape, domain="com.microsoft",
                       iterations=self.hc_iters, epsilon=self.hc_eps)

        pre_u = self.unsq(f"{name}/pre/u", pre, [-1], ir.DataType.FLOAT, pre_shape + [1])
        prod = self.op("Mul", [pre_u, xf], f"{name}/y/mul", ir.DataType.FLOAT, bs_shape + [hc, dim])
        y = self.op("ReduceSum", [prod, self.const("INT64", [2])], f"{name}/y",
                    ir.DataType.FLOAT, bs_shape + [dim], keepdims=0)
        return self.cast(f"{name}/y/cast", y, self.io_dtype, bs_shape + [dim]), post, comb

    def make_hc_post(self, name, x, residual, post, comb, bs_shape):
        """``post[..,h] * x + sum_g comb[..,g,h] * residual[..,g,:]`` -> [.., hc, dim]."""
        hc, dim = self.hc, self.dim
        out_shape = bs_shape + [hc, dim]
        xf = self.cast(f"{name}/castx", x, ir.DataType.FLOAT, bs_shape + [dim])
        xu = self.unsq(f"{name}/xu", xf, [-2], ir.DataType.FLOAT, bs_shape + [1, dim])
        pu = self.unsq(f"{name}/pu", post, [-1], ir.DataType.FLOAT, bs_shape + [hc, 1])
        term1 = self.op("Mul", [pu, xu], f"{name}/t1", ir.DataType.FLOAT, out_shape)

        rf = self.cast(f"{name}/castr", residual, ir.DataType.FLOAT, out_shape)
        ru = self.unsq(f"{name}/ru", rf, [-2], ir.DataType.FLOAT, bs_shape + [hc, 1, dim])
        cu = self.unsq(f"{name}/cu", comb, [-1], ir.DataType.FLOAT, bs_shape + [hc, hc, 1])
        prod = self.op("Mul", [cu, ru], f"{name}/t2/mul", ir.DataType.FLOAT,
                       bs_shape + [hc, hc, dim])
        term2 = self.op("ReduceSum", [prod, self.const("INT64", [2])], f"{name}/t2",
                        ir.DataType.FLOAT, out_shape, keepdims=0)
        out = self.op("Add", [term1, term2], f"{name}/sum", ir.DataType.FLOAT, out_shape)
        return self.cast(name, out, self.io_dtype, out_shape)

    def make_hc_norm(self, name, y, norm_key, bs_shape):
        return self.cast(f"{name}_b",
                         self.make_rmsnorm(name, y, self.init_w(norm_key, to=ir.DataType.FLOAT),
                                           bs_shape + [self.dim]),
                         self.io_dtype, bs_shape + [self.dim])

    def make_hc_mix(self, name, x, residual, post, comb, nxt, bs_shape):
        """Post mix of the block that just ran, then the next block's pre mix and norm.

        ``nxt`` is ``(pre_name, norm_name, prefix, norm_key)`` for the pre mix that follows.
        """
        pre_name, norm_name, prefix, norm_key = nxt
        hc, dim = self.hc, self.dim
        if not self.hc_fused:
            h = self.make_hc_post(name, x, residual, post, comb, bs_shape)
            y, post, comb = self.make_hc_pre(pre_name, h, prefix, bs_shape)
            return h, post, comb, self.make_hc_norm(norm_name, y, norm_key, bs_shape)

        F = ir.DataType.FLOAT
        outs = [f"{name}/output_{i}" for i in range(4)]
        self.make_node(
            "HyperConnectionMix",
            inputs=[x, residual, post, comb,
                    self.init(self.sd[f"{prefix}_fn"].T, f"{prefix}_fn/T", to=F),
                    self.init(self.sd[f"{prefix}_scale"].reshape(-1)[:3], f"{prefix}_scale/v", to=F),
                    self.init(self.sd[f"{prefix}_base"].reshape(-1)[:self.hc_mix_dim],
                              f"{prefix}_base/v", to=F),
                    self.init_w(norm_key, to=F)],
            outputs=outs, name=name, domain="com.microsoft",
            sinkhorn_iterations=self.hc_iters, epsilon=self.eps, hc_epsilon=self.hc_eps,
            sinkhorn_epsilon=self.hc_eps, post_alpha=2.0)
        self.make_value(outs[0], self.io_dtype, shape=bs_shape + [hc, dim])
        self.make_value(outs[1], F, shape=bs_shape + [hc])
        self.make_value(outs[2], F, shape=bs_shape + [hc, hc])
        self.make_value(outs[3], self.io_dtype, shape=bs_shape + [dim])
        return outs[0], outs[1], outs[2], outs[3]

    def make_hc_head(self, name, x, prefix, bs_shape):
        hc, dim = self.hc, self.dim
        flat_shape = bs_shape + [hc * dim]
        xf = self.cast(f"{name}/castf", x, ir.DataType.FLOAT, bs_shape + [hc, dim])
        xflat = self.reshape(f"{name}/flat", xf, [0, 0, hc * dim], ir.DataType.FLOAT, flat_shape)
        mixes = self._hc_mixes(f"{name}/mixes", xflat, prefix, flat_shape)
        sc = self.scalar_init(float(self.sd[f"{prefix}_scale"][0]), f"{prefix}_scale/e0")
        s = self.op("Mul", [mixes, sc], f"{name}/scale", ir.DataType.FLOAT, bs_shape + [hc])
        b = self.init_w(f"{prefix}_base")
        s = self.op("Add", [s, b], f"{name}/bias", ir.DataType.FLOAT, bs_shape + [hc])
        pre = self.op("Sigmoid", [s], f"{name}/sig", ir.DataType.FLOAT, bs_shape + [hc])
        pre = self.op("Add", [pre, self.const("FLOAT", self.hc_eps)], f"{name}/eps",
                      ir.DataType.FLOAT, bs_shape + [hc])
        pu = self.unsq(f"{name}/u", pre, [-1], ir.DataType.FLOAT, bs_shape + [hc, 1])
        prod = self.op("Mul", [pu, xf], f"{name}/mul", ir.DataType.FLOAT, bs_shape + [hc, dim])
        y = self.op("ReduceSum", [prod, self.const("INT64", [2])], f"{name}/sum",
                    ir.DataType.FLOAT, bs_shape + [dim], keepdims=0)
        return self.cast(name, y, self.io_dtype, bs_shape + [dim])

    # ------------------------------------------------------------------ #
    # KV compressor
    # ------------------------------------------------------------------ #

    def make_compressor(self, name, layer_id, ratio, x, ctx, prefix=None, head_dim=None,
                        state="cstate", quant=True, rotate=False):
        """Gated pooling of ``ratio`` consecutive tokens into one latent row.

        Returns (rows [B, J, head_dim] io_dtype, first_slot [B, 1], last_slot [B, 1], J,
                 J_g, present_cstate_kv, present_cstate_score). ``J`` depends only on the
        sequence length, never on ``past_lens``, so the shape stays inferable even
        though the slots each row lands in differ.  ``J_g`` is the same value computed
        on the device, for callers that feed it to a kernel instead of to a Range.

        The indexer owns a second, narrower compressor over the same tokens, hence the
        ``prefix``/``head_dim``/``state`` parameters; it also rotates and rounds its rows to
        FP4 instead of the FP8 round trip the attention cache uses.
        """
        p = prefix or f"layers.{layer_id}.attn.compressor"
        d = head_dim or self.head_dim
        r, co, Lst = ratio, self.coff(ratio), self.state_len(ratio)
        rd = self.rope_head_dim
        nd = d - rd
        F = ir.DataType.FLOAT
        I = ir.DataType.INT64

        xf = self.cast(f"{name}/castx", x, F, [B, S, self.dim])
        kv = self.op("MatMul", [xf, self.init(self.sd[f"{p}.wkv.weight"].T, f"{p}.wkv/T", to=F)],
                     f"{name}/kv", F, [B, S, co * d])
        sc = self.op("MatMul", [xf, self.init(self.sd[f"{p}.wgate.weight"].T, f"{p}.wgate/T",
                                              to=F)],
                     f"{name}/sc", F, [B, S, co * d])

        if self.comp_fused:
            # `J` stays on the host: it feeds a Range, and `row_count` is the same count on the
            # device for the Clip bounds our callers build.
            jm1 = self.op("Sub", [ctx["S"], self.const("INT64", 1)], f"{name}/jm1", I, [])
            jdiv = self.op("Div", [jm1, self.const("INT64", r)], f"{name}/jdiv", I, [])
            J = self.op("Add", [jdiv, self.const("INT64", 2)], f"{name}/J", I, [])
            outs = [f"{name}/output_{i}" for i in range(6)]
            self.make_node(
                "DSV4Compressor",
                inputs=[kv, sc, f"past_{state}_kv_{layer_id}", f"past_{state}_score_{layer_id}",
                        self.init_w(f"{p}.ape", to=F),
                        self.init_w(f"{p}.norm.weight", to=F),
                        "rope/cos_1", "rope/sin_1", "past_lens"],
                outputs=outs, name=name, domain="com.microsoft",
                ratio=r, coff=co, head_dim=d, rope_head_dim=rd, max_seq_len=self.max_seq_len,
                epsilon=self.eps, act_quant=int(bool(quant)), rotate_fp4=int(bool(rotate)),
                dtype=int(self.io_dtype))
            self.make_value(outs[0], self.io_dtype, shape=[B, None, d])
            self.make_value(outs[1], I, shape=[B, 1])
            self.make_value(outs[2], I, shape=[B, 1])
            self.make_value(outs[3], I, shape=[])
            self.make_value(outs[4], F, shape=[B, Lst, co * d])
            self.make_value(outs[5], F, shape=[B, Lst, co * d])
            return outs[0], outs[1], outs[2], J, outs[3], outs[4], outs[5]

        full_kv = self.op("Concat", [f"past_{state}_kv_{layer_id}", kv], f"{name}/fkv",
                          F, [B, None, co * d], axis=1)
        full_sc = self.op("Concat", [f"past_{state}_score_{layer_id}", sc], f"{name}/fsc",
                          F, [B, None, co * d], axis=1)

        # `n_full`/`n_max` are only ever read as data by the Less and Clip below, so they
        # take the device-resident length; `jm1`/`jdiv`/`J` feed a Range and must stay on
        # the host.  `J_g` is the same count for the Clip bounds our callers build.
        n_full = self.op("Add", [ctx["Sg"], self.const("INT64", Lst)], f"{name}/nfull", I, [])
        n_max = self.op("Sub", [n_full, self.const("INT64", 1)], f"{name}/nmax", I, [])
        base = self.op("Sub", [ctx["past3"], self.const("INT64", Lst)], f"{name}/base", I, [B, 1, 1])
        first_slot = self.op("Div", [ctx["past2"], self.const("INT64", r)],
                             f"{name}/first", I, [B, 1])
        end = self.op("Sub", [ctx["total2"], self.const("INT64", 1)], f"{name}/end", I, [B, 1])
        last_slot = self.op("Div", [end, self.const("INT64", r)], f"{name}/last", I, [B, 1])
        jm1 = self.op("Sub", [ctx["S"], self.const("INT64", 1)], f"{name}/jm1", I, [])
        jdiv = self.op("Div", [jm1, self.const("INT64", r)], f"{name}/jdiv", I, [])
        J = self.op("Add", [jdiv, self.const("INT64", 2)], f"{name}/J", I, [])
        jm1_g = self.op("Sub", [ctx["Sg"], self.const("INT64", 1)], f"{name}/jm1g", I, [])
        jdiv_g = self.op("Div", [jm1_g, self.const("INT64", r)], f"{name}/jdivg", I, [])
        J_g = self.op("Add", [jdiv_g, self.const("INT64", 2)], f"{name}/Jg", I, [])

        rng = self.op("Range", [self.const("INT64", 0), J, self.const("INT64", 1)],
                      f"{name}/rng", I, [None])
        slot = self.op("Add", [rng, first_slot], f"{name}/slot", I, [B, None])
        slot2 = self.unsq(f"{name}/slot2", slot, [2], I, [B, None, 1])
        slot_r = self.op("Mul", [slot2, self.const("INT64", r)], f"{name}/slotr", I, [B, None, 1])
        off = self.unsq(f"{name}/off", self.init(torch.arange(r, dtype=torch.int64), f"{name}/arange"),
                        [0], I, [1, r])
        pos_cur = self.op("Add", [slot_r, off], f"{name}/pos", I, [B, None, r])

        def window(tag, pos_t):
            idx = self.op("Sub", [pos_t, base], f"{name}/{tag}/idx", I, [B, None, r])
            ge0 = self.op("GreaterOrEqual", [idx, self.const("INT64", 0)],
                          f"{name}/{tag}/ge0", ir.DataType.BOOL, [B, None, r])
            lt = self.op("Less", [idx, n_full], f"{name}/{tag}/lt", ir.DataType.BOOL, [B, None, r])
            inrange = self.op("And", [ge0, lt], f"{name}/{tag}/and", ir.DataType.BOOL, [B, None, r])
            fut = self.op("Less", [pos_t, ctx["total3"]], f"{name}/{tag}/fut",
                          ir.DataType.BOOL, [B, None, r])
            valid = self.op("And", [inrange, fut], f"{name}/{tag}/valid",
                            ir.DataType.BOOL, [B, None, r])
            if tag == "prev":
                nneg = self.op("GreaterOrEqual", [pos_t, self.const("INT64", 0)],
                               f"{name}/{tag}/nneg", ir.DataType.BOOL, [B, None, r])
                valid = self.op("And", [valid, nneg], f"{name}/{tag}/valid2",
                                ir.DataType.BOOL, [B, None, r])
            cl = self.op("Clip", [idx, self.const("INT64", 0), n_max],
                         f"{name}/{tag}/clip", I, [B, None, r])
            # [B, J*r, 1]: GatherND index vectors, one per batch row.
            flat = self.reshape(f"{name}/{tag}/flat", cl, [0, -1, 1], I, [B, None, 1])
            return flat, valid

        def gather(tag, src, flat):
            g = self.op("GatherND", [src, flat], f"{name}/{tag}/g", F, [B, None, co * d],
                        batch_dims=1)
            return self.reshape(f"{name}/{tag}/g4", g, [0, -1, r, co * d], F, [B, None, r, co * d])

        ape = self.init_w(f"{p}.ape")
        if co == 2:
            pos_prev = self.op("Sub", [pos_cur, self.const("INT64", r)], f"{name}/posp", I,
                               [B, None, r])
            fc, vc = window("cur", pos_cur)
            fp_, vp = window("prev", pos_prev)
            kv_p = self.slice1(f"{name}/kvp", gather("kvprev", full_kv, fp_), 0, d, F,
                               [B, None, r, d])
            kv_c = self.slice1(f"{name}/kvc", gather("kvcur", full_kv, fc), d, INT64_MAX, F,
                               [B, None, r, d])
            ape_p = self.init(self.sd[f"{p}.ape"][:, :d], f"{p}.ape/lo")
            ape_c = self.init(self.sd[f"{p}.ape"][:, d:], f"{p}.ape/hi")
            sc_p = self.op("Add", [self.slice1(f"{name}/scp", gather("scprev", full_sc, fp_),
                                               0, d, F, [B, None, r, d]), ape_p],
                           f"{name}/scpa", F, [B, None, r, d])
            sc_c = self.op("Add", [self.slice1(f"{name}/scc", gather("sccur", full_sc, fc),
                                               d, INT64_MAX, F, [B, None, r, d]), ape_c],
                           f"{name}/scca", F, [B, None, r, d])
            pooled_kv = self.op("Concat", [kv_p, kv_c], f"{name}/pkv", F, [B, None, 2 * r, d], axis=2)
            pooled_sc = self.op("Concat", [sc_p, sc_c], f"{name}/psc", F, [B, None, 2 * r, d], axis=2)
            valid = self.op("Concat", [vp, vc], f"{name}/valid", ir.DataType.BOOL,
                            [B, None, 2 * r], axis=2)
            span = 2 * r
        else:
            fc, valid = window("cur", pos_cur)
            pooled_kv = gather("kv", full_kv, fc)
            pooled_sc = self.op("Add", [gather("sc", full_sc, fc), ape], f"{name}/psc",
                                F, [B, None, r, d])
            span = r

        vmask = self.unsq(f"{name}/vmask", valid, [3], ir.DataType.BOOL, [B, None, span, 1])
        masked = self.op("Where", [vmask, pooled_sc, self.const("FLOAT", NEG_INF)],
                         f"{name}/masked", F, [B, None, span, d])
        wsm = self.op("Softmax", [masked], f"{name}/wsm", F, [B, None, span, d], axis=2)
        prod = self.op("Mul", [pooled_kv, wsm], f"{name}/prod", F, [B, None, span, d])
        pooled = self.op("ReduceSum", [prod, self.const("INT64", [2])], f"{name}/pooled",
                         F, [B, None, d], keepdims=0)

        pooled_b = self.cast(f"{name}/pooled_b", pooled, self.io_dtype, [B, None, d])
        normed = self.make_rmsnorm(f"{name}/norm", pooled_b,
                                   self.init_w(f"{p}.norm.weight",
                                               to=ir.DataType.FLOAT),
                                   [B, None, d])

        pos_slot = self.op("Clip", [self.reshape(f"{name}/slotflat", slot_r, [0, -1], I, [B, None]),
                                    self.const("INT64", 0),
                                    self.const("INT64", self.max_seq_len - 1)],
                           f"{name}/posslot", I, [B, None])
        cos = self.op("Gather", [f"rope/cos_1", pos_slot], f"{name}/cosg", F, [B, None, rd], axis=0)
        sin = self.op("Gather", [f"rope/sin_1", pos_slot], f"{name}/sing", F, [B, None, rd], axis=0)
        nope = self.slice1(f"{name}/nope", normed, 0, nd, F, [B, None, nd])
        rope = self.slice1(f"{name}/ropep", normed, nd, INT64_MAX, F, [B, None, rd])
        rope = self.make_rope(f"{name}/rope", rope, cos, sin, [B, None, rd])
        if quant:
            nope = self.make_act_quant(f"{name}/aq", nope, [B, None, nd])
        rows = self.op("Concat", [nope, rope], f"{name}/rows", F, [B, None, d], axis=-1)
        if rotate:
            rows = self.make_rotate_fp4(f"{name}/fp4", rows, [B, None, d])
        rows = self.cast(f"{name}/rows_b", rows, self.io_dtype, [B, None, d])

        st_shape = [B, Lst, co * d]
        pkv = self.op("Slice", [full_kv, self.const("INT64", [-Lst]),
                                self.const("INT64", [INT64_MAX]), self.const("INT64", [1])],
                      f"{name}/pstate_kv", F, st_shape)
        psc = self.op("Slice", [full_sc, self.const("INT64", [-Lst]),
                                self.const("INT64", [INT64_MAX]), self.const("INT64", [1])],
                      f"{name}/pstate_sc", F, st_shape)
        return rows, first_slot, last_slot, J, J_g, pkv, psc

    # ------------------------------------------------------------------ #
    # attention
    # ------------------------------------------------------------------ #

    def make_attention(self, name, layer_id, x, ctx):
        p = f"layers.{layer_id}.attn"
        ratio = self.compress_ratios[layer_id]
        # Tensor parallel: heads and output-projection groups are sliced together
        # (heads are laid out contiguously within a group, so the two splits agree).
        # KV is MQA -- a single latent row -- so it stays replicated.
        H, D, rd, nd, W = (self.n_heads_local, self.head_dim, self.rope_head_dim,
                           self.nope_dim, self.window)
        G, R = self.o_groups_local, self.o_lora_rank
        dh = self.n_heads * self.head_dim // self.o_groups
        F, I, BOOL = ir.DataType.FLOAT, ir.DataType.INT64, ir.DataType.BOOL
        io = self.io_dtype
        tbl = 1 if ratio else 0

        cos = self.op("Gather", [f"rope/cos_{tbl}", ctx["pos"]], f"{name}/cos", F, [B, S, rd], axis=0)
        sin = self.op("Gather", [f"rope/sin_{tbl}", ctx["pos"]], f"{name}/sin", F, [B, S, rd], axis=0)
        cos_q = self.unsq(f"{name}/cosq", cos, [2], F, [B, S, 1, rd])
        sin_q = self.unsq(f"{name}/sinq", sin, [2], F, [B, S, 1, rd])
        cos_k, sin_k = cos, sin

        def rope_last(tag, t, shape, c, s, inverse=False):
            n = self.slice1(f"{tag}/nope", t, 0, nd, io, shape[:-1] + [nd])
            r_ = self.slice1(f"{tag}/rope", t, nd, INT64_MAX, io, shape[:-1] + [rd])
            r_ = self.cast(f"{tag}/ropef", r_, F, shape[:-1] + [rd])
            r_ = self.make_rope(f"{tag}/rot", r_, c, s, shape[:-1] + [rd], inverse=inverse)
            r_ = self.cast(f"{tag}/ropeb", r_, io, shape[:-1] + [rd])
            return self.op("Concat", [n, r_], tag, io, shape, axis=-1)

        # ---- Q ----
        qa = self.proj(f"{name}/qa", x, f"{p}.wq_a.weight", [B, S, self.q_lora_rank])
        qn = self.make_rmsnorm(f"{name}/qnorm", qa,
                               self.init_w(f"{p}.q_norm.weight",
                                           to=ir.DataType.FLOAT),
                               [B, S, self.q_lora_rank])
        qn = self.cast(f"{name}/qnorm_b", qn, io, [B, S, self.q_lora_rank])
        q = self.proj(f"{name}/qb", qn, f"{p}.wq_b.weight", [B, S, H * D], shard_axis=0)
        q = self.reshape(f"{name}/q4", q, [0, 0, H, D], io, [B, S, H, D])
        q = self.make_weightless_rmsnorm(f"{name}/qrms", q, [B, S, H, D], io)
        q = rope_last(f"{name}/qrope", q, [B, S, H, D], cos_q, sin_q)

        # ---- KV (single latent row shared by all heads) ----
        kv = self.proj(f"{name}/kv", x, f"{p}.wkv.weight", [B, S, D])
        kv = self.make_rmsnorm(f"{name}/kvnorm", kv,
                               self.init_w(f"{p}.kv_norm.weight",
                                           to=ir.DataType.FLOAT),
                               [B, S, D])
        kv = self.cast(f"{name}/kvnorm_b", kv, io, [B, S, D])
        kv = rope_last(f"{name}/kvrope", kv, [B, S, D], cos_k, sin_k)
        kv_n = self.slice1(f"{name}/kvn", kv, 0, nd, io, [B, S, nd])
        kv_r = self.slice1(f"{name}/kvr", kv, nd, INT64_MAX, io, [B, S, rd])
        kv_n = self.make_act_quant(f"{name}/kvaq",
                                   self.cast(f"{name}/kvnf", kv_n, F, [B, S, nd]), [B, S, nd])
        kv_n = self.cast(f"{name}/kvnb", kv_n, io, [B, S, nd])
        kv = self.op("Concat", [kv_n, kv_r], f"{name}/kvq", io, [B, S, D], axis=-1)

        if self.paged:
            o, presents = self.make_paged_attention(name, layer_id, ratio, q, kv, x, ctx,
                                                    qr=qn, cos_q=cos_q, sin_q=sin_q)
        else:
            o, presents = self.make_dense_attention(name, layer_id, ratio, q, kv, x, ctx)

        o = rope_last(f"{name}/orope", o, [B, S, H, D], cos_q, sin_q, inverse=True)

        # ---- grouped output projection ----
        o = self.reshape(f"{name}/og", o, [0, 0, G, -1], io, [B, S, G, dh])
        o = self.op("Transpose", [o], f"{name}/ot", io, [G, B, S, dh], perm=[2, 0, 1, 3])
        o = self.reshape(f"{name}/of", o, [G, -1, dh], io, [G, None, dh])
        wa = self.shard(self.sd[f"{p}.wo_a.weight"].reshape(self.o_groups, R, dh), 0)
        o = self.op("MatMul", [o, self.init(wa.transpose(1, 2), f"{p}.wo_a/G", to=io)],
                    f"{name}/oa", io, [G, None, R])
        o = self.op("Transpose", [o], f"{name}/oat", io, [None, G, R], perm=[1, 0, 2])
        o = self.reshape(f"{name}/oaf", o, [-1, G * R], io, [None, G * R])
        o = self.proj(f"{name}/ob2", o, f"{p}.wo_b.weight", [None, self.dim], shard_axis=1)
        # Each rank holds a column block of wo_b, so this is a partial sum.
        o = self.all_reduce(f"{name}/ar", o, io, [None, self.dim])
        o = self.dyn_reshape(f"{name}/out", o, ctx["bs"], [self.dim], io, [B, S, self.dim])
        return o, presents

    def make_dense_attention(self, name, layer_id, ratio, q, kv, x, ctx):
        """Window ring plus compressed rows concatenated into one dense score tensor.

        Returns ``(o [B, S, H, D] io_dtype, presents)``. The ring holds exactly ``window`` rows
        and the fresh ``kv`` is read alongside it rather than through it, because a chunk of
        ``S > 1`` tokens would otherwise overwrite rows its own earlier tokens still need.
        """
        p = f"layers.{layer_id}.attn"
        H, D, W = self.n_heads_local, self.head_dim, self.window
        F, I, BOOL = ir.DataType.FLOAT, ir.DataType.INT64, ir.DataType.BOOL
        io = self.io_dtype

        # ---- sliding-window ring cache ----
        ring = f"past_kv_{layer_id}"
        j = self.init(torch.arange(W, dtype=torch.int64), "ring/arange")
        qpos = ctx["qpos"]
        qposw = self.op("Sub", [qpos, self.const("INT64", W)], f"{name}/qposw", I, [B, S, 1])

        def ring_pos(tag, endval):
            t = self.op("Sub", [endval, j], f"{name}/{tag}/t", I, [B, W])
            fd = self.floordiv(f"{name}/{tag}/fd", t, W, [B, W])
            m = self.op("Mul", [fd, self.const("INT64", W)], f"{name}/{tag}/m", I, [B, W])
            return self.op("Add", [m, j], f"{name}/{tag}/q", I, [B, W])

        pl_1 = self.op("Sub", [ctx["past2"], self.const("INT64", 1)], f"{name}/pl1", I, [B, 1])
        tot_1 = self.op("Sub", [ctx["total2"], self.const("INT64", 1)], f"{name}/tot1", I, [B, 1])
        q_old = ring_pos("old", pl_1)
        q_new = ring_pos("new", tot_1)

        q_old2 = self.unsq(f"{name}/qold2", q_old, [1], I, [B, 1, W])
        m1 = self.op("GreaterOrEqual", [q_old2, self.const("INT64", 0)], f"{name}/m1", BOOL, [B, 1, W])
        m2 = self.op("Less", [q_old2, ctx["past3"]], f"{name}/m2", BOOL, [B, 1, W])
        m3 = self.op("Greater", [q_old2, qposw], f"{name}/m3", BOOL, [B, S, W])
        ring_mask = self.op("And", [self.op("And", [m1, m2], f"{name}/m12", BOOL, [B, 1, W]), m3],
                            f"{name}/ringmask", BOOL, [B, S, W])

        fpos = self.unsq(f"{name}/fpos", ctx["pos"], [1], I, [B, 1, S])
        f1 = self.op("LessOrEqual", [fpos, qpos], f"{name}/f1", BOOL, [B, S, S])
        f2 = self.op("Greater", [fpos, qposw], f"{name}/f2", BOOL, [B, S, S])
        fresh_mask = self.op("And", [f1, f2], f"{name}/freshmask", BOOL, [B, S, S])

        tf1 = self.op("GreaterOrEqual", [q_new, ctx["past2"]], f"{name}/tf1", BOOL, [B, W])
        tf2 = self.op("GreaterOrEqual", [q_new, self.const("INT64", 0)], f"{name}/tf2", BOOL, [B, W])
        take_fresh = self.unsq(f"{name}/tf", self.op("And", [tf1, tf2], f"{name}/tfa", BOOL, [B, W]),
                               [2], BOOL, [B, W, 1])
        smax_idx = self.op("Sub", [ctx["S"], self.const("INT64", 1)], f"{name}/smaxidx", I, [])
        gidx = self.op("Clip", [self.op("Sub", [q_new, ctx["past2"]], f"{name}/gi", I, [B, W]),
                                self.const("INT64", 0), smax_idx], f"{name}/gic", I, [B, W])
        src = self.batched_gather(f"{name}/src", kv, gidx, W, io, [B, W, D])
        present_kv = self.op("Where", [take_fresh, src, ring], f"{name}/present_kv", io, [B, W, D])

        keys = [ring, kv]
        masks = [ring_mask, fresh_mask]
        presents = {"kv": present_kv}

        if ratio:
            rows, first_slot, last_slot, _J, J_g, pkv, psc = self.make_compressor(
                f"{name}/comp", layer_id, ratio, x, ctx)
            C = self.comp_capacity(ratio)
            cidx = self.init(torch.arange(C, dtype=torch.int64), f"comp/arange_{C}")
            ksel = self.op("Sub", [cidx, first_slot], f"{name}/ksel", I, [B, C])
            n1 = self.op("GreaterOrEqual", [ksel, self.const("INT64", 0)], f"{name}/n1", BOOL, [B, C])
            n2 = self.op("LessOrEqual", [cidx, last_slot], f"{name}/n2", BOOL, [B, C])
            take_new = self.unsq(f"{name}/tn", self.op("And", [n1, n2], f"{name}/na", BOOL, [B, C]),
                                 [2], BOOL, [B, C, 1])
            jmax = self.op("Sub", [J_g, self.const("INT64", 1)], f"{name}/jmax", I, [])
            kclip = self.op("Clip", [ksel, self.const("INT64", 0), jmax], f"{name}/kclip", I, [B, C])
            newc = self.batched_gather(f"{name}/newc", rows, kclip, C, io, [B, C, D])
            present_comp = self.op("Where", [take_new, newc, f"past_comp_{layer_id}"],
                                   f"{name}/present_comp", io, [B, C, D])
            qp1 = self.op("Add", [qpos, self.const("INT64", 1)], f"{name}/qp1", I, [B, S, 1])
            qpr = self.op("Div", [qp1, self.const("INT64", ratio)], f"{name}/qpr", I, [B, S, 1])
            comp_mask = self.op("Less", [self.unsq(f"{name}/cidx2", cidx, [0, 1], I, [1, 1, C]), qpr],
                                f"{name}/compmask", BOOL, [B, S, C])
            keys.append(present_comp)
            masks.append(comp_mask)
            presents.update(comp=present_comp, cstate_kv=pkv, cstate_score=psc)

        k = self.op("Concat", keys, f"{name}/keys", io, [B, None, D], axis=1)
        mask = self.op("Concat", masks, f"{name}/mask", BOOL, [B, S, None], axis=2)

        qf = self.cast(f"{name}/qf", q, F, [B, S, H, D])
        kf = self.cast(f"{name}/kf", k, F, [B, None, D])
        scores = self.op("Einsum", [qf, kf], f"{name}/scores", F, [B, S, H, None],
                         equation="bshd,bnd->bshn")
        scores = self.op("Mul", [scores, self.const("FLOAT", self.softmax_scale)],
                         f"{name}/scaled", F, [B, S, H, None])
        m4 = self.unsq(f"{name}/mask4", mask, [2], BOOL, [B, S, 1, None])
        scores = self.op("Where", [m4, scores, self.const("FLOAT", NEG_INF)],
                         f"{name}/masked", F, [B, S, H, None])
        smx = self.op("ReduceMax", [scores, self.const("INT64", [-1])], f"{name}/smax",
                      F, [B, S, H, 1], keepdims=1)
        pex = self.op("Exp", [self.op("Sub", [scores, smx], f"{name}/shift", F, [B, S, H, None])],
                      f"{name}/exp", F, [B, S, H, None])
        denom = self.op("ReduceSum", [pex, self.const("INT64", [-1])], f"{name}/den",
                        F, [B, S, H, 1], keepdims=1)
        sink = self.init(self.shard(self.sd[f"{p}.attn_sink"], 0).reshape(1, 1, H, 1),
                         f"{p}.attn_sink")
        sink_e = self.op("Exp", [self.op("Sub", [sink, smx], f"{name}/sinkshift", F, [B, S, H, 1])],
                         f"{name}/sinkexp", F, [B, S, H, 1])
        denom = self.op("Add", [denom, sink_e], f"{name}/den2", F, [B, S, H, 1])
        o = self.op("Einsum", [pex, kf], f"{name}/ctx", F, [B, S, H, D], equation="bshn,bnd->bshd")
        o = self.op("Div", [o, denom], f"{name}/norm", F, [B, S, H, D])
        return self.cast(f"{name}/ob", o, io, [B, S, H, D]), presents

    def make_indexer(self, name, layer_id, ratio, x, qr, cos_q, sin_q, ctx):
        """Lightning Indexer: which compressed rows each query is allowed to attend to.

        Returns ``(csel [B, S, k] int64, presents)`` where ``csel`` holds logical cache
        positions ready to concatenate into ``kv_indices`` (``-1`` == masked).

        It scores every compressed row with its own narrower compressor and keeps the top
        ``k = min(index_topk, comp_capacity)``. That ``k`` is a build-time constant, while the
        reference recomputes ``min(index_topk, end_pos // ratio)`` each step -- but whenever the
        latter is smaller, every query's visible row count is below it too, so both end up
        selecting the whole visible set. The extra picks here are all invalid and get masked.

        Its cache is a plain dense tensor, not a paged one: the scoring einsum reads all of it
        every step, so paging would buy nothing.
        """
        p = f"layers.{layer_id}.attn.indexer"
        NH, HD, rd = self.index_n_heads, self.index_head_dim, self.rope_head_dim
        nd = HD - rd
        C, L = self.comp_capacity(ratio), self.max_seq_len
        k = self.index_width(ratio)
        F, I, BOOL = ir.DataType.FLOAT, ir.DataType.INT64, ir.DataType.BOOL
        io = self.io_dtype

        # The heads are replicated, not sharded: the cache they read is already whole on
        # every rank, so sharding would divide the arithmetic without dividing the traffic
        # that bounds it, and would cost an all-reduce of the score per layer.
        q_raw = self.proj(f"{name}/qb", qr, f"{p}.wq_b.weight", [B, S, NH * HD])
        w_raw = self.proj(f"{name}/w", x, f"{p}.weights_proj.weight", [B, S, NH])

        # ---- this step's rows, written into the dense indexer cache ----
        rows, first_slot, last_slot, _J, J_g, pkv, psc = self.make_compressor(
            f"{name}/ic", layer_id, ratio, x, ctx, prefix=f"{p}.compressor",
            head_dim=HD, state="icstate", quant=False, rotate=True)

        if self.index_fused:
            outs = [f"{name}/output_0", f"{name}/output_1"]
            self.make_node(
                "LightningIndexer",
                inputs=[q_raw, cos_q, sin_q, rows, first_slot, last_slot,
                        f"past_icache_{layer_id}", w_raw, "past_lens"],
                outputs=outs, name=name, domain="com.microsoft",
                num_heads=NH, head_dim=HD, rope_head_dim=rd, ratio=ratio, topk=k,
                max_seq_len=L, rotate_fp4=1,
                scale=(HD ** -0.5) * self.index_n_heads ** -0.5)
            self.make_value(outs[0], I, shape=[B, S, k])
            self.make_value(outs[1], io, shape=[B, C, HD])
            return outs[0], {"icache": outs[1], "icstate_kv": pkv, "icstate_score": psc}

        q = self.cast(f"{name}/qf", self.reshape(f"{name}/q4", q_raw, [0, 0, NH, HD], io,
                                                 [B, S, NH, HD]), F, [B, S, NH, HD])
        q = self.op("Concat",
                    [self.slice1(f"{name}/qn", q, 0, nd, F, [B, S, NH, nd]),
                     self.make_rope(f"{name}/qrot",
                                    self.slice1(f"{name}/qr", q, nd, INT64_MAX, F,
                                                [B, S, NH, rd]),
                                    cos_q, sin_q, [B, S, NH, rd])],
                    f"{name}/qcat", F, [B, S, NH, HD], axis=-1)
        q = self.make_rotate_fp4(f"{name}/qfp4", q, [B, S, NH, HD])

        cidx = self.init(torch.arange(C, dtype=torch.int64), f"comp/arange_{C}")
        ksel = self.op("Sub", [cidx, first_slot], f"{name}/ksel", I, [B, C])
        take = self.op("And",
                       [self.op("GreaterOrEqual", [ksel, self.const("INT64", 0)],
                                f"{name}/kge", BOOL, [B, C]),
                        self.op("LessOrEqual", [cidx, last_slot], f"{name}/kle", BOOL, [B, C])],
                       f"{name}/take", BOOL, [B, C])
        kcl = self.op("Clip", [ksel, self.const("INT64", 0),
                               self.op("Sub", [J_g, self.const("INT64", 1)], f"{name}/jm1", I, [])],
                      f"{name}/kclip", I, [B, C])
        present = self.op("Where",
                          [self.unsq(f"{name}/take3", take, [2], BOOL, [B, C, 1]),
                           self.batched_gather(f"{name}/newc", rows, kcl, C, io, [B, C, HD]),
                           f"past_icache_{layer_id}"],
                          f"{name}/icache", io, [B, C, HD])

        # ---- score ----
        w = self.op("Mul", [self.cast(f"{name}/wf", w_raw, F, [B, S, NH]),
                            self.const("FLOAT", (HD ** -0.5) * self.index_n_heads ** -0.5)],
                    f"{name}/ws", F, [B, S, NH])
        score = self.op("Einsum", [q, self.cast(f"{name}/pf", present, F, [B, C, HD])],
                        f"{name}/score", F, [B, S, NH, C], equation="bshd,btd->bsht")
        score = self.op("Mul", [self.op("Relu", [score], f"{name}/relu", F, [B, S, NH, C]),
                                self.unsq(f"{name}/w4", w, [3], F, [B, S, NH, 1])],
                        f"{name}/wsc", F, [B, S, NH, C])
        score = self.op("ReduceSum", [score, self.const("INT64", [2])], f"{name}/sum",
                        F, [B, S, C], keepdims=0)

        # ---- top-k over the rows this query can actually see ----
        qpr = self.op("Div", [self.op("Add", [ctx["qpos"], self.const("INT64", 1)],
                                      f"{name}/qp1", I, [B, S, 1]), self.const("INT64", ratio)],
                      f"{name}/qpr", I, [B, S, 1])
        vis = self.op("Less", [self.unsq(f"{name}/cidx2", cidx, [0, 1], I, [1, 1, C]), qpr],
                      f"{name}/vis", BOOL, [B, S, C])
        score = self.op("Where", [vis, score, self.const("FLOAT", NEG_INF)],
                        f"{name}/masked", F, [B, S, C])
        node = f"{name}/topk"
        vals, idx = f"{node}/output_0", f"{node}/output_1"
        self.make_node("TopK", inputs=[score, self.const("INT64", [k])], outputs=[vals, idx],
                       name=node, axis=-1, largest=1, sorted=1)
        self.make_value(vals, F, shape=[B, S, k])
        self.make_value(idx, I, shape=[B, S, k])
        csel = self.op("Where",
                       [self.op("Less", [idx, qpr], f"{name}/pick", BOOL, [B, S, k]),
                        self.op("Add", [idx, self.const("INT64", L)], f"{name}/abs", I, [B, S, k]),
                        self.const("INT64", -1)],
                       f"{name}/csel", I, [B, S, k])
        return csel, {"icache": present, "icstate_kv": pkv, "icstate_score": psc}

    def make_paged_attention(self, name, layer_id, ratio, q, kv, x, ctx,
                             qr=None, cos_q=None, sin_q=None):
        """The same attention as one ``com.microsoft.PagedAttention`` node (LATENT layout).

        Returns ``(o [B, S, H, D] io_dtype, presents)``. One flat paged cache per layer holds
        both KV streams, in a logical position space of

            ``[0, max_seq_len)``                              one row per token
            ``[max_seq_len, max_seq_len + comp_capacity)``    the compressor's row for slot ``c``

        The window region deliberately is *not* a ring. The op stores every row of the step
        before it attends, so a chunk of ``S > 1`` tokens would overwrite rows its own earlier
        tokens still need. Positions are absolute instead and the block table recycles the
        physical blocks behind them, which costs ``max_seq_len / block_size`` int32 per sequence
        and nothing else.

        The compressor's rows are appended to ``key`` beyond ``token_count`` (paged_attention.md
        section 12.11): they are stored at their ``slot_mapping`` slots and are already visible to
        this step's ``kv_indices``, which is what the model needs and what a chained ``ScatterND``
        could not give, since ``key_cache`` may be touched by exactly one node.
        """
        p = f"layers.{layer_id}.attn"
        H, D, W, L = self.n_heads_local, self.head_dim, self.window, self.max_seq_len
        I, I32, BOOL = ir.DataType.INT64, ir.DataType.INT32, ir.DataType.BOOL
        io = self.io_dtype
        bsize = self.block_size
        table = f"block_table_{ratio}"
        table64 = self.cast(f"{name}/bt64", table, I, [B, self.group_blocks(ratio)])

        def to_slot(tag, logical, shape, rows):
            """Logical position -> flat cache slot, through this group's block table."""
            blk = self.batched_gather(
                f"{name}/{tag}/blk", table64,
                self.op("Div", [logical, self.const("INT64", bsize)], f"{name}/{tag}/bi", I, shape),
                rows, I, shape)
            return self.op("Add",
                           [self.op("Mul", [blk, self.const("INT64", bsize)],
                                    f"{name}/{tag}/base", I, shape),
                            self.op("Mod", [logical, self.const("INT64", bsize)],
                                    f"{name}/{tag}/off", I, shape)],
                           f"{name}/{tag}/slot", I, shape)

        # ---- one stored KV row per token, at its own absolute position ----
        keys = [self.reshape(f"{name}/kpack", kv, [-1, D], io, [None, D])]
        slots = [self.reshape(f"{name}/tokslot", to_slot("tok", ctx["pos"], [B, S], S),
                              [-1], I, [None])]

        # ---- selection: the sliding window, as absolute positions ----
        jw = self.init(torch.arange(W, dtype=torch.int64), f"paged/arange_{W}")
        win = self.op("Add", [self.op("Sub", [ctx["qpos"], self.const("INT64", W - 1)],
                                      f"{name}/wbase", I, [B, S, 1]), jw],
                      f"{name}/win", I, [B, S, W])
        # Positions above qpos cannot occur (the offsets stop at W-1); negative ones are the
        # not-yet-full window and drop out as -1.
        sel = [self.op("Where", [self.op("GreaterOrEqual", [win, self.const("INT64", 0)],
                                         f"{name}/wge", BOOL, [B, S, W]),
                                 win, self.const("INT64", -1)],
                       f"{name}/winsel", I, [B, S, W])]
        presents = {}

        if ratio:
            rows, first_slot, last_slot, J, _J_g, pkv, psc = self.make_compressor(
                f"{name}/comp", layer_id, ratio, x, ctx)
            C = self.comp_capacity(ratio)
            presents.update(cstate_kv=pkv, cstate_score=psc)

            # ---- surplus KV rows: the compressor's J candidate rows for this step ----
            jr = self.op("Range", [self.const("INT64", 0), J, self.const("INT64", 1)],
                         f"{name}/jr", I, [None])
            c = self.op("Add", [jr, first_slot], f"{name}/c", I, [B, None])
            live = self.op("LessOrEqual", [c, last_slot], f"{name}/clive", BOOL, [B, None])
            # Clamped so a dead row still indexes the table in range; the -1 slot suppresses it.
            clog = self.op("Clip", [self.op("Add", [c, self.const("INT64", L)],
                                            f"{name}/clog", I, [B, None]),
                                    self.const("INT64", L), self.const("INT64", L + C - 1)],
                           f"{name}/clogc", I, [B, None])
            cslot = self.op("Where", [live, to_slot("crow", clog, [B, None], None),
                                      self.const("INT64", -1)],
                            f"{name}/cslot", I, [B, None])
            keys.append(self.reshape(f"{name}/rpack", rows, [-1, D], io, [None, D]))
            slots.append(self.reshape(f"{name}/cslotf", cslot, [-1], I, [None]))

            # ---- selection: which compressed rows this token may look at ----
            if self.has_indexer(ratio):
                csel, ipres = self.make_indexer(f"{name}/idx", layer_id, ratio, x, qr,
                                                cos_q, sin_q, ctx)
                presents.update(ipres)
            else:
                # Ratio-128 layers have no indexer: the model attends to all their valid rows.
                cidx = self.init(torch.arange(C, dtype=torch.int64), f"comp/arange_{C}")
                qpr = self.op("Div", [self.op("Add", [ctx["qpos"], self.const("INT64", 1)],
                                              f"{name}/qp1", I, [B, S, 1]),
                                      self.const("INT64", ratio)], f"{name}/qpr", I, [B, S, 1])
                csel = self.op(
                    "Where",
                    [self.op("Less", [self.unsq(f"{name}/cidx2", cidx, [0, 1], I, [1, 1, C]), qpr],
                             f"{name}/cmask", BOOL, [B, S, C]),
                     self.op("Add", [cidx, self.const("INT64", L)], f"{name}/cabs", I, [C]),
                     self.const("INT64", -1)],
                    f"{name}/csel", I, [B, S, C])
            sel.append(csel)

        width = W + (self.index_width(ratio) if ratio else 0)
        key = (keys[0] if len(keys) == 1 else
               self.op("Concat", keys, f"{name}/key", io, [None, D], axis=0))
        slot = (slots[0] if len(slots) == 1 else
                self.op("Concat", slots, f"{name}/slot", I, [None], axis=0))
        kvi = (sel[0] if len(sel) == 1 else
               self.op("Concat", sel, f"{name}/sel", I, [B, S, width], axis=-1))

        node = f"{name}/paged"
        out, cache_out = f"{node}/output_0", f"{node}/cache_0"
        self.make_node(
            "PagedAttention",
            inputs=[
                self.reshape(f"{name}/qpack", q, [-1, H * D], io, [None, H * D]),
                key,
                "",                                  # value: absent, V is K
                f"past_kv_{layer_id}",
                "",                                  # value_cache: absent
                ctx["cum"],
                ctx["past32"],
                table,
                "", "",                              # cos/sin: RoPE is applied in the graph
                self.cast(f"{name}/slot32", slot, I32, [None]),
                self.init(self.shard(self.sd[f"{p}.attn_sink"], 0).reshape(H),
                          f"{p}.attn_sink", to=io),
                "", "", "", "", "",                  # q/k norm, k/v scale, attention_metadata
                self.cast(f"{name}/kvi32",
                          self.reshape(f"{name}/kvi", kvi, [-1, width], I, [None, width]),
                          I32, [None, width]),
            ],
            outputs=[out, cache_out],
            name=node,
            domain="com.microsoft",
            num_heads=H,
            kv_num_heads=1,
            kv_cache_layout="LATENT",
            v_head_size=D,
        )
        # softmax_scale is left at the op's default 1/sqrt(head_size), which is exactly
        # DeepSeek's `self.softmax_scale = head_dim ** -0.5`.
        self.make_value(out, io, shape=[None, H * D])
        self.make_value(cache_out, io, shape=self.input_shapes[f"past_kv_{layer_id}"])
        presents["kv"] = cache_out
        return self.dyn_reshape(f"{name}/o4", out, ctx["bs"], [H, D], io, [B, S, H, D]), presents

    # ------------------------------------------------------------------ #
    # MoE
    # ------------------------------------------------------------------ #

    def _swiglu(self, name, gate, up, shape):
        lim = self.swiglu_limit
        F = ir.DataType.FLOAT
        if lim and lim > 0:
            up = self.op("Clip", [up, self.const("FLOAT", -lim), self.const("FLOAT", lim)],
                         f"{name}/clipu", F, shape)
            gate = self.op("Clip", [gate, "", self.const("FLOAT", lim)], f"{name}/clipg", F, shape)
        s = self.op("Sigmoid", [gate], f"{name}/sig", F, shape)
        g = self.op("Mul", [gate, s], f"{name}/silu", F, shape)
        return self.op("Mul", [g, up], name, F, shape)

    def make_routing(self, name, layer_id, xf, nvec, ctx):
        """sqrtsoftplus scores + noaux_tc selection bias (or hash routing).

        Returns (idx [N, k] int64, weights [N, k] FLOAT already normalised, i.e.
        summing to 1 before ``routed_scaling_factor``).
        """
        p = f"layers.{layer_id}.ffn"
        E, k = self.n_experts, self.topk
        F, I = ir.DataType.FLOAT, ir.DataType.INT64
        gw = self.init(self.sd[f"{p}.gate_weight"].T, f"{p}.gate_weight/T",
                       to=ir.DataType.FLOAT)
        scores = self.op("MatMul", [xf, gw], f"{name}/scores", F, [None, E])
        sp = self.op("Softplus", [scores], f"{name}/softplus", F, [None, E])
        orig = self.op("Sqrt", [sp], f"{name}/sqrt", F, [None, E])

        if layer_id < self.n_hash_layers:
            flat_ids = self.reshape(f"{name}/ids", "input_ids", [-1], I, [None])
            idx = self.op("Gather", [self.init_w(f"{p}.tid2eid"), flat_ids],
                          f"{name}/idx", I, [None, k], axis=0)
        else:
            bias = self.init_w(f"{p}.gate_bias")
            sel = self.op("Add", [orig, bias], f"{name}/sel", F, [None, E])
            vals = f"{name}/topk/output_0"
            idx = f"{name}/topk/output_1"
            self.make_node("TopK", inputs=[sel, self.const("INT64", [k])], outputs=[vals, idx],
                           name=f"{name}/topk", axis=-1, largest=1, sorted=1)
            self.make_value(vals, F, shape=[None, k])
            self.make_value(idx, I, shape=[None, k])

        w = self.op("GatherElements", [orig, idx], f"{name}/w", F, [None, k], axis=1)
        wsum = self.op("ReduceSum", [w, self.const("INT64", [-1])], f"{name}/wsum",
                       F, [None, 1], keepdims=1)
        wn = self.op("Div", [w, wsum], f"{name}/wn", F, [None, k])
        return idx, wn

    def make_moe(self, name, layer_id, x, ctx):
        p = f"layers.{layer_id}.ffn"
        E, k, dim, mi = self.n_experts, self.topk, self.dim, self.moe_inter
        F, I = ir.DataType.FLOAT, ir.DataType.INT64
        io = self.io_dtype

        xflat = self.reshape(f"{name}/flat", x, [-1, dim], io, [None, dim])
        xf = self.cast(f"{name}/flatf", xflat, F, [None, dim])
        idx, wn = self.make_routing(f"{name}/route", layer_id, xf, None, ctx)

        nrow = self.op("Shape", [xflat], f"{name}/nshape", I, [1], start=0, end=1)
        eshape = self.op("Concat", [nrow, self.const("INT64", [E])],
                         f"{name}/eshape", I, [2], axis=0)

        # Expert parallel: this rank owns experts [lo, lo + E_loc).  It sees only
        # the matching column block of the router, so the kernel's softmax hands
        # back w_e / W_local; multiplying the rank output by W_local and then
        # all-reducing recovers sum_e w_e * E_e(x) exactly.  Tokens with no local
        # expert get an all -inf row, whose degenerate uniform softmax is
        # annihilated by W_local == 0.
        E_loc, lo = self.n_experts_local, self.expert_lo
        zeros = self.op("ConstantOfShape", [eshape], f"{name}/zeros", F, [None, E],
                        value=ir.tensor([0.0], dtype=ir.DataType.FLOAT))
        dense_w = self.op("ScatterElements", [zeros, idx, wn], f"{name}/densew", F,
                          [None, E], axis=1)

        if self.moe_impl == "qmoe":
            # QMoE selects its own top-k from `router_probs` via softmax. Feeding
            # log(weight) on the selected experts and -inf elsewhere makes that
            # softmax reproduce the reference weights exactly.
            logw = self.op("Log", [wn], f"{name}/logw", F, [None, k])
            base = self.op("ConstantOfShape", [eshape], f"{name}/negbase", F, [None, E],
                           value=ir.tensor([NEG_INF], dtype=ir.DataType.FLOAT))
            router = self.op("ScatterElements", [base, idx, logw], f"{name}/router",
                             F, [None, E], axis=1)
            if self.world > 1:
                router = self.slice1(f"{name}/router_loc", router, lo, lo + E_loc,
                                     F, [None, E_loc])
            router = self.cast(f"{name}/routerb", router, io, [None, E_loc])
            fc1_w, fc1_s, fc2_w, fc2_s = self.moe_qweights(p)
            out = f"{name}/qmoe/output_0"
            inputs = [xflat, router,
                      fc1_w, fc1_s, "",
                      fc2_w, fc2_s, "",
                      "", "", "", "", "", "", "",
                      self.g_moe(p, "fc1"), self.g_moe(p, "fc2")]
            self.make_node("QMoE", inputs=inputs, outputs=[out], name=f"{name}/qmoe",
                           domain="com.microsoft", activation_type="swiglu",
                           expert_weight_bits=4, k=min(k, E_loc), normalize_routing_weights=0,
                           swiglu_fusion=1, swiglu_limit=self.swiglu_limit,
                           activation_alpha=1.0, activation_beta=0.0,
                           use_sparse_mixer=0, quant_type="fp4", block_size=32)
            self.make_value(out, io, shape=[None, dim])
            y = self.cast(f"{name}/qmoef", out, F, [None, dim])
            dw_loc = (self.slice1(f"{name}/dwloc", dense_w, lo, lo + E_loc, F, [None, E_loc])
                      if self.world > 1 else dense_w)
            wloc = self.op("ReduceSum", [dw_loc, self.const("INT64", [-1])], f"{name}/wloc",
                           F, [None, 1], keepdims=1)
            wloc = self.op("Mul", [wloc, self.const("FLOAT", float(self.route_scale))],
                           f"{name}/wlocs", F, [None, 1])
            y = self.op("Mul", [y, wloc], f"{name}/scaled", F, [None, dim])
        else:
            gate_w = self.op("Mul", [dense_w, self.const("FLOAT", float(self.route_scale))],
                             f"{name}/gate", F, [None, E])
            if self.world > 1:
                gate_w = self.slice1(f"{name}/gate_loc", gate_w, lo, lo + E_loc, F, [None, E_loc])
            w1 = self.init(self.eshard(self.sd[f"{p}.w1"], 0).float(), f"{p}.w1")
            w2 = self.init(self.eshard(self.sd[f"{p}.w2"], 0).float(), f"{p}.w2")
            w3 = self.init(self.eshard(self.sd[f"{p}.w3"], 0).float(), f"{p}.w3")
            h = self.op("Einsum", [xf, w1], f"{name}/h", F, [None, E_loc, mi], equation="nd,eid->nei")
            u = self.op("Einsum", [xf, w3], f"{name}/u", F, [None, E_loc, mi], equation="nd,eid->nei")
            act = self._swiglu(f"{name}/act", h, u, [None, E_loc, mi])
            gu = self.unsq(f"{name}/gu", gate_w, [-1], F, [None, E_loc, 1])
            act = self.op("Mul", [act, gu], f"{name}/gated", F, [None, E_loc, mi])
            y = self.op("Einsum", [act, w2], f"{name}/y", F, [None, dim], equation="nei,edi->nd")

        # shared expert, tensor-parallel over the intermediate dim
        mil = self.moe_inter_local
        sg = self.proj(f"{name}/sg", xflat, f"{p}.sw1.weight", [None, mil], shard_axis=0)
        su = self.proj(f"{name}/su", xflat, f"{p}.sw3.weight", [None, mil], shard_axis=0)
        sact = self._swiglu(f"{name}/sact",
                            self.cast(f"{name}/sgf", sg, F, [None, mil]),
                            self.cast(f"{name}/suf", su, F, [None, mil]), [None, mil])
        sy = self.proj(f"{name}/sy", self.cast(f"{name}/sactb", sact, io, [None, mil]),
                       f"{p}.sw2.weight", [None, dim], shard_axis=1)
        y = self.op("Add", [y, self.cast(f"{name}/syf", sy, F, [None, dim])],
                    f"{name}/total", F, [None, dim])
        y = self.cast(f"{name}/tob", y, io, [None, dim])
        # One collective covers both the expert-parallel and the shared-expert split.
        y = self.all_reduce(f"{name}/ar", y, io, [None, dim])
        return self.dyn_reshape(f"{name}/out", y, ctx["bs"], [dim], io, [B, S, dim])

    def moe_qweights(self, prefix):
        """Initializer names for ``(fc1_w, fc1_scales, fc2_w, fc2_scales)`` in QMoE fp4 layout.

        The CUDA kernel has no separate fc3 GEMM, so gate and up must arrive
        pre-concatenated in fc1; interleaving them along the output dim is
        ``swiglu_fusion=1``. Interleaving commutes with MXFP4 (blocks run along
        the input dim), so a checkpoint that is already quantised can supply
        ``fc1_q``/``fc1_s`` directly and skip the round trip.
        """
        names = [f"{prefix}.fc1_q", f"{prefix}.fc1_s", f"{prefix}.fc2_q", f"{prefix}.fc2_s"]
        if names[0] in self.values:
            return names
        if names[0] in self.sd:
            tensors = [self.eshard(self.sd[n], 0) for n in names]
        else:
            w1, w3, w2 = (self.eshard(self.sd[f"{prefix}.w{i}"], 0).float() for i in (1, 3, 2))
            fc1 = torch.stack([w1, w3], dim=2).reshape(w1.shape[0], -1, w1.shape[2])
            b1, s1 = mxfp4_quantize(fc1)
            b2, s2 = mxfp4_quantize(w2)
            tensors = [pack_for_qmoe(b1), s1, pack_for_qmoe(b2), s2]
        self.init(tensors[0], names[0])
        self.scale_init(names[1], tensors[1])
        self.init(tensors[2], names[2])
        self.scale_init(names[3], tensors[3])
        return names

    def scale_init(self, name, tensor):
        """Register a uint8 tensor reinterpreted as FLOAT8E8M0 block scales."""
        if name in self.values:
            return name
        u8 = tensor.detach().cpu().contiguous().view(torch.uint8)
        if self.writer is not None:
            return self.init(self.writer.add(u8, name, dtype=ir.DataType.FLOAT8E8M0), name)
        t = ir.Tensor(u8.numpy(), dtype=ir.DataType.FLOAT8E8M0, name=name)
        v = self.make_value(name, ir.DataType.FLOAT8E8M0, t.shape)
        v.const_value = t
        self.model.graph.register_initializer(v)
        return name

    def g_moe(self, prefix, proj):
        return self.init(torch.ones(self.n_experts_local, dtype=torch.float32),
                         f"{prefix}.{proj}_g")

    # ------------------------------------------------------------------ #
    # block / model
    # ------------------------------------------------------------------ #

    def make_block(self, layer_id, h, y, post, comb, ctx):
        """``y``, ``post`` and ``comb`` come from the pre mix the previous block already ran.

        Returns the same carried triple for the next block; the last block has no following
        pre mix, so it ends on a bare post and returns ``None`` for the three.
        """
        n = f"/layers.{layer_id}"
        bs = [B, S]
        p = f"layers.{layer_id}"

        a, presents = self.make_attention(f"{n}/attn", layer_id, y, ctx)
        h, post, comb, y = self.make_hc_mix(
            f"{n}/hc_attn_post", a, h, post, comb,
            (f"{n}/hc_ffn", f"{n}/ffn_norm", f"{p}.hc_ffn", f"{p}.ffn_norm.weight"), bs)

        f_ = self.make_moe(f"{n}/ffn", layer_id, y, ctx)
        nxt = layer_id + 1
        if nxt < self.num_layers:
            h, post, comb, y = self.make_hc_mix(
                f"{n}/hc_ffn_post", f_, h, post, comb,
                (f"/layers.{nxt}/hc_attn", f"/layers.{nxt}/attn_norm",
                 f"layers.{nxt}.hc_attn", f"layers.{nxt}.attn_norm.weight"), bs)
        else:
            h = self.make_hc_post(f"{n}/hc_ffn_post", f_, h, post, comb, bs)
            y = post = comb = None
        return h, y, post, comb, presents

    def make_dsv4_graph(self):
        F, I = ir.DataType.FLOAT, ir.DataType.INT64
        io = self.io_dtype
        hc, dim = self.hc, self.dim

        self.make_inputs_and_outputs()
        self.make_rope_tables()

        bs = self.op("Shape", ["input_ids"], "/shape", I, [2])
        Sv = self.op("Gather", [bs, self.const("INT64", 1)], "/seqlen", I, [], axis=0)
        rng = self.op("Range", [self.const("INT64", 0), Sv, self.const("INT64", 1)],
                      "/range", I, [S])
        # The same sequence length again, but on the device.  `Sv` comes off `Shape`, so
        # ORT keeps it and everything derived from it on the CPU; any device op that
        # consumes one of those scalars as *data* then needs a MemcpyFromHost, and a
        # single Memcpy node anywhere in the graph disqualifies the whole session from
        # CUDA Graph capture.  `rng` is a real device tensor -- Range takes CPU scalars
        # but does not produce one -- so reducing it recovers the length on the GPU for
        # two extra nodes.  Scalars that feed shapes or another Range keep using `S`.
        Sg = self.op("Add", [self.op("ReduceMax", [rng, self.const("INT64", [0])],
                                     "/seqmaxg", I, [], keepdims=0),
                             self.const("INT64", 1)], "/seqleng", I, [])
        # Every cache schedule below is per sequence, so each quantity derived from
        # past_lens is kept at the rank the consumer needs to broadcast against.
        past2 = self.unsq("/past2", "past_lens", [1], I, [B, 1])
        past3 = self.unsq("/past3", past2, [2], I, [B, 1, 1])
        pos = self.op("Add", [rng, past2], "/pos", I, [B, S])
        qpos = self.unsq("/qpos", pos, [2], I, [B, S, 1])
        total2 = self.op("Add", [past2, Sg], "/total", I, [B, 1])
        total3 = self.unsq("/total3", total2, [2], I, [B, 1, 1])
        ctx = {"bs": bs, "S": Sv, "Sg": Sg, "pos": pos, "qpos": qpos, "past2": past2,
               "past3": past3, "total2": total2, "total3": total3}

        if self.paged:
            # PagedAttention speaks a packed [token_count] layout. input_ids is [B, S], so every
            # sequence contributes the same S tokens and the packing is just a reshape: the
            # cumulative lengths are 0, S, 2S, ...
            bv = self.op("Gather", [bs, self.const("INT64", 0)], "/batch", I, [], axis=0)
            tot = self.op("Mul", [self.op("Add", [bv, self.const("INT64", 1)], "/b1", I, []), Sv],
                          "/cumend", I, [])
            cum = self.op("Range", [self.const("INT64", 0), tot, Sv], "/cumrange", I, [None])
            ctx["cum"] = self.cast("/cum32", cum, ir.DataType.INT32, [None])
            ctx["past32"] = self.cast("/past32", "past_lens", ir.DataType.INT32, [B])

        emb = self.op("Gather", [self.init_w("embed.weight", to=io),
                                 "input_ids"], "/embed", io, [B, S, dim], axis=0)
        h = self.unsq("/embed_u", emb, [2], io, [B, S, 1, dim])
        tail = self.const("INT64", [hc, dim])
        eshape = self.op("Concat", [bs, tail], "/hc_expand_shape", I, [4], axis=0)
        h = self.op("Expand", [h, eshape], "/hc_expand", io, [B, S, hc, dim])

        # Every other pre mix is fused into the post mix that precedes it; the first block has
        # no preceding post, so it keeps the unrolled pre.
        y, post, comb = self.make_hc_pre("/layers.0/hc_attn", h, "layers.0.hc_attn", [B, S])
        y = self.make_hc_norm("/layers.0/attn_norm", y, "layers.0.attn_norm.weight", [B, S])

        for layer_id in range(self.num_layers):
            h, y, post, comb, presents = self.make_block(layer_id, h, y, post, comb, ctx)
            for key, val in presents.items():
                out = f"present_{key}_{layer_id}"
                self.make_node("Identity", inputs=[val], outputs=[out], name=f"/out/{out}")

        h = self.make_hc_head("/hc_head", h, "hc_head", [B, S])
        n = self.make_rmsnorm("/norm", h, self.init_w("norm.weight",
                                                      to=ir.DataType.FLOAT),
                              [B, S, dim])
        n = self.cast("/norm_b", n, io, [B, S, dim])
        # Keep the vocabulary projection in the activation type; a float copy of
        # it would be the single largest initializer in the model.
        lg = self.op("MatMul", [n, self.init(self.sd["head.weight"].T, "head/T", to=io)],
                     "/lm_head", io, [B, S, self.vocab_size])
        self.make_node("Cast", inputs=[lg], outputs=["logits"], name="/logits",
                       to=int(ir.DataType.FLOAT))

    def save_model(self, out_dir):
        if self.writer is None:
            return super().save_model(out_dir)

        # The initializers already live in the external-data blob, so this only
        # has to serialize the graph -- no second pass over the weights.
        print(f"Saving ONNX model in {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        self.writer.move_to(out_dir)
        self.model.graph.sort()
        out_path = os.path.join(out_dir, self.filename)
        if os.path.exists(out_path):
            os.remove(out_path)
        ir.save(self.model, out_path)
        if os.path.isdir(self.cache_dir) and not os.listdir(self.cache_dir):
            os.rmdir(self.cache_dir)

    def make_model(self, input_path):
        if not self.sd:
            raise ValueError("state dict must be populated (see load_dsv4_weights) before building")
        self.make_dsv4_graph()
