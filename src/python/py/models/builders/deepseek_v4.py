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
# The drafter's second sequence axis: how many positions the target accepted since
# the last draft.  Its block axis is `S`, which is `dspark_block_size` there.
MAIN = "main_len"


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

        # `wkv`/`wgate` have no `.scale` in the checkpoint, so they arrive as plain bf16 and the
        # fp32 projections were an upcast of the activations, a widened GEMM and a Cast per site
        # for no precision anyone chose. The operator widens what it reads anyway.
        self.comp_bf16 = extra_options.get("dsv4_comp_bf16", "1") not in ("0", 0, False)

        # Likewise for what is left of the indexer once its compressor is fused: a rotation, a
        # cache refresh, a scoring einsum over every cached row and a top-k, ~60 nodes on each
        # of the 21 ratio-4 layers. One operator covers everything after its two projections.
        self.index_fused = extra_options.get("dsv4_index_fused", "1") not in ("0", 0, False)

        # The attention path spends 66 nodes per layer on plumbing that never leaves a token:
        # two RMS norms, two partial rotations, the FP8 round trip on the latent KV row, the
        # inverse rotation on the way out and the regrouping for the output projection.  Two
        # operators cover all of it -- one on each side of the attention kernel.
        self.qkv_fused = extra_options.get("dsv4_qkv_fused", "1") not in ("0", 0, False)

        # The routing decision reads a token's own 256 gate scores and writes back its six
        # weights, but the graph spells it as nineteen nodes: sqrtsoftplus, a top-k, a
        # normalisation and then two scatters into dense expert-wide rows purely to hand QMoE
        # its log-domain router input and to recover this rank's share of the weight.
        self.route_fused = extra_options.get("dsv4_route_fused", "1") not in ("0", 0, False)

        # The shared expert's gated activation is eight elementwise passes over the same
        # buffer, one of which exists only to widen to float and one only to narrow back.
        self.swiglu_fused = extra_options.get("dsv4_swiglu_fused", "1") not in ("0", 0, False)

        # Sibling projections that read the same activation can share one GEMV, the way
        # vLLM folds them through ``stacked_params_mapping``.  It halves their kernel time
        # (a 1024+512 pair over K=4096 goes 16.0 -> 8.4 us, a 256+256 pair 15.3 -> 7.9 us),
        # but the ``Split`` that hands the halves back costs more than that in exposed
        # CUDA-graph node dispatch: measured 82.5 -> 78.6 tok/s at batch 1.  Off by default,
        # kept selectable because the win reappears for any consumer that can read the
        # fused tensor directly and skip the split.  See
        # dev/docs/memory/dsv4_perf_it13_sibling_gemm_fusion.md.
        self.proj_fusion = extra_options.get("dsv4_proj_fused", "0") not in ("0", 0, False)

        # The shared expert is the one pair whose consumer can read the fused tensor: with
        # `up` omitted, `DSV4SwiGLU` slices the `[.., 2 * mil]` projection itself.  That
        # makes this fusion *remove* a node per layer instead of relocating one, which is
        # the shape it13 established as the only one that pays -- 322 fp8 nodes -> 279,
        # target verify 14.83 -> 14.55 ms, decode 252.5 -> 256.6 tok/s.  It needs an ORT
        # whose DSV4SwiGLU schema takes one input.
        self.swiglu_proj_fusion = (
            extra_options.get("dsv4_swiglu_proj_fused", "1") not in ("0", 0, False))

        # The vocabulary projection is the one weight that was left replicated: every rank
        # holds all 4096 x 129280 of it and computes the same logits from the same
        # activation.  At decode that is a pure weight-streaming GEMV -- 1.06 GB at 4.8 TB/s,
        # measured 238.6 us of a 9.6 ms step, the single most expensive kernel launch in the
        # graph -- and seven eighths of the work is redundant.  Sharding the vocabulary and
        # gathering the pieces back costs one collective on 129280 floats.  See
        # dev/docs/memory/dsv4_perf_it15_lm_head_shard.md.
        self.lm_head_shard = extra_options.get("dsv4_lm_head_shard", "1") not in ("0", 0, False)

        # Precision of the vocabulary projection and the Markov head.  bf16 keeps 7
        # mantissa bits, so every logit carries ~0.4% relative error; on logits of
        # magnitude ~20 that is ~0.08 absolute, enough to flip an argmax on a near-tie
        # across a 129280-wide vocabulary and cost acceptance.  fp16 carries 10 bits at
        # the same 2 bytes per element, so it is free in both memory and bandwidth.
        # These knobs deliberately stop at the head: the body cannot run in fp16 (its
        # later-layer activations overflow the 65504 range), but the head is downstream
        # of the final norm and its logits stay within ~1e2, so fp16 is safe there.
        # ``dsv4_markov_dtype`` defaults to ``dsv4_head_dtype`` but can be raised on its
        # own: the two Markov tensors are 66 MiB each against 1.06 GB for head.weight.
        _hd = {"io": self.io_dtype, "fp16": ir.DataType.FLOAT16, "fp32": ir.DataType.FLOAT}
        self.head_dtype = _hd[extra_options.get("dsv4_head_dtype", "io")]
        self.markov_dtype = _hd[extra_options.get("dsv4_markov_dtype",
                                                  extra_options.get("dsv4_head_dtype", "io"))]
        self.mtp_sample_top_k = int(extra_options.get("dsv4_mtp_sample_top_k", 0) or 0)
        if self.mtp_sample_top_k < 0:
            raise ValueError("dsv4_mtp_sample_top_k must be non-negative")
        self.mtp_sample_top_p = float(extra_options.get("dsv4_mtp_sample_top_p", 1.0))
        self.mtp_sample_temperature = float(
            extra_options.get("dsv4_mtp_sample_temperature", 1.0))
        if not 0.0 < self.mtp_sample_top_p <= 1.0:
            raise ValueError("dsv4_mtp_sample_top_p must be in (0, 1]")
        if self.mtp_sample_temperature <= 0.0:
            raise ValueError("dsv4_mtp_sample_temperature must be positive")
        if self.mtp_sample_top_k:
            self.graph.opset_imports["com.microsoft.genai"] = 1

        # Two knobs that drain the float32 island the reference implementation leaves
        # in the decode path.  Each cast there is a full kernel launch on a [1, dim]
        # row, so they cost far more in dispatch than in arithmetic; see
        # dev/docs/memory/dsv4_perf_it16_fp32_island.md.
        self.norm_bf16 = extra_options.get("dsv4_norm_bf16", "1") not in ("0", 0, False)
        self.moe_combine_bf16 = extra_options.get("dsv4_moe_combine_bf16", "1") not in ("0", 0, False)

        # DSpark drafts from the target model's own hidden states rather than from a
        # separate small model: `mtp.0.main_proj` consumes the hc-mean of a few late
        # layers, concatenated in layer order (inference/model.py:920).  The target
        # graph is the only place those states exist, so it has to hand them out.  Off
        # by default -- an unused [B, S, 12288] output is pure cost when no drafter is
        # attached, and every published decode number was measured without it.
        self.mtp = extra_options.get("dsv4_mtp", "0") not in ("0", 0, False)
        self.mtp_target_layers = [i for i in getattr(c, "dspark_target_layer_ids", ())
                                  if i < self.num_layers]

        # `dsv4_graph=mtp` emits the drafter instead of the target.  It is a separate
        # model because the two run on different schedules -- verify then draft, one
        # `Run` each -- so there is nothing to gain from sharing a graph, and keeping
        # them apart leaves the target's measured decode path untouched.
        self.graph_kind = extra_options.get("dsv4_graph", "target")
        if self.graph_kind not in ("target", "mtp"):
            raise ValueError(f"dsv4_graph must be 'target' or 'mtp', got {self.graph_kind!r}")
        # The block width is a runtime width, not a trained one: the drafter fills a
        # block of `K` positions in one pass from "anchor + K-1 noise", and no weight
        # is shaped by `K` (the transformer sees K query rows, the Markov head loops
        # K times over the same two tensors).  `dspark_block_size` is only the width
        # the checkpoint was trained at, and vLLM's speculator rejects widths *below*
        # it while recommending 7 (`vllm/config/speculative.py:1009`).  Override it
        # here rather than editing the checkpoint config.
        self.dspark_block = int(extra_options.get("dsv4_mtp_block", 0) or 0) or \
            int(getattr(c, "dspark_block_size", 0) or 0)
        self.noise_token = int(getattr(c, "dspark_noise_token_id", 0) or 0)
        self.markov_rank = int(getattr(c, "dspark_markov_rank", 0) or 0)
        # `num_nextn_predict_layers` says 1, but the checkpoint carries three full
        # DSpark stages; trust the tensors, not the config.
        self.n_mtp = int(extra_options.get("dsv4_mtp_stages", 3))

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
        if self.graph_kind == "mtp":
            self.filename = "mtp.onnx"
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
                mtp_base=self.num_layers,
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

    def all_reduce(self, name, x, dtype, shape, declared=False):
        """``declared`` marks an output the base builder already registered (``logits``),
        where re-declaring the value would duplicate the graph output."""
        if self.world == 1:
            return x
        self.make_node("AllReduce", inputs=[x], outputs=[name], name=name, domain="com.microsoft")
        if not declared:
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

        if self.graph_kind == "mtp":
            return self._make_mtp_io()

        self.input_names = {"input_ids": "input_ids", "past_lens": "past_lens"}
        self.input_types = {"input_ids": ir.DataType.INT64, "past_lens": ir.DataType.INT64}
        self.input_shapes = {"input_ids": [B, S], "past_lens": [B]}
        self.output_names = {"logits": "logits"}
        self.output_types = {"logits": ir.DataType.FLOAT}
        self.output_shapes = {"logits": [B, S, self.vocab_size]}

        if self.mtp and self.mtp_target_layers:
            self.model.metadata_props["dsv4_mtp_target_layers"] = ",".join(
                str(i) for i in self.mtp_target_layers)
            self.output_names["main_hidden"] = "main_hidden"
            self.output_types["main_hidden"] = self.io_dtype
            self.output_shapes["main_hidden"] = [B, S, self.dim * len(self.mtp_target_layers)]

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
        # The value has to survive a round trip through the initializer's *name*:
        # base.make_constant recovers it with ast.literal_eval on the last path
        # segment.  ir.DataType is an IntEnum whose repr is its member name, so a
        # dtype that reaches here by mistake produces a name that cannot be parsed
        # back -- assert on it rather than letting it become the enum's value.
        if isinstance(value, (list, tuple)):
            bad = [v for v in value if type(v) not in (int, float)]
            assert not bad, f"non-literal constant value {bad} for {dtype}"
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

    def proj_fused(self, name, x, keys, sizes, shapes, shard_axis=None, split=True):
        """One ``x @ concat(W).T`` in place of several, split back afterwards.

        At M == 1 a projection is 1-4 MB, far less than the bytes H200 needs in
        flight to saturate HBM, so each of these launches sits at its latency
        floor rather than at a bandwidth limit -- and the grid is only
        ``ceil(N / 16)`` blocks, well under a wave for the narrow ones.  Siblings
        that read the same activation therefore pay twice for one round trip.
        Concatenating them along N buys back a launch and doubles the blocks;
        vLLM folds the same pairs through ``stacked_params_mapping``.

        The checkpoint's fp8 scales are already expanded to one row per output
        channel, so both the weights and the scales concatenate on axis 0 with
        no alignment constraint on N.

        ``split=False`` hands back the concatenated tensor for a consumer that
        can slice it itself, which is the only shape of this fusion that pays --
        see the class comment on ``proj_fusion``.
        """
        io = self.io_dtype
        qs = [self.sd.qweight(k) if hasattr(self.sd, "qweight") else None for k in keys]
        fused_shape = list(shapes[0][:-1]) + [sum(sizes)]
        if any(q is None for q in qs):
            # Unquantized parity checkpoint: no fp8 layout to concatenate.
            if not split:
                ws = [self.sd[k] for k in keys]
                if shard_axis is not None:
                    ws = [self.shard(w, shard_axis) for w in ws]
                w = torch.cat(ws, 0)
                return self.op("MatMul", [x, self.init(w.T, f"{name}/T", to=io)],
                               f"{name}/fused", io, fused_shape)
            return [self.proj(f"{name}/{i}", x, k, s, shard_axis=shard_axis)
                    for i, (k, s) in enumerate(zip(keys, shapes))]

        ws, ss = [], []
        for w, scale in qs:
            if shard_axis is not None:
                w = self.shard(w, shard_axis)
                scale = self.shard(scale, shard_axis)
            ws.append(w)
            ss.append(scale)

        fused = self.op("MatMulBlockQuantizedFp8Weight",
                        [x,
                         self.init(torch.cat(ws, 0), f"{name}/q"),
                         self.init(torch.cat(ss, 0), f"{name}/s")],
                        f"{name}/fused", io, fused_shape,
                        domain="com.microsoft", block_size=FP8_BLOCK)
        if not split:
            return fused

        outs = [f"{name}/split_{i}" for i in range(len(keys))]
        self.make_node("Split", inputs=[fused, self.const("INT64", list(sizes))],
                       outputs=outs, name=f"{name}/split", axis=-1)
        for out, shape in zip(outs, shapes):
            self.make_value(out, io, shape=shape)
        return outs

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

    def make_rmsnorm(self, name, x, weight, shape, dtype=None):
        """Reference computes in fp32 with an fp32 weight; returns ``dtype``.

        ``dtype`` is the tensor type the node runs in, defaulting to the reference's
        float32.  ``stash_type=1`` keeps the accumulator in float32 whatever the
        tensor type, so asking for the io dtype changes only the rounding of the
        input read, the scale and the output write -- and it drops the pair of casts
        that would otherwise bracket every call.  ``weight`` must already be in
        ``dtype``.
        """
        dtype = ir.DataType.FLOAT if dtype is None else dtype
        xf = x if dtype != ir.DataType.FLOAT else self.cast(f"{name}/castf", x, dtype, shape)
        out = f"{name}/output_0"
        self.make_node("SimplifiedLayerNormalization", inputs=[xf, weight], outputs=[out],
                       name=name, axis=-1, epsilon=self.eps, stash_type=1)
        self.make_value(out, dtype, shape=shape)
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

        # The operator reads the projections once and widens them, so it can take them in the
        # activation type.  The unrolled subgraph below cannot: it is the fp32 reference.
        pdt = self.io_dtype if (self.comp_bf16 and self.comp_fused) else F
        xp = x if pdt == self.io_dtype else self.cast(f"{name}/castx", x, F, [B, S, self.dim])

        if self.comp_fused:
            # Both projections read the same `x` and neither is used anywhere else, so they are
            # one GEMM over a weight stacked along N.  The operator splits the result by stride
            # rather than by a Split node, which would give the saving straight back.
            w = torch.cat([self.sd[f"{p}.wkv.weight"], self.sd[f"{p}.wgate.weight"]], 0)
            kv = self.op("MatMul", [xp, self.init(w.T, f"{p}.wkvgate/T", to=pdt)],
                         f"{name}/kv", pdt, [B, S, 2 * co * d])
            sc = ""
        else:
            kv = self.op("MatMul",
                         [xp, self.init(self.sd[f"{p}.wkv.weight"].T, f"{p}.wkv/T", to=pdt)],
                         f"{name}/kv", pdt, [B, S, co * d])
            sc = self.op("MatMul",
                         [xp, self.init(self.sd[f"{p}.wgate.weight"].T, f"{p}.wgate/T", to=pdt)],
                         f"{name}/sc", pdt, [B, S, co * d])

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

        # The rope tables are gathered at the same positions in every layer of a group.
        rope = ctx.setdefault("rope", {})
        if tbl not in rope:
            r = f"/rope/t{tbl}"
            c = self.op("Gather", [f"rope/cos_{tbl}", ctx["pos"]], f"{r}/cos", F, [B, S, rd],
                        axis=0)
            s = self.op("Gather", [f"rope/sin_{tbl}", ctx["pos"]], f"{r}/sin", F, [B, S, rd],
                        axis=0)
            rope[tbl] = (c, s,
                         self.unsq(f"{r}/cosq", c, [2], F, [B, S, 1, rd]),
                         self.unsq(f"{r}/sinq", s, [2], F, [B, S, 1, rd]))
        cos, sin, cos_q, sin_q = rope[tbl]
        cos_k, sin_k = cos, sin

        def rope_last(tag, t, shape, c, s, inverse=False):
            n = self.slice1(f"{tag}/nope", t, 0, nd, io, shape[:-1] + [nd])
            r_ = self.slice1(f"{tag}/rope", t, nd, INT64_MAX, io, shape[:-1] + [rd])
            r_ = self.cast(f"{tag}/ropef", r_, F, shape[:-1] + [rd])
            r_ = self.make_rope(f"{tag}/rot", r_, c, s, shape[:-1] + [rd], inverse=inverse)
            r_ = self.cast(f"{tag}/ropeb", r_, io, shape[:-1] + [rd])
            return self.op("Concat", [n, r_], tag, io, shape, axis=-1)

        # ---- Q ----
        if self.proj_fusion:
            # wq_a and wkv both read the attention norm, so they can ride one GEMV.
            qa, kv_raw = self.proj_fused(f"{name}/qakv", x,
                                         [f"{p}.wq_a.weight", f"{p}.wkv.weight"],
                                         [self.q_lora_rank, D],
                                         [[B, S, self.q_lora_rank], [B, S, D]])
        else:
            qa = self.proj(f"{name}/qa", x, f"{p}.wq_a.weight", [B, S, self.q_lora_rank])
            kv_raw = self.proj(f"{name}/kv", x, f"{p}.wkv.weight", [B, S, D])
        # Not `nd`: that is the nope dim, and `rope_last` below closes over it, so
        # reusing the name here silently turns every later nope/rope split into a
        # slice at the dtype's enum value.  Only the unfused paths call it late
        # enough to be hit, which is why the fused export never showed it.
        ndt = io if self.norm_bf16 else ir.DataType.FLOAT
        qn = self.make_rmsnorm(f"{name}/qnorm", qa,
                               self.init_w(f"{p}.q_norm.weight", to=ndt),
                               [B, S, self.q_lora_rank], dtype=ndt)
        if ndt != io:
            qn = self.cast(f"{name}/qnorm_b", qn, io, [B, S, self.q_lora_rank])
        q_raw = self.proj(f"{name}/qb", qn, f"{p}.wq_b.weight", [B, S, H * D], shard_axis=0)

        # ---- KV (single latent row shared by all heads) ----
        kv_w = self.init_w(f"{p}.kv_norm.weight", to=ir.DataType.FLOAT)

        if self.qkv_fused:
            outs = [f"{name}/qkv_0", f"{name}/qkv_1"]
            self.make_node(
                "DSV4QKVNormRope",
                inputs=[q_raw, kv_raw, kv_w, cos, sin],
                outputs=outs, name=f"{name}/qkv", domain="com.microsoft",
                num_heads=H, head_dim=D, rope_head_dim=rd, epsilon=self.eps, act_quant=1)
            self.make_value(outs[0], io, shape=[B, S, H, D])
            self.make_value(outs[1], io, shape=[B, S, D])
            q, kv = outs
        else:
            q = self.reshape(f"{name}/q4", q_raw, [0, 0, H, D], io, [B, S, H, D])
            q = self.make_weightless_rmsnorm(f"{name}/qrms", q, [B, S, H, D], io)
            q = rope_last(f"{name}/qrope", q, [B, S, H, D], cos_q, sin_q)

            kv = self.make_rmsnorm(f"{name}/kvnorm", kv_raw, kv_w, [B, S, D])
            kv = self.cast(f"{name}/kvnorm_b", kv, io, [B, S, D])
            kv = rope_last(f"{name}/kvrope", kv, [B, S, D], cos_k, sin_k)
            kv_n = self.slice1(f"{name}/kvn", kv, 0, nd, io, [B, S, nd])
            kv_r = self.slice1(f"{name}/kvr", kv, nd, INT64_MAX, io, [B, S, rd])
            kv_n = self.make_act_quant(f"{name}/kvaq",
                                       self.cast(f"{name}/kvnf", kv_n, F, [B, S, nd]),
                                       [B, S, nd])
            kv_n = self.cast(f"{name}/kvnb", kv_n, io, [B, S, nd])
            kv = self.op("Concat", [kv_n, kv_r], f"{name}/kvq", io, [B, S, D], axis=-1)

        if self.paged:
            # `o` is still flat here -- (tokens, H * D) -- because the fused tail below wants
            # it that way and the unfused tail has to reshape it anyway.
            o, presents = self.make_paged_attention(name, layer_id, ratio, q, kv, x, ctx,
                                                    qr=qn, cos_q=cos_q, sin_q=sin_q)
            if self.qkv_fused:
                o = self.op("DSV4InvRopeGroup", [o, cos, sin], f"{name}/of", io, [G, None, dh],
                            domain="com.microsoft", num_heads=H, head_dim=D, rope_head_dim=rd,
                            num_groups=G)
            else:
                o = self.dyn_reshape(f"{name}/o4", o, ctx["bs"], [H, D], io, [B, S, H, D])
        else:
            o, presents = self.make_dense_attention(name, layer_id, ratio, q, kv, x, ctx)

        if not (self.paged and self.qkv_fused):
            o = rope_last(f"{name}/orope", o, [B, S, H, D], cos_q, sin_q, inverse=True)
            o = self.reshape(f"{name}/og", o, [0, 0, G, -1], io, [B, S, G, dh])
            o = self.op("Transpose", [o], f"{name}/ot", io, [G, B, S, dh], perm=[2, 0, 1, 3])
            o = self.reshape(f"{name}/of", o, [G, -1, dh], io, [G, None, dh])

        # ---- grouped output projection ----
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

    def _paged_group(self, ratio, ctx):
        """The slot mapping and row selection every layer of one cache group shares.

        All of it is a function of ``past_lens``, ``S`` and ``ratio`` alone, so the 43 layers
        collapse onto one of these per distinct ratio.  ``first_slot``/``last_slot`` are
        recomputed here with the compressor kernel's own formulas rather than read off each
        layer's ``DSV4Compressor`` outputs: those are distinct nodes per layer, so everything
        downstream of them is beyond the reach of common subexpression elimination.
        """
        memo = ctx.setdefault("paged_groups", {})
        if ratio in memo:
            return memo[ratio]

        I, I32, BOOL = ir.DataType.INT64, ir.DataType.INT32, ir.DataType.BOOL
        L, W, bsize = self.max_seq_len, self.window, self.block_size
        g = f"/paged/r{ratio}"
        table64 = self.cast(f"{g}/bt64", f"block_table_{ratio}", I,
                            [B, self.group_blocks(ratio)])

        def to_slot(tag, logical, shape, rows):
            """Logical position -> flat cache slot, through this group's block table."""
            blk = self.batched_gather(
                f"{g}/{tag}/blk", table64,
                self.op("Div", [logical, self.const("INT64", bsize)], f"{g}/{tag}/bi", I, shape),
                rows, I, shape)
            return self.op("Add",
                           [self.op("Mul", [blk, self.const("INT64", bsize)],
                                    f"{g}/{tag}/base", I, shape),
                            self.op("Mod", [logical, self.const("INT64", bsize)],
                                    f"{g}/{tag}/off", I, shape)],
                           f"{g}/{tag}/slot", I, shape)

        # ---- one stored KV row per token, at its own absolute position ----
        slots = [self.reshape(f"{g}/tokslot", to_slot("tok", ctx["pos"], [B, S], S),
                              [-1], I, [None])]

        # ---- selection: the sliding window, as absolute positions ----
        # The window does not depend on the group either, so it is shared across all of them.
        if "winsel" not in ctx:
            jw = self.init(torch.arange(W, dtype=torch.int64), f"paged/arange_{W}")
            win = self.op("Add", [self.op("Sub", [ctx["qpos"], self.const("INT64", W - 1)],
                                          "/paged/wbase", I, [B, S, 1]), jw],
                          "/paged/win", I, [B, S, W])
            # Positions above qpos cannot occur (the offsets stop at W-1); negative ones are the
            # not-yet-full window and drop out as -1.
            ctx["winsel"] = self.op(
                "Where", [self.op("GreaterOrEqual", [win, self.const("INT64", 0)],
                                  "/paged/wge", BOOL, [B, S, W]),
                          win, self.const("INT64", -1)],
                "/paged/winsel", I, [B, S, W])
        sel = [ctx["winsel"]]

        if ratio:
            C = self.comp_capacity(ratio)
            # `DSV4CompressorStateKernel`: first = past / ratio, last = (past + S - 1) / ratio,
            # row_count = (S - 1) / ratio + 2.  `J` feeds a Range, so it takes the host-side `S`.
            first_slot = self.op("Div", [ctx["past2"], self.const("INT64", ratio)],
                                 f"{g}/first", I, [B, 1])
            last_slot = self.op("Div", [self.op("Sub", [ctx["total2"], self.const("INT64", 1)],
                                                f"{g}/end", I, [B, 1]),
                                        self.const("INT64", ratio)], f"{g}/last", I, [B, 1])
            jm1 = self.op("Sub", [ctx["S"], self.const("INT64", 1)], f"{g}/jm1", I, [])
            jdiv = self.op("Div", [jm1, self.const("INT64", ratio)], f"{g}/jdiv", I, [])
            J = self.op("Add", [jdiv, self.const("INT64", 2)], f"{g}/J", I, [])

            # ---- surplus KV rows: the compressor's J candidate rows for this step ----
            jr = self.op("Range", [self.const("INT64", 0), J, self.const("INT64", 1)],
                         f"{g}/jr", I, [None])
            c = self.op("Add", [jr, first_slot], f"{g}/c", I, [B, None])
            live = self.op("LessOrEqual", [c, last_slot], f"{g}/clive", BOOL, [B, None])
            # Clamped so a dead row still indexes the table in range; the -1 slot suppresses it.
            clog = self.op("Clip", [self.op("Add", [c, self.const("INT64", L)],
                                            f"{g}/clog", I, [B, None]),
                                    self.const("INT64", L), self.const("INT64", L + C - 1)],
                           f"{g}/clogc", I, [B, None])
            cslot = self.op("Where", [live, to_slot("crow", clog, [B, None], None),
                                      self.const("INT64", -1)],
                            f"{g}/cslot", I, [B, None])
            slots.append(self.reshape(f"{g}/cslotf", cslot, [-1], I, [None]))

            if not self.has_indexer(ratio):
                # Ratio-128 layers have no indexer: the model attends to all their valid rows.
                cidx = self.init(torch.arange(C, dtype=torch.int64), f"comp/arange_{C}")
                qpr = self.op("Div", [self.op("Add", [ctx["qpos"], self.const("INT64", 1)],
                                              f"{g}/qp1", I, [B, S, 1]),
                                      self.const("INT64", ratio)], f"{g}/qpr", I, [B, S, 1])
                sel.append(self.op(
                    "Where",
                    [self.op("Less", [self.unsq(f"{g}/cidx2", cidx, [0, 1], I, [1, 1, C]), qpr],
                             f"{g}/cmask", BOOL, [B, S, C]),
                     self.op("Add", [cidx, self.const("INT64", L)], f"{g}/cabs", I, [C]),
                     self.const("INT64", -1)],
                    f"{g}/csel", I, [B, S, C]))

        slot = (slots[0] if len(slots) == 1 else
                self.op("Concat", slots, f"{g}/slot", I, [None], axis=0))
        group = {"slot32": self.cast(f"{g}/slot32", slot, I32, [None]), "sel": sel}

        # Only the indexed groups still owe a per-layer `csel`; everything else finishes here.
        if not self.has_indexer(ratio):
            width = W + (self.index_width(ratio) if ratio else 0)
            kvi = (sel[0] if len(sel) == 1 else
                   self.op("Concat", sel, f"{g}/sel", I, [B, S, width], axis=-1))
            group["kvi32"] = self.cast(
                f"{g}/kvi32", self.reshape(f"{g}/kvi", kvi, [-1, width], I, [None, width]),
                I32, [None, width])

        memo[ratio] = group
        return group

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
        H, D, W = self.n_heads_local, self.head_dim, self.window
        I, I32 = ir.DataType.INT64, ir.DataType.INT32
        io = self.io_dtype
        table = f"block_table_{ratio}"
        grp = self._paged_group(ratio, ctx)

        # ---- one stored KV row per token, at its own absolute position ----
        keys = [self.reshape(f"{name}/kpack", kv, [-1, D], io, [None, D])]
        presents = {}
        kvi32 = grp.get("kvi32")

        if ratio:
            rows, _first, _last, _J, _J_g, pkv, psc = self.make_compressor(
                f"{name}/comp", layer_id, ratio, x, ctx)
            presents.update(cstate_kv=pkv, cstate_score=psc)
            # The rows themselves are per layer; the slots they land in came from the group.
            keys.append(self.reshape(f"{name}/rpack", rows, [-1, D], io, [None, D]))

            # ---- selection: which compressed rows this token may look at ----
            if self.has_indexer(ratio):
                csel, ipres = self.make_indexer(f"{name}/idx", layer_id, ratio, x, qr,
                                                cos_q, sin_q, ctx)
                presents.update(ipres)
                width = W + self.index_width(ratio)
                kvi = self.op("Concat", grp["sel"] + [csel], f"{name}/sel", I,
                              [B, S, width], axis=-1)
                kvi32 = self.cast(
                    f"{name}/kvi32", self.reshape(f"{name}/kvi", kvi, [-1, width], I,
                                                  [None, width]), I32, [None, width])

        key = (keys[0] if len(keys) == 1 else
               self.op("Concat", keys, f"{name}/key", io, [None, D], axis=0))

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
                grp["slot32"],
                self.init(self.shard(self.sd[f"{p}.attn_sink"], 0).reshape(H),
                          f"{p}.attn_sink", to=io),
                "", "", "", "", "",                  # q/k norm, k/v scale, attention_metadata
                kvi32,
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
        return out, presents

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

    def make_routing(self, name, layer_id, xf, ctx):
        """The gate GEMM, plus the fixed expert choice on the hash-routed layers.

        Returns ``(scores [N, E] FLOAT, expert_ids [N, k] int64 or None)``.  Both
        routing paths below start here: hash routing already knows *which* experts a
        token uses and needs the scores only for their weights, while ``noaux_tc``
        derives the choice from the scores as well.
        """
        p = f"layers.{layer_id}.ffn"
        E, k = self.n_experts, self.topk
        F, I = ir.DataType.FLOAT, ir.DataType.INT64
        gw = self.init(self.sd[f"{p}.gate_weight"].T, f"{p}.gate_weight/T",
                       to=ir.DataType.FLOAT)
        scores = self.op("MatMul", [xf, gw], f"{name}/scores", F, [None, E])
        if layer_id >= self.n_hash_layers:
            return scores, None
        flat_ids = self.reshape(f"{name}/ids", "input_ids", [-1], I, [None])
        return scores, self.op("Gather", [self.init_w(f"{p}.tid2eid"), flat_ids],
                               f"{name}/idx", I, [None, k], axis=0)

    def make_topk_weights(self, name, layer_id, scores, expert_ids):
        """sqrtsoftplus affinities, the noaux_tc selection and the normalised weights.

        Returns (idx [N, k] int64, weights [N, k] FLOAT already normalised, i.e.
        summing to 1 before ``routed_scaling_factor``).
        """
        p = f"layers.{layer_id}.ffn"
        E, k = self.n_experts, self.topk
        F, I = ir.DataType.FLOAT, ir.DataType.INT64
        sp = self.op("Softplus", [scores], f"{name}/softplus", F, [None, E])
        orig = self.op("Sqrt", [sp], f"{name}/sqrt", F, [None, E])

        idx = expert_ids
        if idx is None:
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
        scores, expert_ids = self.make_routing(f"{name}/route", layer_id, xf, ctx)

        # Expert parallel: this rank owns experts [lo, lo + E_loc).  It sees only
        # the matching column block of the router, so the kernel's softmax hands
        # back w_e / W_local; multiplying the rank output by W_local and then
        # all-reducing recovers sum_e w_e * E_e(x) exactly.  Tokens with no local
        # expert get an all -inf row, whose degenerate uniform softmax is
        # annihilated by W_local == 0.
        E_loc, lo = self.n_experts_local, self.expert_lo

        if self.moe_impl == "qmoe" and self.route_fused:
            # One operator for the whole decision.  Everything the unfused chain
            # below builds -- the affinities, the selection, the normalisation, the
            # two dense E-wide scatters and the two reductions over them -- is
            # per-token work over 256 scores, so it collapses into one block per
            # token that never spills the row out of shared memory.
            outs = [f"{name}/router", f"{name}/wlocs"]
            self.make_node(
                "DSV4MoERouter",
                inputs=[scores,
                        "" if expert_ids is not None else self.init_w(f"{p}.gate_bias"),
                        expert_ids if expert_ids is not None else ""],
                outputs=outs, name=f"{name}/route/pick", domain="com.microsoft",
                topk=k, local_expert_start=lo, local_expert_count=E_loc,
                route_scale=float(self.route_scale), dtype=int(io))
            self.make_value(outs[0], io, shape=[None, E_loc])
            self.make_value(outs[1], F, shape=[None, 1])
            router, wloc = outs
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
            # The routed output, the shared expert's output and the all-reduce that
            # follows are all in the io dtype; widening to float32 just to scale by a
            # per-token constant and add one tensor costs three casts of a [1, dim]
            # row per layer, each its own kernel launch.  Scale the [None, 1] weight
            # instead, which is 4096x smaller.
            if self.moe_combine_bf16:
                wl = self.cast(f"{name}/wlocb", wloc, io, [None, 1])
                y = self.op("Mul", [out, wl], f"{name}/scaled", io, [None, dim])
                return self.make_moe_shared(name, layer_id, xflat, y, ctx, io)
            y = self.cast(f"{name}/qmoef", out, F, [None, dim])
            y = self.op("Mul", [y, wloc], f"{name}/scaled", F, [None, dim])
            return self.make_moe_shared(name, layer_id, xflat, y, ctx, F)

        idx, wn = self.make_topk_weights(f"{name}/route", layer_id, scores, expert_ids)

        nrow = self.op("Shape", [xflat], f"{name}/nshape", I, [1], start=0, end=1)
        eshape = self.op("Concat", [nrow, self.const("INT64", [E])],
                         f"{name}/eshape", I, [2], axis=0)
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

        return self.make_moe_shared(name, layer_id, xflat, y, ctx, F)

    def make_moe_shared(self, name, layer_id, xflat, y, ctx, ydtype):
        """The shared expert, added to the routed output ``y`` and all-reduced.

        ``ydtype`` is the type ``y`` arrives in, and the type the sum is formed in.

        Tensor-parallel over the intermediate dim, so one collective at the end
        covers both this split and the expert-parallel one above it.
        """
        p = f"layers.{layer_id}.ffn"
        F = ir.DataType.FLOAT
        io, dim, mil = self.io_dtype, self.dim, self.moe_inter_local

        if self.swiglu_proj_fusion and self.swiglu_fused:
            sgu = self.proj_fused(f"{name}/sgu", xflat,
                                  [f"{p}.sw1.weight", f"{p}.sw3.weight"],
                                  [mil, mil], [[None, mil], [None, mil]],
                                  shard_axis=0, split=False)
            sg, su = sgu, None
        elif self.proj_fusion:
            sg, su = self.proj_fused(f"{name}/sgu", xflat,
                                     [f"{p}.sw1.weight", f"{p}.sw3.weight"],
                                     [mil, mil], [[None, mil], [None, mil]], shard_axis=0)
        else:
            sg = self.proj(f"{name}/sg", xflat, f"{p}.sw1.weight", [None, mil], shard_axis=0)
            su = self.proj(f"{name}/su", xflat, f"{p}.sw3.weight", [None, mil], shard_axis=0)
        if self.swiglu_fused:
            # The activation stays in the io dtype end to end: the operator widens to
            # float internally, so the two casts around it -- and the six elementwise
            # passes between them -- leave the graph.  With one fused projection it also
            # slices the halves itself, so the fusion costs no `Split`.
            sactb = self.op("DSV4SwiGLU", [sg] if su is None else [sg, su], f"{name}/sact",
                            io, [None, mil],
                            domain="com.microsoft", limit=float(self.swiglu_limit or 0.0))
        else:
            sact = self._swiglu(f"{name}/sact",
                                self.cast(f"{name}/sgf", sg, F, [None, mil]),
                                self.cast(f"{name}/suf", su, F, [None, mil]), [None, mil])
            sactb = self.cast(f"{name}/sactb", sact, io, [None, mil])
        sy = self.proj(f"{name}/sy", sactb, f"{p}.sw2.weight", [None, dim], shard_axis=1)
        if ydtype == io:
            y = self.op("Add", [y, sy], f"{name}/total", io, [None, dim])
        else:
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

        main_hiddens = []
        for layer_id in range(self.num_layers):
            h, y, post, comb, presents = self.make_block(layer_id, h, y, post, comb, ctx)
            for key, val in presents.items():
                out = f"present_{key}_{layer_id}"
                self.make_node("Identity", inputs=[val], outputs=[out], name=f"/out/{out}")
            if self.mtp and layer_id in self.mtp_target_layers:
                # The carried `h` is the post-mix residual, which is what the reference
                # appends before the next block runs.  The reduction runs in float
                # because there is no bfloat16 ReduceMean on CUDA and an unplaced node
                # fails session init outright.
                hf = self.cast(f"/mtp/hcf.{layer_id}", h, ir.DataType.FLOAT,
                               [B, S, self.hc, dim])
                mean = self.op("ReduceMean", [hf, self.const("INT64", [2])],
                               f"/mtp/hcmean.{layer_id}", ir.DataType.FLOAT,
                               [B, S, dim], keepdims=0)
                main_hiddens.append(
                    self.cast(f"/mtp/hcb.{layer_id}", mean, io, [B, S, dim]))
        if main_hiddens:
            self.make_node("Concat", inputs=main_hiddens, outputs=["main_hidden"],
                           name="/mtp/main_hidden", axis=-1)

        h = self.make_hc_head("/hc_head", h, "hc_head", [B, S])
        nd = io if self.norm_bf16 else ir.DataType.FLOAT
        n = self.make_rmsnorm("/norm", h, self.init_w("norm.weight", to=nd),
                              [B, S, dim], dtype=nd)
        if nd != io:
            n = self.cast("/norm_b", n, io, [B, S, dim])
        # head.weight is the single largest initializer, so this is the one precision
        # knob that costs real memory: fp16 is free (still 2 bytes), fp32 is +1.06 GB.
        hd = self.head_dtype
        nh = n if hd == io else self.cast("/lm_head_in", n, hd, [B, S, dim])
        shard_head = (self.lm_head_shard and self.world > 1
                      and self.vocab_size % self.world == 0)
        if shard_head:
            # head.weight is [vocab, dim], so a contiguous slice along axis 0 is this
            # rank's vocabulary range.  The pieces are reunified with a zero pad and an
            # AllReduce rather than an AllGather: every output element comes from exactly
            # one rank and the others contribute zero, so the sum is exact, and unlike
            # com.microsoft.AllGather -- which concatenates on axis 0 and reaches any other
            # axis through a pair of transposes and two temporaries -- this needs no
            # reshaping at all.  Measured on this box, a 517 KB fp32 AllReduce is 22.9 us
            # against 41.4 us for the equivalent 64 KB AllGather.
            vloc = self.vocab_size // self.world
            lg = self.op("MatMul", [nh, self.init(self.shard(self.sd["head.weight"], 0).T,
                                                  "head/T", to=hd)],
                         "/lm_head", hd, [B, S, vloc])
            # The cast has to precede the pad: CUDA Pad has no bfloat16 kernel, and an
            # unassigned node fails session initialization outright.  It is also the
            # cheaper order, since the cast then runs on an eighth of the vocabulary.
            lgf = self.cast("/lm_head_f", lg, ir.DataType.FLOAT, [B, S, vloc])
            pads = [0, 0, self.rank * vloc, 0, 0, (self.world - 1 - self.rank) * vloc]
            wide = self.op("Pad", [lgf, self.const("INT64", pads)], "/lm_head_pad",
                           ir.DataType.FLOAT, [B, S, self.vocab_size])
            self.all_reduce("logits", wide, ir.DataType.FLOAT,
                            [B, S, self.vocab_size], declared=True)
        else:
            lg = self.op("MatMul", [nh, self.init(self.sd["head.weight"].T, "head/T", to=hd)],
                         "/lm_head", hd, [B, S, self.vocab_size])
            self.make_node("Cast", inputs=[lg], outputs=["logits"], name="/logits",
                           to=int(ir.DataType.FLOAT))

    # ------------------------------------------------------------------ #
    # DSpark drafter
    # ------------------------------------------------------------------ #

    def _make_mtp_io(self):
        """One draft pass: eat the target's new main states, emit a block of ids.

        ``main_hidden`` carries every position the target accepted since the last draft,
        not just the newest one.  The ring holds one row per accepted position, so
        feeding them one at a time would need a ``Run`` per token and defeat the point;
        taking the whole run at once keeps the cache exact for any accept length.
        """
        W, D, K = self.window, self.head_dim, self.dspark_block
        cache_block = self.block_size
        cache_blocks = (W + K + cache_block - 1) // cache_block
        self.model.metadata_props.update({
            "dsv4_mtp_stages": str(self.n_mtp),
            "dsv4_mtp_block_size": str(K),
            "dsv4_mtp_sample_top_k": str(self.mtp_sample_top_k),
            "dsv4_mtp_sample_top_p": str(self.mtp_sample_top_p),
            "dsv4_mtp_sample_temperature": str(self.mtp_sample_temperature),
            "dsv4_mtp_noise_token": str(self.noise_token),
            "dsv4_window": str(W),
            "dsv4_mtp_cache_block_size": str(cache_block),
        })
        self.input_names = {n: n for n in ("main_hidden", "input_ids", "past_lens")}
        self.input_types = {"main_hidden": self.io_dtype, "input_ids": ir.DataType.INT64,
                            "past_lens": ir.DataType.INT64}
        self.input_shapes = {
            "main_hidden": [B, MAIN, self.dim * len(self.mtp_target_layers)],
            "input_ids": [B, 1], "past_lens": [B]}
        self.output_names = {"output_ids": "output_ids", "confidence": "confidence"}
        self.output_types = {"output_ids": ir.DataType.INT64, "confidence": ir.DataType.FLOAT}
        self.output_shapes = {"output_ids": [B, K + 1], "confidence": [B, K]}
        if self.mtp_sample_top_k:
            Q = self.mtp_sample_top_k
            if Q > self.vocab_size:
                raise ValueError(f"dsv4_mtp_sample_top_k={Q} exceeds vocab size {self.vocab_size}")
            self.input_names["draft_uniform"] = "draft_uniform"
            self.input_types["draft_uniform"] = ir.DataType.FLOAT
            self.input_shapes["draft_uniform"] = [B, K]
            self.output_names.update({"draft_indices": "draft_indices",
                                      "draft_probs": "draft_probs"})
            self.output_types.update({"draft_indices": ir.DataType.INT64,
                                      "draft_probs": ir.DataType.FLOAT})
            self.output_shapes.update({"draft_indices": [B, K, Q],
                                       "draft_probs": [B, K, Q]})
        self.cache_spec = []
        for s in range(self.n_mtp):
            lid = self.num_layers + s
            self.cache_spec.append((lid, "kv"))
            for side, pre in (("input", "past"), ("output", "present")):
                getattr(self, f"{side}_names")[f"{pre}_kv_{lid}"] = f"{pre}_kv_{lid}"
                getattr(self, f"{side}_types")[f"{pre}_kv_{lid}"] = self.io_dtype
                getattr(self, f"{side}_shapes")[f"{pre}_kv_{lid}"] = [
                    cache_blocks, cache_block, 1, D]

    def make_dspark_attention(self, name, lid, x, main_x, ctx):
        """Dense MQA over the window ring of main states plus the draft block itself.

        ``get_dspark_topk_idxs`` (model.py:744) hands every query the same row set --
        every live ring row and every row of the draft block -- so the reference's
        ``sparse_attn`` is dense here, and the block attends to itself in both
        directions rather than causally.
        """
        p = f"layers.{lid}.attn"
        H, D, rd, npd = self.n_heads_local, self.head_dim, self.rope_head_dim, self.nope_dim
        G, R, W = self.o_groups_local, self.o_lora_rank, self.window
        dh = self.n_heads * self.head_dim // self.o_groups
        F = ir.DataType.FLOAT
        io = self.io_dtype
        kv_w = self.init_w(f"{p}.kv_norm.weight", to=F)

        def rope_last(tag, t, shape, c, s, inverse=False):
            n_ = self.slice1(f"{tag}/nope", t, 0, npd, io, shape[:-1] + [npd])
            r_ = self.slice1(f"{tag}/rope", t, npd, INT64_MAX, io, shape[:-1] + [rd])
            r_ = self.cast(f"{tag}/ropef", r_, F, shape[:-1] + [rd])
            r_ = self.make_rope(f"{tag}/rot", r_, c, s, shape[:-1] + [rd], inverse=inverse)
            r_ = self.cast(f"{tag}/ropeb", r_, io, shape[:-1] + [rd])
            return self.op("Concat", [n_, r_], tag, io, shape, axis=-1)

        def kv_tail(tag, t, shape, c, s):
            t = rope_last(f"{tag}/rope", t, shape, c, s)
            n_ = self.slice1(f"{tag}/n", t, 0, npd, io, shape[:-1] + [npd])
            r_ = self.slice1(f"{tag}/r", t, npd, INT64_MAX, io, shape[:-1] + [rd])
            n_ = self.make_act_quant(f"{tag}/aq",
                                     self.cast(f"{tag}/nf", n_, F, shape[:-1] + [npd]),
                                     shape[:-1] + [npd])
            n_ = self.cast(f"{tag}/nb", n_, io, shape[:-1] + [npd])
            return self.op("Concat", [n_, r_], tag, io, shape, axis=-1)

        # ---- the accepted run's main states land in the ring ----
        mkv = self.proj(f"{name}/mkv", main_x, f"{p}.wkv.weight", [B, MAIN, D])
        mkv = self.make_rmsnorm(f"{name}/mkvn", mkv, kv_w, [B, MAIN, D])
        mkv = self.cast(f"{name}/mkvb", mkv, io, [B, MAIN, D])
        mkv = kv_tail(f"{name}/mkv2", mkv, [B, MAIN, D], ctx["mcos"], ctx["msin"])
        # ---- the draft block's own q and kv ----
        qa = self.proj(f"{name}/qa", x, f"{p}.wq_a.weight", [B, S, self.q_lora_rank])
        nd_ = io if self.norm_bf16 else F
        qn = self.make_rmsnorm(f"{name}/qnorm", qa, self.init_w(f"{p}.q_norm.weight", to=nd_),
                               [B, S, self.q_lora_rank], dtype=nd_)
        if nd_ != io:
            qn = self.cast(f"{name}/qnorm_b", qn, io, [B, S, self.q_lora_rank])
        q = self.proj(f"{name}/qb", qn, f"{p}.wq_b.weight", [B, S, H * D], shard_axis=0)
        q = self.reshape(f"{name}/q4", q, [0, 0, H, D], io, [B, S, H, D])
        q = self.make_weightless_rmsnorm(f"{name}/qrms", q, [B, S, H, D], io)
        q = rope_last(f"{name}/qrope", q, [B, S, H, D], ctx["cos_q"], ctx["sin_q"])

        kv = self.proj(f"{name}/kv", x, f"{p}.wkv.weight", [B, S, D])
        kv = self.make_rmsnorm(f"{name}/kvn", kv, kv_w, [B, S, D])
        kv = self.cast(f"{name}/kvb", kv, io, [B, S, D])
        kv = kv_tail(f"{name}/kv2", kv, [B, S, D], ctx["cos"], ctx["sin"])

        key = self.op("Concat", [mkv, kv], f"{name}/keys", io, [B, None, D], axis=1)
        out, cache_out = f"{name}/paged/output_0", f"present_kv_{lid}"
        self.make_node(
            "PagedAttention",
            inputs=[
                self.reshape(f"{name}/qpack", q, [-1, H * D], io, [None, H * D]),
                self.reshape(f"{name}/kpack", key, [-1, D], io, [None, D]),
                "", f"past_kv_{lid}", "", ctx["cum32"], ctx["zero_past32"],
                ctx["block_table32"], "", "", ctx["slot32"],
                self.init(self.shard(self.sd[f"{p}.attn_sink"], 0).reshape(H),
                          f"{p}.attn_sink", to=io),
                "", "", "", "", "", ctx["kvi32"],
            ],
            outputs=[out, cache_out], name=f"{name}/paged", domain="com.microsoft",
            num_heads=H, kv_num_heads=1, kv_cache_layout="LATENT", v_head_size=D)
        self.make_value(out, io, shape=[None, H * D])
        self.make_value(cache_out, io, shape=self.input_shapes[f"past_kv_{lid}"])
        o = self.dyn_reshape(f"{name}/o4", out, ctx["bs"], [H, D], io, [B, S, H, D])

        o = rope_last(f"{name}/orope", o, [B, S, H, D], ctx["cos_q"], ctx["sin_q"], inverse=True)
        o = self.reshape(f"{name}/og", o, [0, 0, G, -1], io, [B, S, G, dh])
        o = self.op("Transpose", [o], f"{name}/ot", io, [G, B, S, dh], perm=[2, 0, 1, 3])
        o = self.reshape(f"{name}/of", o, [G, -1, dh], io, [G, None, dh])
        wa = self.shard(self.sd[f"{p}.wo_a.weight"].reshape(self.o_groups, R, dh), 0)
        o = self.op("MatMul", [o, self.init(wa.transpose(1, 2), f"{p}.wo_a/G", to=io)],
                    f"{name}/oa", io, [G, None, R])
        o = self.op("Transpose", [o], f"{name}/oat", io, [None, G, R], perm=[1, 0, 2])
        o = self.reshape(f"{name}/oaf", o, [-1, G * R], io, [None, G * R])
        o = self.proj(f"{name}/ob2", o, f"{p}.wo_b.weight", [None, self.dim], shard_axis=1)
        o = self.all_reduce(f"{name}/ar", o, io, [None, self.dim])
        return self.dyn_reshape(f"{name}/out", o, ctx["bs"], [self.dim], io, [B, S, self.dim])

    def make_dspark_block(self, stage, h, y, post, comb, main_x, ctx):
        lid = self.num_layers + stage
        n, p = f"/mtp.{stage}", f"layers.{lid}"
        bs = [B, S]
        a = self.make_dspark_attention(f"{n}/attn", lid, y, main_x, ctx)
        h, post, comb, y = self.make_hc_mix(
            f"{n}/hc_attn_post", a, h, post, comb,
            (f"{n}/hc_ffn", f"{n}/ffn_norm", f"{p}.hc_ffn", f"{p}.ffn_norm.weight"), bs)
        f_ = self.make_moe(f"{n}/ffn", lid, y, ctx)
        if stage + 1 < self.n_mtp:
            nl = lid + 1
            h, post, comb, y = self.make_hc_mix(
                f"{n}/hc_ffn_post", f_, h, post, comb,
                (f"/mtp.{stage + 1}/hc_attn", f"/mtp.{stage + 1}/attn_norm",
                 f"layers.{nl}.hc_attn", f"layers.{nl}.attn_norm.weight"), bs)
        else:
            h = self.make_hc_post(f"{n}/hc_ffn_post", f_, h, post, comb, bs)
            y = post = comb = None
        return h, y, post, comb

    def make_mtp_graph(self):
        F, I, BOOL = ir.DataType.FLOAT, ir.DataType.INT64, ir.DataType.BOOL
        io, hc, dim = self.io_dtype, self.hc, self.dim
        W, D, K, rd = self.window, self.head_dim, self.dspark_block, self.rope_head_dim
        first, last = self.num_layers, self.num_layers + self.n_mtp - 1

        self.make_inputs_and_outputs()
        self.make_rope_tables()

        shp = self.op("Shape", ["input_ids"], "/shape", I, [2])
        bv = self.op("Gather", [shp, self.const("INT64", 0)], "/batch", I, [], axis=0)
        bv1 = self.unsq("/bv1", bv, [0], I, [1])
        # Every block tensor is [B, K]; `S` is the draft block length in this graph.
        bs = self.op("Concat", [bv1, self.const("INT64", [K])], "/bs", I, [2], axis=0)

        mshape = self.op("Shape", ["main_hidden"], "/mshape", I, [3])
        Mv = self.op("Gather", [mshape, self.const("INT64", 1)], "/mlen", I, [], axis=0)
        Mv1 = self.unsq("/mlen1", Mv, [0], I, [1])
        mrng = self.op("Range", [self.const("INT64", 0), Mv, self.const("INT64", 1)],
                       "/mrange", I, [MAIN])
        past2 = self.unsq("/past2", "past_lens", [1], I, [B, 1])
        mpos = self.op("Add", [mrng, past2], "/mpos", I, [B, MAIN])
        # The draft block sits immediately after the accepted run (model.py:772).
        dpos = self.op("Add", [self.init(torch.arange(K, dtype=torch.int64), "/draft_arange"),
                               self.op("Add", [past2, Mv], "/dstart", I, [B, 1])],
                       "/dpos", I, [B, S])

        def gather_rope(tag, pos, shape):
            c = self.op("Gather", ["rope/cos_0", pos], f"{tag}/cos", F, shape, axis=0)
            s = self.op("Gather", ["rope/sin_0", pos], f"{tag}/sin", F, shape, axis=0)
            return c, s

        mcos, msin = gather_rope("/rope/main", mpos, [B, MAIN, rd])
        cos, sin = gather_rope("/rope/draft", dpos, [B, S, rd])

        # Accepted rows retain the reference ring's logical slots. Draft rows occupy the
        # five slots after it; sparse indices make all of them visible to every query.
        slot = self.op("Mod", [mpos, self.const("INT64", W)], "/slot", I, [B, MAIN])
        draft_slot = self.init(torch.arange(W, W + K, dtype=torch.int64), "/draft_slot")
        draft_slot = self.op("Expand", [self.unsq("/draft_slot_u", draft_slot, [0], I, [1, K]),
                        bs], "/draft_slot_b", I, [B, K])
        slot = self.op("Concat", [slot, draft_slot], "/cache_slot", I, [B, None], axis=1)
        slot32 = self.cast("/cache_slot32", self.reshape("/cache_slot_flat", slot, [-1], I,
                                 [None]), ir.DataType.INT32, [None])

        # A ring row is live once the sequence has reached it.  The draft rows are always
        # live, so they ride along as position -1, which is below any total.
        jext = torch.cat([torch.arange(W, dtype=torch.int64), torch.full((K,), -1, dtype=torch.int64)])
        live = self.op("Less", [self.init(jext, "/jext"),
                    self.op("Add", [past2, Mv], "/total", I, [B, 1])],
                   "/live", BOOL, [B, W + K])
        selected = self.op("Where", [live, self.init(torch.arange(W + K, dtype=torch.int64),
                                  "/selected_rows"),
                         self.const("INT64", -1)],
                   "/selected", I, [B, W + K])
        selected = self.op("Expand", [self.unsq("/selected_u", selected, [1], I,
                             [B, 1, W + K]),
                          self.op("Concat", [bs, self.const("INT64", [W + K])],
                              "/selected_shape", I, [3], axis=0)],
                   "/selected_b", I, [B, S, W + K])
        kvi32 = self.cast("/selected32", self.reshape("/selected_flat", selected,
                                   [-1, W + K], I, [None, W + K]),
                  ir.DataType.INT32, [None, W + K])
        block_table32 = self.init(torch.arange((W + K + self.block_size - 1) // self.block_size,
                            dtype=torch.int32).reshape(1, -1),
                      "/mtp_block_table")
        cum32 = self.init(torch.tensor([0, K], dtype=torch.int32), "/mtp_cum")
        zero_past32 = self.cast("/zero_past32", self.op("Sub", ["past_lens", "past_lens"],
                                "/zero_past", I, [B]),
                    ir.DataType.INT32, [B])

        nd_ = io if self.norm_bf16 else F
        main_x = self.proj("/mtp/main_proj", "main_hidden", f"layers.{first}.main_proj.weight",
                           [B, MAIN, dim])
        main_x = self.make_rmsnorm("/mtp/main_norm", main_x,
                                   self.init_w(f"layers.{first}.main_norm.weight", to=nd_),
                                   [B, MAIN, dim], dtype=nd_)
        if nd_ != io:
            main_x = self.cast("/mtp/main_norm_b", main_x, io, [B, MAIN, dim])

        ctx = {"bs": bs, "mcos": mcos, "msin": msin, "cos": cos, "sin": sin,
               "cos_q": self.unsq("/rope/cosq", cos, [2], F, [B, S, 1, rd]),
               "sin_q": self.unsq("/rope/sinq", sin, [2], F, [B, S, 1, rd]),
             "slot32": slot32, "kvi32": kvi32, "block_table32": block_table32,
             "cum32": cum32, "zero_past32": zero_past32}

        # The block is the accepted token followed by noise; the drafter fills it in one
        # pass rather than autoregressively (model.py:855).
        noise = self.op("Expand", [self.const("INT64", self.noise_token),
                                   self.op("Concat", [bv1, self.const("INT64", [K - 1])],
                                           "/noiseshape", I, [2], axis=0)],
                        "/noise", I, [B, K - 1])
        draft_ids = self.op("Concat", ["input_ids", noise], "/draft_ids", I, [B, S], axis=1)
        emb = self.op("Gather", [self.init_w("embed.weight", to=io), draft_ids],
                      "/mtp/embed", io, [B, S, dim], axis=0)
        h = self.op("Expand", [self.unsq("/mtp/embed_u", emb, [2], io, [B, S, 1, dim]),
                               self.op("Concat", [bs, self.const("INT64", [hc, dim])],
                                       "/hc_expand_shape", I, [4], axis=0)],
                    "/mtp/hc_expand", io, [B, S, hc, dim])

        y, post, comb = self.make_hc_pre(f"/mtp.0/hc_attn", h, f"layers.{first}.hc_attn", [B, S])
        y = self.make_hc_norm(f"/mtp.0/attn_norm", y, f"layers.{first}.attn_norm.weight", [B, S])
        for stage in range(self.n_mtp):
            h, y, post, comb = self.make_dspark_block(stage, h, y, post, comb, main_x, ctx)

        x = self.make_hc_head("/mtp/hc_head", h, f"layers.{last}.hc_head", [B, S])
        n_ = self.make_rmsnorm("/mtp/norm", x, self.init_w(f"layers.{last}.norm.weight", to=nd_),
                               [B, S, dim], dtype=nd_)
        if nd_ != io:
            n_ = self.cast("/mtp/norm_b", n_, io, [B, S, dim])
        logits = self.make_mtp_logits(n_)
        self.make_mtp_head(x, logits)

    def make_mtp_logits(self, n_):
        """The drafter shares the target's lm head (model.py:903)."""
        F, io, V = ir.DataType.FLOAT, self.io_dtype, self.vocab_size
        hd = self.head_dtype
        nh = n_ if hd == io else self.cast("/mtp/lm_head_in", n_, hd, [B, S, self.dim])
        if self.lm_head_shard and self.world > 1 and V % self.world == 0:
            vloc = V // self.world
            lg = self.op("MatMul", [nh, self.init(self.shard(self.sd["head.weight"], 0).T,
                                                  "head/T", to=hd)], "/mtp/lm_head", hd,
                         [B, S, vloc])
            lgf = self.cast("/mtp/lm_head_f", lg, F, [B, S, vloc])
            pads = [0, 0, self.rank * vloc, 0, 0, (self.world - 1 - self.rank) * vloc]
            wide = self.op("Pad", [lgf, self.const("INT64", pads)], "/mtp/lm_head_pad",
                           F, [B, S, V])
            return self.all_reduce("/mtp/logits", wide, F, [B, S, V])
        lg = self.op("MatMul", [nh, self.init(self.sd["head.weight"].T, "head/T", to=hd)],
                     "/mtp/lm_head", hd, [B, S, V])
        return self.cast("/mtp/logits", lg, F, [B, S, V])

    def make_mtp_head(self, x, logits):
        """Sample the block through the rank-256 Markov head (model.py:864).

        Only this head runs autoregressively -- the transformer above produced all ``K``
        positions in one pass -- so the loop is ``K`` narrow GEMVs, not ``K`` blocks.
        Both Markov tensors are replicated rather than sharded: they are 66 MiB each,
        and sharding them would put a full-vocabulary collective inside the loop.
        """
        F, I, io = ir.DataType.FLOAT, ir.DataType.INT64, self.io_dtype
        md = self.markov_dtype
        K, V, R, last = self.dspark_block, self.vocab_size, self.markov_rank, \
            self.num_layers + self.n_mtp - 1
        mp = f"layers.{last}.markov_head"
        w1 = self.init_w(f"{mp}.markov_w1.weight", to=md)
        w2 = self.init(self.sd[f"{mp}.markov_w2.weight"].T, f"{mp}.markov_w2/T", to=md)

        cur = self.op("Squeeze", ["input_ids", self.const("INT64", [1])], "/mtp/tok0", I, [B])
        ids, embeds, draft_indices, draft_probs = [cur], [], [], []
        for i in range(K):
            e = self.op("Gather", [w1, cur], f"/mtp/mk{i}/embed", md, [B, R], axis=0)
            # The confidence head consumes the embedding in the activation type.
            embeds.append(e if md == io else self.cast(f"/mtp/mk{i}/embed_b", e, io, [B, R]))
            bias = self.op("MatMul", [e, w2], f"/mtp/mk{i}/bias", md, [B, V])
            bf = bias if md == F else self.cast(f"/mtp/mk{i}/biasf", bias, F, [B, V])
            li = self.op("Add", [self.op("Gather", [logits, self.const("INT64", i)],
                                         f"/mtp/mk{i}/slice", F, [B, V], axis=1), bf],
                         f"/mtp/mk{i}/logits", F, [B, V])
            if self.mtp_sample_top_k:
                Q = self.mtp_sample_top_k
                uniform = self.op("Gather", ["draft_uniform", self.const("INT64", i)],
                                  f"/mtp/mk{i}/uniform", F, [B], axis=1)
                cur, indices, probs = (f"/mtp/mk{i}/sample", f"/mtp/mk{i}/indices",
                                       f"/mtp/mk{i}/probs")
                self.make_node(
                    "FusedTopKSample", inputs=[li, uniform], outputs=[cur, indices, probs],
                    name=f"/mtp/mk{i}/sample", domain="com.microsoft.genai", top_k=Q,
                    top_p=self.mtp_sample_top_p, temperature=self.mtp_sample_temperature)
                self.make_value(cur, I, shape=[B])
                self.make_value(indices, I, shape=[B, Q])
                self.make_value(probs, F, shape=[B, Q])
                draft_indices.append(self.unsq(f"/mtp/mk{i}/indices_u", indices, [1],
                                               I, [B, 1, Q]))
                draft_probs.append(self.unsq(f"/mtp/mk{i}/probs_u", probs, [1],
                                             F, [B, 1, Q]))
            else:
                cur = self.op("ArgMax", [li], f"/mtp/mk{i}/argmax", I, [B], axis=-1,
                              keepdims=0)
            ids.append(cur)

        self.make_node("Concat", inputs=[self.unsq(f"/mtp/id{i}", t, [1], I, [B, 1])
                                         for i, t in enumerate(ids)],
                       outputs=["output_ids"], name="/mtp/output_ids", axis=1)
        if self.mtp_sample_top_k:
            self.make_node("Concat", inputs=draft_indices, outputs=["draft_indices"],
                           name="/mtp/draft_indices", axis=1)
            self.make_node("Concat", inputs=draft_probs, outputs=["draft_probs"],
                           name="/mtp/draft_probs", axis=1)

        me = self.op("Concat", [self.unsq(f"/mtp/me{i}", e, [1], io, [B, 1, R])
                                for i, e in enumerate(embeds)],
                     "/mtp/markov_embed", io, [B, S, R], axis=1)
        cin = self.op("Concat", [x, me], "/mtp/conf_in", io, [B, S, self.dim + R], axis=-1)
        cw = self.init(self.sd[f"layers.{last}.confidence_head.proj.weight"].T,
                       "confidence/T", to=F)
        conf = self.op("MatMul", [self.cast("/mtp/conf_f", cin, F, [B, S, self.dim + R]), cw],
                       "/mtp/conf", F, [B, S, 1])
        self.make_node("Squeeze", inputs=[conf, self.const("INT64", [2])],
                       outputs=["confidence"], name="/mtp/confidence")

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
        if self.graph_kind == "mtp":
            self.make_mtp_graph()
            return
        self.make_dsv4_graph()
