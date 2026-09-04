# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Builder for a DFlash 2 block-diffusion draft model (``dflash2.onnx``).

DFlash 2 is a block drafter: given the target's auxiliary hidden states it predicts a
whole block of tokens in one pass, keeps the top ``selector_top_k`` candidates at every
position, and a low-rank selector scores the edges between consecutive positions so the
runtime can trace one coherent path through the lattice.

Three things make its graph different from a normal decoder, and all three are visible
in the emitted ONNX:

1. **Cross-attention over target hidden states.** The draft layers never run over the
   context. Every layer's context K/V is projected from the *same* target-derived
   hidden state (``fc`` -> ``hidden_norm``), so only the ``block_size`` query rows carry
   a hidden stream. Context and query rows are merged into one packed
   ``PagedAttention`` call by gathering per-row K/V from ``concat(query, context)``;
   the context rows' query is a duplicate of row 0 and their attention output is
   dropped. This is what fills the draft KV cache — no separate cache-store op.
2. **Non-causal attention** (``is_causal=0``): the query block attends to itself
   bidirectionally, with ``local_window_size`` still bounding the mask on the left.
3. **Two-tap dynamic grouped convolutions** around attention and the MLP, whose
   coefficients are predicted per token and which reset at each block boundary.

The drafter checkpoint ships no embedding and no LM head; both are the target's and are
emitted with the target's initializer names so ``share_external_initializers`` can fold
them back onto ``text.onnx.data``.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import onnx_ir as ir
import torch

from .block_drafter import BlockDrafterBuilder


class DFlash2Builder(BlockDrafterBuilder):
    """Emits ``dflash2.onnx`` from a DFlash 2 draft checkpoint plus the target's
    embedding and LM head."""

    def __init__(
        self,
        draft_dir,
        target_dir,
        io_dtype,
        paged_block_size,
        max_position_embeddings,
        filename="dflash2.onnx",
        num_draft_tokens=None,
    ):
        self.draft_dir = draft_dir
        self.target_dir = target_dir
        # The drafter is a bf16 checkpoint and its activations genuinely leave the fp16 range
        # (the fc output alone reaches ~1.4e4 and the MLP product overflows two layers in), so the
        # body runs in bf16. Only the tensors it shares with the fp16 target -- the aux hidden
        # states, the embedding table and the FP8 LM head -- stay at the target's dtype.
        self.io_dtype = ir.DataType.BFLOAT16
        self.external_dtype = io_dtype
        self.filename = filename
        self.paged_block_size = paged_block_size

        with open(os.path.join(draft_dir, "config.json")) as f:
            cfg = json.load(f)
        self.cfg = cfg
        dfl = cfg["dflash_config"]
        self.dflash_config = dfl

        self.hidden_size = int(cfg["hidden_size"])
        self.num_layers = int(cfg["num_hidden_layers"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.num_kv_heads = int(cfg["num_key_value_heads"])
        self.head_size = int(cfg["head_dim"])
        self.intermediate_size = int(cfg["intermediate_size"])
        self.vocab_size = int(cfg["vocab_size"])
        self.rms_eps = float(cfg["rms_norm_eps"])
        self.max_position = int(max_position_embeddings or cfg["max_position_embeddings"])
        rope_parameters = cfg["rope_parameters"]
        rope_type = str(rope_parameters.get("rope_type", "default")).lower()
        if rope_type != "default":
            raise ValueError(f"DFlash 2 does not support the '{rope_type}' RoPE type; expected 'default'.")
        self.rope_theta = float(rope_parameters["rope_theta"])
        self.sliding_window = self.resolve_sliding_window(cfg)
        # `is_causal` is explicit in a DFlash 2 config and is what makes the block bidirectional.
        self.is_causal = bool(cfg.get("is_causal", False))

        self.taps = int(dfl["conv_kernel_size"])
        self.group_size = int(dfl["conv_group_size"])
        self.num_groups = self.hidden_size // self.group_size
        self.selector_rank = int(dfl["selector_rank"])
        self.selector_top_k = int(dfl["selector_top_k"])
        self.mask_token_id = int(dfl["mask_token_id"])
        self.validate_token_metadata(cfg, dfl)
        self.target_layer_ids = list(dfl["target_layer_ids"])
        # Query tokens per request: the anchor (bonus) token plus one mask token per draft.
        checkpoint_block_size = int(dfl["block_size"])
        if num_draft_tokens is not None and not 1 <= int(num_draft_tokens) < checkpoint_block_size:
            raise ValueError(
                f"num_draft_tokens must be between 1 and {checkpoint_block_size - 1}, the checkpoint draft limit."
            )
        self.block_size = int(num_draft_tokens) + 1 if num_draft_tokens is not None else checkpoint_block_size
        self.num_draft_tokens = self.block_size - 1
        self.input_embedding_scale = float(dfl.get("input_embedding_scale", 1.0))
        self.aux_hidden_size = self.hidden_size * len(self.target_layer_ids)
        self.make_graph("dflash2_graph", "dflash2")

    # ------------------------------------------------------------ rope caches

    def _make_rope_caches(self):
        # PagedAttention type-constrains cos/sin to the query's element type.
        dim = self.head_size
        inv_freq = 1.0 / (self.rope_theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        pos = np.arange(self.max_position, dtype=np.float64)[:, None]
        freqs = pos * inv_freq[None, :]
        for name, fn in (("dflash2.cos_cache", np.cos), ("dflash2.sin_cache", np.sin)):
            self.make_initializer(torch.from_numpy(fn(freqs).astype(np.float32)), name, to=self.io_dtype)

    # ------------------------------------------------------------------ conv

    def _block_shift_mask(self, num_block_rows_source):
        """``[num_block, 1]`` -- 0 on the first row of every block, 1 elsewhere.

        A tap-``t`` term must not read across a block boundary; the reference masks it
        with ``position_within_block >= t``, and with ``taps == 2`` that is exactly
        "not the first row".
        """
        shape = self.unary("Shape", "/dflash2/conv_mask/Shape", num_block_rows_source, ir.DataType.INT64, [2])
        rows = self.binary("Gather", "/dflash2/conv_mask/Gather", shape, self.const(0), ir.DataType.INT64, [])
        rng = self.out("/dflash2/conv_mask/Range")
        self.make_node(
            "Range",
            [self.const(0), rows, self.const(1)],
            [rng],
            name="/dflash2/conv_mask/Range",
        )
        self.make_value(rng, ir.DataType.INT64, ["num_block"])
        mod = self.binary(
            "Mod", "/dflash2/conv_mask/Mod", rng, self.const(self.block_size), ir.DataType.INT64, ["num_block"]
        )
        gt = self.binary("Greater", "/dflash2/conv_mask/Greater", mod, self.const(0), ir.DataType.BOOL, ["num_block"])
        cast = self.unary("Cast", "/dflash2/conv_mask/Cast", gt, self.io_dtype, ["num_block"], to=self.io_dtype)
        return self.reshape("/dflash2/conv_mask/Reshape", cast, [-1, 1], self.io_dtype, ["num_block", 1])

    def _conv_coefficients(self, prefix, x, kernel_weight, rows):
        """``kernel_projection(x)`` split into ``[side][tap]`` per-group deltas."""
        proj = self.matmul(
            f"{prefix}/kernel_projection/MatMul",
            x,
            kernel_weight,
            self.hidden_size,
            2 * self.taps * self.num_groups,
            rows,
        )
        flat = self.reshape(
            f"{prefix}/kernel_projection/Reshape",
            proj,
            [-1, 2 * self.taps, self.num_groups],
            self.io_dtype,
            ["num_block", 2 * self.taps, self.num_groups],
        )
        outs = [f"{prefix}/kernel_projection/Split/output_{i}" for i in range(2 * self.taps)]
        self.make_node(
            "Split", [flat], outs, name=f"{prefix}/kernel_projection/Split", axis=1, num_outputs=2 * self.taps
        )
        deltas = []
        for i, o in enumerate(outs):
            self.make_value(o, self.io_dtype, ["num_block", 1, self.num_groups])
            deltas.append(
                self.reshape(
                    f"{prefix}/kernel_projection/delta{i}",
                    o,
                    [-1, self.num_groups, 1],
                    self.io_dtype,
                    ["num_block", self.num_groups, 1],
                )
            )
        # Split output index is side * taps + tap (the reference reshapes to [T, 2, taps, G]).
        return [deltas[side * self.taps : (side + 1) * self.taps] for side in range(2)]

    def _grouped_conv(self, prefix, x, deltas, base_kernel, shift_mask, rows):
        """``sum_tap (base[tap] + delta[tap]) * shift(x, tap)`` inside each block."""
        terms = []
        for tap in range(self.taps):
            base_name = f"{prefix}.base_kernel.{tap}"
            self.make_initializer(
                base_kernel[tap].reshape(1, self.num_groups, self.group_size), base_name, to=self.io_dtype
            )
            coef3 = self.binary(
                "Add",
                f"{prefix}/coef{tap}/Add",
                base_name,
                deltas[tap],
                self.io_dtype,
                ["num_block", self.num_groups, self.group_size],
            )
            coef = self.reshape(
                f"{prefix}/coef{tap}/Reshape",
                coef3,
                [-1, self.hidden_size],
                self.io_dtype,
                ["num_block", self.hidden_size],
            )
            if tap == 0:
                shifted = x
            else:
                # shifted[t] = x[t - tap]. Rows 0..tap-1 are masked out below, so the
                # front is filled with x[:tap] rather than zeros -- ORT has no
                # bfloat16 Pad kernel, and Concat needs no new constant.
                head = self.out(f"{prefix}/shift{tap}/Head")
                self.make_node(
                    "Slice",
                    [x, self.const([0]), self.const([tap]), self.const([0])],
                    [head],
                    name=f"{prefix}/shift{tap}/Head",
                )
                self.make_value(head, self.io_dtype, [tap, self.hidden_size])
                sl = self.out(f"{prefix}/shift{tap}/Slice")
                self.make_node(
                    "Slice",
                    [x, self.const([0]), self.const([-tap]), self.const([0])],
                    [sl],
                    name=f"{prefix}/shift{tap}/Slice",
                )
                # A distinct symbolic dim: this is num_block - tap rows, and claiming
                # "num_block" lets the memory planner reuse this buffer for the result.
                self.make_value(sl, self.io_dtype, [f"num_block_minus_{tap}", self.hidden_size])
                cat = self.out(f"{prefix}/shift{tap}/Concat")
                self.make_node("Concat", [head, sl], [cat], name=f"{prefix}/shift{tap}/Concat", axis=0)
                self.make_value(cat, self.io_dtype, ["num_block", self.hidden_size])
                shifted = cat
            term = self.binary(
                "Mul", f"{prefix}/term{tap}/Mul", shifted, coef, self.io_dtype, ["num_block", self.hidden_size]
            )
            if tap > 0:
                term = self.binary(
                    "Mul", f"{prefix}/term{tap}/Mask", term, shift_mask, self.io_dtype, ["num_block", self.hidden_size]
                )
            terms.append(term)
        out = terms[0]
        for i, term in enumerate(terms[1:], start=1):
            out = self.binary("Add", f"{prefix}/sum{i}/Add", out, term, self.io_dtype, ["num_block", self.hidden_size])
        return out

    # ------------------------------------------------------------------ graph

    def make_model(self):
        w = self.load_weights()
        self.declare_io()
        self._make_rope_caches()

        rows_q = "num_block"

        # --- context path: the target's aux hidden states become every layer's K/V ---
        aux = self.unary(
            "Cast",
            "/dflash2/aux/Cast",
            "aux_hidden_states",
            self.io_dtype,
            ["num_ctx", self.aux_hidden_size],
            to=self.io_dtype,
        )
        ctx = self.matmul("/dflash2/fc/MatMul", aux, w["fc.weight"], self.aux_hidden_size, self.hidden_size, "num_ctx")
        ctx_n = self.rms_norm("/dflash2/hidden_norm", ctx, w["hidden_norm.weight"], "num_ctx")
        ctx_kv = []
        for i in range(self.num_layers):
            k = self.matmul(
                f"/dflash2/layers.{i}/ctx_k/MatMul",
                ctx_n,
                w[f"layers.{i}.self_attn.k_proj.weight"],
                self.hidden_size,
                self.num_kv_heads * self.head_size,
                "num_ctx",
                weight_name=f"dflash2.layers.{i}.self_attn.k_proj.weight",
            )
            v = self.matmul(
                f"/dflash2/layers.{i}/ctx_v/MatMul",
                ctx_n,
                w[f"layers.{i}.self_attn.v_proj.weight"],
                self.hidden_size,
                self.num_kv_heads * self.head_size,
                "num_ctx",
                weight_name=f"dflash2.layers.{i}.self_attn.v_proj.weight",
            )
            ctx_kv.append((k, v))

        # --- query path ---
        self.make_initializer(w["embed_tokens.weight"], "model.embed_tokens.weight", to=self.external_dtype)
        emb_ext = self.binary(
            "Gather",
            "/dflash2/embed_tokens/Gather",
            "model.embed_tokens.weight",
            "input_ids",
            self.external_dtype,
            [rows_q, self.hidden_size],
        )
        emb = self.unary(
            "Cast", "/dflash2/embed_tokens/Cast", emb_ext, self.io_dtype, [rows_q, self.hidden_size], to=self.io_dtype
        )
        if self.input_embedding_scale != 1.0:
            scale = self.const(np.full((1,), self.input_embedding_scale, np.float32), ir.DataType.FLOAT)
            scale_cast = self.unary(
                "Cast", "/dflash2/embed_tokens/ScaleCast", scale, self.io_dtype, [1], to=self.io_dtype
            )
            emb = self.binary(
                "Mul", "/dflash2/embed_tokens/Mul", emb, scale_cast, self.io_dtype, [rows_q, self.hidden_size]
            )

        shift_mask = self._block_shift_mask(emb)

        hidden, residual = emb, None
        for i in range(self.num_layers):
            p = f"/dflash2/layers.{i}"
            if residual is None:
                x = self.rms_norm(f"{p}/input_layernorm", hidden, w[f"layers.{i}.input_layernorm.weight"], rows_q)
                residual = hidden
            else:
                x, residual = self.skip_rms_norm(
                    f"{p}/input_layernorm", residual, hidden, w[f"layers.{i}.input_layernorm.weight"], rows_q
                )

            # attention_conv.prepare / .finish
            attn_deltas = self._conv_coefficients(
                f"{p}/attention_conv", x, w[f"layers.{i}.attention_conv.kernel_projection.weight"], rows_q
            )
            x = self._grouped_conv(
                f"{p}/attention_conv/prepare",
                x,
                attn_deltas[0],
                w[f"layers.{i}.attention_conv.base_kernel"][0],
                shift_mask,
                rows_q,
            )

            attn_out = self._make_attention(i, x, ctx_kv[i], rows_q)
            attn_out = self._grouped_conv(
                f"{p}/attention_conv/finish",
                attn_out,
                attn_deltas[1],
                w[f"layers.{i}.attention_conv.base_kernel"][1],
                shift_mask,
                rows_q,
            )

            y, residual = self.skip_rms_norm(
                f"{p}/post_attention_layernorm",
                residual,
                attn_out,
                w[f"layers.{i}.post_attention_layernorm.weight"],
                rows_q,
            )

            mlp_deltas = self._conv_coefficients(
                f"{p}/mlp_conv", y, w[f"layers.{i}.mlp_conv.kernel_projection.weight"], rows_q
            )
            y = self._grouped_conv(
                f"{p}/mlp_conv/prepare", y, mlp_deltas[0], w[f"layers.{i}.mlp_conv.base_kernel"][0], shift_mask, rows_q
            )
            m = self._make_mlp(i, y, w, rows_q)
            hidden = self._grouped_conv(
                f"{p}/mlp_conv/finish", m, mlp_deltas[1], w[f"layers.{i}.mlp_conv.base_kernel"][1], shift_mask, rows_q
            )

        final, _ = self.skip_rms_norm("/dflash2/norm", residual, hidden, w["norm.weight"], rows_q, want_sum=False)

        self._make_candidates_and_selector(final, w)
        self.graph.sort()
        return self.model

    def _make_attention(self, i, x, ctx_kv, rows_q):
        p = f"/dflash2/layers.{i}"
        w = self.weights
        q = self.matmul(
            f"{p}/attn/q_proj/MatMul",
            x,
            w[f"layers.{i}.self_attn.q_proj.weight"],
            self.hidden_size,
            self.num_heads * self.head_size,
            rows_q,
        )
        k = self.matmul(
            f"{p}/attn/k_proj/MatMul",
            x,
            w[f"layers.{i}.self_attn.k_proj.weight"],
            self.hidden_size,
            self.num_kv_heads * self.head_size,
            rows_q,
            weight_name=f"dflash2.layers.{i}.self_attn.k_proj.weight",
        )
        v = self.matmul(
            f"{p}/attn/v_proj/MatMul",
            x,
            w[f"layers.{i}.self_attn.v_proj.weight"],
            self.hidden_size,
            self.num_kv_heads * self.head_size,
            rows_q,
            weight_name=f"dflash2.layers.{i}.self_attn.v_proj.weight",
        )

        # Interleave the block rows and the context rows into one packed token stream.
        # `qkv_row_map` indexes concat(block, context); `q_row_map` indexes the block rows
        # alone (context rows point at row 0 and their output is dropped).
        kv_dim = self.num_kv_heads * self.head_size
        k_cat = self.out(f"{p}/attn/k_concat")
        self.make_node("Concat", [k, ctx_kv[0]], [k_cat], name=f"{p}/attn/k_concat", axis=0)
        self.make_value(k_cat, self.io_dtype, ["num_rows", kv_dim])
        v_cat = self.out(f"{p}/attn/v_concat")
        self.make_node("Concat", [v, ctx_kv[1]], [v_cat], name=f"{p}/attn/v_concat", axis=0)
        self.make_value(v_cat, self.io_dtype, ["num_rows", kv_dim])

        q_all = self.binary(
            "Gather",
            f"{p}/attn/q_gather",
            q,
            "q_row_map",
            self.io_dtype,
            ["num_tokens", self.num_heads * self.head_size],
        )
        k_all = self.binary("Gather", f"{p}/attn/k_gather", k_cat, "qkv_row_map", self.io_dtype, ["num_tokens", kv_dim])
        v_all = self.binary("Gather", f"{p}/attn/v_gather", v_cat, "qkv_row_map", self.io_dtype, ["num_tokens", kv_dim])

        q_norm = self.make_initializer(
            w[f"layers.{i}.self_attn.q_norm.weight"], f"dflash2.layers.{i}.self_attn.q_norm.weight", to=self.io_dtype
        )
        k_norm = self.make_initializer(
            w[f"layers.{i}.self_attn.k_norm.weight"], f"dflash2.layers.{i}.self_attn.k_norm.weight", to=self.io_dtype
        )

        attn_name = f"{p}/attn/PagedAttention"
        attn_out = self.out(attn_name)
        self.make_node(
            "PagedAttention",
            [
                q_all,
                k_all,
                v_all,
                f"past_key_values.{i}.key",
                f"past_key_values.{i}.value",
                "cumulative_sequence_lengths",
                "past_sequence_lengths",
                "block_table",
                "dflash2.cos_cache",
                "dflash2.sin_cache",
                "",  # slot_mapping: derived from block_table + past_sequence_lengths
                "",  # head_sink
                q_norm,
                k_norm,
                "",
                "",  # k_scale / v_scale
                "attention_metadata",
            ],
            [attn_out, f"present.{i}.key", f"present.{i}.value"],
            name=attn_name,
            domain="com.microsoft",
            num_heads=self.num_heads,
            kv_num_heads=self.num_kv_heads,
            local_window_size=self.sliding_window,
            is_causal=1 if self.is_causal else 0,
            do_rotary=1,
            rotary_interleaved=0,
            qk_norm_epsilon=self.rms_eps,
        )
        self.make_value(attn_out, self.io_dtype, ["num_tokens", self.num_heads * self.head_size])
        for suffix in ("key", "value"):
            self.make_value(
                f"present.{i}.{suffix}",
                self.io_dtype,
                ["num_blocks", self.paged_block_size, self.num_kv_heads, self.head_size],
            )

        block_out = self.binary(
            "Gather",
            f"{p}/attn/out_gather",
            attn_out,
            "block_row_index",
            self.io_dtype,
            [rows_q, self.num_heads * self.head_size],
        )
        return self.matmul(
            f"{p}/attn/o_proj/MatMul",
            block_out,
            w[f"layers.{i}.self_attn.o_proj.weight"],
            self.num_heads * self.head_size,
            self.hidden_size,
            rows_q,
        )

    def _make_mlp(self, i, x, w, rows_q):
        p = f"/dflash2/layers.{i}/mlp"
        gate = self.matmul(
            f"{p}/gate_proj/MatMul",
            x,
            w[f"layers.{i}.mlp.gate_proj.weight"],
            self.hidden_size,
            self.intermediate_size,
            rows_q,
        )
        up = self.matmul(
            f"{p}/up_proj/MatMul",
            x,
            w[f"layers.{i}.mlp.up_proj.weight"],
            self.hidden_size,
            self.intermediate_size,
            rows_q,
        )
        sig = self.unary("Sigmoid", f"{p}/act/Sigmoid", gate, self.io_dtype, [rows_q, self.intermediate_size])
        silu = self.binary("Mul", f"{p}/act/Mul", gate, sig, self.io_dtype, [rows_q, self.intermediate_size])
        prod = self.binary("Mul", f"{p}/act/MulUp", silu, up, self.io_dtype, [rows_q, self.intermediate_size])
        return self.matmul(
            f"{p}/down_proj/MatMul",
            prod,
            w[f"layers.{i}.mlp.down_proj.weight"],
            self.intermediate_size,
            self.hidden_size,
            rows_q,
        )

    def _make_candidates_and_selector(self, final, w):
        n_spec, top_k, rank = self.num_draft_tokens, self.selector_top_k, self.selector_rank

        # Only the mask rows predict; the anchor at offset 0 is the bonus token.
        h3 = self.reshape(
            "/dflash2/select/Reshape",
            final,
            [-1, self.block_size, self.hidden_size],
            self.io_dtype,
            ["batch_size", self.block_size, self.hidden_size],
        )
        sl = self.out("/dflash2/select/Slice")
        self.make_node(
            "Slice",
            [h3, self.const([1]), self.const([self.block_size]), self.const([1])],
            [sl],
            name="/dflash2/select/Slice",
        )
        self.make_value(sl, self.io_dtype, ["batch_size", n_spec, self.hidden_size])
        hsel = self.reshape(
            "/dflash2/select/Flatten", sl, [-1, self.hidden_size], self.io_dtype, ["num_sample", self.hidden_size]
        )

        # LM head (the target's, shared on disk) -> top-k candidates per position.
        logits = self.make_lm_head(hsel, "num_sample")
        logits32 = self.unary(
            "Cast",
            "/dflash2/topk/Cast",
            logits,
            ir.DataType.FLOAT,
            ["num_sample", self.vocab_size],
            to=ir.DataType.FLOAT,
        )
        vals, idx = "/dflash2/topk/values", "/dflash2/topk/indices"
        self.make_node(
            "TopK",
            [logits32, self.const([top_k])],
            [vals, idx],
            name="/dflash2/topk/TopK",
            axis=-1,
            largest=1,
            sorted=1,
        )
        self.make_value(vals, ir.DataType.FLOAT, ["num_sample", top_k])
        self.make_value(idx, ir.DataType.INT64, ["num_sample", top_k])

        cand = self.reshape(
            "/dflash2/topk/cand", idx, [-1, n_spec, top_k], ir.DataType.INT64, ["batch_size", n_spec, top_k]
        )
        unary_logits = self.reshape(
            "/dflash2/topk/unary", vals, [-1, n_spec, top_k], ir.DataType.FLOAT, ["batch_size", n_spec, top_k]
        )

        # Low-rank edge scores between position l-1 and l.
        hp = self.matmul(
            "/dflash2/selector/hidden_projection/MatMul",
            hsel,
            w["candidate_selector.hidden_projection.weight"],
            self.hidden_size,
            rank,
            "num_sample",
        )
        hp3 = self.reshape(
            "/dflash2/selector/hp", hp, [-1, n_spec, 1, rank], self.io_dtype, ["batch_size", n_spec, 1, rank]
        )

        ids2 = self.reshape(
            "/dflash2/selector/ids",
            "input_ids",
            [-1, self.block_size],
            ir.DataType.INT64,
            ["batch_size", self.block_size],
        )
        anchor = self.out("/dflash2/selector/anchor")
        self.make_node(
            "Slice",
            [ids2, self.const([0]), self.const([1]), self.const([1])],
            [anchor],
            name="/dflash2/selector/anchor",
        )
        self.make_value(anchor, ir.DataType.INT64, ["batch_size", 1])
        anchor3 = self.reshape("/dflash2/selector/anchor3", anchor, [-1, 1, 1], ir.DataType.INT64, ["batch_size", 1, 1])
        anchor_tiled = self.binary(
            "Tile",
            "/dflash2/selector/anchor_tile",
            anchor3,
            self.const([1, 1, top_k]),
            ir.DataType.INT64,
            ["batch_size", 1, top_k],
        )
        prev_tail = self.out("/dflash2/selector/prev_tail")
        self.make_node(
            "Slice",
            [cand, self.const([0]), self.const([n_spec - 1]), self.const([1])],
            [prev_tail],
            name="/dflash2/selector/prev_tail",
        )
        self.make_value(prev_tail, ir.DataType.INT64, ["batch_size", n_spec - 1, top_k])
        prev = self.out("/dflash2/selector/prev")
        self.make_node("Concat", [anchor_tiled, prev_tail], [prev], name="/dflash2/selector/prev", axis=1)
        self.make_value(prev, ir.DataType.INT64, ["batch_size", n_spec, top_k])

        self.make_initializer(
            w["candidate_selector.predecessor_codebook"],
            "dflash2.candidate_selector.predecessor_codebook",
            to=self.io_dtype,
        )
        self.make_initializer(
            w["candidate_selector.successor_codebook"],
            "dflash2.candidate_selector.successor_codebook",
            to=self.io_dtype,
        )
        pred = self.binary(
            "Gather",
            "/dflash2/selector/pred_gather",
            "dflash2.candidate_selector.predecessor_codebook",
            prev,
            self.io_dtype,
            ["batch_size", n_spec, top_k, rank],
        )
        succ = self.binary(
            "Gather",
            "/dflash2/selector/succ_gather",
            "dflash2.candidate_selector.successor_codebook",
            cand,
            self.io_dtype,
            ["batch_size", n_spec, top_k, rank],
        )
        pw = self.binary("Mul", "/dflash2/selector/pw", pred, hp3, self.io_dtype, ["batch_size", n_spec, top_k, rank])
        succ_t = self.unary(
            "Transpose",
            "/dflash2/selector/succ_t",
            succ,
            self.io_dtype,
            ["batch_size", n_spec, rank, top_k],
            perm=[0, 1, 3, 2],
        )
        pair = self.binary(
            "MatMul", "/dflash2/selector/pair", pw, succ_t, self.io_dtype, ["batch_size", n_spec, top_k, top_k]
        )
        pair32 = self.unary(
            "Cast",
            "/dflash2/selector/pair_cast",
            pair,
            ir.DataType.FLOAT,
            ["batch_size", n_spec, top_k, top_k],
            to=ir.DataType.FLOAT,
        )
        unary4 = self.reshape(
            "/dflash2/selector/unary4",
            unary_logits,
            [-1, n_spec, 1, top_k],
            ir.DataType.FLOAT,
            ["batch_size", n_spec, 1, top_k],
        )
        self.make_node("Add", [pair32, unary4], ["draft_scores"], name="/dflash2/selector/scores")
        self.make_value("draft_scores", ir.DataType.FLOAT, ["batch_size", n_spec, top_k, top_k])
        self.make_node(
            "Cast", [cand], ["draft_candidate_ids"], name="/dflash2/selector/cand_cast", to=ir.DataType.INT32
        )
        self.make_value("draft_candidate_ids", ir.DataType.INT32, ["batch_size", n_spec, top_k])

        self.graph.outputs.append(self.values["draft_candidate_ids"])
        self.graph.outputs.append(self.values["draft_scores"])
        for i in range(self.num_layers):
            self.graph.outputs.append(self.values[f"present.{i}.key"])
            self.graph.outputs.append(self.values[f"present.{i}.value"])

    # --------------------------------------------------------------- weights

    def load_weights(self):
        import safetensors.torch as safetensors_torch  # noqa: PLC0415

        weights = {}
        for shard in sorted(glob.glob(os.path.join(self.draft_dir, "*.safetensors"))):
            with safetensors_torch.safe_open(shard, framework="pt") as f:
                for key in f.keys():  # noqa: SIM118 -- safetensors handles are not iterable
                    weights[key] = f.get_tensor(key)
        if "fc.weight" not in weights:
            raise ValueError(f"'{self.draft_dir}' does not look like a DFlash 2 draft checkpoint (no fc.weight).")

        embed_keys = {"model.embed_tokens.weight", "model.language_model.embed_tokens.weight"}
        for shard in sorted(glob.glob(os.path.join(self.target_dir, "*.safetensors"))):
            if os.path.basename(shard).startswith("model_mtp"):
                continue
            with safetensors_torch.safe_open(shard, framework="pt") as f:
                for key in f.keys():  # noqa: SIM118 -- safetensors handles are not iterable
                    if key in embed_keys:
                        weights["embed_tokens.weight"] = f.get_tensor(key)
                    elif key == "lm_head.weight":
                        weights["lm_head.weight"] = f.get_tensor(key)
                    elif key in ("lm_head.weight_scale", "lm_head.weight_global_scale"):
                        weights["lm_head.weight_scale"] = f.get_tensor(key)
        for required in ("embed_tokens.weight", "lm_head.weight"):
            if required not in weights:
                raise ValueError(f"Could not find '{required}' in the target checkpoint '{self.target_dir}'.")
        self.validate_weight_shapes(
            weights,
            {
                "fc.weight": (self.hidden_size, self.aux_hidden_size),
                "embed_tokens.weight": (self.vocab_size, self.hidden_size),
                "lm_head.weight": (self.vocab_size, self.hidden_size),
            },
        )
        self.weights = weights
        return weights
