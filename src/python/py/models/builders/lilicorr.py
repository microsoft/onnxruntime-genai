# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Builder for a LiLiCorr block drafter (``dflash2.onnx``), e.g. NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.

LiLiCorr is a DFlash 2 drafter whose low-rank candidate selector is replaced by a *candidate lattice
correlator*. The backbone is the DFlash 2 one (target aux hidden states -> ``fc`` -> per-layer context K/V,
non-causal block attention with two-tap dynamic grouped convolutions) plus a learned per-head attention sink.
The head then works on the ``block_size - 1`` mask rows:

1. the target's LM head gives top-``k`` candidates and their log-probabilities for every mask row;
2. every candidate becomes a token of a ``(n_slots * k)``-long lattice: ``token_proj(embedding) + pass_hidden_proj(hidden)
   + feature_mlp(log-prob features) + slot/rank embeddings``;
3. a small 2-layer transformer with a learned relative-slot attention bias mixes the lattice;
4. outgoing/incoming factor vectors are normalised and ``dot(out[s, i], in[s + 1, j])`` scores the transition from
   candidate ``i`` of slot ``s`` to candidate ``j`` of slot ``s + 1``; the anchor (the last committed context row,
   ``hidden_norm(fc(aux))``) supplies the first-slot scores.

The graph outputs the same lattice as DFlash 2 (``draft_candidate_ids`` ``[B, n, k]``, ``draft_scores``
``[B, n, k, k]`` with identical rows for the first slot), so the runtime's greedy walk is unchanged. The only extra
input is ``anchor_row_index`` ``[B]``: the packed ``aux_hidden_states`` row of each request's last context row (-1: none).

Weights: the checkpoint is modelopt NVFP4 (``W4A16``) for the large projections; they are dequantised to bf16 (default)
or re-quantised to int4/int8 block-wise. The embedding and LM head are the target's. For the Nemotron target the LM head
is ggml Q8_0 and is converted losslessly to an int8 ``MatMulNBits``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnx_ir as ir
import torch

from .dflash2 import DFlash2Builder


class LiLiCorrBuilder(DFlash2Builder):
    """Emits ``dflash2.onnx`` for a LiLiCorr drafter checkpoint plus the target's embedding and LM head."""

    def __init__(
        self,
        draft_dir,
        target_dir,
        paged_block_size,
        max_position_embeddings=None,
        filename="dflash2.onnx",
        num_draft_tokens=None,
        precision="bf16",
        lm_head_precision="int8",
    ):
        quant = None
        if precision in ("int4", "int8"):
            quant = {"bits": 4 if precision == "int4" else 8, "block_size": 32, "prepack": 0}
        elif precision != "bf16":
            raise ValueError(f"precision must be bf16, int4 or int8, got {precision!r}.")
        self.lm_head_precision = lm_head_precision
        super().__init__(
            draft_dir,
            target_dir,
            ir.DataType.BFLOAT16,
            paged_block_size,
            max_position_embeddings,
            filename=filename,
            num_draft_tokens=num_draft_tokens,
            quant=quant,
        )

    def make_drafter_config_init(self, cfg, max_position_embeddings, num_draft_tokens):
        if cfg.get("dflash_config", {}).get("projector_type") != "lilicorr" or not cfg["dflash_config"].get("lilicorr_enabled", True):
            raise ValueError(f"'{self.draft_dir}' is not a LiLiCorr draft checkpoint (dflash_config.projector_type != 'lilicorr').")
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
        if cfg.get("rope_scaling"):
            raise ValueError("LiLiCorr export supports only the default RoPE.")
        self.rope_theta = float(cfg["rope_theta"])
        self.sliding_window = self.resolve_sliding_window(cfg)
        self.is_causal = bool(dfl.get("causal", False))
        if self.is_causal:
            raise ValueError("LiLiCorr expects a non-causal block drafter.")
        if not dfl.get("attention_sink_bias", cfg.get("attention_sink_bias", False)):
            raise ValueError("LiLiCorr export expects attention_sink_bias=true.")

        self.taps = int(dfl["conv_kernel_size"])
        self.group_size = int(dfl["conv_group_size"])
        self.num_groups = self.hidden_size // self.group_size
        self.selector_top_k = int(dfl["lilicorr_candidate_topk"])
        self.selector_rank = 0
        self.mask_token_id = int(dfl["mask_token_id"])
        self.validate_token_metadata(cfg, dfl)
        self.target_layer_ids = list(dfl["target_layer_ids"])
        checkpoint_block_size = int(cfg["block_size"])
        if num_draft_tokens is not None and not 1 <= int(num_draft_tokens) < checkpoint_block_size:
            raise ValueError(f"num_draft_tokens must be between 1 and {checkpoint_block_size - 1}.")
        self.block_size = int(num_draft_tokens) + 1 if num_draft_tokens is not None else checkpoint_block_size
        self.num_draft_tokens = self.block_size - 1
        if self.block_size != checkpoint_block_size:
            # The lattice head's slot/relative-bias parameters are tied to the trained block; a shorter draft uses the
            # leading slots, but the conv block boundary and PagedAttention block shrink with it.
            print(f"LiLiCorr: drafting {self.num_draft_tokens} of {checkpoint_block_size - 1} trained slots.")
        self.input_embedding_scale = 1.0
        self.aux_hidden_size = self.hidden_size * len(self.target_layer_ids)

        self.lh = int(dfl["lilicorr_hidden_size"])
        self.l_layers = int(dfl["lilicorr_num_layers"])
        self.l_heads = int(dfl["lilicorr_num_heads"])
        self.l_mlp = int(self.lh * float(dfl["lilicorr_mlp_ratio"]))
        self.factor_dim = int(dfl["lilicorr_factor_dim"])
        self.vector_eps = float(dfl["lilicorr_vector_eps"])
        self.logit_scale = float(dfl["lilicorr_logit_scale"])

    # ---------------------------------------------------------------- weights

    def load_drafter_weights(self):
        try:
            from loaders.modelopt import ModeloptModel  # noqa: PLC0415
        except ImportError:
            from onnxruntime_genai.models.loaders.modelopt import ModeloptModel  # noqa: PLC0415

        weights = ModeloptModel.load_lilicorr_weights(
            self.draft_dir,
            self.target_dir,
            dequantize_lm_head=self.lm_head_precision != "int8",
        )
        expected_shapes = {
            "fc.weight": (self.hidden_size, self.aux_hidden_size),
            "embed_tokens.weight": (self.vocab_size, self.hidden_size),
        }
        if "lm_head.qweight" in weights:
            expected_shapes["lm_head.qweight"] = (self.vocab_size, self.hidden_size // 32, 32)
            expected_shapes["lm_head.scales"] = (self.vocab_size, self.hidden_size // 32)
        else:
            expected_shapes["lm_head.weight"] = (self.vocab_size, self.hidden_size)
        self.validate_weight_shapes(weights, expected_shapes)
        self.weights = weights
        return weights

    # ---------------------------------------------------------------- hooks

    def head_sink_input(self, layer_id):
        name = f"dflash2.layers.{layer_id}.self_attn.sinks"
        if name not in self.values:
            self.make_initializer(self.weights[f"layers.{layer_id}.self_attn.attention_sink_bias"], name, to=self.io_dtype)
        return name

    def declare_io(self):
        super().declare_io()
        self.graph.inputs.append(self.make_value("anchor_row_index", ir.DataType.INT32, ["batch_size"]))

    def genai_config_section(self):
        section = super().genai_config_section()
        section["inputs"]["anchor_row_index"] = "anchor_row_index"
        section["aux_hidden_state_layers"] = list(self.target_layer_ids)
        return section

    # ---------------------------------------------------------------- candidates + lattice head

    def make_l2_normalize(self, name, root_input, shape):
        """``root_input / max(||root_input||_2, eps)`` over the last axis, fp32."""
        sq = self.binary("Mul", f"{name}/sq", root_input, root_input, ir.DataType.FLOAT, shape)
        ss = self.binary(
            "ReduceSum", f"{name}/sum", sq, self.const([-1]), ir.DataType.FLOAT, [*shape[:-1], 1], keepdims=1
        )
        nrm = self.unary("Sqrt", f"{name}/sqrt", ss, ir.DataType.FLOAT, [*shape[:-1], 1])
        eps = self.const(np.full((1,), self.vector_eps, np.float32), ir.DataType.FLOAT)
        nrm = self.binary("Max", f"{name}/max", nrm, eps, ir.DataType.FLOAT, [*shape[:-1], 1])
        return self.binary("Div", f"{name}/div", root_input, nrm, ir.DataType.FLOAT, shape)

    def make_lattice_attention_bias(self):
        """``[1, heads, n*k, n*k]``: relative-slot bias + same-slot bias, as vLLM's ``_build_attention_bias`` (bf16-rounded)."""
        n, k = self.num_draft_tokens, self.selector_top_k
        rel_bias = self.weights["lilicorr.relative_slot_bias"].to(torch.bfloat16)  # [heads, 2*block-1]
        same = self.weights["lilicorr.same_slot_bias"].to(torch.bfloat16)
        slot = torch.arange(n).repeat_interleave(k)
        rel = slot.view(-1, 1) - slot.view(1, -1)
        bias = rel_bias[:, rel + int(self.cfg["block_size"]) - 1]
        bias = bias + (slot.view(-1, 1) == slot.view(1, -1)).unsqueeze(0).to(bias.dtype) * same.view(-1, 1, 1)
        return bias.to(torch.float32).unsqueeze(0).contiguous()

    def make_candidates(self, final):
        """Project the ``block_size - 1`` mask rows through the target LM head to top-``k`` candidates and log-probs."""
        n, k, h = self.num_draft_tokens, self.selector_top_k, self.hidden_size
        p = "/lilicorr"
        bf, f32 = self.io_dtype, ir.DataType.FLOAT

        h3 = self.reshape(f"{p}/select/Reshape", final, [-1, self.block_size, h], bf, ["batch_size", self.block_size, h])
        sl = self.make_slice_range(f"{p}/select/Slice", h3, 1, self.block_size, 1, bf, ["batch_size", n, h])
        hsel = self.reshape(f"{p}/select/Flatten", sl, [-1, h], bf, ["num_sample", h])
        logits = self.make_lm_head(hsel, "num_sample")
        logits32 = self.make_cast(f"{p}/logits/Cast", logits, f32, ["num_sample", self.vocab_size])
        vals, idx = self.make_topk(
            f"{p}/topk/TopK", logits32, k, f"{p}/topk/values", f"{p}/topk/indices", ["num_sample", k]
        )
        # Log-partition from the (descending) top-k max. ReduceLogSumExp / ReduceMax over a [rows, vocab] tensor run a
        # cuDNN reduction that is ~100x slower than ORT's own last-axis ReduceSum.
        top_logit = self.make_slice_range(f"{p}/logits/max", vals, 0, 1, 1, f32, ["num_sample", 1])
        shifted = self.binary("Sub", f"{p}/logits/shift", logits32, top_logit, f32, ["num_sample", self.vocab_size])
        expd = self.unary("Exp", f"{p}/logits/exp", shifted, f32, ["num_sample", self.vocab_size])
        denom = self.binary(
            "ReduceSum", f"{p}/logits/sum", expd, self.const([-1]), f32, ["num_sample", 1], keepdims=1
        )
        log_denom = self.unary("Log", f"{p}/logits/log", denom, f32, ["num_sample", 1])
        lse = self.binary("Add", f"{p}/logits/lse", log_denom, top_logit, f32, ["num_sample", 1])
        logp = self.binary("Sub", f"{p}/logp", vals, lse, f32, ["num_sample", k])
        cand = self.reshape(f"{p}/cand", idx, [-1, n, k], ir.DataType.INT64, ["batch_size", n, k])
        return hsel, idx, logp, cand

    def make_candidate_features(self, logp):
        """Build the 5 per-candidate log-probability features and project them through ``feature_mlp``."""
        k, lh = self.selector_top_k, self.lh
        w = self.weights
        p = "/lilicorr"
        bf, f32 = self.io_dtype, ir.DataType.FLOAT

        prob = self.unary("Exp", f"{p}/feat/exp", logp, f32, ["num_sample", k])
        mx = self.make_slice_range(f"{p}/feat/max", logp, 0, 1, 1, f32, ["num_sample", 1])
        rel = self.binary("Sub", f"{p}/feat/rel", logp, mx, f32, ["num_sample", k])
        cols = [
            self.binary("Unsqueeze", f"{p}/feat/u{i}", tensor, self.const([-1]), f32, ["num_sample", k, 1])
            for i, tensor in enumerate((logp, prob, rel))
        ]
        shape_of = self.unary("Shape", f"{p}/feat/shape", cols[0], ir.DataType.INT64, [3])
        rank_frac = np.arange(k, dtype=np.float32).reshape(1, k, 1) / max(k - 1, 1)
        top1 = np.zeros((1, k, 1), np.float32)
        top1[0, 0, 0] = 1.0
        for i, arr in enumerate((rank_frac, top1)):
            const_col = self.const(arr, ir.DataType.FLOAT)
            cols.append(self.binary("Expand", f"{p}/feat/c{i}", const_col, shape_of, f32, ["num_sample", k, 1]))
        feats = self.make_concat(f"{p}/feat/concat", cols, f32, ["num_sample", k, 5], axis=-1)
        feats = self.make_cast(f"{p}/feat/cast", feats, bf, ["num_sample", k, 5])
        ln_out = self.make_layer_norm(
            f"{p}/feat/ln",
            feats,
            w["lilicorr.feature_norm.weight"],
            w["lilicorr.feature_norm.bias"],
            ["num_sample", k, 5],
            weight_name="lilicorr.feature_norm.weight",
            bias_name="lilicorr.feature_norm.bias",
        )
        up = self.make_linear(
            f"{p}/feature_mlp/up",
            ln_out,
            w["lilicorr.feature_mlp.up_proj.weight"],
            w["lilicorr.feature_mlp.up_proj.bias"],
            ["num_sample", k, lh],
        )
        up = self.make_silu(f"{p}/feature_mlp/act", up, bf, ["num_sample", k, lh])
        return self.make_linear(
            f"{p}/feature_mlp/down",
            up,
            w["lilicorr.feature_mlp.down_proj.weight"],
            w["lilicorr.feature_mlp.down_proj.bias"],
            ["num_sample", k, lh],
        )

    def make_lattice_inputs(self, hsel, idx, logp):
        """Combine token embeddings, pass hidden states, log-prob features, and slot/rank embeddings."""
        n, k, lh, h = self.num_draft_tokens, self.selector_top_k, self.lh, self.hidden_size
        w = self.weights
        p = "/lilicorr"
        bf = self.io_dtype

        cand_emb = self.binary(
            "Gather", f"{p}/cand_embed", "model.embed_tokens.weight", idx, self.external_dtype, ["num_sample", k, h]
        )
        if self.io_dtype != self.external_dtype:
            cand_emb = self.make_cast(f"{p}/cand_embed/Cast", cand_emb, bf, ["num_sample", k, h])
        tok = self.make_linear(
            f"{p}/token_proj", cand_emb, w["lilicorr.token_proj.weight"], w["lilicorr.token_proj.bias"], ["num_sample", k, lh]
        )
        pas = self.make_linear(
            f"{p}/pass_hidden_proj",
            hsel,
            w["lilicorr.pass_hidden_proj.weight"],
            w["lilicorr.pass_hidden_proj.bias"],
            ["num_sample", lh],
        )
        pas = self.binary("Unsqueeze", f"{p}/pass_unsq", pas, self.const([1]), bf, ["num_sample", 1, lh])
        fm = self.make_candidate_features(logp)

        hid = self.binary("Add", f"{p}/hid/a0", tok, pas, bf, ["num_sample", k, lh])
        hid = self.binary("Add", f"{p}/hid/a1", hid, fm, bf, ["num_sample", k, lh])
        hid = self.reshape(f"{p}/hid/r4", hid, [-1, n, k, lh], bf, ["batch_size", n, k, lh])
        slot = w["lilicorr.slot_embedding"][0, 0, :n]  # [n, 1, lh]
        rank = w["lilicorr.rank_embedding"][0, 0, 0]  # [k, lh]
        self.make_initializer(slot.reshape(1, n, 1, lh), "lilicorr.slot_embedding", to=bf)
        self.make_initializer(rank.reshape(1, 1, k, lh), "lilicorr.rank_embedding", to=bf)
        hid = self.binary("Add", f"{p}/hid/a2", hid, "lilicorr.slot_embedding", bf, ["batch_size", n, k, lh])
        hid = self.binary("Add", f"{p}/hid/a3", hid, "lilicorr.rank_embedding", bf, ["batch_size", n, k, lh])
        lat = n * k
        return self.reshape(f"{p}/hid/flat", hid, [-1, lat, lh], bf, ["batch_size", lat, lh])

    def make_lattice_layer(self, layer_id, x, scale):
        """One relative-slot biased self-attention + SiLU-MLP transformer layer over the ``(n * k)`` lattice."""
        lat, lh = self.num_draft_tokens * self.selector_top_k, self.lh
        heads, hdim = self.l_heads, lh // self.l_heads
        w = self.weights
        q = f"/lilicorr/layers.{layer_id}"
        bf, f32 = self.io_dtype, ir.DataType.FLOAT

        xn = self.rms_norm(f"{q}/attn_norm", x, w[f"lilicorr.layers.{layer_id}.attn_norm.weight"], ["batch_size", lat, lh])
        qkv = self.make_linear(
            f"{q}/in_proj",
            xn,
            w[f"lilicorr.layers.{layer_id}.attn.in_proj_weight"],
            w[f"lilicorr.layers.{layer_id}.attn.in_proj_bias"],
            ["batch_size", lat, 3 * lh],
        )
        parts = self.make_split_equal(
            f"{q}/qkv/split", qkv, [f"{q}/qkv/split_{i}" for i in range(3)], bf, ["batch_size", lat, lh], axis=-1
        )
        hs = []
        for i, part in enumerate(parts):
            r = self.reshape(f"{q}/qkv/r{i}", part, [-1, lat, heads, hdim], bf, ["batch_size", lat, heads, hdim])
            perm = [0, 2, 3, 1] if i == 1 else [0, 2, 1, 3]
            shape = ["batch_size", heads, hdim, lat] if i == 1 else ["batch_size", heads, lat, hdim]
            hs.append(self.unary("Transpose", f"{q}/qkv/t{i}", r, bf, shape, perm=perm))
        sc = self.binary("MatMul", f"{q}/scores", hs[0], hs[1], bf, ["batch_size", heads, lat, lat])
        sc = self.make_cast(f"{q}/scores/cast", sc, f32, ["batch_size", heads, lat, lat])
        sc = self.binary("Mul", f"{q}/scores/scale", sc, scale, f32, ["batch_size", heads, lat, lat])
        sc = self.binary("Add", f"{q}/scores/bias", sc, "lilicorr.attn_bias", f32, ["batch_size", heads, lat, lat])
        pr = self.unary("Softmax", f"{q}/softmax", sc, f32, ["batch_size", heads, lat, lat], axis=-1)
        pr = self.make_cast(f"{q}/softmax/cast", pr, bf, ["batch_size", heads, lat, lat])
        ao = self.binary("MatMul", f"{q}/context", pr, hs[2], bf, ["batch_size", heads, lat, hdim])
        ao = self.unary("Transpose", f"{q}/context/t", ao, bf, ["batch_size", lat, heads, hdim], perm=[0, 2, 1, 3])
        ao = self.reshape(f"{q}/context/flat", ao, [-1, lat, lh], bf, ["batch_size", lat, lh])
        ao = self.make_linear(
            f"{q}/out_proj",
            ao,
            w[f"lilicorr.layers.{layer_id}.attn.out_proj.weight"],
            w[f"lilicorr.layers.{layer_id}.attn.out_proj.bias"],
            ["batch_size", lat, lh],
        )
        x = self.binary("Add", f"{q}/attn_res", x, ao, bf, ["batch_size", lat, lh])
        mn = self.rms_norm(f"{q}/mlp_norm", x, w[f"lilicorr.layers.{layer_id}.mlp_norm.weight"], ["batch_size", lat, lh])
        m = self.make_linear(
            f"{q}/mlp/up",
            mn,
            w[f"lilicorr.layers.{layer_id}.mlp.up_proj.weight"],
            w[f"lilicorr.layers.{layer_id}.mlp.up_proj.bias"],
            ["batch_size", lat, self.l_mlp],
        )
        m = self.make_silu(f"{q}/mlp/act", m, bf, ["batch_size", lat, self.l_mlp])
        m = self.make_linear(
            f"{q}/mlp/down",
            m,
            w[f"lilicorr.layers.{layer_id}.mlp.down_proj.weight"],
            w[f"lilicorr.layers.{layer_id}.mlp.down_proj.bias"],
            ["batch_size", lat, lh],
        )
        return self.binary("Add", f"{q}/mlp_res", x, m, bf, ["batch_size", lat, lh])

    def make_lattice_transformer(self, x):
        """Run the lattice transformer layers and return the normalized ``[batch_size, n, k, lh]`` states."""
        n, k, lh = self.num_draft_tokens, self.selector_top_k, self.lh
        lat = n * k
        hdim = lh // self.l_heads
        self.make_initializer(self.make_lattice_attention_bias(), "lilicorr.attn_bias", to=ir.DataType.FLOAT)
        scale = self.const(np.full((1,), hdim**-0.5, np.float32), ir.DataType.FLOAT)
        for layer_id in range(self.l_layers):
            x = self.make_lattice_layer(layer_id, x, scale)
        x = self.rms_norm(
            "/lilicorr/output_norm", x, self.weights["lilicorr.output_norm.weight"], ["batch_size", lat, lh]
        )
        return self.reshape("/lilicorr/out4", x, [-1, n, k, lh], self.io_dtype, ["batch_size", n, k, lh])

    def make_anchor_state(self):
        """Extract ``hidden_norm(fc(aux)[anchor_row_index])`` and project/mask it to ``[batch_size, lh]``."""
        lh, h = self.lh, self.hidden_size
        w = self.weights
        p = "/lilicorr"
        bf = self.io_dtype

        idx32 = "anchor_row_index"
        valid = self.binary(
            "Greater", f"{p}/anchor/valid", idx32, self.const(np.array([-1], np.int32), ir.DataType.INT32), ir.DataType.BOOL, ["batch_size"]
        )
        valid = self.make_cast(f"{p}/anchor/valid_cast", valid, bf, ["batch_size"])
        valid = self.binary("Unsqueeze", f"{p}/anchor/valid_unsq", valid, self.const([-1]), bf, ["batch_size", 1])
        safe = self.binary(
            "Max", f"{p}/anchor/safe", idx32, self.const(np.array([0], np.int32), ir.DataType.INT32), ir.DataType.INT32, ["batch_size"]
        )
        row = self.binary("Gather", f"{p}/anchor/row", self.ctx_pre_norm, safe, bf, ["batch_size", h], axis=0)
        row = self.rms_norm(
            f"{p}/anchor/hidden_norm", row, w["hidden_norm.weight"], "batch_size", weight_name="dflash2.hidden_norm.weight"
        )
        anchor = self.make_linear(
            f"{p}/context_proj", row, w["lilicorr.context_proj.weight"], w["lilicorr.context_proj.bias"], ["batch_size", lh]
        )
        anchor = self.binary("Mul", f"{p}/anchor/mask", anchor, valid, bf, ["batch_size", lh])
        return self.rms_norm(f"{p}/anchor_norm", anchor, w["lilicorr.anchor_norm.weight"], ["batch_size", lh])

    def make_factor_vectors(self, hid4, anchor):
        """Compute L2-normalized outgoing/incoming lattice factor vectors and the anchor outgoing vector."""
        n, k, lh, fd = self.num_draft_tokens, self.selector_top_k, self.lh, self.factor_dim
        w = self.weights
        p = "/lilicorr"
        bf, f32 = self.io_dtype, ir.DataType.FLOAT

        fw = w["lilicorr.factor_input_proj.weight"]
        w_self, w_anchor, w_cross = (fw[:, i * lh : (i + 1) * lh].contiguous() for i in range(3))
        a4 = self.reshape(f"{p}/anchor/a4", anchor, [-1, 1, 1, lh], bf, ["batch_size", 1, 1, lh])
        pre = self.make_linear(f"{p}/factor/self", hid4, w_self, w["lilicorr.factor_input_proj.bias"], ["batch_size", n, k, lh])
        pa = self.make_linear(f"{p}/factor/anchor", a4, w_anchor, None, ["batch_size", 1, 1, lh])
        cross_in = self.binary("Mul", f"{p}/factor/cross_in", hid4, a4, bf, ["batch_size", n, k, lh])
        pc = self.make_linear(f"{p}/factor/cross", cross_in, w_cross, None, ["batch_size", n, k, lh])
        pre = self.binary("Add", f"{p}/factor/add0", pre, pa, bf, ["batch_size", n, k, lh])
        pre = self.binary("Add", f"{p}/factor/add1", pre, pc, bf, ["batch_size", n, k, lh])
        fh = self.make_silu(f"{p}/factor/act", pre, bf, ["batch_size", n, k, lh])
        ew = torch.cat([w["lilicorr.out_head.weight"], w["lilicorr.in_head.weight"]], 0)
        eb = torch.cat([w["lilicorr.out_head.bias"], w["lilicorr.in_head.bias"]], 0)
        edges = self.make_linear(f"{p}/edges", fh, ew, eb, ["batch_size", n, k, 2 * fd])
        edges = self.make_cast(f"{p}/edges/cast", edges, f32, ["batch_size", n, k, 2 * fd])
        out_raw, in_raw = self.make_split_equal(
            f"{p}/edges/split", edges, [f"{p}/edges/out", f"{p}/edges/in"], f32, ["batch_size", n, k, fd], axis=-1
        )
        out_vec = self.make_l2_normalize(f"{p}/out_norm", out_raw, ["batch_size", n, k, fd])
        in_vec = self.make_l2_normalize(f"{p}/in_norm", in_raw, ["batch_size", n, k, fd])

        ao_raw = self.make_linear(
            f"{p}/anchor_out_head", anchor, w["lilicorr.anchor_out_head.weight"], w["lilicorr.anchor_out_head.bias"], ["batch_size", fd]
        )
        ao_raw = self.make_cast(f"{p}/anchor_out_head/cast", ao_raw, f32, ["batch_size", fd])
        anchor_out = self.make_l2_normalize(f"{p}/anchor_out_norm", ao_raw, ["batch_size", fd])
        anchor_out = self.reshape(f"{p}/anchor_out4", anchor_out, [-1, 1, 1, fd], f32, ["batch_size", 1, 1, fd])
        return out_vec, in_vec, anchor_out

    def make_lattice_scores(self, hid4, anchor, cand):
        """Score first-slot and pairwise slot transitions and emit ``draft_scores`` and ``draft_candidate_ids``."""
        n, k, fd = self.num_draft_tokens, self.selector_top_k, self.factor_dim
        p = "/lilicorr"
        f32 = ir.DataType.FLOAT
        out_vec, in_vec, anchor_out = self.make_factor_vectors(hid4, anchor)

        # The start scores are row 0 of anchor . in_vec. Scoring every row and slicing the result keeps `in_vec`
        # consumed by a single Slice: with two slices of it (rows 0 and 1..) ORT fuses them into a Split whose kernel
        # allocates per call, which cannot be captured in a CUDA graph.
        start = self.binary("Mul", f"{p}/start/mul", in_vec, anchor_out, f32, ["batch_size", n, k, fd])
        start = self.binary("ReduceSum", f"{p}/start/sum", start, self.const([-1]), f32, ["batch_size", n, k], keepdims=0)
        start = self.make_slice_range(f"{p}/start/first", start, 0, 1, 1, f32, ["batch_size", 1, k])
        start = self.binary("Unsqueeze", f"{p}/start/unsq", start, self.const([2]), f32, ["batch_size", 1, 1, k])
        start = self.binary("Expand", f"{p}/start/expand", start, self.const([1, 1, k, k]), f32, ["batch_size", 1, k, k])
        scale_c = self.const(np.full((1,), self.logit_scale, np.float32), f32)
        start = self.binary("Mul", f"{p}/start/scale", start, scale_c, f32, ["batch_size", 1, k, k])

        out_head = self.make_slice_range(f"{p}/out_head", out_vec, 0, n - 1, 1, f32, ["batch_size", n - 1, k, fd])
        in_tail = self.make_slice_range(f"{p}/in_tail", in_vec, 1, n, 1, f32, ["batch_size", n - 1, k, fd])
        in_tail_t = self.unary("Transpose", f"{p}/in_tail_t", in_tail, f32, ["batch_size", n - 1, fd, k], perm=[0, 1, 3, 2])
        pairs = self.binary("MatMul", f"{p}/pairs", out_head, in_tail_t, f32, ["batch_size", n - 1, k, k])
        pairs = self.binary("Mul", f"{p}/pairs/scale", pairs, scale_c, f32, ["batch_size", n - 1, k, k])
        self.make_concat(f"{p}/scores", [start, pairs], f32, ["batch_size", n, k, k], axis=1, output="draft_scores")
        self.make_cast(f"{p}/cand_cast", cand, ir.DataType.INT32, ["batch_size", n, k], output="draft_candidate_ids")

    def make_candidates_and_selector(self, final, w):
        hsel, idx, logp, cand = self.make_candidates(final)
        lattice = self.make_lattice_inputs(hsel, idx, logp)
        hid4 = self.make_lattice_transformer(lattice)
        anchor = self.make_anchor_state()
        self.make_lattice_scores(hid4, anchor, cand)

        self.graph.outputs.append(self.values["draft_candidate_ids"])
        self.graph.outputs.append(self.values["draft_scores"])
        for i in range(self.num_layers):
            self.graph.outputs.append(self.values[f"present.{i}.key"])
            self.graph.outputs.append(self.values[f"present.{i}.value"])


def main():
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="Export a LiLiCorr drafter (dflash2.onnx) for onnxruntime-genai.")
    ap.add_argument("--draft_dir", required=True)
    ap.add_argument("--target_dir", required=True, help="Target checkpoint (embedding + LM head).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--paged_block_size", type=int, default=256)
    ap.add_argument("--num_draft_tokens", type=int, default=None)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "int4", "int8"])
    ap.add_argument("--lm_head_precision", default="int8", choices=["int8", "bf16"])
    ap.add_argument("--filename", default="dflash2.onnx")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    b = LiLiCorrBuilder(
        a.draft_dir, a.target_dir, a.paged_block_size, num_draft_tokens=a.num_draft_tokens,
        filename=a.filename, precision=a.precision, lm_head_precision=a.lm_head_precision,
    )
    b.make_model()
    b.save_model(a.output_dir)
    with open(os.path.join(a.output_dir, "dflash2_genai_config_section.json"), "w") as f:
        json.dump({"dflash2": b.genai_config_section()}, f, indent=2)
    print("saved", os.path.join(a.output_dir, a.filename))


if __name__ == "__main__":
    main()
