# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Builder for a DSpark block-diffusion draft model (``dspark.onnx``).

DSpark is SpecForge's DFlash backbone plus a low-rank Markov (bigram) head. Like DFlash 2 it
is a *block* drafter: one pass over the target's auxiliary hidden states proposes a whole
block of tokens. It differs from DFlash 2 in four ways that are all visible in the graph:

1. **No dynamic convolutions.** The layers are plain pre-norm Qwen3 blocks.
2. **Checkpoint-specific full attention.** This Qwen3.8 checkpoint has only
    ``full_attention`` layers, so its KV cache covers the whole context
    (``local_window_size = -1``). DSpark itself can use sliding-window attention; for a
    uniformly windowed checkpoint the builder forwards its positive ``sliding_window`` and
    the runtime uses a fixed cache ring.
3. **YaRN RoPE** (factor 32 over an 8192-token pretraining window) rather than default RoPE.
4. **A Markov head instead of a candidate selector.** ``markov_w2[cur] . markov_w1[prev]`` is a
   learned bigram bias added to the draft logits. Restricted to the per-position top-k it is
   exactly the pairwise edge score of a lattice, so the runtime walks it with the same greedy
   path search DFlash 2 uses -- ``draft_candidate_ids`` / ``draft_scores`` mean the same thing.

Every block row predicts: row ``j`` carries the anchor token's embedding when ``j == 0`` and a
MASK embedding otherwise, and predicts the token at ``anchor_position + j + 1``. (DFlash 2's
row 0 predicts nothing, which is why its block is one row wider for the same draft count.)

The drafter checkpoint ships no embedding and no LM head; both are the target's and are
emitted with the target's initializer names so ``share_external_initializers`` can fold them
back onto ``text.onnx.data``.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import onnx_ir as ir
import torch

from .block_drafter import BlockDrafterBuilder


class DSparkBuilder(BlockDrafterBuilder):
    """Emits ``dspark.onnx`` from a DSpark draft checkpoint plus the target's embedding and
    LM head."""

    def __init__(
        self,
        draft_dir,
        target_dir,
        io_dtype,
        paged_block_size,
        max_position_embeddings,
        filename="dspark.onnx",
        num_draft_tokens=None,
        top_k=16,
    ):
        self.draft_dir = draft_dir
        self.target_dir = target_dir
        # The drafter is a bf16 checkpoint whose activations leave the fp16 range (the fc output
        # alone reaches ~1e4), so the body runs in bf16. Only the tensors it shares with the fp16
        # target -- the aux hidden states, the embedding table and the FP8 LM head -- stay at the
        # target's dtype.
        self.io_dtype = ir.DataType.BFLOAT16
        self.external_dtype = io_dtype
        self.filename = filename
        self.paged_block_size = paged_block_size

        with open(os.path.join(draft_dir, "config.json")) as f:
            cfg = json.load(f)
        self.cfg = cfg
        dfl = cfg.get("dflash_config") or {}
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
        self.rope_parameters = dict(cfg["rope_parameters"])
        rope_type = str(self.rope_parameters.get("rope_type", "default")).lower()
        if rope_type not in ("default", "yarn"):
            raise ValueError(f"DSpark does not support the '{rope_type}' RoPE type; expected 'default' or 'yarn'.")
        self.sliding_window = int(cfg["sliding_window"]) if cfg.get("use_sliding_window") else -1
        # The dual-source block attends to itself bidirectionally; DFlash's attention module
        # hard-codes is_causal=False and the published configs do not override it.
        self.is_causal = bool(cfg.get("is_causal", False))

        self.mask_token_id = int(dfl["mask_token_id"])
        self.validate_token_metadata(cfg, dfl)
        self.target_layer_ids = list(dfl["target_layer_ids"])
        self.markov_rank = int(cfg.get("markov_rank", dfl.get("markov_rank", 0)))
        if self.markov_rank <= 0:
            raise ValueError("A DSpark drafter needs markov_rank > 0.")
        # One query row per draft; a smaller request just takes a prefix of the same block.
        checkpoint_block_size = int(cfg["block_size"])
        try:
            self.block_size = int(num_draft_tokens) if num_draft_tokens is not None else checkpoint_block_size
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"num_draft_tokens must be an integer between 2 and the checkpoint block size ({checkpoint_block_size})."
            ) from error
        if not 2 <= self.block_size <= checkpoint_block_size:
            raise ValueError(
                f"num_draft_tokens must be between 2 and the checkpoint block size ({checkpoint_block_size})."
            )
        self.num_draft_tokens = self.block_size
        try:
            self.top_k = int(top_k)
        except (TypeError, ValueError) as error:
            raise ValueError("top_k must be a positive integer.") from error
        if not 1 <= self.top_k <= self.vocab_size:
            raise ValueError(f"top_k must be between 1 and the vocabulary size ({self.vocab_size}).")
        self.selector_top_k = self.top_k
        self.aux_hidden_size = self.hidden_size * len(self.target_layer_ids)
        self.make_graph("dspark_graph", "dspark")

    # ------------------------------------------------------------ rope caches

    def yarn_inv_freq(self):
        """HF ``_compute_yarn_parameters`` for this checkpoint's ``rope_parameters``."""
        params = self.rope_parameters
        base = float(params["rope_theta"])
        dim = int(self.head_size * float(params.get("partial_rotary_factor", 1.0)))
        factor = float(params["factor"])
        original_max = float(params["original_max_position_embeddings"])
        beta_fast = float(params.get("beta_fast") or 32)
        beta_slow = float(params.get("beta_slow") or 1)

        attention_factor = params.get("attention_factor")
        if attention_factor is None:
            attention_factor = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0

        def correction_dim(rotations):
            return (dim * math.log(original_max / (rotations * 2 * math.pi))) / (2 * math.log(base))

        low, high = correction_dim(beta_fast), correction_dim(beta_slow)
        if params.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001

        pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
        extrapolation = 1.0 / pos_freqs
        interpolation = 1.0 / (factor * pos_freqs)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0.0, 1.0)
        weight = 1.0 - ramp
        inv_freq = interpolation * (1.0 - weight) + extrapolation * weight
        return inv_freq.astype(np.float64), float(attention_factor)

    def make_rope_caches(self):
        # PagedAttention type-constrains cos/sin to the query's element type. YaRN's attention
        # factor scales cos/sin, exactly as HF's rotary embedding does, so it is baked in here.
        if str(self.rope_parameters.get("rope_type", "default")).lower() == "yarn":
            inv_freq, attention_factor = self.yarn_inv_freq()
        else:
            dim = self.head_size
            base = float(self.rope_parameters["rope_theta"])
            inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
            attention_factor = 1.0
        pos = np.arange(self.max_position, dtype=np.float64)[:, None]
        freqs = pos * inv_freq[None, :]
        for name, fn in (("dspark.cos_cache", np.cos), ("dspark.sin_cache", np.sin)):
            values = (fn(freqs) * attention_factor).astype(np.float32)
            self.make_initializer(torch.from_numpy(values), name, to=self.io_dtype)

    # ------------------------------------------------------------------ graph

    def make_model(self):
        w = self.load_weights()
        self.declare_io()
        self.make_rope_caches()

        rows_q = "num_block"

        # --- context path: the target's aux hidden states become every layer's K/V ---
        aux = self.unary(
            "Cast",
            "/dspark/aux/Cast",
            "aux_hidden_states",
            self.io_dtype,
            ["num_ctx", self.aux_hidden_size],
            to=self.io_dtype,
        )
        ctx = self.matmul("/dspark/fc/MatMul", aux, w["fc.weight"], self.aux_hidden_size, self.hidden_size, "num_ctx")
        ctx_n = self.rms_norm("/dspark/hidden_norm", ctx, w["hidden_norm.weight"], "num_ctx")
        ctx_kv = []
        for i in range(self.num_layers):
            k = self.matmul(
                f"/dspark/layers.{i}/ctx_k/MatMul",
                ctx_n,
                w[f"layers.{i}.self_attn.k_proj.weight"],
                self.hidden_size,
                self.num_kv_heads * self.head_size,
                "num_ctx",
                weight_name=f"dspark.layers.{i}.self_attn.k_proj.weight",
            )
            v = self.matmul(
                f"/dspark/layers.{i}/ctx_v/MatMul",
                ctx_n,
                w[f"layers.{i}.self_attn.v_proj.weight"],
                self.hidden_size,
                self.num_kv_heads * self.head_size,
                "num_ctx",
                weight_name=f"dspark.layers.{i}.self_attn.v_proj.weight",
            )
            ctx_kv.append((k, v))

        # --- query path ---
        self.make_initializer(w["embed_tokens.weight"], "model.embed_tokens.weight", to=self.external_dtype)
        emb_ext = self.binary(
            "Gather",
            "/dspark/embed_tokens/Gather",
            "model.embed_tokens.weight",
            "input_ids",
            self.external_dtype,
            [rows_q, self.hidden_size],
        )
        hidden = self.unary(
            "Cast", "/dspark/embed_tokens/Cast", emb_ext, self.io_dtype, [rows_q, self.hidden_size], to=self.io_dtype
        )

        residual = None
        for i in range(self.num_layers):
            p = f"/dspark/layers.{i}"
            if residual is None:
                x = self.rms_norm(f"{p}/input_layernorm", hidden, w[f"layers.{i}.input_layernorm.weight"], rows_q)
                residual = hidden
            else:
                x, residual = self.skip_rms_norm(
                    f"{p}/input_layernorm", residual, hidden, w[f"layers.{i}.input_layernorm.weight"], rows_q
                )

            attn_out = self.make_attention(i, x, ctx_kv[i], rows_q)
            y, residual = self.skip_rms_norm(
                f"{p}/post_attention_layernorm",
                residual,
                attn_out,
                w[f"layers.{i}.post_attention_layernorm.weight"],
                rows_q,
            )
            hidden = self.make_mlp(i, y, w, rows_q)

        final, _ = self.skip_rms_norm("/dspark/norm", residual, hidden, w["norm.weight"], rows_q, want_sum=False)

        self.make_candidates_and_markov(final, w)
        self.graph.sort()
        return self.model

    def make_attention(self, i, x, ctx_kv, rows_q):
        p = f"/dspark/layers.{i}"
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
            weight_name=f"dspark.layers.{i}.self_attn.k_proj.weight",
        )
        v = self.matmul(
            f"{p}/attn/v_proj/MatMul",
            x,
            w[f"layers.{i}.self_attn.v_proj.weight"],
            self.hidden_size,
            self.num_kv_heads * self.head_size,
            rows_q,
            weight_name=f"dspark.layers.{i}.self_attn.v_proj.weight",
        )

        # Interleave the block rows and the context rows into one packed token stream.
        # `qkv_row_map` indexes concat(block, context); `q_row_map` indexes the block rows alone
        # (context rows point at row 0 and their output is dropped).
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
            w[f"layers.{i}.self_attn.q_norm.weight"], f"dspark.layers.{i}.self_attn.q_norm.weight", to=self.io_dtype
        )
        k_norm = self.make_initializer(
            w[f"layers.{i}.self_attn.k_norm.weight"], f"dspark.layers.{i}.self_attn.k_norm.weight", to=self.io_dtype
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
                "dspark.cos_cache",
                "dspark.sin_cache",
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

    def make_mlp(self, i, x, w, rows_q):
        p = f"/dspark/layers.{i}/mlp"
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

    def make_candidates_and_markov(self, final, w):
        """Top-k per position plus the Markov bigram bias, packaged as a candidate lattice."""
        n_spec, top_k, rank = self.num_draft_tokens, self.top_k, self.markov_rank

        logits = self.make_lm_head(final, "num_block")
        logits32 = self.unary(
            "Cast", "/dspark/topk/Cast", logits, ir.DataType.FLOAT, ["num_block", self.vocab_size], to=ir.DataType.FLOAT
        )
        vals, idx = "/dspark/topk/values", "/dspark/topk/indices"
        self.make_node(
            "TopK", [logits32, self.const([top_k])], [vals, idx], name="/dspark/topk/TopK", axis=-1, largest=1, sorted=1
        )
        self.make_value(vals, ir.DataType.FLOAT, ["num_block", top_k])
        self.make_value(idx, ir.DataType.INT64, ["num_block", top_k])

        cand = self.reshape(
            "/dspark/topk/cand", idx, [-1, n_spec, top_k], ir.DataType.INT64, ["batch_size", n_spec, top_k]
        )
        unary_logits = self.reshape(
            "/dspark/topk/unary", vals, [-1, n_spec, top_k], ir.DataType.FLOAT, ["batch_size", n_spec, top_k]
        )

        # The bias at slot l is conditioned on the token at slot l-1: the anchor for l == 0 and
        # slot l-1's own candidate afterwards. That is a pairwise edge, so it walks like a lattice.
        ids2 = self.reshape(
            "/dspark/markov/ids", "input_ids", [-1, self.block_size], ir.DataType.INT64, ["batch_size", self.block_size]
        )
        anchor = self.out("/dspark/markov/anchor")
        self.make_node(
            "Slice", [ids2, self.const([0]), self.const([1]), self.const([1])], [anchor], name="/dspark/markov/anchor"
        )
        self.make_value(anchor, ir.DataType.INT64, ["batch_size", 1])
        anchor3 = self.reshape("/dspark/markov/anchor3", anchor, [-1, 1, 1], ir.DataType.INT64, ["batch_size", 1, 1])
        anchor_tiled = self.binary(
            "Tile",
            "/dspark/markov/anchor_tile",
            anchor3,
            self.const([1, 1, top_k]),
            ir.DataType.INT64,
            ["batch_size", 1, top_k],
        )
        prev_tail = self.out("/dspark/markov/prev_tail")
        self.make_node(
            "Slice",
            [cand, self.const([0]), self.const([n_spec - 1]), self.const([1])],
            [prev_tail],
            name="/dspark/markov/prev_tail",
        )
        self.make_value(prev_tail, ir.DataType.INT64, ["batch_size", n_spec - 1, top_k])
        prev = self.out("/dspark/markov/prev")
        self.make_node("Concat", [anchor_tiled, prev_tail], [prev], name="/dspark/markov/prev", axis=1)
        self.make_value(prev, ir.DataType.INT64, ["batch_size", n_spec, top_k])

        self.make_initializer(w["markov_head.markov_w1.weight"], "dspark.markov_w1", to=self.io_dtype)
        self.make_initializer(w["markov_head.markov_w2.weight"], "dspark.markov_w2", to=self.io_dtype)
        pred = self.binary(
            "Gather",
            "/dspark/markov/pred_gather",
            "dspark.markov_w1",
            prev,
            self.io_dtype,
            ["batch_size", n_spec, top_k, rank],
        )
        succ = self.binary(
            "Gather",
            "/dspark/markov/succ_gather",
            "dspark.markov_w2",
            cand,
            self.io_dtype,
            ["batch_size", n_spec, top_k, rank],
        )
        # The bias is added to a float32 logit, and a bf16 rank-256 dot product carries enough
        # relative error to reorder the lattice, so the (tiny) gathered slices go to float32 first.
        pred32 = self.unary(
            "Cast",
            "/dspark/markov/pred_cast",
            pred,
            ir.DataType.FLOAT,
            ["batch_size", n_spec, top_k, rank],
            to=ir.DataType.FLOAT,
        )
        succ32 = self.unary(
            "Cast",
            "/dspark/markov/succ_cast",
            succ,
            ir.DataType.FLOAT,
            ["batch_size", n_spec, top_k, rank],
            to=ir.DataType.FLOAT,
        )
        succ_t = self.unary(
            "Transpose",
            "/dspark/markov/succ_t",
            succ32,
            ir.DataType.FLOAT,
            ["batch_size", n_spec, rank, top_k],
            perm=[0, 1, 3, 2],
        )
        pair32 = self.binary(
            "MatMul", "/dspark/markov/pair", pred32, succ_t, ir.DataType.FLOAT, ["batch_size", n_spec, top_k, top_k]
        )
        unary4 = self.reshape(
            "/dspark/markov/unary4",
            unary_logits,
            [-1, n_spec, 1, top_k],
            ir.DataType.FLOAT,
            ["batch_size", n_spec, 1, top_k],
        )
        self.make_node("Add", [pair32, unary4], ["draft_scores"], name="/dspark/markov/scores")
        self.make_value("draft_scores", ir.DataType.FLOAT, ["batch_size", n_spec, top_k, top_k])
        self.make_node("Cast", [cand], ["draft_candidate_ids"], name="/dspark/markov/cand_cast", to=ir.DataType.INT32)
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
        for required in ("fc.weight", "markov_head.markov_w1.weight", "markov_head.markov_w2.weight"):
            if required not in weights:
                raise ValueError(f"'{self.draft_dir}' does not look like a DSpark checkpoint (no {required}).")

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
        self.validate_weights(weights)
        self.weights = weights
        return weights

    def validate_weights(self, weights):
        self.validate_weight_shapes(
            weights,
            {
                "fc.weight": (self.hidden_size, self.aux_hidden_size),
                "embed_tokens.weight": (self.vocab_size, self.hidden_size),
                "lm_head.weight": (self.vocab_size, self.hidden_size),
                "markov_head.markov_w1.weight": (self.vocab_size, self.markov_rank),
                "markov_head.markov_w2.weight": (self.vocab_size, self.markov_rank),
            },
        )
