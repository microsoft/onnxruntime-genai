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

import glob
import json
import os

import numpy as np
import onnx_ir as ir
import torch

from .dflash2 import DFlash2Builder

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequantize_nvfp4(weight, weight_scale, weight_scale_2, device=None):
    """modelopt NVFP4: uint8 ``[N, K/2]`` (low nibble first), float8_e4m3 block-16 scales, fp32 global scale -> ``[N, K]`` fp32."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)
    w = weight.to(device)
    values = torch.stack([lut[(w & 0xF).long()], lut[(w >> 4).long()]], dim=-1).reshape(w.shape[0], -1)
    scale = weight_scale.to(device).to(torch.float32).repeat_interleave(16, dim=-1) * weight_scale_2.to(device).to(torch.float32)
    return (values * scale).cpu()


def split_ggml_q8_0(blocks):
    """ggml Q8_0 ``uint8 [N, K/32, 34]`` -> (uint8 ``[N, K/32, 32]`` with zero point 128, fp16 scales ``[N, K/32]``)."""
    d = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1)
    q = (blocks[..., 2:].contiguous().view(torch.int8).to(torch.int16) + 128).to(torch.uint8)
    return q, d


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
        self.draft_dir = draft_dir
        self.target_dir = target_dir
        self.io_dtype = ir.DataType.BFLOAT16
        self.external_dtype = ir.DataType.BFLOAT16
        if precision in ("int4", "int8"):
            self.quant_bits = 4 if precision == "int4" else 8
            self.quant_block_size = 32
            self.quant_prepack = 0
        elif precision != "bf16":
            raise ValueError(f"precision must be bf16, int4 or int8, got {precision!r}.")
        self.lm_head_precision = lm_head_precision
        self.lm_head_quant = None
        self.embed_quant = None
        self.filename = filename
        self.paged_block_size = paged_block_size
        self.mlp_attrs = {"fuse_gate_up": True}
        self.attn_attrs = {"fuse_qkv": False}
        self.include_attention_metadata = True
        self.uses_q_row_map = True

        with open(os.path.join(draft_dir, "config.json")) as f:
            cfg = json.load(f)
        if cfg.get("dflash_config", {}).get("projector_type") != "lilicorr" or not cfg["dflash_config"].get("lilicorr_enabled", True):
            raise ValueError(f"'{draft_dir}' is not a LiLiCorr draft checkpoint (dflash_config.projector_type != 'lilicorr').")
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
        self.make_graph("dflash2_graph", "dflash2")

    # ---------------------------------------------------------------- weights

    def _draft_linear(self, f, name):
        """Dense ``[N, K]`` weight (dequantising NVFP4) and optional bias of a checkpoint linear."""
        keys = f.keys()
        weight = f.get_tensor(f"{name}.weight")
        if f"{name}.weight_scale" in keys:
            weight = dequantize_nvfp4(weight, f.get_tensor(f"{name}.weight_scale"), f.get_tensor(f"{name}.weight_scale_2"))
        bias = f.get_tensor(f"{name}.bias") if f"{name}.bias" in keys else None
        return weight.to(torch.bfloat16), (bias.to(torch.bfloat16) if bias is not None else None)

    def load_weights(self):
        import safetensors.torch as safetensors_torch  # noqa: PLC0415

        weights = {}
        quantized_suffixes = (".weight_scale", ".weight_scale_2")
        for shard in sorted(glob.glob(os.path.join(self.draft_dir, "*.safetensors"))):
            with safetensors_torch.safe_open(shard, framework="pt") as f:
                names = set(f.keys())  # noqa: SIM118
                for key in sorted(names):
                    if key.endswith(quantized_suffixes):
                        continue
                    base = key[: -len(".weight")] if key.endswith(".weight") else None
                    if base is not None and f"{base}.weight_scale" in names:
                        weights[key] = self._draft_linear(f, base)[0]
                    else:
                        weights[key] = f.get_tensor(key)
        if "fc.weight" not in weights or "lilicorr.token_proj.weight" not in weights:
            raise ValueError(f"'{self.draft_dir}' does not look like a LiLiCorr draft checkpoint.")

        index_path = os.path.join(self.target_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        embed_key = next(k for k in ("language_model.backbone.embeddings.weight", "model.embed_tokens.weight",
                                     "model.language_model.embed_tokens.weight") if k in weight_map)
        head_key = next(k for k in ("language_model.lm_head.weight", "lm_head.weight") if k in weight_map)
        for key, name in ((embed_key, "embed_tokens.weight"), (head_key, "lm_head.weight")):
            with safetensors_torch.safe_open(os.path.join(self.target_dir, weight_map[key]), framework="pt") as f:
                weights[name] = f.get_tensor(key)
        if weights["lm_head.weight"].dtype == torch.uint8 and weights["lm_head.weight"].dim() == 3:
            weights["lm_head.q8_0"] = split_ggml_q8_0(weights["lm_head.weight"])
        self.validate_weight_shapes(
            weights,
            {
                "fc.weight": (self.hidden_size, self.aux_hidden_size),
                "embed_tokens.weight": (self.vocab_size, self.hidden_size),
            },
        )
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

    def make_lm_head(self, root, rows="num_block"):
        """The target's head: ggml Q8_0 -> int8 ``MatMulNBits`` (exact), or a dense bf16 matmul."""
        name = "/lm_head/MatMul"
        output = self.out(name)
        q8 = self.weights.get("lm_head.q8_0")
        if q8 is not None and self.lm_head_precision == "int8":
            q, d = q8
            self.make_initializer(q, "lm_head.MatMul.weight_Q8")
            self.make_initializer(d, "lm_head.MatMul.weight_scales", to=self.io_dtype)
            self.make_node(
                "MatMulNBits",
                [root, "lm_head.MatMul.weight_Q8", "lm_head.MatMul.weight_scales"],
                [output],
                name=name,
                domain="com.microsoft",
                bits=8,
                block_size=32,
                K=self.hidden_size,
                N=self.vocab_size,
            )
        else:
            if q8 is not None:
                q, d = q8
                weight = (q.to(torch.float32) - 128.0) * d.to(torch.float32).unsqueeze(-1)
                weight = weight.reshape(self.vocab_size, self.hidden_size)
            else:
                weight = self.weights["lm_head.weight"]
            self.make_initializer(weight.T, "lm_head.MatMul.weight", to=self.io_dtype)
            self.make_node("MatMul", [root, "lm_head.MatMul.weight"], [output], name=name)
        self.make_value(output, self.io_dtype, [rows, self.vocab_size])
        return output

    # ---------------------------------------------------------------- generic emit helpers

    def _op(self, op, name, inputs, dtype, shape, domain="", n_out=1, **attrs):
        outs = [f"{name}/output_{i}" for i in range(n_out)]
        self.make_node(op, inputs, outs, name=name, domain=domain, **attrs)
        for o in outs:
            self.make_value(o, dtype, shape)
        return outs[0] if n_out == 1 else outs

    def _cast(self, name, x, dtype, shape):
        return self._op("Cast", name, [x], dtype, shape, to=dtype)

    def _linear(self, name, x, weight, bias, out_shape):
        """``x @ weight.T + bias`` for an arbitrary-rank ``x`` (dense, bf16)."""
        wname = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(weight.T, wname, to=self.io_dtype)
        y = self._op("MatMul", f"{name}/MatMul", [x, wname], self.io_dtype, out_shape)
        if bias is not None:
            bname = f"{name[1:].replace('/', '.')}.bias"
            self.make_initializer(bias, bname, to=self.io_dtype)
            y = self._op("Add", f"{name}/Add", [y, bname], self.io_dtype, out_shape)
        return y

    def _silu(self, name, x, dtype, shape):
        s = self._op("Sigmoid", f"{name}/Sigmoid", [x], dtype, shape)
        return self._op("Mul", f"{name}/Mul", [x, s], dtype, shape)

    def _rms(self, name, x, weight, shape):
        wname = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(weight, wname, to=self.io_dtype)
        out = f"{name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization", [x, wname], [out, "", ""], name=name, axis=-1, epsilon=self.rms_eps, stash_type=1
        )
        self.make_value(out, self.io_dtype, shape)
        return out

    def _l2_normalize(self, name, x, shape):
        """``x / max(||x||_2, eps)`` over the last axis, fp32."""
        sq = self._op("Mul", f"{name}/sq", [x, x], ir.DataType.FLOAT, shape)
        axes = self.const([-1])
        ss = self._op("ReduceSum", f"{name}/sum", [sq, axes], ir.DataType.FLOAT, shape[:-1] + [1], keepdims=1)
        nrm = self._op("Sqrt", f"{name}/sqrt", [ss], ir.DataType.FLOAT, shape[:-1] + [1])
        eps = self.const(np.full((1,), self.vector_eps, np.float32), ir.DataType.FLOAT)
        nrm = self._op("Max", f"{name}/max", [nrm, eps], ir.DataType.FLOAT, shape[:-1] + [1])
        return self._op("Div", f"{name}/div", [x, nrm], ir.DataType.FLOAT, shape)

    # ---------------------------------------------------------------- candidates + lattice head

    def _lattice_attention_bias(self):
        """``[1, heads, n*k, n*k]``: relative-slot bias + same-slot bias, as vLLM's ``_build_attention_bias`` (bf16-rounded)."""
        n, k = self.num_draft_tokens, self.selector_top_k
        rel_bias = self.weights["lilicorr.relative_slot_bias"].to(torch.bfloat16)  # [heads, 2*block-1]
        same = self.weights["lilicorr.same_slot_bias"].to(torch.bfloat16)
        slot = torch.arange(n).repeat_interleave(k)
        rel = slot.view(-1, 1) - slot.view(1, -1)
        bias = rel_bias[:, rel + int(self.cfg["block_size"]) - 1]
        bias = bias + (slot.view(-1, 1) == slot.view(1, -1)).unsqueeze(0).to(bias.dtype) * same.view(-1, 1, 1)
        return bias.to(torch.float32).unsqueeze(0).contiguous()

    def _make_candidates_and_selector(self, final, w):
        n, k, lh, H = self.num_draft_tokens, self.selector_top_k, self.lh, self.hidden_size
        W = self.weights
        p = "/lilicorr"
        bf, f32 = self.io_dtype, ir.DataType.FLOAT

        # ---- mask rows -> top-k candidates (target LM head) ----
        h3 = self.reshape(f"{p}/select/Reshape", final, [-1, self.block_size, H], bf, ["batch_size", self.block_size, H])
        sl = self.out(f"{p}/select/Slice")
        self.make_node("Slice", [h3, self.const([1]), self.const([self.block_size]), self.const([1])], [sl], name=f"{p}/select/Slice")
        self.make_value(sl, bf, ["batch_size", n, H])
        hsel = self.reshape(f"{p}/select/Flatten", sl, [-1, H], bf, ["num_sample", H])
        logits = self.make_lm_head(hsel, "num_sample")
        logits32 = self._cast(f"{p}/logits/Cast", logits, f32, ["num_sample", self.vocab_size])
        vals, idx = f"{p}/topk/values", f"{p}/topk/indices"
        self.make_node("TopK", [logits32, self.const([k])], [vals, idx], name=f"{p}/topk/TopK", axis=-1, largest=1, sorted=1)
        self.make_value(vals, f32, ["num_sample", k])
        self.make_value(idx, ir.DataType.INT64, ["num_sample", k])
        # log-partition from the (descending) top-k max. ReduceLogSumExp / ReduceMax over a [rows, vocab] tensor run a cuDNN
        # reduction that is ~100x slower than ORT's own last-axis ReduceSum.
        top_logit = self._op("Slice", f"{p}/logits/max", [vals, self.const([0]), self.const([1]), self.const([1])], f32, ["num_sample", 1])
        shifted = self._op("Sub", f"{p}/logits/shift", [logits32, top_logit], f32, ["num_sample", self.vocab_size])
        expd = self._op("Exp", f"{p}/logits/exp", [shifted], f32, ["num_sample", self.vocab_size])
        denom = self._op("ReduceSum", f"{p}/logits/sum", [expd, self.const([-1])], f32, ["num_sample", 1], keepdims=1)
        log_denom = self._op("Log", f"{p}/logits/log", [denom], f32, ["num_sample", 1])
        lse = self._op("Add", f"{p}/logits/lse", [log_denom, top_logit], f32, ["num_sample", 1])
        logp = self._op("Sub", f"{p}/logp", [vals, lse], f32, ["num_sample", k])
        cand = self.reshape(f"{p}/cand", idx, [-1, n, k], ir.DataType.INT64, ["batch_size", n, k])

        # ---- lattice tokens ----
        cand_emb = self._op("Gather", f"{p}/cand_embed", ["model.embed_tokens.weight", idx], self.external_dtype, ["num_sample", k, H])
        if self.io_dtype != self.external_dtype:
            cand_emb = self._cast(f"{p}/cand_embed/Cast", cand_emb, bf, ["num_sample", k, H])
        tok = self._linear(f"{p}/token_proj", cand_emb, W["lilicorr.token_proj.weight"], W["lilicorr.token_proj.bias"], ["num_sample", k, lh])
        pas = self._linear(f"{p}/pass_hidden_proj", hsel, W["lilicorr.pass_hidden_proj.weight"], W["lilicorr.pass_hidden_proj.bias"], ["num_sample", lh])
        pas = self._op("Unsqueeze", f"{p}/pass_unsq", [pas, self.const([1])], bf, ["num_sample", 1, lh])

        # candidate log-prob features -> LayerNorm(5) -> MLP
        prob = self._op("Exp", f"{p}/feat/exp", [logp], f32, ["num_sample", k])
        mx = self._op("Slice", f"{p}/feat/max", [logp, self.const([0]), self.const([1]), self.const([1])], f32, ["num_sample", 1])
        rel = self._op("Sub", f"{p}/feat/rel", [logp, mx], f32, ["num_sample", k])
        cols = [
            self._op("Unsqueeze", f"{p}/feat/u{i}", [t, self.const([-1])], f32, ["num_sample", k, 1])
            for i, t in enumerate((logp, prob, rel))
        ]
        shape_of = self._op("Shape", f"{p}/feat/shape", [cols[0]], ir.DataType.INT64, [3])
        rank_frac = np.arange(k, dtype=np.float32).reshape(1, k, 1) / max(k - 1, 1)
        top1 = np.zeros((1, k, 1), np.float32)
        top1[0, 0, 0] = 1.0
        for i, arr in enumerate((rank_frac, top1)):
            c = self.const(arr, ir.DataType.FLOAT)
            cols.append(self._op("Expand", f"{p}/feat/c{i}", [c, shape_of], f32, ["num_sample", k, 1]))
        feats = self._op("Concat", f"{p}/feat/concat", cols, f32, ["num_sample", k, 5], axis=-1)
        feats = self._cast(f"{p}/feat/cast", feats, bf, ["num_sample", k, 5])
        self.make_initializer(W["lilicorr.feature_norm.weight"], "lilicorr.feature_norm.weight", to=bf)
        self.make_initializer(W["lilicorr.feature_norm.bias"], "lilicorr.feature_norm.bias", to=bf)
        ln_out = f"{p}/feat/ln/output_0"
        self.make_node(
            "LayerNormalization",
            [feats, "lilicorr.feature_norm.weight", "lilicorr.feature_norm.bias"],
            [ln_out],
            name=f"{p}/feat/ln",
            axis=-1,
            epsilon=1e-5,
        )
        self.make_value(ln_out, bf, ["num_sample", k, 5])
        up = self._linear(f"{p}/feature_mlp/up", ln_out, W["lilicorr.feature_mlp.up_proj.weight"], W["lilicorr.feature_mlp.up_proj.bias"], ["num_sample", k, lh])
        up = self._silu(f"{p}/feature_mlp/act", up, bf, ["num_sample", k, lh])
        fm = self._linear(f"{p}/feature_mlp/down", up, W["lilicorr.feature_mlp.down_proj.weight"], W["lilicorr.feature_mlp.down_proj.bias"], ["num_sample", k, lh])

        # hidden = token + pass + features + slot embedding + rank embedding   ([B, n, k, lh])
        hid = self._op("Add", f"{p}/hid/a0", [tok, pas], bf, ["num_sample", k, lh])
        hid = self._op("Add", f"{p}/hid/a1", [hid, fm], bf, ["num_sample", k, lh])
        hid = self.reshape(f"{p}/hid/r4", hid, [-1, n, k, lh], bf, ["batch_size", n, k, lh])
        slot = self.weights["lilicorr.slot_embedding"][0, 0, :n]  # [n, 1, lh]
        rank = self.weights["lilicorr.rank_embedding"][0, 0, 0]  # [k, lh]
        self.make_initializer(slot.reshape(1, n, 1, lh), "lilicorr.slot_embedding", to=bf)
        self.make_initializer(rank.reshape(1, 1, k, lh), "lilicorr.rank_embedding", to=bf)
        hid = self._op("Add", f"{p}/hid/a2", [hid, "lilicorr.slot_embedding"], bf, ["batch_size", n, k, lh])
        hid = self._op("Add", f"{p}/hid/a3", [hid, "lilicorr.rank_embedding"], bf, ["batch_size", n, k, lh])
        lat = n * k
        x = self.reshape(f"{p}/hid/flat", hid, [-1, lat, lh], bf, ["batch_size", lat, lh])

        # ---- lattice transformer ----
        self.make_initializer(self._lattice_attention_bias(), "lilicorr.attn_bias", to=f32)
        heads, hdim = self.l_heads, lh // self.l_heads
        scale = self.const(np.full((1,), hdim**-0.5, np.float32), f32)
        for li in range(self.l_layers):
            q = f"{p}/layers.{li}"
            xn = self._rms(f"{q}/attn_norm", x, W[f"lilicorr.layers.{li}.attn_norm.weight"], ["batch_size", lat, lh])
            qkv = self._linear(f"{q}/in_proj", xn, W[f"lilicorr.layers.{li}.attn.in_proj_weight"], W[f"lilicorr.layers.{li}.attn.in_proj_bias"], ["batch_size", lat, 3 * lh])
            parts = [f"{q}/qkv/split_{i}" for i in range(3)]
            self.make_node("Split", [qkv], parts, name=f"{q}/qkv/split", axis=-1, num_outputs=3)
            hs = []
            for i, part in enumerate(parts):
                self.make_value(part, bf, ["batch_size", lat, lh])
                r = self.reshape(f"{q}/qkv/r{i}", part, [-1, lat, heads, hdim], bf, ["batch_size", lat, heads, hdim])
                perm = [0, 2, 3, 1] if i == 1 else [0, 2, 1, 3]
                hs.append(self._op("Transpose", f"{q}/qkv/t{i}", [r], bf, ["batch_size", heads, hdim, lat] if i == 1 else ["batch_size", heads, lat, hdim], perm=perm))
            sc = self._op("MatMul", f"{q}/scores", [hs[0], hs[1]], bf, ["batch_size", heads, lat, lat])
            sc = self._cast(f"{q}/scores/cast", sc, f32, ["batch_size", heads, lat, lat])
            sc = self._op("Mul", f"{q}/scores/scale", [sc, scale], f32, ["batch_size", heads, lat, lat])
            sc = self._op("Add", f"{q}/scores/bias", [sc, "lilicorr.attn_bias"], f32, ["batch_size", heads, lat, lat])
            pr = self._op("Softmax", f"{q}/softmax", [sc], f32, ["batch_size", heads, lat, lat], axis=-1)
            pr = self._cast(f"{q}/softmax/cast", pr, bf, ["batch_size", heads, lat, lat])
            ao = self._op("MatMul", f"{q}/context", [pr, hs[2]], bf, ["batch_size", heads, lat, hdim])
            ao = self._op("Transpose", f"{q}/context/t", [ao], bf, ["batch_size", lat, heads, hdim], perm=[0, 2, 1, 3])
            ao = self.reshape(f"{q}/context/flat", ao, [-1, lat, lh], bf, ["batch_size", lat, lh])
            ao = self._linear(f"{q}/out_proj", ao, W[f"lilicorr.layers.{li}.attn.out_proj.weight"], W[f"lilicorr.layers.{li}.attn.out_proj.bias"], ["batch_size", lat, lh])
            x = self._op("Add", f"{q}/attn_res", [x, ao], bf, ["batch_size", lat, lh])
            mn = self._rms(f"{q}/mlp_norm", x, W[f"lilicorr.layers.{li}.mlp_norm.weight"], ["batch_size", lat, lh])
            m = self._linear(f"{q}/mlp/up", mn, W[f"lilicorr.layers.{li}.mlp.up_proj.weight"], W[f"lilicorr.layers.{li}.mlp.up_proj.bias"], ["batch_size", lat, self.l_mlp])
            m = self._silu(f"{q}/mlp/act", m, bf, ["batch_size", lat, self.l_mlp])
            m = self._linear(f"{q}/mlp/down", m, W[f"lilicorr.layers.{li}.mlp.down_proj.weight"], W[f"lilicorr.layers.{li}.mlp.down_proj.bias"], ["batch_size", lat, lh])
            x = self._op("Add", f"{q}/mlp_res", [x, m], bf, ["batch_size", lat, lh])
        x = self._rms(f"{p}/output_norm", x, W["lilicorr.output_norm.weight"], ["batch_size", lat, lh])
        hid4 = self.reshape(f"{p}/out4", x, [-1, n, k, lh], bf, ["batch_size", n, k, lh])

        # ---- anchor: hidden_norm(fc(aux)[last context row]) -> context_proj -> anchor_norm ----
        idx32 = "anchor_row_index"
        valid = self._op("Greater", f"{p}/anchor/valid", [idx32, self.const(np.array([-1], np.int32), ir.DataType.INT32)], ir.DataType.BOOL, ["batch_size"])
        valid = self._cast(f"{p}/anchor/valid_cast", valid, bf, ["batch_size"])
        valid = self._op("Unsqueeze", f"{p}/anchor/valid_unsq", [valid, self.const([-1])], bf, ["batch_size", 1])
        safe = self._op("Max", f"{p}/anchor/safe", [idx32, self.const(np.array([0], np.int32), ir.DataType.INT32)], ir.DataType.INT32, ["batch_size"])
        row = self._op("Gather", f"{p}/anchor/row", [self.ctx_pre_norm, safe], bf, ["batch_size", H], axis=0)
        row = self.rms_norm(f"{p}/anchor/hidden_norm", row, self.weights["hidden_norm.weight"], "batch_size", weight_name="dflash2.hidden_norm.weight")
        anchor = self._linear(f"{p}/context_proj", row, W["lilicorr.context_proj.weight"], W["lilicorr.context_proj.bias"], ["batch_size", lh])
        anchor = self._op("Mul", f"{p}/anchor/mask", [anchor, valid], bf, ["batch_size", lh])
        anchor = self._rms(f"{p}/anchor_norm", anchor, W["lilicorr.anchor_norm.weight"], ["batch_size", lh])

        # ---- factor vectors ----
        fw = W["lilicorr.factor_input_proj.weight"]
        w_self, w_anchor, w_cross = (fw[:, i * lh:(i + 1) * lh].contiguous() for i in range(3))
        a4 = self.reshape(f"{p}/anchor/a4", anchor, [-1, 1, 1, lh], bf, ["batch_size", 1, 1, lh])
        pre = self._linear(f"{p}/factor/self", hid4, w_self, W["lilicorr.factor_input_proj.bias"], ["batch_size", n, k, lh])
        pa = self._linear(f"{p}/factor/anchor", a4, w_anchor, None, ["batch_size", 1, 1, lh])
        cross_in = self._op("Mul", f"{p}/factor/cross_in", [hid4, a4], bf, ["batch_size", n, k, lh])
        pc = self._linear(f"{p}/factor/cross", cross_in, w_cross, None, ["batch_size", n, k, lh])
        pre = self._op("Add", f"{p}/factor/add0", [pre, pa], bf, ["batch_size", n, k, lh])
        pre = self._op("Add", f"{p}/factor/add1", [pre, pc], bf, ["batch_size", n, k, lh])
        fh = self._silu(f"{p}/factor/act", pre, bf, ["batch_size", n, k, lh])
        ew = torch.cat([W["lilicorr.out_head.weight"], W["lilicorr.in_head.weight"]], 0)
        eb = torch.cat([W["lilicorr.out_head.bias"], W["lilicorr.in_head.bias"]], 0)
        edges = self._linear(f"{p}/edges", fh, ew, eb, ["batch_size", n, k, 2 * self.factor_dim])
        edges = self._cast(f"{p}/edges/cast", edges, f32, ["batch_size", n, k, 2 * self.factor_dim])
        fd = self.factor_dim
        out_raw, in_raw = f"{p}/edges/out", f"{p}/edges/in"
        self.make_node("Split", [edges], [out_raw, in_raw], name=f"{p}/edges/split", axis=-1, num_outputs=2)
        self.make_value(out_raw, f32, ["batch_size", n, k, fd])
        self.make_value(in_raw, f32, ["batch_size", n, k, fd])
        out_vec = self._l2_normalize(f"{p}/out_norm", out_raw, ["batch_size", n, k, fd])
        in_vec = self._l2_normalize(f"{p}/in_norm", in_raw, ["batch_size", n, k, fd])

        ao_raw = self._linear(f"{p}/anchor_out_head", anchor, W["lilicorr.anchor_out_head.weight"], W["lilicorr.anchor_out_head.bias"], ["batch_size", fd])
        ao_raw = self._cast(f"{p}/anchor_out_head/cast", ao_raw, f32, ["batch_size", fd])
        anchor_out = self._l2_normalize(f"{p}/anchor_out_norm", ao_raw, ["batch_size", fd])
        anchor_out = self.reshape(f"{p}/anchor_out4", anchor_out, [-1, 1, 1, fd], f32, ["batch_size", 1, 1, fd])

        in_first = self._op("Slice", f"{p}/in_first", [in_vec, self.const([0]), self.const([1]), self.const([1])], f32, ["batch_size", 1, k, fd])
        start = self._op("Mul", f"{p}/start/mul", [in_first, anchor_out], f32, ["batch_size", 1, k, fd])
        start = self._op("ReduceSum", f"{p}/start/sum", [start, self.const([-1])], f32, ["batch_size", 1, k], keepdims=0)
        start = self._op("Unsqueeze", f"{p}/start/unsq", [start, self.const([2])], f32, ["batch_size", 1, 1, k])
        start = self._op("Expand", f"{p}/start/expand", [start, self.const([1, 1, k, k])], f32, ["batch_size", 1, k, k])
        scale_c = self.const(np.full((1,), self.logit_scale, np.float32), f32)
        start = self._op("Mul", f"{p}/start/scale", [start, scale_c], f32, ["batch_size", 1, k, k])

        out_head = self._op("Slice", f"{p}/out_head", [out_vec, self.const([0]), self.const([n - 1]), self.const([1])], f32, ["batch_size", n - 1, k, fd])
        in_tail = self._op("Slice", f"{p}/in_tail", [in_vec, self.const([1]), self.const([n]), self.const([1])], f32, ["batch_size", n - 1, k, fd])
        in_tail_t = self._op("Transpose", f"{p}/in_tail_t", [in_tail], f32, ["batch_size", n - 1, fd, k], perm=[0, 1, 3, 2])
        pairs = self._op("MatMul", f"{p}/pairs", [out_head, in_tail_t], f32, ["batch_size", n - 1, k, k])
        pairs = self._op("Mul", f"{p}/pairs/scale", [pairs, scale_c], f32, ["batch_size", n - 1, k, k])
        self.make_node("Concat", [start, pairs], ["draft_scores"], name=f"{p}/scores", axis=1)
        self.make_value("draft_scores", f32, ["batch_size", n, k, k])
        self.make_node("Cast", [cand], ["draft_candidate_ids"], name=f"{p}/cand_cast", to=ir.DataType.INT32)
        self.make_value("draft_candidate_ids", ir.DataType.INT32, ["batch_size", n, k])

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
