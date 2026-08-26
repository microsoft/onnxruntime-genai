# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
"""ONNX model builders for Qwen-3.8 Flash Next (Hugging Face architecture family ``Qwen4Exp``).

Qwen4Exp is a hybrid multimodal MoE decoder.  Relative to Qwen3.5 (``Qwen35MoeTextModel``)
it adds four structural features that have no counterpart anywhere else in this repository:

1. **Hyper-connections.**  The residual stream is widened to ``hc_count`` parallel streams
   (``[B, S, hc_count * hidden_size]``).  Before every block a ``GatedResidual`` mixes the
   streams down to ``[B, S, hidden_size]`` and produces per-stream injection weights; after
   the block the output is broadcast back into the widened stream.  There are **no**
   ``input_layernorm`` / ``post_attention_layernorm`` modules, so the base class's
   Skip(Simplified)LayerNormalization residual chaining does not apply and
   :meth:`Qwen4ExpTextModel.make_layer` fully overrides the base implementation.

2. **PLE (Per-Layer Embeddings) with hashed n-grams.**  Selected layers hash the raw
   ``input_ids`` into ``(ngram_size - 1) * heads_per_ngram`` deterministic n-gram ids, embed
   them, gate them against the residual stream and add a dilated depthwise short convolution.
   The hashing is *deterministic* (splitmix64 + prime-sized per-head vocabularies) and is
   reproduced exactly here so the exported multipliers/vocab sizes/offsets match the reference.

3. **QSA (Qwen Sparse Attention).**  Full-attention layers own a small side "indexer"
   projection whose keys are mean-pooled into blocks of ``indexer_compress_ratio`` tokens and
   scored to select a token budget for the main attention.

4. **MoE + always-on shared expert**, identical in layout to Qwen3.5-MoE, so
   :meth:`Qwen35MoeTextModel.make_moe` is reused unchanged.

Contrib operator contracts
--------------------------
The four features above are emitted as ``com.microsoft`` operators.  These schemas are
**defined by this builder**; matching ONNX Runtime kernels must implement them exactly.
``test/python/builder/test_qwen4exp_*.py`` pins every one of them.

``NGramHashMapping`` (attribute ``version = 2``)
    inputs  : ``input_ids``   int64 ``[B, S]``
              ``layer_multipliers`` int64 ``[ngram_size]``
              ``head_vocab_sizes``  int64 ``[ngram_heads]``
              ``head_offsets``      int64 ``[ngram_heads]``
              ``past_token_state``  int64 ``[B, ngram_size - 1]``
    outputs : ``ngram_ids``    int64 ``[B, S, ngram_heads]``
              ``present_token_state`` int64 ``[B, ngram_size - 1]``
    attrs   : ``ngram_size``, ``heads_per_ngram``, ``eos_token_id``, ``version``
    The token state is the trailing ``ngram_size - 1`` tokens of the previous forward, EOS
    filled on the first step (*not* zero filled).  ``version = 2`` selects the
    "shift right, ignoring EOS boundaries" history semantics used by Qwen4Exp.

``EngramGate``
    inputs  : ``key_normed`` ``[B, S, hc_count * H]``
              ``query_normed`` ``[B, S, hc_count * H]``
              ``value`` ``[B, S, H]``
    outputs : ``gated_value`` ``[B, S, hc_count * H]``
    attrs   : ``num_streams`` (= ``hc_count``), ``hidden_size`` (= ``H``), ``epsilon``
    Semantics per stream ``c``:
        ``g = sum_h key[c,h] * query[c,h] / sqrt(H)``
        ``g = sign(g) * sqrt(max(|g|, epsilon))``
        ``out[c] = sigmoid(g) * value``

``ShortConvWithState`` / ``ShortConv``
    inputs  : ``X`` ``[B, C, S]``, ``weight`` ``[C, 1, K]``
              (``ShortConvWithState`` only) ``past_conv_state`` ``[B, C, state_len]``
    outputs : ``Y`` ``[B, C, S]`` (+ ``present_conv_state`` for the stateful variant)
    attrs   : ``dilation``, ``activation`` (``"silu"``), ``group`` (= ``C``)
    ``state_len`` is ``(K - 1) * dilation``; the op left-pads with the state, runs the
    dilated depthwise convolution, and applies ``activation``.

``QwenSparseAttention`` (contiguous KV cache) / ``SparsePagedAttention`` (paged KV cache)
    A GroupQueryAttention/PagedAttention drop-in that additionally consumes the QSA indexer
    query/key streams and restricts each query to ``indexer_token_budget`` selected keys.
    See :meth:`Qwen4ExpTextModel._make_qsa_attention_op` for the exact input order.

Multi-file export
-----------------
``Qwen4ExpForConditionalGeneration`` exports three graphs, mirroring the runtime layout that
``genai_config.json`` already understands for Qwen VL models:

* ``text.onnx``      -- the decoder (built by :class:`Qwen4ExpModel` itself)
* ``embedding.onnx`` -- token embedding + image/video feature scatter
* ``vision.onnx``    -- the ViT tower + patch merger

``Qwen4ExpForCausalLM`` exports the decoder alone as ``model.onnx`` and takes ``input_ids``
directly.

.. warning::
   ``vision.onnx`` takes *pre-computed geometry* (position-embedding interpolation indices and
   weights, rotary cos/sin, and a block-diagonal attention bias) instead of ``image_grid_thw``.
   The reference implementation derives these with per-image Python loops over ``grid_thw``
   that cannot be expressed as static ONNX.  The host image processor must supply them.
   This is a deliberate deviation from the ``{pixel_values, image_grid_thw}`` convention used
   by the other Qwen VL runtime paths and requires matching pre-processor support.
"""

import copy
import json
import math
import os

import onnx_ir as ir
import torch

from .base import Model
from .qwen import Qwen35MoeTextModel

#####################################################################################
# Deterministic n-gram hashing (mirrors modeling_qwen4_exp.py)
#####################################################################################

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def splitmix64(value: int) -> int:
    """SplitMix64 finalizer, used to derive the per-layer n-gram hash multipliers."""
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> list:
    """Odd, per-position hash multipliers for one PLE layer.

    Each multiplier is odd and bounded so that ``token_id * multiplier`` stays inside int64.
    """
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (splitmix64(value) % half_bound) + 1)
    return multipliers


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


def ngram_head_vocab_layout(ngram_vocab_size_base: int, ngram_heads: int, ple_layer_index: int):
    """Per-head vocabulary sizes and offsets for one PLE layer.

    Head ``i`` of PLE layer ``l`` uses the ``(l * ngram_heads + i + 1)``-th prime strictly
    greater than ``ngram_vocab_size_base - 1``, so no two heads (across layers) collide.

    Returns ``(head_vocab_sizes, head_offsets, total_vocab_size)``.
    """
    head_vocab_sizes = []
    head_offsets = []
    total = 0
    for head_idx in range(ngram_heads):
        global_head_idx = ple_layer_index * ngram_heads + head_idx
        size = find_nth_prime_after(ngram_vocab_size_base - 1, global_head_idx + 1)
        head_vocab_sizes.append(size)
        head_offsets.append(total)
        total += size
    return head_vocab_sizes, head_offsets, total


def padded_ngram_vocab_size(total_vocab_size: int, divisor: int) -> int:
    return math.ceil(total_vocab_size / divisor) * divisor


class _PackedExpertsView:
    """Adapter that makes ``Qwen4ExpTextExperts`` usable by ``Qwen35MoeTextModel.make_moe``.

    Qwen3.5's MoE builder probes ``next(iter(mlp.experts), None)`` to detect a ModelOpt NVFP4
    checkpoint whose experts are stored as a per-expert module list.  Qwen4Exp only ever stores
    the packed 3-D ``gate_up_proj``/``down_proj`` parameters, which the builder already handles,
    so iteration yields nothing and the packed path is taken.
    """

    def __init__(self, experts):
        self._experts = experts

    def __getattr__(self, name):
        return getattr(self._experts, name)

    def __iter__(self):
        return iter(())


class _PackedMoeView:
    """``mlp`` view whose ``experts`` attribute is a :class:`_PackedExpertsView`."""

    def __init__(self, mlp):
        self._mlp = mlp
        self.experts = _PackedExpertsView(mlp.experts)

    def __getattr__(self, name):
        return getattr(self._mlp, name)


#####################################################################################
# Text decoder
#####################################################################################


class Qwen4ExpTextModel(Qwen35MoeTextModel):
    """Qwen-3.8 Flash Next text decoder.

    Inherits the hybrid (GatedDeltaNet linear attention + gated full attention) machinery and
    the MoE + shared-expert MLP from :class:`Qwen35MoeTextModel`, and replaces:

    * the residual/layernorm chaining with hyper-connections,
    * ``GroupQueryAttention`` with ``QwenSparseAttention``/``SparsePagedAttention``,
    * the final norm with the ``hyper_connection_mixer`` gated residual,

    while adding the PLE n-gram path on ``config.ple_layer_ids`` layers.
    """

    def _get_model_type(self, config):
        return "Qwen4Exp_textForCausalLM" if self.is_text_only else "Qwen4ExpForConditionalGeneration"

    #: Qwen4Exp names its attention layers ``qwen_sparse_attention``. Published checkpoints may
    #: still say ``full_attention``; the HF config normalizes those to ``qwen_sparse_attention``.
    QSA_LAYER_TYPE = "qwen_sparse_attention"

    def _resolve_layer_types(self, config, num_layers):
        """Normalize ``qwen_sparse_attention`` to the ``full_attention`` name the base uses.

        Every Qwen4Exp attention layer is a QSA layer, so it maps 1:1 onto the Qwen3.5
        full-attention path; only the attention op itself differs.  ``_make_full_attention``
        swaps in ``QwenSparseAttention``/``SparsePagedAttention`` when the module actually
        carries an indexer.
        """
        text_config = getattr(config, "text_config", config)
        configured = getattr(text_config, "layer_types", None)
        if configured is not None and self.QSA_LAYER_TYPE in configured:
            normalized = ["full_attention" if lt == self.QSA_LAYER_TYPE else lt for lt in configured]
            # `_resolve_layer_types` re-reads the config, so rewrite it on both views.
            text_config.layer_types = normalized
            if text_config is not config:
                config.layer_types = list(normalized)
        return super()._resolve_layer_types(config, num_layers)

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        text_config = getattr(config, "text_config", config)

        # `Qwen35MoeTextModel` reads num_local_experts / intermediate_size off the *text* config.
        if getattr(text_config, "num_experts", None) is not None and not hasattr(text_config, "num_local_experts"):
            text_config.num_local_experts = text_config.num_experts
        if not hasattr(text_config, "intermediate_size") and hasattr(text_config, "moe_intermediate_size"):
            text_config.intermediate_size = text_config.moe_intermediate_size

        # Hyper-connections.
        self.hc_count = int(getattr(text_config, "hc_count", 1))
        self.hc_lowrank = int(getattr(text_config, "hc_lowrank", 64))

        # PLE / n-gram hashing.  `ple_layer_ids` is 1-based in the checkpoint config.
        self.ple_layer_ids = list(getattr(text_config, "ple_layer_ids", []) or [])
        self.ngram_size = int(getattr(text_config, "ngram_size", 4))
        self.heads_per_ngram = int(getattr(text_config, "heads_per_ngram", 1))
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ngram_vocab_size_base = int(getattr(text_config, "ngram_vocab_size_base", 0))
        self.ngram_vocab_divisor = int(getattr(text_config, "make_ngram_vocab_size_divisible_by", 1))
        self.ple_embed_dim = int(getattr(text_config, "ple_embed_dim", 0))
        self.ple_conv_kernel_size = int(getattr(text_config, "ple_conv_kernel_size", 1))
        self.ple_conv_state_len = (self.ple_conv_kernel_size - 1) * self.ngram_size
        self.ngram_seed = int(getattr(text_config, "seed", 0))
        # `ple_layer_index` per decoder layer, or None when the layer has no PLE module.
        self.ple_layer_index = {}
        for layer_id_1based in self.ple_layer_ids:
            self.ple_layer_index[layer_id_1based - 1] = self.ple_layer_ids.index(layer_id_1based)

        # QSA indexer.
        self.indexer_n_heads = int(getattr(text_config, "indexer_n_heads", 0))
        self.indexer_kv_heads = int(getattr(text_config, "indexer_kv_heads", 1))
        self.indexer_head_dim = int(getattr(text_config, "indexer_head_dim", 0))
        self.indexer_budget = int(getattr(text_config, "indexer_budget", 0))
        self.indexer_compress_ratio = int(getattr(text_config, "indexer_compress_ratio", 1))

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.hc_hidden_size = self.hc_count * self.hidden_size

        # Set by `_make_full_attention` so the `make_attention_op` callback can pick the QSA op.
        self._pending_indexer = None

        eos_token_id = getattr(text_config, "eos_token_id", 0)
        self.ngram_eos_token_id = int(eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id)

        # PLE hashes the *raw* token ids, so the decoder keeps `input_ids` as a graph input even
        # in multimodal mode where the base class drops it in favour of `inputs_embeds`.
        if self.ple_layer_ids:
            self.input_names["input_ids"] = "input_ids"
            self.input_types["input_ids"] = ir.DataType.INT64
            self.input_shapes["input_ids"] = (
                ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
            )

        # Multimodal exports write the decoder as `text.onnx`; text-only keeps `model.onnx`.
        if not self.is_text_only and not extra_options.get("filename", ""):
            self.filename = "text.onnx"

        # Hyper-connections carry their own norms; nothing pre-bakes a +1 offset into a
        # Skip(Simplified)LayerNormalization here, but `make_qk_norm` (Q/K norm) still relies on
        # `layernorm_attrs["add_offset"]`, which `Qwen35TextModel` already set to 1.

        self._add_ple_and_indexer_cache_io()

    #################################################################################
    # State / cache I/O
    #################################################################################

    def _add_ple_and_indexer_cache_io(self):
        """Add the PLE short-conv + token-history state and the QSA indexer key cache.

        ``Qwen35TextModel._setup_hybrid_cache_io`` has already produced the per-layer KV cache
        (full-attention layers) and conv/recurrent state (linear-attention layers).
        """
        for layer_id, layer_type in enumerate(self.layer_types):
            if layer_type == "full_attention" and self.indexer_head_dim:
                shape = [
                    "batch_size",
                    self.indexer_kv_heads,
                    "total_sequence_length",
                    self.indexer_head_dim,
                ]
                self.input_names[f"past_state.{layer_id}.indexer_key"] = f"past_key_values.{layer_id}.indexer_key"
                self.input_types[f"past_state.{layer_id}.indexer_key"] = self.io_dtype
                self.input_shapes[f"past_state.{layer_id}.indexer_key"] = list(shape)

                self.output_names[f"present_state.{layer_id}.indexer_key"] = f"present.{layer_id}.indexer_key"
                self.output_types[f"present_state.{layer_id}.indexer_key"] = self.io_dtype
                self.output_shapes[f"present_state.{layer_id}.indexer_key"] = list(shape)

            if layer_id in self.ple_layer_index:
                conv_shape = ["batch_size", self.hc_hidden_size, self.ple_conv_state_len]
                self.input_names[f"past_state.{layer_id}.ple_conv"] = f"past_key_values.{layer_id}.ple_conv_state"
                self.input_types[f"past_state.{layer_id}.ple_conv"] = self.io_dtype
                self.input_shapes[f"past_state.{layer_id}.ple_conv"] = list(conv_shape)

                self.output_names[f"present_state.{layer_id}.ple_conv"] = f"present.{layer_id}.ple_conv_state"
                self.output_types[f"present_state.{layer_id}.ple_conv"] = self.io_dtype
                self.output_shapes[f"present_state.{layer_id}.ple_conv"] = list(conv_shape)

                token_shape = ["batch_size", self.ngram_size - 1]
                self.input_names[f"past_state.{layer_id}.ple_tokens"] = f"past_key_values.{layer_id}.ple_tokens"
                self.input_types[f"past_state.{layer_id}.ple_tokens"] = ir.DataType.INT64
                self.input_shapes[f"past_state.{layer_id}.ple_tokens"] = list(token_shape)

                self.output_names[f"present_state.{layer_id}.ple_tokens"] = f"present.{layer_id}.ple_tokens"
                self.output_types[f"present_state.{layer_id}.ple_tokens"] = ir.DataType.INT64
                self.output_shapes[f"present_state.{layer_id}.ple_tokens"] = list(token_shape)

    def make_decoder_state_groups(self, inputs, outputs):
        state_groups = super().make_decoder_state_groups(inputs, outputs)
        if not self.use_paged_attention:
            return state_groups

        full_attention_layers = [i for i, lt in enumerate(self.layer_types) if lt == "full_attention"]
        ple_layers = sorted(self.ple_layer_index)
        has_extra_state = bool(ple_layers) or bool(full_attention_layers and self.indexer_head_dim)

        # `Qwen35TextModel` returns an empty manifest when every layer is full attention, which
        # is the legacy "no manifest needed" signal. Qwen4Exp always owns extra per-layer state,
        # so the paged KV group has to be described explicitly in that case.
        if not state_groups and has_extra_state and full_attention_layers:
            state_groups.append(self.make_paged_key_value_state_group(full_attention_layers))

        if ple_layers:
            inputs["past_ple_conv_names"] = "past_key_values.%d.ple_conv_state"
            inputs["past_ple_token_names"] = "past_key_values.%d.ple_tokens"
            outputs["present_ple_conv_names"] = "present.%d.ple_conv_state"
            outputs["present_ple_token_names"] = "present.%d.ple_tokens"
            state_groups.extend(
                [
                    {"kind": "fixed_ple_conv", "layer_ids": ple_layers},
                    {"kind": "fixed_ple_tokens", "layer_ids": ple_layers},
                ]
            )

        if full_attention_layers and self.indexer_head_dim:
            inputs["past_indexer_key_names"] = "past_key_values.%d.indexer_key"
            outputs["present_indexer_key_names"] = "present.%d.indexer_key"
            state_groups.append({"kind": "paged_indexer_key", "layer_ids": full_attention_layers})

        return state_groups

    #################################################################################
    # Small graph helpers
    #################################################################################

    def _make_silu(self, basename, root_input, dtype, shape):
        sigmoid_name = f"{basename}/Sigmoid"
        self.make_sigmoid(sigmoid_name, root_input, dtype, shape=shape)
        mul_name = f"{basename}/Mul"
        self.make_mul(mul_name, [root_input, f"{sigmoid_name}/output_0"], dtype=dtype, shape=shape)
        return f"{mul_name}/output_0"

    def _make_grouped_rms_norm(self, basename, root_input, weight, seq_dim):
        """``(1 + weight) * RMSNorm(x)`` where the norm groups are ``hidden_size`` wide.

        ``x`` is ``[B, S, hc_count * hidden_size]`` and the RMS is taken independently over each
        of the ``hc_count`` groups, while ``weight`` spans the whole ``hc_count * hidden_size``
        axis.  The scale therefore cannot be folded into the normalization op.
        """
        wide_shape = ["batch_size", seq_dim, self.hc_hidden_size]
        grouped_shape = ["batch_size", seq_dim, self.hc_count, self.hidden_size]

        reshape_name = f"{basename}/group/Reshape"
        self.make_reshape(
            reshape_name,
            [root_input, f"/model/constants/INT64/[0, 0, {self.hc_count}, {self.hidden_size}]"],
            self.io_dtype,
            grouped_shape,
        )

        ones_name = f"{basename}/norm/ones"
        self.make_initializer(torch.ones(self.hidden_size), ones_name, to=self.io_dtype)

        norm_name = f"{basename}/norm/SimplifiedLayerNormalization"
        norm_output = f"{norm_name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[f"{reshape_name}/output_0", ones_name],
            outputs=[norm_output],
            name=norm_name,
            domain=None,
            axis=-1,
            stash_type=1,
            epsilon=self.layernorm_attrs["epsilon"],
        )
        self.make_value(norm_output, self.io_dtype, shape=grouped_shape)

        flatten_name = f"{basename}/flatten/Reshape"
        self.make_reshape(
            flatten_name,
            [norm_output, f"/model/constants/INT64/[0, 0, {self.hc_hidden_size}]"],
            self.io_dtype,
            wide_shape,
        )

        # (1 + weight), pre-baked exactly like the base class's `add_offset` handling.
        scale_name = f"{basename}/norm/weight"
        self.make_initializer(weight + 1, scale_name, to=self.io_dtype)

        scale_mul_name = f"{basename}/norm/Mul"
        self.make_mul(scale_mul_name, [f"{flatten_name}/output_0", scale_name], dtype=self.io_dtype, shape=wide_shape)
        return f"{scale_mul_name}/output_0"

    #################################################################################
    # Hyper-connections
    #################################################################################

    def _make_gated_residual(self, basename, gated_residual, root_input, use_combine=True):
        """``Qwen4ExpTextGatedResidual``.

        Returns ``(mixed_input, hyper_input, injection_weights)``; the last two are ``None``
        when ``use_combine`` is false (the final ``hyper_connection_mixer``).
        """
        seq_dim = self.sequence_dim_name()
        wide_shape = ["batch_size", seq_dim, self.hc_hidden_size]
        grouped_shape = ["batch_size", seq_dim, self.hc_count, self.hidden_size]
        narrow_shape = ["batch_size", seq_dim, self.hidden_size]

        normed = self._make_grouped_rms_norm(f"{basename}/hc_norm", root_input, gated_residual.hc_norm.weight, seq_dim)

        down_name = self.make_matmul(
            gated_residual.input_mix_weight_down, f"{basename}/input_mix_weight_down/MatMul", normed
        )
        down_scaled_name = f"{basename}/input_mix_weight_down/Div"
        self.make_div(
            down_scaled_name,
            [f"{down_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{float(self.hc_count)}"],
            dtype=self.io_dtype,
            shape=["batch_size", seq_dim, self.hc_lowrank],
        )
        silu_output = self._make_silu(
            f"{basename}/input_mix_weight_down/silu",
            f"{down_scaled_name}/output_0",
            self.io_dtype,
            ["batch_size", seq_dim, self.hc_lowrank],
        )

        up_name = self.make_matmul(gated_residual.input_mix_weight_up, f"{basename}/input_mix_weight_up/MatMul", silu_output)
        mix_sigmoid_name = f"{basename}/input_mix_weight_up/Sigmoid"
        self.make_sigmoid(mix_sigmoid_name, f"{up_name}/output_0", self.io_dtype, shape=wide_shape)

        # mixed = mean_over_streams(mix * normed)
        mix_reshape_name = f"{basename}/mix/Reshape"
        self.make_reshape(
            mix_reshape_name,
            [f"{mix_sigmoid_name}/output_0", f"/model/constants/INT64/[0, 0, {self.hc_count}, {self.hidden_size}]"],
            self.io_dtype,
            grouped_shape,
        )
        normed_reshape_name = f"{basename}/mix/normed/Reshape"
        self.make_reshape(
            normed_reshape_name,
            [normed, f"/model/constants/INT64/[0, 0, {self.hc_count}, {self.hidden_size}]"],
            self.io_dtype,
            grouped_shape,
        )
        weighted_name = f"{basename}/mix/Mul"
        self.make_mul(
            weighted_name,
            [f"{mix_reshape_name}/output_0", f"{normed_reshape_name}/output_0"],
            dtype=self.io_dtype,
            shape=grouped_shape,
        )
        mixed_mean_name = f"{basename}/mix/ReduceMean"
        self.make_reduce_mean(
            mixed_mean_name,
            [f"{weighted_name}/output_0", "/model/constants/INT64/[-2]"],
            dtype=self.io_dtype,
            shape=narrow_shape,
        )
        mixed_output = f"{mixed_mean_name}/output_0"

        if not use_combine:
            return mixed_output, None, None

        inject_name = self.make_matmul(gated_residual.block_inject_weight, f"{basename}/block_inject_weight/MatMul", normed)
        inject_scaled_name = f"{basename}/block_inject_weight/Div"
        self.make_div(
            inject_scaled_name,
            [f"{inject_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{float(self.hc_count)}"],
            dtype=self.io_dtype,
            shape=["batch_size", seq_dim, self.hc_count],
        )
        inject_sigmoid_name = f"{basename}/block_inject_weight/Sigmoid"
        self.make_sigmoid(
            inject_sigmoid_name,
            f"{inject_scaled_name}/output_0",
            self.io_dtype,
            shape=["batch_size", seq_dim, self.hc_count],
        )
        inject_mul_name = f"{basename}/block_inject_weight/Mul"
        self.make_mul(
            inject_mul_name,
            [f"{inject_sigmoid_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/2.0"],
            dtype=self.io_dtype,
            shape=["batch_size", seq_dim, self.hc_count],
        )

        return mixed_output, root_input, f"{inject_mul_name}/output_0"

    def _make_injection(self, basename, hyper_input, block_output, injection_weights):
        """``hyper_input + (block_output[..., None, :] * injection_weights[..., :, None]).flatten(-2)``"""
        seq_dim = self.sequence_dim_name()
        grouped_shape = ["batch_size", seq_dim, self.hc_count, self.hidden_size]
        wide_shape = ["batch_size", seq_dim, self.hc_hidden_size]

        block_unsqueeze_name = f"{basename}/block/Unsqueeze"
        self.make_unsqueeze(
            block_unsqueeze_name,
            [block_output, "/model/constants/INT64/[-2]"],
            self.io_dtype,
            ["batch_size", seq_dim, 1, self.hidden_size],
        )
        weight_unsqueeze_name = f"{basename}/weights/Unsqueeze"
        self.make_unsqueeze(
            weight_unsqueeze_name,
            [injection_weights, "/model/constants/INT64/[-1]"],
            self.io_dtype,
            ["batch_size", seq_dim, self.hc_count, 1],
        )
        injection_name = f"{basename}/Mul"
        self.make_mul(
            injection_name,
            [f"{block_unsqueeze_name}/output_0", f"{weight_unsqueeze_name}/output_0"],
            dtype=self.io_dtype,
            shape=grouped_shape,
        )
        flatten_name = f"{basename}/Reshape"
        self.make_reshape(
            flatten_name,
            [f"{injection_name}/output_0", f"/model/constants/INT64/[0, 0, {self.hc_hidden_size}]"],
            self.io_dtype,
            wide_shape,
        )
        add_name = f"{basename}/Add"
        self.make_add(add_name, [hyper_input, f"{flatten_name}/output_0"], dtype=self.io_dtype, shape=wide_shape)
        return f"{add_name}/output_0"

    #################################################################################
    # PLE
    #################################################################################

    def _make_ple_layer(self, layer_id, ple, root_input):
        """``Qwen4ExpTextPLELayer``: hashed n-gram embedding, gated against the residual streams,
        plus a dilated depthwise short convolution.  Returns the widened output tensor name."""
        basename = f"/model/layers.{layer_id}/ple"
        seq_dim = self.sequence_dim_name()
        wide_shape = ["batch_size", seq_dim, self.hc_hidden_size]
        ple_layer_index = self.ple_layer_index[layer_id]

        # --- deterministic hashing constants ---
        multipliers = build_layer_multipliers(self.vocab_size, self.ngram_size, ple_layer_index, self.ngram_seed)
        head_vocab_sizes, head_offsets, total_vocab = ngram_head_vocab_layout(
            self.ngram_vocab_size_base, self.ngram_heads, ple_layer_index
        )

        multipliers_name = f"model.layers.{layer_id}.ple.layer_multipliers"
        vocab_sizes_name = f"model.layers.{layer_id}.ple.ngram_heads_vocab_sizes"
        offsets_name = f"model.layers.{layer_id}.ple.ngram_heads_offsets"
        self.make_initializer(torch.tensor(multipliers, dtype=torch.int64), multipliers_name)
        self.make_initializer(torch.tensor(head_vocab_sizes, dtype=torch.int64), vocab_sizes_name)
        self.make_initializer(torch.tensor(head_offsets, dtype=torch.int64), offsets_name)

        # --- NGramHashMapping ---
        hash_name = f"{basename}/NGramHashMapping"
        ngram_ids = f"{hash_name}/output_0"
        present_tokens = self.output_names[f"present_state.{layer_id}.ple_tokens"]
        self.make_node(
            "NGramHashMapping",
            inputs=[
                self.input_names["input_ids"],
                multipliers_name,
                vocab_sizes_name,
                offsets_name,
                self.input_names[f"past_state.{layer_id}.ple_tokens"],
            ],
            outputs=[ngram_ids, present_tokens],
            name=hash_name,
            domain="com.microsoft",
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
            eos_token_id=self.ngram_eos_token_id,
            version=2,
        )
        self.make_value(ngram_ids, ir.DataType.INT64, shape=["batch_size", seq_dim, self.ngram_heads])
        self.make_value(present_tokens, ir.DataType.INT64, shape=["batch_size", self.ngram_size - 1])

        # --- n-gram embedding table ---
        head_dim_per_ngram = self.ple_embed_dim // self.ngram_heads
        embed_name = f"model.layers.{layer_id}.ple.ple_embedding.ngram_embedding.weight"
        self.make_initializer(ple.ple_embedding.ngram_embedding.weight, embed_name, to=self.io_dtype)
        gather_name = f"{basename}/ple_embedding/Gather"
        self.make_gather(
            gather_name,
            [embed_name, ngram_ids],
            dtype=self.io_dtype,
            shape=["batch_size", seq_dim, self.ngram_heads, head_dim_per_ngram],
            axis=0,
        )
        embed_flat_name = f"{basename}/ple_embedding/Reshape"
        self.make_reshape(
            embed_flat_name,
            [f"{gather_name}/output_0", f"/model/constants/INT64/[0, 0, {self.ple_embed_dim}]"],
            self.io_dtype,
            ["batch_size", seq_dim, self.ple_embed_dim],
        )
        embeddings = f"{embed_flat_name}/output_0"

        # --- key / value projections ---
        key_name = self.make_matmul(ple.key_proj, f"{basename}/key_proj/MatMul", embeddings)
        key_normed = self._make_grouped_rms_norm(
            f"{basename}/norm_key", f"{key_name}/output_0", ple.norm_key.weight, seq_dim
        )
        value_name = self.make_matmul(ple.value_proj, f"{basename}/value_proj/MatMul", embeddings)
        query_normed = self._make_grouped_rms_norm(f"{basename}/norm_query", root_input, ple.norm_query.weight, seq_dim)

        # --- EngramGate ---
        gate_name = f"{basename}/EngramGate"
        gated_value = f"{gate_name}/output_0"
        self.make_node(
            "EngramGate",
            inputs=[key_normed, query_normed, f"{value_name}/output_0"],
            outputs=[gated_value],
            name=gate_name,
            domain="com.microsoft",
            num_streams=self.hc_count,
            hidden_size=self.hidden_size,
            epsilon=1e-6,
        )
        self.make_value(gated_value, self.io_dtype, shape=wide_shape)

        gated_value_normed = self._make_grouped_rms_norm(
            f"{basename}/norm_conv", gated_value, ple.norm_conv.weight, seq_dim
        )

        # --- dilated depthwise short convolution ---
        conv_output = self._make_ple_short_conv(layer_id, ple, gated_value_normed)

        add_name = f"{basename}/Add"
        self.make_add(add_name, [gated_value, conv_output], dtype=self.io_dtype, shape=wide_shape)
        return f"{add_name}/output_0"

    def _make_ple_short_conv(self, layer_id, ple, root_input):
        basename = f"/model/layers.{layer_id}/ple/conv1d"
        seq_dim = self.sequence_dim_name()
        channels_first_shape = ["batch_size", self.hc_hidden_size, seq_dim]

        transpose_in_name = f"{basename}/in/Transpose"
        self.make_transpose(transpose_in_name, root_input, self.io_dtype, channels_first_shape, perm=[0, 2, 1])

        weight_name = f"model.layers.{layer_id}.ple.conv1d.weight"
        self.make_initializer(ple.conv1d.weight, weight_name, to=self.io_dtype)

        conv_name = f"{basename}/ShortConvWithState"
        conv_output = f"{conv_name}/output_0"
        present_conv = self.output_names[f"present_state.{layer_id}.ple_conv"]
        self.make_node(
            "ShortConvWithState",
            inputs=[
                f"{transpose_in_name}/output_0",
                weight_name,
                self.input_names[f"past_state.{layer_id}.ple_conv"],
            ],
            outputs=[conv_output, present_conv],
            name=conv_name,
            domain="com.microsoft",
            dilation=self.ngram_size,
            activation="silu",
            group=self.hc_hidden_size,
        )
        self.make_value(conv_output, self.io_dtype, shape=channels_first_shape)
        self.make_value(present_conv, self.io_dtype, shape=["batch_size", self.hc_hidden_size, self.ple_conv_state_len])

        transpose_out_name = f"{basename}/out/Transpose"
        self.make_transpose(
            transpose_out_name,
            conv_output,
            self.io_dtype,
            ["batch_size", seq_dim, self.hc_hidden_size],
            perm=[0, 2, 1],
        )
        return f"{transpose_out_name}/output_0"

    #################################################################################
    # QSA attention
    #################################################################################

    def _make_qsa_indexer(self, layer_id, indexer, root_input):
        """Emit the QSA indexer query/key projections + per-head RMS norms.

        Block pooling, RoPE on block starts, ReLU scoring and top-k selection live inside the
        ``QwenSparseAttention`` / ``SparsePagedAttention`` kernel.
        """
        basename = f"/model/layers.{layer_id}/attn/indexer"
        seq_dim = self.sequence_dim_name()
        q_dim = self.indexer_n_heads * self.indexer_head_dim
        k_dim = self.indexer_kv_heads * self.indexer_head_dim

        qk_name = self.make_matmul(indexer.index_qk_proj, f"{basename}/index_qk_proj/MatMul", root_input)

        split_name = f"{basename}/Split"
        q_output = f"{split_name}/output_0"
        k_output = f"{split_name}/output_1"
        self.make_node(
            "Split",
            [f"{qk_name}/output_0", f"/model/constants/INT64/[{q_dim}, {k_dim}]"],
            [q_output, k_output],
            name=split_name,
            axis=-1,
        )
        self.make_value(q_output, self.io_dtype, ["batch_size", seq_dim, q_dim])
        self.make_value(k_output, self.io_dtype, ["batch_size", seq_dim, k_dim])

        q_normed = self._make_indexer_norm(f"{basename}/q_layernorm", q_output, indexer.q_layernorm.weight, self.indexer_n_heads)
        k_normed = self._make_indexer_norm(f"{basename}/k_layernorm", k_output, indexer.k_layernorm.weight, self.indexer_kv_heads)
        return q_normed, k_normed

    def _make_indexer_norm(self, basename, root_input, weight, num_heads):
        """Per-head RMSNorm over ``indexer_head_dim`` with the ``(1 + weight)`` offset."""
        seq_dim = self.sequence_dim_name()
        flat_shape = ["batch_size", seq_dim, num_heads * self.indexer_head_dim]
        head_shape = ["batch_size", seq_dim, num_heads, self.indexer_head_dim]

        reshape_name = f"{basename}/Reshape"
        self.make_reshape(
            reshape_name,
            [root_input, f"/model/constants/INT64/[0, 0, {num_heads}, {self.indexer_head_dim}]"],
            self.io_dtype,
            head_shape,
        )

        weight_name = f"{basename}/weight"
        self.make_initializer(weight + self.layernorm_attrs["add_offset"], weight_name, to=self.io_dtype)

        norm_name = f"{basename}/SimplifiedLayerNormalization"
        norm_output = f"{norm_name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[f"{reshape_name}/output_0", weight_name],
            outputs=[norm_output],
            name=norm_name,
            domain=None,
            axis=-1,
            stash_type=1,
            epsilon=self.layernorm_attrs["epsilon"],
        )
        self.make_value(norm_output, self.io_dtype, shape=head_shape)

        flatten_name = f"{basename}/flatten/Reshape"
        self.make_reshape(
            flatten_name,
            [norm_output, f"/model/constants/INT64/[0, 0, {num_heads * self.indexer_head_dim}]"],
            self.io_dtype,
            flat_shape,
        )
        return f"{flatten_name}/output_0"

    def _make_qsa_attention_op(self, name, layer_id, q_path, k_path, v_path, indexer_q, indexer_k, **kwargs):
        seq_dim = self.sequence_dim_name()
        output = f"{name}/output_0"
        past_k = kwargs.get("past_k") or f"past_key_values.{layer_id}.key"
        past_v = kwargs.get("past_v") or f"past_key_values.{layer_id}.value"
        present_k = kwargs.get("present_k") or f"present.{layer_id}.key"
        present_v = kwargs.get("present_v") or f"present.{layer_id}.value"
        present_indexer_key = self.output_names.get(f"present_state.{layer_id}.indexer_key", "")
        past_indexer_key = self.input_names.get(f"past_state.{layer_id}.indexer_key", "")

        common_attrs = {
            "num_heads": self.num_attn_heads,
            "kv_num_heads": self.num_kv_heads,
            "scale": self.attention_attrs["scale"],
            "softcap": self.attention_attrs["softcap"],
            "do_rotary": 0,
            "indexer_num_heads": self.indexer_n_heads,
            "indexer_kv_num_heads": self.indexer_kv_heads,
            "indexer_head_size": self.indexer_head_dim,
            "indexer_token_budget": self.indexer_budget,
            "indexer_compress_ratio": self.indexer_compress_ratio,
        }

        if self.use_paged_attention:
            # SparsePagedAttention:
            #   0..2  query, key, value
            #   3..4  key_cache, value_cache   (updated in place)
            #   5..8  cumulative_sequence_lengths, past_sequence_lengths, block_table,
            #         attention_metadata
            #   9..10 cos_cache, sin_cache     (unused, mRoPE is applied in the graph)
            #  11..13 indexer_query, indexer_key, past_indexer_key
            inputs = [
                q_path,
                k_path,
                v_path,
                past_k,
                past_v,
                self.input_names["cumulative_sequence_lengths"],
                self.input_names["past_sequence_lengths"],
                self.input_names["block_table"],
                self.input_names["attention_metadata"],
                "",
                "",
                indexer_q,
                indexer_k,
                past_indexer_key,
            ]
            outputs = [output, present_indexer_key] if present_indexer_key else [output]
            self.make_node(
                "SparsePagedAttention",
                inputs=inputs,
                outputs=outputs,
                name=name,
                domain="com.microsoft",
                **common_attrs,
            )
        else:
            # QwenSparseAttention:
            #   0..2  query, key, value
            #   3..4  past_key, past_value
            #   5..6  seqlens_k, total_sequence_length
            #   7..8  cos_cache, sin_cache     (unused, mRoPE is applied in the graph)
            #   9..11 indexer_query, indexer_key, past_indexer_key
            inputs = [
                q_path,
                k_path,
                v_path,
                past_k,
                past_v,
                f"{self.mask_attrs['seqlens_k']}/output_0",
                f"{self.mask_attrs['total_seq_len']}/output_0",
                "",
                "",
                indexer_q,
                indexer_k,
                past_indexer_key,
            ]
            outputs = [output, present_k, present_v]
            if present_indexer_key:
                outputs.append(present_indexer_key)
            self.make_node(
                "QwenSparseAttention",
                inputs=inputs,
                outputs=outputs,
                name=name,
                domain="com.microsoft",
                **common_attrs,
            )
        self.make_value(
            output, self.io_dtype, shape=["batch_size", seq_dim, self.num_attn_heads * self.head_size]
        )
        if present_indexer_key:
            self.make_value(
                present_indexer_key,
                self.io_dtype,
                shape=["batch_size", self.indexer_kv_heads, "total_sequence_length", self.indexer_head_dim],
            )
        return output

    def make_attention_op(self, name, **kwargs):
        """Route Qwen4Exp full attention through the sparse (QSA) attention op.

        ``_make_full_attention`` stashes the indexer query/key tensor names in
        ``self._pending_indexer`` before delegating to the Qwen3.5 gated-attention body, which
        eventually calls back into this method.
        """
        pending = getattr(self, "_pending_indexer", None)
        if not pending:
            return super().make_attention_op(name, **kwargs)
        indexer_q, indexer_k = pending
        self._make_qsa_attention_op(
            name,
            kwargs["layer_id"],
            kwargs["q_path"],
            kwargs["k_path"],
            kwargs["v_path"],
            indexer_q,
            indexer_k,
            past_k=kwargs.get("past_k", ""),
            past_v=kwargs.get("past_v", ""),
            present_k=kwargs.get("present_k", ""),
            present_v=kwargs.get("present_v", ""),
        )

    def _make_full_attention(self, layer_id, attn, root_input):
        """Same gated attention as Qwen3.5 plus the QSA indexer side-projection."""
        indexer = getattr(attn, "indexer", None)
        if indexer is None or not self.indexer_head_dim:
            super()._make_full_attention(layer_id, attn, root_input)
            return

        indexer_q, indexer_k = self._make_qsa_indexer(layer_id, indexer, root_input)
        self._pending_indexer = (indexer_q, indexer_k)
        try:
            super()._make_full_attention(layer_id, attn, root_input)
        finally:
            self._pending_indexer = None

    def sequence_dim_name(self):
        return "num_tokens" if self.use_paged_attention else "sequence_length"

    #################################################################################
    # Layer assembly
    #################################################################################

    def make_layer(self, layer_id, layer):
        """Hyper-connection decoder layer.

        ``layernorm_attrs["root_input"]`` holds the widened residual stream on entry, and the
        widened stream is written back to it on exit.  Attention/MoE write their (narrow) block
        output to ``layernorm_attrs["skip_input"]``, which is re-injected into the streams.
        """
        seq_dim = self.sequence_dim_name()
        wide_shape = ["batch_size", seq_dim, self.hc_hidden_size]

        if self.layernorm_attrs["first_layernorm"]:
            # inputs_embeds -> repeat over hc_count streams
            tile_name = "/model/hyper_connections/Tile"
            self.make_tile(
                tile_name,
                [self.layernorm_attrs["root_input"], f"/model/constants/INT64/[1, 1, {self.hc_count}]"],
                self.io_dtype,
                wide_shape,
            )
            hidden_states = f"{tile_name}/output_0"
        else:
            hidden_states = self.layernorm_attrs["root_input"]

        # --- PLE ---
        if layer_id in self.ple_layer_index:
            ple_output = self._make_ple_layer(layer_id, layer.ple, hidden_states)
            ple_add_name = f"/model/layers.{layer_id}/ple/residual/Add"
            self.make_add(ple_add_name, [hidden_states, ple_output], dtype=self.io_dtype, shape=wide_shape)
            hidden_states = f"{ple_add_name}/output_0"

        # --- attention block ---
        mixed, hyper_input, injection = self._make_gated_residual(
            f"/model/layers.{layer_id}/attn_hyper_connection", layer.attn_hyper_connection, hidden_states
        )
        is_linear = self.layer_types[layer_id] == "linear_attention"
        attn_module = layer.linear_attn if is_linear else layer.self_attn
        self.layernorm_attrs["output_0"] = mixed
        self.make_attention(layer_id, attn_module, root_input=mixed)
        hidden_states = self._make_injection(
            f"/model/layers.{layer_id}/attn_hyper_connection/inject",
            hyper_input,
            self.layernorm_attrs["skip_input"],
            injection,
        )

        # --- MoE block ---
        mixed, hyper_input, injection = self._make_gated_residual(
            f"/model/layers.{layer_id}/mlp_hyper_connection", layer.mlp_hyper_connection, hidden_states
        )
        self.layernorm_attrs["output_0"] = mixed
        self.make_moe(layer_id, layer.mlp, root_input=mixed)
        hidden_states = self._make_injection(
            f"/model/layers.{layer_id}/mlp_hyper_connection/inject",
            hyper_input,
            self.layernorm_attrs["skip_input"],
            injection,
        )

        self.layernorm_attrs["root_input"] = hidden_states
        self.layernorm_attrs["skip_input"] = hidden_states
        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    def make_moe(self, layer_id, mlp, root_input):
        """Routed experts + shared expert, using the inherited Qwen3.5 MoE emitter.

        Qwen4Exp stores the routed experts as packed 3-D tensors, so the module is wrapped in a
        view that satisfies the base class's per-expert iteration probe.
        """
        if not isinstance(getattr(mlp, "experts", None), torch.nn.ModuleList):
            mlp = _PackedMoeView(mlp)
        super().make_moe(layer_id, mlp, root_input)

    def make_mlp(self, layer_id, mlp, root_input):
        # Every Qwen4Exp layer is MoE; keep the base dispatch honest.
        self.make_moe(layer_id, mlp, root_input)

    #################################################################################
    # Final mixer / model glue
    #################################################################################

    def has_final_norm(self, module, orig_model):
        model = orig_model
        for attr_path in (("model", "language_model"), ("model",), ()):
            target = model
            for attr in attr_path:
                target = getattr(target, attr, None)
                if target is None:
                    break
            if target is not None and getattr(target, "hyper_connection_mixer", None) is module:
                return True
        return False

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        if location == "final_norm":
            self._make_final_mixer(layernorm)
            return
        super().make_layernorm(layer_id, layernorm, skip, simple, location)

    def _make_final_mixer(self, gated_residual):
        """``hyper_connection_mixer``: widened residual stream -> ``[B, S, hidden_size]``."""
        mixed, _, _ = self._make_gated_residual(
            "/model/hyper_connection_mixer",
            gated_residual,
            self.layernorm_attrs["root_input"],
            use_combine=False,
        )
        if self.include_hidden_states or self.exclude_lm_head:
            hidden_states_name = self.output_names["hidden_states"]
            identity_name = "/model/hyper_connection_mixer/hidden_states/Identity"
            self.make_node(
                "Identity", inputs=[mixed], outputs=[hidden_states_name], name=identity_name
            )
            self.make_value(
                hidden_states_name,
                self.io_dtype,
                shape=["batch_size", self.sequence_dim_name(), self.hidden_size],
            )
            mixed = hidden_states_name
        self.layernorm_attrs["output_0"] = mixed

    def load_weights(self, input_path):
        weights = super().load_weights(input_path)
        # `Model.make_model` does `del self.weights` when it finishes; the composite builder
        # needs the vision tower and token embedding afterwards, so keep a second reference.
        self._hf_model = weights
        return weights


#####################################################################################
# embedding.onnx
#####################################################################################


class Qwen4ExpEmbeddingModel(Model):
    """``embedding.onnx``: token embedding lookup with image/video feature scatter.

    inputs  : ``input_ids`` int64 ``[batch_size, sequence_length]``
              ``image_features`` ``[num_image_tokens, hidden_size]``
    outputs : ``inputs_embeds`` ``[batch_size, sequence_length, hidden_size]``

    Rows of ``input_ids`` equal to ``image_token_id`` or ``video_token_id`` are replaced, in
    order, by the rows of ``image_features``.  Passing a zero-row ``image_features`` makes the
    scatter a no-op, which is the text-only path.
    """

    DEFAULT_FILENAME = "embedding.onnx"

    def _get_model_type(self, config):
        return "Qwen4ExpEmbeddingModel"

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        extra_options = dict(extra_options)
        extra_options["filename"] = self.DEFAULT_FILENAME
        extra_options.pop("use_paged_attention", None)
        extra_options.pop("enable_mtp", None)
        super().__init__(config, io_dtype, io_dtype, ep, cache_dir, extra_options)

        self.image_token_id = int(getattr(config, "image_token_id", -1))
        self.video_token_id = int(getattr(config, "video_token_id", self.image_token_id))
        self.embed_tokens = None

    def make_inputs_and_outputs(self):
        self.model.graph.inputs.extend(
            [
                self.make_value("input_ids", ir.DataType.INT64, shape=["batch_size", "sequence_length"]),
                self.make_value("image_features", self.io_dtype, shape=["num_image_tokens", self.hidden_size]),
            ]
        )
        self.model.graph.outputs.append(
            self.make_value(
                "inputs_embeds", self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size]
            )
        )

    def make_model(self, input_path):
        self.make_inputs_and_outputs()

        embed_weight = "model.embed_tokens.weight"
        self.make_initializer(self.embed_tokens.weight, embed_weight, to=self.io_dtype)

        gather_name = "/model/embed_tokens/Gather"
        self.make_gather(
            gather_name,
            [embed_weight, "input_ids"],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
            axis=0,
        )
        embeds = f"{gather_name}/output_0"

        shape_name = "/model/embeds/Shape"
        self.make_shape(shape_name, embeds, shape=[3])

        flat_embeds_name = "/model/embeds/Reshape"
        self.make_reshape(
            flat_embeds_name,
            [embeds, f"/model/constants/INT64/[-1, {self.hidden_size}]"],
            self.io_dtype,
            ["batch_size * sequence_length", self.hidden_size],
        )

        flat_ids_name = "/model/input_ids/Reshape"
        self.make_reshape(
            flat_ids_name,
            ["input_ids", "/model/constants/INT64/[-1]"],
            ir.DataType.INT64,
            ["batch_size * sequence_length"],
        )

        image_eq_name = "/model/image_token/Equal"
        self.make_equal(
            image_eq_name,
            [f"{flat_ids_name}/output_0", f"/model/constants/INT64/{self.image_token_id}"],
            shape=["batch_size * sequence_length"],
        )
        video_eq_name = "/model/video_token/Equal"
        self.make_equal(
            video_eq_name,
            [f"{flat_ids_name}/output_0", f"/model/constants/INT64/{self.video_token_id}"],
            shape=["batch_size * sequence_length"],
        )
        or_name = "/model/multimodal_token/Or"
        or_output = f"{or_name}/output_0"
        self.make_node(
            "Or",
            inputs=[f"{image_eq_name}/output_0", f"{video_eq_name}/output_0"],
            outputs=[or_output],
            name=or_name,
        )
        self.make_value(or_output, ir.DataType.BOOL, shape=["batch_size * sequence_length"])

        nonzero_name = "/model/multimodal_token/NonZero"
        nonzero_output = f"{nonzero_name}/output_0"
        self.make_node("NonZero", inputs=[or_output], outputs=[nonzero_output], name=nonzero_name)
        self.make_value(nonzero_output, ir.DataType.INT64, shape=[1, "num_image_tokens"])

        indices_name = "/model/multimodal_token/Transpose"
        self.make_transpose(
            indices_name, nonzero_output, ir.DataType.INT64, ["num_image_tokens", 1], perm=[1, 0]
        )

        scatter_name = "/model/image_features/ScatterND"
        scatter_output = f"{scatter_name}/output_0"
        self.make_node(
            "ScatterND",
            inputs=[f"{flat_embeds_name}/output_0", f"{indices_name}/output_0", "image_features"],
            outputs=[scatter_output],
            name=scatter_name,
        )
        self.make_value(
            scatter_output, self.io_dtype, shape=["batch_size * sequence_length", self.hidden_size]
        )

        unflatten_name = "/model/inputs_embeds/Reshape"
        self.make_node(
            "Reshape",
            inputs=[scatter_output, f"{shape_name}/output_0"],
            outputs=["inputs_embeds"],
            name=unflatten_name,
        )


#####################################################################################
# vision.onnx
#####################################################################################


class Qwen4ExpVisionModel(Model):
    """``vision.onnx``: the Qwen4Exp ViT tower plus the spatial patch merger.

    inputs
        ``pixel_values``           ``[num_patches, in_channels * temporal_patch_size * patch_size ** 2]``
        ``pos_embed_indices``      int64 ``[num_patches, 4]``  -- bilinear source cells
        ``pos_embed_weights``      ``[num_patches, 4]``        -- bilinear weights
        ``vision_cos`` / ``vision_sin`` ``[num_patches, head_dim]``
        ``vision_attention_bias``  ``[1, 1, num_patches, num_patches]`` -- 0 inside an image, -inf across images
    outputs
        ``image_features`` ``[num_patches / spatial_merge_size ** 2, out_hidden_size]``

    The geometry inputs replace the reference implementation's per-image Python loops over
    ``grid_thw`` and must be produced by the host pre-processor.
    """

    DEFAULT_FILENAME = "vision.onnx"

    def _get_model_type(self, config):
        return "Qwen4ExpVisionModel"

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        extra_options = dict(extra_options)
        extra_options["filename"] = self.DEFAULT_FILENAME
        extra_options["exclude_lm_head"] = True
        extra_options.pop("use_paged_attention", None)
        extra_options.pop("enable_mtp", None)

        vision_config = config.vision_config
        text_like = copy.deepcopy(getattr(config, "text_config", config))
        text_like.hidden_size = vision_config.hidden_size
        text_like.num_attention_heads = vision_config.num_heads
        text_like.num_key_value_heads = vision_config.num_heads
        text_like.head_dim = vision_config.hidden_size // vision_config.num_heads
        text_like.num_hidden_layers = vision_config.depth
        text_like.intermediate_size = vision_config.intermediate_size
        super().__init__(text_like, io_dtype, io_dtype, ep, cache_dir, extra_options)

        self.vision_config = vision_config
        self.vision_hidden_size = vision_config.hidden_size
        self.vision_intermediate_size = vision_config.intermediate_size
        self.vision_num_heads = vision_config.num_heads
        self.vision_head_dim = self.vision_hidden_size // self.vision_num_heads
        self.vision_depth = vision_config.depth
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.in_channels = vision_config.in_channels
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size
        self.merger_hidden_size = self.vision_hidden_size * self.spatial_merge_size**2
        self.patch_dim = self.in_channels * self.temporal_patch_size * self.patch_size**2
        self.visual = None

    def make_inputs_and_outputs(self):
        self.model.graph.inputs.extend(
            [
                self.make_value("pixel_values", self.io_dtype, shape=["num_patches", self.patch_dim]),
                self.make_value("pos_embed_indices", ir.DataType.INT64, shape=["num_patches", 4]),
                self.make_value("pos_embed_weights", self.io_dtype, shape=["num_patches", 4]),
                self.make_value("vision_cos", self.io_dtype, shape=["num_patches", self.vision_head_dim]),
                self.make_value("vision_sin", self.io_dtype, shape=["num_patches", self.vision_head_dim]),
                self.make_value(
                    "vision_attention_bias", self.io_dtype, shape=[1, 1, "num_patches", "num_patches"]
                ),
            ]
        )
        self.model.graph.outputs.append(
            self.make_value("image_features", self.io_dtype, shape=["num_merged_patches", self.out_hidden_size])
        )

    def _make_linear(self, basename, module, root_input, out_features, shape):
        matmul_name = self.make_matmul(module, f"{basename}/MatMul", root_input)
        output = f"{matmul_name}/output_0"
        if getattr(module, "bias", None) is not None:
            bias_name = f"{basename}/bias"
            self.make_initializer(module.bias, bias_name, to=self.io_dtype)
            add_name = f"{basename}/Add"
            self.make_add(add_name, [output, bias_name], dtype=self.io_dtype, shape=shape)
            output = f"{add_name}/output_0"
        return output

    def _make_layernorm(self, basename, module, root_input, shape, normalized_size):
        weight_name = f"{basename}/weight"
        bias_name = f"{basename}/bias"
        self.make_initializer(module.weight, weight_name, to=self.io_dtype)
        self.make_initializer(module.bias, bias_name, to=self.io_dtype)
        name = f"{basename}/LayerNormalization"
        output = f"{name}/output_0"
        self.make_node(
            "LayerNormalization",
            inputs=[root_input, weight_name, bias_name],
            outputs=[output],
            name=name,
            domain=None,
            axis=-1,
            stash_type=1,
            epsilon=1e-6,
        )
        self.make_value(output, self.io_dtype, shape=shape)
        return output

    def _make_gelu(self, basename, root_input, shape):
        name = f"{basename}/Gelu"
        output = f"{name}/output_0"
        self.make_node("Gelu", inputs=[root_input], outputs=[output], name=name, domain=None, approximate="none")
        self.make_value(output, self.io_dtype, shape=shape)
        return output

    def make_model(self, input_path):
        self.make_inputs_and_outputs()
        visual = self.visual
        patch_shape = ["num_patches", self.vision_hidden_size]

        # --- patch embedding (Conv3d with stride == kernel == a flat MatMul over the patch) ---
        patch_weight = "model.visual.patch_embed.proj.weight"
        self.make_initializer(
            visual.patch_embed.proj.weight.reshape(self.vision_hidden_size, self.patch_dim).transpose(0, 1),
            patch_weight,
            to=self.io_dtype,
        )
        patch_matmul_name = "/model/visual/patch_embed/MatMul"
        patch_output = f"{patch_matmul_name}/output_0"
        self.make_node(
            "MatMul", inputs=["pixel_values", patch_weight], outputs=[patch_output], name=patch_matmul_name
        )
        self.make_value(patch_output, self.io_dtype, shape=patch_shape)

        patch_bias = "model.visual.patch_embed.proj.bias"
        self.make_initializer(visual.patch_embed.proj.bias, patch_bias, to=self.io_dtype)
        patch_bias_name = "/model/visual/patch_embed/Add"
        self.make_add(patch_bias_name, [patch_output, patch_bias], dtype=self.io_dtype, shape=patch_shape)
        hidden_states = f"{patch_bias_name}/output_0"

        # --- interpolated learned position embedding ---
        pos_weight = "model.visual.pos_embed.weight"
        self.make_initializer(visual.pos_embed.weight, pos_weight, to=self.io_dtype)
        pos_gather_name = "/model/visual/pos_embed/Gather"
        self.make_gather(
            pos_gather_name,
            [pos_weight, "pos_embed_indices"],
            dtype=self.io_dtype,
            shape=["num_patches", 4, self.vision_hidden_size],
            axis=0,
        )
        pos_weights_name = "/model/visual/pos_embed/weights/Unsqueeze"
        self.make_unsqueeze(
            pos_weights_name,
            ["pos_embed_weights", "/model/constants/INT64/[-1]"],
            self.io_dtype,
            ["num_patches", 4, 1],
        )
        pos_mul_name = "/model/visual/pos_embed/Mul"
        self.make_mul(
            pos_mul_name,
            [f"{pos_gather_name}/output_0", f"{pos_weights_name}/output_0"],
            dtype=self.io_dtype,
            shape=["num_patches", 4, self.vision_hidden_size],
        )
        pos_sum_name = "/model/visual/pos_embed/ReduceSum"
        self.make_reduce_sum(
            pos_sum_name,
            [f"{pos_mul_name}/output_0", "/model/constants/INT64/[1]"],
            dtype=self.io_dtype,
            shape=patch_shape,
        )
        pos_add_name = "/model/visual/pos_embed/Add"
        self.make_add(
            pos_add_name, [hidden_states, f"{pos_sum_name}/output_0"], dtype=self.io_dtype, shape=patch_shape
        )
        hidden_states = f"{pos_add_name}/output_0"

        # --- transformer blocks ---
        for block_id, block in enumerate(visual.blocks):
            hidden_states = self._make_vision_block(block_id, block, hidden_states)

        # --- patch merger ---
        merger = visual.merger
        normed = self._make_layernorm(
            "/model/visual/merger/norm", merger.norm, hidden_states, patch_shape, self.vision_hidden_size
        )
        merge_reshape_name = "/model/visual/merger/Reshape"
        merged_shape = ["num_merged_patches", self.merger_hidden_size]
        self.make_reshape(
            merge_reshape_name,
            [normed, f"/model/constants/INT64/[-1, {self.merger_hidden_size}]"],
            self.io_dtype,
            merged_shape,
        )
        fc1 = self._make_linear(
            "/model/visual/merger/linear_fc1",
            merger.linear_fc1,
            f"{merge_reshape_name}/output_0",
            self.merger_hidden_size,
            merged_shape,
        )
        act = self._make_gelu("/model/visual/merger/act", fc1, merged_shape)
        fc2_matmul = self.make_matmul(merger.linear_fc2, "/model/visual/merger/linear_fc2/MatMul", act)
        fc2_bias_name = "/model/visual/merger/linear_fc2/bias"
        self.make_initializer(merger.linear_fc2.bias, fc2_bias_name, to=self.io_dtype)
        self.make_node(
            "Add",
            inputs=[f"{fc2_matmul}/output_0", fc2_bias_name],
            outputs=["image_features"],
            name="/model/visual/merger/linear_fc2/Add",
        )

    def _make_vision_block(self, block_id, block, root_input):
        basename = f"/model/visual/blocks.{block_id}"
        patch_shape = ["num_patches", self.vision_hidden_size]

        normed = self._make_layernorm(
            f"{basename}/norm1", block.norm1, root_input, patch_shape, self.vision_hidden_size
        )
        attn_output = self._make_vision_attention(block_id, block.attn, normed)
        add1_name = f"{basename}/attn/Add"
        self.make_add(add1_name, [root_input, attn_output], dtype=self.io_dtype, shape=patch_shape)
        hidden_states = f"{add1_name}/output_0"

        normed2 = self._make_layernorm(
            f"{basename}/norm2", block.norm2, hidden_states, patch_shape, self.vision_hidden_size
        )
        mlp_shape = ["num_patches", self.vision_intermediate_size]
        fc1 = self._make_linear(f"{basename}/mlp/linear_fc1", block.mlp.linear_fc1, normed2, self.vision_intermediate_size, mlp_shape)
        act = self._make_gelu(f"{basename}/mlp/act", fc1, mlp_shape)
        fc2 = self._make_linear(f"{basename}/mlp/linear_fc2", block.mlp.linear_fc2, act, self.vision_hidden_size, patch_shape)
        add2_name = f"{basename}/mlp/Add"
        self.make_add(add2_name, [hidden_states, fc2], dtype=self.io_dtype, shape=patch_shape)
        return f"{add2_name}/output_0"

    def _make_vision_attention(self, block_id, attn, root_input):
        basename = f"/model/visual/blocks.{block_id}/attn"
        patch_shape = ["num_patches", self.vision_hidden_size]
        qkv_shape = ["num_patches", 3 * self.vision_hidden_size]

        qkv = self._make_linear(f"{basename}/qkv", attn.qkv, root_input, 3 * self.vision_hidden_size, qkv_shape)

        split_name = f"{basename}/Split"
        q_out, k_out, v_out = f"{split_name}/output_0", f"{split_name}/output_1", f"{split_name}/output_2"
        self.make_node(
            "Split",
            [qkv, f"/model/constants/INT64/{[self.vision_hidden_size] * 3}"],
            [q_out, k_out, v_out],
            name=split_name,
            axis=-1,
        )
        for value_name in (q_out, k_out, v_out):
            self.make_value(value_name, self.io_dtype, patch_shape)

        q_roped = self._make_vision_rotary(f"{basename}/q_rope", q_out)
        k_roped = self._make_vision_rotary(f"{basename}/k_rope", k_out)

        # [num_patches, hidden] -> [1, num_patches, hidden] for MultiHeadAttention.
        q_3d = self._make_batchify(f"{basename}/q", q_roped)
        k_3d = self._make_batchify(f"{basename}/k", k_roped)
        v_3d = self._make_batchify(f"{basename}/v", v_out)

        attn_name = f"{basename}/MultiHeadAttention"
        attn_output = f"{attn_name}/output_0"
        self.make_node(
            "MultiHeadAttention",
            inputs=[q_3d, k_3d, v_3d, "", "", "vision_attention_bias"],
            outputs=[attn_output],
            name=attn_name,
            domain="com.microsoft",
            num_heads=self.vision_num_heads,
            scale=1.0 / math.sqrt(self.vision_head_dim),
            unidirectional=0,
        )
        self.make_value(attn_output, self.io_dtype, shape=[1, "num_patches", self.vision_hidden_size])

        squeeze_name = f"{basename}/output/Reshape"
        self.make_reshape(
            squeeze_name,
            [attn_output, f"/model/constants/INT64/[-1, {self.vision_hidden_size}]"],
            self.io_dtype,
            patch_shape,
        )
        return self._make_linear(
            f"{basename}/proj", attn.proj, f"{squeeze_name}/output_0", self.vision_hidden_size, patch_shape
        )

    def _make_batchify(self, basename, root_input):
        name = f"{basename}/Unsqueeze"
        self.make_unsqueeze(
            name,
            [root_input, "/model/constants/INT64/[0]"],
            self.io_dtype,
            [1, "num_patches", self.vision_hidden_size],
        )
        return f"{name}/output_0"

    def _make_vision_rotary(self, basename, root_input):
        """``x * cos + rotate_half(x) * sin`` per head over the full ``vision_head_dim``."""
        head_shape = ["num_patches", self.vision_num_heads, self.vision_head_dim]
        half_shape = ["num_patches", self.vision_num_heads, self.vision_head_dim // 2]
        flat_shape = ["num_patches", self.vision_hidden_size]

        reshape_name = f"{basename}/Reshape"
        self.make_reshape(
            reshape_name,
            [root_input, f"/model/constants/INT64/[-1, {self.vision_num_heads}, {self.vision_head_dim}]"],
            self.io_dtype,
            head_shape,
        )
        heads = f"{reshape_name}/output_0"

        cos_name = f"{basename}/cos/Unsqueeze"
        self.make_unsqueeze(
            cos_name,
            ["vision_cos", "/model/constants/INT64/[-2]"],
            self.io_dtype,
            ["num_patches", 1, self.vision_head_dim],
        )
        sin_name = f"{basename}/sin/Unsqueeze"
        self.make_unsqueeze(
            sin_name,
            ["vision_sin", "/model/constants/INT64/[-2]"],
            self.io_dtype,
            ["num_patches", 1, self.vision_head_dim],
        )

        split_name = f"{basename}/half/Split"
        first_half, second_half = f"{split_name}/output_0", f"{split_name}/output_1"
        self.make_node(
            "Split",
            [heads, f"/model/constants/INT64/{[self.vision_head_dim // 2] * 2}"],
            [first_half, second_half],
            name=split_name,
            axis=-1,
        )
        self.make_value(first_half, self.io_dtype, half_shape)
        self.make_value(second_half, self.io_dtype, half_shape)

        neg_name = f"{basename}/half/Neg"
        neg_output = f"{neg_name}/output_0"
        self.make_node("Neg", inputs=[second_half], outputs=[neg_output], name=neg_name)
        self.make_value(neg_output, self.io_dtype, half_shape)

        rotate_name = f"{basename}/half/Concat"
        self.make_concat(rotate_name, [neg_output, first_half], dtype=self.io_dtype, shape=head_shape, axis=-1)

        cos_mul_name = f"{basename}/cos/Mul"
        self.make_mul(cos_mul_name, [heads, f"{cos_name}/output_0"], dtype=self.io_dtype, shape=head_shape)
        sin_mul_name = f"{basename}/sin/Mul"
        self.make_mul(
            sin_mul_name,
            [f"{rotate_name}/output_0", f"{sin_name}/output_0"],
            dtype=self.io_dtype,
            shape=head_shape,
        )
        add_name = f"{basename}/Add"
        self.make_add(
            add_name,
            [f"{cos_mul_name}/output_0", f"{sin_mul_name}/output_0"],
            dtype=self.io_dtype,
            shape=head_shape,
        )
        flatten_name = f"{basename}/flatten/Reshape"
        self.make_reshape(
            flatten_name,
            [f"{add_name}/output_0", f"/model/constants/INT64/[-1, {self.vision_hidden_size}]"],
            self.io_dtype,
            flat_shape,
        )
        return f"{flatten_name}/output_0"


#####################################################################################
# Composite multimodal builder
#####################################################################################


class Qwen4ExpModel(Qwen4ExpTextModel):
    """``Qwen4ExpForConditionalGeneration``: exports ``text.onnx`` + ``embedding.onnx`` + ``vision.onnx``.

    This class *is* the text decoder builder (so every decoder behaviour is inherited rather
    than duplicated) and additionally drives the two auxiliary graphs off the same loaded
    checkpoint.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.multimodal_config = config
        self.embedding_model = Qwen4ExpEmbeddingModel(
            config, io_dtype, onnx_dtype, ep, cache_dir, extra_options
        )
        self.vision_model = (
            Qwen4ExpVisionModel(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
            if getattr(config, "vision_config", None) is not None
            else None
        )

    def make_model(self, input_path):
        super().make_model(input_path)

        hf_model = getattr(self, "_hf_model", None)
        if hf_model is None:
            raise RuntimeError("Qwen4Exp composite export requires the loaded Hugging Face model.")

        inner = getattr(hf_model, "model", hf_model)
        language_model = getattr(inner, "language_model", inner)
        self.embedding_model.embed_tokens = language_model.embed_tokens
        self.embedding_model.make_model(input_path)

        if self.vision_model is not None:
            self.vision_model.visual = inner.visual
            self.vision_model.make_model(input_path)

        del self._hf_model

    def save_model(self, out_dir):
        super().save_model(out_dir)
        self.embedding_model.save_model(out_dir)
        if self.vision_model is not None:
            self.vision_model.save_model(out_dir)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        genai_config["model"]["decoder"]["filename"] = self.filename
        genai_config["model"]["embedding"] = {
            "filename": self.embedding_model.filename,
            "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
            "outputs": {"inputs_embeds": "inputs_embeds"},
        }
        if self.vision_model is not None:
            genai_config["model"]["vision"] = {
                "filename": self.vision_model.filename,
                "spatial_merge_size": self.vision_model.spatial_merge_size,
                "inputs": {
                    "pixel_values": "pixel_values",
                    "pos_embed_indices": "pos_embed_indices",
                    "pos_embed_weights": "pos_embed_weights",
                    "vision_cos": "vision_cos",
                    "vision_sin": "vision_sin",
                    "vision_attention_bias": "vision_attention_bias",
                },
                "outputs": {"image_features": "image_features"},
            }

        for key in ("image_token_id", "video_token_id", "vision_start_token_id"):
            value = getattr(self.multimodal_config, key, None)
            if value is not None:
                genai_config["model"][key] = value

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)
        print("Added 'embedding' and 'vision' sections to genai_config.json")
