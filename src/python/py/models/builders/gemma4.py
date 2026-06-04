# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
"""
Gemma 4 model builder (Part 1 of N: scaffolding).

This builder adds partial support for Google's Gemma 4 model family
(E2B / E4B / 26B-A4B / 31B), released April 2026. It covers the architectural
features that can be handled entirely on the builder side by inheriting
from Gemma 3 and overriding the right hooks:

  * `layer_types`-driven sliding-vs-full attention pattern (Gemma 3 used a
    hard-coded every-6th-layer rule; Gemma 4 exposes a full per-layer list).
  * Per-layer KV-cache head dimension (sliding layers use `head_dim=256`,
    full layers use `global_head_dim=512`) - emitted through the existing
    `make_key_value_cache_shape(layer_id, shape)` extension point.
  * Proportional RoPE for full-attention layers (`rope_theta=1_000_000`,
    `partial_rotary_factor=0.25`), alongside the default sliding RoPE
    (`rope_theta=10_000`).
  * Multimodal wrapper (`Gemma4ForConditionalGeneration`) weight prefix
    stripping - handled at dispatch in `builder.py`.

The following items are explicitly NOT handled by this scaffold and will
raise `NotImplementedError` at build time. They require C++ runtime changes
tracked as follow-up work (see PR description and issue #2062):

  * Per-Layer Embeddings (PLE). Gemma 4's `embed_tokens` produces both
    `inputs_embeds` [B, S, hidden] and `per_layer_inputs`
    [B, S, num_hidden_layers, hidden_size_per_layer_input]. Each transformer
    layer consumes its own slice. The GenAI runtime currently routes a single
    embedding tensor through the decoder stack; supporting PLE requires a
    new side-channel input plumbed through `make_inputs_and_outputs` and the
    C++ model loader.

  * KV cache sharing. Gemma 4 E2B has 35 decoder layers but only 15 own
    unique KV caches (`num_kv_shared_layers=20`). Shared layers alias the
    KV I/O of an earlier layer. The runtime's `kv_cache.cpp` currently
    assumes one cache pair per layer.

  * Per-layer `head_size` in the runtime. Even once this builder emits
    correctly-shaped per-layer `past_key_values.{i}` tensors, `kv_cache.cpp`
    allocates buffers with a single `model.decoder.head_size` value from
    `genai_config.json`. The config schema needs to grow support for either
    a per-layer list or a `head_size` + `global_head_size` pair driven by
    the same `layer_types` pattern.

This file is deliberately small and documented heavily so a maintainer
picking up the follow-up work has a clear, verifiable starting point.

References:
  * Issue: https://github.com/microsoft/onnxruntime-genai/issues/2062
  * Gemma 4 E2B config:
    https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json
  * HuggingFace model card:
    https://huggingface.co/google/gemma-4-E2B-it
  * Community ONNX export (for architectural reference only; uses a
    different I/O contract than GenAI):
    https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX
"""

from .gemma import Gemma3Model


class Gemma4TextModel(Gemma3Model):
    """Text-only decoder for Gemma 4 (E2B / E4B / 26B / 31B).

    Inherits everything that still applies from Gemma 3:
      - Gemma-family layer norm / embedding scaling
      - q_norm / k_norm inside attention
      - Sliding-window attention toggle per layer
      - Multi-cache RoPE (global vs local theta)

    Overrides:
      - `is_local`            -> driven by `config.layer_types`
      - `make_key_value_cache_shape` -> per-layer `head_size` selection
      - `_rope_config_for_layer` / cache wiring for Gemma 4's two RoPE flavours
      - `__init__`            -> records Gemma 4-specific attrs (global head
        dim, KV sharing count, PLE dims, double-wide MLP flag) and raises
        `NotImplementedError` when those features are actually required.
    """

    _SLIDING = "sliding_attention"
    _FULL = "full_attention"

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Gemma 4's multimodal wrapper stores the decoder config under
        # `text_config`; the dispatcher in `builder.py` flattens it before
        # we get here (see the `Gemma4ForConditionalGeneration` branch),
        # but we defensively re-read the fields we need regardless.
        self._layer_types = list(getattr(config, "layer_types", []))
        self._head_dim_sliding = int(getattr(config, "head_dim", 256))
        self._head_dim_full = int(getattr(config, "global_head_dim", self._head_dim_sliding))
        self._hidden_size_per_layer_input = int(getattr(config, "hidden_size_per_layer_input", 0))
        self._vocab_size_per_layer_input = int(getattr(config, "vocab_size_per_layer_input", 0))
        self._num_kv_shared_layers = int(getattr(config, "num_kv_shared_layers", 0))
        self._use_double_wide_mlp = bool(getattr(config, "use_double_wide_mlp", False))

        rope_params = getattr(config, "rope_parameters", {}) or {}
        full_params = rope_params.get(self._FULL, {}) if isinstance(rope_params, dict) else {}
        slid_params = rope_params.get(self._SLIDING, {}) if isinstance(rope_params, dict) else {}
        self._rope_theta_full = float(full_params.get("rope_theta", 1_000_000.0))
        self._rope_theta_sliding = float(slid_params.get("rope_theta", getattr(config, "rope_theta", 10_000.0)))
        self._rope_partial_factor_full = float(full_params.get("partial_rotary_factor", 1.0))
        self._rope_type_full = str(full_params.get("rope_type", "default"))

        # Bail early if the caller asks us to actually emit any of the
        # not-yet-implemented pieces. We can still build config-only output
        # so downstream tooling can at least inspect a minimal genai_config.
        if not extra_options.get("config_only", False):
            self._raise_if_unsupported_requested(extra_options)

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # After Gemma3's __init__ runs, `self.rope_local_theta` has been
        # used to build the "local" (sliding) RoPE cache and the "global"
        # RoPE cache uses `config.rope_theta`. For Gemma 4 the split lives
        # inside `rope_parameters`; we re-point the attributes so inherited
        # code that reads them does the right thing for Gemma 4.
        self.rope_local_theta = self._rope_theta_sliding

    def _raise_if_unsupported_requested(self, extra_options):
        """Fail fast with an actionable message before we emit half a graph.

        A future PR will flip each of these to a supported path once the
        corresponding runtime change lands. Until then the only supported
        workflow against this builder is `config_only=true`, which is still
        useful for validating the dispatcher and genai_config shape.
        """
        missing = []
        if self._hidden_size_per_layer_input > 0:
            missing.append(
                "Per-Layer Embeddings (PLE): "
                f"hidden_size_per_layer_input={self._hidden_size_per_layer_input}. "
                "Requires a new `per_layer_inputs` side-channel input both in "
                "the ONNX graph and in the C++ model loader."
            )
        if self._num_kv_shared_layers > 0:
            missing.append(
                "KV cache sharing: "
                f"num_kv_shared_layers={self._num_kv_shared_layers}. "
                "Requires per-layer KV aliasing in kv_cache.cpp."
            )
        if self._head_dim_full != self._head_dim_sliding:
            missing.append(
                "Variable attention head dimension: "
                f"head_dim={self._head_dim_sliding} on sliding layers, "
                f"global_head_dim={self._head_dim_full} on full layers. "
                "This builder can emit correct per-layer shapes via "
                "`make_key_value_cache_shape`, but kv_cache.cpp allocates "
                "using a single `model.decoder.head_size` value from "
                "genai_config.json and must be taught to read per-layer shapes."
            )
        if missing:
            details = "\n  - ".join(missing)
            raise NotImplementedError(
                "Gemma 4 support in onnxruntime-genai is not yet complete. "
                "This builder scaffolding (tracked in issue #2062) currently "
                "supports `config_only=true` runs only. Blockers:\n  - "
                f"{details}\n"
                "Use `--extra_options config_only=true` to generate "
                "genai_config.json and processing files while runtime support "
                "is being implemented."
            )

    # -- Attention pattern -------------------------------------------------

    def is_local(self, layer_id):
        """Gemma 4 exposes the attention type explicitly in `layer_types`.

        Return True when the layer is sliding (local), False for full (global).
        Falls back to Gemma 3's every-6th rule if `layer_types` was not
        provided (shouldn't happen on published checkpoints but keeps tests
        with trimmed configs working).
        """
        if self._layer_types and layer_id < len(self._layer_types):
            return self._layer_types[layer_id] == self._SLIDING
        return super().is_local(layer_id)

    # -- KV cache shape ----------------------------------------------------

    def make_key_value_cache_shape(self, layer_id, shape):
        """Emit per-layer KV cache shapes for Gemma 4's variable head dim.

        Gemma 4 uses `head_dim=256` on sliding-attention layers and
        `global_head_dim=512` on full-attention layers. `shape` comes in
        as `[batch, num_kv_heads, past_sequence_length, head_size]` where
        `head_size` reflects the base (sliding) value.

        We replace the last dim with the layer-specific head dim. The
        trt-rtx 'sliding' token-name rewrite from the base implementation
        is preserved.
        """
        base_shape = super().make_key_value_cache_shape(layer_id, shape)
        if not self._layer_types or layer_id >= len(self._layer_types):
            return base_shape
        layer_head_dim = (
            self._head_dim_sliding
            if self._layer_types[layer_id] == self._SLIDING
            else self._head_dim_full
        )
        return [base_shape[0], base_shape[1], base_shape[2], layer_head_dim]

    # -- RoPE --------------------------------------------------------------

    def make_rotary_embedding_multi_cache(self):
        """Build the two RoPE caches Gemma 4 needs.

        Sliding layers: standard RoPE with theta=10_000, partial_rotary=1.0.
        Full layers:    'proportional' RoPE with theta=1_000_000 and
                        partial_rotary_factor=0.25.

        Gemma 3's implementation already emits two caches (global + local),
        but assumes both share the same head dim. Until the runtime learns
        per-layer head-dim awareness, we continue emitting both caches
        sized to the sliding head dim. This is why any config with
        `global_head_dim != head_dim` trips `_raise_if_unsupported_requested`
        above. When that gate lifts, this method should be extended to
        emit each cache at its own head-dim using a temporary
        `self.head_size` swap (same pattern as `make_attention` uses for
        `window_size`).
        """
        # Full-attention (global) cache. Gemma 3's own
        # `make_rotary_embedding_caches` override accepts explicit
        # cos/sin_cache_name kwargs and forwards to its super, so we
        # delegate through the normal MRO rather than skipping Gemma 3.
        self.rope_attrs["partial_rotary_factor"] = self._rope_partial_factor_full
        self.rope_attrs["theta"] = self._rope_theta_full
        self.cos_cache_global_name = "cos_cache_global"
        self.sin_cache_global_name = "sin_cache_global"
        super().make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_global_name,
            sin_cache_name=self.sin_cache_global_name,
        )

        # Sliding (local) cache
        self.rope_attrs["create_caches"] = True
        self.rope_attrs["partial_rotary_factor"] = 1.0
        self.rope_attrs["theta"] = self._rope_theta_sliding
        self.cos_cache_local_name = "cos_cache_local"
        self.sin_cache_local_name = "sin_cache_local"
        super().make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_local_name,
            sin_cache_name=self.sin_cache_local_name,
        )
