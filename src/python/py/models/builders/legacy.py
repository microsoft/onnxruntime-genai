# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# -------------------------------------------------------------------------
"""Builders for decoder models that predate the Llama-style module layout."""

import json
from pathlib import Path

import onnx_ir as ir
import torch

from .base import Model

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover - older transformers packages
    Conv1D = ()


LAYER_TYPES = {
    "GPT2Block",
    "GPTNeoBlock",
    "GPTBigCodeBlock",
    "OPTDecoderLayer",
    "BloomBlock",
    "GPTNeoXLayer",
    "CodeGenBlock",
}


def _config_value(config, name):
    try:
        return getattr(config, name)
    except AttributeError:
        return None


def _prepare_config(config) -> None:
    """Fill the fields Model.__init__ reads on older decoder configs."""
    if not _config_value(config, "hidden_act"):
        config.hidden_act = _config_value(config, "activation_function") or "gelu"
    if not _config_value(config, "intermediate_size"):
        config.intermediate_size = (
            _config_value(config, "n_inner")
            or _config_value(config, "ffn_dim")
            or 4 * config.hidden_size
        )
    if config.__dict__.get("num_key_value_heads", 0) is None:
        del config.__dict__["num_key_value_heads"]
    # Bloom has no position table. A missing length makes Model.__init__ raise.
    if _config_value(config, "max_position_embeddings") is None and _config_value(config, "seq_length") is None:
        length = _config_value(config, "n_positions") or _config_value(config, "max_sequence_length") or 2048
        try:
            config.max_position_embeddings = int(length)
        except (AttributeError, TypeError, ValueError):
            object.__setattr__(config, "max_position_embeddings", int(length))


def _json_field(config, name, default):
    value = getattr(config, name, None)
    if value is not None:
        return value
    path = Path(getattr(config, "_name_or_path", "")) / "config.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")).get(name, default)
    return default


def _conv1d_to_linear(module) -> torch.nn.Linear:
    """Conv1D stores weight as (in, out). Linear stores (out, in)."""
    if isinstance(module, torch.nn.Linear):
        return module
    weight = module.weight.detach()
    bias = None if module.bias is None else module.bias.detach()
    if isinstance(module, Conv1D) or (hasattr(module, "nf") and weight.shape[1] == module.nf):
        out_features, in_features = weight.shape[1], weight.shape[0]
        linear = torch.nn.Linear(in_features, out_features, bias=bias is not None)
        linear.weight = torch.nn.Parameter(weight.transpose(0, 1).contiguous(), requires_grad=False)
    else:
        out_features, in_features = weight.shape
        linear = torch.nn.Linear(in_features, out_features, bias=bias is not None)
        linear.weight = torch.nn.Parameter(weight.contiguous(), requires_grad=False)
    if bias is not None:
        linear.bias = torch.nn.Parameter(bias.contiguous(), requires_grad=False)
    else:
        linear.bias = None
    return linear


def _reorder_codegen_qkv(qkv, num_heads: int, head_size: int) -> None:
    """CodeGen stores each group as Q, V, K. The builder expects Q|K|V."""
    groups = 4
    local = head_size * num_heads // groups
    weight = qkv.weight.detach().view(groups, 3, local, qkv.weight.shape[1])
    query, value, key = weight[:, 0], weight[:, 1], weight[:, 2]
    hidden = num_heads * head_size
    qkv.weight = torch.nn.Parameter(
        torch.cat([query, key, value], dim=0).reshape(3 * hidden, -1).contiguous(),
        requires_grad=False,
    )


def _reorder_bloom_qkv(qkv, num_heads: int) -> None:
    """Bloom interleaves Q, K, V inside each head. The builder expects Q|K|V."""
    hidden = qkv.weight.shape[0] // 3
    head = hidden // num_heads
    weight = qkv.weight.detach().view(num_heads, 3, head, qkv.weight.shape[1])
    qkv.weight = torch.nn.Parameter(
        torch.cat([weight[:, 0], weight[:, 1], weight[:, 2]], dim=0).reshape(3 * hidden, -1).contiguous(),
        requires_grad=False,
    )
    if qkv.bias is not None:
        bias = qkv.bias.detach().view(num_heads, 3, head)
        qkv.bias = torch.nn.Parameter(
            torch.cat([bias[:, 0], bias[:, 1], bias[:, 2]], dim=0).reshape(-1).contiguous(),
            requires_grad=False,
        )


def _alias_fc(mlp) -> None:
    """Point the FC1/FC2 MLP builder at this module's first and second linear."""
    if hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
        return
    first = getattr(mlp, "c_fc", None) or getattr(mlp, "fc_in", None) or getattr(mlp, "dense_h_to_4h", None)
    second = getattr(mlp, "c_proj", None) or getattr(mlp, "fc_out", None) or getattr(mlp, "dense_4h_to_h", None)
    mlp.fc1 = _conv1d_to_linear(first)
    mlp.fc2 = _conv1d_to_linear(second)
    # dense_h_to_4h is a packed gate/up projection for Llama-style MLPs.
    # These models use a single GeLU/ReLU expansion, so hide that name.
    modules = getattr(mlp, "_modules", None)
    if modules is not None:
        modules.pop("dense_h_to_4h", None)


class LegacyModel(Model):
    """Shared wiring for pre-Llama decoders: standard LayerNorm and an FC MLP."""

    FINAL_NORM = ("transformer", "ln_f")
    LAYER_TYPES = LAYER_TYPES

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        _prepare_config(config)
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["simple"] = False
        for name in ("layer_norm_epsilon", "layer_norm_eps"):
            epsilon = getattr(config, name, None)
            if epsilon:
                self.layernorm_attrs["epsilon"] = float(epsilon)
                break
        self.mlp_attrs["use_proj"] = False
        self.mlp_attrs["use_fc"] = True
        self.position_weight = None
        self.position_offset = 0
        self.embed_layernorm = None
        self.project_in = None
        self.project_out = None
        self.project_out_done = False

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        saved = self.model_type
        # The runtime only accepts a fixed set of type names. These graphs use
        # the generic decoder session, which still feeds position_ids.
        self.model_type = "decoderFor"
        try:
            return super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)
        finally:
            self.model_type = saved

    def _disable_rope(self) -> None:
        self.attention_attrs["rope"] = False
        self.attention_attrs["use_rope_in_attn"] = False
        self.rope_attrs["rotary_embedding_dim"] = 0
        self.rope_attrs["create_caches"] = False
        self.input_names["position_ids"] = "position_ids"

    def is_layer(self, module):
        return module.__class__.__name__ in self.LAYER_TYPES or super().is_layer(module)

    def has_final_norm(self, module, orig_model):
        model = orig_model.base_model.model if orig_model.__class__.__name__.startswith("Peft") else orig_model
        current = model
        for name in self.FINAL_NORM:
            current = getattr(current, name, None)
            if current is None:
                return super().has_final_norm(module, orig_model)
        return module == current

    def make_embedding(self, embedding):
        original_hidden = self.hidden_size
        if self.project_in is not None:
            self.hidden_size = int(self.project_in.in_features)
        super().make_embedding(embedding)
        self.hidden_size = original_hidden
        if self.project_in is not None:
            produced = self.make_matmul(
                self.project_in,
                "/model/embed_tokens/project_in/MatMul",
                self.layernorm_attrs["root_input"],
            )
            projected = f"{produced}/output_0"
            self.layernorm_attrs["root_input"] = projected
            self.layernorm_attrs["skip_input"] = projected
        if self.embed_layernorm is not None:
            self.make_layernorm(0, self.embed_layernorm, skip=False, simple=False, location="embed")
            self.layernorm_attrs["root_input"] = self.layernorm_attrs["output_0"]
            self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        if self.position_weight is None:
            return
        # OPT stores two padding rows in front of the table and adds the offset
        # at gather time. A scalar Add there becomes an extra fusion input and
        # the EP rejects the subgraph, so drop those rows from the table instead.
        weight = self.position_weight
        if self.position_offset:
            weight = weight[self.position_offset :].contiguous()
        weight_name = "model.embed_positions.weight"
        self.make_initializer(weight, weight_name, to=self.io_dtype)
        position_ids = self.input_names["position_ids"]
        gather_name = "/model/embed_positions/Gather"
        gather_output = f"{gather_name}/output_0"
        self.make_node("Gather", inputs=[weight_name, position_ids], outputs=[gather_output], name=gather_name)
        self.make_value(gather_output, self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])
        add_name = "/model/embed_tokens/AddPositions"
        self.make_add(
            add_name,
            [self.layernorm_attrs["root_input"], gather_output],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )
        added = f"{add_name}/output_0"
        self.layernorm_attrs["root_input"] = added
        self.layernorm_attrs["skip_input"] = added


class GPT2Model(LegacyModel):
    """GPT-2, DistilGPT-2, and GPTBigCode. Absolute positions, no rotary embeddings."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self._disable_rope()

    def load_weights(self, input_path):
        model = super().load_weights(input_path)
        transformer = model.transformer
        self.position_weight = transformer.wpe.weight.detach()
        for block in transformer.h:
            attention = block.attn
            attention.query_key_value = _conv1d_to_linear(attention.c_attn)
            attention.o_proj = _conv1d_to_linear(attention.c_proj)
            _alias_fc(block.mlp)
            block.input_layernorm = block.ln_1
            block.post_attention_layernorm = block.ln_2
            block.self_attn = attention
        return model


class GPTNeoModel(LegacyModel):
    """GPT-Neo. Local layers use the configured attention window."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self._disable_rope()
        # GPT-Neo softmaxes the raw query-key product. The shared attention
        # path divides by sqrt(head_size), which this model does not.
        self.attention_attrs["scale"] = 1.0
        pattern, repeats = config.attention_types[0]
        self.attention_pattern = list(pattern) * int(repeats)
        self.local_window = int(config.window_size)

    def load_weights(self, input_path):
        model = super().load_weights(input_path)
        transformer = model.transformer
        self.position_weight = transformer.wpe.weight.detach()
        for block in transformer.h:
            block.input_layernorm = block.ln_1
            block.post_attention_layernorm = block.ln_2
            block.self_attn = block.attn.attention
            _alias_fc(block.mlp)
        return model

    def make_layer(self, layer_id, layer):
        kind = self.attention_pattern[layer_id]
        self.window_size = self.local_window if kind == "local" else -1
        super().make_layer(layer_id, layer)


class OPTModel(LegacyModel):
    """OPT. Learned positions are stored with the two padding rows OPT adds."""

    FINAL_NORM = ("model", "decoder", "final_layer_norm")

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self._disable_rope()
        self.position_offset = 2
        # 125M applies LayerNorm before each branch. 350M applies it after the residual add.
        self.do_layer_norm_before = bool(getattr(config, "do_layer_norm_before", True))

    def load_weights(self, input_path):
        model = super().load_weights(input_path)
        decoder = model.model.decoder
        self.position_weight = decoder.embed_positions.weight.detach()
        self.project_in = decoder.project_in
        self.project_out = decoder.project_out
        for layer in decoder.layers:
            layer.input_layernorm = layer.self_attn_layer_norm
            layer.post_attention_layernorm = layer.final_layer_norm
            layer.mlp = torch.nn.Module()
            layer.mlp.fc1 = layer.fc1
            layer.mlp.fc2 = layer.fc2
        return model

    def make_layer(self, layer_id, layer):
        if self.do_layer_norm_before:
            return super().make_layer(layer_id, layer)
        residual = self.layernorm_attrs["root_input"]
        self.make_attention(layer_id, layer.self_attn, root_input=residual)
        self.layernorm_attrs["root_input"] = residual
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=True,
            simple=False,
            location="input",
        )
        normed = self.layernorm_attrs["output_0"]
        self.make_mlp(layer_id, layer.mlp, root_input=normed)
        self.layernorm_attrs["root_input"] = normed
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=True,
            simple=False,
            location="post_attention",
        )
        normed = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["root_input"] = normed
        self.layernorm_attrs["skip_input"] = normed
        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    def has_lm_head(self, module):
        matched = super().has_lm_head(module)
        if matched and self.project_out is not None and not self.project_out_done:
            produced = self.make_matmul(
                self.project_out,
                "/model/project_out/MatMul",
                self.layernorm_attrs["output_0"],
            )
            self.layernorm_attrs["output_0"] = f"{produced}/output_0"
            self.project_out_done = True
        return matched


class BloomModel(LegacyModel):
    """Bloom. ALiBi is not an input of this GroupQueryAttention, so scores are unscaled."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self._disable_rope()

    def load_weights(self, input_path):
        model = super().load_weights(input_path)
        transformer = model.transformer
        self.embed_layernorm = transformer.word_embeddings_layernorm
        for block in transformer.h:
            _reorder_bloom_qkv(block.self_attention.query_key_value, self.num_attn_heads)
            block.self_attn = block.self_attention
            _alias_fc(block.mlp)
        return model


class ParallelLegacyModel(LegacyModel):
    """x + Attn(LN(x)) + MLP(LN(x)), with one or two layer norms."""

    two_norms = False

    def make_layer(self, layer_id, layer):
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=False,
            location="input",
        )
        residual = (
            self.layernorm_attrs["output_3"]
            if not self.layernorm_attrs["first_layernorm"]
            else self.layernorm_attrs["root_input"]
        )
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
        attention_out = self.layernorm_attrs["skip_input"]
        if self.two_norms:
            self.layernorm_attrs["root_input"] = residual
            self.make_layernorm(
                layer_id,
                layer.post_attention_layernorm,
                skip=False,
                simple=False,
                location="post_attention",
            )
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])
        mlp_out = self.layernorm_attrs["skip_input"]
        added = f"/model/layers.{layer_id}/residual_add/Add"
        self.make_add(
            added,
            [attention_out, mlp_out],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )
        self.layernorm_attrs["root_input"] = residual
        self.layernorm_attrs["skip_input"] = f"{added}/output_0"
        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True


class GPTNeoXModel(ParallelLegacyModel):
    """Pythia. Partial rotary embeddings and two layer norms on the same residual."""

    FINAL_NORM = ("gpt_neox", "final_layer_norm")
    two_norms = True

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        rotary_pct = _json_field(config, "rotary_pct", 0.25)
        if not getattr(config, "partial_rotary_factor", None):
            config.partial_rotary_factor = rotary_pct
        if not getattr(config, "rope_theta", None):
            config.rope_theta = _json_field(config, "rotary_emb_base", 10000)
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def load_weights(self, input_path):
        model = super().load_weights(input_path)
        for layer in model.gpt_neox.layers:
            _reorder_bloom_qkv(layer.attention.query_key_value, self.num_attn_heads)
            layer.self_attn = layer.attention
            _alias_fc(layer.mlp)
        return model


class CodeGenModel(ParallelLegacyModel):
    """CodeGen-mono. One layer norm feeds both branches, then a partial rotary width."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        head_size = config.hidden_size // config.num_attention_heads
        rotary_dim = _json_field(config, "rotary_dim", 0) or 0
        if rotary_dim and not getattr(config, "partial_rotary_factor", None):
            config.partial_rotary_factor = rotary_dim / head_size
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # CodeGen rotates adjacent pairs, the same layout as GPT-J.
        self.rope_attrs["interleaved"] = 1

    def load_weights(self, input_path):
        model = super().load_weights(input_path)
        for block in model.transformer.h:
            _reorder_codegen_qkv(block.attn.qkv_proj, self.num_attn_heads, self.head_size)
            block.input_layernorm = block.ln_1
            block.self_attn = block.attn
            _alias_fc(block.mlp)
        return model
