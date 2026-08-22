# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# ------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.

import copy
import glob
import os
from types import SimpleNamespace

import onnx_ir as ir
import torch

from .quant_config import resolve_dtype
from .qwen import Qwen35MoeTextModel, Qwen35TextModel


def mtp_dtypes_from_quant_config(quant_config):
    io_dtype = {
        "fp16": ir.DataType.FLOAT16,
        "bf16": ir.DataType.BFLOAT16,
        "fp32": ir.DataType.FLOAT,
    }[quant_config.io_dtype]
    weights = resolve_dtype(quant_config.weights.type)
    if weights.kind == "mx":
        raise ValueError(
            f"MTP dense weights.type={weights.name} is not supported; use int4/int8/none for dense weights "
            "and select mxfp4/nvfp4 independently through moe.type"
        )
    if weights.kind != "int":
        return io_dtype, io_dtype

    signed = weights.signed is not False and quant_config.weights.symmetric
    if weights.bits == 8:
        onnx_dtype = ir.DataType.INT8 if signed else ir.DataType.UINT8
    else:
        onnx_dtype = ir.DataType.INT4 if signed else ir.DataType.UINT4
    return io_dtype, onnx_dtype


class _LinearWeight:
    """Lightweight stand-in for ``nn.Linear`` exposing only ``weight`` (and a
    ``None`` bias), so ``make_matmul`` / ``make_lm_head`` can consume a raw weight
    tensor loaded directly from safetensors."""

    def __init__(self, weight, weight_scale=None, weight_scale_2=None, input_scale=None, bias=None):
        self.weight = weight
        self.weight_scale = weight_scale
        self.weight_scale_2 = weight_scale_2
        self.input_scale = input_scale
        self.bias = bias


class _RMSNormWeight:
    """Lightweight stand-in for an RMSNorm module exposing only ``weight``."""

    def __init__(self, weight):
        self.weight = weight


class Qwen35MtpHead(Qwen35MoeTextModel):
    """Qwen3.6 multi-token-prediction (MTP) self-speculative head builder.

    Emits a separate ``mtp.onnx`` graph that predicts token ``t_{i+2}`` from the
    main model's last hidden state ``h_i`` (post-final-norm) and the just-emitted
    token ``t_{i+1}``::

        h'_i   = fc(concat[ pre_fc_norm_embedding(embed(t_{i+1})),
                            pre_fc_norm_hidden(h_i) ])
        h''_i  = MtpDecoderLayer(h'_i)       # one full-attention + MoE layer
        logits = lm_head(mtp.norm(h''_i))

    The single MTP decoder layer is a ``full_attention`` GQA + MoE layer, so it
    reuses the parent's ``_make_full_attention`` / ``make_moe`` / mRoPE machinery
    unchanged. The ``mtp.*`` weights are loaded directly from the source
    safetensors because HF ``transformers`` discards them on ``from_pretrained``.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Mark as the MTP head so the parent does not recursively build another.
        self.is_mtp_head = True

        # The MTP head is a single full-attention decoder layer.
        config = copy.deepcopy(config)
        text_config = getattr(config, "text_config", config)
        text_config.num_hidden_layers = 1
        text_config.layer_types = ["full_attention"]
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]

        # Keep a copy of the (single-layer, full-attention) text config so the HF
        # ``Qwen3_5MoeDecoderLayer`` for the MTP layer can be instantiated later.
        self._mtp_layer_config = copy.deepcopy(text_config)
        self._mtp_layer_config.layer_types = ["full_attention"]
        self._mtp_layer_config.num_hidden_layers = 1

        # Force a single hidden layer regardless of the original config value.
        extra_options = copy.deepcopy(extra_options)
        extra_options["num_hidden_layers"] = 1
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # With no explicit override, a ModelOpt checkpoint keeps the original
        # per-tensor MTP formats instead of dequantizing and requantizing them.
        self._preserve_modelopt_mtp = self._should_preserve_modelopt_mtp(self.quant_type, extra_options)

        # The MTP head consumes the main model's last hidden state as an extra
        # input (alongside the standard input_ids / position_ids / KV cache).
        self.input_names["hidden_states"] = "hidden_states"
        self.input_types["hidden_states"] = self.io_dtype
        self.input_shapes["hidden_states"] = ["batch_size", "sequence_length", self.hidden_size]

    @staticmethod
    def _should_preserve_modelopt_mtp(quant_type, extra_options):
        return quant_type in {"modelopt", "compressed-tensors"} and "_quant_config" not in extra_options

    def make_model(self, input_path):
        # Inputs/outputs: standard decoder I/O plus the extra hidden_states input.
        self.make_inputs_and_outputs()

        if self.kv_cache_quant_type != "none":
            self.make_kv_cache_scale_initializers()

        # Load MTP-specific weights (discarded by HF ``from_pretrained``).
        self._load_mtp_weights(input_path)

        # Preprocessing: GQA mask (seqlens_k / total_seq_len) + mRoPE position_ids.
        self.make_preprocessing_nodes()

        # h'_i = fc(concat[pre_fc_norm_embedding(embed(t_{i+1})),
        #                   pre_fc_norm_hidden(h_i)])
        projected = self._make_mtp_input_projection()
        self.layernorm_attrs["root_input"] = projected
        self.layernorm_attrs["skip_input"] = projected
        self.layernorm_attrs["first_layernorm"] = True

        # One full-attention + MoE decoder layer (reuses parent machinery).
        self.make_layer(0, self._mtp_layer)

        # Final norm (mtp.norm) -> lm_head.
        self.make_layernorm(1, _RMSNormWeight(self._mtp_norm_weight), skip=True, simple=True, location="final_norm")
        # Capture the post-final-norm hidden BEFORE lm_head consumes it, so it can be
        # exported as the recurrent feedback output for multi-token speculation.
        mtp_norm_output = self.layernorm_attrs["output_0"]
        self.make_lm_head(self._lm_head)

        # A multi-token (num_speculative_tokens>1) loop feeds this back as the next chained
        # draft's `hidden_states` input, since the module is recurrent:
        # h_out = norm(layer(fc(embed, h_in))), same as vLLM's Qwen3.5 MTP. The name differs from
        # the `hidden_states` INPUT to avoid an ONNX name collision, and it is harmless at N=1
        # (genai's ExtraOutputs ignores unused outputs).
        hs_out = "hidden_states_out"
        self.make_node(
            "Identity",
            inputs=[mtp_norm_output],
            outputs=[hs_out],
            name="/model/mtp/hidden_states_out/Identity",
        )
        hs_val = self.make_value(hs_out, self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])
        self.model.graph.outputs.append(hs_val)

        self.make_postprocessing_nodes()

        # Free the large MTP layer module now that the graph is built.
        del self._mtp_layer

    def _load_mtp_weights(self, input_path):
        import safetensors.torch as safetensors_torch  # noqa: PLC0415

        model_dir = input_path if input_path and os.path.isdir(input_path) else self.model_name_or_path
        shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No .safetensors files found in '{model_dir}' for MTP weight loading.")

        mtp_state = {}
        embed_weight = None
        # The lm_head in a Model Optimizer NVFP4 checkpoint is stored packed:
        # "lm_head.weight" is uint8 [N, K/2] E2M1 codes, with a per-block E4M3
        # "lm_head.weight_scale" [N, K/16] and a per-tensor FP32 "lm_head.weight_scale_2".
        # It must be dequantized to a plain BF16 [N, K] weight the same way the main
        # model does (see ModeloptModel._dequant_linear); feeding the packed uint8
        # tensor straight into make_lm_head halves K (K/2 read as K) and corrupts the
        # LM head. Collect all three tensors and reconstruct below.
        lm_head_weight = None
        lm_head_weight_scale = None
        lm_head_weight_scale_2 = None
        # The embedding tensor name varies: plain text models use
        # "model.embed_tokens.weight" while the Qwen3.6 VL checkpoint nests it
        # under "model.language_model.embed_tokens.weight".
        embed_keys = {"model.embed_tokens.weight", "model.language_model.embed_tokens.weight"}
        for shard in shards:
            with safetensors_torch.safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    if key.startswith("mtp."):
                        mtp_state[key] = f.get_tensor(key)
                    elif key in embed_keys:
                        embed_weight = f.get_tensor(key)
                    elif key == "lm_head.weight":
                        lm_head_weight = f.get_tensor(key)
                    elif key == "lm_head.weight_scale":
                        lm_head_weight_scale = f.get_tensor(key)
                    elif key == "lm_head.weight_scale_2":
                        lm_head_weight_scale_2 = f.get_tensor(key)
                    elif key == "lm_head.weight_global_scale":
                        # compressed-tensors stores the reciprocal of the NVFP4 global scale.
                        lm_head_weight_scale_2 = torch.reciprocal(f.get_tensor(key))

        if not mtp_state:
            raise ValueError(f"No 'mtp.*' weights found in '{model_dir}'; this model has no MTP head.")
        if embed_weight is None:
            raise ValueError(
                "Could not find the token embedding weight "
                "('model.embed_tokens.weight' or 'model.language_model.embed_tokens.weight') "
                "for the MTP head embedding."
            )
        if lm_head_weight is None:
            raise ValueError("Could not find 'lm_head.weight' for the MTP head LM head.")

        if self._preserve_modelopt_mtp:
            modelopt_state = dict(mtp_state)
            modelopt_state["lm_head.weight"] = lm_head_weight
            if lm_head_weight_scale is not None:
                modelopt_state["lm_head.weight_scale"] = lm_head_weight_scale
            if lm_head_weight_scale_2 is not None:
                modelopt_state["lm_head.weight_scale_2"] = lm_head_weight_scale_2
            self._load_native_modelopt_mtp_weights(modelopt_state, embed_weight)
            return

        # An explicit precision override requantizes from plain BF16 tensors.
        # Dequantize any native ModelOpt linears before loading the HF layer.
        mtp_state = self._dequantize_modelopt_state(mtp_state)
        lm_head_weight = self._dequantize_modelopt_weight(
            lm_head_weight,
            lm_head_weight_scale,
            lm_head_weight_scale_2,
            "lm_head.weight",
        )

        self._embed_weight = embed_weight
        self._lm_head = _LinearWeight(lm_head_weight)
        self._fc = _LinearWeight(mtp_state["mtp.fc.weight"])
        self._pre_fc_norm_embedding_weight = mtp_state["mtp.pre_fc_norm_embedding.weight"]
        self._pre_fc_norm_hidden_weight = mtp_state["mtp.pre_fc_norm_hidden.weight"]
        self._mtp_norm_weight = mtp_state["mtp.norm.weight"]

        # Build the single MTP decoder layer (full-attention + MoE) and load its
        # weights from the ``mtp.layers.0.*`` entries.
        mtp_layer = self._make_mtp_decoder_layer()
        layer_state = {
            key[len("mtp.layers.0.") :]: value for key, value in mtp_state.items() if key.startswith("mtp.layers.0.")
        }
        missing, unexpected = mtp_layer.load_state_dict(layer_state, strict=False)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise ValueError("Invalid MTP decoder-layer weights: " + ", ".join(details))
        mtp_layer.eval()
        self._mtp_layer = mtp_layer

    def _make_mtp_decoder_layer(self):
        try:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
                Qwen3_5MoeDecoderLayer,
            )
        except ImportError as exc:
            raise ImportError(
                "Building the Qwen3.6 MTP head requires the 'qwen3_5_moe' modeling code in transformers."
            ) from exc
        return Qwen3_5MoeDecoderLayer(self._mtp_layer_config, layer_idx=0)

    @staticmethod
    def _dequantize_modelopt_weight(weight, weight_scale, weight_scale_2, name):
        if weight_scale_2 is not None:
            if weight_scale is None:
                raise ValueError(f"ModelOpt NVFP4 tensor '{name}' is missing its weight_scale.")
            if weight_scale_2.numel() != 1:
                raise ValueError(f"ModelOpt NVFP4 tensor '{name}' weight_scale_2 must be a scalar.")
            if weight_scale.dtype == torch.uint8:
                weight_scale = weight_scale.view(torch.float8_e4m3fn)
            elif weight_scale.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    f"ModelOpt NVFP4 tensor '{name}' weight_scale must be float8_e4m3fn or uint8, "
                    f"got {weight_scale.dtype}."
                )
            low = weight.to(torch.uint8) & 0x0F
            high = weight.to(torch.uint8) >> 4
            codes = torch.stack((low, high), dim=-1).reshape(weight.shape[0], -1).long()
            magnitudes = torch.tensor(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                dtype=torch.float32,
            )[codes & 0x7]
            values = torch.where((codes & 0x8) > 0, -magnitudes, magnitudes)
            block_scales = weight_scale.float()
            if (
                block_scales.ndim != 2
                or block_scales.shape[0] != values.shape[0]
                or block_scales.shape[1] == 0
                or values.shape[1] % block_scales.shape[1] != 0
            ):
                raise ValueError(
                    f"ModelOpt NVFP4 tensor '{name}' has incompatible weight/scale shapes "
                    f"{tuple(weight.shape)} and {tuple(weight_scale.shape)}."
                )
            block_scales = block_scales.repeat_interleave(values.shape[1] // block_scales.shape[1], dim=1)
            return (values * block_scales * float(weight_scale_2.float().item())).to(torch.bfloat16)
        if weight.dtype == torch.float8_e4m3fn:
            if weight_scale is None or weight_scale.numel() not in (1, weight.shape[0]):
                raise ValueError(
                    f"ModelOpt FP8 tensor '{name}' must have a scalar or per-channel weight_scale, "
                    f"got {None if weight_scale is None else tuple(weight_scale.shape)}."
                )
            return (weight.float() * weight_scale.float().reshape(-1, 1)).to(torch.bfloat16)
        return weight

    @classmethod
    def _dequantize_modelopt_state(cls, state):
        result = {}
        metadata_suffixes = (".weight_scale", ".weight_scale_2", ".input_scale")
        for name, tensor in state.items():
            if name.endswith(metadata_suffixes):
                continue
            if not name.endswith(".weight"):
                result[name] = tensor
                continue
            prefix = name.removesuffix(".weight")
            result[name] = cls._dequantize_modelopt_weight(
                tensor,
                state.get(f"{prefix}.weight_scale"),
                state.get(f"{prefix}.weight_scale_2"),
                name,
            )
        return result

    def _load_native_modelopt_mtp_weights(self, state, embed_weight):
        """Build lightweight MTP modules while retaining ModelOpt scale metadata."""

        def required(name):
            tensor = state.get(name)
            if tensor is None:
                raise ValueError(f"ModelOpt MTP checkpoint is missing '{name}'.")
            return tensor

        def linear(prefix):
            return _LinearWeight(
                required(f"{prefix}.weight"),
                weight_scale=state.get(f"{prefix}.weight_scale"),
                weight_scale_2=state.get(f"{prefix}.weight_scale_2"),
                input_scale=state.get(f"{prefix}.input_scale"),
                bias=state.get(f"{prefix}.bias"),
            )

        def norm(name):
            return _RMSNormWeight(required(name))

        layer_prefix = "mtp.layers.0"
        attention_prefix = f"{layer_prefix}.self_attn"
        mlp_prefix = f"{layer_prefix}.mlp"
        attention = SimpleNamespace(
            q_proj=linear(f"{attention_prefix}.q_proj"),
            k_proj=linear(f"{attention_prefix}.k_proj"),
            v_proj=linear(f"{attention_prefix}.v_proj"),
            o_proj=linear(f"{attention_prefix}.o_proj"),
            q_norm=norm(f"{attention_prefix}.q_norm.weight"),
            k_norm=norm(f"{attention_prefix}.k_norm.weight"),
        )
        mlp = self._mtp_mlp_modules(mlp_prefix, linear)
        self._mtp_layer = SimpleNamespace(
            input_layernorm=norm(f"{layer_prefix}.input_layernorm.weight"),
            post_attention_layernorm=norm(f"{layer_prefix}.post_attention_layernorm.weight"),
            self_attn=attention,
            linear_attn=None,
            mlp=mlp,
        )
        self._embed_weight = embed_weight
        self._lm_head = linear("lm_head")
        self._fc = linear("mtp.fc")
        self._pre_fc_norm_embedding_weight = required("mtp.pre_fc_norm_embedding.weight")
        self._pre_fc_norm_hidden_weight = required("mtp.pre_fc_norm_hidden.weight")
        self._mtp_norm_weight = required("mtp.norm.weight")

    def _mtp_mlp_modules(self, mlp_prefix, linear):
        experts = []
        for expert_id in range(self.moe_attrs["num_experts"]):
            expert_prefix = f"{mlp_prefix}.experts.{expert_id}"
            experts.append(
                SimpleNamespace(
                    gate_proj=linear(f"{expert_prefix}.gate_proj"),
                    up_proj=linear(f"{expert_prefix}.up_proj"),
                    down_proj=linear(f"{expert_prefix}.down_proj"),
                )
            )
        shared_prefix = f"{mlp_prefix}.shared_expert"
        return SimpleNamespace(
            gate=linear(f"{mlp_prefix}.gate"),
            experts=experts,
            shared_expert=SimpleNamespace(
                gate_proj=linear(f"{shared_prefix}.gate_proj"),
                up_proj=linear(f"{shared_prefix}.up_proj"),
                down_proj=linear(f"{shared_prefix}.down_proj"),
            ),
            shared_expert_gate=linear(f"{mlp_prefix}.shared_expert_gate"),
        )

    def _make_offset_rmsnorm(self, name, root_input, weight_tensor):
        """Build a non-skip SimplifiedLayerNormalization with the ``(1 + weight)``
        offset (used by the two pre-fc RMSNorms in the MTP head)."""
        weight_name = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(weight_tensor + self.layernorm_attrs["add_offset"], weight_name, to=self.io_dtype)
        output = f"{name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, weight_name],
            outputs=[output],
            name=name,
            epsilon=self.layernorm_attrs["epsilon"],
            axis=-1,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])
        return output

    def _make_mtp_input_projection(self):
        """Build ``fc(concat[pre_fc_norm_embedding(embed(input_ids)),
        pre_fc_norm_hidden(hidden_states)])`` and return its output name."""
        basename = "/model/mtp"

        # embed(input_ids) -> [B, S, H]
        embed_weight = "model.embed_tokens.weight"
        self.make_initializer(self._embed_weight, embed_weight, to=self.io_dtype)
        embed_gather = f"{basename}/embed_tokens/Gather"
        embed_out = f"{embed_gather}/output_0"
        self.make_node(
            "Gather",
            inputs=[embed_weight, self.input_names["input_ids"]],
            outputs=[embed_out],
            name=embed_gather,
        )
        self.make_value(embed_out, self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])

        # pre_fc_norm_embedding(embed) and pre_fc_norm_hidden(hidden_states)
        e_norm = self._make_offset_rmsnorm(
            f"{basename}/pre_fc_norm_embedding", embed_out, self._pre_fc_norm_embedding_weight
        )
        h_norm = self._make_offset_rmsnorm(
            f"{basename}/pre_fc_norm_hidden", self.input_names["hidden_states"], self._pre_fc_norm_hidden_weight
        )

        # concat([e_norm, h_norm], axis=-1) -> [B, S, 2H]
        concat_name = f"{basename}/fc/Concat"
        self.make_concat(
            concat_name,
            [e_norm, h_norm],
            self.io_dtype,
            ["batch_size", "sequence_length", 2 * self.hidden_size],
            axis=-1,
        )

        # fc: [2H -> H]
        fc_name = self.make_matmul(self._fc, f"{basename}/fc/MatMul", f"{concat_name}/output_0")
        return f"{fc_name}/output_0"


class Qwen35DenseMtpHead(Qwen35MtpHead):
    """Dense Qwen3.5/Qwen3.8 MTP head: one full-attention decoder layer with a dense MLP."""

    _uses_moe_layers = False

    def _get_model_type(self, config):
        return Qwen35TextModel._get_model_type(self, config)

    def make_layer(self, layer_id, layer):
        return Qwen35TextModel.make_layer(self, layer_id, layer)

    def _make_mtp_decoder_layer(self):
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Building the dense Qwen3.8 MTP head requires the 'qwen3_5' modeling code in transformers."
            ) from exc
        return Qwen3_5DecoderLayer(self._mtp_layer_config, layer_idx=0)

    def _mtp_mlp_modules(self, mlp_prefix, linear):
        return SimpleNamespace(
            gate_proj=linear(f"{mlp_prefix}.gate_proj"),
            up_proj=linear(f"{mlp_prefix}.up_proj"),
            down_proj=linear(f"{mlp_prefix}.down_proj"),
        )
