# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy
import json
import os

import numpy as np
import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import TorchTensor, to_torch_dtype
from tqdm import tqdm

from .base import Model, parse_hf_token


class MistralModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class MistralNeMoModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


def _dequantize_fp8_weights(model):
    """Dequantize float8_e4m3fn weights in place using per-tensor weight_scale_inv.

    The official Ministral-3B-Instruct-2512 model stores linear layer weights
    as float8_e4m3fn with a per-tensor inverse scale factor (weight_scale_inv).
    Standard PyTorch matmul cannot consume float8 parameters, so we eagerly
    cast them back to float32 before building the ONNX graph.

    Dequantization formula: weight_fp32 = weight_fp8.float() * weight_scale_inv
    """
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        return  # PyTorch version does not support FP8; nothing to do
    for module in model.modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if module.weight.dtype != fp8_dtype:
            continue
        if not hasattr(module, "weight_scale_inv"):
            continue
        scale_inv = module.weight_scale_inv
        with torch.no_grad():
            dequantized = module.weight.float() * scale_inv.float()
            module.weight = torch.nn.Parameter(dequantized, requires_grad=False)


class Ministral3TextModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def load_weights(self, input_path):
        # Mistral3ForConditionalGeneration (model_type="mistral3") is not
        # registered with AutoModelForCausalLM.  Load the full multimodal
        # model directly; make_model will find the embedded language-model
        # sub-modules (embed_tokens, MistralDecoderLayer, norm, lm_head)
        # via the standard module iteration already present in base.Model.
        if "ConditionalGeneration" in self.model_type:
            from transformers import Mistral3ForConditionalGeneration as HF_Mistral3VL

            model = HF_Mistral3VL.from_pretrained(
                self.model_name_or_path, cache_dir=self.cache_dir, token=self.hf_token, trust_remote_code=self.hf_remote
            )
        else:
            model = super().load_weights(input_path)
        # Dequantize FP8 weights if present (official Ministral-3B model uses
        # float8_e4m3fn with per-tensor weight_scale_inv).
        _dequantize_fp8_weights(model)
        return model

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """`minitral3` is not supported as an architecture, let's replace with `mistral`."""
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        genai_config["model"]["type"] = "mistral"

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)


class Ministral3VisionEncoderModel:
    """Direct ``onnx_ir`` graph builder for the Pixtral vision encoder + multimodal projector.

    Builds the ONNX graph manually (analogous to other model builders in this
    codebase) rather than going through :func:`torch.onnx.export`.

    For a fixed image size (``vision_config.image_size`` x same):

    * Input:  ``pixel_values`` [1, num_channels, image_size, image_size]
    * Output: ``image_features`` [num_merged_patches, text_hidden_size]

    The 2-D RoPE (cos/sin) is pre-computed at graph-build time and stored as
    constant initialisers, removing the need for runtime position-id
    computation.

    The model is designed for batch_size = 1 (a single image per forward
    pass), matching how onnxruntime-genai processes multimodal inputs: each
    image is encoded independently before being concatenated into
    ``inputs_embeds``.
    """

    FILENAME = "vision_encoder.onnx"

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.config = config
        self.vision_config = config.vision_config
        self.io_dtype = ir.DataType(io_dtype)
        self.onnx_dtype = ir.DataType(onnx_dtype)
        self.cache_dir = cache_dir
        self.extra_options = extra_options
        self.filename = self.FILENAME
        self.hf_token = parse_hf_token(extra_options.get("hf_token", "true"))
        self.hf_remote = extra_options.get("hf_remote", True)
        self.model_name_or_path = config._name_or_path

        vc = self.vision_config
        self.image_size = vc.image_size
        self.patch_size = vc.patch_size
        self.num_channels = vc.num_channels
        self.n_patches_per_side = self.image_size // self.patch_size
        self.n_patches = self.n_patches_per_side**2
        self.vis_hidden_size = vc.hidden_size
        self.vis_intermediate_size = vc.intermediate_size
        self.vis_num_heads = vc.num_attention_heads
        self.vis_head_dim = (
            vc.head_dim if hasattr(vc, "head_dim") and vc.head_dim else vc.hidden_size // vc.num_attention_heads
        )
        self.vis_num_layers = vc.num_hidden_layers
        self.vis_rms_norm_eps = 1e-5  # hardcoded in PixtralRMSNorm
        self.vis_attn_scale = float(self.vis_head_dim**-0.5)

        tc = config.text_config
        self.spatial_merge_size = config.spatial_merge_size
        self.n_merged_patches = self.n_patches // (self.spatial_merge_size**2)
        self.text_hidden_size = tc.hidden_size
        self.vision_feature_layer = config.vision_feature_layer
        self.projector_hidden_act = config.projector_hidden_act

        # onnx_ir graph state
        self.graph = ir.Graph(
            inputs=(), outputs=(), nodes=(), opset_imports={"": 21, "com.microsoft": 1}, name="pixtral_vision_encoder"
        )
        self.onnx_model = ir.Model(self.graph, ir_version=10, producer_name="onnxruntime-genai")
        self.values = {}
        self.node_names = set()

    # ------------------------------------------------------------------ #
    #  Low-level onnx_ir primitives                                       #
    # ------------------------------------------------------------------ #

    def _val(self, name, dtype=None, shape=None):
        """Obtain or create an IR value by name."""
        if name == "":
            return ir.Value(name="")
        v = self.values.setdefault(name, ir.Value(name=name))
        if dtype is not None:
            v.dtype = ir.DataType(dtype)
        if shape is not None:
            v.shape = ir.Shape(shape)
        return v

    def _node(self, op_type, inputs, outputs, name, domain="", **kwargs):
        """Append an ONNX node to the graph (no-op if name already exists)."""
        if name in self.node_names:
            return
        in_vals = [self._val(n) for n in inputs]
        out_vals = [self._val(n) for n in outputs]
        node = ir.node(op_type, inputs=in_vals, attributes=kwargs, domain=domain, outputs=out_vals, name=name)
        self.graph.append(node)
        self.node_names.add(name)

    def _init(self, tensor, name, to=None):
        """Register a weight tensor as a graph initializer."""
        if to is not None:

            def _lazy():
                return TorchTensor(tensor.to(to_torch_dtype(to)), name=name)

            ir_tensor = ir.LazyTensor(_lazy, dtype=to, shape=ir.Shape(tensor.shape), name=name)
        elif isinstance(tensor, torch.Tensor):
            ir_tensor = TorchTensor(tensor, name=name)
        else:
            # numpy array or other
            ir_tensor = ir.tensor(tensor, name=name)
        value = self._val(name, ir_tensor.dtype, ir_tensor.shape)
        value.const_value = ir_tensor
        self.graph.register_initializer(value)

    def _const_tensor(self, np_data, name):
        """Emit a small constant as an inline ONNX ``Constant`` node.

        Unlike ``_init`` (which registers large weight tensors as initialisers
        and allows them to be offloaded to external data), this method always
        keeps the tensor value inline inside the ONNX graph node.  This is
        required for shape constants consumed by ``Reshape``: ORT's shape
        inference cannot read external-data tensors, so those constants must
        remain embedded in the model file.
        """
        ir_t = ir.tensor(np_data, name=name)
        node_name = f"{name}/Constant"
        # Constant node: no inputs, one output carrying the value.
        self._node("Constant", inputs=[], outputs=[name], name=node_name, value=ir_t)
        self._val(name, ir_t.dtype, ir_t.shape)

    # ------------------------------------------------------------------ #
    #  Mid-level graph-construction helpers                               #
    # ------------------------------------------------------------------ #

    def _rms_norm(self, name, root_input, weight_tensor, weight_name, shape):
        """SimplifiedLayerNormalization (PixtralRMSNorm)."""
        self._init(weight_tensor, weight_name, to=self.io_dtype)
        output = f"{name}/output_0"
        self._node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, weight_name],
            outputs=[output],
            name=name,
            axis=-1,
            epsilon=self.vis_rms_norm_eps,
            stash_type=1,
        )
        self._val(output, self.io_dtype, shape=shape)
        return output

    def _matmul(self, name, root_input, weight_tensor, weight_name, out_shape, bias_tensor=None, bias_name=None):
        """MatMul (weight stored transposed as [in, out]) with optional Add bias."""
        self._init(weight_tensor.T.contiguous(), weight_name, to=self.io_dtype)
        mm_out = f"{name}/output_0"
        self._node("MatMul", inputs=[root_input, weight_name], outputs=[mm_out], name=name)
        self._val(mm_out, self.io_dtype, shape=out_shape)
        if bias_tensor is not None and bias_name is not None:
            if torch.count_nonzero(bias_tensor) > 0:
                self._init(bias_tensor, bias_name, to=self.io_dtype)
                add_name = f"{name}/BiasAdd"
                add_out = f"{add_name}/output_0"
                self._node("Add", inputs=[mm_out, bias_name], outputs=[add_out], name=add_name)
                self._val(add_out, self.io_dtype, shape=out_shape)
                return add_out
        return mm_out

    def _matmul_raw(self, name, a_name, b_name, shape):
        """Raw MatMul between two existing values (weights already in graph)."""
        output = f"{name}/output_0"
        self._node("MatMul", inputs=[a_name, b_name], outputs=[output], name=name)
        self._val(output, self.io_dtype, shape=shape)
        return output

    def _reshape(self, name, root_input, shape_data, dtype, out_shape):
        """Reshape with a constant shape tensor."""
        shape_name = f"{name}/shape"
        self._const_tensor(np.array(shape_data, dtype=np.int64), shape_name)
        output = f"{name}/output_0"
        self._node("Reshape", inputs=[root_input, shape_name], outputs=[output], name=name)
        self._val(output, dtype, shape=out_shape)
        return output

    def _transpose(self, name, root_input, perm, dtype, out_shape):
        output = f"{name}/output_0"
        self._node("Transpose", inputs=[root_input], outputs=[output], name=name, perm=perm)
        self._val(output, dtype, shape=out_shape)
        return output

    def _add(self, name, a, b, dtype, shape):
        output = f"{name}/output_0"
        self._node("Add", inputs=[a, b], outputs=[output], name=name)
        self._val(output, dtype, shape=shape)
        return output

    def _mul(self, name, a, b, dtype, shape):
        output = f"{name}/output_0"
        self._node("Mul", inputs=[a, b], outputs=[output], name=name)
        self._val(output, dtype, shape=shape)
        return output

    def _neg(self, name, root_input, dtype, shape):
        output = f"{name}/output_0"
        self._node("Neg", inputs=[root_input], outputs=[output], name=name)
        self._val(output, dtype, shape=shape)
        return output

    def _concat(self, name, inputs, dtype, shape, axis=-1):
        output = f"{name}/output_0"
        self._node("Concat", inputs=inputs, outputs=[output], name=name, axis=axis)
        self._val(output, dtype, shape=shape)
        return output

    def _softmax(self, name, root_input, dtype, shape, axis=-1):
        output = f"{name}/output_0"
        self._node("Softmax", inputs=[root_input], outputs=[output], name=name, axis=axis)
        self._val(output, dtype, shape=shape)
        return output

    def _slice(self, name, root_input, starts, ends, axes, dtype, out_shape):
        """Slice along axes with scalar integer constants."""
        starts_name = f"{name}/starts"
        ends_name = f"{name}/ends"
        axes_name = f"{name}/axes"
        self._const_tensor(np.array(starts, dtype=np.int64), starts_name)
        self._const_tensor(np.array(ends, dtype=np.int64), ends_name)
        self._const_tensor(np.array(axes, dtype=np.int64), axes_name)
        output = f"{name}/output_0"
        self._node("Slice", inputs=[root_input, starts_name, ends_name, axes_name], outputs=[output], name=name)
        self._val(output, dtype, shape=out_shape)
        return output

    def _scale_mul(self, name, root_input, scale, dtype, shape):
        """Multiply a tensor by a scalar constant."""
        np_dtype = {ir.DataType.FLOAT: np.float32, ir.DataType.FLOAT16: np.float16}.get(dtype, np.float32)
        scale_name = f"{name}/scale"
        self._const_tensor(np.array(scale, dtype=np_dtype), scale_name)
        return self._mul(name, root_input, scale_name, dtype, shape)

    # ------------------------------------------------------------------ #
    #  2-D RoPE (pre-computed at graph-build time)                        #
    # ------------------------------------------------------------------ #

    def _precompute_rope_cos_sin(self):
        """Return cos/sin tensors shaped [1, 1, n_patches, head_dim].

        Pre-computes the Pixtral 2-D rotary embeddings for a fixed image
        grid.  The cos/sin tensors are stored as constant initialisers so
        they require no runtime computation.
        """
        vc = self.vision_config
        head_dim = self.vis_head_dim
        base = vc.rope_parameters["rope_theta"]
        n = self.n_patches_per_side

        h_idx = torch.arange(n)
        w_idx = torch.arange(n)
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs_h = torch.outer(h_idx, freqs[::2]).float()
        freqs_w = torch.outer(w_idx, freqs[1::2]).float()
        inv_freq = torch.cat(
            [freqs_h[:, None, :].repeat(1, n, 1), freqs_w[None, :, :].repeat(n, 1, 1)], dim=-1
        ).reshape(-1, head_dim // 2)
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        # inv_freq: [n_patches, head_dim]

        # Position IDs: row-major (h * max_width + w)
        h_grid, w_grid = torch.meshgrid(h_idx, w_idx, indexing="ij")
        position_ids = (h_grid * n + w_grid).reshape(-1)  # [n_patches]

        freqs_at_pos = inv_freq[position_ids]  # [n_patches, head_dim]
        cos = freqs_at_pos.cos()
        sin = freqs_at_pos.sin()

        # Shape for broadcasting with [1, num_heads, n_patches, head_dim]:
        # apply_rotary_pos_emb(unsqueeze_dim=0) does cos.unsqueeze(0) on
        # input already shaped [1, n_patches, head_dim], producing
        # [1, 1, n_patches, head_dim].
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, n_patches, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        return cos, sin

    def _apply_rope(self, prefix, q_or_k_name, cos_name, sin_name, shape):
        """Apply pre-computed 2-D RoPE: q_embed = q*cos + rotate_half(q)*sin.

        q_or_k_name: value name, shape [1, num_heads, n_patches, head_dim].
        cos_name, sin_name: initialisers of shape [1, 1, n_patches, head_dim].
        Returns the output value name (same shape as input).
        """
        hd = self.vis_head_dim
        half = hd // 2

        # rotate_half: split last dim in two halves, negate second, swap
        q1 = self._slice(
            f"{prefix}/rope/q1",
            q_or_k_name,
            starts=[0],
            ends=[half],
            axes=[-1],
            dtype=self.io_dtype,
            out_shape=shape[:-1] + [half],
        )
        q2 = self._slice(
            f"{prefix}/rope/q2",
            q_or_k_name,
            starts=[half],
            ends=[hd],
            axes=[-1],
            dtype=self.io_dtype,
            out_shape=shape[:-1] + [half],
        )
        neg_q2 = self._neg(f"{prefix}/rope/neg_q2", q2, self.io_dtype, shape[:-1] + [half])
        q_rot = self._concat(f"{prefix}/rope/q_rot", [neg_q2, q1], self.io_dtype, shape, axis=-1)

        q_cos = self._mul(f"{prefix}/rope/q_cos", q_or_k_name, cos_name, self.io_dtype, shape)
        q_sin = self._mul(f"{prefix}/rope/q_rot_sin", q_rot, sin_name, self.io_dtype, shape)
        q_embed = self._add(f"{prefix}/rope/q_embed", q_cos, q_sin, self.io_dtype, shape)
        return q_embed

    # ------------------------------------------------------------------ #
    #  Attention layer                                                     #
    # ------------------------------------------------------------------ #

    def _build_attention(self, layer_id, attn, root_input, cos_name, sin_name):
        """Build one PixtralAttention layer (encoder-style, no KV cache).

        root_input: [1, n_patches, vis_hidden_size]
        Returns: output name, same shape.
        """
        b = f"/vision/layers.{layer_id}/attn"
        n_p = self.n_patches
        d = self.vis_hidden_size
        nh = self.vis_num_heads
        hd = self.vis_head_dim

        # Q / K / V projections (no bias in Pixtral attention)
        q = self._matmul(
            f"{b}/q_proj/MatMul", root_input, attn.q_proj.weight, f"{b}/q_proj/MatMul.weight", out_shape=[1, n_p, d]
        )
        k = self._matmul(
            f"{b}/k_proj/MatMul", root_input, attn.k_proj.weight, f"{b}/k_proj/MatMul.weight", out_shape=[1, n_p, d]
        )
        v = self._matmul(
            f"{b}/v_proj/MatMul", root_input, attn.v_proj.weight, f"{b}/v_proj/MatMul.weight", out_shape=[1, n_p, d]
        )

        qkv_shape_4d = [1, n_p, nh, hd]
        q_4d = self._reshape(f"{b}/q_reshape", q, [1, n_p, nh, hd], self.io_dtype, qkv_shape_4d)
        k_4d = self._reshape(f"{b}/k_reshape", k, [1, n_p, nh, hd], self.io_dtype, qkv_shape_4d)
        v_4d = self._reshape(f"{b}/v_reshape", v, [1, n_p, nh, hd], self.io_dtype, qkv_shape_4d)

        # Transpose to [1, num_heads, n_patches, head_dim]
        qkv_t_shape = [1, nh, n_p, hd]
        q_t = self._transpose(f"{b}/q_t", q_4d, perm=[0, 2, 1, 3], dtype=self.io_dtype, out_shape=qkv_t_shape)
        k_t = self._transpose(f"{b}/k_t", k_4d, perm=[0, 2, 1, 3], dtype=self.io_dtype, out_shape=qkv_t_shape)
        v_t = self._transpose(f"{b}/v_t", v_4d, perm=[0, 2, 1, 3], dtype=self.io_dtype, out_shape=qkv_t_shape)

        # Apply 2-D RoPE to Q and K
        q_rope = self._apply_rope(f"{b}/q", q_t, cos_name, sin_name, qkv_t_shape)
        k_rope = self._apply_rope(f"{b}/k", k_t, cos_name, sin_name, qkv_t_shape)

        # Scaled dot-product attention (encoder, no causal mask)
        # K^T: [1, nh, hd, n_p]
        k_T = self._transpose(f"{b}/k_T", k_rope, perm=[0, 1, 3, 2], dtype=self.io_dtype, out_shape=[1, nh, hd, n_p])
        attn_w = self._matmul_raw(f"{b}/attn_w/MatMul", q_rope, k_T, shape=[1, nh, n_p, n_p])
        # Scale
        attn_ws = self._scale_mul(
            f"{b}/attn_scale", attn_w, scale=self.vis_attn_scale, dtype=self.io_dtype, shape=[1, nh, n_p, n_p]
        )
        attn_probs = self._softmax(f"{b}/attn_softmax", attn_ws, dtype=self.io_dtype, shape=[1, nh, n_p, n_p])
        attn_out_t = self._matmul_raw(f"{b}/attn_out/MatMul", attn_probs, v_t, shape=qkv_t_shape)

        # Transpose + Reshape back to [1, n_patches, hidden_size]
        attn_out = self._transpose(
            f"{b}/attn_out_t", attn_out_t, perm=[0, 2, 1, 3], dtype=self.io_dtype, out_shape=[1, n_p, nh, hd]
        )
        attn_out_2d = self._reshape(f"{b}/attn_out_reshape", attn_out, [1, n_p, d], self.io_dtype, [1, n_p, d])

        # O projection (no bias in Pixtral attention)
        o = self._matmul(
            f"{b}/o_proj/MatMul", attn_out_2d, attn.o_proj.weight, f"{b}/o_proj/MatMul.weight", out_shape=[1, n_p, d]
        )
        return o

    # ------------------------------------------------------------------ #
    #  MLP (SiLU-gated)                                                   #
    # ------------------------------------------------------------------ #

    def _build_mlp(self, layer_id, mlp, root_input):
        """Build one PixtralMLP layer (SiLU(gate_proj) * up_proj, then down_proj).

        SiLU(x) is implemented as ``x * Sigmoid(x)`` using standard ONNX ops.

        root_input: [1, n_patches, vis_hidden_size]
        Returns: output name, same shape.
        """
        b = f"/vision/layers.{layer_id}/mlp"
        n_p = self.n_patches
        d = self.vis_hidden_size
        ff = self.vis_intermediate_size

        gate = self._matmul(
            f"{b}/gate_proj/MatMul",
            root_input,
            mlp.gate_proj.weight,
            f"{b}/gate_proj/MatMul.weight",
            out_shape=[1, n_p, ff],
        )
        up = self._matmul(
            f"{b}/up_proj/MatMul", root_input, mlp.up_proj.weight, f"{b}/up_proj/MatMul.weight", out_shape=[1, n_p, ff]
        )

        # SiLU(gate) * up  (SiLU(x) = x * Sigmoid(x))
        sig_name = f"{b}/act/Sigmoid"
        sig_out = f"{sig_name}/output_0"
        self._node("Sigmoid", inputs=[gate], outputs=[sig_out], name=sig_name)
        self._val(sig_out, self.io_dtype, shape=[1, n_p, ff])

        silu_out = self._mul(f"{b}/act/Mul_silu", gate, sig_out, self.io_dtype, [1, n_p, ff])
        gate_up = self._mul(f"{b}/gate_up/Mul", silu_out, up, self.io_dtype, [1, n_p, ff])

        down = self._matmul(
            f"{b}/down_proj/MatMul",
            gate_up,
            mlp.down_proj.weight,
            f"{b}/down_proj/MatMul.weight",
            out_shape=[1, n_p, d],
        )
        return down

    # ------------------------------------------------------------------ #
    #  Single transformer layer                                           #
    # ------------------------------------------------------------------ #

    def _build_transformer_layer(self, layer_id, layer, root_input, cos_name, sin_name):
        """Build one PixtralAttentionLayer.

        Pipeline:
          attention_norm -> attention -> residual ->
          ffn_norm -> feed_forward -> residual
        """
        b = f"/vision/layers.{layer_id}"
        n_p = self.n_patches
        d = self.vis_hidden_size

        # attention_norm (RMSNorm, no skip)
        norm1_out = self._rms_norm(
            f"{b}/attention_norm/SimplifiedLayerNorm",
            root_input,
            layer.attention_norm.weight,
            f"{b}/attention_norm.weight",
            shape=[1, n_p, d],
        )

        # Attention
        attn_out = self._build_attention(layer_id, layer.attention, norm1_out, cos_name, sin_name)

        # Residual 1
        res1 = self._add(f"{b}/residual1/Add", root_input, attn_out, self.io_dtype, [1, n_p, d])

        # ffn_norm (RMSNorm, no skip)
        norm2_out = self._rms_norm(
            f"{b}/ffn_norm/SimplifiedLayerNorm", res1, layer.ffn_norm.weight, f"{b}/ffn_norm.weight", shape=[1, n_p, d]
        )

        # Feed-forward (SiLU-gated MLP)
        mlp_out = self._build_mlp(layer_id, layer.feed_forward, norm2_out)

        # Residual 2
        res2 = self._add(f"{b}/residual2/Add", res1, mlp_out, self.io_dtype, [1, n_p, d])
        return res2

    # ------------------------------------------------------------------ #
    #  Patch embedding (Conv2d + reshape + RMSNorm)                       #
    # ------------------------------------------------------------------ #

    def _build_patch_embedding(self, vt):
        """Build: pixel_values -> Conv2d -> flatten -> transpose -> ln_pre.

        Returns the value name of shape [1, n_patches, vis_hidden_size].
        """
        # Conv2d weights: [hidden_size, in_channels, patch_size, patch_size]
        conv_w = "vision.patch_conv.weight"
        self._init(vt.patch_conv.weight, conv_w, to=self.io_dtype)

        conv_out = "/vision/patch_conv/output_0"
        self._node(
            "Conv",
            inputs=["pixel_values", conv_w],
            outputs=[conv_out],
            name="/vision/patch_conv/Conv",
            dilations=[1, 1],
            group=1,
            kernel_shape=[self.patch_size, self.patch_size],
            pads=[0, 0, 0, 0],
            strides=[self.patch_size, self.patch_size],
        )
        n_h = n_w = self.n_patches_per_side
        self._val(conv_out, self.io_dtype, shape=[1, self.vis_hidden_size, n_h, n_w])

        # Reshape to [1, hidden_size, n_patches] then Transpose to [1, n_patches, hidden_size]
        reshape1 = self._reshape(
            "/vision/patch_embed/Reshape1",
            conv_out,
            [1, self.vis_hidden_size, self.n_patches],
            self.io_dtype,
            [1, self.vis_hidden_size, self.n_patches],
        )
        transposed = self._transpose(
            "/vision/patch_embed/Transpose",
            reshape1,
            perm=[0, 2, 1],
            dtype=self.io_dtype,
            out_shape=[1, self.n_patches, self.vis_hidden_size],
        )

        # ln_pre (SimplifiedLayerNormalization)
        ln_pre_out = self._rms_norm(
            "/vision/ln_pre/SimplifiedLayerNorm",
            transposed,
            vt.ln_pre.weight,
            "vision.ln_pre.weight",
            shape=[1, self.n_patches, self.vis_hidden_size],
        )
        return ln_pre_out

    # ------------------------------------------------------------------ #
    #  Multimodal projector                                               #
    # ------------------------------------------------------------------ #

    def _build_projector(self, proj, root_input):
        """Build the Mistral3MultiModalProjector.

        root_input: [1, n_patches, vis_hidden_size]

        Pipeline:
          norm -> patch_merger (unfold + linear) -> linear_1 -> gelu -> linear_2

        Returns: value name of shape [n_merged_patches, text_hidden_size].
        """
        n_p = self.n_patches
        d = self.vis_hidden_size
        s = self.spatial_merge_size
        nm = self.n_merged_patches
        n_h = n_w = self.n_patches_per_side
        mh, mw = n_h // s, n_w // s

        assert n_h % s == 0 and n_w % s == 0, f"image grid {n_h}x{n_w} not divisible by spatial_merge_size={s}"

        # --- Projector RMSNorm ---
        proj_norm_eps = float(self.config.text_config.rms_norm_eps)
        norm_w = "vision.projector.norm.weight"
        self._init(proj.norm.weight, norm_w, to=self.io_dtype)
        norm_out = "/vision/projector/norm/SimplifiedLayerNorm/output_0"
        self._node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, norm_w],
            outputs=[norm_out],
            name="/vision/projector/norm/SimplifiedLayerNorm",
            axis=-1,
            epsilon=proj_norm_eps,
            stash_type=1,
        )
        self._val(norm_out, self.io_dtype, shape=[1, n_p, d])

        # Squeeze batch dimension: [1, n_patches, d] -> [n_patches, d]
        squeeze_out = self._reshape("/vision/projector/squeeze", norm_out, [n_p, d], self.io_dtype, [n_p, d])

        # --- Patch Merger (unfold equivalent for non-overlapping windows) ---
        #
        # PyTorch: image_tokens [n_patches, d]
        #   -> view(n_h, n_w, d)
        #   -> permute(2,0,1).unsqueeze(0) -> [1, d, n_h, n_w]
        #   -> unfold(kernel=s, stride=s)  -> [1, d*s*s, n_h//s * n_w//s]
        #   -> view(d*s*s, n_merged).t()   -> [n_merged, d*s*s]
        #
        # Equivalent reshape+transpose+reshape (no overlap, stride==kernel):
        #   [n_patches, d]
        #   -> [n_h, n_w, d]                                Reshape
        #   -> [n_h//s, s, n_w//s, s, d]                   Reshape
        #   -> [n_h//s, n_w//s, d, s, s]  perm=[0,2,4,1,3] Transpose
        #   -> [n_merged, d*s*s]                            Reshape
        r1 = self._reshape("/vision/projector/merge/Reshape1", squeeze_out, [n_h, n_w, d], self.io_dtype, [n_h, n_w, d])
        r2 = self._reshape("/vision/projector/merge/Reshape2", r1, [mh, s, mw, s, d], self.io_dtype, [mh, s, mw, s, d])
        tp = self._transpose(
            "/vision/projector/merge/Transpose",
            r2,
            perm=[0, 2, 4, 1, 3],
            dtype=self.io_dtype,
            out_shape=[mh, mw, d, s, s],
        )
        merged = self._reshape("/vision/projector/merge/Reshape3", tp, [nm, d * s * s], self.io_dtype, [nm, d * s * s])

        # Merging linear (no bias): [nm, d*s*s] -> [nm, d]
        merged_out = self._matmul(
            "/vision/projector/merging_layer/MatMul",
            merged,
            proj.patch_merger.merging_layer.weight,
            "vision.projector.merging_layer.weight",
            out_shape=[nm, d],
        )

        # --- linear_1 + gelu + linear_2 ---
        t_hid = self.text_hidden_size
        lin1_bias = getattr(proj.linear_1, "bias", None)
        lin1_out = self._matmul(
            "/vision/projector/linear_1/MatMul",
            merged_out,
            proj.linear_1.weight,
            "vision.projector.linear_1.weight",
            out_shape=[nm, t_hid],
            bias_tensor=lin1_bias,
            bias_name="vision.projector.linear_1.bias" if lin1_bias is not None else None,
        )

        # GELU activation (default projector_hidden_act is "gelu")
        gelu_out = "/vision/projector/gelu/output_0"
        self._node(
            "Gelu", inputs=[lin1_out], outputs=[gelu_out], name="/vision/projector/gelu/Gelu", domain="com.microsoft"
        )
        self._val(gelu_out, self.io_dtype, shape=[nm, t_hid])

        # linear_2: [nm, text_hidden_size] -> [nm, text_hidden_size]
        lin2_bias = getattr(proj.linear_2, "bias", None)
        lin2_out = self._matmul(
            "/vision/projector/linear_2/MatMul",
            gelu_out,
            proj.linear_2.weight,
            "vision.projector.linear_2.weight",
            out_shape=[nm, t_hid],
            bias_tensor=lin2_bias,
            bias_name="vision.projector.linear_2.bias" if lin2_bias is not None else None,
        )
        return lin2_out

    # ------------------------------------------------------------------ #
    #  Main entry points                                                  #
    # ------------------------------------------------------------------ #

    def _load_hf_model(self, input_path):
        from transformers import Mistral3ForConditionalGeneration

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        return Mistral3ForConditionalGeneration.from_pretrained(
            src, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs
        )

    def make_model(self, input_path):
        """Load HF weights and build the ONNX graph in-memory."""
        hf_model = self._load_hf_model(input_path)
        hf_model.eval()

        vt = hf_model.model.vision_tower  # PixtralVisionModel
        proj = hf_model.model.multi_modal_projector  # Mistral3MultiModalProjector

        # Graph input
        pixel_values_in = self._val(
            "pixel_values", self.io_dtype, shape=[1, self.num_channels, self.image_size, self.image_size]
        )
        self.graph.inputs.append(pixel_values_in)

        # Pre-compute 2-D RoPE cos/sin (shared across all layers)
        cos_t, sin_t = self._precompute_rope_cos_sin()
        cos_name = "vision.rope.cos"
        sin_name = "vision.rope.sin"
        self._init(cos_t, cos_name, to=self.io_dtype)
        self._init(sin_t, sin_name, to=self.io_dtype)

        # Patch embedding
        x = self._build_patch_embedding(vt)

        # Transformer layers
        for layer_id, layer in enumerate(vt.transformer.layers):
            x = self._build_transformer_layer(layer_id, layer, x, cos_name, sin_name)

        # Projector
        image_features = self._build_projector(proj, x)

        # Graph output (rename via Identity so the output has the clean name)
        self._node("Identity", inputs=[image_features], outputs=["image_features"], name="/vision/output/Identity")
        out_val = self._val("image_features", self.io_dtype, shape=[self.n_merged_patches, self.text_hidden_size])
        self.graph.outputs.append(out_val)

        self.graph.sort()

    def save_model(self, out_dir):
        """Save the ONNX model with external data for large weight tensors."""
        out_path = os.path.join(out_dir, self.filename)
        data_path = out_path + ".data"
        if os.path.exists(out_path):
            print(f"Overwriting {out_path}")
            os.remove(out_path)
        if os.path.exists(data_path):
            print(f"Overwriting {data_path}")
            os.remove(data_path)

        print(f"Saving vision encoder ONNX model in {out_dir}")

        with tqdm() as pbar:
            total_set = False

            def callback(tensor: ir.TensorProtocol, metadata: dict):
                nonlocal total_set
                if not total_set:
                    pbar.total = metadata.total
                    total_set = True
                pbar.update()
                pbar.set_description(f"Saving {tensor.name} ({tensor.dtype.short_name()}, {tensor.shape})")

            ir.save(
                self.onnx_model,
                out_path,
                external_data=os.path.basename(data_path),
                size_threshold_bytes=0,
                callback=callback,
            )


class Ministral3ConditionalGenerationModel:
    """Orchestrates exporting both the vision encoder and the text decoder for
    ``Mistral3ForConditionalGeneration`` (Ministral-3-3B-Instruct-2512).

    The exported artifacts are:

    * ``vision_encoder.onnx`` - Pixtral vision tower + multimodal projector
      (one fixed-resolution image in, projected embeddings out).
    * ``model.onnx`` - Mistral text decoder with ``exclude_embeds=True``
      (takes ``inputs_embeds`` from the vision encoder, outputs logits + KV cache).
    * ``genai_config.json`` - extended with a ``vision_encoder`` section.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # --- Vision encoder ---
        self.vision_encoder = Ministral3VisionEncoderModel(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # --- Text decoder ---
        # Flatten text_config attributes onto the top-level config so that the
        # existing Ministral3TextModel constructor finds them (hidden_size, etc.).
        text_obj_config = copy.deepcopy(config)
        text_config = config.text_config
        for key in text_config:
            if not hasattr(text_obj_config, key) or key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "rms_norm_eps",
                "sliding_window",
                "vocab_size",
                "max_position_embeddings",
                "hidden_act",
                "rope_theta",
                "rope_scaling",
                "bos_token_id",
                "eos_token_id",
            ):
                setattr(text_obj_config, key, getattr(text_config, key))

        text_extra_options = dict(extra_options)
        text_extra_options["exclude_embeds"] = True

        self.text_model = Ministral3TextModel(text_obj_config, io_dtype, onnx_dtype, ep, cache_dir, text_extra_options)

    # ------------------------------------------------------------------
    # builder.py interface
    # ------------------------------------------------------------------

    def make_model(self, input_path):
        print("Building vision encoder (Pixtral + multimodal projector) for Mistral3ForConditionalGeneration...")
        self.vision_encoder.make_model(input_path)
        print("Building text decoder for Mistral3ForConditionalGeneration...")
        self.text_model.make_model(input_path)

    def save_model(self, out_dir):
        self.vision_encoder.save_model(out_dir)
        self.text_model.save_model(out_dir)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        # Let the text model write genai_config.json first, then extend it
        # with a vision_encoder section.
        self.text_model.make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        vision_cfg = self.vision_encoder.vision_config
        text_hidden_size = self.text_model.hidden_size
        image_size = vision_cfg.image_size
        patch_size = vision_cfg.patch_size
        spatial_merge_size = self.vision_encoder.config.spatial_merge_size
        num_patches_per_side = image_size // patch_size
        num_merged_patches = (num_patches_per_side**2) // (spatial_merge_size**2)

        genai_config["model"]["vision_encoder"] = {
            "filename": self.vision_encoder.filename,
            "hidden_size": vision_cfg.hidden_size,
            "image_size": image_size,
            "num_channels": vision_cfg.num_channels,
            "num_hidden_layers": vision_cfg.num_hidden_layers,
            "num_merged_patches": num_merged_patches,
            "patch_size": patch_size,
            "spatial_merge_size": spatial_merge_size,
            "text_hidden_size": text_hidden_size,
            "inputs": {"pixel_values": "pixel_values"},
            "outputs": {"image_features": "image_features"},
        }

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.text_model.save_processing(model_name_or_path, extra_kwargs, out_dir)
