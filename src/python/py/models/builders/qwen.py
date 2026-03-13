# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


import onnx_ir as ir
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from .base import Model


class QwenModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Qwen3Model(QwenModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()


class Qwen35Model(Qwen3Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self._normalize_qwen35_config(config)
        self._validate_qwen35_config(config)
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Qwen3_5RMSNorm computes output * (1 + weight) with weight init at 0,
        # so we need to add 1 to the stored weight before exporting (same as Gemma).
        self.layernorm_attrs["add_offset"] = 1

        # HF Qwen3.5 RMSNorm computes normalization in float32, then casts back.
        # Keep this behavior to avoid precision drift in long generations.
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = True
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = True

        self.qwen35_config = config
        self.layer_types = list(getattr(config, "layer_types", ["full_attention"] * self.num_layers))
        self.qwen35_attn_output_gate = getattr(config, "attn_output_gate", False)
        self.qwen35_linear_use_aux_state = any(layer_type == "linear_attention" for layer_type in self.layer_types)
        self._qwen35_linear_num_key_heads = getattr(config, "linear_num_key_heads", 0)
        self._qwen35_linear_num_value_heads = getattr(config, "linear_num_value_heads", 0)

        # Qwen3.5 linear attention is very sensitive to low-bit quantization.
        # Keep critical linear-attention projections in higher precision by default
        # to avoid generation collapse (e.g., repeated punctuation tokens).
        if self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4}:
            int4_nodes_to_exclude = self.quant_attrs["int4"]["nodes_to_exclude"]
            for layer_id, layer_type in enumerate(self.layer_types):
                if layer_type != "linear_attention":
                    continue
                int4_nodes_to_exclude.extend(
                    [
                        f"/model/layers.{layer_id}/linear_attn/in_proj_qkv/MatMul",
                        f"/model/layers.{layer_id}/linear_attn/in_proj_z/MatMul",
                        f"/model/layers.{layer_id}/linear_attn/in_proj_b/MatMul",
                        f"/model/layers.{layer_id}/linear_attn/in_proj_a/MatMul",
                        f"/model/layers.{layer_id}/linear_attn/out_proj/MatMul",
                    ]
                )

    @staticmethod
    def _normalize_qwen35_config(config):
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            for key in text_config:
                if not hasattr(config, key):
                    setattr(config, key, getattr(text_config, key))

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is None and text_config is not None:
            rope_parameters = getattr(text_config, "rope_parameters", None)

        if rope_parameters is None:
            return

        def _get_rope_param(params, key):
            # rope_parameters may be a plain dict (from JSON) or a config object with attributes
            if isinstance(params, dict):
                return params.get(key)
            return getattr(params, key, None)

        # Always prefer values from rope_parameters over class-level defaults.
        # HF config base class defines partial_rotary_factor=1.0 and rope_theta=10000
        # as class attributes, so hasattr() returns True even when the top-level JSON
        # does not set them — causing the hasattr guard to silently keep wrong defaults.
        rope_theta = _get_rope_param(rope_parameters, "rope_theta")
        if rope_theta is not None:
            config.rope_theta = rope_theta

        partial_rotary_factor = _get_rope_param(rope_parameters, "partial_rotary_factor")
        if partial_rotary_factor is not None:
            config.partial_rotary_factor = partial_rotary_factor

        # Qwen3.5 places several RoPE fields under rope_parameters. Lift the
        # fields that base.Model expects so they are consumed consistently.
        mrope_interleaved = _get_rope_param(rope_parameters, "mrope_interleaved")
        if mrope_interleaved is not None:
            config.rope_interleaved = bool(mrope_interleaved)

        mrope_section = _get_rope_param(rope_parameters, "mrope_section")
        rope_type = _get_rope_param(rope_parameters, "rope_type")
        if mrope_section is not None and (
            not hasattr(config, "rope_scaling") or getattr(config, "rope_scaling") is None
        ):
            # Keep HF-style section values unchanged. Qwen2.5-VL and Qwen3.5
            # consumers in this repo apply model-specific handling downstream.
            config.rope_scaling = {
                "type": rope_type if rope_type is not None else "mrope",
                "mrope_section": mrope_section,
            }

        if not hasattr(config, "rope_scaling"):
            config.rope_scaling = None

    @staticmethod
    def _validate_qwen35_config(config):
        unsupported_layer_types = sorted(set(getattr(config, "layer_types", [])) - {"full_attention", "linear_attention"})
        if unsupported_layer_types:
            raise NotImplementedError(
                "Qwen3.5 text export is not supported for layer_types "
                f"{unsupported_layer_types}. The current builder does not implement Qwen3.5 GatedDeltaNet "
                "linear attention layers outside the Qwen3.5 full/linear hybrid decoder layout."
            )

    def make_layer(self, layer_id, layer):
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )

        layer_type = self.layer_types[layer_id] if layer_id < len(self.layer_types) else "full_attention"
        if layer_type == "linear_attention":
            self.make_linear_attention(layer_id, layer.linear_attn, root_input=self.layernorm_attrs["output_0"])
        else:
            self._make_qwen35_passthrough_auxiliary_states(layer_id)
            self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])

        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    def make_linear_attention(self, layer_id, linear_attn, root_input, **kwargs):
        self._make_qwen35_linear_placeholder_present_kv(layer_id, root_input)

        qkv_conv_input = self._make_qwen35_linear_conv_input(layer_id, linear_attn, root_input)
        query, key, value = self._make_qwen35_linear_qkv(layer_id, qkv_conv_input)
        z_path = self._make_qwen35_linear_proj_with_shape(
            layer_id,
            linear_attn.in_proj_z,
            "in_proj_z",
            root_input,
            ["batch_size", "sequence_length", self.num_linear_v_heads * self.linear_head_v_dim],
        )
        beta_path = self._make_qwen35_linear_proj_with_shape(
            layer_id,
            linear_attn.in_proj_b,
            "in_proj_b",
            root_input,
            ["batch_size", "sequence_length", self.num_linear_v_heads],
        )
        a_path = self._make_qwen35_linear_proj_with_shape(
            layer_id,
            linear_attn.in_proj_a,
            "in_proj_a",
            root_input,
            ["batch_size", "sequence_length", self.num_linear_v_heads],
        )

        z_path = self._make_qwen35_linear_reshape_heads(
            layer_id,
            z_path,
            "z",
            self.num_linear_v_heads,
            self.linear_head_v_dim,
        )
        beta_path = self._make_qwen35_linear_reshape_scalar_heads(layer_id, beta_path, "beta")
        g_path = self._make_qwen35_linear_g_projection(layer_id, a_path, linear_attn)

        if self.num_linear_v_heads != self.num_linear_k_heads:
            repeat_factor = self.num_linear_v_heads // self.num_linear_k_heads
            query = self._make_qwen35_repeat_linear_heads(
                layer_id, query, "query", self.num_linear_k_heads, repeat_factor, self.linear_head_k_dim
            )
            key = self._make_qwen35_repeat_linear_heads(
                layer_id, key, "key", self.num_linear_k_heads, repeat_factor, self.linear_head_k_dim
            )

        core_attn = self._make_qwen35_linear_recurrent_attention(
            layer_id,
            query=query,
            key=key,
            value=value,
            beta=beta_path,
            g=g_path,
        )
        gated_output = self._make_qwen35_linear_gated_norm(layer_id, core_attn, z_path, linear_attn)
        flattened_output = self._make_qwen35_linear_flatten_heads(layer_id, gated_output)

        out_name = f"/model/layers.{layer_id}/linear_attn/out_proj/MatMul"
        out_proj = self.make_matmul(linear_attn.out_proj, out_name, flattened_output)
        output_path = f"{out_proj}/output_0"
        if linear_attn.out_proj.bias is not None:
            out_add_name = f"/model/layers.{layer_id}/linear_attn/out_proj/Add"
            self.make_add_bias(linear_attn.out_proj.bias, out_add_name, root_input=output_path)
            output_path = f"{out_add_name}/output_0"

        self.layernorm_attrs["skip_input"] = output_path

    def _make_qwen35_linear_placeholder_present_kv(self, layer_id, root_input):
        zero_hidden = self._make_qwen35_linear_zero_kv_hidden(layer_id, root_input)
        self.make_repeat_kv(
            layer_id,
            root_input=zero_hidden,
            past_kv=f"past_key_values.{layer_id}.key",
            present_kv=f"present.{layer_id}.key",
        )
        self.make_repeat_kv(
            layer_id,
            root_input=zero_hidden,
            past_kv=f"past_key_values.{layer_id}.value",
            present_kv=f"present.{layer_id}.value",
        )

    def _make_qwen35_passthrough_auxiliary_states(self, layer_id):
        if not self.qwen35_linear_use_aux_state:
            return

        present_conv_name = f"present_conv_states.{layer_id}"
        self.make_node(
            "Identity",
            inputs=[f"past_conv_states.{layer_id}"],
            outputs=[present_conv_name],
            name=f"/model/layers.{layer_id}/attn/present_conv_states/Identity",
        )
        self.make_value(present_conv_name, self.io_dtype, shape=self.output_shapes["present_conv_states"])

        present_recurrent_name = f"present_recurrent_states.{layer_id}"
        self.make_node(
            "Identity",
            inputs=[f"past_recurrent_states.{layer_id}"],
            outputs=[present_recurrent_name],
            name=f"/model/layers.{layer_id}/attn/present_recurrent_states/Identity",
        )
        self.make_value(
            present_recurrent_name,
            ir.DataType.FLOAT,
            shape=self.output_shapes["present_recurrent_states"],
        )

    def _make_qwen35_linear_zero_kv_hidden(self, layer_id, root_input):
        shape_name = f"/model/layers.{layer_id}/linear_attn/kv_placeholder/Shape"
        self.make_shape(shape_name, root_input, shape=[3])

        batch_dim_name = f"/model/layers.{layer_id}/linear_attn/kv_placeholder/BatchDim"
        self.make_gather(
            batch_dim_name,
            [f"{shape_name}/output_0", "/model/constants/INT64/0"],
            dtype=ir.DataType.INT64,
            shape=[],
            axis=0,
        )
        batch_unsqueeze_name = f"/model/layers.{layer_id}/linear_attn/kv_placeholder/BatchUnsqueeze"
        self.make_unsqueeze(
            batch_unsqueeze_name,
            [f"{batch_dim_name}/output_0", "/model/constants/INT64/[0]"],
            dtype=ir.DataType.INT64,
            shape=[1],
        )

        seq_dim_name = f"/model/layers.{layer_id}/linear_attn/kv_placeholder/SeqDim"
        self.make_gather(
            seq_dim_name,
            [f"{shape_name}/output_0", "/model/constants/INT64/1"],
            dtype=ir.DataType.INT64,
            shape=[],
            axis=0,
        )
        seq_unsqueeze_name = f"/model/layers.{layer_id}/linear_attn/kv_placeholder/SeqUnsqueeze"
        self.make_unsqueeze(
            seq_unsqueeze_name,
            [f"{seq_dim_name}/output_0", "/model/constants/INT64/[0]"],
            dtype=ir.DataType.INT64,
            shape=[1],
        )

        hidden_shape_name = f"/model/layers.{layer_id}/linear_attn/kv_placeholder/HiddenShape"
        self.make_concat(
            hidden_shape_name,
            [
                f"{batch_unsqueeze_name}/output_0",
                f"{seq_unsqueeze_name}/output_0",
                f"/model/constants/INT64/[{self.num_kv_heads * self.head_size}]",
            ],
            dtype=ir.DataType.INT64,
            shape=[3],
            axis=0,
        )

        zero_name = f"/model/layers.{layer_id}/linear_attn/kv_placeholder/ConstantOfShape"
        self.make_constant_of_shape(
            zero_name,
            f"{hidden_shape_name}/output_0",
            ir.tensor([0], dtype=self.io_dtype),
            self.io_dtype,
            ["batch_size", "sequence_length", self.num_kv_heads * self.head_size],
        )
        return f"{zero_name}/output_0"

    def make_attention_input_proj(self, layer_id, attention, root_input, **kwargs):
        if not self.qwen35_attn_output_gate:
            return super().make_attention_input_proj(layer_id, attention, root_input, **kwargs)

        self.make_attention_unpacked(layer_id, attention, root_input, **kwargs)

        gate_hidden_size = self.num_attn_heads * self.head_size
        q_proj_shape = ["batch_size", "sequence_length", gate_hidden_size * 2]
        q_path = self._make_qwen35_proj_with_bias(layer_id, attention.q_proj, "q_proj", root_input, q_proj_shape)
        self.attention_attrs["q_path"] = self._make_qwen35_attention_gate_slice(
            layer_id,
            q_path,
            "query",
            start=0,
            end=gate_hidden_size,
            shape=["batch_size", "sequence_length", gate_hidden_size],
        )
        self.attention_attrs["q_gate_path"] = self._make_qwen35_attention_gate_slice(
            layer_id,
            q_path,
            "gate",
            start=gate_hidden_size,
            end=gate_hidden_size * 2,
            shape=["batch_size", "sequence_length", gate_hidden_size],
        )

        kv_hidden_size = self.num_kv_heads * self.head_size
        kv_shape = ["batch_size", "sequence_length", kv_hidden_size]
        self.attention_attrs["k_path"] = self._make_qwen35_proj_with_bias(
            layer_id, attention.k_proj, "k_proj", root_input, kv_shape
        )
        self.attention_attrs["v_path"] = self._make_qwen35_proj_with_bias(
            layer_id, attention.v_proj, "v_proj", root_input, kv_shape
        )

    def make_attention_output_proj(self, layer_id, attention, root_input, **kwargs):
        if not self.qwen35_attn_output_gate:
            return super().make_attention_output_proj(layer_id, attention, root_input, **kwargs)

        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        attn_output = f"{attn_name}/output_0"
        hidden_shape = ["batch_size", "sequence_length", self.num_attn_heads * self.head_size]

        gate_path = self.attention_attrs.pop("q_gate_path")
        gate_name = f"/model/layers.{layer_id}/attn/q_gate/Sigmoid"
        self.make_sigmoid(gate_name, gate_path, self.io_dtype, hidden_shape)

        gated_attn_name = f"/model/layers.{layer_id}/attn/q_gate/Mul"
        self.make_mul(
            gated_attn_name,
            [attn_output, f"{gate_name}/output_0"],
            self.io_dtype,
            hidden_shape,
        )
        attn_output = f"{gated_attn_name}/output_0"

        o_proj = "o_proj" if hasattr(attention, "o_proj") else "dense"
        o_matmul_basename = f"/model/layers.{layer_id}/attn/o_proj/MatMul"
        o_weight = getattr(attention, o_proj)
        o_matmul_name = self.make_matmul(o_weight, o_matmul_basename, attn_output)

        o_bias_exists = getattr(attention, o_proj).bias is not None
        output_path = f"{o_matmul_name}/output_0"
        if o_bias_exists:
            o_add_name = f"/model/layers.{layer_id}/attn/o_proj/Add"
            o_bias = getattr(attention, o_proj).bias
            self.make_add_bias(o_bias, o_add_name, root_input=f"{o_matmul_name}/output_0")
            output_path = f"{o_add_name}/output_0"

        self.layernorm_attrs["skip_input"] = output_path

    def _make_qwen35_proj_with_bias(self, layer_id, proj, proj_name, root_input, shape):
        proj_matmul_name = f"/model/layers.{layer_id}/attn/{proj_name}/MatMul"
        self.make_matmul(proj, proj_matmul_name, root_input)
        proj_path = f"{proj_matmul_name}/output_0"

        proj_bias_exists = proj.bias is not None and torch.count_nonzero(proj.bias) > 0
        if proj_bias_exists:
            proj_add_name = f"/model/layers.{layer_id}/attn/{proj_name}/Add"
            self.make_add_bias(proj.bias, proj_add_name, root_input=proj_path)
            proj_path = f"{proj_add_name}/output_0"

        self.make_value(proj_path, self.io_dtype, shape=shape)
        return proj_path

    @property
    def num_linear_k_heads(self):
        return self._qwen35_linear_num_key_heads

    @property
    def num_linear_v_heads(self):
        return self._qwen35_linear_num_value_heads

    @property
    def linear_head_k_dim(self):
        return self.input_shapes["past_recurrent_states"][2]

    @property
    def linear_head_v_dim(self):
        return self.input_shapes["past_recurrent_states"][3]

    @property
    def linear_conv_kernel_dim(self):
        return self.input_shapes["past_conv_states"][2]

    @property
    def linear_conv_dim(self):
        return self.input_shapes["past_conv_states"][1]

    def _make_qwen35_linear_proj_with_shape(self, layer_id, proj, proj_name, root_input, shape):
        proj_matmul_name = f"/model/layers.{layer_id}/linear_attn/{proj_name}/MatMul"
        self.make_matmul(proj, proj_matmul_name, root_input)
        proj_path = f"{proj_matmul_name}/output_0"

        if proj.bias is not None and torch.count_nonzero(proj.bias) > 0:
            proj_add_name = f"/model/layers.{layer_id}/linear_attn/{proj_name}/Add"
            self.make_add_bias(proj.bias, proj_add_name, root_input=proj_path)
            proj_path = f"{proj_add_name}/output_0"

        self.make_value(proj_path, self.io_dtype, shape=shape)
        return proj_path

    def _make_qwen35_linear_conv_input(self, layer_id, linear_attn, root_input):
        qkv_path = self._make_qwen35_linear_proj_with_shape(
            layer_id,
            linear_attn.in_proj_qkv,
            "in_proj_qkv",
            root_input,
            ["batch_size", "sequence_length", self.linear_conv_dim],
        )

        qkv_transpose_name = f"/model/layers.{layer_id}/linear_attn/in_proj_qkv/Transpose"
        self.make_transpose(
            qkv_transpose_name,
            qkv_path,
            self.io_dtype,
            ["batch_size", self.linear_conv_dim, "sequence_length"],
            perm=[0, 2, 1],
        )

        past_conv_name = f"past_conv_states.{layer_id}"
        past_trim_name = f"/model/layers.{layer_id}/linear_attn/past_conv_states/Slice"
        self.make_slice(
            past_trim_name,
            [
                past_conv_name,
                "/model/constants/INT64/[1]",
                f"/model/constants/INT64/[{self.linear_conv_kernel_dim}]",
                "/model/constants/INT64/[2]",
                "/model/constants/INT64/[1]",
            ],
            dtype=self.io_dtype,
            shape=["batch_size", self.linear_conv_dim, self.linear_conv_kernel_dim - 1],
        )

        conv_concat_name = f"/model/layers.{layer_id}/linear_attn/conv_input/Concat"
        conv_concat_output = f"{conv_concat_name}/output_0"
        self.make_node(
            "Concat",
            inputs=[f"{past_trim_name}/output_0", f"{qkv_transpose_name}/output_0"],
            outputs=[conv_concat_output],
            name=conv_concat_name,
            axis=2,
        )
        self.make_value(conv_concat_output, self.io_dtype, shape=["batch_size", self.linear_conv_dim, "unk"])

        present_conv_name = f"present_conv_states.{layer_id}"
        present_conv_slice_name = f"/model/layers.{layer_id}/linear_attn/present_conv_states/Slice"
        self.make_slice(
            present_conv_slice_name,
            [
                conv_concat_output,
                f"/model/constants/INT64/[-{self.linear_conv_kernel_dim}]",
                "/model/constants/INT64/[9223372036854775807]",
                "/model/constants/INT64/[2]",
                "/model/constants/INT64/[1]",
            ],
            dtype=self.io_dtype,
            shape=["batch_size", self.linear_conv_dim, self.linear_conv_kernel_dim],
        )
        self.make_node("Identity", inputs=[f"{present_conv_slice_name}/output_0"], outputs=[present_conv_name], name=f"/model/layers.{layer_id}/linear_attn/present_conv_states/Identity")

        conv_weight_name = f"model.layers.{layer_id}.linear_attn.conv1d.weight"
        self.make_initializer(linear_attn.conv1d.weight, conv_weight_name, to=self.io_dtype)
        conv_inputs = [conv_concat_output, conv_weight_name]
        if linear_attn.conv1d.bias is not None:
            conv_bias_name = f"model.layers.{layer_id}.linear_attn.conv1d.bias"
            self.make_initializer(linear_attn.conv1d.bias, conv_bias_name, to=self.io_dtype)
            conv_inputs.append(conv_bias_name)

        conv_name = f"/model/layers.{layer_id}/linear_attn/conv1d/Conv"
        conv_output = f"{conv_name}/output_0"
        self.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=[conv_output],
            name=conv_name,
            group=self.linear_conv_dim,
            kernel_shape=[self.linear_conv_kernel_dim],
            pads=[0, 0],
            strides=[1],
        )
        self.make_value(conv_output, self.io_dtype, shape=["batch_size", self.linear_conv_dim, "sequence_length"])

        sigmoid_name = f"/model/layers.{layer_id}/linear_attn/conv1d/Sigmoid"
        self.make_sigmoid(sigmoid_name, conv_output, self.io_dtype, ["batch_size", self.linear_conv_dim, "sequence_length"])
        silu_name = f"/model/layers.{layer_id}/linear_attn/conv1d/Mul"
        self.make_mul(
            silu_name,
            [conv_output, f"{sigmoid_name}/output_0"],
            self.io_dtype,
            ["batch_size", self.linear_conv_dim, "sequence_length"],
        )

        conv_transpose_name = f"/model/layers.{layer_id}/linear_attn/conv1d/Transpose"
        self.make_transpose(
            conv_transpose_name,
            f"{silu_name}/output_0",
            self.io_dtype,
            ["batch_size", "sequence_length", self.linear_conv_dim],
            perm=[0, 2, 1],
        )
        return f"{conv_transpose_name}/output_0"

    def _make_qwen35_linear_qkv(self, layer_id, qkv_path):
        query_width = self.num_linear_k_heads * self.linear_head_k_dim
        key_width = query_width
        value_width = self.num_linear_v_heads * self.linear_head_v_dim

        query_path = self._make_qwen35_attention_gate_slice(
            layer_id,
            qkv_path,
            "linear_query",
            start=0,
            end=query_width,
            shape=["batch_size", "sequence_length", query_width],
        )
        key_path = self._make_qwen35_attention_gate_slice(
            layer_id,
            qkv_path,
            "linear_key",
            start=query_width,
            end=query_width + key_width,
            shape=["batch_size", "sequence_length", key_width],
        )
        value_path = self._make_qwen35_attention_gate_slice(
            layer_id,
            qkv_path,
            "linear_value",
            start=query_width + key_width,
            end=query_width + key_width + value_width,
            shape=["batch_size", "sequence_length", value_width],
        )
        query_path = self._make_qwen35_linear_reshape_heads(
            layer_id, query_path, "query", self.num_linear_k_heads, self.linear_head_k_dim
        )
        key_path = self._make_qwen35_linear_reshape_heads(
            layer_id, key_path, "key", self.num_linear_k_heads, self.linear_head_k_dim
        )
        value_path = self._make_qwen35_linear_reshape_heads(
            layer_id, value_path, "value", self.num_linear_v_heads, self.linear_head_v_dim
        )
        return query_path, key_path, value_path

    def _make_qwen35_linear_reshape_heads(self, layer_id, root_input, name_suffix, num_heads, head_dim):
        reshape_name = f"/model/layers.{layer_id}/linear_attn/{name_suffix}/Reshape"
        self.make_reshape(
            reshape_name,
            [root_input, f"/model/constants/INT64/[0, 0, {num_heads}, {head_dim}]"] ,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", num_heads, head_dim],
        )
        return f"{reshape_name}/output_0"

    def _make_qwen35_linear_reshape_scalar_heads(self, layer_id, root_input, name_suffix):
        reshape_name = f"/model/layers.{layer_id}/linear_attn/{name_suffix}/Reshape"
        self.make_reshape(
            reshape_name,
            [root_input, f"/model/constants/INT64/[0, 0, {self.num_linear_v_heads}]"] ,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.num_linear_v_heads],
        )
        return f"{reshape_name}/output_0"

    def _make_qwen35_linear_flatten_heads(self, layer_id, root_input):
        reshape_name = f"/model/layers.{layer_id}/linear_attn/output/Reshape"
        self.make_reshape(
            reshape_name,
            [root_input, f"/model/constants/INT64/[0, 0, {self.num_linear_v_heads * self.linear_head_v_dim}]"] ,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.num_linear_v_heads * self.linear_head_v_dim],
        )
        return f"{reshape_name}/output_0"

    def _make_qwen35_linear_g_projection(self, layer_id, a_path, linear_attn):
        dt_bias_name = f"model.layers.{layer_id}.linear_attn.dt_bias"
        self.make_initializer(linear_attn.dt_bias, dt_bias_name, to=ir.DataType.FLOAT)
        a_cast_name = f"/model/layers.{layer_id}/linear_attn/a/Cast"
        self.make_cast(a_cast_name, a_path, ir.DataType.FLOAT, ["batch_size", "sequence_length", self.num_linear_v_heads])

        add_name = f"/model/layers.{layer_id}/linear_attn/g/Add"
        self.make_add(
            add_name,
            [f"{a_cast_name}/output_0", dt_bias_name],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads],
        )

        softplus_name = f"/model/layers.{layer_id}/linear_attn/g/Softplus"
        self.make_node(
            "Softplus",
            inputs=[f"{add_name}/output_0"],
            outputs=[f"{softplus_name}/output_0"],
            name=softplus_name,
        )
        self.make_value(
            f"{softplus_name}/output_0",
            ir.DataType.FLOAT,
            shape=["batch_size", "sequence_length", self.num_linear_v_heads],
        )

        a_log_name = f"model.layers.{layer_id}.linear_attn.A_log"
        self.make_initializer(linear_attn.A_log, a_log_name, to=ir.DataType.FLOAT)
        exp_name = f"/model/layers.{layer_id}/linear_attn/g/Exp"
        self.make_node("Exp", inputs=[a_log_name], outputs=[f"{exp_name}/output_0"], name=exp_name)
        self.make_value(f"{exp_name}/output_0", ir.DataType.FLOAT, shape=[self.num_linear_v_heads])

        neg_name = f"/model/layers.{layer_id}/linear_attn/g/Neg"
        self.make_node("Neg", inputs=[f"{exp_name}/output_0"], outputs=[f"{neg_name}/output_0"], name=neg_name)
        self.make_value(f"{neg_name}/output_0", ir.DataType.FLOAT, shape=[self.num_linear_v_heads])

        mul_name = f"/model/layers.{layer_id}/linear_attn/g/Mul"
        self.make_mul(
            mul_name,
            [f"{neg_name}/output_0", f"{softplus_name}/output_0"],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads],
        )
        return f"{mul_name}/output_0"

    def _make_qwen35_linear_cast(self, layer_id, root_input, name_suffix, shape):
        cast_name = f"/model/layers.{layer_id}/linear_attn/{name_suffix}/Cast"
        self.make_cast(cast_name, root_input, ir.DataType.FLOAT, shape)
        return f"{cast_name}/output_0"

    def _make_qwen35_linear_l2norm(self, layer_id, root_input, name_suffix, shape):
        basename = f"/model/layers.{layer_id}/linear_attn/{name_suffix}/l2norm"
        cast_input = self._make_qwen35_linear_cast(layer_id, root_input, f"{name_suffix}_float", shape)
        pow_name = f"{basename}/Pow"
        self.make_node(
            "Pow",
            inputs=[cast_input, "/model/constants/FLOAT/2"],
            outputs=[f"{pow_name}/output_0"],
            name=pow_name,
        )
        self.make_value(f"{pow_name}/output_0", ir.DataType.FLOAT, shape=shape)

        sum_name = f"{basename}/ReduceSum"
        self.make_reduce_sum(
            sum_name,
            [f"{pow_name}/output_0", "/model/constants/INT64/[-1]"],
            ir.DataType.FLOAT,
            [shape[0], shape[1], shape[2], 1],
            keepdims=True,
        )

        eps_name = f"{basename}/Add"
        self.make_add(
            eps_name,
            [f"{sum_name}/output_0", "/model/constants/FLOAT/1e-12"],
            ir.DataType.FLOAT,
            [shape[0], shape[1], shape[2], 1],
        )
        sqrt_name = f"{basename}/Sqrt"
        self.make_sqrt(sqrt_name, [f"{eps_name}/output_0"], ir.DataType.FLOAT, [shape[0], shape[1], shape[2], 1])
        div_name = f"{basename}/Div"
        self.make_div(div_name, [cast_input, f"{sqrt_name}/output_0"], ir.DataType.FLOAT, shape)
        return f"{div_name}/output_0"

    def _make_qwen35_linear_scale_query(self, layer_id, root_input):
        scale_name = f"/model/layers.{layer_id}/linear_attn/query_scale/Mul"
        self.make_mul(
            scale_name,
            [root_input, f"/model/constants/FLOAT/{1.0 / (self.linear_head_k_dim ** 0.5)}"],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_k_dim],
        )
        return f"{scale_name}/output_0"

    def _make_qwen35_linear_token_major(self, layer_id, root_input, name_suffix, shape, perm):
        transpose_name = f"/model/layers.{layer_id}/linear_attn/{name_suffix}/Transpose"
        self.make_transpose(transpose_name, root_input, ir.DataType.FLOAT, shape, perm=perm)
        return f"{transpose_name}/output_0"

    def _make_qwen35_linear_recurrent_attention(self, layer_id, query, key, value, beta, g):
        query = self._make_qwen35_linear_l2norm(
            layer_id,
            query,
            "query",
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_k_dim],
        )
        key = self._make_qwen35_linear_l2norm(
            layer_id,
            key,
            "key",
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_k_dim],
        )
        query = self._make_qwen35_linear_scale_query(layer_id, query)
        value = self._make_qwen35_linear_cast(
            layer_id,
            value,
            "value",
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
        )
        beta = self._make_qwen35_linear_cast(
            layer_id,
            beta,
            "beta",
            ["batch_size", "sequence_length", self.num_linear_v_heads],
        )

        beta_sigmoid_name = f"/model/layers.{layer_id}/linear_attn/beta/Sigmoid"
        self.make_sigmoid(
            beta_sigmoid_name,
            beta,
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads],
        )
        beta = f"{beta_sigmoid_name}/output_0"

        query = self._make_qwen35_linear_token_major(
            layer_id,
            query,
            "query_token_major",
            ["sequence_length", "batch_size", self.num_linear_v_heads, self.linear_head_k_dim],
            perm=[1, 0, 2, 3],
        )
        key = self._make_qwen35_linear_token_major(
            layer_id,
            key,
            "key_token_major",
            ["sequence_length", "batch_size", self.num_linear_v_heads, self.linear_head_k_dim],
            perm=[1, 0, 2, 3],
        )
        value = self._make_qwen35_linear_token_major(
            layer_id,
            value,
            "value_token_major",
            ["sequence_length", "batch_size", self.num_linear_v_heads, self.linear_head_v_dim],
            perm=[1, 0, 2, 3],
        )
        beta = self._make_qwen35_linear_token_major(
            layer_id,
            beta,
            "beta_token_major",
            ["sequence_length", "batch_size", self.num_linear_v_heads],
            perm=[1, 0, 2],
        )
        g = self._make_qwen35_linear_token_major(
            layer_id,
            g,
            "g_token_major",
            ["sequence_length", "batch_size", self.num_linear_v_heads],
            perm=[1, 0, 2],
        )

        q_shape_name = f"/model/layers.{layer_id}/linear_attn/query_token_major/Shape"
        self.make_shape(q_shape_name, query, shape=[4])
        loop_count_name = f"/model/layers.{layer_id}/linear_attn/query_token_major/Gather"
        self.make_gather(
            loop_count_name,
            [f"{q_shape_name}/output_0", "/model/constants/INT64/0"],
            dtype=ir.DataType.INT64,
            shape=[],
            axis=0,
        )
        initial_state_name = f"/model/layers.{layer_id}/linear_attn/recurrent_state/Cast"
        self.make_cast(
            initial_state_name,
            f"past_recurrent_states.{layer_id}",
            ir.DataType.FLOAT,
            ["batch_size", self.num_linear_v_heads, self.linear_head_k_dim, self.linear_head_v_dim],
        )

        core_output_name = f"/model/layers.{layer_id}/linear_attn/Loop/output_1"
        present_state_name = f"present_recurrent_states.{layer_id}"
        loop_name = f"/model/layers.{layer_id}/linear_attn/Loop"
        # Dummy output names for carried-through loop variables (read-only sequences)
        query_final = f"/model/layers.{layer_id}/linear_attn/Loop/_query_final"
        key_final = f"/model/layers.{layer_id}/linear_attn/Loop/_key_final"
        value_final = f"/model/layers.{layer_id}/linear_attn/Loop/_value_final"
        beta_final = f"/model/layers.{layer_id}/linear_attn/Loop/_beta_final"
        g_final = f"/model/layers.{layer_id}/linear_attn/Loop/_g_final"
        body_graph = self._make_qwen35_linear_loop_body(layer_id)
        self.make_node(
            "Loop",
            inputs=[
                f"{loop_count_name}/output_0",
                "/model/constants/BOOL/True",
                f"{initial_state_name}/output_0",
                query,
                key,
                value,
                beta,
                g,
            ],
            outputs=[present_state_name, query_final, key_final, value_final, beta_final, g_final, core_output_name],
            name=loop_name,
            body=body_graph,
        )
        self.make_value(
            present_state_name,
            ir.DataType.FLOAT,
            shape=["batch_size", self.num_linear_v_heads, self.linear_head_k_dim, self.linear_head_v_dim],
        )
        self.make_value(
            query_final,
            ir.DataType.FLOAT,
            shape=["sequence_length", "batch_size", self.num_linear_v_heads, self.linear_head_k_dim],
        )
        self.make_value(
            key_final,
            ir.DataType.FLOAT,
            shape=["sequence_length", "batch_size", self.num_linear_v_heads, self.linear_head_k_dim],
        )
        self.make_value(
            value_final,
            ir.DataType.FLOAT,
            shape=["sequence_length", "batch_size", self.num_linear_v_heads, self.linear_head_v_dim],
        )
        self.make_value(
            beta_final,
            ir.DataType.FLOAT,
            shape=["sequence_length", "batch_size", self.num_linear_v_heads],
        )
        self.make_value(
            g_final,
            ir.DataType.FLOAT,
            shape=["sequence_length", "batch_size", self.num_linear_v_heads],
        )
        self.make_value(
            core_output_name,
            ir.DataType.FLOAT,
            shape=["sequence_length", "batch_size", self.num_linear_v_heads, self.linear_head_v_dim],
        )

        output_transpose_name = f"/model/layers.{layer_id}/linear_attn/output/Transpose"
        self.make_transpose(
            output_transpose_name,
            core_output_name,
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
            perm=[1, 0, 2, 3],
        )
        return f"{output_transpose_name}/output_0"

    def _make_qwen35_linear_loop_body(self, layer_id):
        float_tensor = ir.TensorType(ir.DataType.FLOAT)
        bool_tensor = ir.TensorType(ir.DataType.BOOL)
        int_tensor = ir.TensorType(ir.DataType.INT64)

        iteration = ir.Value(name=f"loop_{layer_id}_iter", type=int_tensor, shape=ir.Shape([]))
        cond_in = ir.Value(name=f"loop_{layer_id}_cond_in", type=bool_tensor, shape=ir.Shape([]))
        state_in = ir.Value(
            name=f"loop_{layer_id}_state_in",
            type=float_tensor,
            shape=ir.Shape(["batch_size", self.num_linear_v_heads, self.linear_head_k_dim, self.linear_head_v_dim]),
        )
        query_seq = ir.Value(
            name=f"loop_{layer_id}_query_seq",
            type=float_tensor,
            shape=ir.Shape(["sequence_length", "batch_size", self.num_linear_v_heads, self.linear_head_k_dim]),
        )
        key_seq = ir.Value(
            name=f"loop_{layer_id}_key_seq",
            type=float_tensor,
            shape=ir.Shape(["sequence_length", "batch_size", self.num_linear_v_heads, self.linear_head_k_dim]),
        )
        value_seq = ir.Value(
            name=f"loop_{layer_id}_value_seq",
            type=float_tensor,
            shape=ir.Shape(["sequence_length", "batch_size", self.num_linear_v_heads, self.linear_head_v_dim]),
        )
        beta_seq = ir.Value(
            name=f"loop_{layer_id}_beta_seq",
            type=float_tensor,
            shape=ir.Shape(["sequence_length", "batch_size", self.num_linear_v_heads]),
        )
        g_seq = ir.Value(
            name=f"loop_{layer_id}_g_seq",
            type=float_tensor,
            shape=ir.Shape(["sequence_length", "batch_size", self.num_linear_v_heads]),
        )

        def value(name, dtype, shape):
            return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))

        def node(op_type, inputs, outputs, name, **attributes):
            return ir.node(op_type, inputs=inputs, outputs=outputs, attributes=attributes, name=name)

        axis2 = ir.tensor([2], dtype=ir.DataType.INT64)
        axis3 = ir.tensor([3], dtype=ir.DataType.INT64)
        reduce_k = ir.tensor([2], dtype=ir.DataType.INT64)
        axes2_const = value(f"loop_{layer_id}_axes2_const", ir.DataType.INT64, [1])
        axes3_const = value(f"loop_{layer_id}_axes3_const", ir.DataType.INT64, [1])
        reduce_k_const = value(f"loop_{layer_id}_reduce_k_const", ir.DataType.INT64, [1])
        cond_true = value(f"loop_{layer_id}_cond_true", ir.DataType.BOOL, [])
        query_t = value(f"loop_{layer_id}_query_t", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_k_dim])
        key_t = value(f"loop_{layer_id}_key_t", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_k_dim])
        value_t = value(f"loop_{layer_id}_value_t", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_v_dim])
        beta_t = value(f"loop_{layer_id}_beta_t", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads])
        g_t = value(f"loop_{layer_id}_g_t", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads])
        g_exp = value(f"loop_{layer_id}_g_exp", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads])
        g_unsq2 = value(f"loop_{layer_id}_g_unsq2", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, 1])
        g_unsq3 = value(f"loop_{layer_id}_g_unsq3", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, 1, 1])
        state_scaled = value(
            f"loop_{layer_id}_state_scaled", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_k_dim, self.linear_head_v_dim]
        )
        k_unsq = value(f"loop_{layer_id}_k_unsq", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_k_dim, 1])
        q_unsq = value(f"loop_{layer_id}_q_unsq", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_k_dim, 1])
        beta_unsq = value(f"loop_{layer_id}_beta_unsq", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, 1])
        state_times_k = value(
            f"loop_{layer_id}_state_times_k", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_k_dim, self.linear_head_v_dim]
        )
        kv_mem = value(f"loop_{layer_id}_kv_mem", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_v_dim])
        delta_sub = value(f"loop_{layer_id}_delta_sub", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_v_dim])
        delta = value(f"loop_{layer_id}_delta", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_v_dim])
        delta_unsq = value(f"loop_{layer_id}_delta_unsq", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, 1, self.linear_head_v_dim])
        state_update = value(
            f"loop_{layer_id}_state_update", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_k_dim, self.linear_head_v_dim]
        )
        state_out = value(
            f"loop_{layer_id}_state_out", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_k_dim, self.linear_head_v_dim]
        )
        state_times_q = value(
            f"loop_{layer_id}_state_times_q", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_k_dim, self.linear_head_v_dim]
        )
        out_t = value(f"loop_{layer_id}_out_t", ir.DataType.FLOAT, ["batch_size", self.num_linear_v_heads, self.linear_head_v_dim])

        body_nodes = [
            ir.node(
                "Constant",
                inputs=[],
                outputs=[cond_true],
                name=f"loop_{layer_id}_cond_constant",
                attributes={"value": ir.tensor(True, dtype=ir.DataType.BOOL)},
            ),
            ir.node(
                "Constant",
                inputs=[],
                outputs=[axes2_const],
                name=f"loop_{layer_id}_axes2_constant",
                attributes={"value": axis2},
            ),
            ir.node(
                "Constant",
                inputs=[],
                outputs=[axes3_const],
                name=f"loop_{layer_id}_axes3_constant",
                attributes={"value": axis3},
            ),
            ir.node(
                "Constant",
                inputs=[],
                outputs=[reduce_k_const],
                name=f"loop_{layer_id}_reduce_k_constant",
                attributes={"value": reduce_k},
            ),
            node("Gather", [query_seq, iteration], [query_t], f"loop_{layer_id}_gather_query", axis=0),
            node("Gather", [key_seq, iteration], [key_t], f"loop_{layer_id}_gather_key", axis=0),
            node("Gather", [value_seq, iteration], [value_t], f"loop_{layer_id}_gather_value", axis=0),
            node("Gather", [beta_seq, iteration], [beta_t], f"loop_{layer_id}_gather_beta", axis=0),
            node("Gather", [g_seq, iteration], [g_t], f"loop_{layer_id}_gather_g", axis=0),
            node("Exp", [g_t], [g_exp], f"loop_{layer_id}_exp_g"),
            node("Unsqueeze", [g_exp, axes2_const], [g_unsq2], f"loop_{layer_id}_unsqueeze_g_2"),
            node("Unsqueeze", [g_unsq2, axes3_const], [g_unsq3], f"loop_{layer_id}_unsqueeze_g_3"),
            node("Mul", [state_in, g_unsq3], [state_scaled], f"loop_{layer_id}_mul_state_g"),
            node("Unsqueeze", [key_t, axes3_const], [k_unsq], f"loop_{layer_id}_unsqueeze_k"),
            node("Mul", [state_scaled, k_unsq], [state_times_k], f"loop_{layer_id}_mul_state_k"),
            node("ReduceSum", [state_times_k, reduce_k_const], [kv_mem], f"loop_{layer_id}_reduce_k", keepdims=0),
            node("Sub", [value_t, kv_mem], [delta_sub], f"loop_{layer_id}_sub_delta"),
            node("Unsqueeze", [beta_t, axes2_const], [beta_unsq], f"loop_{layer_id}_unsqueeze_beta"),
            node("Mul", [delta_sub, beta_unsq], [delta], f"loop_{layer_id}_mul_beta"),
            node("Unsqueeze", [delta, axes2_const], [delta_unsq], f"loop_{layer_id}_unsqueeze_delta"),
            node("Mul", [k_unsq, delta_unsq], [state_update], f"loop_{layer_id}_state_update"),
            node("Add", [state_scaled, state_update], [state_out], f"loop_{layer_id}_state_add"),
            node("Unsqueeze", [query_t, axes3_const], [q_unsq], f"loop_{layer_id}_unsqueeze_q"),
            node("Mul", [state_out, q_unsq], [state_times_q], f"loop_{layer_id}_mul_state_q"),
            node("ReduceSum", [state_times_q, reduce_k_const], [out_t], f"loop_{layer_id}_reduce_q", keepdims=0),
        ]

        return ir.Graph(
            inputs=[iteration, cond_in, state_in, query_seq, key_seq, value_seq, beta_seq, g_seq],
            outputs=[cond_true, state_out, query_seq, key_seq, value_seq, beta_seq, g_seq, out_t],
            nodes=body_nodes,
            name=f"qwen35_linear_loop_body_{layer_id}",
            opset_imports={"": 21},
        )

    def _make_qwen35_linear_gated_norm(self, layer_id, hidden_states, gate, linear_attn):
        basename = f"/model/layers.{layer_id}/linear_attn/gated_norm"
        gate_cast = self._make_qwen35_linear_cast(
            layer_id,
            gate,
            "gate_norm",
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
        )
        hidden_cast = self._make_qwen35_linear_cast(
            layer_id,
            hidden_states,
            "hidden_norm",
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
        )
        pow_name = f"{basename}/Pow"
        self.make_node(
            "Pow",
            inputs=[hidden_cast, "/model/constants/FLOAT/2"],
            outputs=[f"{pow_name}/output_0"],
            name=pow_name,
        )
        self.make_value(
            f"{pow_name}/output_0",
            ir.DataType.FLOAT,
            shape=["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
        )
        mean_name = f"{basename}/ReduceMean"
        self.make_reduce_mean(
            mean_name,
            [f"{pow_name}/output_0", "/model/constants/INT64/[-1]"],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads, 1],
            keepdims=True,
        )
        add_name = f"{basename}/Add"
        self.make_add(
            add_name,
            [f"{mean_name}/output_0", f"/model/constants/FLOAT/{linear_attn.norm.variance_epsilon}"],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads, 1],
        )
        sqrt_name = f"{basename}/Sqrt"
        self.make_sqrt(sqrt_name, [f"{add_name}/output_0"], ir.DataType.FLOAT, ["batch_size", "sequence_length", self.num_linear_v_heads, 1])
        div_name = f"{basename}/Div"
        self.make_div(
            div_name,
            [hidden_cast, f"{sqrt_name}/output_0"],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
        )

        weight_name = f"model.layers.{layer_id}.linear_attn.norm.weight"
        self.make_initializer(linear_attn.norm.weight, weight_name, to=self.io_dtype)
        weighted_name = f"{basename}/Mul"
        self.make_mul(
            weighted_name,
            [f"{div_name}/output_0", weight_name],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
        )
        gate_sigmoid_name = f"{basename}/Sigmoid"
        self.make_sigmoid(
            gate_sigmoid_name,
            gate_cast,
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
        )
        gate_silu_name = f"{basename}/GateMul"
        self.make_mul(
            gate_silu_name,
            [gate_cast, f"{gate_sigmoid_name}/output_0"],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
        )
        output_name = f"{basename}/OutputMul"
        self.make_mul(
            output_name,
            [f"{weighted_name}/output_0", f"{gate_silu_name}/output_0"],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
        )
        cast_back_name = f"{basename}/Cast"
        self.make_cast(
            cast_back_name,
            f"{output_name}/output_0",
            self.io_dtype,
            ["batch_size", "sequence_length", self.num_linear_v_heads, self.linear_head_v_dim],
        )
        return f"{cast_back_name}/output_0"

    def _make_qwen35_repeat_linear_heads(self, layer_id, root_input, name_suffix, num_heads, repeat_factor, head_dim):
        basename = f"/model/layers.{layer_id}/linear_attn/{name_suffix}/repeat_heads"
        unsqueeze_name = f"{basename}/Unsqueeze"
        self.make_unsqueeze(
            unsqueeze_name,
            [root_input, "/model/constants/INT64/[3]"],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", num_heads, 1, head_dim],
        )
        expand_name = f"{basename}/Expand"
        self.make_expand(
            expand_name,
            [
                f"{unsqueeze_name}/output_0",
                f"/model/constants/INT64/[1, 1, {num_heads}, {repeat_factor}, {head_dim}]",
            ],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", num_heads, repeat_factor, head_dim],
        )
        reshape_name = f"{basename}/Reshape"
        self.make_reshape(
            reshape_name,
            [f"{expand_name}/output_0", f"/model/constants/INT64/[0, 0, {num_heads * repeat_factor}, {head_dim}]"] ,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", num_heads * repeat_factor, head_dim],
        )
        return f"{reshape_name}/output_0"

    def _make_qwen35_attention_gate_slice(self, layer_id, root_input, name_suffix, start, end, shape):
        slice_name = f"/model/layers.{layer_id}/attn/q_proj/{name_suffix}/Slice"
        slice_inputs = [
            root_input,
            f"/model/constants/INT64/[{start}]",
            f"/model/constants/INT64/[{end}]",
            "/model/constants/INT64/[-1]",
            "/model/constants/INT64/[1]",
        ]
        self.make_slice(slice_name, slice_inputs, dtype=self.io_dtype, shape=shape)
        return f"{slice_name}/output_0"


class Qwen25VLTextModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # The HF model (Qwen2RMSNorm) *always* computes LayerNorm in float32.
        # By inheriting from `base.Model`, all `layernorm_attrs["cast"]` flags
        # are `False`. This causes parity loss and type mismatch error.
        #
        # SOLUTION: Manually set all `cast` flags to `True`. This forces the
        # builder to cast bf16 inputs -> fp32, compute LN, and cast fp32
        # outputs -> bf16, matching the HF model and fixing both errors.
        #
        print("Forcing LayerNorm computation to float32 (and enabling all casts) for Qwen2.5-VL parity.")
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = True
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = True

        # Qwen2's RoPE *always* computes in float32.
        # We must replicate this behavior.
        print("Forcing RoPE computation to float32 for Qwen2.5-VL parity.")
        if "rope_cast" not in self.attention_attrs:
            self.attention_attrs["rope_cast"] = {}
        self.attention_attrs["rope_cast"]["use_fp32"] = True

        # Check rope type since huggingface model supports yarn but that is not recommended as mentioned in model card. Example:
        #    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24,24]}
        if config.rope_scaling and "type" in config.rope_scaling:
            assert config.rope_scaling["type"] in ["mrope", "default"]

        # Qwen 2.5 VL applies RoPE manually before attention, not fused in the op
        self.attention_attrs["use_rope_in_attn"] = False

        # We need separate Q, K, V tensors to apply MRoPE manually.
        # Packed MatMul provides a single output which would require splitting.
        self.attention_attrs["use_packed_matmul"] = False

        if "position_ids" not in self.input_names:
            print("Re-adding 'position_ids' to self.input_names.")
            self.input_names.append("position_ids")

        self.mrope_sections = self.rope_attrs.get("mrope", {}).get("sections", [])
        if not self.mrope_sections:
            raise ValueError("MRoPE sections not found in config.text_config.rope_scaling.mrope_section")

        # The HF logic is `mrope_section * 2`, not `[s * 2 for s in mrope_section]`.
        # This results in [16, 24, 24, 16, 24, 24]
        self.mrope_splits = self.mrope_sections * 2

        if sum(self.mrope_splits) != self.head_size:
            # The sum (128) should now correctly match self.head_size (128)
            raise ValueError(
                f"MRoPE splits {self.mrope_splits} sum ({sum(self.mrope_splits)}) does not match head size ({self.head_size})"
            )

        # Force GroupQueryAttention since make_attention() below only implements GQA.
        self.attention_attrs["op_type"] = "GroupQueryAttention"

        if not self.is_gqa_supported():
            print(f"Warning: {self.ep} does not support GQA for {self.io_dtype}, so GQA might fallback to CPU!")

        # Create and save the inv_freq tensor
        self.make_inv_freq_tensor()

    def make_inv_freq_tensor(self):
        """
        Calculates and saves the `inv_freq` tensor as an initializer.
        This is copied from base.py:make_rotary_embedding_caches_from_scratch
        """
        dim = int(self.rope_attrs["partial_rotary_factor"] * self.head_size)
        inv_freq = 1.0 / (
            self.rope_attrs["rescale_factors"]
            * (self.rope_attrs["theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        )

        # The HF model expects H/2, not R/2
        if dim != self.head_size:
            print(
                f"Warning: partial_rotary_factor ({self.rope_attrs['partial_rotary_factor']}) is not 1. This might be unsupported."
            )
            inv_freq = inv_freq[: (self.head_size // 2)]

        self.make_initializer(inv_freq, "model.inv_freq", to=ir.DataType.FLOAT)
        print("Created and saved 'model.inv_freq' initializer.")

    def make_inputs_and_outputs(self):
        # Qwen2.5-VL uses 3D position_ids
        self.input_shapes["position_ids"] = [3, "batch_size", "sequence_length"]

        # Call the base Model's make_inputs_and_outputs (skipping MistralModel's)
        super().make_inputs_and_outputs()

    def make_dynamic_rope_caches(self, layer_id, basename):
        # Make nodes for the Dynamic RoPE Cache subgraph
        #
        # Re-implements Qwen2_5_VLRotaryEmbedding.forward using ONNX ops.
        # Takes 3D position_ids and inv_freq and dynamically creates
        # the cos/sin caches.
        #
        #         inv_freq (H/2)                                     position_ids (3, B, S)
        #             |                                                      |
        #         Unsqueeze                                              Unsqueeze
        #             |                                                      |
        #           Expand                                                  Cast
        #      (3, B, H/2, 1)                                           (3, B, 1, S)
        #             |                                                      |
        #             +--------------------------+---------------------------+
        #                                        |
        #                                      MatMul
        #                                   (3, B, H/2, S)
        #                                        |
        #                                    Transpose
        #                                   (3, B, S, H/2)
        #                                        |
        #                                     Concat
        #                                  (3, B, S, H)
        #                                        |
        #                          +-------------+-------------+
        #                          |                           |
        #                         Cos                         Sin
        #                          |                           |
        #                         Mul                         Mul
        #                   (apply scaling)             (apply scaling)
        #
        pos_ids_name = "position_ids"
        inv_freq_name = "model.inv_freq"
        head_dim_half = self.head_size // 2

        # Get Batch Size from position_ids.shape[1]
        shape_pos_ids_name = f"{basename}/pos_ids/Shape"
        shape_pos_ids_output = f"{shape_pos_ids_name}/output_0"
        self.make_shape(shape_pos_ids_name, pos_ids_name, [3])

        gather_batch_size_name = f"{basename}/pos_ids/Gather"
        gather_batch_size_output = f"{gather_batch_size_name}/output_0"
        self.make_gather(
            gather_batch_size_name,
            [shape_pos_ids_output, "/model/constants/INT64/[1]"],
            ir.DataType.INT64,
            [1],
            axis=0,
        )

        # Expand inv_freq: [H/2] -> [1, 1, H/2, 1]
        unsqueeze_1_name = f"{basename}/inv_freq/Unsqueeze"
        unsqueeze_1_output = f"{unsqueeze_1_name}/output_0"
        self.make_unsqueeze(
            unsqueeze_1_name,
            [inv_freq_name, "/model/constants/INT64/[0, 1, 3]"],
            ir.DataType.FLOAT,
            [1, 1, head_dim_half, 1],
        )

        # Create target shape for Expand: [3, B, H/2, 1]
        concat_expand_shape_name = f"{basename}/expand_shape/Concat"
        concat_expand_shape_output = f"{concat_expand_shape_name}/output_0"
        self.make_concat(
            concat_expand_shape_name,
            [
                "/model/constants/INT64/[3]",
                gather_batch_size_output,
                f"/model/constants/INT64/[{head_dim_half}, 1]",
            ],
            ir.DataType.INT64,
            [4],
            axis=0,
        )

        expand_name = f"{basename}/inv_freq/Expand"
        expand_output = f"{expand_name}/output_0"
        self.make_expand(
            expand_name,
            [unsqueeze_1_output, concat_expand_shape_output],
            ir.DataType.FLOAT,
            [3, "batch_size", head_dim_half, 1],
        )

        # Expand position_ids: [3, B, S] -> [3, B, 1, S]
        unsqueeze_2_name = f"{basename}/pos_ids/Unsqueeze"
        unsqueeze_2_output = f"{unsqueeze_2_name}/output_0"
        self.make_unsqueeze(
            unsqueeze_2_name,
            [pos_ids_name, "/model/constants/INT64/[2]"],
            ir.DataType.INT64,
            [3, "batch_size", 1, "sequence_length"],
        )

        # Cast position_ids to float
        cast_name = f"{basename}/pos_ids/Cast"
        cast_output = f"{cast_name}/output_0"
        self.make_cast(
            cast_name,
            unsqueeze_2_output,
            ir.DataType.FLOAT,
            [3, "batch_size", 1, "sequence_length"],
        )

        # MatMul: [3, B, H/2, 1] @ [3, B, 1, S] -> [3, B, H/2, S]
        matmul_name = f"{basename}/freqs/MatMul"
        matmul_output = f"{matmul_name}/output_0"
        self.make_node("MatMul", [expand_output, cast_output], [matmul_output], name=matmul_name)
        self.make_value(
            matmul_output,
            ir.DataType.FLOAT,
            [3, "batch_size", head_dim_half, "sequence_length"],
        )

        # Transpose: [3, B, H/2, S] -> [3, B, S, H/2]
        transpose_name = f"{basename}/freqs/Transpose"
        transpose_output = f"{transpose_name}/output_0"
        self.make_transpose(
            transpose_name,
            matmul_output,
            ir.DataType.FLOAT,
            [3, "batch_size", "sequence_length", head_dim_half],
            perm=[0, 1, 3, 2],
        )

        # Concat (freqs, freqs): [3, B, S, H/2] -> [3, B, S, H]
        concat_name = f"{basename}/Concat"
        concat_output = f"{concat_name}/output_0"
        self.make_concat(
            concat_name,
            [transpose_output, transpose_output],
            ir.DataType.FLOAT,
            [3, "batch_size", "sequence_length", self.head_size],
            axis=-1,
        )

        # Cos(emb) and Sin(emb)
        cos_name = f"{basename}/Cos"
        cos_output = f"{cos_name}/output_0"
        self.make_node("Cos", [concat_output], [cos_output], name=cos_name)
        self.make_value(
            cos_output,
            ir.DataType.FLOAT,
            [3, "batch_size", "sequence_length", self.head_size],
        )

        sin_name = f"{basename}/Sin"
        sin_output = f"{sin_name}/output_0"
        self.make_node("Sin", [concat_output], [sin_output], name=sin_name)
        self.make_value(
            sin_output,
            ir.DataType.FLOAT,
            [3, "batch_size", "sequence_length", self.head_size],
        )

        return cos_output, sin_output

    def make_mrope_flattened_caches(self, layer_id, dyn_cos, dyn_sin):
        # Converts the 3D MRoPE caches [3, B, S, H] into flattened, interleaved caches [B*S, H/2]
        # suitable for the RotaryEmbedding operator.
        # The logic is:
        #   1. Slice dynamic caches to H/2.
        #   2. Split into 3 chunks based on mrope_sections (e.g. 16, 24, 24).
        #   3. Gather Temporal(0), Height(1), Width(2) specific slices for each chunk.
        #   4. Concat back to H/2.
        #   5. Flatten to [B*S, H/2].
        # The subgraph looks like:
        #      dyn_cos (3, B, S, H)
        #             |
        #           Slice
        #      (3, B, S, H/2)
        #             |
        #           Split
        #   (3, B, S, sections[i])
        #       /     |     \
        #  Gather  Gather  Gather
        #   idx=0   idx=1   idx=2
        #    /        |       \
        # Squeeze  Squeeze  Squeeze
        #    \        |       /
        #     \       |      /
        #      \      |     /
        #          Concat
        #       (B, S, H/2)
        #             |
        #          Reshape
        #        (B*S, H/2)

        basename = f"/model/layers.{layer_id}/attn/mrope_flattened_cache"

        def process_cache(input_name, name_suffix):
            # 1. Slice to H/2: [3, B, S, H] -> [3, B, S, H/2]
            slice_name = f"{basename}/{name_suffix}/half/Slice"
            slice_output = f"{slice_name}/output_0"
            self.make_slice(
                slice_name,
                [
                    input_name,
                    "/model/constants/INT64/[0]",
                    f"/model/constants/INT64/[{self.head_size // 2}]",
                    "/model/constants/INT64/[-1]",
                ],
                ir.DataType.FLOAT,
                [3, "batch_size", "sequence_length", self.head_size // 2],
            )

            # Create a Constant node for mrope_sections: [16, 24, 24]
            sections_name = f"{basename}/mrope_sections/Constant"
            sections_output = f"{basename}/mrope_sections"
            self.make_node(
                "Constant",
                [],
                [sections_output],
                name=sections_name,
                value=ir.tensor(torch.tensor(self.mrope_sections, dtype=torch.int64), name=sections_output),
            )
            self.make_value(sections_output, ir.DataType.INT64, [3])

            # 2. Split: [3, B, S, H/2] -> 3 * [3, B, S, section_dim]
            split_name = f"{basename}/{name_suffix}/Split"
            split_outputs = [f"{split_name}/output_{i}" for i in range(3)]
            self.make_node(
                "Split",
                [slice_output, sections_output],
                split_outputs,
                name=split_name,
                axis=-1,
            )

            # 3. Gather + Squeeze: Reorder T, H, W
            gathered_chunks = []
            for i in range(3):
                # Chunk 0->T(0), Chunk 1->H(1), Chunk 2->W(2)
                gather_name = f"{basename}/{name_suffix}/chunk_{i}/Gather"
                gather_output = f"{gather_name}/output_0"
                self.make_node(
                    "Gather",
                    [split_outputs[i], f"/model/constants/INT64/[{i}]"],
                    [gather_output],
                    name=gather_name,
                    axis=0,
                )
                # Gather output is [1, B, S, dim]

                squeeze_name = f"{basename}/{name_suffix}/chunk_{i}/Squeeze"
                squeeze_output = f"{squeeze_name}/output_0"
                self.make_squeeze(
                    squeeze_name,
                    [gather_output, "/model/constants/INT64/[0]"],
                    ir.DataType.FLOAT,
                    ["batch_size", "sequence_length", self.mrope_sections[i]],
                )
                gathered_chunks.append(squeeze_output)

            # 4. Concat: -> [B, S, H/2]
            concat_name = f"{basename}/{name_suffix}/Concat"
            concat_output = f"{concat_name}/output_0"
            self.make_concat(
                concat_name,
                gathered_chunks,
                ir.DataType.FLOAT,
                ["batch_size", "sequence_length", self.head_size // 2],
                axis=-1,
            )

            # 5. Flatten: -> [B*S, H/2]
            reshape_name = f"{basename}/{name_suffix}_flat/Reshape"
            reshape_output = f"{reshape_name}/output_0"
            self.make_reshape(
                reshape_name,
                [concat_output, f"/model/constants/INT64/[-1, {self.head_size // 2}]"],
                ir.DataType.FLOAT,
                ["total_token_count", self.head_size // 2],
            )
            return reshape_output

        flat_cos = process_cache(dyn_cos, "cos")
        flat_sin = process_cache(dyn_sin, "sin")

        return flat_cos, flat_sin

    def apply_mrope_rotation(self, layer_id, q_or_k_path, q_or_k_shape, dyn_cos, dyn_sin, num_heads, basename):
        # Make nodes for the MRoPE rotation subgraph using RotaryEmbedding op
        #
        # 1. Flatten 3D caches [3, B, S, H] -> [B*S, H/2] (via make_mrope_flattened_caches)
        # 2. Generate linear position IDs [B, S] (0 .. B*S-1)
        # 3. Apply RotaryEmbedding
        #
        #      dyn_cos (3, B, S, H)   dyn_sin (3, B, S, H)
        #              |                      |
        #    make_mrope_flattened_caches (slice, split, gather, concat, flatten)
        #              |                      |
        #        flat_cos               flat_sin
        #      (B*S, H/2)             (B*S, H/2)
        #              |                      |
        #              +-----------+----------+
        #                          |
        #      q_or_k              |              position_ids
        #    (B, S, N*H)           |            (0 .. B*S-1)
        #        |                 |                 |
        #     Reshape              |              Reshape
        #        |                 |                 |
        #    Transpose             |                 |
        #   (B, N, S, H)           |               (B, S)
        #        |                 |                 |
        #        +--------+--------+--------+--------+
        #                 |                 |
        #          RotaryEmbedding (com.microsoft)
        #                 |
        #            output (B, N, S, H)
        #                 |
        #             Transpose
        #                 |
        #              Reshape
        #            (B, S, N*H)

        # 1. Prepare flattened MRoPE caches [B*S, H/2]
        #    This slices, splits, and re-assembles the 3D dynamic caches into the correct per-token layout.
        flat_cos, flat_sin = self.make_mrope_flattened_caches(layer_id, dyn_cos, dyn_sin)

        # 2. Prepare position_ids [B, S] (values 0 to B*S - 1)
        #    RotaryEmbedding will use these indices to access the flattened cache.
        #    Get B*S from q_or_k shape. q_or_k is [B, S, N*H].
        shape_node = f"{basename}/Shape"
        self.make_shape(shape_node, q_or_k_path, [3])

        # Extract B and S
        batch_size_node = f"{basename}/BatchSize/Gather"
        batch_size_out = f"{batch_size_node}/output_0"
        self.make_gather(
            batch_size_node, [f"{shape_node}/output_0", "/model/constants/INT64/[0]"], ir.DataType.INT64, [], 0
        )

        seq_len_node = f"{basename}/SeqLen/Gather"
        seq_len_out = f"{seq_len_node}/output_0"
        self.make_gather(
            seq_len_node, [f"{shape_node}/output_0", "/model/constants/INT64/[1]"], ir.DataType.INT64, [], 0
        )

        # Calculate Total Tokens = B * S
        mul_len_node = f"{basename}/TotalLen/Mul"
        mul_len_out = f"{mul_len_node}/output_0"
        self.make_node("Mul", [batch_size_out, seq_len_out], [mul_len_out], name=mul_len_node)
        self.make_value(mul_len_out, ir.DataType.INT64, [])

        # Range(0, TotalTokens)
        range_node = f"{basename}/Range"
        range_out = f"{range_node}/output_0"
        self.make_node(
            "Range", ["/model/constants/INT64/0", mul_len_out, "/model/constants/INT64/1"], [range_out], name=range_node
        )
        self.make_value(range_out, ir.DataType.INT64, ["total_token_count"])

        # Slice Position IDs shape from input shape (take first 2 dims)
        slice_shape_node = f"{basename}/SliceShape"
        slice_shape_out = f"{slice_shape_node}/output_0"
        self.make_slice(
            slice_shape_node,
            [
                f"{shape_node}/output_0",
                "/model/constants/INT64/[0]",
                "/model/constants/INT64/[2]",
                "/model/constants/INT64/[0]",
            ],
            ir.DataType.INT64,
            [2],
        )

        # Reshape Range output to [B, S]
        pos_ids_reshape_node = f"{basename}/PosIds/Reshape"
        pos_ids_out = f"{pos_ids_reshape_node}/output_0"
        self.make_reshape(
            pos_ids_reshape_node, [range_out, slice_shape_out], ir.DataType.INT64, ["batch_size", "sequence_length"]
        )

        # 3. Prepare Q/K input [B, N, S, H]
        #    Input is [B, S, N*H]. Reshape -> [B, S, N, H] -> Transpose -> [B, N, S, H]
        reshape_in_node = f"{basename}/Input/Reshape"
        reshape_in_out = f"{reshape_in_node}/output_0"
        self.make_reshape(
            reshape_in_node,
            [q_or_k_path, f"/model/constants/INT64/[0, 0, {num_heads}, {self.head_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", num_heads, self.head_size],
        )

        transpose_in_node = f"{basename}/Input/Transpose"
        transpose_in_out = f"{transpose_in_node}/output_0"
        target_shape_bnsh = ["batch_size", num_heads, "sequence_length", self.head_size]
        self.make_transpose(transpose_in_node, reshape_in_out, self.io_dtype, target_shape_bnsh, [0, 2, 1, 3])

        # 4. Handle Type Casting
        #    RotaryEmbedding requires input, cos, sin to be same type.
        #    Qwen2.5-VL forces float32 computation.
        force_fp32 = self.attention_attrs.get("rope_cast", {}).get("use_fp32", False)
        compute_dtype = ir.DataType.FLOAT if force_fp32 else self.io_dtype

        rope_input = transpose_in_out
        if force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            cast_in_node = f"{basename}/Input/Cast"
            rope_input = f"{cast_in_node}/output_0"
            self.make_cast(cast_in_node, transpose_in_out, compute_dtype, target_shape_bnsh)

        rope_cos = flat_cos
        rope_sin = flat_sin
        # Note: dyn_cos is Float. flat_cos is Float. If compute_dtype is not Float (e.g. fp16), we must cast cache.
        if compute_dtype != ir.DataType.FLOAT:
            # Cache is Float, we need FP16
            cast_cos_node = f"{basename}/Cos/Cast"
            rope_cos = f"{cast_cos_node}/output_0"
            self.make_cast(cast_cos_node, flat_cos, compute_dtype, ["total_token_count", self.head_size // 2])

            cast_sin_node = f"{basename}/Sin/Cast"
            rope_sin = f"{cast_sin_node}/output_0"
            self.make_cast(cast_sin_node, flat_sin, compute_dtype, ["total_token_count", self.head_size // 2])

        # 5. RotaryEmbedding Node
        rope_node = f"{basename}/RotaryEmbedding"
        rope_output = f"{rope_node}/output_0"
        self.make_node(
            "RotaryEmbedding",
            [rope_input, pos_ids_out, rope_cos, rope_sin],
            [rope_output],
            name=rope_node,
            domain="com.microsoft",
            rotary_embedding_dim=self.head_size,
            num_heads=num_heads,
            interleaved=0,  # False, matches rotate_half logic
        )
        self.make_value(rope_output, compute_dtype, target_shape_bnsh)

        # 6. Post-process Output
        #    Cast back if needed -> Transpose -> Reshape
        final_rope_output = rope_output
        if force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            cast_out_node = f"{basename}/Output/Cast"
            final_rope_output = f"{cast_out_node}/output_0"
            self.make_cast(cast_out_node, rope_output, self.io_dtype, target_shape_bnsh)

        transpose_out_node = f"{basename}/Output/Transpose"
        transpose_out_out = f"{transpose_out_node}/output_0"
        self.make_transpose(
            transpose_out_node,
            final_rope_output,
            self.io_dtype,
            ["batch_size", "sequence_length", num_heads, self.head_size],
            [0, 2, 1, 3],
        )

        reshape_out_node = f"{basename}/Output/Reshape"
        reshape_out_out = f"{reshape_out_node}/output_0"
        self.make_reshape(
            reshape_out_node,
            [transpose_out_out, f"/model/constants/INT64/[0, 0, {num_heads * self.head_size}]"],
            self.io_dtype,
            q_or_k_shape,
        )

        return reshape_out_out

    def make_attention_qk_subgraph(self, layer_id, attention, root_input, **kwargs):
        # Make nodes for the Attention subgraph (with MRoPE)
        #
        #        q_path    k_path    v_path
        #          |        |        |
        #          |        |        +-----------------+
        #          |        |                          |
        #   (make_dynamic_rope_caches)                 |
        #          |                                   |
        #    +-----+-----+                             |
        #    |           |                             |
        # dyn_cos     dyn_sin                          |
        #    |           |                             |
        #    v           v                             |
        # (apply_mrope_rotation for Q)                 |
        #          |                                   |
        #        Q_Rot                                 |
        #          |     (apply_mrope_rotation for K)  |
        #          |                 |                 |
        #          |               K_Rot               |
        #          |                 |                 |
        #          +--------+--------+                 |
        #                   |                          |
        #           GroupQueryAttention <--------------+
        #                   |

        # 1. Calculate shapes for MRoPE rotation
        q_shape = [
            "batch_size",
            "sequence_length",
            self.num_attn_heads * self.head_size,
        ]
        k_shape = [
            "batch_size",
            "sequence_length",
            self.num_kv_heads * self.head_size,
        ]

        # 2. Apply 3D RoPE (MRoPE)
        cos_dynamic, sin_dynamic = self.make_dynamic_rope_caches(
            layer_id, basename=f"/model/layers.{layer_id}/attn/mrope_dynamic_cache"
        )

        # Apply rotation to Q
        self.attention_attrs["q_path"] = self.apply_mrope_rotation(
            layer_id,
            self.attention_attrs["q_path"],
            q_shape,
            cos_dynamic,
            sin_dynamic,
            self.num_attn_heads,
            basename=f"/model/layers.{layer_id}/attn/q_mrope",
        )

        # Apply rotation to K
        self.attention_attrs["k_path"] = self.apply_mrope_rotation(
            layer_id,
            self.attention_attrs["k_path"],
            k_shape,
            cos_dynamic,
            sin_dynamic,
            self.num_kv_heads,
            basename=f"/model/layers.{layer_id}/attn/k_mrope",
        )

        # 3. Call GroupQueryAttention op
        past_k = f"past_key_values.{layer_id}.key"
        past_v = f"past_key_values.{layer_id}.value"
        present_k = f"present.{layer_id}.key"
        present_v = f"present.{layer_id}.value"

        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        self.make_attention_op(
            attn_name,
            q_path=self.attention_attrs["q_path"],
            k_path=self.attention_attrs["k_path"],
            v_path=self.attention_attrs["v_path"],
            past_k=past_k,
            past_v=past_v,
            present_k=present_k,
            present_v=present_v,
            # Pass empty strings for fused caches since we applied RoPE manually
            cos_cache="",
            sin_cache="",
            **kwargs,
        )

    def load_weights(self, input_path):
        # For quantized models (e.g., Quark, AWQ, GPTQ) or GGUF, use base class logic
        # which loads weights directly via QuantModel
        if self.quant_type is not None or input_path.endswith(".gguf"):
            return super().load_weights(input_path)

        # For non-quantized models, load the Hugging Face model
        print("Loading Qwen2_5_VLForConditionalGeneration model...")
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
        )
