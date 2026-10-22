# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
import onnx_ir as ir


class Engine:
    """Engine graph, cache-layout, and runtime configuration expansions."""

    def make_paged_mrope_inputs(self, name, root_input, position_ids, dtype, num_heads):
        input_unsqueeze_name = f"{name}/input/Unsqueeze"
        self.make_unsqueeze(
            input_unsqueeze_name,
            [root_input, "/model/constants/INT64/[0]"],
            dtype,
            [1, "num_tokens", self.head_size * num_heads],
        )
        position_ids_unsqueeze_name = f"{name}/position_ids/Unsqueeze"
        self.make_unsqueeze(
            position_ids_unsqueeze_name,
            [position_ids, "/model/constants/INT64/[1]"],
            ir.DataType.INT64,
            [3, 1, "num_tokens"],
        )
        return (
            f"{input_unsqueeze_name}/output_0",
            f"{position_ids_unsqueeze_name}/output_0",
            f"{name}/rank3_output_0",
        )

    def make_paged_mrope_output(self, name, root_input, output, dtype, num_heads):
        self.make_node(
            "Squeeze",
            inputs=[root_input, "/model/constants/INT64/[0]"],
            outputs=[output],
            name=f"{name}/output/Squeeze",
        )
        self.make_value(output, dtype, shape=["num_tokens", self.head_size * num_heads])

    def prunes_hidden_rows(self):
        # The hidden_states output is the final norm's output, so it pins every row before the LM head.
        return self.use_paged_attention and self.prune_lm_head and not self.include_hidden_states

    def make_selected_hidden_rows(self):
        """Gather the residual-stream rows the LM head reads, so every later op runs on those rows only."""
        if self.hidden_rows_dim == "num_logits":
            return
        selected = {}
        for key in ("root_input", "skip_input"):
            name = self.layernorm_attrs[key]
            if name not in selected:
                gather_name = f"/model/selected_rows/{key}/Gather"
                self.make_gather(
                    gather_name,
                    [name, self.input_names["logits_indices"]],
                    dtype=self.values[name].dtype,
                    shape=["num_logits", self.hidden_size],
                    axis=0,
                )
                selected[name] = f"{gather_name}/output_0"
            self.layernorm_attrs[key] = selected[name]
        self.hidden_rows_dim = "num_logits"

    def make_decoder_state_groups(self, inputs, outputs):
        """Group paged decoder layers by the runtime state they maintain."""
        if not self.use_paged_attention:
            return []

        full_attention_layers = [
            layer_id
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type in {"full_attention", "sliding_attention"}
        ]
        conv_layers = [
            layer_id
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type in {"conv", "linear_attention"}
        ]
        recurrent_layers = [
            layer_id for layer_id, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"
        ]

        state_groups = []
        if full_attention_layers:
            state_groups.append(self.make_paged_key_value_state_group(full_attention_layers))
        if conv_layers:
            state_groups.append(
                {
                    "kind": "fixed_conv",
                    "layer_ids": conv_layers,
                }
            )
        if recurrent_layers:
            state_groups.append(
                {
                    "kind": "fixed_recurrent",
                    "layer_ids": recurrent_layers,
                }
            )
        return state_groups

    def make_paged_key_value_state_group(self, layer_ids):
        return {
            "kind": "paged_kv",
            "layer_ids": layer_ids,
        }

    def is_windowed_paged_layer(self, layer_id):
        """True when this layer's paged KV cache is a ring sized to the window rather than to the
        full context. Such layers read a different block table and a differently sized cache."""
        return self.has_windowed_paged_layers() and self.is_local(layer_id)

    def has_windowed_paged_layers(self):
        """True when at least one layer is actually served from the ring.

        A model can carry `sliding_window` in its config without the builder knowing which layers
        it applies to, in which case no layer reads the ring and nothing extra must be emitted.
        The runtime also needs at least one full-context layer to size the shared paged cache, so
        an all-local export falls back to full paged caches.
        """
        if (
            not self.context_length_attrs["window_kv_cache"]
            or not self.use_paged_attention
            or self.window_size is None
            or self.window_size <= 0
        ):
            return False
        local_layers = [self.is_local(layer_id) for layer_id in range(self.num_layers)]
        return any(local_layers) and not all(local_layers)

    def make_varlen_causal_conv_with_state(self, name, **kwargs):
        """Emit packed causal convolution with optional compact state updates."""
        inputs = [
            kwargs["root_input"],
            kwargs["weight"],
            kwargs["cumulative_sequence_length"],
            kwargs["bias"],
            kwargs["past_conv_state"],
        ]
        state_update_capacity = kwargs.get("state_update_capacity", 0)
        if state_update_capacity:
            inputs.append(kwargs["state_update_capture_count"])
            inputs.append(kwargs.get("state_update_active", ""))

        output = f"{name}/output_0"
        present_conv = kwargs["present_conv_state"]
        outputs = [output, present_conv]
        if state_update_capacity:
            outputs.append(kwargs["state_update_value"])

        attributes = {
            "activation": kwargs.get("activation", "silu"),
            "dilation": kwargs.get("dilation", 1),
        }
        if state_update_capacity:
            attributes["state_update_capacity"] = state_update_capacity
        self.make_node(
            "VarlenCausalConvWithState",
            inputs=inputs,
            outputs=outputs,
            name=name,
            domain="com.microsoft",
            **attributes,
        )
        self.make_value(output, self.io_dtype, shape=kwargs["output_shape"])
        self.make_value(present_conv, self.io_dtype, shape=kwargs["present_conv_shape"])
        if state_update_capacity:
            self.make_value(kwargs["state_update_value"], self.io_dtype, shape=kwargs["state_update_value_shape"])

    def make_varlen_gated_delta_net(self, name, **kwargs):
        """Emit packed GatedDeltaNet with optional compact transition capsules."""
        decay, beta = self.make_gated_delta_net_gates(name, kwargs)
        inputs = [
            kwargs["q_path"],
            kwargs["k_path"],
            kwargs["v_path"],
            kwargs["cumulative_sequence_length"],
            decay,
            beta,
            kwargs["past_recurrent_state"],
        ]
        a_log = kwargs.get("a_log", "")
        dt_bias = kwargs.get("dt_bias", "")
        if bool(a_log) != bool(dt_bias):
            raise ValueError("a_log and dt_bias must be set together")
        state_update_capacity = kwargs.get("state_update_capacity", 0)
        if a_log or state_update_capacity:
            inputs.extend([a_log, dt_bias])
        state_update_capture_count = kwargs.get("state_update_capture_count", "")
        state_update_active = kwargs.get("state_update_active", "")
        if bool(state_update_capture_count) != bool(state_update_active):
            raise ValueError("state_update_capture_count and state_update_active must be provided together")
        if state_update_capacity:
            if not state_update_capture_count:
                raise ValueError(
                    "state_update_capture_count and state_update_active are required when state_update_capacity is set"
                )
            inputs.extend([state_update_capture_count, state_update_active])

        output = f"{name}/output_0"
        present_recurrent = kwargs["present_recurrent_state"]
        outputs = [output, present_recurrent]
        if state_update_capacity:
            outputs.append(kwargs["state_update_capsule"])

        attributes = self.make_gated_delta_net_attributes(kwargs)
        if state_update_capacity:
            attributes["state_update_capacity"] = state_update_capacity
        self.make_node(
            "GatedDeltaNet",
            inputs=inputs,
            outputs=outputs,
            name=name,
            domain="com.microsoft",
            **attributes,
        )
        self.make_value(output, self.io_dtype, shape=kwargs["output_shape"])
        self.make_value(present_recurrent, ir.DataType.FLOAT, shape=kwargs["present_recurrent_shape"])
        if state_update_capacity:
            self.make_value(
                kwargs["state_update_capsule"],
                ir.DataType.FLOAT,
                shape=kwargs["state_update_capsule_shape"],
            )

    def make_paged_rotary_multi_cache_subgraph(self):
        # PagedAttention has no `attention_mask` input to derive the current sequence length from,
        # so approximate the effective context length from the block table:
        #
        #   block_table --> Shape --> Gather(idx=1) --> Multiply(x block_size) --> (context length)
        #
        # The result feeds the Greater node that selects the short vs. long rotary cache.
        basename = "/model/context_length_subgraph"
        shape_name = f"{basename}/Shape"
        shape_output = f"{shape_name}/output_0"
        self.make_shape(shape_name, "block_table", shape=[2])

        gather_name = f"{basename}/Gather"
        gather_output = f"{gather_name}/output_0"
        self.make_gather(
            gather_name, [shape_output, "/model/constants/INT64/1"], dtype=ir.DataType.INT64, shape=None, axis=0
        )

        # max_num_blocks (block_table dim 1) x block_size approximates the max representable
        # context length. block_size is a build-time constant (matches the engine's block_size).
        block_size = self.attention_attrs["paged_block_size"]
        mul_name = f"{basename}/Mul"
        mul_inputs = [gather_output, f"/model/constants/INT64/{block_size}"]
        self.make_mul(mul_name, mul_inputs, dtype=ir.DataType.INT64, shape=None)

    def make_paged_attention(self, name, **kwargs):
        inputs = [
            kwargs["q_path"],
            kwargs["k_path"],
            kwargs["v_path"],
            kwargs.get("past_k", ""),
            kwargs.get("past_v", ""),
            kwargs["cumulative_sequence_lengths"],
            kwargs["past_sequence_lengths"],
            kwargs["block_table"],
            kwargs.get("cos_cache", ""),
            kwargs.get("sin_cache", ""),
        ]

        q_norm_weight, k_norm_weight = self.get_qk_norm_weight_inputs(**kwargs)
        k_scale_name, v_scale_name = self.get_kv_cache_scale_inputs(**kwargs)

        # Optional trailing inputs of PagedAttention, in schema order:
        #   10: slot_mapping, 11: head_sink, 12: q_norm_weight, 13: k_norm_weight, 14: k_scale, 15: v_scale,
        #   16: attention_metadata
        # The scheduler-provided slot_mapping is not used here; slots are derived from
        # past_seqlens / cumulative_sequence_length / block_table by the op.
        # attention_metadata carries [max_query_len_bound, max_kv_len_bound, max_kv_len_lower_bound]
        # in CPU memory so the op can select a backend and size its launch without a device-to-host
        # readback of the sequence lengths. Feeding it is what removes the per-node stream
        # synchronization on the decode path.
        self.extend_with_optional_inputs(
            inputs,
            [
                "",  # slot_mapping
                kwargs.get("sinks", ""),  # head_sink
                q_norm_weight,
                k_norm_weight,
                k_scale_name,
                v_scale_name,
                kwargs.get("attention_metadata", ""),
            ],
        )

        output = f"{name}/output_0"
        outputs = [output, kwargs.get("present_k", ""), kwargs.get("present_v", "")]
        # PagedAttention derives the cache element type from the tensor itself, so unlike
        # GroupQueryAttention it has no `kv_cache_bit_width` attribute.
        attributes = self.get_attention_op_attributes(**kwargs)
        if self.kv_cache_attrs.get("bit_width", 0) == 4:
            attributes.update(k_cache_dtype="int4", v_cache_dtype="int4")
        rotation = self.kv_cache_attrs.get("rotation", "NONE")
        if rotation != "NONE":
            attributes.update(qk_rotation=rotation, v_rotation=rotation)
        if self.kv_cache_attrs.get("quant_mode") == "PER_TOKEN":
            layer_id = kwargs["layer_id"]
            inputs.extend([""] * (17 - len(inputs)))
            inputs.extend(self.input_names[f"past_key_values.{side}_scale"][layer_id] for side in ("key", "value"))
            outputs.extend(self.output_names[f"present.{side}_scale"][layer_id] for side in ("key", "value"))
        self.make_node(
            "PagedAttention",
            inputs=inputs,
            outputs=outputs,
            name=name,
            domain="com.microsoft",
            **attributes,
        )
        self.make_value(output, self.io_dtype, shape=["num_tokens", self.head_size * self.num_attn_heads])

    def make_paged_bindings_init(self):
        self.input_names.update(
            {
                "block_table": "block_table",
                "block_table_windowed": "block_table_windowed",
                "cumulative_sequence_lengths": "cumulative_sequence_lengths",
                "past_sequence_lengths": "past_sequence_lengths",
                "attention_metadata": "attention_metadata",
                "logits_indices": "logits_indices",
            }
        )
        self.input_types.update(
            {
                "block_table": ir.DataType.INT32,
                "block_table_windowed": ir.DataType.INT32,
                "cumulative_sequence_lengths": ir.DataType.INT32,
                "past_sequence_lengths": ir.DataType.INT32,
                "attention_metadata": ir.DataType.INT32,
                "logits_indices": ir.DataType.INT32,
            }
        )
        self.input_shapes.update(
            {
                "block_table": ["batch_size", "max_num_blocks"],
                "block_table_windowed": ["batch_size", "max_num_blocks"],
                "cumulative_sequence_lengths": ["batch_size + 1"],
                "past_sequence_lengths": ["batch_size"],
                "attention_metadata": [3],
                "logits_indices": ["num_logits"],
            }
        )

    def make_paged_inputs_init(self):
        if self.use_paged_attention:
            self.input_shapes["input_ids"] = ["num_tokens"]
            self.input_shapes["past_key_values.key"] = ["num_blocks", "block_size", self.num_kv_heads, self.head_size]
            self.input_shapes["past_key_values.value"] = ["num_blocks", "block_size", self.num_kv_heads, self.head_size]
            if "attention_mask" in self.input_names:
                del self.input_names["attention_mask"]
            if not self.has_windowed_paged_layers():
                del self.input_names["block_table_windowed"]
            if not self.extra_options.get("prune_lm_head", False):
                del self.input_names["logits_indices"]
        else:
            for name in [
                "block_table",
                "block_table_windowed",
                "cumulative_sequence_lengths",
                "past_sequence_lengths",
                "attention_metadata",
                "logits_indices",
            ]:
                del self.input_names[name]

    def make_paged_outputs_init(self):
        if self.use_paged_attention:
            self.output_shapes["present.key"] = ["num_blocks", "block_size", self.num_kv_heads, self.head_size]
            self.output_shapes["present.value"] = ["num_blocks", "block_size", self.num_kv_heads, self.head_size]
            self.output_shapes["hidden_states"] = ["num_tokens", self.hidden_size]
            logits_first_dim = "num_logits" if self.prune_lm_head else "num_tokens"
            self.output_shapes["logits"] = [logits_first_dim, self.vocab_size]

    def make_paged_attention_init(self):
        valid_paged_configurations = {
            ("cuda", ir.DataType.FLOAT16),
            ("cuda", ir.DataType.BFLOAT16),
            ("webgpu", ir.DataType.FLOAT16),
        }
        if (self.ep, self.io_dtype) not in valid_paged_configurations:
            raise NotImplementedError(
                "PagedAttention (use_paged_attention=true) requires CUDA with FP16/BF16 or WebGPU "
                f"with FP16, not ({self.ep}, {self.io_dtype})."
            )
        block_size = int(self.extra_options.get("paged_block_size", 256))
        self.attention_attrs["paged_block_size"] = block_size
        if "multi_cache" in self.rope_attrs and self.original_context_length % block_size != 0:
            raise ValueError(
                "paged_block_size must evenly divide original_max_position_embeddings "
                "for models that use short and long rotary caches."
            )
        self.attention_attrs["op_type"] = "PagedAttention"
        print("PagedAttention (PA) is used in this model.")

        # Packed Q/K/V MatMul is used unless disabled by LoRA/QLoRA or Q/K norm.
        self.attention_attrs["use_packed_matmul"] = (
            not self.matmul_attrs["use_lora"]
            and not self.attention_attrs["q_norm"]
            and not self.attention_attrs["k_norm"]
            and self.extra_options.get("fuse_qkv", not self.extra_options.get("disable_qkv_fusion", False))
        )

        # Some architectures require a separate RoPE op before PagedAttention.
        self.attention_attrs["use_rope_in_attn"] = self.is_fused_rope_supported()
        if self.attention_attrs["use_rope_in_attn"] and "position_ids" in self.input_names:
            del self.input_names["position_ids"]

    def make_paged_config_inputs(self, inputs):
        if self.use_paged_attention:
            inputs["block_table"] = self.input_names["block_table"]
            if self.has_windowed_paged_layers():
                inputs["block_table_windowed"] = self.input_names["block_table_windowed"]
            inputs["cumulative_sequence_lengths"] = self.input_names["cumulative_sequence_lengths"]
            inputs["past_sequence_lengths"] = self.input_names["past_sequence_lengths"]
            if "attention_metadata" in self.input_names:
                inputs["attention_metadata"] = self.input_names["attention_metadata"]
            if "logits_indices" in self.input_names:
                inputs["logits_indices"] = self.input_names["logits_indices"]

    def make_paged_window_config(self, genai_config):
        if self.has_windowed_paged_layers():
            # The runtime sizes the ring from `window_size` and the prefill chunk size, and
            # builds `block_table_windowed` by repeating each request's ring across the columns.
            # Only these layers read that table and the smaller `num_blocks_windowed` cache.
            genai_config["model"]["decoder"]["sliding_window"] = {
                "window_size": self.window_size,
                "slide_key_value_cache": False,
                "slide_inputs": False,
                "layers": [layer_id for layer_id in range(self.num_layers) if self.is_windowed_paged_layer(layer_id)],
                "cache_slack": 0,  # ring capacity comes from chunk_size, not from slack
            }
            # The ring only holds `chunk_size + window_size - 1` positions, so a prefill that
            # ran in one shot would overwrite positions it still had to attend to. Chunking is
            # not optional for this model, hence a default rather than an opt-in.
            genai_config["search"]["chunk_size"] = int(
                self.extra_options.get("paged_chunk_size", self.attention_attrs["paged_block_size"])
            )
        elif self.use_paged_attention and "paged_chunk_size" in self.extra_options:
            # Nothing forces chunking without a ring, but this caps one request where
            # max_scheduled_tokens only caps the step, so concurrent prefills interleave
            # instead of running one at a time.
            genai_config["search"]["chunk_size"] = int(self.extra_options["paged_chunk_size"])

    def make_paged_engine_config(self, genai_config):
        if self.use_paged_attention:
            dynamic_batching = {
                "block_size": self.attention_attrs["paged_block_size"],
                "max_batch_size": int(self.extra_options.get("max_batch_size", 100)),
            }
            if self.has_windowed_paged_layers():
                dynamic_batching["prefix_caching"] = False
            if "num_blocks" in self.extra_options:
                dynamic_batching["num_blocks"] = int(self.extra_options["num_blocks"])
            else:
                dynamic_batching["gpu_utilization_factor"] = float(
                    self.extra_options.get("gpu_utilization_factor", 0.6)
                )
            if "max_scheduled_tokens" in self.extra_options:
                dynamic_batching["max_scheduled_tokens"] = int(self.extra_options["max_scheduled_tokens"])
            genai_config["engine"] = {"dynamic_batching": dynamic_batching}

    def make_paged_lm_head(self, basename, root_input):
        seq_dim = "num_logits"
        if self.hidden_rows_dim != "num_logits":
            gather_name = f"{basename}/selected_hidden_states/Gather"
            gather_inputs = [root_input, self.input_names["logits_indices"]]
            self.make_gather(
                gather_name,
                gather_inputs,
                dtype=self.io_dtype,
                shape=["num_logits", self.hidden_size],
                axis=0,
            )
            root_input = f"{gather_name}/output_0"
        # Outputs built after the LM head, such as aux_hidden_states, keep every row.
        self.hidden_rows_dim = "num_tokens"
        self.output_shapes["logits"] = ["num_logits", self.vocab_size]
        return root_input, seq_dim
