# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Procedural batch-one Qwen Flash target graph for the native S4 executor."""

from dataclasses import dataclass

import numpy as np
from onnx import TensorProto as T
from onnx import helper, numpy_helper


@dataclass(frozen=True)
class Weight:
    name: str
    dtype: int
    shape: tuple
    source: str
    transform: str = "identity"


def validate_config(config):
    text = config.get("text_config", {})
    expected = {
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "hc_count": 4,
        "hc_lowrank": 320,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "ple_embed_dim": 2560,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "indexer_compress_ratio": 4,
        "indexer_budget": 2048,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "max_position_embeddings": 262144,
        "vocab_size": 248320,
        "ple_layer_ids": [2],
        "output_gate_type": "sigmoid",
    }
    if config.get("model_type") != "qwen4_exp":
        raise ValueError("The procedural target exporter requires model_type=qwen4_exp")
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f"Unsupported target text_config.{key}: expected {value!r}, got {text.get(key)!r}")
    layers = ["linear_attention" if i % 4 != 3 else "full_attention" for i in range(48)]
    if text.get("layer_types") != layers:
        raise ValueError("Unsupported target attention layer ordering")
    rope = text.get("rope_parameters", {})
    if rope.get("rope_theta") != 10000000 or rope.get("partial_rotary_factor") != 0.25:
        raise ValueError("Unsupported target rotary parameters")
    if text.get("rms_norm_eps") != 1e-6 or not text.get("norm_topk_prob"):
        raise ValueError("Unsupported target normalization parameters")
    groups = config.get("quantization_config", {}).get("config_groups", {}).values()
    if not any(
        group.get("weights", {}).get("num_bits") == 4 and group.get("weights", {}).get("group_size") == 16
        for group in groups
    ):
        raise ValueError("The procedural target exporter requires group-16 NVFP4 expert weights")
    return text


class TargetGraph:
    def __init__(self, config, *, calibration=False, calibration_capacity=4096):
        self.text = validate_config(config)
        if calibration_capacity < 4 or calibration_capacity > 262144:
            raise ValueError("Calibration capacity must be in [4, 262144]")
        self.calibration = calibration
        self.capacity = calibration_capacity if calibration else 262148
        self.index_capacity = (self.capacity + 3) // 4
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.constants = []
        self.weights = {}
        self._constant_names = {}
        self.status = []
        self.captures = []

    def const(self, values, dtype=np.int64):
        value = np.array(values, dtype=dtype)
        key = (value.dtype.str, value.shape, value.tobytes())
        if key not in self._constant_names:
            name = f"constant.{len(self.constants)}"
            self._constant_names[key] = name
            self.constants.append(numpy_helper.from_array(value, name))
        return self._constant_names[key]

    def io(self, name, dtype, shape, *, output=False):
        (self.outputs if output else self.inputs).append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def node(self, op, inputs, outputs, *, native=False, **attributes):
        if isinstance(outputs, str):
            outputs = [outputs]
        self.nodes.append(
            helper.make_node(
                op,
                inputs,
                outputs,
                name=next(name for name in outputs if name),
                domain="com.microsoft" if native else "",
                **attributes,
            )
        )
        return outputs[0]

    def weight(self, name, dtype, shape, source=None, transform="identity"):
        if source is None:
            source = name.replace("model.", "model.language_model.", 1)
        value = Weight(name, dtype, tuple(shape), source, transform)
        if name in self.weights and self.weights[name] != value:
            raise ValueError(f"Conflicting target weight declaration: {name}")
        self.weights[name] = value
        return name

    def dense(self, name, source, shape, transform="identity"):
        return self.weight(name, T.BFLOAT16, shape, source, transform)

    def projection(self, root, prefix, n, k, *, source=None, output=None, block=None):
        block = block or k
        source = source or prefix.replace("model.", "model.language_model.", 1) + ".weight"
        weight = self.weight(prefix + ".weight_t", T.FLOAT8E4M3FN, (n, k), source, f"fp8:{block}")
        scale = self.weight(prefix + ".weight_t.fp8_scale", T.FLOAT, (n, k // block), source, f"fp8_scale:{block}")
        inputs = [root, weight, scale]
        if prefix != "lm_head":
            inputs += ["", "", "prefill_numerics"]
        return self.node(
            "MatMulBlockQuantizedFp8Weight", inputs, output or prefix + ".out", native=True, block_size=block
        )

    def reshape(self, root, shape, name):
        return self.node("Reshape", [root, self.const(shape)], name)

    def slice(self, root, start, end, axis, name):
        return self.node("Slice", [root, self.const([start]), self.const([end]), self.const([axis])], name)

    def state(self, layer, suffix, dtype, shape, *, prefixes=True):
        tail = f"{layer}.{suffix}"
        name = self.io("past_key_values." + tail, dtype, shape)
        outputs = [self.io("present." + tail, dtype, shape, output=True)]
        if prefixes:
            for tag in ("present_mid.", "present_mid2.", "present_mid3."):
                outputs.append(self.io(tag + tail, dtype, shape, output=True))
        return name, outputs

    def hyper(self, streams, layer, location):
        prefix = f"model.layers.{layer}.{location}_hyper_connection"
        scale = self.weight(
            prefix + ".norm",
            T.BFLOAT16,
            (10240,),
            prefix.replace("model.", "model.language_model.", 1) + ".hc_norm.weight",
            "norm",
        )
        weights = [
            self.dense(
                prefix + "." + field + ".transposed",
                prefix.replace("model.", "model.language_model.", 1) + "." + field + ".weight",
                shape,
                "transpose",
            )
            for field, shape in (
                ("input_mix_weight_down", (10240, 320)),
                ("input_mix_weight_up", (320, 10240)),
                ("block_inject_weight", (10240, 4)),
            )
        ]
        self.node(
            "HyperConnectionLowRankMix",
            [streams, scale, *weights],
            [prefix + ".mixed", prefix + ".injection"],
            native=True,
            epsilon=1e-6,
            gate_div=4.0,
            gate_mul=2.0,
            num_branches=4,
            reduction_scale=0.25,
            silu_alpha=0.25,
        )
        return prefix + ".mixed", prefix + ".injection"

    def post_mix(self, streams, block, injection, name):
        return self.node("HyperConnectionPostMix", [streams, block, injection], name, native=True, num_branches=4)

    def linear_attention(self, root, layer):
        prefix = f"model.layers.{layer}.linear_attn"
        qkv = self.projection(root, prefix + ".in_proj_qkv", 10240, 2560)
        z = self.projection(root, prefix + ".in_proj_z", 6144, 2560)
        a = self.projection(root, prefix + ".in_proj_a", 48, 2560)
        b = self.projection(root, prefix + ".in_proj_b", 48, 2560)
        conv_weight = self.weight(prefix + ".conv1d.weight", T.BFLOAT16, (10240, 1, 4))
        past, states = self.state(layer, "conv_state", T.BFLOAT16, [1, 10240, 4])
        conv = self.node(
            "CausalConvWithState",
            [qkv, conv_weight, "", past],
            [prefix + ".conv", *states],
            native=True,
            ndim=1,
            channels_last=1,
            state_channels_last=0,
            state_length=4,
            input_channels=10240,
            activation="silu",
        )
        past, states = self.state(layer, "recurrent_state", T.FLOAT, [1, 48, 128, 128])
        a_log = self.weight(prefix + ".A_log", T.FLOAT, (48,))
        dt = self.weight(prefix + ".dt_bias", T.FLOAT, (48,))
        delta = self.node(
            "GatedDeltaNet",
            [conv, "", "", "", a, b, past, a_log, dt, "", "", "prefill_numerics"],
            [prefix + ".delta", *states],
            native=True,
            update_rule="gated_delta",
            gate_activation="softplus",
            beta_activation="sigmoid",
            qk_l2_norm=1,
            scale=0.0,
            state_output="mid_state",
        )
        delta = self.reshape(delta, [-1, 128], prefix + ".delta.flat")
        z = self.reshape(z, [-1, 128], prefix + ".z.flat")
        norm = self.weight(prefix + ".norm.weight", T.BFLOAT16, (128,))
        result = self.node(
            "GatedRMSNorm", [delta, norm, z], prefix + ".gated", native=True, epsilon=1e-6, activation="sigmoid"
        )
        result = self.reshape(result, [1, -1, 6144], prefix + ".heads")
        return self.projection(result, prefix + ".out_proj", 2560, 6144)

    def ple(self, streams):
        prefix = "model.layers.1.ple"
        embedding = self.io("ple_embeddings.1", T.BFLOAT16, [1, "sequence_length", 2560])
        key = self.projection(embedding, prefix + ".key_proj", 10240, 2560)
        value = self.projection(embedding, prefix + ".value_proj", 10240, 2560)
        normalized = []
        for root, field in ((key, "norm_key"), (streams, "norm_query")):
            scale = self.weight(
                prefix + "." + field + ".scale",
                T.BFLOAT16,
                (10240,),
                prefix.replace("model.", "model.language_model.", 1) + "." + field + ".weight",
                "norm",
            )
            normalized.append(
                self.node(
                    "BranchwiseRMSNorm", [root, scale], prefix + "." + field, native=True, num_branches=4, epsilon=1e-6
                )
            )
        gated = self.node(
            "DotProductGate",
            [*normalized, value],
            prefix + ".gated",
            native=True,
            divisor=2560**0.5,
            epsilon=1e-6,
            group_width=2560,
        )
        scale = self.weight(
            prefix + ".norm_conv.scale",
            T.BFLOAT16,
            (10240,),
            prefix.replace("model.", "model.language_model.", 1) + ".norm_conv.weight",
            "norm",
        )
        norm = self.node(
            "BranchwiseRMSNorm", [gated, scale], prefix + ".norm_conv", native=True, num_branches=4, epsilon=1e-6
        )
        weight = self.weight(prefix + ".conv1d.weight", T.BFLOAT16, (10240, 1, 4))
        past, states = self.state(1, "ple_conv_state", T.BFLOAT16, [1, 10240, 9])
        conv = self.node(
            "CausalConvWithState",
            [norm, weight, "", past],
            [prefix + ".conv", *states],
            native=True,
            ndim=1,
            channels_last=1,
            state_channels_last=0,
            state_length=9,
            input_channels=10240,
            dilation=3,
            activation="silu",
        )
        past, states = self.state(1, "ple_context", T.INT64, [1, 2])
        tokens = self.node("Concat", [past, "input_ids"], prefix + ".tokens", axis=1)
        self.slice(tokens, -2, 9223372036854775807, 1, states[0])
        for count, output in enumerate(states[1:], 1):
            first = self.slice("input_ids", 0, count, 1, prefix + f".prefix.{count}.tokens")
            joined = self.node("Concat", [past, first], prefix + f".prefix.{count}.context", axis=1)
            self.slice(joined, -2, 9223372036854775807, 1, output)
        added = self.node("Add", [gated, conv], prefix + ".update")
        return self.node("Add", [streams, added], prefix + ".streams")

    def sparse_attention(self, root, layer):
        prefix = f"model.layers.{layer}.self_attn"
        q = self.projection(root, prefix + ".q_proj", 12288, 2560)
        k = self.projection(root, prefix + ".k_proj", 512, 2560)
        v = self.projection(root, prefix + ".v_proj", 512, 2560)
        q_heads = self.reshape(q, [1, -1, 24, 512], prefix + ".q_heads")
        k_heads = self.reshape(k, [1, -1, 2, 256], prefix + ".k_heads")
        rotated = []
        for heads, field in ((q_heads, "q_norm"), (k_heads, "k_norm")):
            scale = self.weight(
                prefix + "." + field + ".scale",
                T.FLOAT,
                (256,),
                prefix.replace("model.", "model.language_model.", 1) + "." + field + ".weight",
                "norm_float",
            )
            rotated.append(
                self.node(
                    "RMSNormRotary",
                    [heads, scale, "position_ids", "rope.cos", "rope.sin", "prefill_numerics"],
                    prefix + "." + field,
                    native=True,
                    epsilon=1e-6,
                    flatten_batch=1,
                    input_start=0,
                    input_width=256,
                    rotary_embedding_dim=64,
                    use_fp32_arithmetic=1,
                )
            )
        index = self.projection(root, prefix + ".indexer.index_qk_proj", 640, 2560)
        index = self.reshape(index, [-1, 640], prefix + ".index.flat")
        index_norms = [
            self.weight(
                prefix + ".indexer." + field + ".scale",
                T.BFLOAT16,
                (128,),
                prefix.replace("model.", "model.language_model.", 1) + ".indexer." + field + ".weight",
                "norm",
            )
            for field in ("q_layernorm", "k_layernorm")
        ]
        index_key, index_key_out = self.state(
            layer, "index_key", T.FLOAT, [1, self.index_capacity, 128], prefixes=False
        )
        buffer, buffer_out = self.state(layer, "index_buffer", T.BFLOAT16, [1, 7, 128])
        lengths, lengths_out = self.state(layer, "index_lengths", T.INT32, [1, 2])
        scoring = self.io(
            f"prefill_cache.past_key_values.{layer}.index_key.scoring", T.FLOAT, [1, "prefill_scoring_capacity", 128]
        )
        scoring_out = self.io(
            f"prefill_cache_out.past_key_values.{layer}.index_key.scoring",
            T.FLOAT,
            [1, "prefill_scoring_capacity", 128],
            output=True,
        )
        status = prefix + ".index.status"
        self.status.append(status)
        self.node(
            "PackedSparseAttentionIndexer",
            [
                index,
                "",
                *index_norms,
                "index_rope.cos",
                "index_rope.sin",
                "cumulative_sequence_lengths",
                "committed_length",
                "",
                "",
                "",
                "positions.flat",
                index_key,
                buffer,
                "",
                lengths,
                "",
                "",
                "",
                "",
                "",
                "",
                "prefill_numerics",
                scoring,
            ],
            [
                prefix + ".selected",
                prefix + ".counts",
                *index_key_out,
                buffer_out[0],
                "",
                lengths_out[0],
                "",
                "",
                buffer_out[1],
                lengths_out[1],
                status,
                buffer_out[2],
                lengths_out[2],
                buffer_out[3],
                lengths_out[3],
                scoring_out,
            ],
            native=True,
            compress_ratio=4,
            epsilon=1e-6,
            policy_mode="qsa",
            scale=128**-0.5,
            state_capacity=self.index_capacity,
            token_budget=2048,
        )
        selected = self.reshape(prefix + ".selected", [-1, 2051], prefix + ".selected.flat")
        counts = self.reshape(prefix + ".counts", [-1], prefix + ".counts.flat")
        dtype = T.BFLOAT16 if self.calibration else T.INT8
        caches, cache_outputs, numerical, numerical_out = [], [], [], []
        scales = []
        for field in ("key", "value"):
            cache, outputs = self.state(layer, field, dtype, [1, 2, self.capacity, 256], prefixes=False)
            caches.append(cache)
            cache_outputs.extend(outputs)
            if not self.calibration:
                numerical.append(
                    self.io(
                        f"prefill_cache.past_key_values.{layer}.{field}",
                        T.BFLOAT16,
                        [1, 2, "prefill_cache_capacity", 256],
                    )
                )
                numerical_out.append(
                    self.io(
                        f"prefill_cache_out.past_key_values.{layer}.{field}",
                        T.BFLOAT16,
                        [1, 2, "prefill_cache_capacity", 256],
                        output=True,
                    )
                )
            scales.append(
                ""
                if self.calibration
                else self.weight(
                    prefix + "." + field + ".cache_scale", T.FLOAT, (512,), "", f"cache_scale:{layer}:{field}"
                )
            )
        status = prefix + ".attention.status"
        self.status.append(status)
        attention = self.node(
            "DynamicSparseAttention",
            [
                *rotated,
                v,
                *caches,
                "",
                "",
                selected,
                counts,
                "seqlens_k",
                self.const([self.capacity], np.int32),
                "",
                "",
                "",
                "",
                "",
                "",
                "committed_length",
                *scales,
                self.const([0], np.int32) if self.calibration else "prefill_numerics",
                *numerical,
            ],
            [prefix + ".attention", *cache_outputs, status, *numerical_out],
            native=True,
            attention_mode="selected_only",
            is_causal=1,
            kv_num_heads=2,
            num_heads=24,
            scale=0.0625,
            selected_kv_source="main",
        )
        gated = self.node(
            "SigmoidGateMul",
            [attention, q, "", "prefill_numerics"],
            prefix + ".gated",
            native=True,
            head_dim=256,
            num_heads=24,
            q_offset=0,
            q_stride=512,
            tok_stride=12288,
        )
        if self.calibration:
            for field, root in (("key", rotated[1]), ("value", v)):
                value = self.node("Cast", [root], prefix + "." + field + ".fp32", to=T.FLOAT)
                value = self.node("Abs", [value], prefix + "." + field + ".abs")
                name = f"calibration.{layer}.{field}.amax"
                self.node("ReduceMax", [value, self.const([0, 1])], name, keepdims=0)
                self.io(name, T.FLOAT, [512], output=True)
        return self.projection(gated, prefix + ".o_proj", 2560, 6144)

    def moe(self, root, layer):
        prefix = f"model.layers.{layer}.mlp"
        router = self.projection(root, prefix + ".gate", 512, 2560)
        router = self.reshape(router, [-1, 512], prefix + ".router")
        weights = []
        for field, n, k in (("fc1", 1280, 2560), ("fc2", 2560, 640)):
            for suffix, dtype, shape in (
                ("experts_weights", T.UINT8, (512, n, k // 4)),
                ("scales", T.BFLOAT16, (512, n, k // 64)),
            ):
                weights.append(
                    self.weight(
                        prefix + "." + field + "_" + suffix, dtype, shape, str(layer), f"experts:{field}:{suffix}"
                    )
                )
        routed = self.node(
            "QMoE",
            [root, router, weights[0], weights[1], "", weights[2], weights[3]],
            prefix + ".routed",
            native=True,
            activation_type="swiglu",
            normalize_routing_weights=1,
            k=10,
            expert_weight_bits=2,
            block_size=64,
            swiglu_fusion=1,
            quant_type="int",
            weights_prepacked=1,
        )
        gate = self.projection(root, prefix + ".shared_expert.gate_proj", 640, 2560)
        up = self.projection(root, prefix + ".shared_expert.up_proj", 640, 2560)
        sigmoid = self.node("Sigmoid", [gate], prefix + ".shared.sigmoid")
        silu = self.node("Mul", [gate, sigmoid], prefix + ".shared.silu")
        activated = self.node("Mul", [silu, up], prefix + ".shared.activated")
        down = self.projection(activated, prefix + ".shared_expert.down_proj", 2560, 640)
        shared_gate = self.dense(
            prefix + ".shared_expert_gate.weight_t",
            prefix.replace("model.", "model.language_model.", 1) + ".shared_expert_gate.weight",
            (2560, 1),
            "transpose",
        )
        gate = self.node("MatMul", [root, shared_gate], prefix + ".shared.gate")
        gate = self.node("Sigmoid", [gate], prefix + ".shared.gate.sigmoid")
        shared = self.node("Mul", [down, gate], prefix + ".shared.output")
        return self.node("Add", [routed, shared], prefix + ".output")

    def build(self):
        for name, dtype, shape in (
            ("input_ids", T.INT64, [1, "sequence_length"]),
            ("position_ids", T.INT64, [1, "sequence_length"]),
            ("attention_mask", T.INT64, [1, "sequence_length"]),
            ("past_position_ids", T.INT64, [1, self.capacity]),
            ("logits_indices", T.INT64, ["logits_rows"]),
            ("committed_length", T.INT32, [1]),
            ("sequence_length_minus_one", T.INT32, [1]),
            ("cumulative_sequence_lengths", T.INT32, [2]),
            ("prefill_numerics", T.INT32, [1]),
        ):
            self.io(name, dtype, shape)
        self.node("Identity", ["past_position_ids"], "present_position_ids")
        self.io("present_position_ids", T.INT64, [1, self.capacity], output=True)
        self.reshape("position_ids", [-1], "positions.flat")
        self.node("Add", ["committed_length", "sequence_length_minus_one"], "seqlens_k")
        for prefix, width in (("rope", 32), ("index_rope", 64)):
            for field in ("cos", "sin"):
                self.weight(prefix + "." + field, T.BFLOAT16, (self.capacity, width), "", f"rope:{field}:{width}")
        embedding = self.weight("model.embed_tokens.weight", T.BFLOAT16, (248320, 2560))
        hidden = self.node("Gather", [embedding, "input_ids"], "embedding", axis=0)
        hidden = self.node("Concat", [hidden] * 4, "streams.initial", axis=-1)
        for layer in range(48):
            if layer == 1:
                hidden = self.ple(hidden)
            mixed, inject = self.hyper(hidden, layer, "attn")
            attention = self.sparse_attention(mixed, layer) if layer % 4 == 3 else self.linear_attention(mixed, layer)
            hidden = self.post_mix(hidden, attention, inject, f"layer.{layer}.attention.streams")
            mixed, inject = self.hyper(hidden, layer, "mlp")
            result = self.moe(mixed, layer)
            hidden = self.post_mix(hidden, result, inject, "hidden_states" if layer == 47 else f"layer.{layer}.streams")
            if layer in (5, 15, 25, 35, 45):
                streams = self.reshape(hidden, [1, -1, 4, 2560], f"capture.{layer}.streams")
                fp32 = self.node("Cast", [streams], f"capture.{layer}.fp32", to=T.FLOAT)
                mean = self.node("ReduceMean", [fp32, self.const([2])], f"capture.{layer}.mean", keepdims=0)
                self.captures.append(self.node("Cast", [mean], f"capture.{layer}.bf16", to=T.BFLOAT16))
        self.io("hidden_states", T.BFLOAT16, [1, "sequence_length", 10240], output=True)
        self.node("Concat", self.captures, "aux_hidden_states", axis=-1)
        self.io("aux_hidden_states", T.BFLOAT16, [1, "sequence_length", 12800], output=True)
        prefix = "model.hyper_connection_mixer"
        scale = self.weight(
            prefix + ".norm", T.BFLOAT16, (10240,), "model.language_model.hyper_connection_mixer.hc_norm.weight", "norm"
        )
        norm = self.node(
            "BranchwiseRMSNorm", [hidden, scale], prefix + ".normed", native=True, num_branches=4, epsilon=1e-6
        )
        down = self.projection(norm, prefix + ".input_mix_weight_down", 320, 10240)
        silu = self.node("ScaledSiLU", [down], prefix + ".silu", native=True, alpha=0.25)
        up = self.projection(silu, prefix + ".input_mix_weight_up", 10240, 320)
        mixed = self.node(
            "HyperConnectionPreMix",
            [norm, up],
            prefix + ".mixed",
            native=True,
            num_branches=4,
            reduction_scale=0.25,
            sigmoid_mode=2,
        )
        tail = self.slice(mixed, -4, 9223372036854775807, 1, "logits.tail")
        selected = self.node("Gather", [tail, "logits_indices"], "logits.hidden", axis=1)
        self.projection(selected, "lm_head", 248320, 2560, source="lm_head.weight", output="logits", block=128)
        self.io("logits", T.BFLOAT16, [1, "logits_rows", 248320], output=True)
        self.node("ArgMaxLastAxis", ["logits"], "next_tokens", native=True, keepdims=0)
        self.io("next_tokens", T.INT64, [1, "logits_rows"], output=True)
        self.node("Concat", self.status, "validation.aggregate", axis=0)
        self.io("validation.aggregate", T.INT32, [24], output=True)
        if not self.calibration:
            self.fold()
        model = helper.make_model(
            helper.make_graph(self.nodes, "QwenFlashProceduralTarget", self.inputs, self.outputs, self.constants),
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
            producer_name="onnxruntime-genai.qwen38_graph",
        )
        model.ir_version = 10
        helper.set_model_props(
            model,
            {
                "qwen38.exporter": "procedural-v1",
                "experiment.validation_status_names": str(
                    [f"layer.{layer}.{kind}" for layer in range(3, 48, 4) for kind in ("index", "attention")]
                ).replace("'", '"'),
            },
        )
        return model

    def fold(self):
        for name, shape in (
            ("fold.ctl", [16]),
            ("fold.ctx_commit", [2]),
            ("fold.ctx_a", [8]),
            ("fold.ctx_b", [2]),
            ("fold.ptrs", [2]),
        ):
            self.io(name, T.INT64, shape)
        ids = self.slice("input_ids", -4, 9223372036854775807, 1, "fold.ids")
        hidden = self.slice("hidden_states", -4, 9223372036854775807, 1, "fold.hidden")
        outputs = [
            "fold.head_ids",
            "fold.head_c",
            "fold.slots",
            "fold.lengths",
            "fold.cos",
            "fold.sin",
            "fold.selection",
            "fold.head_hidden",
            "fold.mid_ctl",
            "fold.committed",
        ]
        self.node(
            "FoldMid2",
            [
                "next_tokens",
                ids,
                "validation.aggregate",
                "fold.ctl",
                "fold.ctx_commit",
                "fold.ctx_a",
                "fold.ctx_b",
                "fold.ptrs",
                "rope.cos",
                "rope.sin",
                hidden,
            ],
            outputs,
            native=True,
            head_rows=4,
            head_capacity=self.capacity,
        )
        self.node(
            "FoldTail",
            ["fold.selection", self.const([0], np.int32), "fold.mid_ctl", "fold.ptrs"],
            ["fold.tail_done", "fold.tail_ctl"],
            native=True,
            ctl_slot=2,
        )
        self.io("fold.tail_done", T.INT32, [1], output=True)


def build_target_graph(config, **kwargs):
    builder = TargetGraph(config, **kwargs)
    model = builder.build()
    return model, list(builder.weights.values())
