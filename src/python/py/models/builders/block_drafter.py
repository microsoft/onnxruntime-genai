# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import tempfile

import numpy as np
import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import TorchTensor, to_torch_dtype
from quantization import CudaQuantizer
from tqdm import tqdm

from .base import Model


# Deliberately not a `Model` subclass: a block drafter emits a hand-written graph, so it shares
# only the two stateless `Model` helpers below rather than the decoder builder's contract.
class BlockDrafterBuilder:
    """Shared graph, initializer, I/O, and file plumbing for block drafters."""

    # Weight-only quantization of the drafter body. ``quant_bits = None`` keeps every
    # projection dense; subclasses set these from the target's quantization settings so the
    # drafter lands in the same format as the model it drafts for.
    quant_bits = None
    quant_block_size = 32
    quant_prepack = 0
    # Set only when the target's own LM head is symmetric/`default` quantized, which is the one
    # convention whose initializer names and bytes the drafter can reproduce and share.
    lm_head_quant = None

    def make_graph(self, graph_name, const_prefix):
        self.values: dict[str, ir.Value] = {}
        self.node_names: set[str] = set()
        self.graph = ir.Graph(
            inputs=(),
            outputs=(),
            nodes=(),
            opset_imports={"": 21, "com.microsoft": 1},
            name=graph_name,
        )
        self.model = ir.Model(self.graph, ir_version=10, producer_name="onnxruntime-genai")
        self.const_cache: dict[str, str] = {}
        self.const_prefix = const_prefix

    def make_value(self, name, dtype=None, shape=None):
        if name == "":
            return ir.Value(name="")
        value = self.values.setdefault(name, ir.Value(name=name))
        if dtype is not None:
            value.dtype = ir.DataType(dtype)
        if shape is not None:
            value.shape = ir.Shape(shape)
        return value

    def make_node(self, op_type, inputs, outputs, *, name, domain="", **attrs):
        if name in self.node_names:
            raise ValueError(f"duplicate node name {name}")
        node = ir.node(
            op_type,
            inputs=[self.make_value(input_name) for input_name in inputs],
            attributes=attrs,
            domain=domain,
            outputs=[self.make_value(output_name) for output_name in outputs],
            name=name,
        )
        self.graph.append(node)
        self.node_names.add(name)

    def make_initializer(self, tensor, name, to=None):
        if to is not None:

            def tensor_func(value=tensor, dtype=to):
                return TorchTensor(value.to(to_torch_dtype(dtype)).contiguous(), name=name)

            ir_tensor = ir.LazyTensor(tensor_func, dtype=to, shape=ir.Shape(tensor.shape), name=name)
        elif isinstance(tensor, torch.Tensor):
            ir_tensor = TorchTensor(tensor.contiguous(), name=name)
        else:
            ir_tensor = ir.tensor(tensor, name=name)
        value = self.make_value(name, ir_tensor.dtype, ir_tensor.shape)
        value.const_value = ir_tensor
        self.graph.register_initializer(value)
        return name

    def const(self, values, dtype=ir.DataType.INT64):
        """Emit a small Constant node once so shape inference can read it."""
        array = np.asarray(values)
        key = f"{dtype}:{array.shape}:{array.tobytes().hex()}"
        if key in self.const_cache:
            return self.const_cache[key]
        name = f"{self.const_prefix}.const.{len(self.const_cache)}"
        np_dtype = {
            ir.DataType.INT64: np.int64,
            ir.DataType.INT32: np.int32,
            ir.DataType.FLOAT: np.float32,
        }[dtype]
        tensor = ir.tensor(array.astype(np_dtype), name=name)
        self.make_node("Constant", [], [name], name=f"{name}/Constant", value=tensor)
        self.make_value(name, dtype, tensor.shape)
        self.const_cache[key] = name
        return name

    def out(self, name):
        return f"{name}/output_0"

    def unary(self, op, name, root_input, dtype, shape, domain="", **attrs):
        output = self.out(name)
        self.make_node(op, [root_input], [output], name=name, domain=domain, **attrs)
        self.make_value(output, dtype, shape)
        return output

    def binary(self, op, name, lhs, rhs, dtype, shape):
        output = self.out(name)
        self.make_node(op, [lhs, rhs], [output], name=name)
        self.make_value(output, dtype, shape)
        return output

    def reshape(self, name, root_input, shape_const, dtype, shape):
        return self.binary("Reshape", name, root_input, self.const(shape_const), dtype, shape)

    def matmul(self, name, root_input, weight_tensor, in_features, out_features, rows, weight_name=None, quantize=True):
        """Emit ``root_input @ weight.T`` for a torch ``[out, in]`` weight."""
        expected_shape = (out_features, in_features)
        if tuple(weight_tensor.shape) != expected_shape:
            raise ValueError(f"Weight for '{name}' has shape {tuple(weight_tensor.shape)}, expected {expected_shape}.")
        initializer_name = weight_name or (name[1:].replace("/", ".") + ".weight")
        if quantize and self.quant_bits and in_features % self.quant_block_size == 0:
            return self.matmul_nbits(name, root_input, weight_tensor, in_features, out_features, rows, initializer_name)
        if initializer_name not in self.values:
            self.make_initializer(weight_tensor.T, initializer_name, to=self.io_dtype)
        output = self.out(name)
        self.make_node("MatMul", [root_input, initializer_name], [output], name=name)
        self.make_value(output, self.io_dtype, [rows, out_features])
        return output

    def matmul_nbits(self, name, root_input, weight_tensor, in_features, out_features, rows, initializer_name):
        """Emit a weight-only quantized ``MatMulNBits`` for a torch ``[out, in]`` weight.

        ``MatMulNBits`` consumes ``[N, K]`` directly, so unlike the dense path the weight is
        not transposed. Repeat call sites reuse the initializer the first one registered.
        """
        # The prepacked fpA_intB kernel takes FP16 activations only, so a bf16 body has to ship
        # the plain blockwise layout even when the target it drafts for is prepacked.
        prepack = self.quant_prepack if self.io_dtype == ir.DataType.FLOAT16 else 0
        qweight_name = f"{initializer_name}_Q{self.quant_bits}"
        scales_name = f"{initializer_name}_scales"
        if qweight_name not in self.values:
            if prepack:
                qweight, scales = CudaQuantizer.matmulnbits_prepacked_blockwise_quantize(
                    weight_tensor,
                    self.quant_bits,
                    self.quant_block_size,
                    force_arch=90 if prepack == 2 else 80,
                )
            else:
                qweight, scales = CudaQuantizer.matmulnbits_blockwise_quantize(
                    weight_tensor, self.quant_bits, self.quant_block_size, flatten_qweight=False
                )
            self.make_initializer(qweight, qweight_name)
            self.make_initializer(scales, scales_name, to=self.io_dtype)
        attributes = {
            "bits": self.quant_bits,
            "block_size": self.quant_block_size,
            "K": in_features,
            "N": out_features,
        }
        if prepack:
            attributes["weight_prepacked"] = prepack
        output = self.out(name)
        self.make_node(
            "MatMulNBits",
            [root_input, qweight_name, scales_name],
            [output],
            name=name,
            domain="com.microsoft",
            **attributes,
        )
        self.make_value(output, self.io_dtype, [rows, out_features])
        return output

    def rms_norm(self, name, root_input, weight_tensor, rows, weight_name=None):
        initializer_name = weight_name or (name[1:].replace("/", ".") + ".weight")
        if initializer_name not in self.values:
            self.make_initializer(weight_tensor, initializer_name, to=self.io_dtype)
        output = self.out(name)
        self.make_node(
            "SimplifiedLayerNormalization",
            [root_input, initializer_name],
            [output, "", ""],
            name=name,
            axis=-1,
            epsilon=self.rms_eps,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, [rows, self.hidden_size])
        return output

    def skip_rms_norm(self, name, root, skip, weight_tensor, rows, want_sum=True):
        initializer_name = name[1:].replace("/", ".") + ".weight"
        self.make_initializer(weight_tensor, initializer_name, to=self.io_dtype)
        output = self.out(name)
        summed_output = f"{name}/output_3"
        self.make_node(
            "SkipSimplifiedLayerNormalization",
            [root, skip, initializer_name],
            [output, "", "", summed_output] if want_sum else [output],
            name=name,
            domain="com.microsoft",
            epsilon=self.rms_eps,
        )
        self.make_value(output, self.io_dtype, [rows, self.hidden_size])
        if want_sum:
            self.make_value(summed_output, self.io_dtype, [rows, self.hidden_size])
        return output, (summed_output if want_sum else None)

    def make_lm_head(self, root, rows="num_block"):
        weight = self.weights["lm_head.weight"]
        if str(weight.dtype).startswith("torch.float8") and weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"FP8 LM head weight must be float8_e4m3fn, got {weight.dtype}.")
        expected_shape = (self.vocab_size, self.hidden_size)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(f"LM head weight has shape {tuple(weight.shape)}, expected {expected_shape}.")

        if self.io_dtype != self.external_dtype:
            root = self.unary(
                "Cast", "/lm_head/Cast", root, self.external_dtype, [rows, self.hidden_size], to=self.external_dtype
            )

        name = "/lm_head/MatMul"
        output = self.out(name)
        if self.lm_head_quant is not None:
            self.make_lm_head_nbits(name, root, output, weight)
        elif weight.dtype != torch.float8_e4m3fn:
            self.make_initializer(weight.T, "lm_head.MatMul.weight", to=self.external_dtype)
            self.make_node("MatMul", [root, "lm_head.MatMul.weight"], [output], name=name)
        else:
            weight_scale = self.weights.get("lm_head.weight_scale")
            if weight_scale is None:
                raise ValueError("FP8 LM head weight is missing 'lm_head.weight_scale'.")
            scale = Model.prepare_matmul_block_quantized_scales(weight_scale, self.vocab_size, 1)
            if scale is None:
                raise ValueError(
                    f"FP8 LM head weight scale has shape {tuple(weight_scale.shape)}, "
                    f"expected a scalar or [{self.vocab_size}, 1]."
                )
            self.make_initializer(weight.contiguous(), "lm_head.MatMul.fp8_weight")
            self.make_initializer(scale, "lm_head.MatMul.fp8_weight_scale", to=ir.DataType.FLOAT)
            self.make_node(
                "MatMulBlockQuantizedFp8Weight",
                [root, "lm_head.MatMul.fp8_weight", "lm_head.MatMul.fp8_weight_scale"],
                [output],
                name=name,
                domain="com.microsoft",
                block_size=int(weight.shape[1]),
            )
        self.make_value(output, self.external_dtype, [rows, self.vocab_size])
        return output

    def make_lm_head_nbits(self, name, root, output, weight):
        """Emit the LM head under the *target's* initializer names so the two fold into one copy.

        The drafter's head is the target's `lm_head.weight`, so quantizing it the same way
        reproduces the target's bytes and `share_initializers` collapses them. Scales stay at
        `external_dtype` (the target's IO dtype), not the drafter's bf16 body dtype, because a
        byte difference there would silently cost a duplicated copy instead of failing.
        """
        bits = self.lm_head_quant["bits"]
        block_size = self.lm_head_quant["block_size"]
        prepack = self.lm_head_quant["prepack"]
        if prepack:
            qweight, scales = CudaQuantizer.matmulnbits_prepacked_blockwise_quantize(
                weight, bits, block_size, force_arch=90 if prepack == 2 else 80
            )
        else:
            qweight, scales = CudaQuantizer.matmulnbits_blockwise_quantize(
                weight, bits, block_size, flatten_qweight=False
            )
        qweight_name = f"lm_head.MatMul.weight_Q{bits}"
        scales_name = "lm_head.MatMul.weight_scales"
        self.make_initializer(qweight, qweight_name)
        self.make_initializer(scales, scales_name, to=self.external_dtype)
        attributes = {"bits": bits, "block_size": block_size, "K": self.hidden_size, "N": self.vocab_size}
        if prepack:
            attributes["weight_prepacked"] = prepack
        self.make_node(
            "MatMulNBits",
            [root, qweight_name, scales_name],
            [output],
            name=name,
            domain="com.microsoft",
            **attributes,
        )

    def resolve_sliding_window(self, config):
        """Return the single ``local_window_size`` every layer runs with, or -1 for full attention."""
        if not config.get("use_sliding_window"):
            return -1
        window = int(config["sliding_window"])
        # PagedAttention takes one window per model, so a checkpoint that windows only a suffix of
        # its layers (HF's `max_window_layers`, or a mixed `layer_types`) cannot be exported as-is.
        max_window_layers = int(config.get("max_window_layers", 0))
        if max_window_layers not in (0, self.num_layers):
            raise ValueError(
                "A windowed block drafter must set max_window_layers to 0 or num_hidden_layers; "
                "mixed layer windows are not supported."
            )
        configured_layer_types = config.get("layer_types")
        if configured_layer_types is not None and len(configured_layer_types) != self.num_layers:
            raise ValueError(
                f"A windowed block drafter must declare one layer type per hidden layer; got "
                f"{len(configured_layer_types)} for {self.num_layers} layers."
            )
        layer_types = set(configured_layer_types or ["sliding_attention"])
        if layer_types != {"sliding_attention"}:
            raise ValueError(
                "A windowed block drafter must use sliding attention on every layer, got "
                + ", ".join(sorted(layer_types))
                + "."
            )
        return window

    def validate_token_metadata(self, config, drafter_config):
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError(f"mask_token_id must be between 0 and vocabulary size - 1 ({self.vocab_size - 1}).")
        unsupported_id_maps = {
            "draft_to_target_id_map",
            "draft_to_target_map",
            "draft_to_target_mapping",
            "token_id_map",
            "token_mapping",
            "vocab_mapping",
        }
        configured_id_maps = sorted(
            unsupported_id_maps.intersection(config) | unsupported_id_maps.intersection(drafter_config)
        )
        if configured_id_maps:
            raise ValueError(
                "Block-drafter token ID remapping is not implemented; remove " + ", ".join(configured_id_maps) + "."
            )

    def validate_weight_shapes(self, weights, expected_shapes):
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(weights[name].shape)
            if actual_shape != expected_shape:
                raise ValueError(f"Weight '{name}' has shape {actual_shape}, expected {expected_shape}.")

    def declare_io(self):
        declarations = [
            ("aux_hidden_states", self.external_dtype, ["num_ctx", self.aux_hidden_size]),
            ("input_ids", ir.DataType.INT64, ["num_block"]),
            ("q_row_map", ir.DataType.INT32, ["num_tokens"]),
            ("qkv_row_map", ir.DataType.INT32, ["num_tokens"]),
            ("block_row_index", ir.DataType.INT32, ["num_block"]),
            ("cumulative_sequence_lengths", ir.DataType.INT32, ["batch_size + 1"]),
            ("past_sequence_lengths", ir.DataType.INT32, ["batch_size"]),
            ("block_table", ir.DataType.INT32, ["batch_size", "max_num_blocks"]),
            ("attention_metadata", ir.DataType.INT32, [3]),
        ]
        for name, dtype, shape in declarations:
            self.graph.inputs.append(self.make_value(name, dtype, shape))
        for layer_id in range(self.num_layers):
            for suffix in ("key", "value"):
                self.graph.inputs.append(
                    self.make_value(
                        f"past_key_values.{layer_id}.{suffix}",
                        self.io_dtype,
                        ["num_blocks", self.paged_block_size, self.num_kv_heads, self.head_size],
                    )
                )

    def save_model(self, out_dir):
        out_path = os.path.join(out_dir, self.filename)
        data_path = out_path + ".data"
        with tempfile.TemporaryDirectory(dir=out_dir, prefix=f".{self.filename}.") as staging_dir:
            staged_path = os.path.join(staging_dir, self.filename)
            staged_data_path = staged_path + ".data"
            with tqdm() as progress:
                total_set = False

                def callback(tensor, metadata):
                    nonlocal total_set
                    if not total_set:
                        progress.total = metadata.total
                        total_set = True
                    progress.update()
                    progress.set_description(f"Saving {tensor.name} ({tensor.dtype.short_name()}, {tensor.shape})")

                Model.stamp_build_metadata(self.model)
                ir.save(
                    self.model,
                    staged_path,
                    external_data=os.path.basename(staged_data_path),
                    size_threshold_bytes=0,
                    callback=callback,
                )
            self.replace_model_files(staged_path, staged_data_path, out_path, data_path, staging_dir)

    def replace_model_files(self, staged_path, staged_data_path, out_path, data_path, staging_dir):
        replacements = ((staged_data_path, data_path), (staged_path, out_path))
        backups = []
        installed = []
        try:
            for _, destination in replacements:
                if os.path.exists(destination):
                    backup = os.path.join(staging_dir, os.path.basename(destination) + ".bak")
                    os.replace(destination, backup)
                    backups.append((backup, destination))
            for source, destination in replacements:
                os.replace(source, destination)
                installed.append(destination)
        except Exception:
            for path in reversed(installed):
                if os.path.exists(path):
                    os.remove(path)
            for backup, destination in reversed(backups):
                os.replace(backup, destination)
            raise

    def genai_config_section(self):
        return {
            "filename": self.filename,
            "num_hidden_layers": self.num_layers,
            "num_key_value_heads": self.num_kv_heads,
            "head_size": self.head_size,
            "block_size": self.block_size,
            "num_draft_tokens": self.num_draft_tokens,
            "selector_top_k": self.selector_top_k,
            "mask_token_id": self.mask_token_id,
            "sliding_window": self.sliding_window,
            "main_aux_hidden_states": "aux_hidden_states",
            "inputs": {
                "aux_hidden_states": "aux_hidden_states",
                "input_ids": "input_ids",
                "q_row_map": "q_row_map",
                "qkv_row_map": "qkv_row_map",
                "block_row_index": "block_row_index",
                "cumulative_sequence_lengths": "cumulative_sequence_lengths",
                "past_sequence_lengths": "past_sequence_lengths",
                "block_table": "block_table",
                "attention_metadata": "attention_metadata",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value",
            },
            "outputs": {
                "candidate_ids": "draft_candidate_ids",
                "scores": "draft_scores",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value",
            },
        }
