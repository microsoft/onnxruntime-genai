# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os

import numpy as np
import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import TorchTensor, to_torch_dtype
from quantization import CudaQuantizer
from tqdm import tqdm

from .base import Model


# Block drafters share Model's graph lifecycle while retaining their packed block I/O.
class BlockDrafterBuilder(Model):
    """Shared graph, initializer, I/O, and file plumbing for block drafters."""

    # Weight-only quantization of the drafter body. ``quant_bits = None`` keeps every
    # projection dense; subclasses set these from the target's quantization settings so the
    # drafter lands in the same format as the model it drafts for.
    quant_bits = None
    # False when Q comes from the same packed rows as K/V, so the graph has no `q_row_map` input.
    uses_q_row_map = True
    quant_block_size = 32
    quant_prepack = 0
    # Set only when the target's own LM head is symmetric/`default` quantized, which is the one
    # convention whose initializer names the drafter can reuse.
    lm_head_quant = None
    # Set only when the target quantizes its embedding table into `GatherBlockQuantized`.
    embed_quant = None
    # Record the nodes `make_lm_head_adopted` and `make_embedding` left without weights for
    # `adopt_target_tensors` to fill from the saved target.
    lm_head_adoption = None
    embed_adoption = None

    def __init__(self, graph_name="block_drafter", const_prefix="drafter"):
        self.graph_attrs = Model.default_graph_attrs()
        super().__init__(graph_name=graph_name, graph_opset=self.graph_attrs["ai.onnx"])
        self.graph.opset_imports["com.microsoft"] = self.graph_attrs["com.microsoft"]
        self.const_cache = {}
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
        return node

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

    def binary(self, op, name, lhs, rhs, dtype, shape, domain="", **attrs):
        output = self.out(name)
        self.make_node(op, [lhs, rhs], [output], name=name, domain=domain, **attrs)
        self.make_value(output, dtype, shape)
        return output

    def reshape(self, name, root_input, shape_const, dtype, shape):
        return self.binary("Reshape", name, root_input, self.const(shape_const), dtype, shape)

    def make_cast(self, name, root_input, dtype, shape, output=None):
        output = output or self.out(name)
        self.make_node("Cast", [root_input], [output], name=name, to=dtype)
        self.make_value(output, dtype, shape)
        return output

    def make_slice_range(self, name, root_input, start, end, axis, dtype, shape):
        self.make_slice(
            name,
            [root_input, self.const([start]), self.const([end]), self.const([axis])],
            dtype,
            shape,
        )
        return self.out(name)

    def make_split_equal(self, name, root_input, outputs, dtype, shape, axis=-1):
        self.make_split(
            name, [root_input], outputs, [dtype] * len(outputs), [shape] * len(outputs), axis=axis, num_outputs=len(outputs)
        )
        return outputs

    def make_concat(self, name, inputs, dtype, shape, axis=0, output=None):
        output = output or self.out(name)
        self.make_node("Concat", inputs, [output], name=name, axis=axis)
        self.make_value(output, dtype, shape)
        return output

    def make_silu(self, name, root_input, dtype, shape):
        sigmoid = self.unary("Sigmoid", f"{name}/Sigmoid", root_input, dtype, shape)
        return self.binary("Mul", f"{name}/Mul", root_input, sigmoid, dtype, shape)

    def make_linear(self, name, root_input, weight_tensor, bias_tensor, output_shape):
        out_features, in_features = (int(dim) for dim in weight_tensor.shape)
        weight_name = f"{name[1:].replace('/', '.')}.weight"
        output = self.matmul(
            f"{name}/MatMul",
            root_input,
            weight_tensor,
            in_features,
            out_features,
            output_shape[:-1],
            weight_name=weight_name,
            quantize=False,
        )
        if bias_tensor is not None:
            bias_name = f"{name[1:].replace('/', '.')}.bias"
            self.make_initializer(bias_tensor, bias_name, to=self.io_dtype)
            output = self.binary("Add", f"{name}/Add", output, bias_name, self.io_dtype, output_shape)
        return output

    def make_layer_norm(
        self, name, root_input, weight_tensor, bias_tensor, shape, weight_name=None, bias_name=None, epsilon=1e-5
    ):
        weight_name = weight_name or (name[1:].replace("/", ".") + ".weight")
        bias_name = bias_name or (name[1:].replace("/", ".") + ".bias")
        self.make_initializer(weight_tensor, weight_name, to=self.io_dtype)
        self.make_initializer(bias_tensor, bias_name, to=self.io_dtype)
        output = self.out(name)
        self.make_node("LayerNormalization", [root_input, weight_name, bias_name], [output], name=name, axis=-1, epsilon=epsilon)
        self.make_value(output, self.io_dtype, shape)
        return output

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
        output_shape = [*rows, out_features] if isinstance(rows, (list, tuple)) else [rows, out_features]
        self.make_node("MatMul", [root_input, initializer_name], [output], name=name)
        self.make_value(output, self.io_dtype, output_shape)
        return output

    def matmul_nbits(self, name, root_input, weight_tensor, in_features, out_features, rows, initializer_name):
        """Emit a weight-only quantized ``MatMulNBits`` for a torch ``[out, in]`` weight.

        ``MatMulNBits`` consumes ``[N, K]`` directly, so unlike the dense path the weight is
        not transposed. Repeat call sites reuse the initializer the first one registered.
        """
        n_tile = {2: 128, 4: 64, 8: 32}.get(self.quant_bits)
        supported_blocks = (32, 64, 128) if self.quant_prepack == 1 else (64, 128)
        if self.quant_bits == 2:
            supported_blocks = (64, 128)
        prepack = (
            self.quant_prepack
            if n_tile
            and out_features % n_tile == 0
            and self.quant_block_size in supported_blocks
            and in_features % self.quant_block_size == 0
            and (self.quant_bits != 2 or self.quant_prepack == 1)
            else 0
        )
        qweight_name = f"{initializer_name}_Q{self.quant_bits}"
        scales_name = f"{initializer_name}_scales"
        if qweight_name not in self.values:
            weight_tensor = weight_tensor.to(to_torch_dtype(self.io_dtype))
            use_ort_quantizer = self.quant_block_size in (16, 32, 64, 128, 256)
            if prepack:
                qweight, scales = CudaQuantizer.matmulnbits_prepacked_blockwise_quantize(
                    weight_tensor,
                    self.quant_bits,
                    self.quant_block_size,
                    force_arch=90 if prepack == 2 else 80,
                    use_ort_quantizer=use_ort_quantizer,
                )
            else:
                qweight, scales = CudaQuantizer.matmulnbits_blockwise_quantize(
                    weight_tensor,
                    self.quant_bits,
                    self.quant_block_size,
                    flatten_qweight=False,
                    use_ort_quantizer=use_ort_quantizer,
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
        output_shape = [*rows, out_features] if isinstance(rows, (list, tuple)) else [rows, out_features]
        self.make_node(
            "MatMulNBits",
            [root_input, qweight_name, scales_name],
            [output],
            name=name,
            domain="com.microsoft",
            **attributes,
        )
        self.make_value(output, self.io_dtype, output_shape)
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
        output_shape = list(rows) if isinstance(rows, (list, tuple)) else [rows, self.hidden_size]
        self.make_value(output, self.io_dtype, output_shape)
        return output

    def skip_rms_norm(self, name, root, skip, weight_tensor, rows, want_sum=True, dtype=None):
        dtype = self.io_dtype if dtype is None else dtype
        initializer_name = name[1:].replace("/", ".") + ".weight"
        self.make_initializer(weight_tensor, initializer_name, to=self.io_dtype)
        inputs = [root, skip, initializer_name]
        if dtype != self.io_dtype:
            # Retain the residual sum at the accumulation dtype until the next normalization.
            for i, value in enumerate(inputs):
                if self.values[value].dtype != dtype:
                    inputs[i] = self.unary(
                        "Cast", f"{name}/input{i}/Cast", value, dtype, self.values[value].shape, to=dtype
                    )
        output = self.out(name)
        summed_output = f"{name}/output_3"
        self.make_node(
            "SkipSimplifiedLayerNormalization",
            inputs,
            [output, "", "", summed_output] if want_sum else [output],
            name=name,
            domain="com.microsoft",
            epsilon=self.rms_eps,
        )
        self.make_value(output, dtype, [rows, self.hidden_size])
        if want_sum:
            self.make_value(summed_output, dtype, [rows, self.hidden_size])
        if dtype != self.io_dtype:
            output = self.unary(
                "Cast", f"{name}/Cast", output, self.io_dtype, [rows, self.hidden_size], to=self.io_dtype
            )
        return output, (summed_output if want_sum else None)

    def make_embedding(self, name, rows):
        """Emit the embedding lookup over the *target's* table and return its output value name.

        A quantized target writes `GatherBlockQuantized` over `model.embed_tokens.weight_Q{bits}`
        instead of a dense `Gather`. The drafter has to match it, or `share_initializers` finds
        no common name and the drafter keeps a full dense copy of the largest tensor in the model.
        No weights are produced for the quantized form: `adopt_target_tensors` copies the
        target's once it has been saved.
        """
        if self.exclude_embeds:
            return "inputs_embeds"
        if self.target_model is not None and self.embed_quant is None:
            return self.make_decoder_embedding(name, rows)
        output = self.out(name)
        if self.embed_quant is None:
            self.make_initializer(
                self.weights["embed_tokens.weight"], "model.embed_tokens.weight", to=self.external_dtype
            )
            self.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], [output], name=name)
        else:
            bits = self.embed_quant["bits"]
            qweight_name = f"model.embed_tokens.weight_Q{bits}"
            scales_name = "model.embed_tokens.weight_scales"
            node = self.make_node(
                "GatherBlockQuantized",
                [qweight_name, "input_ids", scales_name],
                [output],
                name=name,
                domain="com.microsoft",
                block_size=self.embed_quant["block_size"],
                gather_axis=0,
                quantize_axis=1,
            )
            self.embed_adoption = {
                "node": node,
                "initializers": (qweight_name, scales_name),
                "op_type": "GatherBlockQuantized",
                # `GatherBlockQuantized` takes the table first and the indices second.
                "weight_input": 0,
                "reason": (
                    "Both models embed with the same table, so a mismatch here means the drafter "
                    "would propose from a different input space than the target."
                ),
            }
        self.make_value(output, self.external_dtype, [rows, self.hidden_size])
        return output

    def make_decoder_embedding(self, name, rows):
        """Reuse the decoder's lookup, including tied-weight transforms and quantized tables."""
        nodes = {node.name: node for node in self.target_model.graph}
        source = next(
            (
                nodes[f"/model/embed_tokens/{op}"].outputs[0]
                for op in ("Slice", "GatherBlockQuantized", "Gather")
                if f"/model/embed_tokens/{op}" in nodes
            ),
            None,
        )
        if source is None:
            raise ValueError("The decoder graph has no token embedding lookup to share.")
        if source.dtype is not None and source.dtype != self.external_dtype:
            raise ValueError("The decoder embedding must match the drafter's I/O dtype.")
        missing = []

        def copy_value(value):
            producer = value.producer()
            if producer is None and value.const_value is None and value in self.target_model.graph.inputs:
                if value.dtype != ir.DataType.INT64:
                    raise ValueError(f"Unexpected decoder embedding input '{value.name}'.")
                return "input_ids"
            if value.name in self.values:
                return value.name
            if producer is None:
                copied = self.make_value(value.name, value.dtype, value.shape)
                copied.const_value = value.const_value
                copied.metadata_props.update(value.metadata_props)
                if copied.const_value is None:
                    missing.append(value.name)
                else:
                    self.graph.register_initializer(copied)
                return value.name
            inputs = [copy_value(input_value) if input_value is not None else "" for input_value in producer.inputs]
            output_names = [output.name for output in producer.outputs]
            node = self.make_node(
                producer.op_type,
                inputs,
                output_names,
                name=producer.name,
                domain=producer.domain,
                **{key: attribute.value for key, attribute in producer.attributes.items()},
            )
            node.metadata_props.update(producer.metadata_props)
            for output in producer.outputs:
                shape = (
                    [
                        rows if isinstance(dim, ir.SymbolicDim) and dim.value == "num_tokens" else dim
                        for dim in output.shape
                    ]
                    if output.shape is not None
                    else None
                )
                copied = self.make_value(output.name, output.dtype, shape)
                copied.metadata_props.update(output.metadata_props)
            return value.name

        root = copy_value(source)
        if missing:
            self.embed_adoption = {"initializers": tuple(missing)}
        output = self.out(name)
        self.rename_value(root, output)
        self.make_value(output, self.external_dtype, [rows, self.hidden_size])
        return output

    def pending_adoptions(self):
        """Return the adoption records still waiting on the target's saved bytes."""
        return [adoption for adoption in (self.embed_adoption, self.lm_head_adoption) if adoption is not None]

    def embedding_initializers(self):
        """Find the lookup's tensors, including a tied LM-head table."""
        names = set()

        def visit(value):
            if value is None:
                return
            if value.name in self.graph.initializers:
                if value.name.startswith(("model.embed_tokens.", "lm_head.MatMul.")):
                    names.add(value.name)
            elif value.producer() is not None:
                for input_value in value.producer().inputs:
                    visit(input_value)

        for prefix in ("dflash2", "dspark"):
            output = self.values.get(f"/{prefix}/embed_tokens/Gather/output_0")
            if output is not None:
                visit(output)
        return frozenset(names)

    def adopt_target_tensors(self, target_model_path, embedding_model_path=None):
        """Fill the drafter's shared tensors from the saved target model.

        Called between the target's save and the drafter's, so `target_model_path` exists and
        still holds its weights in external data that `ir` reads lazily. Each target node is
        found by the initializer it consumes, because the quantizer renames the node (`..._Q4`).
        """
        pending = self.pending_adoptions()
        if not pending:
            return
        target = ir.load(target_model_path)
        for adoption in pending:
            source = (
                ir.load(embedding_model_path)
                if adoption is self.embed_adoption and embedding_model_path is not None
                else target
            )
            if "node" in adoption:
                self.adopt_target_node(
                    source,
                    adoption,
                    os.path.basename(embedding_model_path or target_model_path)
                    if adoption is self.embed_adoption
                    else os.path.basename(target_model_path),
                )
            else:
                for initializer_name in adoption["initializers"]:
                    tensor = source.graph.initializers[initializer_name].const_value
                    value = self.make_value(initializer_name, tensor.dtype, tensor.shape)
                    value.const_value = tensor
                    self.graph.register_initializer(value)
        self.embed_adoption = None
        self.lm_head_adoption = None

    def adopt_target_node(self, target, adoption, target_model_name):
        """Copy one target node's initializers and attributes onto the placeholder the drafter emitted."""
        qweight_name = adoption["initializers"][0]
        weight_input = adoption["weight_input"]
        target_node = next(
            (
                node
                for node in target.graph
                if (node.domain, node.op_type) == ("com.microsoft", adoption["op_type"])
                and len(node.inputs) > weight_input
                and node.inputs[weight_input] is not None
                and node.inputs[weight_input].name == qweight_name
            ),
            None,
        )
        if target_node is None:
            raise ValueError(
                f"The block drafter expected to adopt '{qweight_name}', but '{target_model_name}' has no "
                f"{adoption['op_type']} over it. {adoption['reason']}"
            )
        for initializer_name in adoption["initializers"]:
            tensor = target.graph.initializers[initializer_name].const_value
            value = self.make_value(initializer_name, tensor.dtype, tensor.shape)
            value.const_value = tensor
            self.graph.register_initializer(value)
        node = adoption["node"]
        for attribute_name in list(node.attributes):
            del node.attributes[attribute_name]
        for attribute_name, attribute in target_node.attributes.items():
            node.attributes[attribute_name] = attribute

    def make_lm_head(self, root, rows="num_block"):
        if "lm_head.qweight" in self.weights:
            qweight = self.weights["lm_head.qweight"]
            scales = self.weights["lm_head.scales"]
            expected_qweight_shape = (self.vocab_size, self.hidden_size // 32, 32)
            if tuple(qweight.shape) != expected_qweight_shape:
                raise ValueError(
                    f"LM head qweight has shape {tuple(qweight.shape)}, expected {expected_qweight_shape}."
                )
            # Q8_0 block scales are FP16. BF16 cannot represent them exactly, so a BF16 graph runs this head in FP16
            # with explicit boundary casts instead of rounding the scales.
            head_dtype = ir.DataType.FLOAT16 if self.external_dtype == ir.DataType.BFLOAT16 else self.external_dtype
            if self.io_dtype != head_dtype:
                root = self.unary("Cast", "/lm_head/Cast", root, head_dtype, [rows, self.hidden_size], to=head_dtype)
            name = "/lm_head/MatMul"
            output = self.out(name)
            self.make_initializer(qweight, "lm_head.MatMul.weight_Q8")
            self.make_initializer(scales, "lm_head.MatMul.weight_scales", to=head_dtype)
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
            self.make_value(output, head_dtype, [rows, self.vocab_size])
            if head_dtype != self.external_dtype:
                output = self.unary(
                    "Cast",
                    "/lm_head/logits/Cast",
                    output,
                    self.external_dtype,
                    [rows, self.vocab_size],
                    to=self.external_dtype,
                )
            return output

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
        if weight.dtype == torch.float8_e4m3fn:
            # A prequantized head wins over any requested weight precision, for the target as
            # well as here. Quantizing this one to `--precision` instead would leave the drafter
            # proposing tokens scored by a head the target never verifies with.
            if self.lm_head_quant is not None:
                print(
                    "The target's LM head is prequantized FP8, which overrides the requested weight "
                    "precision, so the block drafter shares that head instead of quantizing its own."
                )
                self.lm_head_quant = None
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
        elif self.lm_head_quant is None:
            self.make_initializer(weight.T, "lm_head.MatMul.weight", to=self.external_dtype)
            self.make_node("MatMul", [root, "lm_head.MatMul.weight"], [output], name=name)
        elif self.lm_head_quant["adopt_target"]:
            self.make_lm_head_adopted(name, root, output)
        else:
            self.make_lm_head_nbits(name, root, output, weight)
        self.make_value(output, self.external_dtype, [rows, self.vocab_size])
        return output

    def make_lm_head_adopted(self, name, root, output):
        """Emit the LM head as `MatMulNBits` under the *target's* initializer names.

        No weights are produced here: `adopt_target_tensors` copies the target's once it has
        been saved. Quantizing the same tensor a second time would round it through a second
        implementation (`CudaQuantizer` here, ORT's `MatMulNBitsQuantizer` there), which both
        defeats `share_initializers` and leaves the drafter scoring drafts with a head the
        target does not verify with.
        """
        bits = self.lm_head_quant["bits"]
        qweight_name = f"lm_head.MatMul.weight_Q{bits}"
        scales_name = "lm_head.MatMul.weight_scales"
        node = self.make_node(
            "MatMulNBits",
            [root, qweight_name, scales_name],
            [output],
            name=name,
            domain="com.microsoft",
            bits=bits,
            block_size=self.lm_head_quant["block_size"],
            K=self.hidden_size,
            N=self.vocab_size,
        )
        self.lm_head_adoption = {
            "node": node,
            "initializers": (qweight_name, scales_name),
            "op_type": "MatMulNBits",
            # `MatMulNBits` takes the activations first and the packed weight second.
            "weight_input": 1,
            "reason": (
                "Both models run the same head, so a mismatch here means the target would reject nearly every draft."
            ),
        }

    def make_lm_head_nbits(self, name, root, output, weight):
        """Quantize a private `MatMulNBits` LM head for a drafter that cannot adopt the target's.

        Only reached when the target's prepacked layout needs FP16 activations the drafter does
        not have, so the head is re-quantized here into the portable raw blockwise layout and
        kept out of `share_initializers`.
        """
        bits = self.lm_head_quant["bits"]
        block_size = self.lm_head_quant["block_size"]
        prepack = self.lm_head_quant["prepack"]
        weight = weight.to(to_torch_dtype(self.external_dtype))
        use_ort_quantizer = block_size in (16, 32, 64, 128, 256)
        if prepack:
            qweight, scales = CudaQuantizer.matmulnbits_prepacked_blockwise_quantize(
                weight,
                bits,
                block_size,
                force_arch=90 if prepack == 2 else 80,
                use_ort_quantizer=use_ort_quantizer,
            )
        else:
            qweight, scales = CudaQuantizer.matmulnbits_blockwise_quantize(
                weight,
                bits,
                block_size,
                flatten_qweight=False,
                use_ort_quantizer=use_ort_quantizer,
            )
        qweight_name = f"lm_head.MatMul.weight_Q{bits}"
        scales_name = "lm_head.MatMul.weight_scales"
        self.make_initializer(qweight, qweight_name)
        self.make_initializer(scales, scales_name, to=self.external_dtype)
        self.make_node(
            "MatMulNBits",
            [root, qweight_name, scales_name],
            [output],
            name=name,
            domain="com.microsoft",
            bits=bits,
            block_size=block_size,
            K=self.hidden_size,
            N=self.vocab_size,
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
        ]
        if self.exclude_embeds:
            declarations.append(("inputs_embeds", self.external_dtype, ["num_block", self.hidden_size]))
        if self.ep == "cuda":
            declarations.append(("attention_metadata", ir.DataType.INT32, [3]))
        for name, dtype, shape in declarations:
            if name == "q_row_map" and not self.uses_q_row_map:
                continue
            self.graph.inputs.append(self.make_value(name, dtype, shape))
        # TODO: this is the target's block size. A drafter usually has the smaller head size and
        # so the larger FlashAttention tile, which means the block that is fastest for the target
        # is not the one that is fastest here. Giving the drafter its own block size would
        # decouple them.
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
        if self.pending_adoptions():
            raise ValueError("adopt_target_tensors must run before saving a drafter that adopts the target's tensors.")
        out_path = os.path.join(out_dir, self.filename)
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
            Model.save_onnx_model(self.model, out_path, callback=callback)

    def genai_config_section(self):
        section = {
            "filename": self.filename,
            "session_options": {"ep.cuda.fpa_intb_gemm": "0"},
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
                "attention_metadata": "attention_metadata" if self.ep == "cuda" else "",
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
        if not self.uses_q_row_map:
            del section["inputs"]["q_row_map"]
        if self.exclude_embeds:
            section["inputs"]["inputs_embeds"] = "inputs_embeds"
        return section
