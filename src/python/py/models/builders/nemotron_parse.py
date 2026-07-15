# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.

import json
import os
import re

import onnx
import onnx_ir as ir
import torch
from onnx import TensorProto, helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, QuantFormat
from packaging.version import Version
from transformers import AutoModel, AutoProcessor
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

_SELF_PRESENT_RE = re.compile(r"^present\.(?P<layer>\d+)\.(?P<slot>key|value)$")
_CROSS_PRESENT_RE = re.compile(r"^cross_present\.\d+\.(key|value)$")
_TENSOR_SCATTER_CACHE_WRITE_INPUT = "cache_write_indices"
_TENSOR_SCATTER_OP_TYPE = "TensorScatter"
_TENSOR_SCATTER_OPSET_VERSION = 24
_TENSOR_SCATTER_MIN_TORCH_VERSION = Version("2.12")


def _supports_direct_tensor_scatter_export():
    return Version(str(torch.__version__).split("+", 1)[0]) >= _TENSOR_SCATTER_MIN_TORCH_VERSION


def _cache_name_templates(kind):
    if kind == "past":
        return (
            "past_key_values.%d.key",
            "past_key_values.%d.value",
            "cross_past_key_values.%d.key",
            "cross_past_key_values.%d.value",
        )
    if kind == "present":
        return (
            "present.%d.key",
            "present.%d.value",
            "cross_present.%d.key",
            "cross_present.%d.value",
        )
    raise ValueError(f"Unsupported cache name kind: {kind}")


def _cache_names(kind, num_layers):
    templates = _cache_name_templates(kind)
    names = []
    for layer in range(num_layers):
        names.extend(template % layer for template in templates)
    return names


def _flatten_encoder_decoder_cache(cache):
    flattened = []
    for layer_cache in cache:
        values = tuple(layer_cache)
        if len(values) >= 6:
            self_key, self_value, _, cross_key, cross_value, _ = values[:6]
        elif len(values) == 4:
            self_key, self_value, cross_key, cross_value = values
        else:
            raise RuntimeError(f"Unexpected cache tuple length: {len(values)}")
        flattened.extend([self_key, self_value, cross_key, cross_value])
    return tuple(flattened)


def _flatten_self_attention_cache(cache):
    flattened = []
    for layer_cache in cache:
        self_key, self_value = tuple(layer_cache)[:2]
        flattened.extend([self_key, self_value])
    return tuple(flattened)


def _self_cache_names(kind, num_layers):
    templates = _cache_name_templates(kind)[:2]
    return [template % layer for layer in range(num_layers) for template in templates]


def _cross_cache_names(kind, num_layers):
    templates = _cache_name_templates(kind)[2:]
    return {template % layer for layer in range(num_layers) for template in templates}


def _tensor_scatter(past_cache, update, write_indices):
    if torch.onnx.is_in_onnx_export():
        present_cache = torch.onnx.ops.symbolic(
            _TENSOR_SCATTER_OP_TYPE,
            (past_cache, update, write_indices),
            attrs={"axis": -2, "mode": "linear"},
            dtype=past_cache.dtype,
            shape=past_cache.shape,
            version=_TENSOR_SCATTER_OPSET_VERSION,
        )
        # The symbolic op's fake implementation creates a CPU tensor. Preserve
        # the cache device so downstream CUDA operations can be captured.
        return present_cache.to(device=past_cache.device)

    present_cache = past_cache.clone()
    for batch_idx in range(past_cache.shape[0]):
        start = int(write_indices[batch_idx].item())
        end = start + update.shape[-2]
        present_cache[batch_idx, :, start:end, :] = update[batch_idx]
    return present_cache


class _TensorScatterDynamicCache(DynamicCache):
    def __init__(self, cache_data, write_indices):
        self.write_indices = write_indices
        self.physical_sequence_length = cache_data[0][0].shape[-2]
        super().__init__(cache_data)

    def _layer_cache(self, layer_idx):
        if hasattr(self, "layers"):
            layer = self.layers[layer_idx]
            return layer.keys, layer.values
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _set_layer_cache(self, layer_idx, keys, values):
        if hasattr(self, "layers"):
            self.layers[layer_idx].keys = keys
            self.layers[layer_idx].values = values
        else:
            self.key_cache[layer_idx] = keys
            self.value_cache[layer_idx] = values

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        del cache_kwargs
        past_keys, past_values = self._layer_cache(layer_idx)
        present_keys = _tensor_scatter(past_keys, key_states, self.write_indices)
        present_values = _tensor_scatter(past_values, value_states, self.write_indices)
        self._set_layer_cache(layer_idx, present_keys, present_values)
        return present_keys, present_values

    def get_seq_length(self, layer_idx=0):
        del layer_idx
        return self.write_indices[0]

    def get_max_cache_shape(self):
        return self.physical_sequence_length


def _resolve_image_size(config, extra_options):
    if "image_height" in extra_options or "image_width" in extra_options:
        return (
            int(extra_options.get("image_height", 768)),
            int(extra_options.get("image_width", 768)),
        )

    image_size = getattr(config, "image_size", None)
    if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        return int(image_size[0]), int(image_size[1])

    return 768, 768


def _attribute_value(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return None


def _has_graph_input(model, name):
    return any(value.name == name for value in model.graph.input)


def _add_cache_write_input(model, name):
    if _has_graph_input(model, name):
        return False
    model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.INT64, ["batch_size"]))
    return True


def _set_dim_value(dim, value):
    dim.ClearField("dim_param")
    dim.dim_value = int(value)


def _set_value_sequence_dim(value, sequence_length):
    tensor_type = value.type.tensor_type
    if len(tensor_type.shape.dim) < 3:
        return False
    _set_dim_value(tensor_type.shape.dim[2], sequence_length)
    return True


def _set_attention_mask_sequence_dim(value, sequence_length):
    tensor_type = value.type.tensor_type
    if len(tensor_type.shape.dim) < 2:
        return False
    _set_dim_value(tensor_type.shape.dim[1], sequence_length)
    return True


def _set_static_self_cache_sequence_shapes(model, cache_sequence_length):
    self_cache_names = set()
    present_names = set()
    for node in model.graph.node:
        if node.op_type != _TENSOR_SCATTER_OP_TYPE or len(node.output) != 1:
            continue
        match = _SELF_PRESENT_RE.match(node.output[0])
        if match:
            layer = match.group("layer")
            slot = match.group("slot")
            self_cache_names.add(f"past_key_values.{layer}.{slot}")
            present_names.add(node.output[0])

    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
            if value.name in self_cache_names or value.name in present_names:
                _set_value_sequence_dim(value, cache_sequence_length)
            elif value.name == "decoder_attention_mask":
                _set_attention_mask_sequence_dim(value, cache_sequence_length)


def _remove_cross_present_graph_outputs(model):
    kept_outputs = [output for output in model.graph.output if _CROSS_PRESENT_RE.match(output.name) is None]
    removed = len(model.graph.output) - len(kept_outputs)
    if removed:
        del model.graph.output[:]
        model.graph.output.extend(kept_outputs)
    return removed


def _ensure_main_opset_version(model, version):
    current_version = next(
        (opset.version for opset in model.opset_import if opset.domain == ""),
        None,
    )
    if current_version is None:
        model.opset_import.append(helper.make_opsetid("", version))
    elif current_version < version:
        model.CopyFrom(onnx.version_converter.convert_version(model, version))


def _patch_self_present_concat_node(node, cache_write_input):
    if node.op_type != "Concat" or len(node.output) != 1:
        return False
    match = _SELF_PRESENT_RE.match(node.output[0])
    if match is None:
        return False

    layer = match.group("layer")
    slot = match.group("slot")
    expected_past = f"past_key_values.{layer}.{slot}"
    if len(node.input) != 2 or node.input[0] != expected_past:
        raise RuntimeError(f"Unexpected self-present Concat inputs for {node.output[0]}: {list(node.input)}")
    axis = _attribute_value(node, "axis")
    if axis not in (-2, 2):
        raise RuntimeError(f"Unexpected Concat axis for {node.output[0]}: {axis}")

    update_input = node.input[1]
    del node.input[:]
    node.input.extend([expected_past, update_input, cache_write_input])
    del node.attribute[:]
    node.attribute.extend(
        [
            helper.make_attribute("axis", -2),
            helper.make_attribute("mode", "linear"),
        ]
    )
    node.op_type = _TENSOR_SCATTER_OP_TYPE
    node.domain = ""
    return True


def _apply_tensor_scatter_to_decoder_model(
    model,
    cache_sequence_length,
    cache_write_input=_TENSOR_SCATTER_CACHE_WRITE_INPUT,
):
    added_cache_write_indices = _add_cache_write_input(model, cache_write_input)
    _ensure_main_opset_version(model, _TENSOR_SCATTER_OPSET_VERSION)

    patched_self_updates = 0
    for node in model.graph.node:
        if _patch_self_present_concat_node(node, cache_write_input):
            patched_self_updates += 1

    removed_cross_outputs = _remove_cross_present_graph_outputs(model)
    _set_static_self_cache_sequence_shapes(model, int(cache_sequence_length))
    return {
        "patched_self_updates": patched_self_updates,
        "removed_cross_outputs": removed_cross_outputs,
        "added_cache_write_indices": added_cache_write_indices,
    }


def _save_model_with_external_data(model, onnx_path):
    external_data_path = onnx_path + ".data"
    if os.path.exists(external_data_path):
        os.unlink(external_data_path)
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(external_data_path),
        size_threshold=1024,
        convert_attribute=False,
    )


class _NemotronParseEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values):
        return self.encoder(pixel_values, return_dict=True).last_hidden_state


class _NemotronParseDecoderPrefill(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.lm_head = model.lm_head

    def forward(self, decoder_input_ids, decoder_attention_mask, encoder_hidden_states):
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=True,
            return_dict=True,
        )
        logits = self.lm_head(decoder_outputs.last_hidden_state[:, -1:, :])
        return (logits,) + _flatten_encoder_decoder_cache(decoder_outputs.past_key_values)


class _NemotronParseDecoderWithPast(torch.nn.Module):
    def __init__(self, model, num_layers):
        super().__init__()
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        self.num_layers = num_layers

    def forward(self, decoder_input_ids, decoder_attention_mask, encoder_hidden_states, *past_tensors):
        self_cache_data = []
        cross_cache_data = []
        for layer in range(self.num_layers):
            base = layer * 4
            self_cache_data.append((past_tensors[base], past_tensors[base + 1]))
            cross_cache_data.append((past_tensors[base + 2], past_tensors[base + 3]))

        past_key_values = EncoderDecoderCache(
            DynamicCache(self_cache_data),
            DynamicCache(cross_cache_data),
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return (logits,) + _flatten_encoder_decoder_cache(decoder_outputs.past_key_values)


class _NemotronParseDecoderWithPastTensorScatter(torch.nn.Module):
    def __init__(self, model, num_layers):
        super().__init__()
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        self.num_layers = num_layers

    def forward(
        self,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_hidden_states,
        cache_write_indices,
        *past_tensors,
    ):
        self_cache_data = []
        cross_cache_data = []
        for layer in range(self.num_layers):
            base = layer * 4
            self_cache_data.append((past_tensors[base], past_tensors[base + 1]))
            cross_cache_data.append((past_tensors[base + 2], past_tensors[base + 3]))

        past_key_values = EncoderDecoderCache(
            _TensorScatterDynamicCache(self_cache_data, cache_write_indices),
            DynamicCache(cross_cache_data),
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return (logits,) + _flatten_self_attention_cache(decoder_outputs.past_key_values)


class NemotronParseModel:
    DECODER_CACHE_EXTRA_OPTIONS = frozenset(
        {
            "decoder_cache_mode",
            "prefill_sequence_length",
            "cache_sequence_length",
        }
    )

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.config = config
        self.io_dtype = io_dtype
        self.onnx_dtype = onnx_dtype
        self.ep = ep
        self.cache_dir = cache_dir
        self.extra_options = extra_options
        self.hf_token = extra_options.get("hf_token", True)
        self.hf_remote = extra_options.get("hf_remote", True)
        self.model_name_or_path = None
        self.model_type = "nemotron_parse"

        self.image_height, self.image_width = _resolve_image_size(config, extra_options)
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height and image_width must be positive.")
        self.decoder_cache_mode = str(
            extra_options.get("decoder_cache_mode", "tensor_scatter")
        ).lower()
        if self.decoder_cache_mode != "tensor_scatter":
            raise ValueError(
                f"decoder_cache_mode={self.decoder_cache_mode} is not supported by this model; "
                "supported modes: tensor_scatter."
            )
        self.prefill_sequence_length = int(
            extra_options.get("prefill_sequence_length", 8)
        )
        self.cache_sequence_length = int(
            extra_options.get(
                "cache_sequence_length", self.config.max_sequence_length
            )
        )
        if self.prefill_sequence_length <= 0:
            raise ValueError("prefill_sequence_length must be positive.")
        if self.cache_sequence_length <= 0:
            raise ValueError("cache_sequence_length must be positive.")
        if self.cache_sequence_length <= self.prefill_sequence_length:
            raise ValueError(
                "cache_sequence_length must leave room for at least one decoded token."
            )
        self.export_components = {
            component.strip()
            for component in extra_options.get("export_components", "encoder,decoder").split(",")
            if component.strip()
        }
        unsupported_components = self.export_components - {"encoder", "decoder"}
        if not self.export_components or unsupported_components:
            raise ValueError(
                "export_components must contain only encoder and/or decoder."
            )
        self.use_direct_tensor_scatter_export = _supports_direct_tensor_scatter_export()
        if not self.use_direct_tensor_scatter_export:
            print(
                f"PyTorch {torch.__version__} does not support the direct Nemotron Parse TensorScatter export path; "
                "using the post-export ONNX rewrite."
            )
        self.export_device = str(extra_options.get("export_device", "cpu")).lower()
        if self.export_device not in {"cpu", "cuda"}:
            raise ValueError("export_device must be cpu or cuda.")

        patch_size = int(getattr(config.encoder, "patch_size", 16))
        encoder_grid_h = self.image_height // patch_size
        encoder_grid_w = self.image_width // patch_size
        compressed_grid_w = ((encoder_grid_w - 4) // 4) + 1
        if encoder_grid_h <= 0 or compressed_grid_w <= 0:
            raise ValueError("The image size is too small for the encoder patch geometry.")
        self.encoder_sequence_length = encoder_grid_h * compressed_grid_w + 1

        self.encoder_filename = "encoder.onnx"
        self.decoder_filename = "decoder.onnx"
        self.decoder_prefill_filename = "decoder_prefill.onnx"
        self.opset_version = int(extra_options.get("opset_version", 20))

    def _provider_options(self):
        if self.ep == "cpu":
            return []
        ep_name = self.ep.replace("trt-rtx", "NvTensorRtRtx")
        attrs = {"enable_cuda_graph": "1"} if self.ep == "trt-rtx" else {}
        return [{ep_name: attrs}]

    def _torch_dtype(self):
        if self.extra_options.get("torch_dtype", None) == "fp32":
            return torch.float32
        if self.extra_options.get("torch_dtype", None) == "fp16":
            return torch.float16
        if self.extra_options.get("torch_dtype", None) == "bf16":
            return torch.bfloat16
        # HF "auto" currently exports BF16 Conv inputs for this model, which
        # ONNX Runtime rejects before provider execution. Use FP16 by default.
        return torch.float16

    def _load_model(self, input_path):
        self.model_name_or_path = input_path if os.path.isdir(input_path) else self.config._name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        torch_dtype = self._torch_dtype()
        model = AutoModel.from_pretrained(
            self.model_name_or_path,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **extra_kwargs,
        )
        if isinstance(torch_dtype, torch.dtype):
            if self.export_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("export_device=cuda requested, but CUDA is not available.")
            model.to(device=torch.device(self.export_device), dtype=torch_dtype)
        model.eval()

        if getattr(model.config.decoder, "_attn_implementation", None) != "eager":
            model.config.decoder._attn_implementation = "eager"
        if getattr(model.decoder.config, "_attn_implementation", None) != "eager":
            model.decoder.config._attn_implementation = "eager"

        return model

    def _maybe_quantize_int4(self, onnx_path):
        if self.onnx_dtype not in {ir.DataType.INT4, ir.DataType.UINT4}:
            return

        print(f"Quantizing MatMul weights in {onnx_path}")
        model = onnx.load(onnx_path, load_external_data=True)
        quantizer = MatMulNBitsQuantizer(
            model,
            block_size=int(self.extra_options.get("int4_block_size", 32)),
            is_symmetric=self.extra_options.get("int4_is_symmetric", True),
            accuracy_level=int(self.extra_options.get("int4_accuracy_level", 0)),
            quant_format=QuantFormat.QDQ if self.extra_options.get("use_qdq", False) else QuantFormat.QOperator,
            op_types_to_quantize=self.extra_options.get("int4_op_types_to_quantize", ("MatMul",)),
            nodes_to_exclude=self.extra_options.get("int4_nodes_to_exclude", []),
        )
        quantizer.process()
        _save_model_with_external_data(quantizer.model.model, onnx_path)

    def _export_encoder(self, model, output_dir):
        out_path = os.path.join(output_dir, self.encoder_filename)
        print(f"Exporting Nemotron Parse encoder to {out_path}")
        wrapper = _NemotronParseEncoder(model)
        dtype = next(wrapper.parameters()).dtype
        device = next(wrapper.parameters()).device
        pixel_values = torch.randn((1, 3, self.image_height, self.image_width), dtype=dtype, device=device)
        with torch.no_grad():
            wrapper(pixel_values)
        torch.onnx.export(
            wrapper,
            (pixel_values,),
            out_path,
            input_names=["pixel_values"],
            output_names=["encoder_hidden_states"],
            opset_version=self.opset_version,
            external_data=True,
            dynamo=False,
        )
        self._maybe_quantize_int4(out_path)

    def _maybe_apply_tensor_scatter(self, onnx_path):
        if self.use_direct_tensor_scatter_export:
            return

        print(f"Rewriting decoder self KV cache updates to ONNX TensorScatter-24 in {onnx_path}")
        model = onnx.load(onnx_path, load_external_data=True)
        summary = _apply_tensor_scatter_to_decoder_model(
            model,
            cache_sequence_length=self.cache_sequence_length,
        )
        if summary["patched_self_updates"] <= 0:
            raise RuntimeError(f"No self KV Concat nodes were rewritten to TensorScatter in {onnx_path}.")
        _save_model_with_external_data(model, onnx_path)

    def _export_decoder(self, model, output_dir):
        self._export_decoder_prefill(model, output_dir)
        self._export_decoder_with_past(model, output_dir)

    def _cached_decoder_dynamic_axes(self, input_names, output_names):
        dynamic_axes = {
            "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
            "decoder_attention_mask": {0: "batch_size", 1: "total_sequence_length"},
            "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"},
            "logits": {0: "batch_size", 1: "decoder_sequence_length"},
        }
        num_layers = self.config.decoder.decoder_layers
        cross_past_names = _cross_cache_names("past", num_layers)
        cross_present_names = _cross_cache_names("present", num_layers)
        for name in input_names:
            if name in {"decoder_input_ids", "decoder_attention_mask", "encoder_hidden_states"}:
                continue
            if name.endswith(".key") or name.endswith(".value"):
                axis_name = (
                    "encoder_sequence_length"
                    if name in cross_past_names
                    else "past_sequence_length"
                )
                dynamic_axes[name] = {0: "batch_size", 2: axis_name}
        for name in output_names:
            if name == "logits":
                continue
            if name.endswith(".key") or name.endswith(".value"):
                axis_name = (
                    "encoder_sequence_length"
                    if name in cross_present_names
                    else "total_sequence_length"
                )
                dynamic_axes[name] = {0: "batch_size", 2: axis_name}
        return dynamic_axes

    def _export_decoder_prefill(self, model, output_dir):
        out_path = os.path.join(output_dir, self.decoder_prefill_filename)
        print(f"Exporting cached Nemotron Parse decoder prefill to {out_path}")
        wrapper = _NemotronParseDecoderPrefill(model)
        dtype = next(model.decoder.parameters()).dtype
        device = next(model.decoder.parameters()).device
        decoder_input_ids = torch.ones((1, self.prefill_sequence_length), dtype=torch.long, device=device)
        decoder_attention_mask = torch.ones((1, self.prefill_sequence_length), dtype=torch.long, device=device)
        encoder_hidden_states = torch.randn(
            (1, self.encoder_sequence_length, self.config.decoder.d_model), dtype=dtype, device=device
        )
        output_names = ["logits"] + _cache_names("present", self.config.decoder.decoder_layers)
        input_names = ["decoder_input_ids", "decoder_attention_mask", "encoder_hidden_states"]
        torch.onnx.export(
            wrapper,
            (decoder_input_ids, decoder_attention_mask, encoder_hidden_states),
            out_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=self.opset_version,
            external_data=True,
            dynamo=False,
            dynamic_axes=self._cached_decoder_dynamic_axes(input_names, output_names),
        )
        self._maybe_quantize_int4(out_path)

    def _export_decoder_with_past(self, model, output_dir):
        if self.use_direct_tensor_scatter_export:
            self._export_decoder_with_past_tensor_scatter(model, output_dir)
            return

        out_path = os.path.join(output_dir, self.decoder_filename)
        print(f"Exporting cached Nemotron Parse decoder for TensorScatter compatibility rewrite to {out_path}")
        wrapper = _NemotronParseDecoderWithPast(model, self.config.decoder.decoder_layers)
        dtype = next(model.decoder.parameters()).dtype
        device = next(model.decoder.parameters()).device
        num_heads = self.config.decoder.decoder_attention_heads
        head_size = self.config.decoder.d_model // num_heads
        decoder_input_ids = torch.ones((1, 1), dtype=torch.long, device=device)
        decoder_attention_mask = torch.ones((1, self.prefill_sequence_length + 1), dtype=torch.long, device=device)
        encoder_hidden_states = torch.randn(
            (1, self.encoder_sequence_length, self.config.decoder.d_model), dtype=dtype, device=device
        )
        past_tensors = []
        for _ in range(self.config.decoder.decoder_layers):
            past_tensors.extend(
                [
                    torch.randn((1, num_heads, self.prefill_sequence_length, head_size), dtype=dtype, device=device),
                    torch.randn((1, num_heads, self.prefill_sequence_length, head_size), dtype=dtype, device=device),
                    torch.randn((1, num_heads, self.encoder_sequence_length, head_size), dtype=dtype, device=device),
                    torch.randn((1, num_heads, self.encoder_sequence_length, head_size), dtype=dtype, device=device),
                ]
            )
        input_names = [
            "decoder_input_ids",
            "decoder_attention_mask",
            "encoder_hidden_states",
        ] + _cache_names("past", self.config.decoder.decoder_layers)
        output_names = ["logits"] + _cache_names("present", self.config.decoder.decoder_layers)
        torch.onnx.export(
            wrapper,
            (decoder_input_ids, decoder_attention_mask, encoder_hidden_states, *past_tensors),
            out_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=self.opset_version,
            external_data=True,
            dynamo=False,
            dynamic_axes=self._cached_decoder_dynamic_axes(input_names, output_names),
        )
        self._maybe_quantize_int4(out_path)
        self._maybe_apply_tensor_scatter(out_path)

    def _export_decoder_with_past_tensor_scatter(self, model, output_dir):
        out_path = os.path.join(output_dir, self.decoder_filename)
        print(f"Exporting cached Nemotron Parse decoder with direct ONNX TensorScatter-24 to {out_path}")
        wrapper = _NemotronParseDecoderWithPastTensorScatter(model, self.config.decoder.decoder_layers)
        wrapper.eval()
        dtype = next(model.decoder.parameters()).dtype
        device = next(model.decoder.parameters()).device
        num_heads = self.config.decoder.decoder_attention_heads
        head_size = self.config.decoder.d_model // num_heads

        decoder_input_ids = torch.ones((1, 1), dtype=torch.long, device=device)
        decoder_attention_mask = torch.zeros((1, self.cache_sequence_length), dtype=torch.long, device=device)
        decoder_attention_mask[:, : self.prefill_sequence_length + 1] = 1
        encoder_hidden_states = torch.randn(
            (1, self.encoder_sequence_length, self.config.decoder.d_model), dtype=dtype, device=device
        )
        cache_write_indices = torch.full(
            (1,),
            self.prefill_sequence_length,
            dtype=torch.long,
            device=device,
        )

        past_tensors = []
        for _ in range(self.config.decoder.decoder_layers):
            past_tensors.extend(
                [
                    torch.randn((1, num_heads, self.cache_sequence_length, head_size), dtype=dtype, device=device),
                    torch.randn((1, num_heads, self.cache_sequence_length, head_size), dtype=dtype, device=device),
                    torch.randn((1, num_heads, self.encoder_sequence_length, head_size), dtype=dtype, device=device),
                    torch.randn((1, num_heads, self.encoder_sequence_length, head_size), dtype=dtype, device=device),
                ]
            )

        input_names = [
            "decoder_input_ids",
            "decoder_attention_mask",
            "encoder_hidden_states",
            _TENSOR_SCATTER_CACHE_WRITE_INPUT,
        ] + _cache_names("past", self.config.decoder.decoder_layers)
        output_names = ["logits"] + _self_cache_names("present", self.config.decoder.decoder_layers)
        torch.onnx.export(
            wrapper,
            (
                decoder_input_ids,
                decoder_attention_mask,
                encoder_hidden_states,
                cache_write_indices,
                *past_tensors,
            ),
            out_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=_TENSOR_SCATTER_OPSET_VERSION,
            external_data=True,
            dynamo=True,
            optimize=True,
        )
        self._maybe_quantize_int4(out_path)

    def make_model(self, input_path):
        model = self._load_model(input_path)
        self._model = model

    def save_model(self, output_dir):
        try:
            if "encoder" in self.export_components:
                self._export_encoder(self._model, output_dir)
            if "decoder" in self.export_components:
                self._export_decoder(self._model, output_dir)
        finally:
            del self._model

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        decoder_config = self.config.decoder
        genai_config = {
            "model": {
                "type": self.model_type,
                "bos_token_id": self.config.decoder_start_token_id,
                "eos_token_id": decoder_config.eos_token_id,
                "pad_token_id": decoder_config.pad_token_id,
                "context_length": self.cache_sequence_length,
                "vocab_size": decoder_config.vocab_size,
                "vision": {
                    "filename": self.encoder_filename,
                    "inputs": {"pixel_values": "pixel_values"},
                    "outputs": {"image_features": "encoder_hidden_states"},
                    "num_visual_tokens": self.encoder_sequence_length,
                },
                "decoder": {
                    "session_options": {
                        "log_id": "onnxruntime-genai",
                        "provider_options": self._provider_options(),
                    },
                    "filename": self.decoder_filename,
                    "hidden_size": decoder_config.d_model,
                    "head_size": decoder_config.d_model // decoder_config.decoder_attention_heads,
                    "num_attention_heads": decoder_config.decoder_attention_heads,
                    "num_hidden_layers": decoder_config.decoder_layers,
                    "num_key_value_heads": decoder_config.decoder_attention_heads,
                    "inputs": {
                        "input_ids": "decoder_input_ids",
                        "attention_mask": "decoder_attention_mask",
                        "encoder_hidden_states": "encoder_hidden_states",
                    },
                    "outputs": {"logits": "logits"},
                },
            },
            "search": {
                "do_sample": False,
                "early_stopping": True,
                "max_length": self.cache_sequence_length,
                "min_length": 0,
                "num_beams": 1,
                "num_return_sequences": 1,
                "past_present_share_buffer": True,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
            },
        }

        decoder = genai_config["model"]["decoder"]
        decoder["prefill_filename"] = self.decoder_prefill_filename
        decoder["prefill_sequence_length"] = self.prefill_sequence_length
        decoder["cache_update_mode"] = self.decoder_cache_mode
        decoder["inputs"].update(
            {
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value",
                "cross_past_key_names": "cross_past_key_values.%d.key",
                "cross_past_value_names": "cross_past_key_values.%d.value",
                "cache_write_indices": _TENSOR_SCATTER_CACHE_WRITE_INPUT,
            }
        )
        decoder["outputs"].update(
            {
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value",
                "cross_present_key_names": "cross_present.%d.key",
                "cross_present_value_names": "cross_present.%d.value",
            }
        )

        out_path = os.path.join(out_dir, "genai_config.json")
        print(f"Saving GenAI config in {out_path}")
        with open(out_path, "w") as f:
            json.dump(genai_config, f, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            **extra_kwargs,
        )
        image_processor = getattr(processor, "image_processor", None)
        for target in (processor, image_processor):
            if target is None:
                continue
            if hasattr(target, "target_height"):
                target.target_height = self.image_height
            if hasattr(target, "target_width"):
                target.target_width = self.image_width
            if hasattr(target, "final_size"):
                target.final_size = [self.image_height, self.image_width]
            if hasattr(target, "size"):
                target.size = {"height": self.image_height, "width": self.image_width}
        print(f"Saving processing files in {out_dir} for Nemotron Parse")
        processor.save_pretrained(out_dir)
