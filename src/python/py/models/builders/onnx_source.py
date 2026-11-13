"""Build from validated flat ONNX sources without unpacking their weights."""

from __future__ import annotations

from pathlib import Path

import onnx_ir as ir
from loaders.component import ComponentModelLoader

from .base import Model


class OnnxSourceModel(Model):
    def __init__(self, source, decoder_config=None, extra_options=None):
        options = extra_options or {}
        unsupported = set(options) - {"filename", "kv_cache_quant_scheme", "kv_cache_scale_file"}
        if unsupported:
            raise ValueError(f"ONNX source export does not support options {sorted(unsupported)}.")
        scheme = options.get("kv_cache_quant_scheme", "none")
        if scheme not in {"none", "int8_per_channel"}:
            raise ValueError("ONNX source export supports only none or int8_per_channel KV caches.")
        if source.functions or any(
            attribute.type in {ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS}
            for node in source.graph
            for attribute in node.attributes.values()
        ):
            raise ValueError("ONNX source export requires a flat graph without local functions.")
        node_names = [node.name for node in source.graph if node.name]
        if len(node_names) != len(set(node_names)):
            raise ValueError("ONNX source node names must be unique.")
        reserved_node_names = set(node_names)
        super().__init__(
            graph_name=source.graph.name or "main_graph", graph_opset=source.graph.opset_imports.get("", 20)
        )
        self.graph.opset_imports.update(source.graph.opset_imports)
        self.model.ir_version = source.ir_version
        self.model.metadata_props.update(source.metadata_props)
        self.model.doc_string = source.doc_string
        self.graph.doc_string = source.graph.doc_string
        self.graph.metadata_props.update(source.graph.metadata_props)
        self.make_config_init(decoder_config, options)
        cache_names = set()
        layer_by_key = {}
        if scheme != "none":
            source_inputs = {value.name: value for value in source.graph.inputs}
            for layer_id, key_name in self.input_names["past_key_values.key"].items():
                layer_by_key[key_name] = layer_id
                for bindings, keys in (
                    (decoder_config["inputs"], ("past_key_names", "past_value_names")),
                    (decoder_config["outputs"], ("present_key_names", "present_value_names")),
                ):
                    cache_names.update(bindings[key] % layer_id for key in keys)
                for key, cache_slot in (("past_key_names", 3), ("past_value_names", 4)):
                    name = decoder_config["inputs"][key] % layer_id
                    value = source_inputs.get(name)
                    if value is None or value.dtype not in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
                        raise ValueError(f"ONNX source cache {name!r} must be an FP16 or BF16 graph input.")
                    if (
                        value.shape is None
                        or len(value.shape) != 4
                        or list(value.shape)[2:] != [self.num_kv_heads, self.head_size]
                    ):
                        raise ValueError(f"ONNX source cache {name!r} does not match the decoder head geometry.")
                    if any(
                        consumer.domain != "com.microsoft"
                        or consumer.op_type != "PagedAttention"
                        or input_index != cache_slot
                        for consumer, input_index in value.uses()
                    ):
                        raise ValueError(f"ONNX source cache consumer for {name!r} is unsupported for conversion.")
        elif self.kv_cache_attrs["scales_path"]:
            raise ValueError("Unquantized ONNX source export does not use a KV scale file.")

        for value in source.graph.initializers.values():
            self.make_initializer(value.const_value, value.name)
        if scheme != "none":
            for layer_id in self.input_names["past_key_values.key"]:
                if any(name in self.graph.initializers for name in self.get_kv_cache_scale_names(layer_id)):
                    raise ValueError("ONNX source already contains KV scale initializers.")
            self.make_kv_cache_scale_initializers()
        self.graph.inputs.extend(self.make_source_value(value, cache_names) for value in source.graph.inputs)
        quantized_layers = set()
        for index, node in enumerate(source.graph):
            for value in node.outputs:
                self.make_source_value(value, cache_names)
            inputs = [value.name if value is not None else "" for value in node.inputs]
            attributes = dict(node.attributes)
            if scheme != "none" and node.domain == "com.microsoft" and node.op_type == "PagedAttention":
                if len(inputs) < 5 or inputs[3] not in layer_by_key:
                    raise ValueError("ONNX source attention does not match its decoder cache bindings.")
                layer_id = layer_by_key[inputs[3]]
                if inputs[4] != decoder_config["inputs"]["past_value_names"] % layer_id:
                    raise ValueError("ONNX source key/value cache layers do not match.")
                if len(node.outputs) < 3:
                    raise ValueError("ONNX source cache outputs must include the key and value caches.")
                for cache, key in zip(node.outputs[1:3], ("present_key_names", "present_value_names"), strict=True):
                    expected_name = decoder_config["outputs"][key] % layer_id
                    if cache.name != expected_name or not cache.is_graph_output():
                        raise ValueError("ONNX source cache outputs must directly match the decoder bindings.")
                    if cache.uses():
                        raise ValueError(f"ONNX source cache consumer for output {cache.name!r} is unsupported for conversion.")
                inputs.extend([""] * max(0, 16 - len(inputs)))
                inputs[14], inputs[15] = self.get_kv_cache_scale_inputs(layer_id=layer_id)
                attributes.update(k_quant_type="PER_CHANNEL", v_quant_type="PER_CHANNEL")
                quantized_layers.add(layer_id)
            node_name = node.name
            if not node_name:
                generated_name = f"source/{index}/{node.op_type}"
                node_name = generated_name
                suffix = 1
                while node_name in reserved_node_names:
                    node_name = f"{generated_name}/{suffix}"
                    suffix += 1
                reserved_node_names.add(node_name)
            self.make_node(
                node.op_type,
                inputs=inputs,
                outputs=[value.name for value in node.outputs],
                name=node_name,
                domain=node.domain,
                metadata_props=dict(node.metadata_props),
                **attributes,
            )
        if scheme != "none" and quantized_layers != set(self.input_names["past_key_values.key"]):
            raise ValueError("Every configured cache layer must have a PagedAttention node.")
        self.graph.outputs.extend(self.make_source_value(value, cache_names) for value in source.graph.outputs)
        if scheme != "none" or "onnxruntime_genai.kv_cache_quant_scheme" not in self.model.metadata_props:
            self.model.metadata_props["onnxruntime_genai.kv_cache_quant_scheme"] = scheme

    def make_config_init(self, config, extra_options=None):
        options = extra_options or {}
        self.filename = options.get("filename", "model.onnx")
        if Path(self.filename).name != self.filename or not self.filename.endswith(".onnx"):
            raise ValueError("ONNX source output filename must be a plain .onnx filename.")
        self.kv_cache_attrs = {
            "quant_scheme": options.get("kv_cache_quant_scheme", "none"),
            "quant_mode": "PER_CHANNEL",
            "scales_path": options.get("kv_cache_scale_file", ""),
        }
        if self.kv_cache_attrs["quant_scheme"] == "none":
            return
        if config is None:
            raise ValueError("Quantized ONNX source export requires the original decoder configuration.")
        self.num_layers = config["num_hidden_layers"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_size = config["head_size"]
        self.use_paged_attention = True
        state_groups = config.get("state_groups")
        layer_ids = (
            list(range(self.num_layers))
            if state_groups is None
            else [
                layer_id
                for group in state_groups
                if group["kind"] == "paged_kv"
                for layer_id in group["layer_ids"]
            ]
        )
        if (
            not layer_ids
            or any(type(layer_id) is not int or not 0 <= layer_id < self.num_layers for layer_id in layer_ids)
            or len(set(layer_ids)) != len(layer_ids)
        ):
            raise ValueError("ONNX source paged KV layer IDs must be unique integers in the decoder layer range.")
        self.input_names = {
            "past_key_values.key": {
                layer_id: config["inputs"]["past_key_names"] % layer_id for layer_id in layer_ids
            },
        }

    @classmethod
    def from_pretrained(cls, model_path, decoder_config=None, extra_options=None):
        path = Path(model_path).resolve()
        loader = ComponentModelLoader(path, "ONNX source")
        source = loader.load(reject_sparse_initializers=True)
        loader.validate(allow_ort_extensions=True)
        ir.external_data.set_base_dir(source.graph, str(path.parent))
        return cls(source, decoder_config, {"filename": path.name, **(extra_options or {})})

    def make_source_value(self, value, cache_names):
        emitted = self.make_value(
            value.name,
            ir.DataType.INT8 if value.name in cache_names else None,
            value.shape,
            value_type=value.type,
        )
        emitted.doc_string = value.doc_string
        emitted.metadata_props.update(value.metadata_props)
        return emitted

    def save_model(self, out_dir):
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.graph.sort()
        path = directory / self.filename
        self.save_onnx_model(self.model, path, inline_small_initializers=True)
        return path
