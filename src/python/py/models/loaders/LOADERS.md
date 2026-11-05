# Model Loaders

This directory contains the model loaders used by the model builder. Each loader translates a source model or checkpoint format from its original implementation into the common, builder-facing intermediate representation (IR).

The IR presents model structure and tensors through a consistent set of Python objects, including embeddings, decoder layers, attention projections, MLP or MoE blocks, normalization layers, and the language-model head. Builders can therefore traverse the same interface and emit ONNX graphs without depending on the source format's naming, layout, or quantization scheme.

`TensorModule.can_reuse_as_embedding` tells builders whether a language-model head's storage can directly back an embedding lookup. Loaders set it to `false` for source formats whose packed weights require a format-specific operator.

## Loading Flow

1. A builder selects a loader based on the input format and quantization configuration.
2. The loader reads the original checkpoint and maps its tensor names, module hierarchy, and storage format into the common IR.
3. The builder consumes that IR to create ONNX nodes and initializers.

## Loaders

- `gguf.py` maps GGUF tensors and metadata to the common model structure.
- `resolver.py` resolves URI model sources (e.g. `oci://`) to local paths by scheme. Other inputs pass through unchanged.
- `llmman.py` pulls `oci://` (CNCF ModelPack) sources through a running `llmman serve` daemon, for `resolver.py`.
- `base.py` defines the shared quantized-model IR and base loading, unpacking, and repacking behavior.
- `awq.py`, `gptq.py`, `quark.py`, `olive.py`, `modelopt.py`, and `quant_auto.py` implement the source-format-specific quantized checkpoint loaders. See [`quant-auto.md`](quant-auto.md) for the `quant_auto` tensor layout and tied-embedding design.
- `quant_model.py` selects the concrete quantized loader for the requested quantization format.
- `component.py` validates prebuilt ONNX component sources and returns ONNX IR models for package configuration. Raw external-data validation and the ONNX schema checker stay at this source-format boundary because IR deserialization may normalize malformed metadata and does not provide an equivalent schema checker; graph construction does not use those APIs.
- `qwen3_8.py` loads native Qwen3.8 FP8/NVFP4 checkpoints. `engram.py`, `vision.py`, and `mtp.py` own the corresponding component-loading helpers; `qwen.py` preserves the existing MTP loader import paths.

CUDA packing settings are loader inputs, not properties inferred from the export host GPU. Integer QMoE and NVFP4 have different `weights_prepacked` contracts: NVFP4 defaults to legacy N-packed storage, and only explicit mode `1` preserves the newer K-packed layout. Synthetic loader instances in tests must initialize `quant_attrs` just as normal loader construction does.

New source formats should be implemented here and should expose the same model structure consumed by the builders, keeping source-specific logic out of the ONNX graph construction code.

## ONNX API boundary

The loader ONNX import inventory is:

- `component.py`: `onnx` for public `onnx.load(..., load_external_data=False)`, `onnx.TensorProto` identification/external-storage markers, and `onnx.checker.check_model(path)`; `onnx_ir` for public `ir.serde.deserialize_model`, `ir.ExternalTensor`, `ir.DataType`, and `ir.Shape`.
- `gptoss.py`: `onnx_ir` for `ir.DataType.FLOAT8E8M0`.
- `modelopt.py`: `onnx_ir` for `ir.DataType.FLOAT8E4M3FN`.
- All other loader Python modules have no ONNX or ONNX IR imports. No loader uses `onnx.helper`, `onnx.numpy_helper`, or private ONNX/ONNX IR APIs.

The raw dependency in `component.py` is intentional, not an alternative graph construction or serialization path. Public `ir.load(path)` also parses with external data disabled, but immediately deserializes the protobuf. That loses training graphs, sparse initializers, and tensor payloads inconsistent with an attribute's declared type; sparse tensor attributes are currently unsupported. Consequently, traversing only the resulting IR cannot validate every original external-data location and byte range. The loader recursively inspects the raw protobuf first, rejects unsafe paths (including symlink escapes), missing files, and invalid byte ranges without materializing tensor data, and only then converts to IR through public serde.

ONNX IR also has no public equivalent of the ONNX schema checker. Checking a protobuf reserialized from IR is not equivalent to checking the source: IR can normalize an invalid attribute into a valid one. `validate()` therefore checks the original source path, after source external-data validation in the package loading flow. Dedicated component-loader tests demonstrate these normalization losses, the unsupported sparse-attribute case, and the original-source checker requirement.
