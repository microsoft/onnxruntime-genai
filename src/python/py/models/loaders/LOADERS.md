# Model Loaders

This directory contains the model loaders used by the model builder. Each
loader translates a source model or checkpoint format from its original
implementation into the common, builder-facing intermediate representation
(IR).

The IR presents model structure and tensors through a consistent set of Python
objects, including embeddings, decoder layers, attention projections, MLP or
MoE blocks, normalization layers, and the language-model head. Builders can
therefore traverse the same interface and emit ONNX graphs without depending
on the source format's naming, layout, or quantization scheme.

## Loading Flow

1. A builder selects a loader based on the input format and quantization
	configuration.
2. The loader reads the original checkpoint and maps its tensor names, module
	hierarchy, and storage format into the common IR.
3. The builder consumes that IR to create ONNX nodes and initializers.

## Loaders

- `gguf.py` maps GGUF tensors and metadata to the common model structure.
- `base.py` defines the shared quantized-model IR and base loading,
	unpacking, and repacking behavior.
- `awq.py`, `gptq.py`, `quark.py`, `olive.py`, `modelopt.py`, and `quant_auto.py`
	implement the source-format-specific quantized checkpoint loaders.
	See [`quant-auto.md`](quant-auto.md) for the `quant_auto` tensor layout and
	tied-embedding design.
- `quant_model.py` selects the concrete quantized loader for the requested
	quantization format.

New source formats should be implemented here and should expose the same model
structure consumed by the builders, keeping source-specific logic out of the
ONNX graph construction code.
