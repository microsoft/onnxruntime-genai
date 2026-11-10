# WebGPU capture fallback fixtures

`shared.onnx` is a deterministic FP32 decoder with symbolic KV sequence axes.
It uses real `com.microsoft::GroupQueryAttention` with aliased past/present
buffers and a separate logical sequence length. Values in the live KV rows
equal the input token IDs; logits select the next token ID modulo 16 and
depend on the attention output. This fixture verifies that compatible models
keep capture enabled, including across a KV growth boundary.

`control-flow.onnx` adds an input-dependent If to the logits path. ORT rejects
session-level capture for this graph; ordinary eager execution supports it.
It exercises rebuilding the model with capture disabled before it is returned
to the caller, including in-memory loading. Repeated generators reuse the same
eager model and sessions. `sparse-control-flow.onnx` uses nonsequential layer
index 3 and custom KV names. `fixed-control-flow.onnx` fixes the KV sequence
dimension to 8 and verifies that fallback excludes static KV.

`cpu-compute.onnx` reconstructs the original token IDs using CPU-only Unique
and its inverse indices, requiring transfers between CPU and WebGPU without
control flow. This exercises the capture node-assignment restriction.

Regenerate the small ONNX IR 9 / opset 21 files with `python generate.py`.
Tests compare raw logits, generated tokens and KV values with explicit eager
execution, verify effective capture settings and retain unrelated errors.
