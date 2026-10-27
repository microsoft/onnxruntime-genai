# Qwen3.8 Model Builder Design

Qwen3.8 builders live in the [`builders/qwen3_8/`](../builders/qwen3_8/) package. They inherit from the base `Model` infrastructure for graph initialization, node/value creation, initializers, quantization, and serialization.

## Components

- `model.py` coordinates the package and contains the non-paged Engram graph.
- `text.py` builds the decoder, sparse attention, paged Engram, and hyper connections.
- `mtp.py` builds the one-layer MTP head with the same text-building infrastructure.
- `graph.py` supplies the shared companion builder.
- `embedding.py`, `engram.py`, and `vision.py` build their corresponding components.

Component loaders remain in `loaders/`. Shared standard-ONNX decompositions are registered from `builders/expansions/standard.py`; model-specific graph emission remains in the owning builder.

## Index Sharing

Paged MTP graphs declare index-sharing inputs and outputs during initialization. Sparse attention emits the projection-row Gather, index merge, packed indexer, and status output directly. Index sharing is not a separate graph or a conversion pass over a serialized model.

The existing `max_draft_tokens` setting controls the exported draft budget, bounded by the kernel's supported capacity. Main and MTP graphs retain their ordinary cache and state contracts.

## Embeddings and Storage

Non-paged and paged Engram paths reuse the text builder's embedding, projection, normalization, and convolution helpers. Each exported component owns one external-data file. Runtime initializer-sharing metadata may reuse compatible weights without making an ONNX graph depend on another graph's data file.

## Configuration and Layout

Operator settings belong in the existing sparse-indexer, PLE, hyper-connection, attention, and quantization dictionaries. Weight packing is selected explicitly for the deployment target, not inferred from the export host. Integer QMoE, NVFP4, and FP8 have distinct layout contracts; see [packing options](../README.md#qmoe-weights-prepacked) and [quantization design](../quantization/QUANTIZATION.md).

For model-specific command-line examples, see [Model-specific Export Options](model-specific-options.md). General architecture and naming guidance remain in [DESIGN.md](../DESIGN.md).
