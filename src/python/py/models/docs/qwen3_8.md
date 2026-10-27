# Qwen3.8 Model Builder Design

Qwen3.8 builders live in the [`builders/qwen3_8/`](../builders/qwen3_8/) package. They inherit from the base `Model` infrastructure for graph initialization, node/value creation, initializers, quantization, and serialization.

## Components

- `model.py` coordinates the package and exports the standalone Engram component.
- `text.py` builds the decoder, sparse attention, hidden-state-dependent Engram processing, and hyper connections.
- `mtp.py` builds the one-layer MTP head with the same text-building infrastructure.
- `graph.py` supplies the shared companion builder.
- `embedding.py`, `engram.py`, and `vision.py` build their corresponding components.

Component loaders remain in `loaders/`. Shared standard-ONNX decompositions are registered from `builders/expansions/standard.py`; model-specific graph emission remains in the owning builder.

## Index Sharing

Paged MTP graphs declare index-sharing inputs and outputs during initialization. Sparse attention emits the projection-row Gather, index merge, packed indexer, and status output directly. Index sharing is not a separate graph or a conversion pass over a serialized model.

The existing `max_draft_tokens` setting controls the exported draft budget, bounded by the kernel's supported capacity. Main and MTP graphs retain their ordinary cache and state contracts.

## Embeddings and Storage

`Qwen4ExpEngramModel` owns all token hashing and embedding-table lookup for non-paged and paged execution. The decoder always receives `engram_embeddings`; it contains no hash operator, quantized table lookup, or Engram embedding table. The former `Qwen38Engram` decoder helper is removed.

The decoder retains projections, hidden-state-dependent gating, padding masking, convolution, and residual addition because those execute inside the decoder layer. Its small token-history bindings preserve the runtime state interface, and standard-ONNX gathers capture accepted-prefix token history for speculative rollback without duplicating hashing or lookup.

The standalone component preserves its existing runtime interface: token IDs and history are rank two, and embeddings are rank three. The paged implementation uses packed hashing internally; the Engine invokes it per request and packs its output rows for the decoder. Each exported component owns one external-data file, and only the standalone Engram graph stores the Engram lookup table.

## Configuration and Layout

Operator settings belong in the existing sparse-indexer, PLE, hyper-connection, attention, and quantization dictionaries. Weight packing is selected explicitly for the deployment target, not inferred from the export host. Integer QMoE, NVFP4, and FP8 have distinct layout contracts; see [packing options](../README.md#qmoe-weights-prepacked) and [quantization design](../quantization/QUANTIZATION.md).

For model-specific command-line examples, see [Model-specific Export Options](model-specific-options.md). General architecture and naming guidance remain in [DESIGN.md](../DESIGN.md).
