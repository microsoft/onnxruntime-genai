# WebGPU shared KV fixture

A single FP32 `com.microsoft::GroupQueryAttention` node with two query heads,
one KV head and head size 64. Batch and sequence dimensions are symbolic.
KV uses sparse layer index 3 and custom `kv.3.k`/`kv.3.v` → `out.3.k`/`out.3.v`
bindings, with no other cache consumers or weights. ONNX IR 9; opsets 21 and
`com.microsoft` 1.

`shared_cache_growth_tests` loads this file through ORT and checks automatic
sharing, aliasing and growth without exporting temporary models at test runtime.
GPU-only execution also checks attention against a computed reference through
growth, capture/replay and rewind, with changing inputs and poisoned spare
capacity to detect use of stale or masked cache rows.
