# Engine test selection

Use the lowest test layer that can prove the behavior. Add a higher layer only when the behavior crosses a boundary that the lower layer cannot exercise.

| Test layer | Use it when | Typical assertions | Do not use it for |
|---|---|---|---|
| C++ unit test | The behavior belongs to one Engine component or invariant and does not require an ONNX model | admission decisions, block accounting, reservation and rollback, request state transitions, scheduler ordering, failure isolation | public API wiring, model input/output binding, execution-provider behavior |
| Synthetic-model test | The behavior crosses Engine components or the public API, but a tiny deterministic ONNX graph can represent it | exact tokens, isolated-versus-batched equality, staggered admission, packed inputs, block tables, cache reuse, completion and removal | real PagedAttention kernel behavior, tokenizer behavior, real model numerics |
| Real-model integration test | Correctness depends on a real exported model, tokenizer, execution provider, or custom operator | real model loading, PagedAttention execution, isolated-versus-batched equality, request lifecycle behavior, provider-specific integration | latency, throughput, memory, capacity, scaling, or longevity |

## C++ unit tests

Add a unit test under `test/cpp/engine/` when the failure can be reproduced by directly constructing the relevant Engine object or test double.

Unit tests are the default for:

- scheduler and admission policy;
- block-pool and cache accounting;
- reservation, commit, and rollback;
- request lifecycle transitions;
- invalid state and failure paths;
- arithmetic and boundary conditions.

Keep these tests small and deterministic. Prefer a focused test double over loading a model.

## Synthetic-model tests

Add a case to `test/python/test_onnxruntime_genai_engine.py` when the behavior must pass through the packaged Python API, scheduler, paged cache, packed model inputs, sampling, and request lifecycle together.

The checked-in model under `test/models/engine/synthetic-paged/` produces exact predictable tokens and requires no external download. Use it for cases where a component-level test could pass even though the components are wired together incorrectly.

Good synthetic-model scenarios include:

- request rows, sequence lengths, or block tables becoming mixed;
- simultaneous or staggered requests changing each other's output;
- completion or removal affecting a surviving request;
- incorrect max-length or EOS handling;
- Engine teardown and reuse.

Do not treat this test as coverage of the production PagedAttention operator. Its graph intentionally replaces real attention with deterministic ONNX operations.

## Real-model integration tests

Add a case to `test/python/integration/test_integration_engine.py` only when a synthetic graph cannot prove the behavior.

Use the real-model lane for:

- loading a model-builder-produced paged model;
- executing the real `com.microsoft::PagedAttention` operator;
- CUDA provider and platform integration;
- tokenizer and real model input/output contracts;
- verifying that batching, admission, completion, and removal preserve real greedy outputs.

Keep prompts and generation lengths small. Compare a request's batched result with its isolated result instead of hardcoding a long model output.

The real-model suite runs in a separate Linux CUDA stage in the existing integration pipeline.

## Choosing coverage for a bug

1. Add the smallest unit test that reproduces the root cause.
2. Add a synthetic-model regression test if the bug crossed component or public API boundaries.
3. Add a real-model test only if the bug depended on the real operator, exported graph, tokenizer, provider, or platform.
4. Avoid copying the same assertion into every layer when a lower layer already proves it.

Performance and capacity work belongs in the benchmark pipeline, not these correctness suites.

## Useful commands

```bash
# C++ Engine unit tests
./build/Linux/RelWithDebInfo/engine_unit_tests

# Synthetic public-API tests
python -m pytest test/python/test_onnxruntime_genai_engine.py -sv

# Real-model tests (CUDA model root required)
python -m pytest test/python/integration/test_integration_engine.py -sv \
    --run-engine-tests \
    --execution-provider cuda \
    --model-root /path/to/models
```
