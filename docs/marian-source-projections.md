# Marian source-projection caching

An opt-in model export can move source-only decoder projections into the
encoder. GenAI retains those outputs separately for each generator and reuses
them at every decoder step. This does not cache translations across requests,
transform existing graphs, or change learned weights.

## Model interface

Names beginning with `cached_source_projection_` are reserved for encoder
outputs and matching decoder inputs. GenAI discovers the names from both
sessions independently; the sets must match. No fixed projection count or
contiguous numbering is required. The existing six-tensor FP32 export remains
compatible.

Each pair must have the same FP32 or FP16 element type and rank-three
`[batch, source, projection]` shape. All dimensions must match, including
symbolic names for dynamic dimensions. Batch/source dimensions must also match
the encoder's normal output. Projection width must be a positive constant; it
need not equal the encoder hidden size. Static batch/source dimensions are
checked against each request before cache allocation. Mixed static/dynamic
declarations or different symbolic names are rejected rather than assuming
they describe the same dimension.

Each tensor is allocated using its validated element type and projection
width, with batch/source dimensions resolved from the encoder input. The
generator owns the tensors until destruction; other generators using the same
model do not share or overwrite them.

Models without these names retain the uncached execution path. Cached models
currently require CPU execution and one beam. Greedy-equivalent settings
(`do_sample=false`, `top_k=1`, or `temperature=0`) use the same classification
as the generator. Stochastic sampling and other providers are not supported.

## Memory and performance

The extra retained cache storage per generator is
`batch * source_length * sum(projection_width * element_bytes)`. Source length
includes Marian's appended EOS and padding. Six FP32 projections of width 512
retain `batch * source_length * 12288` bytes. This storage does not grow with
generated length, but it does grow with the number of live generators.

Caching moves computation rather than removing the first projection
evaluation. Its benefit depends on source length, decoder steps, physical batch
width, session threading, and concurrent generators. Measure uncached and
cached exports with identical weights, inputs, runtime and search settings;
include encoder execution and generator setup, and report peak process memory
as well as tensor storage. Short generations need not amortize allocation and
binding costs. Synthetic fixture timings are not translation-performance or
quality evidence.

## Regression fixtures

Regenerate the small ONNX test graphs with:

```powershell
python test\python\create\create_marian_cache_models.py --output-dir test\models\marian-cache
```

The decoder's logits depend on every cache tensor, the source token values,
and the decode step. Tests exercise multiple steps, interleaved generators,
different live shapes, destruction/recreation, FP16 with a different projection
count/width, fixed shapes, and the uncached path. Negative fixtures are valid
ONNX sessions with malformed cross-session interfaces, so the runtime must
reject them at model load rather than relying on inference to fail.

Run `unit_tests --gtest_filter=*Marian*` after building the native tests.
