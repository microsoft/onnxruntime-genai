# Marian source-projection caching

With a compatible model export, the encoder computes source-only projections
once and the decoder reuses them at every step. Each generator owns its cache.
GenAI does not change model graphs or weights, or cache translations across
requests.

## Model interface

Encoder outputs and decoder inputs prefixed with `cached_source_projection_`
must have matching names. GenAI checks both sessions independently. Any number
of projections is allowed, and numbering need not be contiguous.

Each pair must use the same FP32 or FP16 type and
`[batch, source, projection]` shape:

- All dimensions must match, including names for dynamic dimensions.
  A dimension cannot be fixed on one side and dynamic on the other.
- Batch and source dimensions must also match the encoder's normal output.
- Projection width must be a positive constant, but can differ from the
  encoder hidden size.

Before allocating each cache tensor, GenAI checks fixed dimensions against the
request and resolves dynamic dimensions from the encoder input. Tensors use
the declared type and projection width and live until their generator is
destroyed; they are never shared between generators.

Models without these names run unchanged. Caching supports CPU execution with
one beam and greedy decoding (`do_sample=false`, `top_k=1`, or `temperature=0`).
It uses the generator's existing greedy-mode check. Stochastic sampling and
other providers are not supported.

## Memory and performance

Cache memory per generator is
`batch * source_length * sum(projection_width * element_bytes)`, including
source EOS and padding. For six FP32 projections of width 512, this is
`batch * source_length * 12288` bytes. Memory grows with batch size, source
length and live generators, not output length.

The projections still run once in the encoder; the saving comes from reusing
them in the decoder. Cache setup can outweigh the savings for short outputs.
Compare cached and uncached models with the same weights, inputs, runtime,
threading and search settings. Include encoder and generator setup time, and
measure peak process memory as well as cache size. Results depend on source and
output lengths, batch size and concurrent generators.

## Tests

Regenerate the small ONNX test graphs with:

```powershell
python test\python\create\create_marian_cache_models.py --output-dir test\models\marian-cache
```

These test models check correctness, not translation speed or quality. Their
outputs depend on every cache tensor, source tokens and decode step. Tests
cover independent generators, different shapes, generator recreation, FP16,
different projection counts and widths, and the uncached path. Invalid
encoder/decoder interfaces must fail at model load.

The padded-batch test uses both attention masks and rows that reach EOS at
different steps. It checks exact tokens and cached/uncached output equality
after early EOS, with both row orders.

Run `unit_tests --gtest_filter=*Marian*` after building the native tests.
