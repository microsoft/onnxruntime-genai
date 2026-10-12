# Nemotron-H exported graphs on CUDA

The `nemotron_h` model type uses the decoder-only runtime with sparse attention
KV discovery and fixed convolution/SSM state management. It is distinct from
the dense `nemotron` and the transducer `nemotron_speech` model types. This is
runtime support for already-exported ONNX graphs, not a new Python model-builder
architecture or support for native NVFP4 checkpoints.

## Nemotron 3.5 Lightning BF16 interface

The experimental Mobius paired task `nemotron-h-mtp` exports
`decoder/model.onnx` and `mtp/model.onnx`. The pinned BF16 checkpoint is
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at revision
`a9904d24bcc1d289a1950fa9d2b978c47cf903b9`.

Configure the target decoder with `num_hidden_layers: 52`, `hidden_size: 2688`,
`num_attention_heads: 32`, `num_key_value_heads: 2`, and `head_size: 128`.
Its vocabulary size is 131072 and its pinned context limit is 262144. A short
generation `max_length` does not change that model context limit.

The target exposes attention KV at global layers 5, 12, 19, 26, 33 and 42.
The runtime discovers these indices from the session; do not renumber the
cache tensors or allocate attention KV for every layer. Omit `decoder.layer_types`:
the convolution-cache path for LFM2 treats every non-attention layer as a
convolution layer, which is not the contract of cacheless Nemotron MoE layers.

Configure the Mamba cache and the post-final-normalization seed explicitly:

```json
{
  "inputs": {
    "past_conv_names": "past_key_values.%d.conv_state",
    "past_recurrent_names": "past_key_values.%d.ssm_state"
  },
  "outputs": {
    "present_conv_names": "present.%d.conv_state",
    "present_recurrent_names": "present.%d.ssm_state",
    "hidden_states": "mtp_seed"
  }
}
```

These fields belong under `model.decoder`. Each Mamba layer has a BF16
`[batch,6144,3]` convolution carry and a **FP32** `[batch,64,128,64]` SSM carry.
Their types come from the session independently; do not cast the SSM carry
to the model's BF16 activation type.

## Optional MTP head

Use the existing [MTP generator](speculative_decoding.md), not a separate
draft-model acceptance implementation. The main configuration maps:

```json
{
  "prefill_head": true,
  "filename": "mtp/model.onnx",
  "num_hidden_layers": 1,
  "num_key_value_heads": 2,
  "head_size": 128,
  "main_hidden_states": "mtp_seed",
  "outputs": { "hidden_states": "mtp_hidden" }
}
```

This block belongs under `model.mtp`; filenames are relative to the
configuration's package directory. Load the head as a separate model whose
decoder has one attention-cache layer, hidden size 2688, two KV heads and head
size 128, with `inputs.hidden_states: "hidden_states"` and
`outputs.hidden_states: "mtp_hidden"`. Its `model.type` can also be
`nemotron_h`. The physical graph also contains a cacheless MoE block.

The controller pairs target `h_i` with token `t_(i+1)` to predict `t_(i+2)`.
`prefill_head: true` initializes prompt context using `tokens[:, 1:]` with
`mtp_seed[:, :-1]`, including shifted alignment across prefill chunks.
The target must emit the complete prompt hidden sequence, not only its final row.
Both fixed and symbolic incomplete outputs report the explicit MTP contract
error. Prompt validation allocates the selected output on the session device
before checking its actual shape, rather than pre-binding the expected row count.
The head owns independent layer-0 attention KV; neither target attention KV
nor target convolution/SSM carry is a head input. The serialized graphs have
separate embedding and LM-head copies, so do not declare shared initializers
unless a different export explicitly provides that ABI.

Begin with `max_draft_tokens: 1`, greedy search, batch size 1 and one beam.
Beam search is rejected for discovered recurrent-state models because their
state manager does not perform beam reordering. The unwindowed export requires
snapshot/restore and replay on rejected verification, not KV-only cropping.
For dynamic concatenating cache exports, set
`search.past_present_share_buffer: false` and disable CUDA graph capture.
Shared-buffer and captured execution require an export with that cache
contract and independent validation; a configuration boolean does not provide it.

## Validation boundary

The checked-in weight-free fixture exercises sparse global layer indices,
mixed BF16/FP32 carry, post-normalization hidden/token alignment, reused-state
generation, reset, and accepted/rejected one- and multi-token MTP rounds on
CPU and CUDA. An independently carried token counter checks target KV length
and the head's one-token-lag cache, so incorrect cropping or borrowed target
KV changes the expected output. It is a controller/state regression, not
pretrained-model parity.

Regenerate this fixture with NumPy and `onnx-ir` installed:
`python test/models/nemotron-h/generate_model.py`. No checkpoint download is needed.

At the published Mobius PR #761 head
`e3d01a7712f38deb962e7107752da7fb668a7054`, the supplied full decoder scorecard
still fails 141 of 240 comparisons against the unchanged finite,
NRMSE <= 0.03 and cosine >= 0.999 gates. Later unpublished exporter candidates
do not establish readiness of that published head. Short token agreement
and a loadable package are insufficient to clear full numerical and longer
generation gates. Keep the exporter package's `runtime_unvalidated` status
until those gates and real-weight runtime generation pass.

A native CUDA probe of that same pinned export reached 257 total tokens with
ordinary greedy decoding, but N1 MTP diverged at token index 51. A subsequent
64-token control confirmed the same divergence for both N1 and N2. A separate
target-only diagnostic reproduced the difference without running the head:
with the same 50-token cached prefix, width 1 predicted token 20166 while widths
2 and 3 predicted token 9314 for the first row. Target single-token versus wide
forward consistency remains a blocking numerical gate, not a reason to add
compensating head alignment or acceptance logic.

Layer-boundary probes at that prefix first locate a BF16 activation difference
at MoE layer 17, following small FP32 router MatMul reduction differences with
identical input rows and unchanged selected experts. These differences amplify
in later layers, but do not establish the cause of every subsequent mismatch.
Disabling CUDA TF32 removes that isolated activation difference; the full
257-token control still diverges at indices 208 (N1) and 109 (N2).
A private graph using FP64 router MatMuls also fails, at indices 105 and 90.
Neither experiment establishes a production precision policy or HuggingFace
parity. A further private normalization-unfusion control with graph optimization
disabled timed out during loading at 600 seconds, before generation; its
numerical result is untested. An optimized control using the corrected standard
RMS kernel also timed out at 600 seconds. A separate 180-second startup trace
completed graph transformation and reported 34077 CUDA nodes and 94 CPU nodes;
allocation planning was its last logged stage before timeout. This localizes
the startup investigation, not a numerical result or a proven planner defect.
The published export was not modified.

A subsequent isolated normalization probe identifies a separate CUDA operator
issue: standard ONNX `RMSNormalization` must round the standardized value to
the input type before applying its scale. Correcting this stage boundary
matches an independent two-stage reference exactly for the tested FP16/BF16
component cases. Historical `SkipSimplifiedLayerNormalization` semantics remain
unchanged, including the pinned export's pre-fused nodes; this component fix
does not establish full-checkpoint parity or replace the earlier supplied
scorecard. With the corrected standard RMS kernel and CUDA TF32 disabled, the
unchanged export again reaches 257 ordinary tokens, but N1 and N2 MTP both
diverge at index 64, accepting 87/129 and 122/169 drafts respectively.

A further CUDA Softplus correction preserves small positive tails with
`log1p`, uses FP32 intermediates for low-precision inputs, and retains FP64
computation for double inputs. Four independent precision regressions fail
before that correction and pass after it, with CPU fallback disabled.
Against the exact corrected provider, the 21 public tests, 34 engine tests
and eight independent CUDA fixture cases remain green. The unchanged
checkpoint still fails the full 257-token control: N1 first diverges at index
70 with 86/131 accepted drafts, and N2 at index 100 with 107/179. All three
sequences contain 257 tokens. Component corrections do not clear this
generation gate; the supplied full-model reference scorecard was not rerun.

The Sigmoid-stage provider also corrects negative-branch cancellation in CUDA Sigmoid,
used by the pinned decoder's 23 FP32 router nodes. Its four independent
precision regressions and the same 21 public, 34 engine and eight CUDA fixture
cases pass. A fresh unchanged-export control against this exact final provider,
with CUDA TF32 and graph capture disabled, still fails ordinary/MTP equivalence:
N1 first diverges at index 90 with 90/126 accepted drafts, and N2 at index 119
with 119/167. Ordinary, N1 and N2 sequences all contain 257 tokens. These are
zero-based indices including the 40-token prompt. The 70/100 divergence above
was measured before the Sigmoid correction; neither result clears the longer
generation gate. A subsequent unchanged-gate rerun against that provider fails
139/240 comparisons under both TF32 policies, with maximum NRMSE
0.10676754207476785 and minimum cosine 0.9942840098112892. The reference uses
independent FP32 source-checkpoint equations, not a direct HuggingFace
model-class invocation. CPU placements are verified integer-only.

The later ORT provider at `0acd2ee5ca0d738767d021be52d1e816ad70913b` also
corrects cancellation and intermediate rounding in fused QuickGelu/SiLU.
Profiling confirms that the decoder's 23 FP32 Swish gates execute through this
fused CUDA kernel. The combined provider suites pass 224 cases with 13 existing
MoE skips; the same 21 public, 34 engine and eight independent CUDA fixture
cases pass against its exact binary. Its CUDA provider SHA256 is
`c54d757853ad295a933d20a55435341694fd1e7845695117f0371f27cc9315ea`.

Full-checkpoint gates still fail against that provider. With TF32 and graph
capture disabled, ordinary, N1 and N2 each contain 257 tokens. N1 first diverges
at zero-based index 109, accepting 92/125 drafts; N2 at index 92, accepting
121/165. Indices include the 40-token prompt. Under the original TF32-on policy,
the unchanged FP32-equation reference gates fail 141/240 comparisons, with
maximum NRMSE 0.1106184779356323 and minimum cosine 0.9938640906045041.
Floating-point computation remains CUDA-only. The decoder/head graph hashes
are unchanged. The earlier 90/119 divergence and 139/240 scorecard failures
predate QuickGelu's correction; neither historical target-only probes nor
component improvements establish the cause of these current failures.

A separate direct upstream reference now runs the unmodified
`transformers==5.19.0` `NemotronHForCausalLM` on CUDA with the same pinned
checkpoint, BF16 parameters, eager attention and TF32 enabled. All decoder
weights load without missing or mismatched keys. The Mamba implementation
uses upstream's PyTorch reference path because `mamba_ssm` is unavailable.
Four forwards carry the upstream cache independently; logits and final
normalized hidden are compared against the saved post-QuickGelu native outputs.
Only 1/8 comparisons passes the unchanged NRMSE/cosine thresholds: prefill
logits pass, but the hidden seed and every cached-step output fail.
Maximum NRMSE is 0.11662261092503201, minimum cosine 0.9941554758121095 and
maximum absolute hidden error 5.5. All sampled logit argmaxes agree; that does
not clear numerical or generation gates. This direct BF16 reference is
additional evidence, not a replacement for the 240-output FP32-equation
scorecard, a full-cache comparison or a full upstream generation test.

An ORT review follow-up at `8c36c6c8e8a7ec43fe503491b82778c2bddce262` fixes
the default FP32 stash for double-input standard RMSNormalization and preserves
double precision in scaling. Explicit double stash remains available for
double inputs/scales; other unsupported stash choices now fail explicitly.
The strict double-stash regression fails before this fix and passes after it.
Against provider SHA256
`068c67babdffb1e14a6ed0c5d32be352ac0bc1be8f3d1de8ca1e87e5dfb812f2`,
the coupled provider suites pass 227 cases with 13 existing skips, and the same
21 public, 34 engine and eight independent CUDA fixture cases pass. The
full-checkpoint measurements above remain explicitly tied to the earlier
post-QuickGelu provider; no new full-model pass is inferred from this follow-up.

A GenAI review regression also reproduces a symbolic last-row-only output
failing inside ORT because its decoder buffer was pre-bound to the full prompt
length. Prompt validation now allocates that selected output on the session
device, checks its actual shape, and copies valid output into the stable buffer.
CUDA copies finish before the temporary allocation is released. Both fixed and
symbolic incomplete seeds are rejected explicitly with and without chunking.
Against the final ORT provider above, 23 public tests, 34 engine tests and eight
independent CUDA fixture cases pass. Full symbolic seeds retain accepted/rejected
N1/N2 generation, chunk alignment, one-token prompts and reset coverage.

A private precision control changes only the initial standard RMS node to
multiply its scale in FP32 before the final BF16 cast. It completes against the
final ORT provider but passes 0/8 direct upstream comparisons and 168/240
FP32-equation comparisons, leaving 72 failures. This is not a fix or a published
exporter candidate; the original decoder/head graphs remain unchanged and the
full-checkpoint gates remain open.
