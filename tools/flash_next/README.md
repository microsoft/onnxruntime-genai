# Flash Next WebGPU GenAI companion

Use with `microsoft/onnxruntime` branch
`sayanshaw/flash-next-webgpu-portable`; see its `tools/flash_next/README.md`.

Includes the previously uncommitted Engine/cache compatibility changes and
the measured GPU-greedy/UInt32-Slice runtime source overlays in normal source
locations. The CPU eight-lane argmax fallback is also retained.
All model weights, Linux binaries, build trees and credentials are excluded.

`source-manifest.json` records source provenance. The historical Linux runtime
reference hash is not a promise of byte-identical Windows output.
This publication is a source snapshot; a combined Windows build and target
model/backend compatibility still require validation.

Preserve the lazy-logits lifecycle, CPU-authoritative sampling/penalty/API
fallbacks, minimum-length EOS suppression and GPU-copy range/type checks.
Do not remove these behaviors to obtain a faster benchmark.

Build matching ORT/GenAI through repository-supported scripts. Do not mix
unrelated released wheels, SDK headers or old private Linux libraries.
New RC/DFlash/custom CUDA graphs under `C:\models` may need a separate model
compatibility port before these older WebGPU runtime sources can execute them.
