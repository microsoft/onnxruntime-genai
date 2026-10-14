# NVTX profiling ranges

GenAI can emit opt-in NVTX3 host-thread ranges for NVIDIA Nsight Systems. This
instruments the batching Engine, not the separate Generator API execution loop.
`ENABLE_NVTX` defaults to `OFF`: disabled builds do not search for or include NVTX
headers, evaluate range labels, or collect timing samples.

## Build

NVTX3 is header-only. Obtain its headers from a CUDA toolkit or a standalone
[NVTX distribution](https://github.com/NVIDIA/NVTX). No NVTX library or CUDA
execution provider is required to enable the host ranges.

Use the normal build driver, adding the CMake options below (and `--use_cuda`
and the usual CUDA options when building for CUDA):

```sh
python build.py --config RelWithDebInfo --cmake_extra_defines ENABLE_NVTX=ON ORTGENAI_NVTX_INCLUDE_DIR=/path/to/include
```

`ORTGENAI_NVTX_INCLUDE_DIR` must contain `nvtx3/nvToolsExt.h`, not the legacy
NVTX2 headers. For example, on Windows it can be
`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/include` (quote the
complete `ORTGENAI_NVTX_INCLUDE_DIR=...` argument when it contains spaces).
CMake also searches `NVTX_ROOT`, `CUDA_PATH`, and the configured CUDA toolkit
include directory. An enabled build fails at configuration if headers are missing;
it never silently disables instrumentation. There is no automatic download.

## Capture And Interpret

Launch the application using the instrumented GenAI build under Nsight Systems:

```sh
nsys profile --trace=cuda,nvtx --output=genai-engine <application> <arguments>
```

Open the report's NVTX and CUDA timelines. Thread-local scope guards close ranges
on normal return, early return, and exception unwinding. Labels contain no prompts,
tokens, or request identifiers. Enabled builds issue NVTX calls even without a
collector attached; use `ENABLE_NVTX=OFF` for builds without instrumentation.

| Range | Host work covered |
| --- | --- |
| `Engine::Step` | One invocation of the dynamic Engine step, including no-work paths |
| `PlanStep`, `ReserveStep` | Scheduling and cache reservation/validation |
| `BeginTransaction`, `TargetDecode`, `GenerateNextTokens` | Checkpointing, target execution, and token selection |
| `PrepareCommit`, `PrepareDflash2Feeds`, `CommitState`, `PublishDflash2Drafts` | Commit preparation, state publication, and optional drafting |
| `TargetBuildInputs`, `TargetSessionRun` | SimpleDecoder input construction and the target session's `Run` call; also present for static batching |
| `D2Layout`, `D2PackAux`, `D2Embed` | DFlash admission/layout, auxiliary packing, and token/embedding preparation |
| `D2Uploads`, `D2SessionRun`, `D2Readback` | Metadata preparation/uploads and bindings, drafter `Run`, then cache bookkeeping, output readback, and host draft selection |

These are **host elapsed intervals**, not GPU kernel durations. Session execution
and copies can enqueue asynchronous work, or wait for it depending on the existing
execution path. In particular, captured asynchronous runs may finish their host
range before GPU execution finishes; readback may include synchronization. GPU
work launched during packing, embedding, or uploads can overlap later host ranges.
Inspect CUDA kernels, copies, streams, and graph activity for device timing. Nested
host ranges are inclusive and must not be summed as independent durations. The
instrumentation adds no synchronization and makes no performance-improvement claim.

## CPU-Only Header Tests

These tests require neither ONNX Runtime nor a model or GPU. They also run in the
normal `ENABLE_TESTS` build. The enabled configuration tests actual NVTX3 header
integration plus a declaration-only counting backend for nested scopes, phase
transitions, empty/idempotent closure, exceptions, and early return:

```sh
cmake -S test/nvtx -B build/nvtx-tests -DENABLE_NVTX=ON -DORTGENAI_NVTX_INCLUDE_DIR=/path/to/include
cmake --build build/nvtx-tests --config Release
ctest --test-dir build/nvtx-tests -C Release --output-on-failure
```

To verify dependency-free builds, configure another directory with
`-DENABLE_NVTX=OFF`, omit the NVTX include option, and run the same build/test
commands. The disabled test also verifies that label expressions are not evaluated.