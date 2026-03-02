# ONNX Runtime GenAI — Feature Specification

> Comprehensive catalog of current capabilities, potential features, known issues, and implementation suggestions.
>
> Generated: March 2026

---

## Table of Contents

1. [Current Features](#1-current-features)
   1. [Model Architectures](#11-model-architectures)
   2. [Inference Pipeline](#12-inference-pipeline)
   3. [Search & Sampling Strategies](#13-search--sampling-strategies)
   4. [Multi-Modal Support](#14-multi-modal-support)
   5. [Engine & Batching](#15-engine--batching)
   6. [KV Cache Management](#16-kv-cache-management)
   7. [Constrained Decoding](#17-constrained-decoding)
   8. [LoRA / Adapter Support](#18-lora--adapter-support)
   9. [Hardware Acceleration](#19-hardware-acceleration)
   10. [Language Bindings & APIs](#110-language-bindings--apis)
   11. [Tokenization](#111-tokenization)
   12. [Logging, Tracing & Profiling](#112-logging-tracing--profiling)
   13. [Configuration System](#113-configuration-system)
   14. [Benchmarking](#114-benchmarking)
   15. [Continuous Decoding](#115-continuous-decoding)
2. [Potential Features & Improvements](#2-potential-features--improvements)
   1. [Speculative Decoding](#21-speculative-decoding)
   2. [Stable Diffusion / Diffusion Models](#22-stable-diffusion--diffusion-models)
   3. [Tensor Parallelism & Multi-GPU](#23-tensor-parallelism--multi-gpu)
   4. [Asynchronous / Streaming Generation](#24-asynchronous--streaming-generation)
   5. [Memory Profiling & Optimization](#25-memory-profiling--optimization)
   6. [Batch Size Auto-Tuning](#26-batch-size-auto-tuning)
   7. [Expanded Model Architecture Support](#27-expanded-model-architecture-support)
   8. [Multinomial & Advanced Sampling](#28-multinomial--advanced-sampling)
   9. [CPU Top-K Fallback](#29-cpu-top-k-fallback)
   10. [iOS Support](#210-ios-support)
   11. [AMD GPU (ROCm) Acceleration](#211-amd-gpu-rocm-acceleration)
   12. [Improved C# Ecosystem](#212-improved-c-ecosystem)
   13. [Enhanced Documentation](#213-enhanced-documentation)
   14. [Expanded Test Coverage](#214-expanded-test-coverage)
   15. [Model Validation & Diagnostics](#215-model-validation--diagnostics)
   16. [Dynamic Thread Scaling & NUMA Awareness](#216-dynamic-thread-scaling--numa-awareness)
   17. [Prefix Caching](#217-prefix-caching)
   18. [Function Calling / Tool Use Framework](#218-function-calling--tool-use-framework)
   19. [Quantization-Aware Generation](#219-quantization-aware-generation)
   20. [Prompt Caching / Session Persistence](#220-prompt-caching--session-persistence)
   21. [Embedding Extraction API](#221-embedding-extraction-api)
   22. [Token Probability / Logprobs API](#222-token-probability--logprobs-api)
   23. [OpenAI-Compatible Server](#223-openai-compatible-server)
3. [Known Issues & Gaps](#3-known-issues--gaps)
4. [Implementation Suggestions](#4-implementation-suggestions)

---

## 1. Current Features

### 1.1 Model Architectures

The library supports 28+ model architectures across five categories:

| Category | Count | Models |
|----------|-------|--------|
| **LLM** (Language) | 21 | ChatGLM, DeepSeek, ERNIE, Gemma, GPT2, gpt-oss, Granite, InternLM2, Llama, Mistral, Nemotron, OLMo, Phi, Phi3, Phi3Small, PhiMoE, Qwen2, Qwen3, SmolLM, SmolLM3, Whisper-Decoder |
| **VLM** (Vision-Language) | 4 | Fara, Gemma3, Phi3v, Qwen2.5-VL |
| **ALM** (Audio-Language) | 1 | Whisper |
| **MMM** (Multi-Modal) | 1 | Phi4MM (vision + audio) |
| **Pipeline** | 1 | Decoder-only pipeline (multi-stage) |

**Key files:** `src/models/model_type.h`, `src/models/model.cpp`

### 1.2 Inference Pipeline

The core inference flow is:

```
Model Loading → Tokenization → Generator Creation → Token Generation Loop → Decoding
```

**Components:**
- **Model** (`src/models/model.h`): Loads ONNX sessions, manages device interfaces, creates states
- **Generator** (`src/generators.h`): Orchestrates the full generation loop with `GenerateNextToken()`, `AppendTokens()`, `RewindToLength()`
- **State** (`src/models/model.h`): Abstract base for model-specific execution logic; each architecture implements `Run()`
- **Config** (`src/config.h`): Reads `genai_config.json` for model metadata, search parameters, and session options
- **Graph Builder/Executor** (`src/models/graph_builder.h`, `graph_executor.h`): Dynamic ONNX graph construction and cached session execution for operations like Cast, TopK, Argmax

### 1.3 Search & Sampling Strategies

| Strategy | Implementation | Description |
|----------|---------------|-------------|
| **Greedy Search** | `GreedySearch_Cpu` | Deterministic argmax selection |
| **Beam Search** | `BeamSearch_Cpu` | Multi-hypothesis search with beam scorer, length penalty, early stopping |
| **Top-K Sampling** | `SampleTopK()` | Samples from top-K most probable tokens |
| **Top-P (Nucleus)** | `SampleTopP()` | Samples from smallest set of tokens with cumulative probability ≥ p |
| **Top-K + Top-P** | `SampleTopKTopP()` | Combined filtering |
| **Temperature** | Applied in softmax | Scales logit distribution before sampling |
| **Repetition Penalty** | `ApplyRepetitionPenalty()` | Penalizes previously generated tokens |
| **Min Length** | `ApplyMinLength()` | Forces minimum generation length |

**CUDA-optimized top-K** includes 9 specialized algorithms:
- Flash Convergent (k ≤ 64, optimal for small k)
- Cascaded Sort (k 16–64, H200-tuned)
- Iterative Sort (k 32–64, adaptive reduction)
- Hybrid Sort (k > 64, two-kernel cooperative)
- Distributed Select Sort (large vocab + small k)
- Per-Batch Radix Sort (small batches)
- Full Sort (fallback, CUB radix)
- Select Sort (debugging/baseline)

A **three-tier benchmark caching** system selects the best algorithm at runtime: local array cache → global thread-safe cache → online benchmarking.

**CUDA sampling** implements fused and multi-stage kernels:
- Fused kernel (k ≤ 256): All steps in shared memory — softmax, CDF, filtering, re-normalization, sampling
- Multi-stage kernel (k > 256): Pipeline decomposition for cache locality

**Key files:** `src/search.h`, `src/cuda/cuda_topk*.cuh`, `src/cuda/cuda_sampling.cu`

### 1.4 Multi-Modal Support

**Vision-Language Models:**
- **Phi3v**: Image preprocessing via `PhiImageProcessor`, embedding fusion through `MultiModalLanguageModel`
- **Qwen2.5-VL**: Specialized `QwenImageProcessor` with spatial merge support, 3D MRoPE position IDs
- **Gemma3**: Image processing via `GemmaImageProcessor`
- **Fara**: Uses Qwen VL pipeline
- **Phi4MM**: Full multi-modal (vision + audio) with separate processor pipelines

**Architecture:** Multi-modal models use `VisionState`, `SpeechState`, `EmbeddingState`, and `DecoderState` substates coordinated by `MultiModalLanguageModel`.

**Audio-Language Models:**
- **Whisper**: Encoder-decoder architecture with `AudioEncoderState` for feature extraction and `WhisperDecoderState` for cross-attention decoding
- Audio feature extraction uses ONNX Runtime Extensions (`OrtxFeatureExtractor`)

**Key files:** `src/models/multi_modal.h`, `src/models/phi_image_processor.h`, `src/models/whisper.h`

### 1.5 Engine & Batching

The **Engine** (`src/engine/engine.h`) provides production-grade concurrent request execution:

| Component | Responsibility |
|-----------|---------------|
| **Engine** | Request lifecycle management, batch orchestration via `Step()` |
| **Scheduler** | Determines which requests to batch together |
| **ModelExecutor** | Actual ONNX Runtime inference execution |
| **CacheManager** | KV cache allocation and reuse |

**Scheduling strategies:**
- **StaticBatchScheduler**: Fixed batch size, simpler logic
- **DynamicBatchScheduler**: Variable batch size, opportunistic batching for higher throughput

**Decoder implementations (engine layer):**
- **VarlenDecoderIO**: Variable-length sequences with PagedAttention operator support
- **StaticBatchDecoderIO**: Fixed batch with padding, standard attention mask

**Key files:** `src/engine/engine.h`, `src/engine/scheduler.h`, `src/engine/decoders/`

### 1.6 KV Cache Management

| Strategy | Class | Use Case |
|----------|-------|----------|
| **Default KV Cache** | `DefaultKeyValueCache` | Standard layer-wise K/V tensors |
| **Combined KV Cache** | `CombinedKeyValueCache` | Merged K/V for GPT-style models |
| **Cross Cache** | `CrossCache` | Encoder-decoder cross-attention |
| **Model-Managed** | `ModelManagedKeyValueCache` | Stateful models that manage their own cache |
| **Windowed KV Cache** | `WindowedKeyValueCache` | Sliding window for long-context efficiency |
| **Paged KV Cache** | `PagedKeyValueCache` | Block-based allocation for dynamic batching |

Paged attention supports block tables, sparse indexing, and per-layer block pools for concurrent request memory management.

**Key files:** `src/models/kv_cache.h`, `src/models/windowed_kv_cache.h`, `src/engine/paged_key_value_cache.h`

### 1.7 Constrained Decoding

Uses **LLGuidance** integration for structured output generation:

| Constraint Type | Description |
|----------------|-------------|
| **JSON Schema** | Force output to conform to a JSON schema |
| **Regex** | Constrain output to match a regular expression |
| **Lark Grammar** | General-purpose context-free grammar constraints |

**Implementation:** `GuidanceLogitsProcessor` masks invalid tokens (sets to -inf) at each step, with async mask computation for performance. Requires `special: true` in `tokenizer.json` for custom tool tokens.

**Build flag:** `USE_GUIDANCE=ON` (off by default)

**Key files:** `src/constrained_logits_processor.h`, `docs/ConstrainedDecoding.md`

### 1.8 LoRA / Adapter Support

- `LoadAdapter(path, name)`: Load LoRA weights from file
- `UnloadAdapter(name)`: Remove adapter
- Reference counting via `AcquireAdapter()` / `ReleaseAdapter()`
- Multiple simultaneous adapters supported (Multi-LoRA)
- Exposed in C, Python, C#, Java, and Objective-C APIs

**Key files:** `src/models/adapters.h`

### 1.9 Hardware Acceleration

| Backend | Status | Platform | Key Features |
|---------|--------|----------|-------------|
| **CPU** | ✅ Stable | All | Baseline, always available |
| **CUDA** | ✅ Stable | Linux, Windows | 9 top-k algorithms, fused sampling kernels, cooperative launches |
| **DirectML** | ✅ Stable | Windows | GPU graph capture, shader-based kernels, adapter selection |
| **NvTensorRtRtx** | ✅ Supported | Linux, Windows | TensorRT-RTX optimized inference |
| **OpenVINO** | ✅ Supported | Linux, Windows | Intel hardware optimization |
| **QNN** | ✅ Supported | Windows (ARM64) | Qualcomm Neural Network for Snapdragon |
| **WebGPU** | ✅ Supported | Browser | Web-based GPU inference |
| **RyzenAI** | ✅ Supported | Windows | AMD AI accelerator |
| **WinML** | ✅ Supported | Windows | Windows ML integration (requires SDK version) |

**Device abstraction:** `DeviceInterface` in `src/smartptrs.h` provides a uniform API across all backends, including `DeviceBuffer`/`DeviceSpan` for device-agnostic memory management.

### 1.10 Language Bindings & APIs

| Language | Status | Key Classes | Notes |
|----------|--------|-------------|-------|
| **C/C++** | ✅ Stable | Full API via `ort_genai_c.h` / `ort_genai.h` | Zero-cost C++ wrappers with automatic resource management |
| **Python** | ✅ Stable | Model, Generator, Tokenizer, MultiModalProcessor | Plus model builder framework for 15+ architectures |
| **C#** | ✅ Stable | 25+ classes including `OnnxRuntimeGenAIChatClient` | Semantic Kernel integration via `Microsoft.Extensions.AI` |
| **Java** | ✅ Source-build | Model, Generator, Tokenizer, SimpleGenAI | Android-first with Gradle, JNI bindings |
| **Objective-C** | 🔧 In Development | Model, Generator, Config, TokenizerStream | iOS/macOS framework with `.mm` implementations |

**C API surface** exposes 15+ element types (float32, int32, float16, bfloat16, etc.) and opaque handles for all core types.

### 1.11 Tokenization

- Encode (single and batch) with special token handling
- Decode (single and batch)
- Streaming decode via `TokenizerStream`
- Chat template application
- BOS/EOS/PAD token management
- Built on ONNX Runtime Extensions tokenizer

### 1.12 Logging, Tracing & Profiling

| Feature | Mechanism |
|---------|-----------|
| **Logging** | Global `LogItems` config: model inputs/outputs, logits values, statistics; ANSI colors; file output; callbacks |
| **Tracing** | Perfetto-compatible trace sink (`ENABLE_TRACING=ON`); `DurationTrace` RAII wrapper; `ORTGENAI_TRACE_FILE_PATH` env var |
| **Profiling** | Per-token runtime profiling via `enable_profiling` runtime option; custom file prefixes |

### 1.13 Configuration System

- **`genai_config.json`**: Model metadata, encoder/decoder config, vision/speech processor config, search parameters, session/run options
- **Runtime Settings** (`src/runtime_settings.h`): Dynamic settings that can't be in config.json — handle-based, serializable to JSON overlay
- **Provider configuration**: Hardware-specific options (OpenVINO device, DML adapter, QNN vendor) via config or runtime API

### 1.14 Benchmarking

| Tool | Language | Features |
|------|----------|----------|
| **C Benchmark** | C | E2E latency (p50/p90/p99/stddev), per-token metrics, warmup, TensorRT support |
| **Python Benchmark** | Python | E2E continuous testing, multimodal support, JSON prompt library |

### 1.15 Continuous Decoding

- `RewindToLength()` on `Generator` to discard part of a conversation and resume
- Session continuation via model state management (`State::RewindTo()`)
- Cached generator in C# `OnnxRuntimeGenAIChatClient` for multi-turn chat

---

## 2. Potential Features & Improvements

### 2.1 Speculative Decoding

**Status:** On the README roadmap

**Description:** Use a smaller "draft" model to generate candidate tokens in parallel, then verify with the larger "target" model. Accepted tokens skip full-model computation, yielding 2–3× speedups for large models.

**Suggestion:**
- Add a `DraftModel` concept that can be attached to a `Generator`
- Implement a `SpeculativeSearch` strategy alongside existing Greedy/Beam search
- The engine's `Scheduler` already supports multi-request batching; extend it to batch draft + verify passes
- Key decision: support self-speculative decoding (early exit from layers of the same model) vs. separate draft model, or both
- Integration point: `src/search.h` — add `SpeculativeSearch` class; `src/generators.h` — add `SetDraftModel()` API

### 2.2 Stable Diffusion / Diffusion Models

**Status:** Listed as "Under development" in README

**Description:** Support image-generation diffusion models (Stable Diffusion, SDXL, FLUX) through the same API surface.

**Suggestion:**
- Diffusion models have a fundamentally different loop (denoising iterations vs. autoregressive token generation)
- Add a `DiffusionGenerator` parallel to the existing `Generator`
- Requires new pipeline components: text encoder, VAE decoder, U-Net/Transformer scheduler
- The existing `MultiModalProcessor` pattern can be reused for text/image conditioning
- Consider a plugin architecture where diffusion-specific logic is loaded on demand

### 2.3 Tensor Parallelism & Multi-GPU

**Status:** Not currently implemented (single-device assumed)

**Description:** Distribute model layers or tensor operations across multiple GPUs for models that don't fit in a single GPU's memory.

**Suggestion:**
- The `DeviceInterface` abstraction already supports multiple device types; extend it to support multiple instances of the same device type
- Add a `DistributedState` that shards the model across devices
- CUDA's NCCL library can handle inter-GPU communication
- Start with pipeline parallelism (different layers on different GPUs) as it's simpler than tensor parallelism
- Integration point: `src/models/decoder_only_pipeline.h` already supports multi-stage execution — extend with cross-device stages

### 2.4 Asynchronous / Streaming Generation

**Status:** Partial — `TokenizerStream` exists for token-by-token streaming; no async generator

**Description:** Fully async generator API that yields tokens via callbacks or async iterators without blocking the calling thread.

**Suggestion:**
- Add `GenerateNextTokenAsync()` returning a future/promise
- Python: expose as an `async for` iterator (`__aiter__` / `__anext__`)
- C#: expose as `IAsyncEnumerable<string>`
- The engine already runs on a separate thread; surface this via the public API
- Consider callback-based API: `generator.OnToken(callback)` for all languages

### 2.5 Memory Profiling & Optimization

**Status:** No memory profiling infrastructure

**Description:** Track peak memory usage, allocation patterns, and fragmentation to help users right-size their hardware.

**Suggestion:**
- Add memory tracking to `DeviceInterface::Allocate()` — record allocation sizes and lifetimes
- Expose via API: `model.GetMemoryUsage()` returning peak, current, and allocated bytes per device
- Add a `ENABLE_MEMORY_PROFILING` build flag to avoid overhead in production
- Report memory breakdown by component: KV cache, model weights, activations, temporary buffers
- Integration point: `src/smartptrs.h` — wrap `DeviceBuffer` allocations with tracking

### 2.6 Batch Size Auto-Tuning

**Status:** Batch size is user-specified; no automatic tuning

**Description:** Automatically determine the optimal batch size based on available memory and model size.

**Suggestion:**
- At model load time, run a small calibration pass to determine memory per request
- Calculate max batch size as `(available_memory - model_weights) / memory_per_request`
- Integrate with the `DynamicBatchScheduler` to set upper bounds automatically
- Expose as `config.search.auto_batch_size = true` in `genai_config.json`

### 2.7 Expanded Model Architecture Support

**Priority additions based on community demand (GitHub issues):**

| Model | Category | Issue # | Notes |
|-------|----------|---------|-------|
| **Qwen3-VL** | VLM | #1989 | Detailed spec available; differs from Qwen2.5-VL in patch size, attention, MLP |
| **Gemma3n** | LLM | #1977 | Nano variant with different architecture |
| **MiniCPM** | VLM | — | Popular compact vision-language model |
| **Florence** | VLM | — | Microsoft's vision foundation model |
| **Moshi** | ALM | — | Audio generation model |
| **Janus** | MMM | — | Unified vision understanding and generation |

**Suggestion:**
- Each new model requires: entry in `model_type.h`, config template, `State` subclass, optional processor
- The existing `decoder_only.h` and `multi_modal.h` base classes handle most of the common logic
- Consider a model registration plugin system to allow third-party model additions without modifying core code

### 2.8 Multinomial & Advanced Sampling

**Status:** Only temperature-scaled softmax sampling; no Min-P, Typical, Mirostat, or classifier-free guidance sampling

**Description:** Additional sampling strategies that improve generation quality for different use cases.

**Suggestion:**
- **Min-P Sampling**: Filter tokens below `min_p * max_probability`; add to `Search` interface as `SampleMinP()`
- **Typical Sampling**: Select tokens whose information content is close to the expected information
- **Mirostat**: Adaptive perplexity-targeting sampler; maintains target surprise level
- **Classifier-Free Guidance (CFG)**: Run model twice (with/without conditioning), interpolate logits
- Add these as options in `genai_config.json` under `search`: `min_p`, `typical_p`, `mirostat_mode`, `mirostat_tau`
- Integration point: `src/search.h` — extend the `Search` interface; `src/cuda/cuda_sampling.cu` — add CUDA kernels

### 2.9 CPU Top-K Fallback

**Status:** All top-K algorithms are CUDA-only; CPU uses basic sorting

**Description:** For small batches on CPU, a SIMD-optimized top-K could avoid GPU kernel launch overhead.

**Suggestion:**
- Implement a `TopKCpu()` using partial sort (e.g., `std::nth_element` + sort of top-K)
- Use AVX2/AVX-512 intrinsics for the comparison-heavy inner loop
- Threshold: use CPU path when `batch_size * vocab_size < threshold` (e.g., 100K elements)
- Integration point: `src/search.h` — add CPU path in `SelectTop()`

### 2.10 iOS Support

**Status:** On the README roadmap; Objective-C bindings exist but iOS deployment is incomplete

**Description:** Full iOS support including CocoaPods/Swift Package Manager distribution.

**Suggestion:**
- The Objective-C bindings in `src/objectivec/` already wrap the core API
- Needs: iOS-specific build configuration, CoreML EP integration, framework packaging
- Publish via CocoaPods (pipeline infrastructure already exists for this in `.pipelines/`)
- Consider Swift wrapper for more idiomatic iOS API
- Test on real devices and simulators for ARM64 + Neural Engine

### 2.11 AMD GPU (ROCm) Acceleration

**Status:** On the README roadmap; ROCm build flags exist (`USE_ROCM`) but no dedicated kernels or testing

**Description:** First-class AMD GPU support via ROCm, including HIP-ported CUDA kernels.

**Suggestion:**
- The `cmake/check_rocm.cmake` already detects ROCm; needs kernel porting
- Use HIPify to auto-convert CUDA kernels in `src/cuda/` to HIP
- Key kernels to port: top-K algorithms, sampling, beam search
- Add ROCm-specific `DeviceInterface` implementation
- Add CI pipeline for ROCm testing (MI250/MI300 recommended)

### 2.12 Improved C# Ecosystem

**Status:** Only 1 test case vs. 30+ Python tests; `OnnxRuntimeGenAIChatClient` needs more features

**Description:** Comprehensive C# testing and API improvements for the .NET ecosystem.

**Suggestion:**
- **Testing**: Port Python test suite to C# — tokenization, generation, multi-modal, adapters, error handling
- **ChatClient**: Expose `Generator` via `GetService<Generator>()` for rewind support (GitHub issue #1553)
- **Async API**: Add `GenerateNextTokenAsync()` for ASP.NET Core integration
- **NuGet**: Improve package metadata, XML documentation, and samples
- **Blazor/MAUI**: Add examples for cross-platform .NET apps

### 2.13 Enhanced Documentation

**Status:** Only 3 doc files (`ConstrainedDecoding.md`, `DownloadModels.md`, `RuntimeOptions.md`); GitHub issue #1990 requests config format documentation

**Description:** Comprehensive developer documentation.

**Suggestion:**
- **Config Format Documentation**: Document all `genai_config.json` fields with types, defaults, and examples (issue #1990)
- **Architecture Guide**: Document the State → Model → Generator → Engine abstraction layers
- **Model Porting Guide**: Step-by-step guide for adding new model architectures
- **Performance Tuning Guide**: Hardware-specific recommendations, batch size selection, KV cache strategies
- **API Reference**: Auto-generate from C header comments; add docstrings to Python bindings
- **Migration Guide**: Document breaking changes between versions
- **Troubleshooting Guide**: Common errors and resolutions (especially DML and QNN issues that appear repeatedly in GitHub issues)

### 2.14 Expanded Test Coverage

**Current gaps identified:**

| Area | Current State | Recommendation |
|------|--------------|----------------|
| **C# Tests** | 1 test case | Port full API surface tests from Python/C++ |
| **Error Handling** | No negative tests | Add invalid input, OOM, malformed config tests |
| **Concurrency** | Basic worker thread tests | Add multi-request concurrent generation tests |
| **Memory** | No leak/peak tests | Add leak detection (LeakChecked<T> exists but isn't tested) |
| **ROCm** | No tests | Add ROCm pipeline when hardware available |
| **OpenVINO** | Config tests only | Add model execution tests |
| **WebGPU** | Minimal | Add browser-based integration tests |
| **Long-running** | None | Add stress tests for extended generation sessions |
| **Edge cases** | Limited | Test boundary batch sizes, extreme max_length, empty inputs |
| **Performance regression** | No automation | Add automated benchmark comparison in CI |

### 2.15 Model Validation & Diagnostics

**Status:** No model compatibility checking before inference

**Description:** Pre-flight validation that catches model configuration errors before expensive inference fails.

**Suggestion:**
- Add `Model::Validate()` that checks: ONNX graph inputs/outputs match config, required files exist, data types are compatible
- Report clear diagnostic messages: "Model expects input 'pixel_values' but config has no vision section"
- Add a CLI tool: `onnxruntime-genai-validate <model_path>` for offline model checking
- Check hardware compatibility: warn if model requires FP16 but device doesn't support it
- Integration point: `src/models/model.cpp` — add validation during `Model::Create()`

### 2.16 Dynamic Thread Scaling & NUMA Awareness

**Status:** Simple thread pool with fixed size; no NUMA awareness

**Description:** Adapt thread count to workload and respect NUMA topology for optimal memory access patterns.

**Suggestion:**
- Replace `ThreadPool` with a work-stealing implementation (e.g., based on Intel TBB or custom)
- Add NUMA-aware allocation: bind threads and memory to the same NUMA node
- Auto-scale thread count based on batch size and system load
- Integration point: `src/models/threadpool.h`

### 2.17 Prefix Caching

**Status:** Not implemented

**Description:** Cache KV states for common prompt prefixes (system prompts, few-shot examples) to avoid recomputation across requests.

**Suggestion:**
- Add a `PrefixCache` keyed by token sequence hash
- When a new request shares a prefix with a cached entry, copy the cached KV state instead of recomputing
- Integrate with the engine's `CacheManager` — extend `PagedCacheManager` to support shared read-only blocks
- Particularly valuable for chat applications where the system prompt is the same across all users
- Integration point: `src/engine/cache_manager.h`, `src/engine/paged_key_value_cache.h`

### 2.18 Function Calling / Tool Use Framework

**Status:** Constrained decoding supports JSON schema output; no structured tool-call framework

**Description:** Built-in support for function calling / tool use with automatic schema enforcement and response parsing.

**Suggestion:**
- Add `ToolDefinition` class: name, description, JSON schema for parameters
- Integrate with constrained decoding to force tool call output format
- Auto-parse tool call responses into structured objects
- Support multiple tool-call patterns: single call, parallel calls, sequential calls
- Provide language-specific ergonomic APIs (Python dataclasses, C# records, Java POJOs)
- Integration point: `src/constrained_logits_processor.h` — extend `GuidanceLogitsProcessor`

### 2.19 Quantization-Aware Generation

**Status:** Quantized models (INT4, INT8) are supported; no runtime quantization control

**Description:** Allow runtime selection of quantization levels and mixed-precision strategies.

**Suggestion:**
- Add `QuantizationConfig` to `genai_config.json`: per-layer precision, KV cache quantization
- Support FP8 KV cache for memory reduction (especially useful for long contexts)
- Add KV cache quantization: store cache in INT8/FP8, dequantize on-the-fly during attention
- Integration point: `src/models/kv_cache.h` — add quantized cache variants

### 2.20 Prompt Caching / Session Persistence

**Status:** `RewindToLength()` supports in-session rewind; no cross-session persistence

**Description:** Save and restore generator state to disk for resumable conversations.

**Suggestion:**
- Serialize `Generator` state (KV cache, token sequence, search state) to a binary format
- Add `Generator::SaveState(path)` and `Generator::LoadState(path)`
- Useful for: long conversations that span application restarts, checkpoint/resume for batch processing
- Consider memory-mapped files for large KV caches to avoid full serialization
- Integration point: `src/generators.h`

### 2.21 Embedding Extraction API

**Status:** `Embeddings` class exists internally but not exposed in public API

**Description:** Allow users to extract intermediate embeddings from the model for retrieval, classification, or visualization.

**Suggestion:**
- Add `Generator::GetEmbeddings(layer_index)` returning a `Tensor`
- Support extracting from any transformer layer, not just the final one
- Useful for RAG pipelines, semantic search, and fine-tuning workflows
- Integration point: `src/models/embeddings.h`, `src/ort_genai_c.h`

### 2.22 Token Probability / Logprobs API

**Status:** `GetLogits()` exposes raw logits; no convenient logprobs API

**Description:** Expose per-token log probabilities for the generated tokens and optionally the top-N alternatives.

**Suggestion:**
- Add `Generator::GetLogProbs()` returning log-probabilities for generated tokens
- Add `Generator::GetTopLogProbs(n)` returning the top-N token alternatives with their probabilities
- Essential for: evaluation pipelines, uncertainty estimation, watermarking, and classifier-free guidance
- Integration point: `src/generators.h`, `src/models/logits.h`

### 2.23 OpenAI-Compatible Server

**Status:** No server component; only library API

**Description:** A standalone HTTP server implementing the OpenAI `/v1/chat/completions` and `/v1/completions` APIs.

**Suggestion:**
- Implement as a separate tool/example using the engine's batching infrastructure
- Support SSE streaming for token-by-token responses
- Map OpenAI parameters (temperature, top_p, max_tokens, stop, etc.) to GenAI search options
- Include model listing endpoint (`/v1/models`)
- Can serve as a drop-in replacement for local LLM serving
- Consider building on top of the existing Python/C# examples

---

## 3. Known Issues & Gaps

### 3.1 DirectML (DML) Stability

Multiple open GitHub issues report DML failures:
- **GPU suspended errors** (887A0005, 887A0006) during batch processing (#628, #975)
- **Graph capture failures** when not all nodes partition to DML (#1522)
- **Parameter errors** in fused nodes (#1535)
- Particularly affects: multi-batch generation, older GPUs (Quadro P5000), and Olive-optimized models

### 3.2 QNN Execution Provider

- Models may silently fall back to CPU instead of running on NPU (#1535)
- Limited model compatibility — not all architectures work with QNN
- Requires specific model optimization for Qualcomm hardware

### 3.3 Long Prompt Handling

- Users report issues with prompts exceeding context length (#1565, #1560)
- Error messages can be unclear — "sequence length (0) exceeds max length"
- `max_length` parameter semantics (input + output) are confusing to users

### 3.4 Model Compatibility

- Some Olive-optimized models fail on certain execution providers
- GGUF model support requires the Python model builder, not native C++ loading
- Custom-exported ONNX models may have input/output name mismatches

### 3.5 Documentation Gaps

- `genai_config.json` format is undocumented (issue #1990)
- No architecture/design documentation for contributors
- Error messages often reference internal file paths without user-actionable guidance
- Hardware-specific setup guides are incomplete (especially QNN, ROCm)

### 3.6 Testing Gaps

- C# binding has only 1 test case
- No automated performance regression detection
- No error/negative path testing
- ROCm and OpenVINO lack model execution tests
- No concurrent multi-request stress tests

---

## 4. Implementation Suggestions

### Priority Tiers

#### Tier 1 — High Impact, Moderate Effort

| Feature | Why | Estimated Scope |
|---------|-----|----------------|
| **Config format documentation** (#1990) | Most frequently requested; unblocks all users | 1 markdown file |
| **Qwen3-VL support** (#1989) | Detailed spec ready; working prototype exists | ~5 files changed |
| **Token logprobs API** | Essential for evaluation and advanced use cases | ~3 files changed |
| **C# test expansion** | Currently 1 test; huge gap for .NET users | ~1 new test file |
| **Async generator API** | Modern apps expect async; engine already threaded | ~5 files per language |

#### Tier 2 — High Impact, Higher Effort

| Feature | Why | Estimated Scope |
|---------|-----|----------------|
| **Speculative decoding** | On roadmap; 2–3× speedup potential | New search strategy + engine changes |
| **Prefix caching** | Major latency reduction for chat workloads | Cache manager extension |
| **Min-P / Typical sampling** | Popular in community; small kernel additions | ~2 files per sampling method |
| **Model validation tool** | Catches errors before expensive inference | New validation module |
| **Memory profiling** | Users need to right-size hardware | DeviceBuffer instrumentation |

#### Tier 3 — Strategic, Significant Effort

| Feature | Why | Estimated Scope |
|---------|-----|----------------|
| **Stable Diffusion** | Under development per README; new paradigm | New pipeline + state classes |
| **Multi-GPU / Tensor parallelism** | Enables larger models | New distributed state + NCCL |
| **iOS support** | On roadmap; ObjC bindings exist | Build config + packaging |
| **ROCm acceleration** | On roadmap; AMD GPU market growing | Kernel porting + testing |
| **OpenAI-compatible server** | Enables drop-in local LLM serving | New tool/example |

#### Tier 4 — Nice to Have

| Feature | Why | Estimated Scope |
|---------|-----|----------------|
| **Function calling framework** | Builds on constrained decoding | New API layer |
| **Session persistence** | Resumable conversations | Serialization infrastructure |
| **Embedding extraction** | RAG and fine-tuning workflows | API surface extension |
| **NUMA awareness** | Server-grade performance | Thread pool replacement |
| **Batch auto-tuning** | Better out-of-box experience | Calibration logic |

### Cross-Cutting Recommendations

1. **Error messages**: Audit all `throw` sites and ensure messages include: what went wrong, why, and what the user can do about it. Many GitHub issues stem from opaque errors.

2. **Telemetry hooks**: Add optional performance counters (tokens/sec, time-to-first-token, queue depth) accessible via API. The existing `Tracing` infrastructure can be extended.

3. **Model registry**: Consider a machine-readable model compatibility matrix (JSON) that tools can query to determine which models work with which hardware/EP combinations.

4. **Plugin architecture**: As the number of model architectures grows (28+), consider a plugin registration system where new models can be added via shared libraries without recompiling the core.

5. **Versioned config schema**: Add a `schema_version` field to `genai_config.json` to enable forward/backward compatibility as the config format evolves.

---

*This document reflects the state of the codebase as of March 2026 and should be updated as features are implemented or requirements change.*
