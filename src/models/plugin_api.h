// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.
#pragma once

// Pipeline-as-Config plugin ABI (issue #2114, PR4) — the "escape hatch" that lets a model author
// ship a fully custom pipeline `Model` in a separate shared library, loaded at runtime when a config
// declares `pipeline.plugin {library, entry_point}`.
//
// STABLE C ABI ONLY. The original issue sketched an entry point that passed std::unique_ptr<Config>
// and returned std::shared_ptr<Model> across `extern "C"`. That is NOT a stable ABI: C++ smart
// pointers, RTTI, exceptions and the allocator are all compiler/STL-version specific and must never
// cross the library boundary. This header therefore exposes only opaque handles, plain pointers and
// an int status code. The C++ <-> opaque-handle adaptation lives entirely INSIDE the runtime (see
// plugin_loader.cpp); plugins never see a Generators C++ type through this boundary.
//
// NOTE: a plugin is an in-process escape hatch built against this exact runtime. The opaque
// `OgaPipelinePluginModel*` it returns is a pointer to a runtime object the plugin constructed; the
// runtime adapts it back on its own side. The C boundary's job is to avoid passing throwing C++
// types/smart pointers across the .so edge, not to allow ABI-mismatched binaries to interoperate.

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the pipeline model produced by the plugin. Its concrete type is private to the
// plugin; the runtime only ever passes it back to the plugin (e.g. to the destroy callback) and,
// internally, reinterprets it as the runtime Model the plugin constructed.
typedef struct OgaPipelinePluginModel OgaPipelinePluginModel;

// Status codes returned by the plugin entry point. Zero means success; any non-zero value is treated
// by the runtime as failure and surfaced as a std::runtime_error. Plugins may define their own
// non-zero codes; only the zero/non-zero distinction is part of the ABI.
typedef enum {
  OgaPipelinePluginStatus_Ok = 0,
  OgaPipelinePluginStatus_Error = 1,
} OgaPipelinePluginStatus;

// Destructor callback for the handle returned by the entry point. The plugin owns the model's memory
// and is the only code that may free it; the runtime calls this exactly once when the wrapping
// std::shared_ptr<Model> is destroyed. Must be tolerant of being called on the value written to
// `model_out` and must not throw across the boundary.
typedef void (*OgaPipelinePluginModelDestroyFn)(OgaPipelinePluginModel* model);

// Entry point a plugin MUST export under the symbol name given by `pipeline.plugin.entry_point`.
//
// Parameters:
//   ort_env     [in]  Opaque pointer to the runtime's OrtEnv (a `OrtEnv*`). Borrowed; the plugin
//                     must NOT take ownership or free it. Valid for the duration of the call and the
//                     lifetime of the returned model.
//   config      [in]  Opaque pointer to the runtime Config (a `Generators::Config*`) describing the
//                     model. OWNERSHIP TRANSFERS to the plugin on success (status == Ok): the plugin
//                     must store/free it (typically by embedding it in the model it returns). On
//                     failure (non-zero status) ownership remains with the runtime, which frees it.
//   model_out   [out] On success the plugin writes its opaque model handle here (non-NULL).
//   destroy_out [out] On success the plugin writes the destructor callback for that handle here
//                     (non-NULL). The runtime invokes it to release the model.
//
// Returns OgaPipelinePluginStatus_Ok (0) on success, non-zero on failure. The plugin must not throw
// C++ exceptions across this boundary; report errors via the status code instead.
typedef int (*OgaCreatePipelineFn)(void* ort_env,
                                   void* config,
                                   OgaPipelinePluginModel** model_out,
                                   OgaPipelinePluginModelDestroyFn* destroy_out);

// =====================================================================================================
// Controller-plugin escape hatch (issue #2114, PR-E / design §8) — bucket C.
//
// The entry point above hands over *model construction only*; the generation loop stays in the
// runtime's Generator. The controller surface below lets a plugin instead OWN the per-step decode
// loop: irregular, stateful control flow (Lookahead's Jacobi n-gram pool, nested cascades, novel
// research) that cannot be expressed as a static DAG or the parameterized `speculative` strategy.
//
// Same ABI contract as above: STABLE C ONLY — opaque handles, plain pointers, int status. No C++
// types, smart pointers, RTTI or exceptions cross the boundary. The runtime adapts its C++ Generator
// to the C `OgaDecodeStepContext` primitive surface on its own side; the controller never sees a
// Generators C++ type. Status convention: 0 == success, non-zero == failure (surfaced as a
// std::runtime_error by the runtime).
// =====================================================================================================

// Opaque handle to the controller object the plugin constructs. Private to the plugin; the runtime
// only passes it back to the plugin's step/destroy callbacks.
typedef struct OgaDecodeController OgaDecodeController;

// Opaque handle to the runtime-side decode state (a `Generators::Generator*` adapter) the controller
// drives. The controller never dereferences it; it only passes it back into the primitive callbacks
// in OgaDecodeStepContext below.
typedef struct OgaDecodeContext OgaDecodeContext;

// The runtime-provided step-primitive vtable. The runtime fills this in (with `host` bound to the
// current decode state) and passes it to the controller's step function. Every callback takes the
// opaque `host` as its first argument and returns an int status (0 == ok) unless noted. Pointers
// returned via out-params (`GetLogits`/`GetHiddenStates`) point into runtime-owned scratch that
// stays valid only until the next callback on the same context.
typedef struct OgaDecodeStepContext {
  OgaDecodeContext* host;  // Pass this back to every callback below. Borrowed; do not free.

  // Current committed sequence length (token count) for batch index 0.
  size_t (*GetSequenceLength)(OgaDecodeContext* host);

  // The model's primary end-of-stream token id, or -1 if none is configured. Query primitive.
  int32_t (*GetEosTokenId)(OgaDecodeContext* host);

  // Non-zero if the runtime considers generation finished (EOS reached / max length). Query primitive.
  int (*IsDone)(OgaDecodeContext* host);

  // Copy up to `capacity` tokens of the committed sequence (batch 0) into `out`; writes the true
  // token count to `*count_out`. Returns 0 on success, non-zero if `out`/`capacity` is insufficient.
  int (*GetTokens)(OgaDecodeContext* host, int32_t* out, size_t capacity, size_t* count_out);

  // Run a forward step on the current next-token(s) if needed and expose the next-position logits.
  // Writes a pointer to the fp32 logits ([vocab] for batch 0) to `*logits_out` and the vocab size to
  // `*vocab_out`. This is the controller's "read logits" + lazy "run a forward step" primitive.
  int (*GetLogits)(OgaDecodeContext* host, const float** logits_out, size_t* vocab_out);

  // Expose the most recent intermediate hidden-state activation (fp32) for batch 0, last position.
  // Writes a pointer to `*hidden_out` and the hidden size to `*hidden_size_out`. Returns non-zero
  // when the model does not expose hidden states.
  int (*GetHiddenStates)(OgaDecodeContext* host, const float** hidden_out, size_t* hidden_size_out);

  // Commit `count` caller-chosen tokens as the accepted next tokens WITHOUT forcing a recompute
  // (the next GetLogits re-evaluates against the extended sequence). This is the controller's
  // "append tokens" / advance primitive. Returns 0 on success.
  int (*AppendTokens)(OgaDecodeContext* host, const int32_t* tokens, size_t count);

  // Roll the runtime state (search sequences + owned KV caches + position state) back to `length`
  // tokens. The rewind/rollback primitive used by speculative / lookahead controllers. Returns 0
  // on success.
  int (*RewindTo)(OgaDecodeContext* host, size_t length);
} OgaDecodeStepContext;

// The controller's per-outer-step callback. The runtime calls this once per Generator::GenerateNextToken
// when a controller is configured, instead of running the built-in sampling step. The controller uses
// the `ctx` primitives to drive its own loop and MUST commit at least progress toward termination
// (typically >= 1 token via ctx->AppendTokens). Writes the number of tokens it emitted this step to
// `*tokens_emitted_out`. Returns 0 on success, non-zero on failure.
typedef int (*OgaControllerStepFn)(OgaDecodeController* self,
                                   OgaDecodeStepContext* ctx,
                                   int* tokens_emitted_out);

// Destructor for the controller handle. Called exactly once by the runtime when the owning Generator
// is destroyed. Must tolerate the handle written to `controller_out` and must not throw.
typedef void (*OgaDecodeControllerDestroyFn)(OgaDecodeController* self);

// Optional second entry point a controller plugin MAY export (alongside OgaCreatePipelineFn), named
// by `pipeline.controller.entry_point`.
//
// Parameters:
//   config       [in]  Opaque, controller-defined config C string from `pipeline.controller.config`
//                      (may be NULL/empty). Borrowed for the duration of the call; the controller
//                      must copy anything it needs to retain.
//   controller_out [out] On success the plugin writes its opaque controller handle here (non-NULL).
//   step_out     [out] On success the plugin writes its per-step callback here (non-NULL).
//   destroy_out  [out] On success the plugin writes the destructor for the handle here (non-NULL).
//
// Returns 0 on success, non-zero on failure. Must not throw across the boundary.
typedef int (*OgaCreateDecodeControllerFn)(void* config,
                                           OgaDecodeController** controller_out,
                                           OgaControllerStepFn* step_out,
                                           OgaDecodeControllerDestroyFn* destroy_out);

#ifdef __cplusplus
}  // extern "C"
#endif
