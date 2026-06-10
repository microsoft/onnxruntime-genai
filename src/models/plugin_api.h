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

#ifdef __cplusplus
}  // extern "C"
#endif
