// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.
#pragma once

#include <memory>

#include "../generators.h"  // Config, Model, OrtEnv

namespace Generators {

class ControllerHook;

// Loads a pipeline Model from an external shared library declared via `pipeline.plugin`
// (issue #2114, PR4). The dynamic-loading body is compiled in only when USE_GENAI_PLUGINS is set;
// otherwise this throws a clear "plugin support is not enabled" error so the default build never
// silently routes a plugin config to a built-in model.
//
// On success ownership of `config` transfers to the plugin (see plugin_api.h); on any failure the
// runtime retains and frees `config`.
std::shared_ptr<Model> LoadPluginPipeline(const Config::Pipeline::Plugin& plugin,
                                          std::unique_ptr<Config> config, OrtEnv& ort_env);

// Loads a decode controller from an external shared library declared via `pipeline.controller`
// (issue #2114 §8, PR-E) — the controller-plugin escape hatch that hands the per-step decode loop to
// a custom plugin (bucket C). Like the pipeline plugin path, the dynamic-loading body is compiled in
// only when USE_GENAI_PLUGINS is set; otherwise this throws a clear "controller plugin support is not
// enabled" error so a default build never silently ignores a declared controller.
std::unique_ptr<ControllerHook> LoadDecodeController(const Config::Pipeline::Controller& controller);

}  // namespace Generators
