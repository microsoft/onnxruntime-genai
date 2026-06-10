// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.
#pragma once

#include <memory>

#include "../generators.h"  // Config, Model, OrtEnv

namespace Generators {

// Loads a pipeline Model from an external shared library declared via `pipeline.plugin`
// (issue #2114, PR4). The dynamic-loading body is compiled in only when USE_GENAI_PLUGINS is set;
// otherwise this throws a clear "plugin support is not enabled" error so the default build never
// silently routes a plugin config to a built-in model.
//
// On success ownership of `config` transfers to the plugin (see plugin_api.h); on any failure the
// runtime retains and frees `config`.
std::shared_ptr<Model> LoadPluginPipeline(const Config::Pipeline::Plugin& plugin,
                                          std::unique_ptr<Config> config, OrtEnv& ort_env);

}  // namespace Generators
