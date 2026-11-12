// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

struct OrtEnv;

namespace Generators {
struct Config;
struct Model;

namespace WebGPU {

using ModelFactory = std::shared_ptr<Model> (*)(OrtEnv&, std::unique_ptr<Config>);

// Applies WebGPU capture defaults and retries recognized initialization refusals
// with fresh eager sessions. The callback preserves the common model dispatch.
std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, std::unique_ptr<Config> config, ModelFactory create_model);

}  // namespace WebGPU
}  // namespace Generators
