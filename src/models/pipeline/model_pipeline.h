// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

namespace Generators {

struct Config;
struct Model;
struct OrtEnv;

std::shared_ptr<Model> CreatePipelineModel(std::unique_ptr<Config>& config, OrtEnv& ort_env);

}  // namespace Generators
