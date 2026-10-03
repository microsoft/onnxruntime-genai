// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

struct OrtEnv;

namespace Generators {

struct Config;
struct Model;

std::shared_ptr<Model> CreatePipelineModel(std::unique_ptr<Config>& config, OrtEnv& ort_env);

}  // namespace Generators
