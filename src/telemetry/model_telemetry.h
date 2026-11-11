// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>

namespace Generators {

struct Model;

std::shared_ptr<Model> CreateModelWithTelemetry(
    const std::function<std::shared_ptr<Model>()>& create_model);

}  // namespace Generators
