// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "kv_cache.h"

namespace Generators {

// Very similar to the DefaultKeyValueCache, but is only created once at the encoder step, then used without modification for every decoder step
struct CrossCache {
  CrossCache(State& state, int sequence_length);

  void AddOutputs(State& state);
  void AddInputs(State& state);
  auto& GetShape() const { return shape_; }
  auto& GetType() const { return type_; }
  auto& GetValues() { return values_; }

 private:
  int layer_count_;

  std::array<int64_t, 4> shape_;
  ONNXTensorElementDataType type_;

  std::vector<std::unique_ptr<OrtValue>> values_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};

}  // namespace Generators
