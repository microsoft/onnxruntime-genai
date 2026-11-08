// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../model.h"

namespace Generators {

// Capacity changes are rare; a provider-matched Pad graph migrates BNSH tensors
// on device without treating opaque WebGPU buffer handles as offset pointers.
struct SharedCacheGrowth {
  static bool CanAutoShare(const Model& model, const GeneratorParams& params);

  static bool IsSupported(const State& state, std::span<const std::string> input_names,
                          std::span<const std::string> output_names);

  explicit SharedCacheGrowth(State& state) : state_{state} {}

  int64_t EnsureCapacity(int total_length, bool has_content,
                         std::vector<std::unique_ptr<OrtValue>>& tensors,
                         size_t input_index, size_t output_index);

 private:
  void Migrate(OrtValue& source, OrtValue& destination, int64_t extra_capacity);

  State& state_;
  std::unique_ptr<OrtSession> migration_session_;
};

}  // namespace Generators
