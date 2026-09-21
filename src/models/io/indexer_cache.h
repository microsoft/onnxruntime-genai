// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/model.h"

namespace Generators {

struct IndexerCache {
  explicit IndexerCache(State& state);

  void Add();
  void Update(DeviceSpan<int32_t> beam_indices, int total_length, int current_length);
  void RewindTo(size_t index);
  bool IsEmpty() const { return layer_indices_.empty(); }

 private:
  State& state_;
  const Model& model_{state_.model_};
  std::vector<int> layer_indices_;
  std::vector<std::unique_ptr<OrtValue>> pasts_;
  std::vector<std::unique_ptr<OrtValue>> presents_;
  std::vector<std::unique_ptr<OrtValue>> empty_pasts_;
  std::unique_ptr<OrtValue> past_sequence_length_;
  std::vector<std::string> input_name_strings_;
  std::vector<std::string> output_name_strings_;
  std::vector<int64_t> shape_;
  ONNXTensorElementDataType type_{};
  bool share_buffer_{false};
  bool first_update_{true};
  size_t input_index_{~0U};
  size_t output_index_{~0U};
};

std::unique_ptr<IndexerCache> CreateIndexerCache(State& state);

}  // namespace Generators