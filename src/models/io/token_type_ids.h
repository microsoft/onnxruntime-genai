// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "models/model.h"

namespace Generators {

// Feeds the optional decoder token_type_ids input ([batch, sequence_length]; 0 = text, 1 = image)
// used by OpenVINO-partitioned Gemma3 decoders. Prompt markers are consumed by the next Update.
struct TokenTypeIds {
  TokenTypeIds(State& state);
  TokenTypeIds(const TokenTypeIds&) = delete;
  TokenTypeIds& operator=(const TokenTypeIds&) = delete;

  void Add();
  void SetPending(std::span<const int32_t> markers);
  void Update(size_t sequence_length);

 private:
  State& state_;
  const Model& model_{state_.model_};
  size_t input_index_{~0U};
  ONNXTensorElementDataType type_;
  std::array<int64_t, 2> shape_{};
  std::unique_ptr<OrtValue> value_;
  std::vector<int32_t> pending_;
};

std::vector<int32_t> ReadTokenTypeIds(const OrtValue& token_type_ids);
void UpdatePendingTokenTypeIds(OrtValue& token_type_ids, std::vector<int32_t>& pending_token_type_ids);

}  // namespace Generators
