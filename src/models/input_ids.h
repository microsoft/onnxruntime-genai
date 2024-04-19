#pragma once

#include "static_buffer.h"

namespace Generators {

struct InputIDs {
  InputIDs(const Model& model, State& state);

  void Add();
  void Update(RoamingArray<int32_t> next_tokens);

  auto& GetShape() const { return shape_; }
  const char* name_;

 private:
  const Model& model_;
  State& state_;
  size_t input_index_{~0U};

  std::array<int64_t, 2> shape_{};
  ONNXTensorElementDataType type_;
  std::unique_ptr<OrtValue> value_;

  // Used for decoding runs with cuda graphs.
  std::unique_ptr<StaticBuffer> sb_input_ids_;
};

}  // namespace Generators
