#pragma once

#include "static_buffer.h"

namespace Generators {

struct Logits {
  Logits(const Model& model, State& state);

  void Add();
  RoamingArray<float> Get();

 private:
  const Model& model_;
  State& state_;
  size_t output_index_{~0U};

  std::array<int64_t, 3> shape_{};
  ONNXTensorElementDataType type_;
  std::unique_ptr<OrtValue> value32_;  // Always fp32 values
  std::unique_ptr<OrtValue> value16_;  // When model output is fp16

  // Used for decoding runs with cuda graphs.
  std::unique_ptr<StaticBuffer> sb_logits32_;
  std::unique_ptr<StaticBuffer> sb_logits16_;
};

}  // namespace Generators
