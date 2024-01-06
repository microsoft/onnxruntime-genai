#pragma once

namespace Generators {

struct Logits {

  Logits(Model& model, const SearchParams& search_params);
  void Update();
  RoamingArray<float> Get();

  Model& model_;

  std::array<int64_t, 3> logits_shape_;
  std::unique_ptr<OrtValue> logits_;
  std::unique_ptr<OrtValue> logits32_;  // When model output is fp16, this holds the fp32 conversion of them
};

}  // namespace Generators
