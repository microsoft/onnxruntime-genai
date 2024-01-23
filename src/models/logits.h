#pragma once

namespace Generators {

struct Logits {
  Logits(Model& model, State& state);

  void Add();
  void Update();
  RoamingArray<float> Get();

 private:
  Model& model_;
  State& state_;
  size_t output_index_{~0U};

  std::array<int64_t, 3> logits_shape_{};
  std::unique_ptr<OrtValue> logits_;
  std::unique_ptr<OrtValue> logits32_;  // When model output is fp16, this holds the fp32 conversion of them
};

}  // namespace Generators
