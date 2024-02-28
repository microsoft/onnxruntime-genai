#pragma once

namespace Generators {

struct Logits {
  Logits(const Model& model, State& state);

  void Add();
  void Update();
  RoamingArray<float> Get();

 private:
  const Model& model_;
  State& state_;
  size_t output_index_{~0U};

  std::array<int64_t, 3> shape_{};
  ONNXTensorElementDataType type_;
  std::unique_ptr<OrtValue> value_;
  std::unique_ptr<OrtValue> value32_;  // When model output is fp16, this holds the fp32 conversion of them
};

}  // namespace Generators
