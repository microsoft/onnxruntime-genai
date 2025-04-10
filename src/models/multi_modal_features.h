// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

struct MultiModalFeatures {
  enum struct Mode {
    Input = 0,
    Output
  };

  MultiModalFeatures(State& state, MultiModalFeatures::Mode mode, const std::string& name, int64_t num_feature_tokens);
  MultiModalFeatures(const MultiModalFeatures&) = delete;
  MultiModalFeatures& operator=(const MultiModalFeatures&) = delete;

  void Add();
  void Update(bool is_prompt);
  void ReuseFeaturesBuffer(MultiModalFeatures& other);

  auto& GetShape() const { return shape_; }
  OrtValue* Get() { return features_.get(); }

 private:
  State& state_;
  const Model& model_{state_.model_};

  std::vector<int64_t> shape_;  // [num_feature_tokens, hidden_size]
  ONNXTensorElementDataType type_;

  const Mode mode_{};
  const std::string name_;

  std::unique_ptr<OrtValue> features_;
  size_t index_{~0U};
};

}  // namespace Generators
