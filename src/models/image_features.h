// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

struct ImageFeatures {
  enum struct Mode {
    Input = 0,
    Output
  };

  ImageFeatures(const Model& model, State& state, ImageFeatures::Mode mode, const std::string& name, int64_t num_image_tokens);
  ImageFeatures(const ImageFeatures&) = delete;
  ImageFeatures& operator=(const ImageFeatures&) = delete;

  void Add();
  void Update();
  void ReuseImageFeaturesBuffer(ImageFeatures& other);

  auto& GetShape() const { return shape_; }
  OrtValue* Get() { return image_features_.get(); }

 private:
  const Model& model_;
  State& state_;

  std::array<int64_t, 2> shape_{};  // [num_image_tokens, hidden_size]
  ONNXTensorElementDataType type_;

  const Mode mode_{};
  const std::string name_;

  std::unique_ptr<OrtValue> image_features_;
  size_t index_{~0U};
};

}  // namespace Generators
