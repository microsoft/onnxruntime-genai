// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

struct AudioFeatures {
  AudioFeatures(State& state, const std::string& name, const std::vector<ExtraInput>& extra_inputs);
  AudioFeatures(const AudioFeatures&) = delete;
  AudioFeatures& operator=(const AudioFeatures&) = delete;

  void Add();
  auto& GetShape() const { return shape_; }
  auto& GetType() const { return type_; }
  OrtValue* Get() { return audio_features_.get(); }

 private:
  State& state_;
  const Model& model_;

  std::vector<int64_t> shape_;  // [batch_size, num_mels, num_frames]
  ONNXTensorElementDataType type_;

  const std::string name_;

  std::unique_ptr<OrtValue> audio_features_;
  size_t index_{~0U};
};

}  // namespace Generators
