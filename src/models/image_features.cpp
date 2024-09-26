// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "model.h"
#include "image_features.h"

namespace Generators {

ImageFeatures::ImageFeatures(const Model& model, State& state, ImageFeatures::Mode mode, const std::string& name, int64_t num_image_tokens)
    : model_{model},
      state_{state},
      shape_{num_image_tokens, model.config_->model.decoder.hidden_size},
      type_{mode == ImageFeatures::Mode::Input
                ? model_.session_info_->GetInputDataType(name)
                : model_.session_info_->GetOutputDataType(name)},
      mode_{mode},
      name_{name} {
  // There are four cases for ImageFeatures:
  // 1) Created as an output for vision model (num_image_tokens > 0)
  //    The tensor needs to be pre-allocated to store the output.
  //    It will be transferred to an input for the embedding model.
  // 2) Created as an output for vision model (num_image_tokens = 0)
  //    The tensor will be pre-allocated to store the empty output.
  //    It will be transferred to an input for the embedding model.
  // 3) Created as an input for embedding model (num_image_tokens > 0)
  //    The tensor does not need to be pre-allocated because it will be created during (1).
  // 4) Created as an input for embedding model (num_image_tokens = 0)
  //    The tensor does not need to be pre-allocated because it will be created during (2).
  if (mode == ImageFeatures::Mode::Output) {
    image_features_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
  }
}

void ImageFeatures::Add() {
  if (mode_ == ImageFeatures::Mode::Input) {
    // In case the image_features are an input to a model, they are added
    // as a nullptr to reserve a slot in the inputs. The image_features
    // input will be overwritten when ReuseImageFeaturesBuffer is invoked.
    index_ = state_.inputs_.size();
    state_.inputs_.push_back(nullptr);
    state_.input_names_.push_back(name_.c_str());
  } else {
    index_ = state_.outputs_.size();
    state_.outputs_.push_back(image_features_.get());
    state_.output_names_.push_back(name_.c_str());
  }
}

void ImageFeatures::Update() {
  // Initialize empty image_features tensor for after-prompt input scenarios
  // num_image_tokens will be 0 when no image is provided
  if (shape_[0] > 0) {  // if num_image_tokens > 0
    shape_[0] = 0;
    image_features_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
    state_.inputs_[index_] = image_features_.get();
  }
}

void ImageFeatures::ReuseImageFeaturesBuffer(ImageFeatures& other) {
  if (mode_ == ImageFeatures::Mode::Output || other.mode_ == ImageFeatures::Mode::Input) {
    throw std::runtime_error("Incorrect usage of the ImageFeatures inputs and outputs.");
  }

  // Share the output ImageFeatures OrtValue* from other with the input ImageFeatures for this.
  image_features_ = std::move(other.image_features_);
  state_.inputs_[index_] = other.state_.outputs_[other.index_];
}

}  // namespace Generators
