// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "model.h"
#include "multi_modal_features.h"

namespace Generators {

MultiModalFeatures::MultiModalFeatures(State& state, MultiModalFeatures::Mode mode, const std::string& name, int64_t num_feature_tokens)
    : state_{state},
      type_{mode == MultiModalFeatures::Mode::Input
                ? model_.session_info_->GetInputDataType(name)
                : model_.session_info_->GetOutputDataType(name)},
      mode_{mode},
      name_{name} {
  shape_.push_back(num_feature_tokens);
  shape_.push_back(model_.config_->model.decoder.hidden_size);

  // There are four cases for MultiModalFeatures:
  // 1) Created as an output for vision or speech model (num_feature_tokens > 0)
  //    The tensor will be pre-allocated to store the output.
  //    It will be transferred to an input for the embedding model.
  // 2) Created as an output for vision or speech model (num_feature_tokens = 0)
  //    The tensor will be pre-allocated to store the empty output.
  //    It will be transferred to an input for the embedding model.
  // 3) Created as an input for embedding model (num_feature_tokens > 0)
  //    The tensor does not need to be pre-allocated because it will be created during (1).
  // 4) Created as an input for embedding model (num_feature_tokens = 0)
  //    The tensor does not need to be pre-allocated because it will be created during (2).
  if (mode == MultiModalFeatures::Mode::Output) {
    features_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), shape_, type_);
  }
}

void MultiModalFeatures::Add() {
  if (mode_ == MultiModalFeatures::Mode::Input) {
    // In case the features are an input to a model, they are added
    // as a nullptr to reserve a slot in the inputs. The features
    // input will be overwritten when ReuseFeaturesBuffer is invoked.
    index_ = state_.inputs_.size();
    state_.inputs_.push_back(nullptr);
    state_.input_names_.push_back(name_.c_str());
  } else {
    index_ = state_.outputs_.size();
    state_.outputs_.push_back(features_.get());
    state_.output_names_.push_back(name_.c_str());
  }
}

void MultiModalFeatures::Update(bool is_prompt) {
  // Initialize empty features tensor for after-prompt input scenarios
  // num_feature_tokens will be 0 when no image is provided
  if (!is_prompt && shape_[0] > 0) {  // if num_image_tokens > 0
    shape_[0] = 0;
    features_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), shape_, type_);
    state_.inputs_[index_] = features_.get();
  }
}

void MultiModalFeatures::ReuseFeaturesBuffer(MultiModalFeatures& other) {
  if (mode_ == MultiModalFeatures::Mode::Output || other.mode_ == MultiModalFeatures::Mode::Input) {
    throw std::runtime_error("Incorrect usage of the MultiModalFeatures inputs and outputs.");
  }

  // Share the output MultiModalFeatures OrtValue* from other with the input MultiModalFeatures for this.
  features_ = std::move(other.features_);
  state_.inputs_[index_] = other.state_.outputs_[other.index_];
}

}  // namespace Generators
