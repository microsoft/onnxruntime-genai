// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <cstring>
#include "generator/generators.h"
#include "models/model.h"
#include "models/io/multi_modal_features.h"

namespace Generators {

MultiModalFeatures::MultiModalFeatures(State& state, MultiModalFeatures::Mode mode, const std::string& name,
                                       int64_t batch_size, int64_t num_feature_tokens)
    : state_{state},
      type_{mode == MultiModalFeatures::Mode::Input
                ? model_.session_info_.GetInputDataType(name)
                : model_.session_info_.GetOutputDataType(name)},
      mode_{mode},
      name_{name} {
  const auto feature_shape = mode_ == MultiModalFeatures::Mode::Input
                                 ? model_.session_info_.GetInputShape(name)
                                 : model_.session_info_.GetOutputShape(name);
  const auto dims = feature_shape.size();

  // If the model expects 3 dimensions, add a batch dimension
  // batch_size <= 0 signals "skip batch dim even if model has 3D output"
  if (dims == 3 && batch_size > 0) {
    shape_.push_back(batch_size);
  }

  shape_.push_back(num_feature_tokens);
  const auto feature_width =
      !feature_shape.empty() && feature_shape.back() > 0 ? feature_shape.back() : model_.config_->model.decoder.hidden_size;
  shape_.push_back(feature_width);

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
    // Written by the encoder session, so it lives on that session's device, not the decoder's
    // (see SessionCanAccess).
    features_ = OrtValue::CreateTensor(state_.p_session_device_->GetAllocator(), shape_, type_);
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
  if (!is_prompt && shape_[shape_.size() - 2] > 0) {  // if num_image_tokens > 0
    shape_[shape_.size() - 2] = 0;
    features_ = OrtValue::CreateTensor(state_.p_session_device_inputs_->GetAllocator(), shape_, type_);
    state_.inputs_[index_] = features_.get();
  }
}

void MultiModalFeatures::ReuseFeaturesBuffer(MultiModalFeatures& other) {
  if (mode_ == MultiModalFeatures::Mode::Output || other.mode_ == MultiModalFeatures::Mode::Input) {
    throw std::runtime_error("Incorrect usage of the MultiModalFeatures inputs and outputs.");
  }

  auto& producer_device = *other.state_.p_session_device_;
  auto& consumer_device = *state_.p_session_device_inputs_;

  if (SessionCanAccess(*state_.p_session_device_, producer_device)) {
    // Share the output MultiModalFeatures OrtValue* from other with the input MultiModalFeatures for this.
    features_ = std::move(other.features_);
    state_.inputs_[index_] = other.state_.outputs_[other.index_];
    return;
  }

  // The encoder ran on a device this session has no EP for, so its buffer cannot be bound here.
  // Stage the features through a copy onto a device this session can read.
  auto info = other.features_->GetTensorTypeAndShapeInfo();
  auto producer_shape = info->GetShape();
  features_ = OrtValue::CreateTensor(consumer_device.GetAllocator(), producer_shape, info->GetElementType());
  if (info->GetElementCount() != 0) {
    ByteWrapTensor(consumer_device, *features_).CopyFrom(ByteWrapTensor(producer_device, *other.features_));
  }
  state_.inputs_[index_] = features_.get();
}

void MultiModalFeatures::AllocateEmptyFeatures() {
  // Skip if already allocated (avoids redundant allocation when called from
  // both EmbeddingState::SetExtraInputs and the pipeline prompt path)
  if (features_ && state_.inputs_[index_] == features_.get()) return;
  features_ = OrtValue::CreateTensor(state_.p_session_device_inputs_->GetAllocator(), shape_, type_);
  state_.inputs_[index_] = features_.get();
}

void MultiModalFeatures::ReshapeFeatures(std::vector<int64_t> new_shape) {
  if (!features_) return;
  auto old_info = features_->GetTensorTypeAndShapeInfo();
  int64_t old_count = static_cast<int64_t>(old_info->GetElementCount());
  int64_t new_count = 1;
  for (auto d : new_shape) new_count *= d;
  if (old_count != new_count || old_count == 0) return;

  auto old_features = std::move(features_);
  // Only encoder outputs are reshaped; they live on the session's own device.
  auto& device = *state_.p_session_device_;
  features_ = OrtValue::CreateTensor(device.GetAllocator(), new_shape, type_);
  auto src = ByteWrapTensor(device, *old_features);
  auto dst = ByteWrapTensor(device, *features_);
  dst.CopyFrom(src);
  shape_ = std::move(new_shape);
  if (mode_ == Mode::Output && index_ != ~0U) {
    state_.outputs_[index_] = features_.get();
  }
}

size_t MergeImageFeaturesIntoEmbeddings(OrtValue& inputs_embeds,
                                        const OrtValue& image_features,
                                        std::span<const int64_t> target_token_rows) {
  const auto embeddings_info = inputs_embeds.GetTensorTypeAndShapeInfo();
  const auto vision_info = image_features.GetTensorTypeAndShapeInfo();
  if (embeddings_info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      vision_info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: both tensors must contain float elements");
  }

  const auto embeddings_shape = embeddings_info->GetShape();
  const auto vision_shape = vision_info->GetShape();
  if (embeddings_shape.empty() || vision_shape.empty()) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: tensors must have a hidden dimension");
  }

  // Hidden size is always the last dimension for both [batch, seq, hidden]/[seq, hidden] embeddings
  // and [num_image_tokens, hidden] image features.
  const int64_t embedding_dim = embeddings_shape.back();
  const int64_t vision_dim = vision_shape.back();
  if (embedding_dim <= 0 || vision_dim <= 0) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: hidden dimensions must be positive");
  }
  if (vision_dim != embedding_dim) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: hidden dimension mismatch - vision_dim=" +
                             std::to_string(vision_dim) + ", embedding_dim=" + std::to_string(embedding_dim));
  }

  const size_t embedding_element_count = embeddings_info->GetElementCount();
  const size_t vision_element_count = vision_info->GetElementCount();
  if (embedding_element_count % static_cast<size_t>(embedding_dim) != 0 ||
      vision_element_count % static_cast<size_t>(vision_dim) != 0) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: tensor element count is not divisible by hidden dimension");
  }
  const size_t embedding_rows = embedding_element_count / static_cast<size_t>(embedding_dim);
  const size_t num_vision_tokens = vision_element_count / static_cast<size_t>(vision_dim);
  const size_t rows_to_write = std::min(target_token_rows.size(), num_vision_tokens);

  for (size_t i = 0; i < rows_to_write; ++i) {
    const int64_t row = target_token_rows[i];
    if (row < 0 || static_cast<uint64_t>(row) >= embedding_rows) {
      throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: destination row " + std::to_string(row) +
                               " is outside embeddings row count " + std::to_string(embedding_rows));
    }
  }

  float* embeddings_data = inputs_embeds.GetTensorMutableData<float>();
  const float* vision_data = image_features.GetTensorData<float>();
  for (size_t consumed = 0; consumed < rows_to_write; ++consumed) {
    const int64_t row = target_token_rows[consumed];
    std::memcpy(embeddings_data + (row * embedding_dim),
                vision_data + (consumed * vision_dim),
                vision_dim * sizeof(float));
  }
  return rows_to_write;
}

}  // namespace Generators
