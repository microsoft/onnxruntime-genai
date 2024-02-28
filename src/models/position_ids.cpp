#include "../generators.h"
#include "model.h"
#include "position_ids.h"
#include "kernels.h"

namespace Generators {

PositionIDs::PositionIDs(const Model& model, State& state, RoamingArray<int32_t>& sequence_lengths_unk)
    : model_{model},
      state_{state} {
  type_ = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.position_ids);
  if (type_ != Ort::TypeToTensorType<int32_t>::type && type_ != Ort::TypeToTensorType<int64_t>::type)
    throw std::runtime_error("position_ids & attention_mask only support int32 or int64 types");

  std::array<int64_t, 2> shape{state_.search_params_.batch_size, state_.search_params_.sequence_length};  // Only batch_size initially, as we haven't expanded over the beams yet
  position_ids_ = OrtValue::CreateTensor(model.allocator_cpu_, shape, type_);
  attention_mask_ = OrtValue::CreateTensor(model.allocator_cpu_, shape, type_);

  initial_sequence_lengths_.resize(state_.search_params_.batch_size * state_.search_params_.num_beams);

  if (type_ == Ort::TypeToTensorType<int32_t>::type)
    InitializeTensors<int32_t>(shape, sequence_lengths_unk);
  else
    InitializeTensors<int64_t>(shape, sequence_lengths_unk);

  position_ids_ = model_.ExpandInputs(position_ids_, state_.search_params_.num_beams);
  attention_mask_ = model_.ExpandInputs(attention_mask_, state_.search_params_.num_beams);
  shape[0] *= state_.search_params_.num_beams;
  position_ids_shape_ = shape;
  attention_mask_shape_ = shape;
}

void PositionIDs::Add() {
  input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(position_ids_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.position_ids.c_str());

  state_.inputs_.push_back(attention_mask_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
}

void PositionIDs::Update(int current_length) {
  // Reallocate position_ids for the 2nd and onward shape
  if (initial_sequence_lengths_.size()) {
    position_ids_shape_[1] = 1;
    position_ids_ = OrtValue::CreateTensor(*model_.allocator_device_, position_ids_shape_, type_);
    state_.inputs_[input_index_] = position_ids_.get();

    // Copy the initial values over to the device specific tensor
    switch (model_.device_type_) {
      case DeviceType::CPU:
        if (type_ == Ort::TypeToTensorType<int32_t>::type)
          std::copy(initial_sequence_lengths_.begin(), initial_sequence_lengths_.end(), position_ids_->GetTensorMutableData<int32_t>());
        else
          std::copy(initial_sequence_lengths_.begin(), initial_sequence_lengths_.end(), position_ids_->GetTensorMutableData<int64_t>());
        break;
#if USE_CUDA
      case DeviceType::CUDA:
        if (type_ == Ort::TypeToTensorType<int32_t>::type)
          cudaMemcpyAsync(position_ids_->GetTensorMutableRawData(), initial_sequence_lengths_.data(), sizeof(int32_t) * initial_sequence_lengths_.size(), cudaMemcpyHostToDevice, model_.cuda_stream_);
        else
          cudaMemcpyAsync(position_ids_->GetTensorMutableRawData(), initial_sequence_lengths_.data(), sizeof(int64_t) * initial_sequence_lengths_.size(), cudaMemcpyHostToDevice, model_.cuda_stream_);
        break;
#endif
      default:
        throw std::runtime_error("PositionIDs::Update - Unsupported device type");
    }
    initial_sequence_lengths_.clear();
  } else {  // Just incrementing existing position IDs
    switch (model_.device_type_) {
      case DeviceType::CPU: {
        if (type_ == Ort::TypeToTensorType<int32_t>::type)
          UpdatePositionIDs<int32_t>();
        else
          UpdatePositionIDs<int64_t>();
        break;
      }
#if USE_CUDA
      case DeviceType::CUDA:
        if (type_ == Ort::TypeToTensorType<int32_t>::type)
          cuda::Launch_UpdatePositionIds(position_ids_->GetTensorMutableData<int32_t>(), static_cast<int>(position_ids_shape_[0]), model_.cuda_stream_);
        else
          cuda::Launch_UpdatePositionIds(position_ids_->GetTensorMutableData<int64_t>(), static_cast<int>(position_ids_shape_[0]), model_.cuda_stream_);
        break;
#endif
      default:
        throw std::runtime_error("PositionIDs::Update - Unsupported device type");
    }
  }

  {
    // Update attention mask
    assert(attention_mask_shape_[1] == current_length - 1);  // We should always be growing by 1
    attention_mask_shape_[1] = current_length;

    std::unique_ptr<OrtValue> next_attention_mask = OrtValue::CreateTensor(*model_.allocator_device_, attention_mask_shape_, type_);

    switch (model_.device_type_) {
      case DeviceType::CPU: {
        if (type_ == Ort::TypeToTensorType<int32_t>::type)
          UpdateAttentionMask(next_attention_mask->GetTensorMutableData<int32_t>(), attention_mask_->GetTensorData<int32_t>(), current_length);
        else
          UpdateAttentionMask(next_attention_mask->GetTensorMutableData<int64_t>(), attention_mask_->GetTensorData<int64_t>(), current_length);
        break;
      }
#if USE_CUDA
      case DeviceType::CUDA:
        if (type_ == Ort::TypeToTensorType<int32_t>::type)
          cuda::Launch_UpdateAttentionMask(next_attention_mask->GetTensorMutableData<int32_t>(), attention_mask_->GetTensorData<int32_t>(), static_cast<int>(attention_mask_shape_[0]), current_length, model_.cuda_stream_);
        else
          cuda::Launch_UpdateAttentionMask(next_attention_mask->GetTensorMutableData<int64_t>(), attention_mask_->GetTensorData<int64_t>(), static_cast<int>(attention_mask_shape_[0]), current_length, model_.cuda_stream_);
        break;
#endif
      default:
        throw std::runtime_error("PositionIDs::Update - Unsupported device type");
    }
    attention_mask_ = std::move(next_attention_mask);
    state_.inputs_[input_index_ + 1] = attention_mask_.get();
  }
}

template <typename T>
void PositionIDs::InitializeTensors(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths) {
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  auto* mask_data = attention_mask_->GetTensorMutableData<T>();
  auto* position_data = position_ids_->GetTensorMutableData<T>();
  const auto* word_id = state_.search_params_.input_ids.data();
  auto* mask = mask_data;
  auto* position = position_data;
  for (int i = 0; i < shape[0]; i++) {
    T abs_position = 0;
    for (int j = 0; j < shape[1]; j++, word_id++, mask++, position++) {
      if (*word_id == state_.search_params_.pad_token_id) {
        *mask = 0;
        *position = 0;
      } else {
        *mask = 1;
        *position = abs_position++;
      }
    }

    for (int k = 0; k < state_.search_params_.num_beams; k++) {
      sequence_lengths[i * state_.search_params_.num_beams + k] = static_cast<int32_t>(abs_position);
      initial_sequence_lengths_[i * state_.search_params_.num_beams + k] = static_cast<int32_t>(abs_position);
    }
  }
}

template <typename T>
void PositionIDs::UpdatePositionIDs() {
  // Increment position IDs
  auto* data = position_ids_->GetTensorMutableData<T>();
  for (int i = 0; i < position_ids_shape_[0]; i++) {
    data[i]++;
  }
};

template <typename T>
void PositionIDs::UpdateAttentionMask(T* data, const T* old_data, int current_length) {
  for (int i = 0; i < attention_mask_shape_[0]; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      data[i * current_length + j] = old_data[i * (current_length - 1) + j];
    }
    data[i * current_length + current_length - 1] = 1;
  }
};

}  // namespace Generators
