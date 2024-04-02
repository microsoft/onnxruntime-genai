#include "../generators.h"
#include "model.h"
#include "position_ids.h"
#include "kernels.h"

namespace Generators {

PositionIDs::PositionIDs(const Model& model, State& state, RoamingArray<int32_t>& sequence_lengths_unk)
    : model_{model},
      state_{state} {
  has_position_ids_ = model_.session_info_->HasInput(model_.config_->model.decoder.inputs.position_ids);
  type_ = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);

  if (type_ != Ort::TypeToTensorType<int32_t>::type && type_ != Ort::TypeToTensorType<int64_t>::type)
    throw std::runtime_error("position_ids & attention_mask only support int32 or int64 types");

  std::array<int64_t, 2> shape{state_.params_->batch_size, state_.params_->sequence_length};  // Only batch_size initially, as we haven't expanded over the beams yet
  position_ids_ = OrtValue::CreateTensor(model.allocator_cpu_, shape, type_);
  position_ids_next_ = OrtValue::CreateTensor(model.allocator_cpu_, std::array<int64_t, 2>{shape[0], 1}, type_);
  attention_mask_ = OrtValue::CreateTensor(model.allocator_cpu_, shape, type_);

  if (type_ == Ort::TypeToTensorType<int32_t>::type)
    InitializeTensors<int32_t>(shape, sequence_lengths_unk);
  else
    InitializeTensors<int64_t>(shape, sequence_lengths_unk);

  position_ids_ = model_.ExpandInputs(position_ids_, state_.params_->search.num_beams);
  position_ids_next_ = model_.ExpandInputs(position_ids_next_, state_.params_->search.num_beams);
  attention_mask_ = model_.ExpandInputs(attention_mask_, state_.params_->search.num_beams);
  shape[0] *= state_.params_->search.num_beams;
  position_ids_shape_ = shape;
  attention_mask_shape_ = shape;
}

void PositionIDs::Add() {
  input_index_ = state_.inputs_.size();

  if (has_position_ids_) {
    state_.inputs_.push_back(position_ids_.get());
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.position_ids.c_str());
  }

  state_.inputs_.push_back(attention_mask_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
}

void PositionIDs::Update(int current_length) {
  if (has_position_ids_) {
    // Reallocate position_ids for the 2nd and onward shape
    if (position_ids_next_) {
      position_ids_ = std::move(position_ids_next_);
      position_ids_shape_[1] = 1;
      state_.inputs_[input_index_] = position_ids_.get();
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
    state_.inputs_[input_index_ + has_position_ids_] = attention_mask_.get();
  }
}

template <typename T>
void PositionIDs::InitializeTensors(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths) {
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  auto* mask_data = attention_mask_->GetTensorMutableData<T>();
  auto* position_data = position_ids_->GetTensorMutableData<T>();
  auto* position_data_next = position_ids_next_->GetTensorMutableData<T>();
  const auto* word_id = state_.params_->input_ids.data();
  auto* mask = mask_data;
  auto* position = position_data;
  for (int i = 0; i < shape[0]; i++) {
    T abs_position = 0;
    for (int j = 0; j < shape[1]; j++, word_id++, mask++, position++) {
      if (*word_id == state_.params_->pad_token_id) {
        *mask = 0;
        *position = 0;
      } else {
        *mask = 1;
        *position = abs_position++;
      }
    }

    position_data_next[i] = abs_position;
    for (int k = 0; k < state_.params_->search.num_beams; k++) {
      sequence_lengths[i * state_.params_->search.num_beams + k] = static_cast<int32_t>(abs_position);
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
