#include "default_position_inputs.h"

#include "generator/generators.h"
#include "models/model.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace Generators {

DefaultPositionInputs::DefaultPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk, const std::string& attention_mask_name,
                                           std::optional<int> static_mask_capacity)
    : model_{model},
      state_{state},
      attention_mask_name_{attention_mask_name} {
  has_mask_input_ = model_.session_info_.HasInput(attention_mask_name_);
  has_posid_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.position_ids);

  type_ = Ort::TypeToTensorType<int32_t>;
  if (has_mask_input_) {
    type_ = model_.session_info_.GetInputDataType(attention_mask_name_);
  }
  if (has_posid_input_) {
    if (has_mask_input_) {
      if (model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.position_ids) != type_) {
        throw std::runtime_error("position_ids & attention_mask must have the same data type");
      }
    }
    type_ = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.position_ids);
  }

  if (type_ != Ort::TypeToTensorType<int32_t> && type_ != Ort::TypeToTensorType<int64_t>)
    throw std::runtime_error("position_ids & attention_mask only support int32 or int64 types");

  std::array<int64_t, 2> shape{state_.params_->search.batch_size, 0};  // Only batch_size initially, as we haven't expanded over the beams yet

  auto sequence_lengths = cpu_span<int32_t>{sequence_lengths_unk.CpuSpan()};
  if (type_ == Ort::TypeToTensorType<int32_t>)
    InitializeSequenceLengths<int32_t>(shape, sequence_lengths);
  else
    InitializeSequenceLengths<int64_t>(shape, sequence_lengths);
  sequence_lengths_unk.CopyCpuToDevice();

  position_ids_shape_ = shape;

  position_ids_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
  position_ids_next_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
  if (has_mask_input_)
    attention_mask_ = CreateAttentionMask(model_, state_, type_, static_mask_capacity);
}

void DefaultPositionInputs::Add() {
  if (has_posid_input_) {
    AddPositionIDs();
  }
  if (has_mask_input_) {
    AddAttentionMask();
  }
}

void DefaultPositionInputs::Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) {
  if (has_posid_input_) {
    // Initialize on first update
    if (is_first_update_) {
      position_ids_shape_[1] = new_length;
      if (type_ == Ort::TypeToTensorType<int32_t>)
        CreateAndInitializePositionIDs<int32_t>(next_tokens, position_ids_shape_);
      else
        CreateAndInitializePositionIDs<int64_t>(next_tokens, position_ids_shape_);
    } else {
      UpdatePositionIDs(total_length, new_length);
    }
  }
  if (has_mask_input_) {
    // Initialize on first update
    if (is_first_update_) {
      std::array<int64_t, 2> shape{state_.params_->search.batch_size, new_length};
      if (type_ == Ort::TypeToTensorType<int32_t>)
        CreateAndInitializeAttentionMask<int32_t>(next_tokens, shape);
      else
        CreateAndInitializeAttentionMask<int64_t>(next_tokens, shape);
    } else {
      UpdateAttentionMask(total_length, new_length);
    }
  }
  is_first_update_ = false;
}

void DefaultPositionInputs::RewindTo(size_t index) {
  // Reset the state of the position inputs
  if (index == 0) {
    is_first_update_ = true;
    // Position ids next is set to nullptr after the first Run() call. This restores it
    if (has_posid_input_)
      position_ids_next_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
    // Rewind the mask input to a previous state
  } else if (has_mask_input_) {
    if (attention_mask_->GetShape()[0] == 1) {
      attention_mask_->RewindTo(index);
    } else
      throw std::runtime_error("DefaultPositionInputs::RewindTo - Unsupported batch size");
  }
}

void DefaultPositionInputs::AddAttentionMask() {
  mask_input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(attention_mask_->GetOrtTensor());
  state_.input_names_.push_back(attention_mask_name_.c_str());
}

void DefaultPositionInputs::AddPositionIDs() {
  posid_input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(position_ids_->GetOrtTensor());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.position_ids.c_str());
}

void DefaultPositionInputs::CreateNextPositionIDsTensor() {
  // position_ids_next_ tensor is allocated and initialized in anticipation of token generation
  if (position_ids_next_ && position_ids_shape_[0] > 1 && position_ids_shape_[1] == 1) {
    position_ids_ = std::move(position_ids_next_);
    position_ids_next_ = nullptr;
  } else {
    const int max_cap = state_.params_->max_graph_capture_length;
    const bool use_static = state_.params_->use_graph_capture && position_ids_shape_[1] >= 1 && position_ids_shape_[1] <= max_cap;
    const size_t static_cap_bytes = use_static ? static_cast<size_t>(position_ids_shape_[0]) * max_cap * Ort::SizeOf(type_) : 0;
    position_ids_->CreateTensor(position_ids_shape_, use_static, static_cap_bytes);
  }
}

void DefaultPositionInputs::UpdatePositionIDs(int total_length, int new_kv_length) {
  if (position_ids_shape_[0] != 1 && !(total_length == 0 || new_kv_length == 1))
    throw std::runtime_error("DefaultPositionInputs::UpdatePositionIDs - batch_size must be 1 for continuous decoding.");

  // Reallocate position_ids when new_kv_length changes
  if (position_ids_shape_[1] != new_kv_length) {
    position_ids_shape_[1] = new_kv_length;
    CreateNextPositionIDsTensor();
    state_.inputs_[posid_input_index_] = position_ids_->GetOrtTensor();
  }
  // Try to update position ids on the device. If it fails, copy to CPU, update there, and copy back to device.
  if (!model_.p_device_inputs_->UpdatePositionIds(position_ids_->GetMutableRawData(), static_cast<int>(position_ids_shape_[0]), total_length, new_kv_length, type_)) {
    auto position_ids_span = position_ids_->GetByteSpan();
    model_.p_device_inputs_->GetCpuFallbackDevice().UpdatePositionIds(position_ids_span.CopyDeviceToCpu().data(), static_cast<int>(position_ids_shape_[0]), total_length, new_kv_length, type_);
    position_ids_span.CopyCpuToDevice();
  }
}

void DefaultPositionInputs::UpdateAttentionMask(int total_length, int new_kv_length) {
  attention_mask_->Update(total_length, new_kv_length);
  state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
}

template <typename T>
void DefaultPositionInputs::CreateAndInitializePositionIDs(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape) {
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  auto position_ids = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  auto* position_data = position_ids->GetTensorMutableData<T>();
  auto position_ids_next = OrtValue::CreateTensor(model_.allocator_cpu_, std::array<int64_t, 2>{shape[0], 1}, type_);
  auto* position_data_next = position_ids_next->GetTensorMutableData<T>();
  // If batch_size is 1 we have no padding, so we do simple ascending
  if (shape[0] == 1) {
    for (int i = 0; i < shape[1]; ++i) {
      position_data[i] = static_cast<T>(i);
    }
    position_data_next[0] = static_cast<T>(shape[1]) - 1;
    // Otherwise we iterate backwards as to not misinterpret any right pad tokens
  } else {
    const auto* word_id = const_cast<DeviceSpan<int32_t>&>(next_tokens).CpuSpan().data() + shape[0] * shape[1] - 1;
    auto* position = position_data + shape[0] * shape[1] - 1;
    bool found_first_non_pad = false;
    for (int i = static_cast<int>(shape[0] - 1); i >= 0; i--) {
      T abs_position = static_cast<T>(shape[1] - 1);
      found_first_non_pad = false;
      for (int j = static_cast<int>(shape[1] - 1); j >= 0; j--, word_id--, position--) {
        // Non-pad tokens are set to their corresponding position
        if (found_first_non_pad) {
          *position = abs_position;
          // If we found first non-padding token, we can now set the rest of the positions to non-0 values
        } else if (*word_id != model_.config_->model.pad_token_id) {
          found_first_non_pad = true;
          *position = abs_position;
          position_data_next[i] = abs_position;
          // We have not found any non-padding token yet so we set the position to 0
        } else {
          *position = 0;
        }
        abs_position--;
      }
    }
  }

  // Move tensors to appropriate device and expand by num_beams
  position_ids_->ort_tensor_ = model_.ExpandInputs(position_ids, state_.params_->search.num_beams);
  position_ids_next_->ort_tensor_ = model_.ExpandInputs(position_ids_next, state_.params_->search.num_beams);
  if (state_.params_->use_graph_capture)
    position_ids_next_->MakeStatic();
  position_ids_shape_[0] *= state_.params_->search.num_beams;
  state_.inputs_[posid_input_index_] = position_ids_->GetOrtTensor();
}

template <typename T>
void DefaultPositionInputs::CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape) {
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  auto attention_mask = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  auto* mask_data = attention_mask->GetTensorMutableData<T>();
  // If batch size is 1, we have no padding, so we simply set all tokens to 1
  if (shape[0] == 1) {
    for (int i = 0; i < shape[1]; ++i) {
      mask_data[i] = 1;
    }
    // Otherwise we iterate backwards as to not misinterpret any right pad tokens
  } else {
    auto* mask = mask_data + shape[0] * shape[1] - 1;
    const auto* word_id = const_cast<DeviceSpan<int32_t>&>(next_tokens).CpuSpan().data() + shape[0] * shape[1] - 1;
    bool found_first_non_pad = false;
    for (int i = static_cast<int>(shape[0] - 1); i >= 0; i--) {
      found_first_non_pad = false;
      for (int j = static_cast<int>(shape[1] - 1); j >= 0; j--, word_id--, mask--) {
        if (found_first_non_pad) {
          *mask = 1;
        } else if (*word_id != model_.config_->model.pad_token_id) {
          found_first_non_pad = true;
          *mask = 1;
        } else {
          *mask = 0;
        }
      }
    }
  }

  attention_mask_->Initialize(std::move(attention_mask));
  state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
}

template <typename T>
void DefaultPositionInputs::InitializeSequenceLengths(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths_unk) {
  for (int i = 0; i < shape[0] * state_.params_->search.num_beams; i++) {
    sequence_lengths_unk[i] = 0;
  }
}

}  // namespace Generators
