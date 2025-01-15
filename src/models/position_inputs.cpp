#include "../generators.h"
#include "model.h"
#include "position_inputs.h"

namespace Generators {

PositionInputs::PositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk)
    : model_{model},
      state_{state} {
  has_mask_input_ = model_.session_info_->HasInput(model_.config_->model.decoder.inputs.attention_mask);
  has_posid_input_ = model_.session_info_->HasInput(model_.config_->model.decoder.inputs.position_ids);

  type_ = Ort::TypeToTensorType<int32_t>;
  if (has_mask_input_) {
    type_ = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);
  }
  if (has_posid_input_) {
    if (has_mask_input_) {
      if (model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.position_ids) != type_) {
        throw std::runtime_error("position_ids & attention_mask must have the same data type");
      }
    }
    type_ = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.position_ids);
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
  attention_mask_shape_ = shape;

  if (state_.GetCapturedGraphInfo()) {
    if (has_posid_input_) {
      sb_position_ids_ = state_.GetCapturedGraphInfo()->sb_position_ids_.get();
    }
    if (has_mask_input_) {
      sb_attention_mask_ = state_.GetCapturedGraphInfo()->sb_attention_mask_.get();
    }
  }
}

void PositionInputs::Add() {
  if (has_posid_input_) {
    AddPositionIDs();
  }
  if (has_mask_input_) {
    AddAttentionMask();
  }
}

void PositionInputs::Update(const DeviceSpan<int32_t>& next_tokens, int total_length, int new_length) {
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
      attention_mask_shape_[1] = new_length;
      if (type_ == Ort::TypeToTensorType<int32_t>)
        CreateAndInitializeAttentionMask<int32_t>(next_tokens, attention_mask_shape_);
      else
        CreateAndInitializeAttentionMask<int64_t>(next_tokens, attention_mask_shape_);
    } else {
      UpdateAttentionMask(total_length, new_length);
    }
  }
  is_first_update_ = false;
}

void PositionInputs::RewindTo(size_t index) {
  // Reset the state of the position inputs
  if (index == 0) {
    is_first_update_ = true;
    is_first_mask_update_ = true;
    // Rewind the mask input to a previous state
  } else if (has_mask_input_) {
    if (attention_mask_shape_[0] == 1) {
      RewindMask(index);
    } else
      throw std::runtime_error("PositionInputs::RewindTo - Unsupported batch size");
  }
}

void PositionInputs::AddAttentionMask() {
  mask_input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(attention_mask_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
}

void PositionInputs::AddPositionIDs() {
  posid_input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(position_ids_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.position_ids.c_str());
}

void PositionInputs::CreateNextPositionIDsTensor() {
  if (!sb_position_ids_) {
    if (position_ids_shape_[1] == 1 && position_ids_next_) {
      position_ids_ = std::move(position_ids_next_);
      position_ids_next_ = nullptr;
    } else {
      position_ids_ = OrtValue::CreateTensor(*model_.allocator_device_, position_ids_shape_, type_);
    }
  } else {
    position_ids_ = sb_position_ids_->CreateTensorOnStaticBuffer(position_ids_shape_, type_);
    if (position_ids_shape_[1] == 1) {
      auto position_ids_span = ByteWrapTensor(*model_.p_device_, *position_ids_);
      auto position_ids_next_span = ByteWrapTensor(*model_.p_device_, *position_ids_next_);
      position_ids_span.CopyFrom(position_ids_next_span);
    }
  }
}

void PositionInputs::UpdatePositionIDs(int total_length, int new_kv_length) {
  if (position_ids_shape_[0] != 1 && !(total_length == 0 || new_kv_length == 1))
    throw std::runtime_error("PositionInputs::UpdatePositionIDs - batch_size must be 1 for continuous decoding.");
  if (DeviceType::DML == model_.device_type_ && !(total_length == 0 || new_kv_length == 1))
    throw std::runtime_error("PositionInputs::UpdatePositionIDs - DML does not support continuous decoding.");

  // Reallocate position_ids when new_kv_length changes
  if (position_ids_shape_[1] != new_kv_length) {
    position_ids_shape_[1] = new_kv_length;
    CreateNextPositionIDsTensor();
    state_.inputs_[posid_input_index_] = position_ids_.get();
  }

  switch (model_.device_type_) {
    case DeviceType::WEBGPU:
    case DeviceType::DML:
    case DeviceType::CPU: {
      type_ == Ort::TypeToTensorType<int32_t> ? UpdatePositionIDsImpl<int32_t>(total_length, new_kv_length)
                                              : UpdatePositionIDsImpl<int64_t>(total_length, new_kv_length);
      break;
    }
    case DeviceType::CUDA: {
      model_.p_device_->UpdatePositionIds(position_ids_->GetTensorMutableRawData(), static_cast<int>(position_ids_shape_[0]), total_length, new_kv_length, type_);
      break;
    }
    default:
      throw std::runtime_error("PositionIDs::Update - Unsupported device type");
  }
}

void PositionInputs::CreateNextAttentionMaskTensor(int total_length) {
  if (!sb_attention_mask_) {
    attention_mask_shape_[1] = total_length;
    attention_mask_next_ = OrtValue::CreateTensor(*model_.allocator_device_, attention_mask_shape_, type_);
  } else {
    attention_mask_shape_[1] = state_.params_->search.max_length;
    attention_mask_next_ = sb_attention_mask_->CreateTensorOnStaticBuffer(attention_mask_shape_, type_);
    if (is_first_mask_update_) {
      ByteWrapTensor(*model_.p_device_, *attention_mask_next_).Zero();
    }
  }
}

void PositionInputs::UpdateAttentionMask(int total_length, int new_kv_length) {
  if (position_ids_shape_[0] != 1 && !(total_length == 0 || new_kv_length == 1))
    throw std::runtime_error("PositionInputs::UpdatePositionIDs - batch_size must be 1 for continuous decoding.");
  if (DeviceType::DML == model_.device_type_ && !(total_length == 0 || new_kv_length == 1))
    throw std::runtime_error("PositionInputs::UpdatePositionIDs - DML does not support continuous decoding.");

  CreateNextAttentionMaskTensor(total_length);
  state_.inputs_[mask_input_index_] = attention_mask_.get();

  switch (model_.device_type_) {
    case DeviceType::WEBGPU:
    case DeviceType::DML:
    case DeviceType::CPU: {
      type_ == Ort::TypeToTensorType<int32_t> ? UpdateAttentionMaskImpl<int32_t>(total_length)
                                              : UpdateAttentionMaskImpl<int64_t>(total_length);
      break;
    }
    case DeviceType::CUDA: {
        int max_length = sb_attention_mask_ ? state_.params_->search.max_length : total_length;
      bool update_only = sb_attention_mask_ && !is_first_mask_update_;
      model_.p_device_->UpdateAttentionMask(attention_mask_next_->GetTensorMutableRawData(),
                                            attention_mask_->GetTensorRawData(),
                                            static_cast<int>(attention_mask_shape_[0]),
                                            new_kv_length,
                                            total_length,
                                            max_length,
                                            update_only,
                                            type_);
      break;
    }

    default:
      throw std::runtime_error("PositionInputs::Update - Unsupported device type");
  }

  attention_mask_ = std::move(attention_mask_next_);
  state_.inputs_[mask_input_index_] = attention_mask_.get();
  is_first_mask_update_ = false;
}

template <typename T>
void PositionInputs::CreateAndInitializePositionIDs(const DeviceSpan<int32_t>& next_tokens, std::array<int64_t, 2> shape) {
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  position_ids_ = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  position_ids_next_ = OrtValue::CreateTensor(model_.allocator_cpu_, std::array<int64_t, 2>{shape[0], 1}, type_);
  auto* position_data = position_ids_->GetTensorMutableData<T>();
  auto* position_data_next = position_ids_next_->GetTensorMutableData<T>();
  const auto* word_id = const_cast<DeviceSpan<int32_t>&>(next_tokens).CpuSpan().data();
  auto* position = position_data;
  for (int i = 0; i < shape[0]; i++) {
    T abs_position = 0;
    for (int j = 0; j < shape[1]; j++, word_id++, position++) {
      if (*word_id == model_.config_->model.pad_token_id) {
        *position = 0;
      } else {
        *position = abs_position++;
      }
    }

    position_data_next[i] = abs_position - 1;
  }

  // Move tensors to appropriate device and expand by num_beams
  position_ids_ = model_.ExpandInputs(position_ids_, state_.params_->search.num_beams);
  position_ids_next_ = model_.ExpandInputs(position_ids_next_, state_.params_->search.num_beams);
  position_ids_shape_[0] *= state_.params_->search.num_beams;
  state_.inputs_[posid_input_index_] = position_ids_.get();
}

template <typename T>
void PositionInputs::CreateAndInitializeAttentionMask(const DeviceSpan<int32_t>& next_tokens, std::array<int64_t, 2> shape) {
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  attention_mask_ = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  auto* mask_data = attention_mask_->GetTensorMutableData<T>();
  const auto* word_id = const_cast<DeviceSpan<int32_t>&>(next_tokens).CpuSpan().data();
  auto* mask = mask_data;
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++, word_id++, mask++) {
      if (*word_id == model_.config_->model.pad_token_id) {
        *mask = 0;
      } else {
        *mask = 1;
      }
    }
  }

  // Move tensors to appropriate device and expand by num_beams
  attention_mask_ = model_.ExpandInputs(attention_mask_, state_.params_->search.num_beams);
  attention_mask_shape_[0] *= state_.params_->search.num_beams;
  state_.inputs_[mask_input_index_] = attention_mask_.get();
}

template <typename T>
void PositionInputs::InitializeSequenceLengths(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths_unk) {
  for (int i = 0; i < shape[0] * state_.params_->search.num_beams; i++) {
    sequence_lengths_unk[i] = 0;
  }
}

template <typename T>
void PositionInputs::UpdatePositionIDsImpl(int total_length, int new_kv_length) {
  auto* data = position_ids_->GetTensorMutableData<T>();
  if (position_ids_shape_[0] == 1) {
    // For batch size == 1 we calculate position ids with total length and new kv length for continuous decoding
    for (int i = 0; i < new_kv_length; i++)
      data[i] = i + total_length - new_kv_length;
  } else {
    // For batch size > 1 we increment position ids by 1... continuous decoding is not supported
    for (int i = 0; i < position_ids_shape_[0]; i++)
      data[i]++;
  }
}

template <typename T>
void PositionInputs::UpdateAttentionMaskImpl(int total_length) {
  auto* data = attention_mask_next_->GetTensorMutableData<T>();
  auto* old_data = attention_mask_->GetTensorData<T>();
  if (attention_mask_shape_[0] == 1) {
    // For batch size == 1 we assume no padding. We make this explicit for continuous decoding.
    for (int i = 0; i < total_length; i++)
      data[i] = 1;
  } else {
    // For batch size > 1 we increment attention mask by 1... continuous decoding is not supported
    for (int i = 0; i < attention_mask_shape_[0]; i++) {
      for (int j = 0; j < total_length - 1; j++) {
        data[i * total_length + j] = old_data[i * (total_length - 1) + j];
      }
      data[i * total_length + total_length - 1] = 1;
    }
  }
}

void PositionInputs::RewindMask(size_t index) {
  if (sb_attention_mask_ && !is_first_mask_update_) {
    throw std::runtime_error("PositionInputs::RewindMask - Static buffer is not supported for continuous decoding.");
#if 0  // TODO: Fix implementation, cudaMemsetAsync of 1 is setting bytes of 1 vs int32's of 1
    int past_length = static_cast<int>(index);
    int max_length = static_cast<int>(state_.params_->search.max_length);
    cudaMemsetAsync(attention_mask_->GetTensorMutableRawData(),
                    0,
                    (type_ == Ort::TypeToTensorType<int32_t> ? sizeof(int32_t) : sizeof(int64_t)) * max_length,
                    model_.cuda_stream_);
    cudaMemsetAsync(attention_mask_->GetTensorMutableRawData(),
                    1,
                    (type_ == Ort::TypeToTensorType<int32_t> ? sizeof(int32_t) : sizeof(int64_t)) * past_length,
                    model_.cuda_stream_);
#endif
  }
}

}  // namespace Generators
