#include "../generators.h"
#include "model.h"
#include "position_inputs.h"

namespace Generators {

DefaultPositionInputs::DefaultPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk)
    : model_{model},
      state_{state} {
  has_mask_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.attention_mask);
  has_posid_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.position_ids);

  type_ = Ort::TypeToTensorType<int32_t>;
  if (has_mask_input_) {
    type_ = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);
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
  attention_mask_shape_ = shape;

  position_ids_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
  position_ids_next_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
  attention_mask_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
  attention_mask_next_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
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

void DefaultPositionInputs::RewindTo(size_t index) {
  // Reset the state of the position inputs
  if (index == 0) {
    is_first_update_ = true;
    // Position ids next is set to nullptr after the first Run() call. This restores it
    if (has_posid_input_)
      position_ids_next_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
    // Rewind the mask input to a previous state
  } else if (has_mask_input_) {
    if (attention_mask_shape_[0] == 1) {
      RewindMask(index);
    } else
      throw std::runtime_error("DefaultPositionInputs::RewindTo - Unsupported batch size");
  }
}

void DefaultPositionInputs::AddAttentionMask() {
  mask_input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(attention_mask_->GetOrtTensor());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
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
    position_ids_->CreateTensor(position_ids_shape_, state_.params_->use_graph_capture && position_ids_shape_[1] == 1);
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
    GetDeviceInterface(DeviceType::CPU)->UpdatePositionIds(position_ids_span.CopyDeviceToCpu().data(), static_cast<int>(position_ids_shape_[0]), total_length, new_kv_length, type_);
    position_ids_span.CopyCpuToDevice();
  }
}

void DefaultPositionInputs::CreateNextAttentionMaskTensor(int total_length) {
  if (state_.params_->use_graph_capture)
    return;
  attention_mask_shape_[1] = total_length;
  attention_mask_next_->CreateTensor(attention_mask_shape_);
}

void DefaultPositionInputs::UpdateAttentionMask(int total_length, int new_kv_length) {
  if (position_ids_shape_[0] != 1 && !(total_length == 0 || new_kv_length == 1))
    throw std::runtime_error("DefaultPositionInputs::UpdatePositionIDs - batch_size must be 1 for continuous decoding.");

  CreateNextAttentionMaskTensor(total_length);

  // Update the attention mask on the device. If it fails, copy to CPU, update there, and copy back to device.
  if (!model_.p_device_inputs_->UpdateAttentionMask(state_.params_->use_graph_capture ? nullptr : attention_mask_next_->GetMutableRawData(),
                                                    attention_mask_->GetMutableRawData(),
                                                    static_cast<int>(attention_mask_shape_[0]),
                                                    new_kv_length,
                                                    total_length,
                                                    state_.params_->search.max_length,
                                                    state_.params_->use_graph_capture,
                                                    type_)) {
    // auto* attention_mask_next_span = state_.params_->use_graph_capture ? &attention_mask_next_->GetByteSpan() : nullptr;
    DeviceSpan<uint8_t> attention_mask_next_span;
    if (!state_.params_->use_graph_capture)
      attention_mask_next_span = attention_mask_next_->GetByteSpan();
    auto attention_mask_span = attention_mask_->GetByteSpan();
    GetDeviceInterface(DeviceType::CPU)->UpdateAttentionMask(state_.params_->use_graph_capture ? nullptr : attention_mask_next_span.CopyDeviceToCpu().data(), attention_mask_span.CopyDeviceToCpu().data(), static_cast<int>(attention_mask_shape_[0]), new_kv_length, total_length, state_.params_->search.max_length, state_.params_->use_graph_capture, type_);
    if (!state_.params_->use_graph_capture)
      attention_mask_next_span.CopyCpuToDevice();
    attention_mask_span.CopyCpuToDevice();
  }

  if (!state_.params_->use_graph_capture) {
    attention_mask_->ort_tensor_ = std::move(attention_mask_next_->ort_tensor_);
    state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
  }
}

template <typename T>
void DefaultPositionInputs::CreateAndInitializePositionIDs(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape) {
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  auto position_ids = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  auto position_ids_next = OrtValue::CreateTensor(model_.allocator_cpu_, std::array<int64_t, 2>{shape[0], 1}, type_);
  auto* position_data = position_ids->GetTensorMutableData<T>();
  auto* position_data_next = position_ids_next->GetTensorMutableData<T>();
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
  position_ids_->ort_tensor_ = model_.ExpandInputs(position_ids, state_.params_->search.num_beams);
  position_ids_next_->ort_tensor_ = model_.ExpandInputs(position_ids_next, state_.params_->search.num_beams);
  if (state_.params_->use_graph_capture)
    position_ids_next_->MakeStatic();
  position_ids_shape_[0] *= state_.params_->search.num_beams;
  state_.inputs_[posid_input_index_] = position_ids_->GetOrtTensor();
}

// Initialize a static attention mask of size max_length and expanded by num_beams
template <typename T>
void DefaultPositionInputs::InitializeStaticMask(OrtValue& cpu_attention_mask) {
  // Create static tensor of size max_length and expanded by num_beams
  attention_mask_shape_[0] *= state_.params_->search.num_beams;
  attention_mask_shape_[1] = state_.params_->search.max_length;
  attention_mask_->CreateTensor(attention_mask_shape_, true);
  auto output_span = attention_mask_->GetDeviceSpan<T>();
  output_span.Zero();
  // Copy the first new_kv_length elements of each sequence num_beams times each
  auto input_span = WrapTensor<T>(*GetDeviceInterface(DeviceType::CPU), cpu_attention_mask);
  auto input_shape = cpu_attention_mask.GetTensorTypeAndShapeInfo()->GetShape();
  auto batch_size = input_shape[0];
  auto num_beams = state_.params_->search.num_beams;
  auto new_kv_length = input_shape[1];
  auto max_length = input_shape[1] * num_beams;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++) {
      auto output_subspan = output_span.subspan((i * num_beams + j) * max_length, new_kv_length);
      auto input_subspan = input_span.subspan(i * new_kv_length, new_kv_length);
      output_subspan.CopyFrom(input_subspan);
    }
  }
}

template void DefaultPositionInputs::InitializeStaticMask<int32_t>(OrtValue& cpu_attention_mask);
template void DefaultPositionInputs::InitializeStaticMask<int64_t>(OrtValue& cpu_attention_mask);

template <typename T>
void DefaultPositionInputs::CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape) {
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  auto attention_mask = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  auto* mask_data = attention_mask->GetTensorMutableData<T>();
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

  if (state_.params_->use_graph_capture) {
    InitializeStaticMask<T>(*attention_mask);
  } else {
    attention_mask = model_.ExpandInputs(attention_mask, state_.params_->search.num_beams);
    attention_mask_->ort_tensor_ = std::move(attention_mask);
    attention_mask_shape_[0] *= state_.params_->search.num_beams;
  }
  state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
}

template <typename T>
void DefaultPositionInputs::InitializeSequenceLengths(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths_unk) {
  for (int i = 0; i < shape[0] * state_.params_->search.num_beams; i++) {
    sequence_lengths_unk[i] = 0;
  }
}

void DefaultPositionInputs::RewindMask(size_t index) {
  if (state_.params_->use_graph_capture) {
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

// TODO: SlidingWindow does not support graph capture
WindowedPositionInputs::WindowedPositionInputs(State& state)
    : state_{state} {
  has_posid_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.position_ids);
  has_mask_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.attention_mask);

  if (has_posid_input_ || has_mask_input_) {
    if (!model_.config_->model.decoder.sliding_window.has_value()) {
      throw std::runtime_error("Sliding a window over position_ids and attention_mask requires sliding_window to be set in the genai_config.json.");
    }
    window_size_ = model_.config_->model.decoder.sliding_window->window_size;

    if (window_size_ == 0) {
      throw std::runtime_error("Window size must be greater than 0");
    }
  }

  if (has_posid_input_) {
    position_ids_type_ = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.position_ids);
    if (position_ids_type_ != Ort::TypeToTensorType<int32_t>)
      throw std::runtime_error("WindowedPositionInputs only supports int32_t position_ids");

    position_ids_shape_ = {1, model_.config_->model.decoder.sliding_window->window_size};
  }

  if (has_mask_input_) {
    attention_mask_type_ = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);
    if (attention_mask_type_ != Ort::TypeToTensorType<int32_t>)
      throw std::runtime_error("WindowedPositionInputs only supports int32_t attention_mask");

    attention_mask_shape_ = {1, model_.config_->model.context_length};
  }
}

void WindowedPositionInputs::Add() {
  if (has_posid_input_) {
    position_ids_index_ = state_.inputs_.size();
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.position_ids.c_str());
    state_.inputs_.push_back(position_ids_.get());
  }

  if (has_mask_input_) {
    attention_mask_index_ = state_.inputs_.size();
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
    state_.inputs_.push_back(attention_mask_.get());
  }
}

void WindowedPositionInputs::Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) {
  if (!has_posid_input_ && !has_mask_input_) {
    return;
  }

  if (window_index_ == 0) {
    num_windows_ = (next_tokens.size() + window_size_ - 1) / window_size_;
    if (has_posid_input_) {
      position_ids_ = OrtValue::CreateTensor(model_.allocator_cpu_, position_ids_shape_, position_ids_type_);

      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 0, position_ids_ -> [0, 0, 1]
      auto* position_ids_data = position_ids_->GetTensorMutableData<int32_t>();
      for (int i = 0, j = 0; i < position_ids_shape_[1]; i++) {
        if (next_tokens.Span()[i] == model_.config_->model.pad_token_id) {
          position_ids_data[i] = 0;
        } else {
          position_ids_data[i] = j++;
        }
      }
    }

    if (has_mask_input_) {
      attention_mask_ = OrtValue::CreateTensor(model_.allocator_cpu_, attention_mask_shape_, attention_mask_type_);

      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 0, attention_mask_ -> ([0] * context_length - window_size_) + [0, 1, 1]
      auto* attention_mask_data = attention_mask_->GetTensorMutableData<int32_t>();
      std::fill_n(attention_mask_data, attention_mask_shape_[1] - window_size_, 0);
      for (size_t i = 0; i < window_size_; i++) {
        attention_mask_data[attention_mask_shape_[1] - window_size_ + i] = next_tokens.CpuSpan()[i] == model_.config_->model.pad_token_id ? 0 : 1;
      }
      for (size_t i = 0; i < window_size_; i++) {
        if (attention_mask_data[attention_mask_shape_[1] - window_size_ + i] == 1) {
          attention_mask_backward_offset_ = attention_mask_shape_[1] - window_size_ + i - 1;
          break;
        }
      }
    }
  } else if (window_index_ < num_windows_) {
    if (has_posid_input_) {
      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 1, position_ids_ -> [2, 3, 4]

      auto* position_ids_data = position_ids_->GetTensorMutableData<int32_t>();
      const auto last_position = position_ids_data[window_size_ - 1];
      std::iota(position_ids_data, position_ids_data + window_size_, last_position + 1);
    }

    if (has_mask_input_) {
      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 1, attention_mask_ -> ([0] * context_length - (2 * window_size_)) + [0, 1, 1, 1, 1, 1]
      auto* attention_mask_data = attention_mask_->GetTensorMutableData<int32_t>();
      std::fill_n(attention_mask_data + attention_mask_backward_offset_ - window_size_ + 1, window_size_, 1);
      attention_mask_backward_offset_ -= window_size_;
    }
  } else {
    // All prompt token chunks have been processed. Now we process the tokens generated by the model.
    if (has_posid_input_) {
      // next_tokens -> [f]
      // position_ids_ -> [5]
      const auto last_position = position_ids_->GetTensorData<int32_t>()[position_ids_shape_[1] - 1];
      if (position_ids_shape_[1] != 1) {
        position_ids_shape_[1] = 1;
        position_ids_ = OrtValue::CreateTensor(model_.allocator_cpu_, position_ids_shape_, position_ids_type_);
      }
      position_ids_->GetTensorMutableData<int32_t>()[0] = last_position + 1;
    }

    if (has_mask_input_) {
      // next_tokens -> [f]
      // attention_mask_ -> ([0] * context_length - (2 * window_size_) - 1) + [0, 1, 1, 1, 1, 1, 1]
      attention_mask_->GetTensorMutableData<int32_t>()[attention_mask_backward_offset_] = 1;
      if (attention_mask_backward_offset_ > 0) {
        attention_mask_backward_offset_ -= 1;
      }
    }
  }

  if (has_posid_input_) {
    state_.inputs_[position_ids_index_] = position_ids_.get();
  }

  if (has_mask_input_) {
    state_.inputs_[attention_mask_index_] = attention_mask_.get();
  }

  window_index_++;
}

std::unique_ptr<PositionInputs> CreatePositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths) {
  if (state.model_.config_->model.decoder.sliding_window.has_value()) {
    return std::make_unique<WindowedPositionInputs>(state);
  } else {
    return std::make_unique<DefaultPositionInputs>(state.model_, state, sequence_lengths);
  }
}

}  // namespace Generators
