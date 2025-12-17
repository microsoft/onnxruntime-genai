#include "../generators.h"
#include "model.h"
#include "position_inputs.h"
#include "model_type.h"
#include <vector>
#include <numeric>
#include <cmath>  // For std::round

namespace Generators {

// Forward declaration
struct Qwen2VLPositionInputs;

// Helper to dispatch type-specific tensor operations
template <typename Func>
void DispatchOnType(ONNXTensorElementDataType type, Func&& func) {
  if (type == Ort::TypeToTensorType<int32_t>)
    func.template operator()<int32_t>();
  else
    func.template operator()<int64_t>();
}

// Functors for Qwen2VL position inputs
struct UpdatePositionIdsFunctor {
  OrtValue* position_ids;
  int base_pos;
  int64_t batch_size;
  int64_t seq_len;
  const std::vector<int64_t>& rope_deltas;

  template <typename T>
  void operator()() {
    auto* data = position_ids->GetTensorMutableData<T>();
    for (int64_t dim = 0; dim < 3; ++dim) {
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
          T delta = static_cast<T>(base_pos + rope_deltas[b]);
          T pos = static_cast<T>(s);
          data[dim * batch_size * seq_len + b * seq_len + s] = delta + pos;
        }
      }
    }
  }
};

struct FillMaskFunctor {
  OrtValue* attention_mask;
  int64_t total_size;

  template <typename T>
  void operator()() {
    auto* mask_data = attention_mask->GetTensorMutableData<T>();
    std::fill_n(mask_data, total_size, static_cast<T>(1));
  }
};

struct InitPositionIdsFunctor {
  Qwen2VLPositionInputs* self;
  DeviceSpan<int32_t> next_tokens;
  std::array<int64_t, 3> position_ids_shape;

  template <typename T>
  void operator()() {
    self->CreateAndInitialize3DPositionIDs<T>(next_tokens, position_ids_shape);
  }
};

struct InitAttentionMaskFunctor {
  Qwen2VLPositionInputs* self;
  DeviceSpan<int32_t> next_tokens;
  std::array<int64_t, 2> attention_mask_shape;

  template <typename T>
  void operator()() {
    self->CreateAndInitializeAttentionMask<T>(next_tokens, attention_mask_shape);
  }
};

DefaultPositionInputs::DefaultPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk, const std::string& attention_mask_name)
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
  if (ShouldUseStaticMaskHandling())
    return;
  attention_mask_shape_[1] = total_length;
  attention_mask_next_->CreateTensor(attention_mask_shape_);
}

void DefaultPositionInputs::UpdateAttentionMask(int total_length, int new_kv_length) {
  if (position_ids_shape_[0] != 1 && !(total_length == 0 || new_kv_length == 1))
    throw std::runtime_error("DefaultPositionInputs::UpdatePositionIDs - batch_size must be 1 for continuous decoding.");

  CreateNextAttentionMaskTensor(total_length);

  // Update the attention mask on the device. If it fails, copy to CPU, update there, and copy back to device.
  if (!model_.p_device_inputs_->UpdateAttentionMask(ShouldUseStaticMaskHandling() ? nullptr : attention_mask_next_->GetMutableRawData(),
                                                    attention_mask_->GetMutableRawData(),
                                                    static_cast<int>(attention_mask_shape_[0]),
                                                    new_kv_length,
                                                    total_length,
                                                    state_.params_->search.max_length,
                                                    ShouldUseStaticMaskHandling(),
                                                    type_)) {
    // auto* attention_mask_next_span = state_.params_->use_graph_capture ? &attention_mask_next_->GetByteSpan() : nullptr;
    DeviceSpan<uint8_t> attention_mask_next_span;
    if (!ShouldUseStaticMaskHandling())
      attention_mask_next_span = attention_mask_next_->GetByteSpan();
    auto attention_mask_span = attention_mask_->GetByteSpan();
    GetDeviceInterface(DeviceType::CPU)->UpdateAttentionMask(ShouldUseStaticMaskHandling() ? nullptr : attention_mask_next_span.CopyDeviceToCpu().data(), attention_mask_span.CopyDeviceToCpu().data(), static_cast<int>(attention_mask_shape_[0]), new_kv_length, total_length, state_.params_->search.max_length, ShouldUseStaticMaskHandling(), type_);
    if (!ShouldUseStaticMaskHandling())
      attention_mask_next_span.CopyCpuToDevice();
    attention_mask_span.CopyCpuToDevice();
  }

  if (!ShouldUseStaticMaskHandling()) {
    attention_mask_->ort_tensor_ = std::move(attention_mask_next_->ort_tensor_);
    state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
  }
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
  // Number of elements per (batch * beam) row in the *output* tensor
  // equals the full max_length configured for generation, which is
  // attention_mask_shape_[1] after the assignment above.
  auto max_length = attention_mask_shape_[1];
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

  if (ShouldUseStaticMaskHandling()) {
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

bool DefaultPositionInputs::ShouldUseStaticMaskHandling() const {
  return state_.params_->use_graph_capture ||
         (state_.params_->IsPastPresentShareBufferEnabled(model_.config_->model.type) &&
          model_.p_device_->GetType() == DeviceType::NvTensorRtRtx);
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

// Qwen2VLPositionInputs implementation
Qwen2VLPositionInputs::Qwen2VLPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk)
    : model_{model},
      state_{state},
      image_token_id_{model.config_->model.image_token_id},
      video_token_id_{model.config_->model.video_token_id},
      vision_start_token_id_{model.config_->model.vision_start_token_id},
      tokens_per_second_{model.config_->model.vision.tokens_per_second},
      spatial_merge_size_{model.config_->model.vision.spatial_merge_size} {
  has_mask_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.attention_mask);
  has_posid_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.position_ids);

  type_ = Ort::TypeToTensorType<int64_t>;  // Default to int64 for Qwen2VL
  if (has_mask_input_) {
    type_ = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);
  }

  if (has_posid_input_) {
    ONNXTensorElementDataType posid_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.position_ids);

    // Set up 3D position IDs shape: [3, batch_size, sequence_length]
    // The 3 dimensions represent temporal, height, and width for mrope
    position_ids_shape_[0] = 3;
    position_ids_shape_[1] = state_.params_->search.batch_size;
    position_ids_shape_[2] = 0;  // Will be set during first update

    position_ids_ = std::make_unique<Tensor>(model_.p_device_inputs_, posid_type);
  }
  if (has_mask_input_) {
    attention_mask_shape_[0] = state_.params_->search.batch_size;
    attention_mask_shape_[1] = 0;  // Will be set during first update
    attention_mask_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
  }
}

void Qwen2VLPositionInputs::SetGridTensors(const std::shared_ptr<Tensor>& image_grid_thw,
                                           const std::shared_ptr<Tensor>& video_grid_thw,
                                           const std::shared_ptr<Tensor>& second_per_grid_ts) {
  image_grid_thw_ = image_grid_thw;
  video_grid_thw_ = video_grid_thw;
  second_per_grid_ts_ = second_per_grid_ts;
}

void Qwen2VLPositionInputs::Add() {
  if (has_posid_input_) {
    AddPositionIDs();
  }
  if (has_mask_input_) {
    AddAttentionMask();
  }
}

void Qwen2VLPositionInputs::AddPositionIDs() {
  posid_input_index_ = state_.inputs_.size();
  state_.inputs_.push_back(position_ids_->GetOrtTensor());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.position_ids.c_str());
}

void Qwen2VLPositionInputs::AddAttentionMask() {
  mask_input_index_ = state_.inputs_.size();
  state_.inputs_.push_back(attention_mask_->GetOrtTensor());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
}

template <typename T>
void Qwen2VLPositionInputs::CreateAndInitialize3DPositionIDs(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 3> shape) {
  // Replicates the logic from HuggingFace's `get_rope_index`
  // `shape` is [3, batch_size, seq_len] (before beam expansion)
  // `next_tokens` is [batch_size, seq_len]
  int64_t batch_size = shape[1];
  int64_t seq_len = shape[2];

  auto position_ids = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  auto* position_data = position_ids->GetTensorMutableData<T>();

  // Get spans for grid_thw tensors (on CPU)
  std::span<const int64_t> image_grid_thw_span;
  if (image_grid_thw_) {
    image_grid_thw_span = std::span(image_grid_thw_->GetData<int64_t>(), image_grid_thw_->GetElementCount());
  }

  std::span<const int64_t> video_grid_thw_span;
  if (video_grid_thw_) {
    video_grid_thw_span = std::span(video_grid_thw_->GetData<int64_t>(), video_grid_thw_->GetElementCount());
  }

  std::span<const float> second_per_grid_ts_span;
  if (second_per_grid_ts_) {
    // Qwen 2.5 processor outputs float32 for this
    if (second_per_grid_ts_->GetType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("second_per_grid_ts must be float32.");
    second_per_grid_ts_span = std::span(second_per_grid_ts_->GetData<float>(), second_per_grid_ts_->GetElementCount());
  }

  auto input_ids_span = next_tokens.CpuSpan();
  int image_index = 0;
  int video_index = 0;
  rope_deltas_.clear();

  for (int64_t b = 0; b < batch_size; ++b) {
    auto input_ids = input_ids_span.subspan(b * seq_len, seq_len);

    int64_t image_nums = 0;
    int64_t video_nums = 0;

    // Count images/videos for this batch item by checking the token *after* vision_start_token_id
    for (int64_t s = 0; s < seq_len - 1; ++s) {
      if (input_ids[s] == vision_start_token_id_) {
        if (input_ids[s + 1] == image_token_id_) {
          image_nums++;
        } else if (input_ids[s + 1] == video_token_id_) {
          video_nums++;
        }
      }
    }

    int64_t st = 0;
    int64_t remain_images = image_nums;
    int64_t remain_videos = video_nums;
    T st_idx = 0;
    T max_pos_for_batch = 0;

    for (int64_t k = 0; k < image_nums + video_nums; ++k) {
      int64_t ed_image = seq_len + 1;
      int64_t ed_video = seq_len + 1;

      // Find next image_token_id (after a vision_start_token_id)
      if (remain_images > 0) {
        for (int64_t s = st; s < seq_len - 1; ++s) {
          if (input_ids[s] == vision_start_token_id_ && input_ids[s + 1] == image_token_id_) {
            ed_image = s + 1;  // Point to the image_token_id
            break;
          }
        }
      }
      // Find next video_token_id (after a vision_start_token_id)
      if (remain_videos > 0) {
        for (int64_t s = st; s < seq_len - 1; ++s) {
          if (input_ids[s] == vision_start_token_id_ && input_ids[s + 1] == video_token_id_) {
            ed_video = s + 1;  // Point to the video_token_id
            break;
          }
        }
      }

      int64_t ed;
      int64_t t, h, w;
      float second_per_grid_t = 0.0f;

      if (ed_image < ed_video) {
        // Process image
        if (image_index * 3 + 2 >= image_grid_thw_span.size())
          throw std::runtime_error("Not enough image_grid_thw data for image tokens.");
        t = image_grid_thw_span[image_index * 3 + 0];
        h = image_grid_thw_span[image_index * 3 + 1];
        w = image_grid_thw_span[image_index * 3 + 2];
        second_per_grid_t = 0.0f;  // Images have 0 time delta
        image_index++;
        remain_images--;
        ed = ed_image;
      } else {
        // Process video
        if (video_index * 3 + 2 >= video_grid_thw_span.size())
          throw std::runtime_error("Not enough video_grid_thw data for video tokens.");
        t = video_grid_thw_span[video_index * 3 + 0];
        h = video_grid_thw_span[video_index * 3 + 1];
        w = video_grid_thw_span[video_index * 3 + 2];
        if (second_per_grid_ts_span.empty() || video_index >= second_per_grid_ts_span.size()) {
          second_per_grid_t = 1.0f;  // Default from Python
        } else {
          second_per_grid_t = second_per_grid_ts_span[video_index];
        }
        video_index++;
        remain_videos--;
        ed = ed_video;
      }

      int64_t llm_grid_t = t;
      int64_t llm_grid_h = h / spatial_merge_size_;
      int64_t llm_grid_w = w / spatial_merge_size_;

      // 1. Fill Text Part
      // Text runs from `st` up to `ed-1` (which is the <|vision_start|> token)
      int64_t text_len = ed - st;
      st_idx = (k > 0 || b > 0) ? max_pos_for_batch + 1 : 0;
      T current_pos = st_idx;

      for (int64_t s = 0; s < text_len; ++s) {
        int64_t current_token_idx = st + s;
        if (input_ids[current_token_idx] == model_.config_->model.pad_token_id) {
          position_data[0 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
          position_data[1 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
          position_data[2 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
        } else {
          position_data[0 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          position_data[1 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          position_data[2 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          max_pos_for_batch = current_pos;
          current_pos++;  // Only increment position for non-pad tokens
        }
      }

      // 2. Fill Vision Part
      st_idx = max_pos_for_batch + 1;
      int64_t vision_len = llm_grid_t * llm_grid_h * llm_grid_w;
      for (int64_t s = 0; s < vision_len; ++s) {
        int64_t gt = s / (llm_grid_h * llm_grid_w);
        int64_t gh = (s / llm_grid_w) % llm_grid_h;
        int64_t gw = s % llm_grid_w;

        // Round to nearest integer for temporal position
        // Note: huggingface code use truncation/floor (time_tensor_long = time_tensor.long() when converting time coordinates.
        // This will cause slight deviation from the reference during parity comparsion.
        T t_pos = static_cast<T>(std::round(gt * second_per_grid_t * tokens_per_second_)) + st_idx;
        T h_pos = static_cast<T>(gh) + st_idx;
        T w_pos = static_cast<T>(gw) + st_idx;

        // Vision tokens are guaranteed not to be padding
        position_data[0 * batch_size * seq_len + b * seq_len + (ed + s)] = t_pos;
        position_data[1 * batch_size * seq_len + b * seq_len + (ed + s)] = h_pos;
        position_data[2 * batch_size * seq_len + b * seq_len + (ed + s)] = w_pos;
        max_pos_for_batch = std::max({max_pos_for_batch, t_pos, h_pos, w_pos});
      }
      st = ed + vision_len;  // New start is after the vision tokens
    }

    // 3. Fill Remaining Text Part
    if (st < seq_len) {
      st_idx = (max_pos_for_batch == 0 && st == 0) ? 0 : max_pos_for_batch + 1;
      int64_t text_len = seq_len - st;
      T current_pos = st_idx;
      for (int64_t s = 0; s < text_len; ++s) {
        int64_t current_token_idx = st + s;
        if (input_ids[current_token_idx] == model_.config_->model.pad_token_id) {
          position_data[0 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
          position_data[1 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
          position_data[2 * batch_size * seq_len + b * seq_len + current_token_idx] = 0;
        } else {
          position_data[0 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          position_data[1 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          position_data[2 * batch_size * seq_len + b * seq_len + current_token_idx] = current_pos;
          max_pos_for_batch = current_pos;
          current_pos++;  // Only increment position for non-pad tokens
        }
      }
    }
    rope_deltas_.push_back(max_pos_for_batch + 1 - seq_len);
  }

  // Move tensor to GPU and expand by num_beams
  position_ids_->ort_tensor_ = model_.ExpandInputs(position_ids, state_.params_->search.num_beams);
  position_ids_shape_[1] *= state_.params_->search.num_beams;
  state_.inputs_[posid_input_index_] = position_ids_->GetOrtTensor();

  // Expand rope_deltas_
  std::vector<int64_t> expanded_deltas;
  for (int64_t delta : rope_deltas_) {
    for (int b = 0; b < state_.params_->search.num_beams; ++b) {
      expanded_deltas.push_back(delta);
    }
  }
  rope_deltas_ = std::move(expanded_deltas);
}

template <typename T>
void Qwen2VLPositionInputs::CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape) {
  auto attention_mask = OrtValue::CreateTensor(model_.allocator_cpu_, shape, type_);
  auto* mask_data = attention_mask->GetTensorMutableData<T>();
  auto input_ids_span = next_tokens.CpuSpan();
  int64_t batch_size = shape[0];
  int64_t seq_len = shape[1];

  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t s = 0; s < seq_len; ++s) {
      int64_t current_token_idx = b * seq_len + s;
      mask_data[current_token_idx] = (input_ids_span[current_token_idx] == model_.config_->model.pad_token_id)
                                         ? static_cast<T>(0)
                                         : static_cast<T>(1);
    }
  }

  // Move tensor to GPU and expand by num_beams
  attention_mask_->ort_tensor_ = model_.ExpandInputs(attention_mask, state_.params_->search.num_beams);
  attention_mask_shape_[0] *= state_.params_->search.num_beams;
  state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
}

void Qwen2VLPositionInputs::Update3DPositionIDs(int base_pos) {
  // This is the generation step (decode)
  // base_pos is cache_position[0]
  auto position_ids = OrtValue::CreateTensor(model_.allocator_cpu_, position_ids_shape_, type_);
  int64_t batch_size = position_ids_shape_[1];  // This is already expanded (batch*beams)
  int64_t seq_len = position_ids_shape_[2];     // This will be 1 for generation

  if (rope_deltas_.size() != static_cast<size_t>(batch_size)) {
    throw std::runtime_error("rope_deltas size mismatch with batch_size * num_beams.");
  }

  DispatchOnType(type_, UpdatePositionIdsFunctor{position_ids.get(), base_pos, batch_size, seq_len, rope_deltas_});

  position_ids_->ort_tensor_ = model_.ExpandInputs(position_ids, 1);  // No beam expansion needed, already expanded
  state_.inputs_[posid_input_index_] = position_ids_->GetOrtTensor();
}

void Qwen2VLPositionInputs::UpdateAttentionMask() {
  auto attention_mask = OrtValue::CreateTensor(model_.allocator_cpu_, attention_mask_shape_, type_);

  DispatchOnType(type_, FillMaskFunctor{attention_mask.get(), attention_mask_shape_[0] * attention_mask_shape_[1]});

  attention_mask_->ort_tensor_ = model_.ExpandInputs(attention_mask, 1);
  state_.inputs_[mask_input_index_] = attention_mask_->GetOrtTensor();
}

void Qwen2VLPositionInputs::Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) {
  if (has_posid_input_) {
    position_ids_shape_[2] = new_length;
    if (is_first_update_) {
      DispatchOnType(type_, InitPositionIdsFunctor{this, next_tokens, position_ids_shape_});
    } else {
      Update3DPositionIDs(total_length - new_length);
    }
  }

  if (has_mask_input_) {
    if (is_first_update_) {
      attention_mask_shape_[1] = new_length;
      DispatchOnType(type_, InitAttentionMaskFunctor{this, next_tokens, attention_mask_shape_});
    } else {
      attention_mask_shape_[1] = total_length;
      UpdateAttentionMask();
    }
  }

  is_first_update_ = false;
}

void Qwen2VLPositionInputs::RewindTo(size_t index) {
  // For Qwen2-VL, we need to handle rewinding for beam search
  // This is a simplified rewind, just updating the shape.
  // A full rewind would require re-calculating rope_deltas if we rewound into the prompt.
  // For now, we assume rewind only happens during generation.
  if (has_posid_input_) {
    position_ids_shape_[2] = static_cast<int64_t>(index);
  }
  if (has_mask_input_) {
    attention_mask_shape_[1] = static_cast<int64_t>(index);
  }
}

std::unique_ptr<PositionInputs> CreatePositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths, const std::string& attention_mask_name) {
  // Check for Qwen25-VL model type which requires 3D position IDs
  if (ModelType::IsQwen25VL(state.model_.config_->model.type)) {
    return std::make_unique<Qwen2VLPositionInputs>(state.model_, state, sequence_lengths);
  }
  if (state.model_.config_->model.decoder.sliding_window.has_value() && state.model_.config_->model.decoder.sliding_window->slide_inputs) {
    return std::make_unique<WindowedPositionInputs>(state);
  } else {
    return std::make_unique<DefaultPositionInputs>(state.model_, state, sequence_lengths, attention_mask_name);
  }
}

}  // namespace Generators
