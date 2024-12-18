#include "../generators.h"
#include "model.h"
#include "position_inputs.h"
#include "kernels.h"

#if USE_DML
#include "../dml/dml_update_mask_kernel.h"
#endif

namespace Generators {

DefaultPositionInputs::DefaultPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk)
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

#if USE_DML
      if (model_.device_type_ == DeviceType::DML) {
        sb_attention_mask_next_ = state_.GetCapturedGraphInfo()->sb_attention_mask_next_.get();
      }
#endif
    }
  }
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
    is_first_mask_update_ = true;
    // Rewind the mask input to a previous state
  } else if (has_mask_input_) {
    if (attention_mask_shape_[0] == 1) {
#if USE_CUDA
      RewindMask(index);
#endif
    } else
      throw std::runtime_error("DefaultPositionInputs::RewindTo - Unsupported batch size");
  }
}

void DefaultPositionInputs::AddAttentionMask() {
  mask_input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(attention_mask_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
}

void DefaultPositionInputs::AddPositionIDs() {
  posid_input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(position_ids_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.position_ids.c_str());
}

#if USE_CUDA || USE_DML
void DefaultPositionInputs::CopyNextPositionIDsToCurrent() {
#if USE_CUDA
  assert(model_.device_type_ == DeviceType::CUDA);
  cudaMemcpyAsync(position_ids_->GetTensorMutableRawData(),
                  position_ids_next_->GetTensorMutableRawData(),
                  (type_ == Ort::TypeToTensorType<int32_t> ? sizeof(int32_t) : sizeof(int64_t)) * position_ids_shape_[0],
                  cudaMemcpyDeviceToDevice,
                  model_.cuda_stream_);
#elif USE_DML
  assert(model_.device_type_ == DeviceType::DML);
  ComPtr<ID3D12Resource> target_resource;
  Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, position_ids_->GetTensorMutableRawData(), &target_resource));
  auto source = std::span(position_ids_next_->GetTensorData<const uint8_t>(), (type_ == Ort::TypeToTensorType<int32_t> ? sizeof(int32_t) : sizeof(int64_t)) * position_ids_shape_[0]);
  model_.GetDmlUploadHeap()->BeginUploadToGpu(
      target_resource.Get(),
      0,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      source);
#endif
}
#endif

void DefaultPositionInputs::CreateNextPositionIDsTensor() {
  if (!sb_position_ids_) {
    if (position_ids_shape_[1] == 1 && position_ids_next_) {
      position_ids_ = std::move(position_ids_next_);
      position_ids_next_ = nullptr;
    } else {
      position_ids_ = OrtValue::CreateTensor(*model_.allocator_device_, position_ids_shape_, type_);
    }
  } else {
#if USE_CUDA || USE_DML
    position_ids_ = sb_position_ids_->CreateTensorOnStaticBuffer(position_ids_shape_, type_);
    if (position_ids_shape_[1] == 1) {
      CopyNextPositionIDsToCurrent();
    }
#endif
  }
}

void DefaultPositionInputs::UpdatePositionIDs(int total_length, int new_kv_length) {
  if (position_ids_shape_[0] != 1 && !(total_length == 0 || new_kv_length == 1))
    throw std::runtime_error("DefaultPositionInputs::UpdatePositionIDs - batch_size must be 1 for continuous decoding.");

  // Reallocate position_ids when new_kv_length changes
  if (position_ids_shape_[1] != new_kv_length) {
    position_ids_shape_[1] = new_kv_length;
    CreateNextPositionIDsTensor();
    state_.inputs_[posid_input_index_] = position_ids_.get();
  }

  switch (model_.device_type_) {
    case DeviceType::WEBGPU:
    case DeviceType::CPU: {
      type_ == Ort::TypeToTensorType<int32_t> ? UpdatePositionIDsImpl<int32_t>(total_length, new_kv_length)
                                              : UpdatePositionIDsImpl<int64_t>(total_length, new_kv_length);
      break;
    }
#if USE_CUDA
    case DeviceType::CUDA: {
      if (type_ == Ort::TypeToTensorType<int32_t>)
        cuda::Launch_UpdatePositionIds(position_ids_->GetTensorMutableData<int32_t>(), static_cast<int>(position_ids_shape_[0]), total_length, new_kv_length, model_.cuda_stream_);
      else
        cuda::Launch_UpdatePositionIds(position_ids_->GetTensorMutableData<int64_t>(), static_cast<int>(position_ids_shape_[0]), total_length, new_kv_length, model_.cuda_stream_);
      break;
    }
#elif USE_DML
    case DeviceType::DML: {
      UpdatePositionIDsImplDML();
      break;
    }
#endif
    default:
      throw std::runtime_error("PositionIDs::Update - Unsupported device type");
  }
}

void DefaultPositionInputs::CreateNextAttentionMaskTensor(int total_length) {
  if (!sb_attention_mask_) {
    attention_mask_shape_[1] = total_length;
    attention_mask_next_ = OrtValue::CreateTensor(*model_.allocator_device_, attention_mask_shape_, type_);
#if USE_DML
    if (model_.device_type_ == DeviceType::DML)
      attention_mask_ = OrtValue::CreateTensor(*model_.allocator_device_, attention_mask_shape_, type_);
#endif
  } else {
#if USE_CUDA
    attention_mask_shape_[1] = state_.params_->search.max_length;
    attention_mask_next_ = sb_attention_mask_->CreateTensorOnStaticBuffer(attention_mask_shape_, type_);
    if (is_first_mask_update_) {
      cudaMemsetAsync(attention_mask_next_->GetTensorMutableRawData(),
                      0,
                      (type_ == Ort::TypeToTensorType<int32_t> ? sizeof(int32_t) : sizeof(int64_t)) * attention_mask_shape_[0] * attention_mask_shape_[1],
                      model_.cuda_stream_);
    }
#elif USE_DML
    attention_mask_shape_[1] = state_.params_->search.max_length;
    attention_mask_ = sb_attention_mask_->CreateTensorOnStaticBuffer(attention_mask_shape_, type_);
    attention_mask_next_ = sb_attention_mask_next_->CreateTensorOnStaticBuffer(attention_mask_shape_, type_);
#endif
  }
}

void DefaultPositionInputs::UpdateAttentionMask(int total_length, int new_kv_length) {
  if (position_ids_shape_[0] != 1 && !(total_length == 0 || new_kv_length == 1))
    throw std::runtime_error("DefaultPositionInputs::UpdatePositionIDs - batch_size must be 1 for continuous decoding.");
  if (DeviceType::DML == model_.device_type_ && !(total_length == 0 || new_kv_length == 1))
    throw std::runtime_error("DefaultPositionInputs::UpdatePositionIDs - DML does not support continuous decoding.");

  CreateNextAttentionMaskTensor(total_length);
  state_.inputs_[mask_input_index_] = attention_mask_.get();

  switch (model_.device_type_) {
    case DeviceType::WEBGPU:
    case DeviceType::CPU:
    case DeviceType::QNN: {
      type_ == Ort::TypeToTensorType<int32_t> ? UpdateAttentionMaskImpl<int32_t>(total_length)
                                              : UpdateAttentionMaskImpl<int64_t>(total_length);
      break;
    }
#if USE_CUDA
    case DeviceType::CUDA: {
      int max_length = sb_attention_mask_ ? state_.params_->search.max_length : total_length;
      bool update_only = sb_attention_mask_ && !is_first_mask_update_;
      if (type_ == Ort::TypeToTensorType<int32_t>) {
        cuda::Launch_UpdateAttentionMask(attention_mask_next_->GetTensorMutableData<int32_t>(),
                                         attention_mask_->GetTensorData<int32_t>(),
                                         static_cast<int>(attention_mask_shape_[0]),
                                         new_kv_length,
                                         total_length,
                                         max_length,
                                         update_only,
                                         model_.cuda_stream_);
      } else {
        cuda::Launch_UpdateAttentionMask(attention_mask_next_->GetTensorMutableData<int64_t>(),
                                         attention_mask_->GetTensorData<int64_t>(),
                                         static_cast<int>(attention_mask_shape_[0]),
                                         new_kv_length,
                                         total_length,
                                         max_length,
                                         update_only,
                                         model_.cuda_stream_);
      }
      break;
    }
#elif USE_DML
    case DeviceType::DML: {
      UpdateAttentionMaskImplDML(total_length);
      break;
    }
#endif
    default:
      throw std::runtime_error("DefaultPositionInputs::Update - Unsupported device type");
  }
#if USE_DML
  if (model_.device_type_ != DeviceType::DML) {
    attention_mask_ = std::move(attention_mask_next_);
  }
#else
  attention_mask_ = std::move(attention_mask_next_);
#endif
  state_.inputs_[mask_input_index_] = attention_mask_.get();
  is_first_mask_update_ = false;
}

template <typename T>
void DefaultPositionInputs::CreateAndInitializePositionIDs(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape) {
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
void DefaultPositionInputs::CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape) {
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
void DefaultPositionInputs::InitializeSequenceLengths(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths_unk) {
  for (int i = 0; i < shape[0] * state_.params_->search.num_beams; i++) {
    sequence_lengths_unk[i] = 0;
  }
}

template <typename T>
void DefaultPositionInputs::UpdatePositionIDsImpl(int total_length, int new_kv_length) {
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

#if USE_DML
void DefaultPositionInputs::UpdatePositionIDsImplDML() {
  ComPtr<ID3D12Resource> target_resource;
  Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, position_ids_->GetTensorMutableRawData(), &target_resource));

  dml_update_position_ids_kernel_ = DmlIncrementValuesKernel(
      model_.GetD3D12Device(),
      model_.GetDmlExecutionContext(),
      static_cast<uint32_t>(position_ids_shape_[0]),
      type_,
      target_resource.Get());

  // Execute the cached command list
  ComPtr<ID3D12Fence> fence;
  uint64_t completion_value;
  model_.GetDmlExecutionContext()->ExecuteCommandList(dml_update_position_ids_kernel_->GetCommandList(), &fence, &completion_value);
}
#endif

template <typename T>
void DefaultPositionInputs::UpdateAttentionMaskImpl(int total_length) {
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

#if USE_DML
void DefaultPositionInputs::UpdateAttentionMaskImplDML(int total_length) {
  ComPtr<ID3D12Resource> attention_mask_resource;
  Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, attention_mask_->GetTensorMutableRawData(), &attention_mask_resource));
  ComPtr<ID3D12Resource> attention_mask_next_resource;
  Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, attention_mask_next_->GetTensorMutableRawData(), &attention_mask_next_resource));
  if (is_first_mask_update_) {
    dml_update_mask_kernel_ = DmlUpdateMaskKernel(
        model_.GetD3D12Device(),
        model_.GetDmlExecutionContext(),
        static_cast<uint32_t>(attention_mask_shape_[0]),
        static_cast<uint32_t>(attention_mask_shape_[1]),
        type_,
        total_length,
        attention_mask_resource.Get(),
        attention_mask_next_resource.Get());
    is_second_mask_update_ = true;
  } else if (is_second_mask_update_) {
    dml_update_mask_kernel_ = DmlUpdateMaskKernel(
        model_.GetD3D12Device(),
        model_.GetDmlExecutionContext(),
        static_cast<uint32_t>(attention_mask_shape_[0]),
        static_cast<uint32_t>(attention_mask_shape_[1]),
        type_,
        1,
        attention_mask_resource.Get(),
        attention_mask_next_resource.Get());
    is_second_mask_update_ = false;
  }
  ComPtr<ID3D12Fence> fence;
  uint64_t completion_value;
  model_.GetDmlExecutionContext()->ExecuteCommandList(dml_update_mask_kernel_->GetCommandList(), &fence, &completion_value);
}
#endif

#if USE_CUDA
void DefaultPositionInputs::RewindMask(size_t index) {
  if (sb_attention_mask_ && !is_first_mask_update_) {
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
  }
}
#endif

WindowedPositionInputs::WindowedPositionInputs(State& state)
    : state_{state} {
  has_posid_input_ = model_.session_info_->HasInput(model_.config_->model.decoder.inputs.position_ids);
  has_mask_input_ = model_.session_info_->HasInput(model_.config_->model.decoder.inputs.attention_mask);

  if (has_posid_input_ || has_mask_input_) {
    if (!model_.config_->model.decoder.sliding_window.has_value()) {
      throw std::runtime_error("Sliding a window over position_ids and attention_mask requires sliding_window to be set in the genai_config.json.");
    }
    window_size_ = model_.config_->model.decoder.sliding_window->window_size;
  }

  if (has_posid_input_) {
    position_ids_type_ = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.position_ids);
    if (position_ids_type_ != Ort::TypeToTensorType<int32_t>)
      throw std::runtime_error("WindowedPositionInputs only supports int32_t position_ids");

    position_ids_shape_ = {1, model_.config_->model.decoder.sliding_window->window_size};
  }

  if (has_mask_input_) {
    attention_mask_type_ = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);
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
