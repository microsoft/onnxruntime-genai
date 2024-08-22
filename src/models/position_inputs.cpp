#include "../generators.h"
#include "model.h"
#include "position_inputs.h"
#include "kernels.h"

#if USE_DML
#include "../dml/dml_update_mask_kernel.h"
#endif

namespace Generators {

PositionInputs::PositionInputs(const Model& model, State& state, RoamingArray<int32_t>& sequence_lengths_unk)
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

  std::array<int64_t, 2> shape{state_.params_->batch_size, state_.params_->sequence_length};  // Only batch_size initially, as we haven't expanded over the beams yet
  position_ids_ = OrtValue::CreateTensor(model.allocator_cpu_, shape, type_);
  position_ids_next_ = OrtValue::CreateTensor(model.allocator_cpu_, std::array<int64_t, 2>{shape[0], 1}, type_);
  attention_mask_ = OrtValue::CreateTensor(model.allocator_cpu_, shape, type_);

  initial_sequence_lengths_.resize(state_.params_->BatchBeamSize());

  if (type_ == Ort::TypeToTensorType<int32_t>)
    InitializeTensors<int32_t>(shape, sequence_lengths_unk);
  else
    InitializeTensors<int64_t>(shape, sequence_lengths_unk);

  position_ids_ = model_.ExpandInputs(position_ids_, state_.params_->search.num_beams);
  position_ids_next_ = model_.ExpandInputs(position_ids_next_, state_.params_->search.num_beams);
  attention_mask_ = model_.ExpandInputs(attention_mask_, state_.params_->search.num_beams);
  shape[0] *= state_.params_->search.num_beams;
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

void PositionInputs::Add() {
  if (has_posid_input_) {
    AddPositionIDs();
  }
  if (has_mask_input_) {
    AddAttentionMask();
  }
}

void PositionInputs::Update(int current_length) {
  if (has_posid_input_) {
    UpdatePositionIDs(current_length);
  }
  if (has_mask_input_) {
    UpdateAttentionMask(current_length);
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

void PositionInputs::UpdatePositionIDs(int current_length) {
  // Reallocate position_ids for the 2nd and onward shape
  if (is_first_posid_update_) {
    position_ids_shape_[1] = 1;
    if (!sb_position_ids_) {
      position_ids_ = std::move(position_ids_next_);
    } else {
#if USE_CUDA
      position_ids_ = sb_position_ids_->CreateTensorOnStaticBuffer(position_ids_shape_, type_);
      assert(model_.device_type_ == DeviceType::CUDA);
      if (type_ == Ort::TypeToTensorType<int32_t>) {
        cudaMemcpyAsync(position_ids_->GetTensorMutableRawData(),
                        position_ids_next_->GetTensorData<int32_t>(),
                        sizeof(int32_t) * position_ids_shape_[0],
                        cudaMemcpyDeviceToDevice,
                        model_.cuda_stream_);
      } else {
        cudaMemcpyAsync(position_ids_->GetTensorMutableRawData(),
                        position_ids_next_->GetTensorData<int64_t>(),
                        sizeof(int64_t) * position_ids_shape_[0],
                        cudaMemcpyDeviceToDevice,
                        model_.cuda_stream_);
      }
#elif USE_DML
      position_ids_ = sb_position_ids_->CreateTensorOnStaticBuffer(position_ids_shape_, type_);
      assert(model_.device_type_ == DeviceType::DML);

      ComPtr<ID3D12Resource> target_resource;
      Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, position_ids_->GetTensorMutableRawData(), &target_resource));

      if (type_ == Ort::TypeToTensorType<int32_t>) {
        auto source = std::span(position_ids_next_->GetTensorData<const uint8_t>(), sizeof(int32_t) * position_ids_shape_[0]);

        model_.GetDmlUploadHeap()->BeginUploadToGpu(
            target_resource.Get(),
            0,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            source);
      } else {
        auto source = std::span(position_ids_next_->GetTensorData<const uint8_t>(), sizeof(int64_t) * position_ids_shape_[0]);

        model_.GetDmlUploadHeap()->BeginUploadToGpu(
            target_resource.Get(),
            0,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            source);
      }
#endif
    }
    is_first_posid_update_ = false;
    state_.inputs_[posid_input_index_] = position_ids_.get();
  } else {  // Just incrementing existing position IDs
    switch (model_.device_type_) {
#if USE_DML
      case DeviceType::DML: {
        ComPtr<ID3D12Resource> target_resource;
        Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, position_ids_->GetTensorMutableRawData(), &target_resource));

        // Lazily create the kernel only the first time it's needed
        if (!dml_update_position_ids_kernel_) {
          dml_update_position_ids_kernel_ = DmlIncrementValuesKernel(
              model_.GetD3D12Device(),
              model_.GetDmlExecutionContext(),
              static_cast<uint32_t>(position_ids_shape_[0]),
              type_,
              target_resource.Get());
        }

        // Execute the cached command list
        ComPtr<ID3D12Fence> fence;
        uint64_t completion_value;
        model_.GetDmlExecutionContext()->ExecuteCommandList(dml_update_position_ids_kernel_->GetCommandList(), &fence, &completion_value);
      } break;
#endif
      case DeviceType::CPU: {
        if (type_ == Ort::TypeToTensorType<int32_t>)
          UpdatePositionIDsImpl<int32_t>();
        else
          UpdatePositionIDsImpl<int64_t>();
        break;
      }
#if USE_CUDA
      case DeviceType::CUDA:
        if (type_ == Ort::TypeToTensorType<int32_t>)
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

void PositionInputs::UpdateAttentionMask(int current_length) {
  // Update attention mask
  if (sb_attention_mask_) {
#if USE_CUDA
    attention_mask_shape_[1] = state_.params_->search.max_length;
    attention_mask_next_ = sb_attention_mask_->CreateTensorOnStaticBuffer(attention_mask_shape_, type_);
    if (is_first_mask_update_) {
      if (type_ == Ort::TypeToTensorType<int32_t>) {
        cudaMemsetAsync(attention_mask_next_->GetTensorMutableRawData(),
                        0,
                        sizeof(int32_t) * attention_mask_shape_[0] * attention_mask_shape_[1],
                        model_.cuda_stream_);
      } else {
        cudaMemsetAsync(attention_mask_next_->GetTensorMutableRawData(),
                        0,
                        sizeof(int64_t) * attention_mask_shape_[0] * attention_mask_shape_[1],
                        model_.cuda_stream_);
      }
    }
#elif USE_DML
    attention_mask_shape_[1] = state_.params_->search.max_length;
    attention_mask_ = sb_attention_mask_->CreateTensorOnStaticBuffer(attention_mask_shape_, type_);
    attention_mask_next_ = sb_attention_mask_next_->CreateTensorOnStaticBuffer(attention_mask_shape_, type_);
#endif
  } else {
    assert(attention_mask_shape_[1] == current_length - 1);  // We should always be growing by 1
    attention_mask_shape_[1] = current_length;

#if USE_DML
    if (model_.device_type_ == DeviceType::DML) {
      attention_mask_ = OrtValue::CreateTensor(*model_.allocator_device_, attention_mask_shape_, type_);
    }
#endif

    attention_mask_next_ = OrtValue::CreateTensor(*model_.allocator_device_, attention_mask_shape_, type_);
  }

  switch (model_.device_type_) {
#if USE_DML
    case DeviceType::DML: {
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
            current_length,
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
      break;
    }
#endif
    case DeviceType::CPU: {
      if (type_ == Ort::TypeToTensorType<int32_t>)
        UpdateAttentionMaskImpl(attention_mask_next_->GetTensorMutableData<int32_t>(),
                                attention_mask_->GetTensorData<int32_t>(),
                                current_length);
      else
        UpdateAttentionMaskImpl(attention_mask_next_->GetTensorMutableData<int64_t>(),
                                attention_mask_->GetTensorData<int64_t>(),
                                current_length);
      break;
    }
#if USE_CUDA
    case DeviceType::CUDA: {
      int max_seq_len = sb_attention_mask_ ? state_.params_->search.max_length : current_length;
      bool update_only = sb_attention_mask_ && !is_first_mask_update_;
      if (type_ == Ort::TypeToTensorType<int32_t>) {
        cuda::Launch_UpdateAttentionMask(attention_mask_next_->GetTensorMutableData<int32_t>(),
                                         attention_mask_->GetTensorData<int32_t>(),
                                         static_cast<int>(attention_mask_shape_[0]),
                                         current_length,
                                         max_seq_len,
                                         update_only,
                                         model_.cuda_stream_);
      } else {
        cuda::Launch_UpdateAttentionMask(attention_mask_next_->GetTensorMutableData<int64_t>(),
                                         attention_mask_->GetTensorData<int64_t>(),
                                         static_cast<int>(attention_mask_shape_[0]),
                                         current_length,
                                         max_seq_len,
                                         update_only,
                                         model_.cuda_stream_);
      }
      break;
    }
#endif
    default:
      throw std::runtime_error("PositionIDs::Update - Unsupported device type");
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
void PositionInputs::InitializeTensors(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths) {
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
      initial_sequence_lengths_[i * state_.params_->search.num_beams + k] = static_cast<int32_t>(abs_position);
    }
  }
}

template <typename T>
void PositionInputs::UpdatePositionIDsImpl() {
  // Increment position IDs
  auto* data = position_ids_->GetTensorMutableData<T>();
  for (int i = 0; i < position_ids_shape_[0]; i++) {
    data[i]++;
  }
};

template <typename T>
void PositionInputs::UpdateAttentionMaskImpl(T* data, const T* old_data, int current_length) {
  for (int i = 0; i < attention_mask_shape_[0]; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      data[i * current_length + j] = old_data[i * (current_length - 1) + j];
    }
    data[i * current_length + current_length - 1] = 1;
  }
};

}  // namespace Generators
