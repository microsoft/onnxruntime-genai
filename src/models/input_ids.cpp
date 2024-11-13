#include "../generators.h"
#include "model.h"
#include "input_ids.h"
#include "kernels.h"

namespace Generators {

InputIDs::InputIDs(State& state)
    : state_{state} {
  name_ = model_.config_->model.decoder.inputs.input_ids.c_str();
  shape_ = {state_.params_->BatchBeamSize(), 0};
  type_ = model_.session_info_->GetInputDataType(name_);

  if (state_.GetCapturedGraphInfo()) {
    sb_input_ids_ = state_.GetCapturedGraphInfo()->sb_input_ids_.get();

#if USE_DML
    if (model_.device_type_ == DeviceType::DML) {
      sb_input_ids_int32_ = state_.GetCapturedGraphInfo()->sb_input_ids_int32_.get();
    }
#endif
  }

  if (model_.session_info_->HasInput(model_.config_->model.decoder.inputs.current_sequence_length) &&
      model_.session_info_->HasInput(model_.config_->model.decoder.inputs.past_sequence_length)) {
    if (state_.params_->BatchBeamSize() != 1) {
      throw std::runtime_error("Batch size must be 1 for current_sequence_length and past_sequence_length inputs");
    }
    const std::array<int64_t, 1> current_sequence_length_shape{1};
    const std::array<int64_t, 2> past_sequence_length_shape{1, 1};

    if (model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.current_sequence_length) != Ort::TypeToTensorType<int32_t> ||
        model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.past_sequence_length) != Ort::TypeToTensorType<int32_t>)
      throw std::runtime_error("current_sequence_length and past_sequence_length must be int32");

    current_sequence_length_ = OrtValue::CreateTensor(model_.allocator_cpu_, current_sequence_length_shape, model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.current_sequence_length));
    *current_sequence_length_->GetTensorMutableData<int32_t>() = 0;

    past_sequence_length_ = OrtValue::CreateTensor(*model_.allocator_device_, past_sequence_length_shape, model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.past_sequence_length));
    *past_sequence_length_->GetTensorMutableData<int32_t>() = -1;
  }
}

void InputIDs::Add() {
  input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(value_.get());
  state_.input_names_.push_back(name_);

  if (current_sequence_length_ && past_sequence_length_) {
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.current_sequence_length.c_str());
    state_.inputs_.push_back(current_sequence_length_.get());
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.past_sequence_length.c_str());
    state_.inputs_.push_back(past_sequence_length_.get());
  }
}

void InputIDs::Update(DeviceSpan<int32_t>& new_tokens) {
  const auto get_unpadded_sequence_length = [](std::span<const int32_t> input_ids,
                                               int32_t pad_token_id) {
    int32_t seq_length = 0;
    for (int32_t i = 0; i < input_ids.size(); i++) {
      if (input_ids[i] == pad_token_id) {
        break;
      }
      seq_length++;
    }
    return seq_length;
  };

  if (current_sequence_length_ && past_sequence_length_) {
    if (state_.params_->BatchBeamSize() != 1) {
      throw std::runtime_error("Batch size must be 1 for current_sequence_length and past_sequence_length inputs");
    }
    auto new_sequence_length = get_unpadded_sequence_length(new_tokens.CpuSpan(), model_.config_->model.pad_token_id);
    *current_sequence_length_->GetTensorMutableData<int32_t>() += new_sequence_length;
    *past_sequence_length_->GetTensorMutableData<int32_t>() += new_sequence_length;
  }

  // Resize input_ids shape based on new_tokens
  // For beam search
  size_t sequence_length = static_cast<size_t>(new_tokens.size()) / state_.params_->BatchBeamSize();
  if (is_prompt_ && state_.params_->search.num_beams > 1)
    sequence_length = static_cast<size_t>(new_tokens.size()) / state_.params_->search.batch_size;

  if (static_cast<size_t>(shape_[1]) != sequence_length) {
    shape_[1] = sequence_length;
    if (!sb_input_ids_) {
      value_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);

#if USE_DML
      if (model_.device_type_ == DeviceType::DML) {
        value_int32_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
      }
#endif
    } else {
      value_ = sb_input_ids_->CreateTensorOnStaticBuffer(shape_, type_);

#if USE_DML
      if (model_.device_type_ == DeviceType::DML) {
        value_int32_ = sb_input_ids_int32_->CreateTensorOnStaticBuffer(shape_, Ort::TypeToTensorType<int32_t>);
      }
#endif
    }

    state_.inputs_[input_index_] = value_.get();
  }

  // Update input_ids with next tokens, converting from 32-bit to 64-bit
  if (type_ == Ort::TypeToTensorType<int64_t>) {
    switch (model_.device_type_) {
      case DeviceType::CUDA: {
#if USE_CUDA
        auto* data = value_->GetTensorMutableData<int64_t>();
        auto next_tokens = new_tokens.Span();
        // For beam search
        if (is_prompt_ && state_.params_->search.num_beams > 1)
          cuda::LaunchExpandAndInt32ToInt64(next_tokens.data(), data, state_.params_->search.num_beams, state_.params_->search.batch_size, static_cast<int>(sequence_length), model_.cuda_stream_);
        else
          cuda::LaunchInt32ToInt64(next_tokens.data(), data, static_cast<int>(next_tokens.size()), model_.cuda_stream_);
#endif
      } break;

      case DeviceType::DML: {
#if USE_DML
        ComPtr<ID3D12Resource> source_resource;
        Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, value_int32_->GetTensorMutableRawData(), &source_resource));

        auto source = std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(new_tokens.CpuSpan().data()),
            new_tokens.CpuSpan().size_bytes());

        model_.GetDmlUploadHeap()->BeginUploadToGpu(
            source_resource.Get(),
            0,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            source);

        DmlHelpers::DmlCastInputToOutput(
            model_.GetDmlExecutionContext(),
            *model_.allocator_device_,
            *value_int32_,
            value_,
            model_.GetDmlDevice(),
            model_.GetOrtDmlApi(),
            input_ids_cast_command_list_state_);
#endif
      } break;
      default: {
        // CPU, WEBGPU
        auto* data = value_->GetTensorMutableData<int64_t>();
        auto next_tokens = new_tokens.Span();
        for (int b = 0; b < shape_[0]; b++) {
          for (int i = 0; i < shape_[1]; i++) {
            // For beam search
            int32_t next_token;
            if (is_prompt_ && state_.params_->search.num_beams > 1)
              next_token = next_tokens[(b / state_.params_->search.num_beams) * shape_[1] + i];
            else
              next_token = next_tokens[b * shape_[1] + i];
            data[b * shape_[1] + i] = next_token;
          }
        }
      }
    }
  } else {
    auto* data = value_->GetTensorMutableData<int32_t>();
#if USE_CUDA
    if (model_.device_type_ == DeviceType::CUDA) {
      if (is_prompt_ && state_.params_->search.num_beams > 1) {
        cuda::LaunchExpand(new_tokens.Span().data(), data, state_.params_->search.num_beams, state_.params_->search.batch_size, static_cast<int>(sequence_length), model_.cuda_stream_);
      } else {
        cudaMemcpyAsync(data, new_tokens.Span().data(), shape_[0] * shape_[1] * sizeof(int32_t), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
      }
    } else
#endif
    {
      // For beam search
      if (is_prompt_ && state_.params_->search.num_beams > 1) {
        for (int b = 0; b < shape_[0]; b++) {
          int in_offset = (b / state_.params_->search.num_beams) * static_cast<int>(shape_[1]);
          int out_offset = b * static_cast<int>(shape_[1]);
          memcpy(data + out_offset, new_tokens.Span().data() + in_offset, shape_[1] * sizeof(int32_t));
        }
      } else {
        memcpy(data, new_tokens.Span().data(), shape_[0] * shape_[1] * sizeof(int32_t));
      }
    }
  }

  is_prompt_ = false;
}

}  // namespace Generators
