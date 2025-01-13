#include "../generators.h"
#include "model.h"
#include "input_ids.h"
#include "kernels.h"

namespace Generators {

DefaultInputIDs::DefaultInputIDs(State& state)
    : state_{state} {
  name_ = model_.config_->model.decoder.inputs.input_ids.c_str();
  shape_ = {state_.params_->search.batch_size, 0};
  type_ = model_.session_info_->GetInputDataType(name_);

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

    past_sequence_length_ = OrtValue::CreateTensor(model_.allocator_cpu_, past_sequence_length_shape, model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.past_sequence_length));
    *past_sequence_length_->GetTensorMutableData<int32_t>() = -1;
  }
}

void DefaultInputIDs::Add() {
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

void DefaultInputIDs::Update(DeviceSpan<int32_t>& new_tokens) {
  // There are three scopes involved when the Update function is called:
  // 1. A new Generator state has been just created. This is a prompt stage, and value_ is a nullptr.
  //    i.e. this is the very first time ever that Update is being called for this Generator.
  // 2. We move to the token generation stage. value_ has already been previously created in the prompt stage.
  //    Update is called on every new token generated.
  // 3. We move from the token generation stage back to the prompt stage (e.g. in continous decoding). value_ is already created.

  // For instances where the value_ is not created, we need handle graph capture correctly.
  // For subsequent prompt stages, the limiting factor is that the subsequent prompts can not
  // be larger than the first prompt (when graph capture is enabled).
  if (!value_) {
    shape_[1] = static_cast<int64_t>(new_tokens.size()) / shape_[0];

    // If 64-bit, convert from 32-bit to 64-bit
    auto input_ids = new_tokens.CopyDeviceToCpu();
    if (type_ == Ort::TypeToTensorType<int64_t>) {
      value_ = OrtValue::CreateTensor(model_.allocator_cpu_, shape_, type_);
      auto* p_data = value_->GetTensorMutableData<int64_t>();
      for (auto v : input_ids) {
        *p_data++ = v;
      }
    } else {
      if (type_ != Ort::TypeToTensorType<int32_t>)
        throw std::runtime_error("InputIDs must be int64 or int32");
      value_ = OrtValue::CreateTensor<int32_t>(model_.allocator_cpu_.GetInfo(), input_ids, shape_);
    }

    value_ = model_.ExpandInputs(value_, state_.params_->search.num_beams);
    shape_[0] *= state_.params_->search.num_beams;

    if (state_.GetCapturedGraphInfo()) {
      sb_input_ids_ = state_.GetCapturedGraphInfo()->sb_input_ids_.get();

#if USE_DML
      if (model_.device_type_ == DeviceType::DML) {
        sb_input_ids_int32_ = state_.GetCapturedGraphInfo()->sb_input_ids_int32_.get();
      }
#endif
    }

    is_prompt_ = false;
    state_.inputs_[input_index_] = value_.get();
    return;
  }

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

WindowedInputIDs::WindowedInputIDs(State& state) : state_{state} {
  name_ = model_.config_->model.decoder.inputs.input_ids.c_str();

  if (!model_.config_->model.decoder.sliding_window.has_value()) {
    throw std::runtime_error("Sliding a window over input_ids requires sliding_window to be set in the genai_config.json.");
  }

  if (state_.params_->BatchBeamSize() != 1) {
    throw std::runtime_error("Batch beam size must be 1 for sliding a window over input_ids.");
  }

  window_size_ = model_.config_->model.decoder.sliding_window->window_size;
  shape_ = {1, model_.config_->model.decoder.sliding_window->window_size};
  type_ = model_.session_info_->GetInputDataType(name_);

  if (type_ != Ort::TypeToTensorType<int32_t>) {
    throw std::runtime_error("WindowedInputIDs only supports int32_t input_ids.");
  }
}

void WindowedInputIDs::Add() {
  input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(value_.get());
  state_.input_names_.push_back(name_);
}

void WindowedInputIDs::Update(DeviceSpan<int32_t>& new_tokens) {
  if (window_index_ == 0) {
    num_windows_ = (new_tokens.size() + window_size_ - 1) / window_size_;

    value_ = OrtValue::CreateTensor(model_.allocator_cpu_, shape_, type_);

    // new_tokens will always be padded so that it's size is a multiple of window_size_
    // new_tokens -> [0, a, b, c, d, e]
    // window_size = 3, num_windows = 2, pad_token = 0
    // window_index = 0, value_ -> [0, a, b]
    std::copy_n(new_tokens.Span().begin(), window_size_, value_->GetTensorMutableData<int32_t>());
  } else if (window_index_ < num_windows_) {
    // new_tokens -> [a, b, c, d, e]
    // window_size = 3, num_windows = 2
    // window_index = 1, value_ -> [c, d, e]
    std::copy_n(new_tokens.Span().begin() + window_index_ * window_size_, window_size_, value_->GetTensorMutableData<int32_t>());
  } else {
    // All prompt token chunks have been processed. Now we process the tokens generated by the model.
    // new_tokens -> [f]
    assert(new_tokens.size() == 1);
    if (shape_[1] != 1) {
      shape_[1] = 1;
      value_ = OrtValue::CreateTensor(model_.allocator_cpu_, shape_, type_);
    }

    value_->GetTensorMutableData<int32_t>()[0] = new_tokens.Span()[0];
  }

  state_.inputs_[input_index_] = value_.get();
  window_index_++;
}

std::unique_ptr<InputIDs> CreateInputIDs(State& state) {
  if (state.model_.config_->model.decoder.sliding_window.has_value()) {
    return std::make_unique<WindowedInputIDs>(state);
  } else {
    return std::make_unique<DefaultInputIDs>(state);
  }
}

}  // namespace Generators
