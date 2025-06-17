#include "../generators.h"
#include "model.h"
#include "input_ids.h"

namespace Generators {

DefaultInputIDs::DefaultInputIDs(State& state)
    : state_{state} {
  name_ = model_.config_->model.decoder.inputs.input_ids.c_str();
  shape_ = {state_.params_->BatchBeamSize(), 0};
  type_ = model_.session_info_.GetInputDataType(name_);

  if (model_.session_info_.HasInput(model_.config_->model.decoder.inputs.current_sequence_length) &&
      model_.session_info_.HasInput(model_.config_->model.decoder.inputs.past_sequence_length)) {
    if (state_.params_->BatchBeamSize() != 1) {
      throw std::runtime_error("Batch size must be 1 for current_sequence_length and past_sequence_length inputs");
    }
    const std::array<int64_t, 1> current_sequence_length_shape{1};
    const std::array<int64_t, 2> past_sequence_length_shape{1, 1};

    if (model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.current_sequence_length) != Ort::TypeToTensorType<int32_t> ||
        model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.past_sequence_length) != Ort::TypeToTensorType<int32_t>)
      throw std::runtime_error("current_sequence_length and past_sequence_length must be int32");

    current_sequence_length_ = OrtValue::CreateTensor(model_.allocator_cpu_, current_sequence_length_shape, model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.current_sequence_length));
    *current_sequence_length_->GetTensorMutableData<int32_t>() = 0;

    past_sequence_length_ = OrtValue::CreateTensor(model_.allocator_cpu_, past_sequence_length_shape, model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.past_sequence_length));
    *past_sequence_length_->GetTensorMutableData<int32_t>() = -1;
  }

  value_ = std::make_unique<Tensor>(model_.p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  cast_value_ = std::make_unique<Tensor>(model_.p_device_inputs_, Ort::TypeToTensorType<int64_t>);
}

void DefaultInputIDs::Add() {
  input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(value_->GetOrtTensor());
  state_.input_names_.push_back(name_);

  if (current_sequence_length_ && past_sequence_length_) {
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.current_sequence_length.c_str());
    state_.inputs_.push_back(current_sequence_length_.get());
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.past_sequence_length.c_str());
    state_.inputs_.push_back(past_sequence_length_.get());
  }
}

void DefaultInputIDs::Update(DeviceSpan<int32_t> new_tokens) {
  auto new_tokens_cpu = new_tokens.CopyDeviceToCpu();

  const auto get_unpadded_sequence_length = [](std::span<const int32_t> input_ids, int32_t pad_token_id) {
    for (int32_t i = 0; i < input_ids.size(); i++) {
      if (input_ids[i] == pad_token_id)
        return i;
    }
    return static_cast<int32_t>(input_ids.size());
  };

  if (current_sequence_length_ && past_sequence_length_) {
    if (state_.params_->BatchBeamSize() != 1) {
      throw std::runtime_error("Batch size must be 1 for current_sequence_length and past_sequence_length inputs");
    }
    auto new_sequence_length = get_unpadded_sequence_length(new_tokens_cpu, model_.config_->model.pad_token_id);
    *current_sequence_length_->GetTensorMutableData<int32_t>() += new_sequence_length;
    *past_sequence_length_->GetTensorMutableData<int32_t>() += new_sequence_length;
  }

  // For beam search, resize input_ids shape based on new_tokens
  size_t sequence_length = static_cast<size_t>(new_tokens.size()) / state_.params_->BatchBeamSize();
  if (is_prompt_ && state_.params_->search.num_beams > 1)
    sequence_length = static_cast<size_t>(new_tokens.size()) / state_.params_->search.batch_size;

  if (static_cast<size_t>(shape_[1]) != sequence_length) {
    shape_[1] = sequence_length;
    value_->CreateTensor(shape_, state_.params_->use_graph_capture && shape_[1] == 1);
    state_.inputs_[input_index_] = value_->GetOrtTensor();
  }

  // Update input_ids with next tokens
  auto data_span = value_->GetDeviceSpan<int32_t>();

  // For beam search
  if (is_prompt_ && state_.params_->search.num_beams > 1) {
    int row_size = static_cast<int>(shape_[1]);
    for (int b = 0; b < shape_[0]; b++) {
      int in_offset = (b / state_.params_->search.num_beams) * row_size;
      int out_offset = b * row_size;
      data_span.subspan(out_offset, row_size).CopyFrom(new_tokens.subspan(in_offset, row_size));
    }
  } else {
    data_span.CopyFrom(new_tokens);
  }

  if (type_ == Ort::TypeToTensorType<int64_t>) {
    if (!cast_value_->ort_tensor_ || static_cast<size_t>(cast_value_->GetShape()[1]) != sequence_length)
      cast_value_->CreateTensor(shape_, state_.params_->use_graph_capture && shape_[1] == 1);
    Cast(*value_->GetOrtTensor(), cast_value_->ort_tensor_, *model_.p_device_inputs_, type_);
    state_.inputs_[input_index_] = cast_value_->GetOrtTensor();
  }

  is_prompt_ = false;
}

WindowedInputIDs::WindowedInputIDs(State& state) : state_{state} {
  if (model_.p_device_inputs_->GetType() != DeviceType::QNN &&
      model_.p_device_inputs_->GetType() != DeviceType::CPU) {
    throw std::runtime_error("Sliding a window over input_ids only works with either the QNN or the CPU provider.");
  }

  name_ = model_.config_->model.decoder.inputs.input_ids.c_str();

  if (!model_.config_->model.decoder.sliding_window.has_value()) {
    throw std::runtime_error("Sliding a window over input_ids requires sliding_window to be set in the genai_config.json.");
  }

  if (state_.params_->BatchBeamSize() != 1) {
    throw std::runtime_error("Batch beam size must be 1 for sliding a window over input_ids.");
  }

  window_size_ = model_.config_->model.decoder.sliding_window->window_size;
  shape_ = {1, model_.config_->model.decoder.sliding_window->window_size};
  type_ = model_.session_info_.GetInputDataType(name_);

  if (type_ != Ort::TypeToTensorType<int32_t> && type_ != Ort::TypeToTensorType<int64_t>) {
    throw std::runtime_error("WindowedInputIDs only supports int32_t and int64_t input_ids.");
  }

  if (model_.session_info_.HasInput(model_.config_->model.decoder.inputs.total_sequence_length) &&
      model_.session_info_.HasInput(model_.config_->model.decoder.inputs.past_sequence_length)) {
    const std::array<int64_t, 1> total_sequence_length_shape{1};
    const std::array<int64_t, 2> past_sequence_length_shape{1, 1};

    if (model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.total_sequence_length) != Ort::TypeToTensorType<int32_t> ||
        model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.past_sequence_length) != Ort::TypeToTensorType<int32_t>)
      throw std::runtime_error("total_sequence_length and past_sequence_length must be int32");

    total_sequence_length_ = OrtValue::CreateTensor(model_.allocator_cpu_, total_sequence_length_shape,
                                                    model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.total_sequence_length));
    *total_sequence_length_->GetTensorMutableData<int32_t>() = state_.params_->search.max_length;

    past_sequence_length_ = OrtValue::CreateTensor(model_.allocator_cpu_, past_sequence_length_shape,
                                                   model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.past_sequence_length));
    *past_sequence_length_->GetTensorMutableData<int32_t>() = -1;
  }
}

void WindowedInputIDs::Add() {
  input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(value_.get());
  state_.input_names_.push_back(name_);

  if (total_sequence_length_ && past_sequence_length_) {
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.total_sequence_length.c_str());
    state_.inputs_.push_back(total_sequence_length_.get());
    state_.input_names_.push_back(model_.config_->model.decoder.inputs.past_sequence_length.c_str());
    state_.inputs_.push_back(past_sequence_length_.get());
  }
}

void WindowedInputIDs::Update(DeviceSpan<int32_t> new_tokens) {
  if (window_index_ == 0) {
    num_windows_ = (new_tokens.size() + window_size_ - 1) / window_size_;

    const auto get_unpadded_sequence_length = [](std::span<const int32_t> tokens,
                                                 int32_t pad_token_id) {
      for (int32_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == pad_token_id) {
          return i;
        }
      }
      return static_cast<int32_t>(tokens.size());
    };

    initial_num_tokens_ += get_unpadded_sequence_length(new_tokens.CpuSpan(), model_.config_->model.pad_token_id);

    value_ = OrtValue::CreateTensor<int32_t>(model_.p_device_inputs_->GetAllocator(), shape_);

    // new_tokens will always be padded so that it's size is a multiple of window_size_
    // new_tokens -> [0, a, b, c, d, e]
    // window_size = 3, num_windows = 2, pad_token = 0
    // window_index = 0, value_ -> [0, a, b]
    std::copy_n(new_tokens.Span().begin(), window_size_, value_->GetTensorMutableData<int32_t>());

    if (past_sequence_length_)
      *past_sequence_length_->GetTensorMutableData<int32_t>() += static_cast<int32_t>(window_size_);
  } else if (window_index_ < num_windows_) {
    // new_tokens -> [a, b, c, d, e]
    // window_size = 3, num_windows = 2
    // window_index = 1, value_ -> [c, d, e]
    std::copy_n(new_tokens.Span().begin() + window_index_ * window_size_, window_size_, value_->GetTensorMutableData<int32_t>());

    if (past_sequence_length_)
      *past_sequence_length_->GetTensorMutableData<int32_t>() += static_cast<int32_t>(window_size_);
  } else {
    // All prompt token chunks have been processed. Now we process the tokens generated by the model.
    // new_tokens -> [f]
    assert(new_tokens.size() == 1);
    if (shape_[1] != 1) {
      shape_[1] = 1;
      value_ = OrtValue::CreateTensor<int32_t>(model_.p_device_inputs_->GetAllocator(), shape_);

      if (type_ == Ort::TypeToTensorType<int64_t>) {
        cast_value_ = OrtValue::CreateTensor<int64_t>(model_.p_device_inputs_->GetAllocator(), shape_);
      }

      if (past_sequence_length_)
        *past_sequence_length_->GetTensorMutableData<int32_t>() = initial_num_tokens_;
    } else {
      if (past_sequence_length_)
        *past_sequence_length_->GetTensorMutableData<int32_t>() += 1;
    }

    value_->GetTensorMutableData<int32_t>()[0] = new_tokens.Span()[0];
  }

  state_.inputs_[input_index_] = value_.get();

  if (type_ == Ort::TypeToTensorType<int64_t>) {
    Cast(*value_, cast_value_, *model_.p_device_inputs_, type_);
    state_.inputs_[input_index_] = cast_value_.get();
  }
  window_index_++;
}

std::unique_ptr<InputIDs> CreateInputIDs(State& state) {
  if (state.model_.config_->model.decoder.sliding_window.has_value() && state.model_.config_->model.decoder.sliding_window->slide_inputs) {
    return std::make_unique<WindowedInputIDs>(state);
  } else {
    return std::make_unique<DefaultInputIDs>(state);
  }
}

}  // namespace Generators
