#include "../generators.h"
#include "model.h"
#include "input_ids.h"

namespace Generators {

InputIDs::InputIDs(State& state)
    : state_{state} {
  name_ = model_.config_->model.decoder.inputs.input_ids.c_str();
  shape_ = {state_.params_->BatchBeamSize(), 0};
  type_ = model_.session_info_->GetInputDataType(name_);

  if (state_.GetCapturedGraphInfo()) {
    sb_input_ids_ = state_.GetCapturedGraphInfo()->sb_input_ids_.get();
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

void InputIDs::Update(DeviceSpan<int32_t> new_tokens) {
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
    if (!sb_input_ids_) {
      value_ = OrtValue::CreateTensor<int32_t>(*model_.allocator_device_, shape_);
    } else {
      value_ = sb_input_ids_->CreateTensorOnStaticBuffer(shape_, Ort::TypeToTensorType<int32_t>);
    }

    state_.inputs_[input_index_] = value_.get();
  }

  // Update input_ids with next tokens
  auto data_span = WrapTensor<int32_t>(*model_.p_device_, *value_); 

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
    Cast(*value_, cast_value_, *model_.p_device_, type_);
    state_.inputs_[input_index_] = cast_value_.get();
  }

  is_prompt_ = false;
}

}  // namespace Generators
