#include "windowed_position_inputs.h"

#include "generator/generators.h"
#include "models/model.h"
#include "models/model_type.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

namespace Generators {

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
    if (position_ids_type_ != Ort::TypeToTensorType<int32_t> && position_ids_type_ != Ort::TypeToTensorType<int64_t>)
      throw std::runtime_error("WindowedPositionInputs only supports int32_t or int64_t position_ids");

    position_ids_shape_ = {1, model_.config_->model.decoder.sliding_window->window_size};
  }

  if (has_mask_input_) {
    attention_mask_type_ = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);
    if (attention_mask_type_ != Ort::TypeToTensorType<int32_t> && attention_mask_type_ != Ort::TypeToTensorType<int64_t>)
      throw std::runtime_error("WindowedPositionInputs only supports int32_t or int64_t attention_mask");

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
      auto fill_first_window = [&](auto* position_ids_data) {
        using T = std::remove_pointer_t<decltype(position_ids_data)>;
        for (int i = 0, j = 0; i < position_ids_shape_[1]; i++) {
          if (next_tokens.Span()[i] == model_.config_->model.pad_token_id) {
            position_ids_data[i] = T{0};
          } else {
            position_ids_data[i] = static_cast<T>(j++);
          }
        }
      };
      if (position_ids_type_ == Ort::TypeToTensorType<int64_t>)
        fill_first_window(position_ids_->GetTensorMutableData<int64_t>());
      else
        fill_first_window(position_ids_->GetTensorMutableData<int32_t>());
    }

    if (has_mask_input_) {
      attention_mask_ = OrtValue::CreateTensor(model_.allocator_cpu_, attention_mask_shape_, attention_mask_type_);

      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 0, attention_mask_ -> ([0] * context_length - window_size_) + [0, 1, 1]
      auto fill_first_mask = [&](auto* attention_mask_data) {
        using T = std::remove_pointer_t<decltype(attention_mask_data)>;
        std::fill_n(attention_mask_data, attention_mask_shape_[1] - window_size_, T{0});
        for (size_t i = 0; i < window_size_; i++) {
          attention_mask_data[attention_mask_shape_[1] - window_size_ + i] = next_tokens.CpuSpan()[i] == model_.config_->model.pad_token_id ? T{0} : T{1};
        }
        for (size_t i = 0; i < window_size_; i++) {
          if (attention_mask_data[attention_mask_shape_[1] - window_size_ + i] == T{1}) {
            attention_mask_backward_offset_ = attention_mask_shape_[1] - window_size_ + i - 1;
            break;
          }
        }
      };
      if (attention_mask_type_ == Ort::TypeToTensorType<int64_t>)
        fill_first_mask(attention_mask_->GetTensorMutableData<int64_t>());
      else
        fill_first_mask(attention_mask_->GetTensorMutableData<int32_t>());
    }
  } else if (window_index_ < num_windows_) {
    if (has_posid_input_) {
      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 1, position_ids_ -> [2, 3, 4]

      auto fill_next_window = [&](auto* position_ids_data) {
        const auto last_position = position_ids_data[window_size_ - 1];
        std::iota(position_ids_data, position_ids_data + window_size_, last_position + 1);
      };
      if (position_ids_type_ == Ort::TypeToTensorType<int64_t>)
        fill_next_window(position_ids_->GetTensorMutableData<int64_t>());
      else
        fill_next_window(position_ids_->GetTensorMutableData<int32_t>());
    }

    if (has_mask_input_) {
      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 1, attention_mask_ -> ([0] * context_length - (2 * window_size_)) + [0, 1, 1, 1, 1, 1]
      auto fill_next_mask = [&](auto* attention_mask_data) {
        using T = std::remove_pointer_t<decltype(attention_mask_data)>;
        std::fill_n(attention_mask_data + attention_mask_backward_offset_ - window_size_ + 1, window_size_, T{1});
        attention_mask_backward_offset_ -= window_size_;
      };
      if (attention_mask_type_ == Ort::TypeToTensorType<int64_t>)
        fill_next_mask(attention_mask_->GetTensorMutableData<int64_t>());
      else
        fill_next_mask(attention_mask_->GetTensorMutableData<int32_t>());
    }
  } else {
    // All prompt token chunks have been processed. Now we process the tokens generated by the model.
    if (has_posid_input_) {
      // next_tokens -> [f]
      // position_ids_ -> [5]
      auto fill_generated = [&](auto* data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const auto last_position = data[position_ids_shape_[1] - 1];
        if (position_ids_shape_[1] != 1) {
          position_ids_shape_[1] = 1;
          position_ids_ = OrtValue::CreateTensor(model_.allocator_cpu_, position_ids_shape_, position_ids_type_);
          data = position_ids_->GetTensorMutableData<T>();
        }
        data[0] = last_position + 1;
      };
      if (position_ids_type_ == Ort::TypeToTensorType<int64_t>)
        fill_generated(position_ids_->GetTensorMutableData<int64_t>());
      else
        fill_generated(position_ids_->GetTensorMutableData<int32_t>());
    }

    if (has_mask_input_) {
      // next_tokens -> [f]
      // attention_mask_ -> ([0] * context_length - (2 * window_size_) - 1) + [0, 1, 1, 1, 1, 1, 1]
      auto fill_generated_mask = [&](auto* data) {
        using T = std::remove_pointer_t<decltype(data)>;
        data[attention_mask_backward_offset_] = T{1};
        if (attention_mask_backward_offset_ > 0) {
          attention_mask_backward_offset_ -= 1;
        }
      };
      if (attention_mask_type_ == Ort::TypeToTensorType<int64_t>)
        fill_generated_mask(attention_mask_->GetTensorMutableData<int64_t>());
      else
        fill_generated_mask(attention_mask_->GetTensorMutableData<int32_t>());
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

}  // namespace Generators
