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
    const int configured_window_size = model_.config_->model.decoder.sliding_window->window_size;
    if (configured_window_size <= 0) {
      throw std::runtime_error("Window size must be greater than 0");
    }
    if (has_mask_input_ && model_.config_->model.context_length < configured_window_size) {
      throw std::runtime_error("Model context_length must be greater than or equal to sliding_window.window_size");
    }
    window_size_ = static_cast<size_t>(configured_window_size);
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

  auto is_valid_token = [&](size_t token_index) {
    return token_index >= valid_begin_ && token_index < valid_end_;
  };

  auto fill_window_positions = [&](auto* position_ids_data, size_t window_offset) {
    using T = std::remove_pointer_t<decltype(position_ids_data)>;
    for (size_t i = 0; i < window_size_; i++) {
      if (!is_valid_token(window_offset + i)) {
        position_ids_data[i] = T{0};
      } else {
        position_ids_data[i] = static_cast<T>(next_position_id_++);
      }
    }
  };

  auto fill_window_mask = [&](auto* attention_mask_data, size_t window_offset, bool zero_prefix) {
    using T = std::remove_pointer_t<decltype(attention_mask_data)>;
    const size_t mask_length = static_cast<size_t>(attention_mask_shape_[1]);
    if (zero_prefix) {
      std::fill_n(attention_mask_data, mask_length - window_size_, T{0});
    } else {
      std::move(attention_mask_data + window_size_, attention_mask_data + mask_length, attention_mask_data);
    }
    for (size_t i = 0; i < window_size_; i++) {
      attention_mask_data[mask_length - window_size_ + i] =
          is_valid_token(window_offset + i) ? T{1} : T{0};
    }
  };

  if (window_index_ == 0 && (is_first_update_ || next_tokens.size() > 1)) {
    if (next_tokens.size() < window_size_ || next_tokens.size() % window_size_ != 0) {
      throw std::runtime_error("WindowedPositionInputs::Update requires next_tokens size to be a positive multiple of window_size");
    }
    num_windows_ = next_tokens.size() / window_size_;
    auto tokens_cpu = next_tokens.CpuSpan();
    valid_begin_ = 0;
    valid_end_ = tokens_cpu.size();
    if (model_.config_->model.decoder.sliding_window->alignment == "left") {
      while (valid_end_ > 0 && tokens_cpu[valid_end_ - 1] == model_.config_->model.pad_token_id) {
        --valid_end_;
      }
    } else {
      while (valid_begin_ < valid_end_ && tokens_cpu[valid_begin_] == model_.config_->model.pad_token_id) {
        ++valid_begin_;
      }
    }

    const bool zero_prefix = is_first_update_;
    if (is_first_update_) {
      next_position_id_ = 0;
    }
    if (has_posid_input_) {
      const auto window_width = static_cast<int64_t>(window_size_);
      if (!position_ids_ || position_ids_shape_[1] != window_width) {
        position_ids_shape_ = {1, window_width};
        position_ids_ = OrtValue::CreateTensor(model_.allocator_cpu_, position_ids_shape_, position_ids_type_);
      }

      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 0, position_ids_ -> [0, 0, 1]
      if (position_ids_type_ == Ort::TypeToTensorType<int64_t>)
        fill_window_positions(position_ids_->GetTensorMutableData<int64_t>(), 0);
      else
        fill_window_positions(position_ids_->GetTensorMutableData<int32_t>(), 0);
    }

    if (has_mask_input_) {
      if (!attention_mask_) {
        attention_mask_ = OrtValue::CreateTensor(model_.allocator_cpu_, attention_mask_shape_, attention_mask_type_);
      }

      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 0, attention_mask_ -> ([0] * context_length - window_size_) + [0, 1, 1]
      if (attention_mask_type_ == Ort::TypeToTensorType<int64_t>)
        fill_window_mask(attention_mask_->GetTensorMutableData<int64_t>(), 0, zero_prefix);
      else
        fill_window_mask(attention_mask_->GetTensorMutableData<int32_t>(), 0, zero_prefix);
    }
    is_first_update_ = false;
    window_index_++;
    if (window_index_ == num_windows_) {
      window_index_ = 0;
      num_windows_ = 0;
    }
  } else if (window_index_ < num_windows_) {
    const size_t window_offset = window_index_ * window_size_;
    if (next_tokens.size() < window_offset + window_size_) {
      throw std::runtime_error("WindowedPositionInputs::Update requires next_tokens to cover the current window");
    }
    if (has_posid_input_) {
      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 1, position_ids_ -> [2, 3, 4]
      if (position_ids_type_ == Ort::TypeToTensorType<int64_t>)
        fill_window_positions(position_ids_->GetTensorMutableData<int64_t>(), window_offset);
      else
        fill_window_positions(position_ids_->GetTensorMutableData<int32_t>(), window_offset);
    }

    if (has_mask_input_) {
      // next_tokens will always be padded so that it's size is a multiple of window_size_
      // next_tokens -> [0, a, b, c, d, e]
      // window_size = 3, num_windows = 2, pad_token = 0
      // window_index = 1, attention_mask_ -> ([0] * context_length - (2 * window_size_)) + [0, 1, 1, 1, 1, 1]
      if (attention_mask_type_ == Ort::TypeToTensorType<int64_t>)
        fill_window_mask(attention_mask_->GetTensorMutableData<int64_t>(), window_offset, false);
      else
        fill_window_mask(attention_mask_->GetTensorMutableData<int32_t>(), window_offset, false);
    }
    window_index_++;
    if (window_index_ == num_windows_) {
      window_index_ = 0;
      num_windows_ = 0;
    }
  } else {
    if (next_tokens.size() != 1) {
      throw std::runtime_error("WindowedPositionInputs::Update expects a single token during decode");
    }
    // All prompt token chunks have been processed. Now we process the tokens generated by the model.
    if (has_posid_input_) {
      // next_tokens -> [f]
      // position_ids_ -> [5]
      auto fill_generated = [&](auto* data) {
        using T = std::remove_pointer_t<decltype(data)>;
        if (position_ids_shape_[1] != 1) {
          position_ids_shape_[1] = 1;
          position_ids_ = OrtValue::CreateTensor(model_.allocator_cpu_, position_ids_shape_, position_ids_type_);
          data = position_ids_->GetTensorMutableData<T>();
        }
        data[0] = static_cast<T>(next_position_id_++);
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
        const size_t mask_length = static_cast<size_t>(attention_mask_shape_[1]);
        if (mask_length > 1) {
          std::move(data + 1, data + mask_length, data);
        }
        data[mask_length - 1] = T{1};
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
}

}  // namespace Generators
