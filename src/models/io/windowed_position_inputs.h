#pragma once

#include "position_inputs.h"

namespace Generators {

// Certain models can only process a fixed number of tokens at a time.
struct WindowedPositionInputs : PositionInputs {
  WindowedPositionInputs(State& state);
  WindowedPositionInputs(const WindowedPositionInputs&) = delete;
  WindowedPositionInputs& operator=(const WindowedPositionInputs&) = delete;

  void Add() override;
  void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) override;
  void RewindTo(size_t index) override {
    throw std::runtime_error("WindowedPositionInputs does not support RewindTo.");
  };

 private:
  State& state_;
  const Model& model_{state_.model_};
  bool has_mask_input_{};
  bool has_posid_input_{};
  std::array<int64_t, 2> position_ids_shape_{};
  ONNXTensorElementDataType position_ids_type_{};
  std::unique_ptr<OrtValue> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_{};
  ONNXTensorElementDataType attention_mask_type_{};
  std::unique_ptr<OrtValue> attention_mask_;
  size_t attention_mask_backward_offset_{~0U};
  size_t attention_mask_index_{~0U};
  size_t position_ids_index_{~0U};
  size_t window_size_{};
  size_t num_windows_{};
  size_t window_index_{};
};

}  // namespace Generators
