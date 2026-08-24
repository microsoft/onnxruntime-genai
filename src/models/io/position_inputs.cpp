#include "position_inputs.h"

#include "default_position_inputs.h"
#include "qwen_vl_position_inputs.h"
#include "windowed_position_inputs.h"
#include "models/model.h"
#include "models/model_type.h"

namespace Generators {

std::unique_ptr<PositionInputs> CreateStandardPositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths, const std::string& attention_mask_name) {
  // Check for Qwen-VL family models which require 3D mRoPE position IDs
  if (ModelType::IsQwenVLFamily(state.model_.config_->model.type)) {
    return std::make_unique<Qwen2VLPositionInputs>(state.model_, state, sequence_lengths);
  }
  if (state.model_.config_->model.decoder.sliding_window.has_value() && state.model_.config_->model.decoder.sliding_window->slide_inputs) {
    return std::make_unique<WindowedPositionInputs>(state);
  } else {
    return std::make_unique<DefaultPositionInputs>(state.model_, state, sequence_lengths, attention_mask_name);
  }
}

}  // namespace Generators
