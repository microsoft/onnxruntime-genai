#pragma once

#include "static_buffer.h"

namespace Generators {

struct InputIDs {
  InputIDs(State& state);
  InputIDs(const InputIDs&) = delete;
  InputIDs& operator=(const InputIDs&) = delete;

  void Add();
  void Update(DeviceSpan<int32_t> next_tokens);

  auto& GetShape() const { return shape_; }
  const char* name_;

  OrtValue* Get() { return value_.get(); }

 private:
  State& state_;
  const Model& model_{state_.model_};
  size_t input_index_{~0U};

  std::array<int64_t, 2> shape_{};
  ONNXTensorElementDataType type_;
  std::unique_ptr<OrtValue> value_;

  // Used for decoding runs with cuda graphs.
  StaticBuffer* sb_input_ids_{};

#if USE_DML
  std::unique_ptr<OrtValue> value_int32_;
  StaticBuffer* sb_input_ids_int32_{};
  DmlReusedCommandListState input_ids_cast_command_list_state_{};
#endif

  std::unique_ptr<OrtValue> current_sequence_length_;
  std::unique_ptr<OrtValue> past_sequence_length_;
};

}  // namespace Generators
