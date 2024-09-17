#pragma once

#include "static_buffer.h"

namespace Generators {

struct InputIDs {
  InputIDs(const Model& model, State& state);
  InputIDs(const InputIDs&) = delete;
  InputIDs& operator=(const InputIDs&) = delete;

  // Register input_ids as ORT session input.
  // Called only once during initialization of state.
  void Add();
  // Resize input_ids to [1], update value with next_tokens.
  // next_tokens is assumed to have length 1.
  void Update(RoamingArray<int32_t> next_tokens);
  // Add tokens to the end of input ids tensor
  // void InputIDs::AddInputTokens(RoamingArray<int32_t> tokens, bool is_first_tokens);

  auto& GetShape() const { return shape_; }
  const char* name_;

  OrtValue* Get() { return value_.get(); }

 private:
  const Model& model_;
  State& state_;
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
};

}  // namespace Generators
