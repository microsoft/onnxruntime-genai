#pragma once

#include "static_buffer.h"

namespace Generators {

struct InputIDs {
  InputIDs(const Model& model, State& state);
  InputIDs(const InputIDs&) = delete;
  InputIDs& operator=(const InputIDs&) = delete;

  void Add();
  void Update(RoamingArray<int32_t> next_tokens);

  auto& GetShape() const { return shape_; }
  const char* name_;

  OrtValue* Get() { return value_.get(); }

 private:
  const Model& model_;
  State& state_;
  size_t input_index_{~0U};

  std::array<int64_t, 1> shape_{};
  ONNXTensorElementDataType type_;
  std::unique_ptr<OrtValue> value_;
  std::unique_ptr<OrtValue> is_prompt_;
  std::vector<int32_t> is_prompt_data_{1};
  std::array<int64_t, 1> is_prompt_shape_{1};

  // Used for decoding runs with cuda graphs.
  StaticBuffer* sb_input_ids_{};

#if USE_DML
  std::unique_ptr<OrtValue> value_int32_;
  StaticBuffer* sb_input_ids_int32_{};
  DmlReusedCommandListState input_ids_cast_command_list_state_{};
#endif
};

}  // namespace Generators
