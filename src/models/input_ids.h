#pragma once

#include "static_buffer.h"

namespace Generators {

struct InputIDsInterface {
  virtual ~InputIDsInterface() = default;
  virtual void Add() = 0;
  virtual std::array<int64_t, 2> GetShape() const = 0;
  virtual void Update(DeviceSpan<int32_t>& next_tokens) = 0;
};

struct InputIDs : InputIDsInterface {
  InputIDs(State& state);
  InputIDs(const InputIDs&) = delete;
  InputIDs& operator=(const InputIDs&) = delete;

  // Register input_ids as ORT session input.
  // Called only once during initialization of state.
  void Add() override;
  // Resize input_ids based on size of next_tokens.
  // Update value with next_tokens.
  void Update(DeviceSpan<int32_t>& next_tokens) override;

  std::array<int64_t, 2> GetShape() const override { return shape_; }
  const char* name_;

  OrtValue* Get() { return value_.get(); }

 private:
  State& state_;
  const Model& model_{state_.model_};
  size_t input_index_{~0U};

  bool is_prompt_{true};

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

struct SlidingWindowInputIDs : public InputIDsInterface {
  SlidingWindowInputIDs(State& state);
  SlidingWindowInputIDs(const SlidingWindowInputIDs&) = delete;
  SlidingWindowInputIDs& operator=(const SlidingWindowInputIDs&) = delete;

  void Add() override;
  void Update(DeviceSpan<int32_t>& next_tokens) override;
  std::array<int64_t, 2> GetShape() const override { return shape_; }

 private:
  State& state_;
  const Model& model_{state_.model_};
  size_t input_index_{~0U};
  size_t window_size_{0};
  size_t num_windows_{1};
  size_t window_index_{0};
  const char* name_;
  std::array<int64_t, 2> shape_{};
  ONNXTensorElementDataType type_;

  std::unique_ptr<OrtValue> value_;
};

std::unique_ptr<InputIDsInterface> CreateInputIDs(State& state);

}  // namespace Generators
