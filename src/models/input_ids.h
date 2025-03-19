#pragma once

namespace Generators {

struct InputIDs {
  virtual ~InputIDs() = default;
  virtual void Add() = 0;
  virtual std::array<int64_t, 2> GetShape() const = 0;
  virtual void Update(DeviceSpan<int32_t> next_tokens) = 0;
};

struct DefaultInputIDs : InputIDs {
  DefaultInputIDs(State& state);
  DefaultInputIDs(const DefaultInputIDs&) = delete;
  DefaultInputIDs& operator=(const DefaultInputIDs&) = delete;

  // Register input_ids as ORT session input.
  // Called only once during initialization of state.
  void Add() override;
  // Resize input_ids based on size of next_tokens.
  // Update value with next_tokens.
  void Update(DeviceSpan<int32_t> next_tokens) override;

  std::array<int64_t, 2> GetShape() const override { return shape_; }
  const char* name_;

  OrtValue* Get() { return value_->GetOrtTensor(); }

 private:
  State& state_;
  const Model& model_{state_.model_};
  size_t input_index_{~0U};

  bool is_prompt_{true};

  std::array<int64_t, 2> shape_{};
  ONNXTensorElementDataType type_;
  std::unique_ptr<Tensor> value_;
  std::unique_ptr<Tensor> cast_value_;

  std::unique_ptr<OrtValue> current_sequence_length_;
  std::unique_ptr<OrtValue> past_sequence_length_;
};

// Certain models can only process a fixed number of tokens at a time.
// For example, given a prompt with 120 tokens, and a model that can only process 20 tokens at a time,
// this class will split the prompt into 6 windows of 20 tokens each.
// At each update step, the next window of tokens is processed.
// This is done until all windows have been processed before switching to the model-generated tokens
// which are processed one token at a time.
// In contrast, DefaultInputIDs processes all prompt tokens at once.
struct WindowedInputIDs : public InputIDs {
  WindowedInputIDs(State& state);
  WindowedInputIDs(const WindowedInputIDs&) = delete;
  WindowedInputIDs& operator=(const WindowedInputIDs&) = delete;

  void Add() override;
  void Update(DeviceSpan<int32_t> next_tokens) override;
  std::array<int64_t, 2> GetShape() const override { return shape_; }

 private:
  State& state_;
  const Model& model_{state_.model_};
  size_t input_index_{~0U};
  size_t window_size_{};
  size_t num_windows_{};
  size_t window_index_{};
  const char* name_;
  std::array<int64_t, 2> shape_{};
  ONNXTensorElementDataType type_;

  std::unique_ptr<OrtValue> value_;
  std::unique_ptr<OrtValue> cast_value_;
  std::unique_ptr<OrtValue> total_sequence_length_;
  std::unique_ptr<OrtValue> past_sequence_length_;
  int32_t initial_num_tokens_{};
};

std::unique_ptr<InputIDs> CreateInputIDs(State& state);

}  // namespace Generators
