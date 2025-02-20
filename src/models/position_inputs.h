#pragma once
#include "static_buffer.h"

namespace Generators {

struct PositionInputs {
  virtual ~PositionInputs() = default;
  virtual void Add() = 0;
  virtual void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) = 0;
  virtual void RewindTo(size_t index) = 0;
};

struct DefaultPositionInputs : PositionInputs {
  DefaultPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk);

  void Add() override;
  void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) override;

  void RewindTo(size_t index) override;

 private:
  void AddAttentionMask();
  void AddPositionIDs();

  void CreateNextPositionIDsTensor();
  void CreateNextAttentionMaskTensor(int total_length);

  void UpdatePositionIDs(int total_length, int new_length);
  void UpdateAttentionMask(int total_length, int new_length);

  template <typename T>
  void InitializeSequenceLengths(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths_unk);
  template <typename T>
  void CreateAndInitializePositionIDs(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape);
  template <typename T>
  void CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape);

  template <typename T>
  void UpdatePositionIDsImpl(int total_length, int new_kv_length);
  template <typename T>
  void UpdateAttentionMaskImpl(int total_length);

  void RewindMask(size_t index);

  const Model& model_;
  State& state_;

  size_t mask_input_index_{~0U};
  size_t posid_input_index_{~0U};

  ONNXTensorElementDataType type_;  // Common type for position_ids and attention_mask

  bool has_mask_input_{};
  bool has_posid_input_{};

  std::array<int64_t, 2> position_ids_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<OrtValue> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<OrtValue> attention_mask_;

  std::unique_ptr<OrtValue> position_ids_next_;    // Replaces position_ids_ after the first Run() call
  std::unique_ptr<OrtValue> attention_mask_next_;  // Replaces attention_mask_ after the first Run() call

  // Used for decoding runs with cuda graphs.
  StaticBuffer* sb_position_ids_{};
  StaticBuffer* sb_attention_mask_{};

  bool is_first_mask_update_{true};
  bool is_first_update_{true};
};

// Certain models can only process a fixed number of tokens at a time.
// For example, given a prompt with 120 tokens, and a model that can only process 20 tokens at a time,
// this class will split the position ids into 6 windows of 20 tokens each.
// At each update step, the next window of position ids is prepared.
// This is done until all windows have been processed before switching to the model-generation phase
// where position ids are prepared one id at a time.
// This class will also prepare the attention mask for each iteration. The attention mask buffer is allocated just
// once and reused for each iteration by setting the mask to 1 for current window tokens and previously active window tokens
// In contrast, DefaultPositionInputs processes all position ids at once.
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

std::unique_ptr<PositionInputs> CreatePositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths);

}  // namespace Generators
