#pragma once

namespace Generators {

struct PositionIDs {
  PositionIDs(const Model& model, State& state, RoamingArray<int32_t>& sequence_lengths);

  void Add();
  void Update(int current_length);

 private:
  template <typename T>
  void InitializeTensors(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths);

  template <typename T>
  void UpdatePositionIDs();
  template <typename T>
  void UpdateAttentionMask(T* data, const T* old_data, int current_length);

  const Model& model_;
  State& state_;
  size_t input_index_{~0U};
  ONNXTensorElementDataType type_;  // Common type for position_ids and attention_mask
  bool has_position_ids_;

  std::array<int64_t, 2> position_ids_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<OrtValue> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<OrtValue> attention_mask_;

  std::unique_ptr<OrtValue> position_ids_next_;  // Replaces position_ids_ after the first Run() call
};

}  // namespace Generators
