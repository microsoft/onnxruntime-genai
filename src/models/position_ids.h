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

  std::array<int64_t, 2> position_ids_shape_{};  // {search_params.batch_size*search_params.beam_size, search_params.sequence_length}
  std::unique_ptr<OrtValue> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_{};  // {search_params.batch_size*search_params.beam_size, search_params.sequence_length}
  std::unique_ptr<OrtValue> attention_mask_;

  std::vector<int32_t> initial_sequence_lengths_;
};

}  // namespace Generators
