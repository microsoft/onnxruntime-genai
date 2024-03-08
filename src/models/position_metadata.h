#pragma once

namespace Generators {

struct PositionMetadata {
  PositionMetadata(const Model& model, State& state, RoamingArray<int32_t>& sequence_lengths);

  void AddAttentionMask();
  void AddPositionIDs();
  void AddSeqlensK();
  void AddTotalSequenceLength();

  void UpdatePositionIDs(int current_length);
  void UpdateAttentionMask(int current_length);

 private:
  template <typename T>
  void InitializeTensors(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths);

  template <typename T>
  void UpdatePositionIDsImpl();
  template <typename T>
  void UpdateAttentionMaskImpl(T* data, const T* old_data, int current_length);

  const Model& model_;
  State& state_;
  size_t mask_input_index_{~0U};
  size_t posid_input_index_{~0U};
  ONNXTensorElementDataType type_;  // Common type for position_ids and attention_mask

  std::array<int64_t, 2> position_ids_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<OrtValue> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<OrtValue> attention_mask_;
  std::array<int64_t, 1> senlens_k_shape_{}; // {params.batch_size}
  std::unique_ptr<OrtValue> seqlens_k_;
  std::array<int64_t, 0> total_sequence_length_shape_{}; // scalar
  std::unique_ptr<OrtValue> total_sequence_length_;

  std::vector<int32_t> initial_sequence_lengths_;
};

}  // namespace Generators
