#pragma once

namespace Generators {

template<typename T>
struct PositionIDs {

  PositionIDs(Model& model, State& state, RoamingArray<int32_t>& sequence_lengths);

  void Add();
  void Update(int current_length);

private:

  Model& model_;
  State& state_;
  size_t input_index_{~0U};

  std::array<int64_t, 2> position_ids_shape_; // {search_params.batch_size*search_params.beam_size, search_params.sequence_length}
  std::unique_ptr<OrtValue> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_; // {search_params.batch_size*search_params.beam_size, search_params.sequence_length}
  std::unique_ptr<OrtValue> attention_mask_;
};

} // namespace Generators
