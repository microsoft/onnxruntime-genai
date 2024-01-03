#pragma once

namespace Generators {

template<typename T>
struct PositionIDs {

  PositionIDs(Model& model, const SearchParams& search_params, Ort::Allocator& allocator, RoamingArray<int32_t>& sequence_lengths);
  void Update(int current_length);

  Model& model_;
  Ort::Allocator& allocator_;

  std::array<int64_t, 2> position_ids_shape_; // {search_params.batch_size*search_params.beam_size, search_params.sequence_length}
  std::unique_ptr<OrtValue> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_; // {search_params.batch_size*search_params.beam_size, search_params.sequence_length}
  std::unique_ptr<OrtValue> attention_mask_;
};

} // namespace Generators
