#pragma once

namespace Generators {

struct InputIDs {
  InputIDs(Model& model, const SearchParams& search_params, Ort::Allocator& allocator);
  void Update(RoamingArray<int32_t> next_tokens);

  Model& model_;
  Ort::Allocator& allocator_;

  std::array<int64_t, 2> input_ids_shape_;
  std::unique_ptr<OrtValue> input_ids_;
};

struct InputIDs64 {
  InputIDs64(Model& model, const SearchParams& search_params, Ort::Allocator& allocator);
  void Update(RoamingArray<int32_t> next_tokens);

  Model& model_;
  Ort::Allocator& allocator_;

  std::array<int64_t, 2> input_ids_shape_;
  std::unique_ptr<OrtValue> input_ids_;
};

}  // namespace Generators
