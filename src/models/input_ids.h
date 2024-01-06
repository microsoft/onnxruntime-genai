#pragma once

namespace Generators {

template<typename T>
struct InputIDs {
  InputIDs(Model& model, const SearchParams& search_params);
  void Update(RoamingArray<int32_t> next_tokens);

  Model& model_;

  std::array<int64_t, 2> input_ids_shape_;
  std::unique_ptr<OrtValue> input_ids_;
};

}  // namespace Generators
