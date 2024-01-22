#pragma once

namespace Generators {

template <typename T>
struct InputIDs {
  InputIDs(Model& model, State& state);

  void Add();
  void Update(RoamingArray<int32_t> next_tokens);

  auto& GetShape() const { return shape_; }
  const char* name_{"input_ids"};

 private:
  Model& model_;
  State& state_;
  size_t input_index_{~0U};

  std::array<int64_t, 2> shape_;
  std::unique_ptr<OrtValue> value_;
};

}  // namespace Generators
