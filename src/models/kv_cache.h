#pragma once

namespace Generators {

struct KV_Cache_Combined {

  KV_Cache_Combined(Model& model, State& state);

  void Add();  // Add to state inputs/outputs
  void Update(std::span<const int32_t> beam_indices, int current_length);

  template <typename ScoreType>
  void PickPastState(std::span<const int32_t> beam_indices, int index);
  void PickPastState(std::span<const int32_t> beam_indices, int index);

private:

  Model& model_;
  State& state_;
  int layer_count_;
  size_t input_index_{~0U}, output_index_{~0U};

  std::array<int64_t, 5> shape_;

  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_, presents_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};

struct KV_Cache {
  KV_Cache(Model& model, State& state);

  void AddEncoder(); // If model has an initial encoder step, this is used
  void Add();
  void Update(std::span<const int32_t> beam_indices, int current_length);
  template <typename ScoreType>
  void PickPastState(std::span<const int32_t> beam_indices, int index);
  void PickPastState(std::span<const int32_t> beam_indices, int index);

private:
  Model& model_;
  State& state_;
  int layer_count_;
  size_t input_index_{~0U}, output_index_{~0U};

  std::array<int64_t, 4> shape_;

  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_, presents_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};

// Very similar to the KV_Cache, but is only created once at the encoder step, then used without modification for every decoder step
struct Cross_Cache {
  Cross_Cache(Model& model, State& state);

  void AddOutputs();
  void AddInputs();

 private:
  Model& model_;
  State& state_;
  int layer_count_;

  std::array<int64_t, 4> shape_;

  std::vector<std::unique_ptr<OrtValue>> values_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};
} // namespace Generators
