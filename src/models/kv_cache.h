#pragma once

#if USE_CUDA
#include "kernels.h"
#endif
#include "static_buffer.h"

namespace Generators {

std::string ComposeKeyValueName(const std::string& template_string, int index);

struct KV_Cache_Combined {
  KV_Cache_Combined(State& state);

  void Add();  // Add to state inputs/outputs
  void Update(std::span<const int32_t> beam_indices, int current_length);

  template <typename ScoreType>
  void PickPastState(std::span<const int32_t> beam_indices, int index);
  void PickPastState(std::span<const int32_t> beam_indices, int index);

 private:
  State& state_;
  const Model& model_{state_.model_};
  int layer_count_;
  size_t input_index_{~0U}, output_index_{~0U};

  std::array<int64_t, 5> shape_;
  ONNXTensorElementDataType type_;

  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_, presents_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};

struct KV_Cache {
  KV_Cache(State& state);

  static bool IsCacheNeeded(const Model& model);

  void Add();
  auto& GetShape() const { return shape_; }
  auto& GetType() const { return type_; }
  auto& GetPresents() { return presents_; }

  void Update(std::span<const int32_t> beam_indices, int current_length);
  template <typename ScoreType>
  void PickPastState(std::span<const int32_t> beam_indices, int index);
  void PickPastState(std::span<const int32_t> beam_indices, int index);

 private:
  State& state_;
  const Model& model_{state_.model_};
  int layer_count_;
  size_t input_index_{~0U}, output_index_{~0U};
  bool past_present_share_buffer_;  // True if model.decoder.past_present_share_buffer is set to true, and we're using cuda, and not beam search

  std::array<int64_t, 4> shape_;
  ONNXTensorElementDataType type_;

  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_, presents_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
  std::vector<StaticBuffer*> sb_kv_caches_;
};

struct Cross_Cache {
  Cross_Cache(State& state, int sequence_length = 0);

  void AddOutputs(State& state);
  void AddInputs(State& state);
  auto& GetShape() const { return shape_; }
  auto& GetType() const { return type_; }
  auto& GetValues() { return values_; }

 private:
  int layer_count_;

  std::array<int64_t, 4> shape_;
  ONNXTensorElementDataType type_;

  std::vector<std::unique_ptr<OrtValue>> values_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};
}  // namespace Generators
