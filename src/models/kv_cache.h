#pragma once

#include "static_buffer.h"

namespace Generators {

struct KeyValueCache {
  virtual ~KeyValueCache() = default;
  virtual void Add() = 0;
  virtual void AddEncoder() = 0;
  virtual void Update(DeviceSpan<int32_t> beam_indices, int total_length) = 0;
  virtual void RewindTo(size_t index) = 0;
};

struct CombinedKeyValueCache : KeyValueCache {
  CombinedKeyValueCache(State& state);

  void Add() override;  // Add to state inputs/outputs
  void AddEncoder() override {
    throw std::runtime_error("CombinedKeyValueCache does not support AddEncoder.");
  };
  void Update(DeviceSpan<int32_t> beam_indices, int total_length) override;
  void RewindTo(size_t index) override;

 private:
  template <typename ScoreType>
  void PickPastState(DeviceSpan<int32_t> beam_indices, int index);
  void PickPastState(DeviceSpan<int32_t> beam_indices, int index);

  template <typename T>
  void RewindPastTensorsTo(size_t index);

  State& state_;
  const Model& model_{state_.model_};
  int layer_count_;
  size_t input_index_{~0U}, output_index_{~0U};

  bool is_first_update_{true};

  std::array<int64_t, 5> shape_;
  ONNXTensorElementDataType type_;

  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_, presents_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};

struct DefaultKeyValueCache : KeyValueCache {
  DefaultKeyValueCache(State& state);

  void AddEncoder() override;  // If model has an initial encoder step, this is used
  // Register input_ids as ORT session input.
  // Called only once during initialization of state.
  void Add() override;
  // Move present to past. Prepare present output for next generation iteration.
  void Update(DeviceSpan<int32_t> beam_indices, int total_length) override;
  void RewindTo(size_t index) override;

 private:
  template <typename ScoreType>
  void PickPastState(DeviceSpan<int32_t> beam_indices, int index);
  void PickPastState(DeviceSpan<int32_t> beam_indices, int index);

  template <typename T>
  void RewindPastTensorsTo(size_t index);

  State& state_;
  const Model& model_{state_.model_};
  int layer_count_;
  size_t input_index_{~0U}, output_index_{~0U};
  bool past_present_share_buffer_;  // True if model.decoder.past_present_share_buffer is set to true, and we're using cuda, and not beam search

  bool is_first_update_{true};

  std::array<int64_t, 4> shape_;
  ONNXTensorElementDataType type_;

  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_, presents_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
  std::vector<StaticBuffer*> sb_kv_caches_;
};

// Very similar to the DefaultKeyValueCache, but is only created once at the encoder step, then used without modification for every decoder step
struct CrossCache {
  CrossCache(State& state);

  void AddOutputs();
  void AddInputs();

 private:
  State& state_;
  const Model& model_{state_.model_};
  int layer_count_;

  std::array<int64_t, 4> shape_;
  ONNXTensorElementDataType type_;

  std::vector<std::unique_ptr<OrtValue>> values_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};

struct WindowedKeyValueCache : KeyValueCache {
  WindowedKeyValueCache(State& state);

  void Add() override;
  void AddEncoder() override {
    throw std::runtime_error("WindowedKeyValueCache does not support AddEncoder.");
  };
  void Update(DeviceSpan<int32_t> beam_indices, int current_length) override;
  void RewindTo(size_t index) override {
    throw std::runtime_error("WindowedKeyValueCache does not support RewindTo.");
  }

 private:
  void Slide();

  State& state_;
  const Model& model_{state_.model_};
  int layer_count_{};
  int window_size_{};
  size_t num_windows_{};
  size_t window_index_{};
  size_t input_index_{~0U}, output_index_{~0U};

  std::array<int64_t, 4> key_cache_shape_in_, key_cache_shape_out_;
  std::array<int64_t, 4> value_cache_shape_in_, value_cache_shape_out_;
  ONNXTensorElementDataType type_;

  std::vector<std::unique_ptr<OrtValue>> key_caches_in_, value_caches_in_;
  std::vector<std::unique_ptr<OrtValue>> key_caches_out_, value_caches_out_;
  std::vector<std::string> input_name_strings_, output_name_strings_;

  bool is_first_update_{true};
};

std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state);

}  // namespace Generators
