#pragma once

#include "model.h"

namespace Generators {

struct KeyValueCache {
  virtual ~KeyValueCache() = default;

  virtual void Add() = 0;

  virtual void AddEncoder() = 0;

  virtual void Update(DeviceSpan<int32_t> beam_indices, int total_length) = 0;

  virtual void RewindTo(size_t index) = 0;

  // Note: PartialTokenGenerationUpdate() is mainly for supporting DecoderOnlyPipelineState usage where we update
  // part of the KV cache after running part of the pipeline.
  // An alternative may be to have a dedicated KV cache per IntermediatePipelineState.

  virtual bool IsPartialTokenGenerationUpdateSupported() const { return false; }

  virtual void PartialTokenGenerationUpdate(DeviceSpan<int32_t> beam_indices, int total_length,
                                            std::span<const size_t> layer_indices_to_update) {
    throw std::runtime_error("PartialTokenGenerationUpdate is not supported.");
  }
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

  DeviceInterface& Device() { return *model_.p_device_kvcache_; }
  Ort::Allocator& Allocator() { return model_.p_device_kvcache_->GetAllocator(); }

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

  DeviceInterface& Device() { return *model_.p_device_kvcache_; }
  Ort::Allocator& Allocator() { return model_.p_device_kvcache_->GetAllocator(); }

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
};

// Very similar to the DefaultKeyValueCache, but is only created once at the encoder step, then used without modification for every decoder step
struct CrossCache {
  CrossCache(State& state);

  void AddOutputs();
  void AddInputs();

 private:
  DeviceInterface& Device() { return *model_.p_device_kvcache_; }
  Ort::Allocator& Allocator() { return model_.p_device_kvcache_->GetAllocator(); }

  State& state_;
  const Model& model_{state_.model_};
  int layer_count_;

  std::array<int64_t, 4> shape_;
  ONNXTensorElementDataType type_;

  std::vector<std::unique_ptr<OrtValue>> values_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};

// A (mostly) NO-OP KeyValueCache variant that is used for stateful models
// i.e. Models that manage KV Cache internally to the session.
struct ModelManagedKeyValueCache : KeyValueCache {
  ModelManagedKeyValueCache(State& state);

  virtual void Add() override;
  virtual void AddEncoder() override;
  virtual void Update(DeviceSpan<int32_t> beam_indices, int total_length) override;
  virtual void RewindTo(size_t index) override;

 private:
  State& state_;
  const Model& model_{state_.model_};
};

std::string ComposeKeyValueName(const std::string& template_string, int index);

std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state);

}  // namespace Generators
