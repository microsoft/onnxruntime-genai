#pragma once

#include "static_buffer.h"
#include "cache_manager.h"

namespace Generators {

struct KV_Cache_Combined {
  KV_Cache_Combined(const Model& model, State& state);

  void Add();  // Add to state inputs/outputs
  void Update(std::span<const int32_t> beam_indices, int current_length);

 private:
  template <typename ScoreType>
  void PickPastState(std::span<const int32_t> beam_indices, int index);
  void PickPastState(std::span<const int32_t> beam_indices, int index);

  const Model& model_;
  State& state_;
  int layer_count_;
  size_t input_index_{~0U}, output_index_{~0U};

  std::array<int64_t, 5> shape_;
  ONNXTensorElementDataType type_;

  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_, presents_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};

class CacheManagerInterface {
 public:
  CacheManagerInterface() = default;

  virtual void Add() = 0;

  virtual void Update(std::span<const int32_t> beam_indices, int current_length) = 0;

  virtual ~CacheManagerInterface() = default;
};

struct KV_Cache : public CacheManagerInterface {
  KV_Cache(const Model& model, State& state);

  void AddEncoder();  // If model has an initial encoder step, this is used
  void Add() override;
  void Update(std::span<const int32_t> beam_indices, int current_length) override;

 private:
  template <typename ScoreType>
  void PickPastState(std::span<const int32_t> beam_indices, int index);
  void PickPastState(std::span<const int32_t> beam_indices, int index);

  const Model& model_;
  State& state_;
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

// Very similar to the KV_Cache, but is only created once at the encoder step, then used without modification for every decoder step
struct Cross_Cache {
  Cross_Cache(const Model& model, State& state);

  void AddOutputs();
  void AddInputs();

 private:
  const Model& model_;
  State& state_;
  int layer_count_;

  std::array<int64_t, 4> shape_;
  ONNXTensorElementDataType type_;

  std::vector<std::unique_ptr<OrtValue>> values_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};

struct PagedCacheOrchestrator : public CacheManagerInterface {
  PagedCacheOrchestrator(const Model& model, State& state);

  void Add() override;
  void Update(std::span<const int32_t> beam_indices, int current_length) override;

 private:
  const Model& model_;
  State& state_;
  int layer_count_;
  size_t input_offset_{~0U};

  std::vector<std::string> input_name_strings_;
  std::unique_ptr<PagedCacheManager> paged_cache_;
};

std::unique_ptr<CacheManagerInterface> CreateCacheManager(const Model& model, State& state);

}  // namespace Generators
