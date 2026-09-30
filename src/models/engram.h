#pragma once

#include "model.h"
#include "io/embeddings.h"

namespace Generators {

struct EngramState : State {
  EngramState(const Model& model, OrtSession& session, const GeneratorParams& params);

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;
  void CopyEmbeddingsTo(Embeddings& destination);
  void CopyEmbeddingsTo(OrtValue& destination);
  void RewindTo(size_t index) override;
  void SnapshotState(size_t position) override;
  void CommitAcceptedPrefix(size_t token_count);

 private:
  struct CacheEntry {
    std::vector<uint8_t> embeddings;
    std::vector<int64_t> present_tokens;
  };

  struct CacheKeyHash {
    size_t operator()(const std::vector<int64_t>& key) const noexcept {
      size_t hash = 1469598103934665603ull;
      for (int64_t value : key) {
        hash ^= static_cast<size_t>(value);
        hash *= 1099511628211ull;
      }
      return hash;
    }
  };

  void InitializeTokens(OrtValue& tokens);
  void CopyTokens(OrtValue& source, OrtValue& destination);

  const Model& model_;
  OrtSession& session_;
  std::unique_ptr<OrtValue> input_ids_;
  std::unique_ptr<OrtValue> embeddings_output_;
  std::unique_ptr<OrtValue> past_tokens_;
  std::unique_ptr<OrtValue> present_tokens_;
  std::unique_ptr<OrtValue> snapshot_tokens_;
  size_t past_input_index_{~0U};
  size_t input_ids_index_{~0U};
  size_t embeddings_output_index_{~0U};
  size_t present_output_index_{~0U};
  ONNXTensorElementDataType embeddings_type_{};
  int64_t embeddings_width_{};
  size_t snapshot_position_{};
  bool snapshot_valid_{};
  bool first_run_{true};
  bool cache_hit_{};
  std::optional<std::vector<int64_t>> active_cache_key_;
  std::unordered_map<std::vector<int64_t>, CacheEntry, CacheKeyHash> cache_;
};

}  // namespace Generators