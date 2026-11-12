// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"
#include "dflash2_drafter.h"
#include "engine/prefix_cache.h"
#include <deque>
#include <mutex>

namespace Generators {

namespace test {
struct NemotronBoltTestAccess;
}

inline constexpr size_t kBoltVerifyRows = 8;

struct NemotronBoltPrefixCache {
  struct Entry : PrefixSnapshotCache::Entry {
    std::vector<int32_t> tokens;
    std::vector<std::vector<Range>> ranges;
    size_t aux_start{};
    int32_t state_row{};
    int32_t anchor{};
  };

  explicit NemotronBoltPrefixCache(std::shared_ptr<PrefixSnapshotCache> storage)
      : storage_{std::move(storage)} {}
  std::shared_ptr<Entry> Find(std::span<const int32_t> tokens);
  void Capture(std::span<const int32_t> tokens, std::span<Tensor* const> caches,
               int32_t state_row, int32_t anchor, Tensor& aux, size_t aux_start, size_t aux_row_bytes);
  void Restore(const Entry& entry, std::span<Tensor* const> caches, Tensor& aux,
               size_t aux_start, size_t aux_row_bytes);
  const PrefixCacheMetrics& Metrics() const { return metrics_; }

 private:
  void Prune();
  std::shared_ptr<PrefixSnapshotCache> storage_;
  std::vector<std::shared_ptr<Entry>> entries_;
  PrefixCacheMetrics metrics_;
};

void ValidateNemotronBoltSearch(const GeneratorParams& params);
size_t NemotronBoltAcceptedPrefix(std::span<const int32_t> drafts,
                                  std::span<const int64_t> predictions);
std::vector<int32_t> NemotronBoltRoundTokens(std::span<const int32_t> drafts,
                                             std::span<const int64_t> predictions,
                                             size_t remaining,
                                             std::span<const int> eos);

struct NemotronBoltModel : Model {
  NemotronBoltModel(std::unique_ptr<Config> config, OrtEnv& env);
  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_;
  std::shared_ptr<Dflash2Model> drafter_model_;
  size_t capacity_{};
  size_t aux_width_{};
  // Target and drafter CUDA graph sessions, and their shared snapshot index, have one host owner at a time.
  mutable std::mutex execution_mutex_;
  std::unique_ptr<NemotronBoltPrefixCache> prefix_cache_;
};

// The target owns dense attention K/V and eight recurrent-state rows. Only the drafter is paged.
// The first base keeps the execution lock alive through member and State destruction.
struct NemotronBoltState : private std::unique_lock<std::mutex>, State {
  NemotronBoltState(const NemotronBoltModel& model, const GeneratorParams& params);
  ~NemotronBoltState() override;
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override;
  void RewindTo(size_t length) override;
  void SetExtraInputs(const std::vector<ExtraInput>& inputs) override;
  void SetActiveAdapter(Adapters* adapters, const std::string& adapter_name) override;
  void Prefill(std::span<const int32_t> tokens);
  int32_t NextToken(size_t public_length);
  SpeculativeStats Stats() const;
  size_t CachedPromptTokens() const { return cached_prompt_tokens_; }

 private:
  friend struct test::NemotronBoltTestAccess;
  Tensor& AddTensor(const std::string& name, bool input, DeviceInterface* device,
                    ONNXTensorElementDataType type, const std::vector<int64_t>& shape);
  std::vector<int64_t> Verify(std::span<const int32_t> tokens, bool want_predictions = true);
  void Propose(size_t context_rows, size_t first_position, int32_t anchor, Tensor* aux_override = nullptr);
  void Reset();

  const NemotronBoltModel& bolt_;
  std::map<std::string, std::unique_ptr<Tensor>> tensors_;
  std::vector<std::string> input_storage_, output_storage_;
  std::vector<Tensor*> caches_;
  DeviceSpan<int32_t> verify_inputs_span_;
  DeviceSpan<int64_t> pred_ids_span_;
  std::unique_ptr<Dflash2Drafter> drafter_;
  std::unique_ptr<Tensor> aux_view_;
  std::unique_ptr<Tensor> prefill_aux_;
  std::unique_ptr<Request> request_;
  size_t length_{};
  int32_t state_row_{};
  int32_t anchor_{};
  std::vector<int32_t> drafts_;
  std::deque<int32_t> pending_;
  bool prefilled_{};
  size_t cached_prompt_tokens_{};
  SpeculativeStats stats_;
};

std::unique_ptr<DecodingStrategy> MakeNemotronBoltStrategy(Generator& generator);
std::unique_ptr<Search> CreateNemotronBoltSearch(const GeneratorParams& params);

}  // namespace Generators
