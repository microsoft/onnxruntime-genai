// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"
#include "dflash2_drafter.h"
#include "state-checkpoint.h"
#include "engine/native-device-memory-budget.h"
#include <deque>
#include <mutex>

namespace Generators {

namespace test {
struct NemotronBoltTestAccess;
}

inline constexpr size_t kBoltVerifyRows = 8;

struct NemotronBoltPrefillPlan {
  NemotronBoltPrefillPlan(size_t token_count, size_t context_window, bool checkpointing);
  size_t NextRows(size_t position, size_t max_rows) const;

  size_t token_count;
  size_t checkpoint;
  size_t draft_start;
  size_t auxiliary_start;
};

void ValidateNemotronBoltSearch(const GeneratorParams& params);
void ValidateNemotronBoltGraphCapture(const Config::SessionOptions& options, bool speculative);
std::unique_ptr<Config> CreateNemotronBoltDrafterConfig(const Config& config);
size_t NemotronBoltAcceptedPrefix(std::span<const int32_t> drafts,
                                  std::span<const int64_t> predictions);
std::vector<int32_t> NemotronBoltRoundTokens(std::span<const int32_t> drafts,
                                             std::span<const int64_t> predictions,
                                             size_t remaining,
                                             std::span<const int> eos);

struct NemotronBoltState;

struct NemotronBoltModel : Model {
  NemotronBoltModel(std::unique_ptr<Config> config, OrtEnv& env);
  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams& params) const override;
  std::unique_ptr<NemotronBoltState> CreateNativeState(const GeneratorParams& params) const;

  std::unique_ptr<OrtSession> session_;
  std::shared_ptr<Dflash2Model> drafter_model_;
  size_t capacity_{};
  size_t aux_width_{};
  // Target and drafter CUDA graph sessions, and their shared snapshot index, have one host owner at a time.
  mutable std::mutex execution_mutex_;
  NativeDeviceMemoryBudget memory_budget_;
  std::unique_ptr<StateCheckpointCache> prefix_cache_;
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
  void Prefill(std::span<const int32_t> tokens, PrefixCacheMetrics* cache_events = nullptr);
  int32_t NextToken(size_t public_length);
  SpeculativeStats Stats() const;
  size_t CachedPromptTokens() const { return cached_prompt_tokens_; }

 private:
  friend struct test::NemotronBoltTestAccess;

  struct PrefillResult {
    int32_t state_row;
    int32_t prediction;
  };

  Tensor& AddTensor(const std::string& name, bool input, DeviceInterface* device,
                    ONNXTensorElementDataType type, const std::vector<int64_t>& shape);
  void BindCheckpointState();
  void RestorePrefix(std::span<const int32_t> tokens);
  PrefillResult RunPrefillTarget(std::span<const int32_t> tokens, size_t auxiliary_start,
                                 bool want_prediction);
  std::vector<int64_t> Verify(std::span<const int32_t> tokens, bool want_predictions = true);
  void Propose(size_t context_rows, size_t first_position, int32_t anchor, Tensor* aux_override = nullptr);
  void Reset();

  const NemotronBoltModel& bolt_;
  StateCheckpointDescription state_description_;
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
  size_t aux_start_{};
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
