// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"
#include "dflash2_drafter.h"
#include "state-checkpoint.h"
#include "engine/native-device-memory-budget.h"
#include <deque>
#include <mutex>
#include <optional>

namespace Generators {

namespace test {
struct NemotronBoltTestAccess;
}

inline constexpr size_t kBoltVerifyRows = 8;
inline constexpr size_t kBoltDefaultPrefillChunkRows = 4096;

struct NemotronBoltPrefillPlan {
  NemotronBoltPrefillPlan(size_t token_count, size_t context_window, bool checkpointing);
  size_t NextRows(size_t position, size_t max_rows) const;

  size_t token_count;
  size_t checkpoint;
  size_t draft_start;
  size_t auxiliary_start;
};

// Rows per dynamic-prefill target pass from search.chunk_size (genai_config.json or set_search_options).
// Values <= 8 select the fixed R8 path; larger values are capped to the KV capacity minus the
// verification scratch.
size_t NemotronBoltPrefillChunkRows(std::optional<size_t> configured, size_t capacity);
size_t NemotronBoltVerifyPrefixRows(size_t position);
size_t NemotronBoltVerifyUsableRows(size_t position, size_t capacity, int max_length);
void ValidateNemotronBoltSearch(const GeneratorParams& params);
void ValidateNemotronBoltGraphCapture(const Config::SessionOptions& options, bool speculative);
std::unique_ptr<Config> CreateNemotronBoltDrafterConfig(const Config& config);
struct NemotronBoltRoundStats {
  size_t accepted{};
  size_t evaluated{};
  bool correction{};
  bool bonus{};
  bool full{};  // All seven proposed drafts matched.
};
NemotronBoltRoundStats GetNemotronBoltRoundStats(std::span<const int32_t> drafts,
                                                 std::span<const int32_t> emitted_tokens);
size_t NemotronBoltAcceptedPrefix(std::span<const int32_t> drafts,
                                  std::span<const int64_t> predictions,
                                  size_t max_accepted = kBoltVerifyRows - 1);
std::vector<int32_t> NemotronBoltRoundTokens(std::span<const int32_t> drafts,
                                             std::span<const int64_t> predictions,
                                             size_t remaining,
                                             std::span<const int> eos,
                                             size_t usable_rows = kBoltVerifyRows);

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
  size_t target_rows_{kBoltVerifyRows};
  bool single_token_decode_{};
  bool dynamic_prefill_{};
  bool has_logit_rows_{};
  bool has_aux_rows_{};
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
  std::vector<int64_t> VerifyBlock(std::span<const int32_t> tokens, bool want_predictions);
  int64_t RunDynamicPrefillTarget(std::span<const int32_t> tokens, size_t auxiliary_start, bool want_prediction);
  static std::unique_ptr<Tensor> SaveVerifyPrefix(Tensor& tensor, size_t prefix_rows);
  static void RestoreVerifyPrefix(Tensor& tensor, Tensor& prefix);
  void Propose(size_t context_rows, size_t first_position, int32_t anchor, Tensor* aux_override = nullptr);
  void Reset();

  const NemotronBoltModel& bolt_;
  StateCheckpointDescription state_description_;
  // Primary graph bindings (persistent caches, scalars, and R=target_rows_ I/O buffers).
  std::map<std::string, std::unique_ptr<Tensor>> tensors_;
  // Zero-copy 1-row OrtValue views into `tensors_` for opt-in `single_token_decode_` R=1 replay.
  std::map<std::string, std::unique_ptr<Tensor>> single_token_tensors_;
  std::vector<size_t> single_token_input_indices_, single_token_output_indices_;
  std::vector<std::string> input_storage_, output_storage_;
  std::vector<Tensor*> caches_;
  std::vector<Tensor*> kv_caches_;
  std::vector<Tensor*> recurrent_caches_;
  bool caches_zeroed_{};
  size_t touched_kv_rows_{};
  size_t input_idx_input_ids_{}, input_idx_logit_rows_{}, input_idx_aux_rows_{};
  size_t output_idx_logits_{}, output_idx_pred_ids_{}, output_idx_aux_{};
  DeviceSpan<int32_t> verify_inputs_span_;
  DeviceSpan<int64_t> pred_ids_span_;
  DeviceSpan<int64_t> prefill_ids_span_;
  DeviceSpan<int64_t> prefill_logit_rows_span_;
  DeviceSpan<int64_t> prefill_aux_rows_span_;
  std::unique_ptr<Tensor> prefill_logits_;
  std::unique_ptr<Tensor> prefill_pred_ids_;
  DeviceSpan<int64_t> prefill_pred_span_;
  std::unique_ptr<Dflash2Drafter> drafter_;
  std::unique_ptr<Tensor> aux_view_;
  std::unique_ptr<Tensor> prefill_aux_;
  std::unique_ptr<Request> request_;
  size_t length_{};
  size_t aux_start_{};
  size_t prefill_chunk_rows_{kBoltVerifyRows};
  size_t verified_rows_{};
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
