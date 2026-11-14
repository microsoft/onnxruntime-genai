// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"
#include "dflash2_drafter.h"
#include "engine/executor_result.h"
#include <array>
#include <deque>
#include <mutex>
#include <optional>

namespace Generators {

namespace test {
struct NemotronBoltTestAccess;
}

inline constexpr size_t kBoltVerifyRows = 8;
inline constexpr size_t kBoltDefaultPrefillChunkRows = 4096;
inline constexpr size_t kBoltPagedBlockSize = 256;
inline constexpr size_t kBoltMaxBatchSize = 64;

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

struct NemotronBoltEngineStep {
  std::array<int32_t, kMaxGeneratedTokensPerStep> tokens{};
  std::array<int32_t, kBoltVerifyRows - 1> drafts{};
  size_t count{};
  size_t prompt_tokens_processed{};
  bool prompt_complete{};
  bool speculative_round{};
  size_t verify_forward_passes{};
  SpeculativeRowMetrics speculation{};
};

struct NemotronBoltSlotTask {
  size_t slot{};
  std::span<const int32_t> full_prompt{};
  size_t prefill_offset{};
  size_t prefill_tokens{};
  size_t remaining_budget{};
  bool is_decode{};
};

struct NemotronBoltModel : Model {
  NemotronBoltModel(std::unique_ptr<Config> config, OrtEnv& env);
  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_;
  std::shared_ptr<Dflash2Model> drafter_model_;
  mutable std::mutex execution_mutex_;
  size_t capacity_{};
  size_t paged_num_blocks_{};
  size_t paged_block_size_{kBoltPagedBlockSize};
  size_t engine_max_batch_size_{1};
  std::optional<size_t> engine_max_scheduled_tokens_;
  size_t aux_width_{};
  size_t target_rows_{kBoltVerifyRows};
  bool paged_attention_{};
  bool batched_recurrent_{};
  bool single_token_decode_{};
  bool dynamic_prefill_{};
  bool has_logit_rows_{};
  bool has_aux_rows_{};
};

// The target owns either paged or dense attention K/V plus eight recurrent-state rows.
struct NemotronBoltState : State {
  NemotronBoltState(const NemotronBoltModel& model, const GeneratorParams& params);
  ~NemotronBoltState() override;
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override;
  void RewindTo(size_t length) override;
  void SetExtraInputs(const std::vector<ExtraInput>& inputs) override;
  void SetActiveAdapter(Adapters* adapters, const std::string& adapter_name) override;
  void Prefill(std::span<const int32_t> tokens);
  int32_t NextToken(size_t public_length);
  size_t PrefillChunkRows() const noexcept { return prefill_chunk_rows_; }
  SpeculativeStats Stats() const;

 private:
  friend struct test::NemotronBoltTestAccess;
  Tensor& AddTensor(const std::string& name, bool input, DeviceInterface* device,
                    ONNXTensorElementDataType type, const std::vector<int64_t>& shape);
  std::vector<int64_t> Verify(std::span<const int32_t> tokens, bool want_predictions = true);
  std::vector<int64_t> VerifyBlock(std::span<const int32_t> tokens, bool want_predictions);
  int64_t PrefillChunk(std::span<const int32_t> tokens, size_t pos, size_t draft_start, bool want_prediction);
  static std::unique_ptr<Tensor> SaveVerifyPrefix(Tensor& tensor, size_t prefix_rows);
  static void RestoreVerifyPrefix(Tensor& tensor, Tensor& prefix);
  void Propose(size_t context_rows, size_t first_position, int32_t anchor, Tensor* aux_override = nullptr);
  bool IsEosToken(int32_t token) const;
  void Reset();

  const NemotronBoltModel& bolt_;
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
  size_t input_idx_attn_meta_{};
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
  size_t prefill_chunk_rows_{kBoltVerifyRows};
  size_t verified_rows_{};
  int32_t state_row_{};
  int32_t anchor_{};
  std::vector<int32_t> drafts_;
  std::deque<int32_t> pending_;
  bool has_drafts_{};
  bool prefilled_{};
  SpeculativeStats stats_;
};

// Multi-slot continuous-batching execution state over shared PagedAttention blocks, banked
// recurrent state [max_slots, 8, ...], and a shared batched DFlash2 drafter.
struct NemotronBoltBatchState : State {
  NemotronBoltBatchState(const NemotronBoltModel& model, const GeneratorParams& params, size_t max_slots);
  ~NemotronBoltBatchState() override;
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override;

  size_t MaxSlots() const noexcept { return max_slots_; }
  size_t PrefillChunkRows() const noexcept { return prefill_chunk_rows_; }
  size_t FreeBlockCount() const noexcept { return free_blocks_.size(); }
  size_t BlocksNeededForSlot(size_t slot, size_t total_seq_len) const;
  void ResetSlot(size_t slot);
  void StepBatch(std::span<const NemotronBoltSlotTask> tasks,
                 std::span<NemotronBoltEngineStep> results);
  SpeculativeStats Stats() const noexcept { return stats_; }

 private:
  struct BatchVerifyBinding {
    std::vector<std::unique_ptr<OrtValue>> owned_inputs;
    std::vector<std::unique_ptr<OrtValue>> owned_outputs;
    std::vector<OrtValue*> inputs;
    std::vector<OrtValue*> outputs;
    std::unique_ptr<Tensor> aux_view;
  };

  struct SlotContext {
    std::unique_ptr<Request> drafter_key;
    std::unique_ptr<Tensor> prefill_aux;
    size_t length{};
    int32_t state_row{};
    int32_t anchor{};
    std::vector<int32_t> drafts;
    bool has_drafts{};
    bool prefilled{};
    bool dirty{};
  };

  Tensor& AddBaseTensor(const std::string& name, bool input, DeviceInterface* device,
                        ONNXTensorElementDataType type, const std::vector<int64_t>& shape);
  void EnsureSlotBlocks(size_t slot, size_t total_seq_len);
  bool IsEosToken(int32_t token) const;
  void ResetAll();
  void StepPureDecode(std::span<const NemotronBoltSlotTask> tasks,
                      std::span<NemotronBoltEngineStep> results);
  void StepMixedOrPrefill(std::span<const NemotronBoltSlotTask> tasks,
                          std::span<NemotronBoltEngineStep> results);

  const NemotronBoltModel& bolt_;
  size_t max_slots_{1};
  size_t max_blocks_per_seq_{1};
  size_t prefill_chunk_rows_{kBoltVerifyRows};
  std::map<std::string, std::unique_ptr<Tensor>> tensors_;
  std::vector<std::string> input_storage_, output_storage_;
  std::vector<Tensor*> kv_caches_;
  std::vector<Tensor*> recurrent_caches_;
  size_t input_idx_input_ids_{}, input_idx_past_seqlens_{}, input_idx_cu_seqlens_{};
  size_t input_idx_state_row_{}, input_idx_state_slots_{}, input_idx_block_table_{};
  size_t input_idx_logit_rows_{}, input_idx_aux_rows_{}, input_idx_attn_meta_{};
  size_t output_idx_logits_{}, output_idx_pred_ids_{}, output_idx_aux_{};

  DeviceSpan<int64_t> decode_ids_span_;
  DeviceSpan<int32_t> decode_meta_span_;
  DeviceSpan<int32_t> decode_block_table_span_;
  DeviceSpan<int64_t> decode_pred_span_;
  std::unique_ptr<Tensor> decode_row_indices_;
  std::unique_ptr<Tensor> decode_attn_meta_;
  std::unique_ptr<Tensor> decode_logits_;
  std::unique_ptr<Tensor> decode_pred_ids_;
  std::unique_ptr<Tensor> decode_aux_;
  std::vector<BatchVerifyBinding> batch_bindings_;  // indexed by active batch size b in [1..max_slots_]
  std::vector<size_t> last_decode_slots_;
  std::vector<size_t> last_decode_block_counts_;

  DeviceSpan<int64_t> mixed_ids_span_;
  DeviceSpan<int32_t> mixed_meta_span_;
  DeviceSpan<int32_t> mixed_block_table_span_;
  DeviceSpan<int64_t> mixed_logit_rows_span_;
  DeviceSpan<int64_t> mixed_aux_rows_span_;
  std::unique_ptr<Tensor> mixed_propose_aux_;

  std::vector<int32_t> free_blocks_;
  std::vector<std::vector<int32_t>> slot_blocks_;
  std::vector<SlotContext> slots_;
  std::unique_ptr<Dflash2Drafter> drafter_;
  SpeculativeStats stats_;
};

std::unique_ptr<DecodingStrategy> MakeNemotronBoltStrategy(Generator& generator);
std::unique_ptr<Search> CreateNemotronBoltSearch(const GeneratorParams& params);

}  // namespace Generators
