// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"
#include "dflash2_drafter.h"
#include <deque>

namespace Generators {

namespace test {
struct NemotronBoltTestAccess;
}

inline constexpr size_t kBoltVerifyRows = 8;
inline constexpr size_t kBoltDefaultPrefillChunkRows = 4096;

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

struct NemotronBoltModel : Model {
  NemotronBoltModel(std::unique_ptr<Config> config, OrtEnv& env);
  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_;
  std::shared_ptr<Dflash2Model> drafter_model_;
  size_t capacity_{};
  size_t aux_width_{};
  size_t target_rows_{kBoltVerifyRows};
  size_t prefill_chunk_rows_{kBoltDefaultPrefillChunkRows};
  bool single_token_decode_{};
  bool dynamic_prefill_{};
  bool has_logit_rows_{};
  bool has_aux_rows_{};
};

// The target owns dense attention K/V and eight recurrent-state rows. Only the drafter is paged.
struct NemotronBoltState : State {
  NemotronBoltState(const NemotronBoltModel& model, const GeneratorParams& params);
  ~NemotronBoltState() override;
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override;
  void RewindTo(size_t length) override;
  void SetExtraInputs(const std::vector<ExtraInput>& inputs) override;
  void SetActiveAdapter(Adapters* adapters, const std::string& adapter_name) override;
  void Prefill(std::span<const int32_t> tokens);
  int32_t NextToken(size_t public_length);
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
  size_t verified_rows_{};
  int32_t state_row_{};
  int32_t anchor_{};
  std::vector<int32_t> drafts_;
  std::deque<int32_t> pending_;
  bool prefilled_{};
  SpeculativeStats stats_;
};

std::unique_ptr<DecodingStrategy> MakeNemotronBoltStrategy(Generator& generator);
std::unique_ptr<Search> CreateNemotronBoltSearch(const GeneratorParams& params);

}  // namespace Generators
