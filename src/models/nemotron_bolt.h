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
inline constexpr size_t kBoltPrefillRows16 = 16;

struct NemotronBoltTargetRowContract {
  bool dynamic_r{};
  size_t aux_width{};
};

struct NemotronBoltPrefillStep {
  size_t pos{};
  size_t valid_rows{};
  size_t profile_rows{};
  int32_t seqlens_k{};
  int32_t state_row_in{};
  int32_t state_row_out{};
  size_t length_after{};
};

void ValidateNemotronBoltSearch(const GeneratorParams& params);
NemotronBoltTargetRowContract ValidateNemotronBoltTargetRowContract(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<std::string>& input_ids_symbolic,
    const std::vector<int64_t>& pred_ids_shape,
    const std::vector<std::string>& pred_ids_symbolic,
    const std::vector<int64_t>& logits_shape,
    const std::vector<std::string>& logits_symbolic,
    const std::vector<int64_t>& aux_shape,
    const std::vector<std::string>& aux_symbolic,
    int64_t expected_vocab_size);
std::vector<NemotronBoltPrefillStep> PlanNemotronBoltPrefillChunks(
    size_t prompt_rows, size_t capacity, bool allow_r16,
    size_t initial_length = 0, int32_t initial_state_row = 0,
    size_t prefix_r8_rows = 0);
size_t NemotronBoltPrefillAuxRowsToRetain(size_t prompt_rows, size_t sliding_window,
                                          size_t block_rows);
void CopyNemotronBoltPrefillAuxRows(Tensor& destination, Tensor& source,
                                    size_t chunk_position, size_t valid_rows,
                                    size_t physical_prefix_rows = 0);
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
  bool supports_r16_prefill_{};
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
  Tensor& AddTensor16(const std::string& name, DeviceInterface* device,
                      ONNXTensorElementDataType type, const std::vector<int64_t>& shape);
  std::vector<int64_t> Verify(std::span<const int32_t> tokens, bool want_predictions = true);
  std::vector<int64_t> PrefillChunk16(std::span<const int32_t> tokens, bool want_predictions = true);
  void Propose(size_t context_rows, size_t first_position, int32_t anchor, Tensor* aux_override = nullptr);
  void Reset();

  const NemotronBoltModel& bolt_;
  std::map<std::string, std::unique_ptr<Tensor>> tensors_;
  std::map<std::string, std::unique_ptr<Tensor>> tensors_16_;
  std::vector<std::string> input_storage_, output_storage_;
  std::vector<OrtValue*> inputs_8_, outputs_8_;
  std::vector<OrtValue*> inputs_16_, outputs_16_;
  std::vector<Tensor*> caches_;
  DeviceSpan<int32_t> verify_inputs_span_;
  DeviceSpan<int64_t> pred_ids_span_;
  DeviceSpan<int64_t> pred_ids_16_span_;
  std::unique_ptr<Dflash2Drafter> drafter_;
  std::unique_ptr<Tensor> aux_view_;
  std::unique_ptr<Tensor> aux_view_16_;
  std::unique_ptr<Tensor> prefill_aux_;
  std::unique_ptr<Request> request_;
  size_t length_{};
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
