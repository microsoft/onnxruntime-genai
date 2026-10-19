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
};

// The target owns dense attention K/V and eight recurrent-state rows. Only the drafter is paged.
struct NemotronBoltState : State {
  NemotronBoltState(const NemotronBoltModel& model, const GeneratorParams& params);
  ~NemotronBoltState() override;
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override;
  void RewindTo(size_t length) override;
  void SetExtraInputs(const std::vector<ExtraInput>& inputs) override;
  void Prefill(std::span<const int32_t> tokens);
  int32_t NextToken(size_t public_length);
  SpeculativeStats Stats() const;

 private:
  friend struct test::NemotronBoltTestAccess;
  Tensor& AddTensor(const std::string& name, bool input, DeviceInterface* device,
                    ONNXTensorElementDataType type, const std::vector<int64_t>& shape);
  std::vector<int64_t> Verify(std::span<const int32_t> tokens);
  void Propose(size_t context_rows, size_t first_position, int32_t anchor);
  void Reset();

  const NemotronBoltModel& bolt_;
  std::map<std::string, std::unique_ptr<Tensor>> tensors_;
  std::vector<std::string> input_storage_, output_storage_;
  std::vector<Tensor*> caches_;
  std::unique_ptr<OrtIoBinding> binding_;
  std::unique_ptr<Dflash2Drafter> drafter_;
  std::unique_ptr<Tensor> aux_view_;
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
