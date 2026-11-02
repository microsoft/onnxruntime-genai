// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Generator behavior for step-driven models (RNNT, TDT, streaming encoder-decoder ASR), which have no
// search. The fake model and state stand in for the real ones, whose test models are not checked in.
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"
#include "generator/generators.h"
#include "models/model.h"
#include "models/transducer_state.h"

#ifndef MODEL_PATH
#define MODEL_PATH "../../test/models/"
#endif

namespace Generators {
namespace {

struct FakeStepState : TransducerState {
  using TransducerState::TransducerState;

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override { return {}; }
  void StepToken() override { chunk_done_ = true; }
};

struct FakeStepModel : Model {
  using Model::Model;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams& params) const override {
    return std::make_unique<FakeStepState>(params, *this);
  }
};

TEST(StepDrivenModelTest, AppendTokensAsksForSetInputs) {
  for (const std::string type : {"nemotron_speech", "parakeet_tdt", "streaming_enc_dec_asr"}) {
    SCOPED_TRACE(type);
    auto config = std::make_unique<Config>(fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"},
                                           R"({"model": {"type": ")" + type + R"("}})");
    auto model = std::make_shared<FakeStepModel>(std::move(config));
    auto params = std::make_shared<GeneratorParams>(*model);
    Generator generator{*model, *params};

    const std::vector<int32_t> ids{1, 2, 3};
    auto expect_error = [&](auto&& append) {
      try {
        append();
        FAIL() << "AppendTokens was accepted";
      } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string{e.what()}.find("AppendTokens is not supported for " + type), std::string::npos) << e.what();
        EXPECT_NE(std::string{e.what()}.find("SetInputs"), std::string::npos) << e.what();
      }
    };
    expect_error([&] { generator.AppendTokens(cpu_span<const int32_t>{ids}); });
    auto device_ids = model->p_device_inputs_->Allocate<int32_t>(ids.size());
    expect_error([&] { generator.AppendTokens(device_ids); });
  }
}

}  // namespace
}  // namespace Generators
