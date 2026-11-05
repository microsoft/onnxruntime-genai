// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "models/engram.h"

namespace Generators::test {
namespace {

TEST(EngramStateReview, CacheKeyIncludesEveryCurrentBatchTokenAfterReset) {
  for (size_t cache_capacity : {size_t{0}, size_t{4}}) {
    auto model = LoadDummyDecoderModel();
    auto& config = model->config_->model.engram;
    config.inputs.input_ids = "engram_input_ids";
    config.inputs.past_tokens = "engram_past_tokens";
    config.outputs.embeddings = "engram_embeddings";
    config.outputs.present_tokens = "engram_present_tokens";
    config.cache_capacity = cache_capacity;
    model->config_->model.decoder.ple_token_pad_id = 0;
    auto session = model->CreateSession(
        GetOrtEnv(), "../synthetic-rotary-history/engram.onnx", model->session_options_.get());
    model->session_info_.Add(*session);
    auto params = CreateGeneratorParams(*model);
    params->search.batch_size = 2;
    EngramState state{*model, *session, *params};
    auto tokens = model->p_device_->Allocate<int32_t>(2);
    tokens.CpuSpan()[0] = 3;
    tokens.CpuSpan()[1] = 4;
    tokens.CopyCpuToDevice();
    state.UpdateInputsOutputs(tokens);
    state.Run(1, tokens);
    const auto* embeddings = state.GetOutput("engram_embeddings")->GetTensorData<float>();
    EXPECT_EQ(embeddings[0], 3.0f);
    EXPECT_EQ(embeddings[1], 4.0f);
    state.RewindTo(0);
    tokens.CpuSpan()[1] = 5;
    tokens.CopyCpuToDevice();
    state.UpdateInputsOutputs(tokens);
    state.Run(1, tokens);
    embeddings = state.GetOutput("engram_embeddings")->GetTensorData<float>();
    EXPECT_EQ(embeddings[0], 3.0f);
    EXPECT_EQ(embeddings[1], 5.0f);
  }
}

}  // namespace
}  // namespace Generators::test
