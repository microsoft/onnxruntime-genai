// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include "constrained_logits_processor.h"
#include "config_utils.h"
#include "generator/generators.h"
#include "models/model.h"
#include "models/io/logits.h"
#include "test_utils.h"

namespace Generators::test {
namespace {

class ForcedTokensProcessor final : public ConstrainedLogitsProcessor {
 public:
  explicit ForcedTokensProcessor(std::vector<int32_t> tokens) : tokens_{std::move(tokens)} {}
  void CommitTokens(std::span<int32_t>) override {}
  void ProcessLogits(DeviceSpan<float>) override {}
  bool AllowsOnlyTokens(size_t, std::span<const int>) override { return false; }
  void Reset() override {}
  std::vector<int32_t> GetFFTokens(size_t) override { return std::exchange(tokens_, {}); }
  std::unique_ptr<ConstrainedLogitsProcessor> Clone() const override {
    return std::make_unique<ForcedTokensProcessor>(*this);
  }

 private:
  std::vector<int32_t> tokens_;
};

TEST(ForcedTokenLogitsTest, SlidingWindowAlignmentExcludesOnlySyntheticPadding) {
  for (const auto* path : {MODEL_PATH "longrope-context-legacy", MODEL_PATH "longrope-context-legacy-runtime-indices"}) {
    for (const auto* alignment : {"left", "right"}) {
      for (bool slide_inputs : {false, true}) {
        for (int forced_length : {2, 3, 4, 6, 7, 8}) {
          for (int32_t final_token : {1, 2}) {
            SCOPED_TRACE(testing::Message() << "path=" << path << " alignment=" << alignment
                                            << " slide_inputs=" << slide_inputs << " forced_length=" << forced_length
                                            << " final_token=" << final_token);
            auto config = CreateConfig(GetOrtEnv(), path);
            config->model.pad_token_id = 1;
            config->model.eos_token_id = {0};
            if (slide_inputs) {
              auto& window = config->model.decoder.sliding_window.emplace();
              window.window_size = 4;
              window.alignment = alignment;
              window.slide_inputs = true;
              window.slide_key_value_cache = false;
            }
            auto model = CreateModel(GetOrtEnv(), std::move(config));
            auto params = std::make_shared<GeneratorParams>(*model);
            auto generator = std::make_unique<Generator>(*model, *params);
            generator->AppendTokens(std::vector<int32_t>{2, 2, 2, 2});
            auto forced_logits = model->p_device_->Allocate<float>(3);
            std::copy_n(std::array<float, 3>{0, 0, 10}.begin(), 3, forced_logits.CpuSpan().begin());
            forced_logits.CopyCpuToDevice();
            generator->SetLogits(forced_logits);
            generator->GenerateNextToken();
            std::vector<int32_t> forced_tokens(forced_length, 2);
            forced_tokens.back() = final_token;
            generator->guidance_logits_processor_ = std::make_unique<ForcedTokensProcessor>(forced_tokens);
            auto logits = generator->GetLogits().CopyDeviceToCpu();
            auto* cache = generator->state_->GetOutput("present.0.key");
            const auto count = cache->GetTensorTypeAndShapeInfo()->GetElementCount();
            const auto* values = cache->GetTensorData<float>();
            const size_t padding = slide_inputs && forced_length % 4 ? 4 - forced_length % 4 : 0;
            const size_t selected_length = count - (std::string_view{alignment} == "left" ? padding : 0);
            const float sum = std::accumulate(values, values + selected_length, 0.0f);
            EXPECT_FLOAT_EQ(logits[0], -sum);
            EXPECT_FLOAT_EQ(logits[1], sum);
            EXPECT_FLOAT_EQ(logits[2], 0);
          }
        }
      }
    }
  }
}

TEST(LastTokenIndicesTest, SelectsEachRowsLastNonpadAndRepeatsForBeams) {
  const std::array<int32_t, 15> tokens{0, 4, 5, 0, 0, 2, 0, 3, 0, 0, 1, 2, 3, 4, 5};
  EXPECT_EQ(LastTokenIndices(tokens, 3, 2, 5, 0), (std::vector<int32_t>{2, 2, 2, 2, 4, 4}));
  EXPECT_EQ(LastTokenIndices(std::array<int32_t, 1>{7}, 1, 1, 1, 0), (std::vector<int32_t>{0}));
}

TEST(ForcedTokenLogitsTest, ExplicitSelectionExtentSurvivesChunkingAndKeepsGeneratedPad) {
  for (const auto* path : {MODEL_PATH "longrope-context-legacy", MODEL_PATH "longrope-context-legacy-runtime-indices"}) {
    for (int chunk : {0, 1, 2, 3}) {
      SCOPED_TRACE(testing::Message() << "path=" << path << " chunk=" << chunk);
      auto config = CreateConfig(GetOrtEnv(), path);
      config->model.pad_token_id = 1;
      auto model = CreateModel(GetOrtEnv(), std::move(config));
      auto params = std::make_shared<GeneratorParams>(*model);
      params->search.chunk_size = chunk;
      auto generator = std::make_unique<Generator>(*model, *params);
      auto tokens = model->p_device_->Allocate<int32_t>(4);
      std::copy_n(std::array<int32_t, 4>{2, 2, 1, 1}.begin(), 4, tokens.CpuSpan().begin());
      tokens.CopyCpuToDevice();
      auto indices = model->p_device_->Allocate<int32_t>(1);
      indices.CpuSpan()[0] = 0;
      indices.CopyCpuToDevice();
      State::PromptInputScope selection_scope(*generator->state_, false, 3);
      auto selected_logits = generator->state_->Run(4, tokens, indices);
      auto logits = selected_logits.CopyDeviceToCpu();
      EXPECT_FLOAT_EQ(logits[0], -5);
      EXPECT_FLOAT_EQ(logits[1], 5);
      EXPECT_FLOAT_EQ(logits[2], 0);
    }
  }
}

TEST(LastTokenIndicesTest, HandlesChunkLocalPaddingIndependentlyOfPastAndStaticMask) {
  std::vector<int32_t> tokens(968, 9);
  std::fill(tokens.end() - 16, tokens.end(), 0);
  EXPECT_EQ(LastTokenIndices(tokens, 1, 1, tokens.size(), 0), (std::vector<int32_t>{951}));
}

TEST(LastTokenIndicesTest, RejectsEmptyRowsAndInvalidDimensions) {
  const std::array<int32_t, 2> tokens{0, 0};
  EXPECT_THROW(LastTokenIndices(tokens, 1, 1, 2, 0), std::invalid_argument);
  EXPECT_THROW(LastTokenIndices(tokens, 0, 1, 2, 0), std::invalid_argument);
  EXPECT_THROW(LastTokenIndices(tokens, 1, 0, 2, 0), std::invalid_argument);
  EXPECT_THROW(LastTokenIndices(tokens, 1, 1, 0, 0), std::invalid_argument);
  EXPECT_THROW(LastTokenIndices(tokens, 2, 1, 2, 0), std::invalid_argument);
  EXPECT_THROW(LastTokenIndices(tokens, 1, 1, std::numeric_limits<size_t>::max(), 0), std::invalid_argument);
  EXPECT_THROW(LastTokenIndices(tokens, 2, std::numeric_limits<size_t>::max(), 1, 0), std::invalid_argument);
}

TEST(ForcedTokenLogitsTest, DeviceAppendRejectsPaddingBeforeMutatingTheSequence) {
#if USE_CUDA
  const std::array providers{false, true};
#else
  const std::array providers{false};
#endif
  for (bool use_cuda : providers) {
    for (const auto* path : {MODEL_PATH "longrope-context", MODEL_PATH "longrope-context-runtime-indices"}) {
      SCOPED_TRACE(testing::Message() << "path=" << path << " cuda=" << use_cuda);
      auto config = CreateConfig(GetOrtEnv(), path);
      config->model.pad_token_id = 0;
      if (use_cuda)
        OverlayConfig(*config, R"({"model":{"decoder":{"session_options":{"provider_options":[{"cuda":{}}]}}}})");
      auto model = CreateModel(GetOrtEnv(), std::move(config));
      auto params = std::make_shared<GeneratorParams>(*model);
      params->search.chunk_size = 4;
      auto generator = std::make_unique<Generator>(*model, *params);
      generator->AppendTokens(std::vector<int32_t>(8, 1));
      auto tokens = model->p_device_->Allocate<int32_t>(4);
      std::fill(tokens.CpuSpan().begin(), tokens.CpuSpan().end(), 0);
      tokens.CopyCpuToDevice();
      EXPECT_THROW(generator->AppendTokens(tokens), std::invalid_argument);
      ASSERT_EQ(generator->GetSequence(0).size(), 8u);
      auto logits = generator->GetLogits().CopyDeviceToCpu();
      EXPECT_FLOAT_EQ(logits[0], 8);
      EXPECT_FLOAT_EQ(logits[1], -8);
      auto* cache = generator->state_->GetOutput("present.0.key");
      EXPECT_EQ(cache->GetTensorTypeAndShapeInfo()->GetElementCount(), 8u);
      tokens.CpuSpan()[0] = 1;
      tokens.CopyCpuToDevice();
      EXPECT_NO_THROW(generator->AppendTokens(tokens));
      EXPECT_EQ(generator->GetSequence(0).size(), 12u);
    }
  }
}

struct LogitsTestState : State {
  using State::State;

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override { return {}; }
};

class RuntimeLastTokenLogitsTest : public testing::Test {
 protected:
  void Initialize(const std::string& filename = "pruned.onnx", bool capture = false, int beams = 1) {
    model_ = CreateModel(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    model_->config_->model.vocab_size = 4;
    session_ = model_->CreateSession(
        GetOrtEnv(), "../last-token-logits/" + filename, model_->session_options_.get());
    model_->session_info_ = SessionInfo{};
    model_->session_info_.Add(*session_);
    params_ = std::make_shared<GeneratorParams>(*model_);
    params_->search.batch_size = 3;
    params_->search.num_beams = beams;
    params_->use_graph_capture = capture;
    state_ = std::make_unique<LogitsTestState>(*params_, *model_);
    state_->input_names_.push_back("input_ids");
    state_->inputs_.push_back(nullptr);
    logits_ = std::make_unique<Logits>(*state_);
  }

  DeviceSpan<int32_t> SetTokens(const std::vector<int32_t>& tokens, int sequence_length) {
    auto next = model_->p_device_->Allocate<int32_t>(tokens.size());
    std::copy(tokens.begin(), tokens.end(), next.CpuSpan().begin());
    next.CopyCpuToDevice();
    const int beams = params_->search.num_beams;
    input_ids_ = OrtValue::CreateTensor<int32_t>(
        model_->allocator_cpu_, std::array<int64_t, 2>{params_->BatchBeamSize(), sequence_length});
    auto* expanded = input_ids_->GetTensorMutableData<int32_t>();
    for (int batch = 0; batch < params_->search.batch_size; ++batch) {
      for (int beam = 0; beam < beams; ++beam)
        std::copy_n(tokens.data() + batch * sequence_length, sequence_length,
                    expanded + (batch * beams + beam) * sequence_length);
    }
    state_->inputs_[0] = input_ids_.get();
    return next;
  }

  void RunChunk(const std::vector<int32_t>& tokens, int sequence_length,
                const std::array<int32_t, 3>& indices, bool expanded = false, bool is_prompt = true) {
    State::PromptInputScope prompt_scope(*state_, is_prompt);
    auto next = SetTokens(tokens, sequence_length);
    if (expanded)
      next = WrapTensor<int32_t>(*model_->p_device_inputs_, *input_ids_);
    logits_->Update(next, sequence_length);
    const int beams = params_->search.num_beams;
    if (model_->session_info_.HasInput("last_token_indices")) {
      ASSERT_EQ(state_->input_names_.size(), 2u);
      EXPECT_STREQ(state_->input_names_[1], "last_token_indices");
      const auto* tensor = state_->inputs_[1];
      EXPECT_EQ(tensor->GetTensorTypeAndShapeInfo()->GetShape(),
                (std::vector<int64_t>{params_->BatchBeamSize(), 1, 1}));
      const auto* values = tensor->GetTensorData<int32_t>();
      for (int batch = 0; batch < 3; ++batch) {
        for (int beam = 0; beam < beams; ++beam)
          EXPECT_EQ(values[batch * beams + beam], indices[batch]);
      }
    }
    session_->Run(nullptr, state_->input_names_.data(), state_->inputs_.data(), state_->inputs_.size(),
                  state_->output_names_.data(), state_->outputs_.data(), state_->outputs_.size());
    auto values = logits_->Get().CpuSpan();
    ASSERT_EQ(values.size(), params_->BatchBeamSize() * 4u);
    for (int batch = 0; batch < 3; ++batch) {
      for (int beam = 0; beam < beams; ++beam) {
        for (int column = 0; column < 4; ++column)
          EXPECT_EQ(values[(batch * beams + beam) * 4 + column],
                    tokens[batch * sequence_length + indices[batch]]);
      }
    }
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<OrtSession> session_;
  std::unique_ptr<LogitsTestState> state_;
  std::unique_ptr<Logits> logits_;
  std::unique_ptr<OrtValue> input_ids_;
};

class RuntimeLastTokenLogitsModesTest : public RuntimeLastTokenLogitsTest,
                                        public testing::WithParamInterface<std::tuple<bool, int>> {};

TEST_F(RuntimeLastTokenLogitsTest, BeamExpandedRowsKeepUnequalPromptLengths) {
  Initialize("pruned.onnx", false, 2);
  logits_->Add();
  RunChunk({4, 0, 0, 5, 6, 0, 7, 8, 9}, 3, {0, 1, 2}, true);
}

TEST_F(RuntimeLastTokenLogitsTest, RejectsInvalidBatchBeamDimensionsBeforeAllocation) {
  Initialize();
  for (const auto& [batch, beams] : {std::pair{0, 1}, std::pair{1, 0}, std::pair{50000, 50000}}) {
    params_->search.batch_size = batch;
    params_->search.num_beams = beams;
    EXPECT_THROW(Logits temporary(*state_), std::invalid_argument);
  }
}

#if USE_CUDA
TEST_F(RuntimeLastTokenLogitsTest, DeviceTokensDoNotReadAnUnpopulatedHostMirror) {
  Initialize();
  logits_->Add();
  auto tokens = SetTokens({4, 0, 0, 5, 6, 0, 7, 8, 9}, 3);
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", {}, {});
  auto cuda_model = CreateModel(GetOrtEnv(), std::move(config));
  auto* cuda = cuda_model->p_device_;
  ASSERT_NE(cuda, nullptr);
  auto device_tokens = cuda->Allocate<int32_t>(tokens.size());
  device_tokens.CopyFrom(tokens);
  auto wrapped = cuda->WrapMemory<int32_t>(device_tokens.Span());
  std::fill(wrapped.CpuSpan().begin(), wrapped.CpuSpan().end(), 0);
  {
    State::PromptInputScope prompt_scope(*state_, true);
    ASSERT_NO_THROW(logits_->Update(wrapped, 3));
  }
  const auto* indices = state_->inputs_[1]->GetTensorData<int32_t>();
  EXPECT_EQ((std::vector<int32_t>{indices, indices + 3}), (std::vector<int32_t>{0, 1, 2}));
  auto decode_tokens = cuda->WrapMemory<int32_t>(device_tokens.Span().first(3));
  auto* index_binding = state_->inputs_[1];
  logits_->Update(decode_tokens, 1);
  auto* output_binding = state_->outputs_[0];
  logits_->Update(decode_tokens, 1);
  EXPECT_EQ(decode_tokens.GetBuffer().p_cpu_, nullptr);
  EXPECT_EQ(state_->inputs_[1], index_binding);
  EXPECT_EQ(state_->outputs_[0], output_binding);
}

TEST_F(RuntimeLastTokenLogitsTest, SingletonPromptUsesDeviceValuesWithStaleHostMirror) {
  Initialize();
  logits_->Add();
  auto tokens = SetTokens({4, 5, 6}, 1);
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", {}, {});
  auto cuda_model = CreateModel(GetOrtEnv(), std::move(config));
  auto* cuda = cuda_model->p_device_;
  ASSERT_NE(cuda, nullptr);
  auto device_tokens = cuda->Allocate<int32_t>(tokens.size());
  device_tokens.CopyFrom(tokens);
  auto wrapped = cuda->WrapMemory<int32_t>(device_tokens.Span());
  std::fill(wrapped.CpuSpan().begin(), wrapped.CpuSpan().end(), 0);
  State::PromptInputScope prompt_scope(*state_, true);
  ASSERT_NO_THROW(logits_->Update(wrapped, 1));
  auto* index_binding = state_->inputs_[1];
  auto* output_binding = state_->outputs_[0];
  const auto* indices = index_binding->GetTensorData<int32_t>();
  EXPECT_EQ((std::vector<int32_t>{indices, indices + 3}), (std::vector<int32_t>{0, 0, 0}));

  tokens = SetTokens({4, 0, 6}, 1);
  device_tokens.CopyFrom(tokens);
  std::fill(wrapped.CpuSpan().begin(), wrapped.CpuSpan().end(), 7);
  EXPECT_THROW(logits_->Update(wrapped, 1), std::invalid_argument);
  EXPECT_EQ(state_->inputs_[1], index_binding);
  EXPECT_EQ(state_->outputs_[0], output_binding);
}
#endif

TEST_P(RuntimeLastTokenLogitsModesTest, BoundIndicesFollowPrefillContinuationAndDecode) {
  Initialize("pruned.onnx", std::get<0>(GetParam()), std::get<1>(GetParam()));
  logits_->Add();
  ASSERT_EQ(state_->inputs_.size(), 2u);
  auto* index_tensor = state_->inputs_[1];
  const auto* index_data = index_tensor->GetTensorData<int32_t>();
  RunChunk({0, 4, 5, 0, 0, 0, 0, 6, 7, 8, 9, 0, 10, 0, 0}, 5, {2, 4, 2});
  auto* output_data = state_->outputs_[0]->GetTensorMutableData<float>();
  RunChunk({11, 12, 13}, 1, {0, 0, 0});
  RunChunk({14, 0, 0, 0, 0, 15, 16, 0, 17, 18, 19, 20}, 4, {0, 2, 3});
  RunChunk({21, 22, 23}, 1, {0, 0, 0});
  RunChunk({24, 25, 26}, 1, {0, 0, 0});
  EXPECT_EQ(state_->inputs_[1], index_tensor);
  EXPECT_EQ(state_->inputs_[1]->GetTensorData<int32_t>(), index_data);
  EXPECT_EQ(state_->outputs_[0]->GetTensorMutableData<float>(), output_data);
}

TEST_P(RuntimeLastTokenLogitsModesTest, OneTokenPrefillInitializesEveryBoundIndex) {
  Initialize("pruned.onnx", std::get<0>(GetParam()), std::get<1>(GetParam()));
  logits_->Add();
  ASSERT_EQ(state_->inputs_.size(), 2u);
  std::fill_n(state_->inputs_[1]->GetTensorMutableData<int32_t>(), params_->BatchBeamSize(), -17);
  RunChunk({4, 5, 6}, 1, {0, 0, 0});
  RunChunk({7, 8, 9}, 1, {0, 0, 0});
}

TEST_P(RuntimeLastTokenLogitsModesTest, FinalSingletonPaddingValidatesWithAnExistingZeroIndex) {
  Initialize("pruned.onnx", std::get<0>(GetParam()), std::get<1>(GetParam()));
  logits_->Add();
  RunChunk({4, 5, 6}, 1, {0, 0, 0});
  auto* index_binding = state_->inputs_[1];
  auto* output_binding = state_->outputs_[0];
  auto next = SetTokens({4, 0, 6}, 1);
  State::PromptInputScope prompt_scope(*state_, true);
  EXPECT_THROW(logits_->Update(next, 1), std::invalid_argument);
  EXPECT_NO_THROW(logits_->Update(next, 1, /*all_tokens_valid=*/false, /*is_final_chunk=*/false));
  EXPECT_EQ(state_->inputs_[1], index_binding);
  EXPECT_EQ(state_->outputs_[0], output_binding);
}

TEST_P(RuntimeLastTokenLogitsModesTest, GeneratedPadDoesNotValidateOrRebind) {
  Initialize("pruned.onnx", std::get<0>(GetParam()), std::get<1>(GetParam()));
  logits_->Add();
  RunChunk({4, 5, 6}, 1, {0, 0, 0});
  auto* index_binding = state_->inputs_[1];
  auto* output_binding = state_->outputs_[0];
  RunChunk({0, 0, 0}, 1, {0, 0, 0}, /*expanded=*/false, /*is_prompt=*/false);
  EXPECT_EQ(state_->inputs_[1], index_binding);
  EXPECT_EQ(state_->outputs_[0], output_binding);
}

TEST_F(RuntimeLastTokenLogitsTest, PromptScopeRestoresTokenOriginAfterAnException) {
  Initialize();
  EXPECT_FALSE(state_->IsPromptInput());
  EXPECT_EQ(state_->InputSelectionLength(), 0u);
  EXPECT_THROW(
      [&] {
        State::PromptInputScope prompt_scope(*state_, true, 3);
        EXPECT_TRUE(state_->IsPromptInput());
        EXPECT_EQ(state_->InputSelectionLength(), 3u);
        {
          State::PromptInputScope nested_scope(*state_, false, 1);
          EXPECT_FALSE(state_->IsPromptInput());
          EXPECT_EQ(state_->InputSelectionLength(), 1u);
        }
        EXPECT_TRUE(state_->IsPromptInput());
        EXPECT_EQ(state_->InputSelectionLength(), 3u);
        throw std::invalid_argument("test prompt failure");
      }(),
      std::invalid_argument);
  EXPECT_FALSE(state_->IsPromptInput());
  EXPECT_EQ(state_->InputSelectionLength(), 0u);
}

TEST_F(RuntimeLastTokenLogitsTest, InvalidChunkPreservesExistingIndexAndOutputBindings) {
  Initialize("pruned.onnx", true, 2);
  logits_->Add();
  ASSERT_EQ(state_->inputs_.size(), 2u);
  RunChunk({4, 0, 5, 6, 7, 0}, 2, {0, 1, 0});
  auto* indices = state_->inputs_[1];
  auto* output = state_->outputs_[0];
  auto next = SetTokens({8, 0, 0, 0, 9, 10}, 2);
  State::PromptInputScope prompt_scope(*state_, true);
  EXPECT_THROW(logits_->Update(next, 2), std::invalid_argument);
  EXPECT_EQ(state_->inputs_[1], indices);
  EXPECT_EQ(state_->outputs_[0], output);
  for (int batch = 0; batch < 3; ++batch) {
    for (int beam = 0; beam < params_->search.num_beams; ++beam)
      EXPECT_EQ(indices->GetTensorData<int32_t>()[batch * params_->search.num_beams + beam], batch == 1 ? 1 : 0);
  }
}

TEST_F(RuntimeLastTokenLogitsTest, RejectsNonInt32IndicesAndUnprunedOutputBeforeBinding) {
  for (const auto* filename : {"invalid-index-type.onnx", "unpruned-with-indices.onnx"}) {
    SCOPED_TRACE(filename);
    logits_.reset();
    state_.reset();
    Initialize(filename);
    EXPECT_THROW(logits_->Add(), std::runtime_error);
    EXPECT_EQ(state_->inputs_.size(), 1u);
    EXPECT_TRUE(state_->outputs_.empty());
  }
}

TEST_F(RuntimeLastTokenLogitsTest, ModelsWithoutReservedInputKeepFullLogitsBehavior) {
  Initialize("full.onnx");
  logits_->Add();
  EXPECT_EQ(state_->input_names_.size(), 1u);
  RunChunk({0, 4, 5, 0, 0, 0, 0, 6, 7, 8, 9, 0, 10, 0, 0}, 5, {2, 4, 2});
  RunChunk({11, 12, 13}, 1, {0, 0, 0});
}

INSTANTIATE_TEST_SUITE_P(CaptureAndBeams, RuntimeLastTokenLogitsModesTest,
                         testing::Combine(testing::Bool(), testing::Values(1, 2)));

}  // namespace
}  // namespace Generators::test
