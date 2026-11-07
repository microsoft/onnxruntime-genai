// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "models/nemotron_bolt.h"
#include "nlohmann/json.hpp"
#include "search.h"

namespace Generators::test {

struct NemotronBoltTestAccess {
  static void ForceRejectedDrafts(NemotronBoltState& state) {
    state.drafts_.assign(7, 0);
  }

  static bool ModelSupportsR16(const NemotronBoltState& state) {
    return state.bolt_.supports_r16_prefill_;
  }

  static bool R16Enabled(const NemotronBoltState& state) {
    return state.r16_enabled_;
  }

  static bool HasR16ProfileTensors(const NemotronBoltState& state) {
    return !state.inputs_16_.empty() && !state.outputs_16_.empty() &&
           !state.tensors_16_.empty() && state.aux_view_16_ != nullptr;
  }

  static size_t VerifyInputsSpanSize(const NemotronBoltState& state) {
    return state.verify_inputs_span_.size();
  }

  static uint64_t WarmupR8Calls(const NemotronBoltState& state) {
    return state.warmup_r8_calls_;
  }

  static uint64_t WarmupR16Calls(const NemotronBoltState& state) {
    return state.warmup_r16_calls_;
  }

  static uint64_t PrefillR8Calls(const NemotronBoltState& state) {
    return state.prefill_r8_calls_;
  }

  static uint64_t PrefillR16Calls(const NemotronBoltState& state) {
    return state.prefill_r16_calls_;
  }

  static uint64_t VerifyR8Calls(const NemotronBoltState& state) {
    return state.verify_r8_calls_;
  }

  static size_t LastPrefillDraftStart(const NemotronBoltState& state) {
    return state.last_prefill_draft_start_;
  }

  static size_t LastPrefillDraftRows(const NemotronBoltState& state) {
    return state.last_prefill_draft_rows_;
  }

  static size_t Length(const NemotronBoltState& state) {
    return state.length_;
  }

  static int32_t StateRow(const NemotronBoltState& state) {
    return state.state_row_;
  }

  static int32_t Anchor(const NemotronBoltState& state) {
    return state.anchor_;
  }

  static std::vector<float> SelectedSsm0RowSlice(NemotronBoltState& state, size_t max_floats = 64) {
    auto it = state.tensors_.find("past_ssm.0");
    if (it == state.tensors_.end() || !it->second) return {};
    Tensor& ssm = *it->second;
    const size_t total_elems = ssm.GetElementCount();
    if (total_elems < 8) return {};
    const size_t row_elems = total_elems / 8;
    const size_t count = std::min(row_elems, max_floats);
    const size_t row_idx = static_cast<size_t>(std::clamp<int32_t>(state.state_row_, 0, 7));
    const size_t byte_offset = row_idx * row_elems * sizeof(float);
    const size_t byte_count = count * sizeof(float);
    auto scratch = state.model_.p_device_->Allocate<uint8_t>(byte_count);
    scratch.CopyFrom(ssm.GetByteSpan().subspan(byte_offset, byte_count));
    auto cpu_bytes = scratch.CopyDeviceToCpu();
    std::vector<float> out(count);
    std::memcpy(out.data(), cpu_bytes.data(), byte_count);
    return out;
  }

  static std::vector<float> SelectedLastLogitsSlice(NemotronBoltState& state, size_t prompt_rows,
                                                    size_t max_elems = 64) {
    const bool last_was_r16 =
        state.r16_enabled_ && (prompt_rows >= kBoltPrefillRows16) && (prompt_rows % kBoltPrefillRows16 == 0);
    Tensor* logits = nullptr;
    size_t last_valid_row = 0;
    if (last_was_r16) {
      auto it16 = state.tensors_16_.find("logits");
      if (it16 == state.tensors_16_.end() || !it16->second) return {};
      logits = it16->second.get();
      last_valid_row = kBoltPrefillRows16 - 1;
    } else {
      auto it8 = state.tensors_.find("logits");
      if (it8 == state.tensors_.end() || !it8->second) return {};
      logits = it8->second.get();
      const size_t rem8 = prompt_rows % kBoltVerifyRows;
      last_valid_row = (rem8 == 0) ? (kBoltVerifyRows - 1) : (rem8 - 1);
    }
    const auto shape = logits->GetShape();
    if (shape.size() != 3 || shape[2] <= 0) return {};
    const size_t vocab_size = static_cast<size_t>(shape[2]);
    const size_t count = std::min(vocab_size, max_elems);
    const size_t byte_offset = last_valid_row * vocab_size * sizeof(uint16_t);
    const size_t byte_count = count * sizeof(uint16_t);
    auto scratch = state.model_.p_device_->Allocate<uint8_t>(byte_count);
    scratch.CopyFrom(logits->GetByteSpan().subspan(byte_offset, byte_count));
    auto cpu_bytes = scratch.CopyDeviceToCpu();
    std::vector<uint16_t> bf16(count);
    std::memcpy(bf16.data(), cpu_bytes.data(), byte_count);
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) {
      const uint32_t bits = static_cast<uint32_t>(bf16[i]) << 16;
      float val = 0.0f;
      std::memcpy(&val, &bits, sizeof(float));
      out[i] = val;
    }
    return out;
  }

  static std::vector<int64_t> InvokePrefillChunk16ForTest(NemotronBoltState& state,
                                                          std::span<const int32_t> tokens) {
    return state.PrefillChunk16(tokens, true);
  }

  static std::shared_ptr<NemotronBoltModel> CreateSyntheticBoltModel(
      const uint8_t* graph_bytes, size_t graph_size,
      const uint8_t* drafter_bytes, size_t drafter_size,
      size_t capacity = 135168) {
    auto config = std::make_unique<Config>();
    config->model.type = "nemotron_bolt";
    config->model.vocab_size = 128;
    config->model.context_length = static_cast<int>(capacity - kBoltVerifyRows);
    config->model.eos_token_id = {2, 11};
    config->model.dflash2.filename = "dflash2.onnx";
    config->model.dflash2.num_hidden_layers = 1;
    config->model.dflash2.num_key_value_heads = 1;
    config->model.dflash2.head_size = 1;
    config->model.dflash2.sliding_window = 1024;
    config->model.dflash2.block_size = 8;
    config->model.dflash2.num_draft_tokens = 7;
    config->model.dflash2.selector_top_k = 1;
    config->model.dflash2.mask_token_id = 0;
    config->model.dflash2.inputs.anchor_row_index = "anchor_row_index";
    config->speculative.max_draft_tokens = 7;
    config->search.max_length = 64;
    config->model_data_spans_.emplace(
        "dflash2.onnx", std::span<const std::byte>{reinterpret_cast<const std::byte*>(drafter_bytes), drafter_size});
    auto drafter_config = CreateDflash2Config(*config);
    drafter_config->model.decoder.session_options.intra_op_num_threads = 1;
    drafter_config->model.decoder.session_options.graph_optimization_level = ORT_DISABLE_ALL;
    auto drafter_model = std::make_shared<Dflash2Model>(std::move(drafter_config), GetOrtEnv());
    auto session_options = OrtSessionOptions::Create();
    session_options->SetIntraOpNumThreads(1);
    session_options->SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    auto session = OrtSession::Create(GetOrtEnv(), graph_bytes, graph_size, session_options.get());
    auto model = std::shared_ptr<NemotronBoltModel>(new NemotronBoltModel(std::move(config)));
    model->p_device_scoring_ = GetDeviceInterface(DeviceType::CPU);
    model->session_ = std::move(session);
    model->drafter_model_ = std::move(drafter_model);
    model->capacity_ = capacity;
    model->session_info_.Add(*model->session_);
    const auto& info = model->session_info_;
    const auto to_sym = [](const std::vector<const char*>& raw) {
      std::vector<std::string> out;
      out.reserve(raw.size());
      for (const char* dim : raw) out.emplace_back(dim ? dim : "");
      return out;
    };
    const auto row_contract = ValidateNemotronBoltTargetRowContract(
        info.GetInputShape("input_ids"), to_sym(info.GetInputSymbolicShape("input_ids")),
        info.GetOutputShape("pred_ids"), to_sym(info.GetOutputSymbolicShape("pred_ids")),
        info.GetOutputShape("logits"), to_sym(info.GetOutputSymbolicShape("logits")),
        info.GetOutputShape("aux_hidden_states"),
        to_sym(info.GetOutputSymbolicShape("aux_hidden_states")),
        model->config_->model.vocab_size);
    model->supports_r16_prefill_ = row_contract.dynamic_r;
    model->aux_width_ = row_contract.aux_width;
    return model;
  }

  static std::vector<int32_t> TargetOnly(NemotronBoltState& state, size_t count) {
    std::vector<int32_t> tokens{state.anchor_};
    for (size_t i = 1; i < count; ++i) {
      const std::vector<int32_t> block(8, tokens.back());
      const auto predictions = state.Verify(block);
      ++state.length_;
      state.state_row_ = 0;
      tokens.push_back(static_cast<int32_t>(predictions[0]));
    }
    return tokens;
  }
};

namespace {

class ScopedBoltEnvVar {
 public:
  ScopedBoltEnvVar(const char* name, std::optional<std::string> value) : name_{name} {
    if (const char* current = std::getenv(name)) {
      original_ = std::string{current};
    }
    Apply(value);
  }

  ~ScopedBoltEnvVar() {
    Apply(original_);
  }

  ScopedBoltEnvVar(const ScopedBoltEnvVar&) = delete;
  ScopedBoltEnvVar& operator=(const ScopedBoltEnvVar&) = delete;

  static void Apply(const std::string& name, const std::optional<std::string>& value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.has_value() ? value->c_str() : "");
#else
    if (value.has_value()) {
      setenv(name.c_str(), value->c_str(), 1);
    } else {
      unsetenv(name.c_str());
    }
#endif
  }

 private:
  void Apply(const std::optional<std::string>& value) const {
    Apply(name_, value);
  }

  std::string name_;
  std::optional<std::string> original_;
};

TEST(NemotronBolt, EveryAcceptanceLengthAndBudget) {
  const std::vector<int32_t> drafts{20, 21, 22, 23, 24, 25, 26};
  const std::vector<int> eos{2, 11};
  for (size_t accepted = 0; accepted <= drafts.size(); ++accepted) {
    std::vector<int64_t> predictions(8, 99);
    std::copy_n(drafts.begin(), accepted, predictions.begin());
    EXPECT_EQ(NemotronBoltAcceptedPrefix(drafts, predictions), accepted);
    for (size_t remaining = 0; remaining <= 9; ++remaining) {
      const auto tokens = NemotronBoltRoundTokens(drafts, predictions, remaining, eos);
      ASSERT_EQ(tokens.size(), std::min(accepted + 1, remaining));
      for (size_t i = 0; i < tokens.size(); ++i)
        EXPECT_EQ(tokens[i], i < accepted ? drafts[i] : 99);
    }
  }
}

TEST(NemotronBolt, StopsAtEitherEosIncludingCorrectionAndBonus) {
  for (int eos_token : {2, 11}) {
    for (size_t eos_row = 0; eos_row < 8; ++eos_row) {
      std::vector<int32_t> drafts{20, 21, 22, 23, 24, 25, 26};
      if (eos_row < 7) drafts[eos_row] = eos_token;
      std::vector<int64_t> predictions(drafts.begin(), drafts.end());
      predictions.push_back(eos_row == 7 ? eos_token : 99);
      const auto tokens = NemotronBoltRoundTokens(drafts, predictions, 8, std::vector<int>{2, 11});
      ASSERT_EQ(tokens.size(), eos_row + 1);
      EXPECT_EQ(tokens.back(), eos_token);
    }
    const auto correction = NemotronBoltRoundTokens(
        std::vector<int32_t>(7, 20), std::vector<int64_t>(8, eos_token), 8,
        std::vector<int>{2, 11});
    EXPECT_EQ(correction, std::vector<int32_t>{eos_token});
  }
}

TEST(NemotronBolt, RejectsInvalidVerificationGeometry) {
  EXPECT_THROW(NemotronBoltAcceptedPrefix(std::vector<int32_t>(6), std::vector<int64_t>(8)),
               std::runtime_error);
  EXPECT_THROW(NemotronBoltAcceptedPrefix(std::vector<int32_t>(7), std::vector<int64_t>(7)),
               std::runtime_error);
}

TEST(NemotronBolt, RejectsUnsupportedSearchInsteadOfIgnoringIt) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.dflash2.sliding_window = 1024;
  config.speculative.max_draft_tokens = 7;
  auto params = std::make_shared<GeneratorParams>(config);
  params->use_graph_capture = true;
  EXPECT_NO_THROW(ValidateNemotronBoltSearch(*params));
  const std::vector<std::function<void(GeneratorParams&)>> changes{
      [](auto& p) { p.search.do_sample = true; },
      [](auto& p) { p.search.batch_size = 2; },
      [](auto& p) { p.search.num_beams = 2; },
      [](auto& p) { p.search.num_return_sequences = 2; },
      [](auto& p) { p.search.min_length = 1; },
      [](auto& p) { p.search.repetition_penalty = 1.1f; },
      [](auto& p) { p.search.no_repeat_ngram_size = 2; },
      [](auto& p) { p.search.diversity_penalty = 1; },
      [](auto& p) { p.search.length_penalty = 2; },
      [](auto& p) { p.search.blank_penalty = 1; },
      [](auto& p) { p.guidance_type = "json_schema"; },
      [](auto& p) { p.guidance_data = "{}"; },
      [](auto& p) { p.guidance_ff_tokens_enabled = true; },
      [](auto& p) { p.speculative.ngram_size = 3; },
      [](auto& p) { p.speculative.ngram_chained_lookup = true; },
      [](auto& p) { p.speculative.min_adaptive_k = 2; },
      [](auto& p) { p.speculative.cooldown = true; },
      [](auto& p) { p.speculative.max_draft_tokens = 4; },
      [](auto& p) { p.search.chunk_size = 8; },
      [](auto& p) { p.use_graph_capture = false; }};
  for (const auto& change : changes) {
    auto invalid = std::make_shared<GeneratorParams>(config);
    invalid->use_graph_capture = true;
    change(*invalid);
    EXPECT_THROW(ValidateNemotronBoltSearch(*invalid), std::runtime_error);
  }
  Config sliding = config;
  sliding.model.vocab_size = 128;
  sliding.model.context_length = 16;
  sliding.search.max_length = 16;
  auto valid_sliding_free = std::make_shared<GeneratorParams>(sliding);
  valid_sliding_free->use_graph_capture = true;
  EXPECT_NO_THROW(CreateSearch(*valid_sliding_free));
  sliding.model.decoder.sliding_window = Config::Model::Decoder::SlidingWindow{8};
  auto invalid_sliding = std::make_shared<GeneratorParams>(sliding);
  invalid_sliding->use_graph_capture = true;
  EXPECT_THROW(ValidateNemotronBoltSearch(*invalid_sliding), std::runtime_error);
  EXPECT_THROW(CreateSearch(*invalid_sliding), std::runtime_error);
}

TEST(NemotronBolt, PublicTokenCommitAndResetHaveNoLookaheadLeak) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.vocab_size = 128;
  config.model.context_length = 16;
  config.model.eos_token_id = {2, 11};
  config.search.max_length = 11;
  auto params = std::make_shared<GeneratorParams>(config);
  auto search = CreateSearch(*params);
  auto prompt = params->p_device->Allocate<int32_t>(3);
  const std::vector<int32_t> prompt_ids{1, 8, 9};
  prompt.CopyFromCpu(prompt_ids);
  for (int request = 0; request < 2; ++request) {
    search->AppendTokens(prompt);
    const auto queued = NemotronBoltRoundTokens(
        std::vector<int32_t>{20, 21, 22, 23, 24, 25, 26},
        std::vector<int64_t>{20, 21, 22, 23, 24, 25, 26, 27}, 8,
        config.model.eos_token_id);
    EXPECT_EQ(search->GetSequenceLength(), 3);
    for (size_t i = 0; i < queued.size(); ++i) {
      EXPECT_FALSE(search->IsDone());
      search->CommitToken(queued[i]);
      EXPECT_EQ(search->GetSequenceLength(), 4 + i);
      const auto visible = search->GetSequence(0).CopyDeviceToCpu();
      EXPECT_EQ(visible.back(), queued[i]);
    }
    EXPECT_TRUE(search->IsDone());
    search->RewindTo(0);
    EXPECT_EQ(search->GetSequenceLength(), 0);
    EXPECT_FALSE(search->IsDone());
  }
}

TEST(NemotronBolt, TerminalEosIsVisibleAndStopsExactlyOnce) {
  for (int token : {2, 11}) {
    Config config;
    config.model.type = "nemotron_bolt";
    config.model.vocab_size = 128;
    config.model.context_length = 16;
    config.model.eos_token_id = {2, 11};
    config.search.max_length = 16;
    auto params = std::make_shared<GeneratorParams>(config);
    auto search = CreateSearch(*params);
    auto prompt = params->p_device->Allocate<int32_t>(1);
    prompt.CpuSpan()[0] = 1;
    search->AppendTokens(prompt);
    search->CommitToken(token);
    EXPECT_TRUE(search->IsDone());
    EXPECT_EQ(search->GetSequenceLength(), 2);
    EXPECT_EQ(search->GetSequence(0).CopyDeviceToCpu().back(), token);
    EXPECT_THROW(search->CommitToken(20), std::runtime_error);
    search->RewindTo(0);
    EXPECT_FALSE(search->IsDone());
    search->AppendTokens(prompt);
    search->CommitToken(token);
    EXPECT_EQ(search->GetSequenceLength(), 2);
  }
}

TEST(NemotronBolt, PromptAtBudgetIsImmediatelyDone) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.vocab_size = 128;
  config.model.context_length = 16;
  config.search.max_length = 1;
  auto params = std::make_shared<GeneratorParams>(config);
  auto search = CreateSearch(*params);
  auto prompt = params->p_device->Allocate<int32_t>(1);
  prompt.CpuSpan()[0] = 20;
  search->AppendTokens(prompt);
  EXPECT_TRUE(search->IsDone());
  EXPECT_EQ(search->GetSequenceLength(), 1);
  EXPECT_THROW(search->CommitToken(21), std::runtime_error);
}

TEST(NemotronBolt, PrefillAuxiliaryRowsPreserveOrderAndIgnorePadding) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  Tensor chunk(device, Ort::TypeToTensorType<float>);
  Tensor prompt(device, Ort::TypeToTensorType<float>);
  chunk.CreateTensor(std::vector<int64_t>{8, 3});
  prompt.CreateTensor(std::vector<int64_t>{19, 3});
  for (size_t start = 0; start < 19; start += 8) {
    auto values = chunk.GetDeviceSpan<float>().CpuSpan();
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(start * 3 + i);
    CopyNemotronBoltPrefillAuxRows(prompt, chunk, start, std::min(size_t{8}, 19 - start));
  }
  const auto result = prompt.GetDeviceSpan<float>().CopyDeviceToCpu();
  for (size_t i = 0; i < result.size(); ++i)
    EXPECT_EQ(result[i], static_cast<float>(i));
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(prompt, chunk, 18, 2), std::runtime_error);
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(prompt, chunk, 0, 9), std::runtime_error);
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(prompt, chunk, 0, 0), std::runtime_error);
}

TEST(NemotronBolt, DisableSwitchValueAndResolutionContract) {
  EXPECT_FALSE(IsNemotronBoltR16PrefillDisabledValue(nullptr));
  EXPECT_FALSE(IsNemotronBoltR16PrefillDisabledValue(""));
  EXPECT_FALSE(IsNemotronBoltR16PrefillDisabledValue("0"));
  EXPECT_FALSE(IsNemotronBoltR16PrefillDisabledValue("false"));
  EXPECT_FALSE(IsNemotronBoltR16PrefillDisabledValue("true"));
  EXPECT_FALSE(IsNemotronBoltR16PrefillDisabledValue("10"));
  EXPECT_TRUE(IsNemotronBoltR16PrefillDisabledValue("1"));

  {
    ScopedBoltEnvVar unset_env("BOLT_DISABLE_R16_PREFILL", std::nullopt);
    EXPECT_FALSE(ResolveNemotronBoltR16Enabled(false));
    EXPECT_TRUE(ResolveNemotronBoltR16Enabled(true));
  }
  for (const char* non_disable : {"", "0", "false", "true", "2"}) {
    ScopedBoltEnvVar env("BOLT_DISABLE_R16_PREFILL", std::string{non_disable});
    EXPECT_FALSE(ResolveNemotronBoltR16Enabled(false));
    EXPECT_TRUE(ResolveNemotronBoltR16Enabled(true)) << "value=" << non_disable;
  }
  {
    ScopedBoltEnvVar disable_env("BOLT_DISABLE_R16_PREFILL", std::string{"1"});
    EXPECT_FALSE(ResolveNemotronBoltR16Enabled(false));
    EXPECT_FALSE(ResolveNemotronBoltR16Enabled(true));
  }
}

TEST(NemotronBolt, ProductionAuxiliaryWindowAndDrafterTrimmingAcrossAlignmentAndWindowCrossings) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  constexpr size_t kContextWindow = 1032;  // sliding_window (1024) + block_size (8)
  constexpr size_t kBlockRows = 8;

  // Verify exact production draft_start, draft_rows, and drafter trimming offsets around
  // the 1032 effective context window and 8-/16-row alignment boundaries.
  {
    const auto w1031 = PlanNemotronBoltPrefillAuxWindow(1031, kContextWindow, kBlockRows);
    EXPECT_EQ(w1031.max_buffer_rows, 1040u);
    EXPECT_EQ(w1031.draft_start, 0u);
    EXPECT_EQ(w1031.draft_rows, 1031u);
    EXPECT_EQ(w1031.ingest_offset, 0u);
    EXPECT_EQ(w1031.ingest_rows, 1031u);
    EXPECT_EQ(w1031.ingest_first_position, 0u);

    const auto w1032 = PlanNemotronBoltPrefillAuxWindow(1032, kContextWindow, kBlockRows);
    EXPECT_EQ(w1032.draft_start, 0u);
    EXPECT_EQ(w1032.draft_rows, 1032u);
    EXPECT_EQ(w1032.ingest_offset, 0u);
    EXPECT_EQ(w1032.ingest_rows, 1032u);
    EXPECT_EQ(w1032.ingest_first_position, 0u);

    // At 1033 tokens, production rounds draft_start down to 0 (retaining 1033 physical rows)
    // and Dflash2Drafter::Propose drops the leading 1 excess row to ingest rows [1..1032].
    const auto w1033 = PlanNemotronBoltPrefillAuxWindow(1033, kContextWindow, kBlockRows);
    EXPECT_EQ(w1033.draft_start, 0u);
    EXPECT_EQ(w1033.draft_rows, 1033u);
    EXPECT_EQ(w1033.ingest_offset, 1u);
    EXPECT_EQ(w1033.ingest_rows, 1032u);
    EXPECT_EQ(w1033.ingest_first_position, 1u);

    const auto w1039 = PlanNemotronBoltPrefillAuxWindow(1039, kContextWindow, kBlockRows);
    EXPECT_EQ(w1039.draft_start, 0u);
    EXPECT_EQ(w1039.draft_rows, 1039u);
    EXPECT_EQ(w1039.ingest_offset, 7u);
    EXPECT_EQ(w1039.ingest_rows, 1032u);
    EXPECT_EQ(w1039.ingest_first_position, 7u);

    const auto w1040 = PlanNemotronBoltPrefillAuxWindow(1040, kContextWindow, kBlockRows);
    EXPECT_EQ(w1040.draft_start, 8u);
    EXPECT_EQ(w1040.draft_rows, 1032u);
    EXPECT_EQ(w1040.ingest_offset, 0u);
    EXPECT_EQ(w1040.ingest_rows, 1032u);
    EXPECT_EQ(w1040.ingest_first_position, 8u);

    const auto w1041 = PlanNemotronBoltPrefillAuxWindow(1041, kContextWindow, kBlockRows);
    EXPECT_EQ(w1041.draft_start, 8u);
    EXPECT_EQ(w1041.draft_rows, 1033u);
    EXPECT_EQ(w1041.ingest_offset, 1u);
    EXPECT_EQ(w1041.ingest_rows, 1032u);
    EXPECT_EQ(w1041.ingest_first_position, 9u);

    const auto w1048 = PlanNemotronBoltPrefillAuxWindow(1048, kContextWindow, kBlockRows);
    EXPECT_EQ(w1048.draft_start, 16u);
    EXPECT_EQ(w1048.draft_rows, 1032u);
    EXPECT_EQ(w1048.ingest_offset, 0u);
    EXPECT_EQ(w1048.ingest_rows, 1032u);
    EXPECT_EQ(w1048.ingest_first_position, 16u);

    const auto w1049 = PlanNemotronBoltPrefillAuxWindow(1049, kContextWindow, kBlockRows);
    EXPECT_EQ(w1049.draft_start, 16u);
    EXPECT_EQ(w1049.draft_rows, 1033u);
    EXPECT_EQ(w1049.ingest_offset, 1u);
    EXPECT_EQ(w1049.ingest_rows, 1032u);
    EXPECT_EQ(w1049.ingest_first_position, 17u);
  }

  std::vector<size_t> lengths{
      1, 7, 8, 9, 15, 16, 32, 33, 57,
      1023, 1024, 1025, 1031, 1032, 1033,
      1039, 1040, 1041, 1047, 1048, 1049,
      4099, 32768};
  for (size_t rem = 1; rem <= 15; ++rem) lengths.push_back(16 + rem);

  for (size_t count : lengths) {
    const auto aux_win = PlanNemotronBoltPrefillAuxWindow(count, kContextWindow, kBlockRows);
    ASSERT_LE(aux_win.draft_rows, aux_win.max_buffer_rows);
    ASSERT_EQ(aux_win.ingest_rows, std::min(count, kContextWindow));
    ASSERT_EQ(aux_win.ingest_first_position, count - aux_win.ingest_rows);

    Tensor full_r8(device, Ort::TypeToTensorType<float>);
    Tensor full_r16(device, Ort::TypeToTensorType<float>);
    Tensor bounded_r8(device, Ort::TypeToTensorType<float>);
    Tensor bounded_r16(device, Ort::TypeToTensorType<float>);
    full_r8.CreateTensor(std::vector<int64_t>{static_cast<int64_t>(count), 3});
    full_r16.CreateTensor(std::vector<int64_t>{static_cast<int64_t>(count), 3});
    bounded_r8.CreateTensor(std::vector<int64_t>{static_cast<int64_t>(aux_win.max_buffer_rows), 3});
    bounded_r16.CreateTensor(std::vector<int64_t>{static_cast<int64_t>(aux_win.max_buffer_rows), 3});

    for (bool allow_r16 : {false, true}) {
      Tensor& full = allow_r16 ? full_r16 : full_r8;
      Tensor& bounded = allow_r16 ? bounded_r16 : bounded_r8;
      std::fill(full.GetDeviceSpan<float>().CpuSpan().begin(),
                full.GetDeviceSpan<float>().CpuSpan().end(), -2.0f);
      std::fill(bounded.GetDeviceSpan<float>().CpuSpan().begin(),
                bounded.GetDeviceSpan<float>().CpuSpan().end(), -1.0f);

      Tensor chunk8(device, Ort::TypeToTensorType<float>);
      Tensor chunk16(device, Ort::TypeToTensorType<float>);
      chunk8.CreateTensor(std::vector<int64_t>{8, 3});
      chunk16.CreateTensor(std::vector<int64_t>{16, 3});

      const auto steps = PlanNemotronBoltPrefillChunks(count, 135168, allow_r16, 0, 0);
      for (const auto& step : steps) {
        Tensor& chunk = (step.profile_rows == 16) ? chunk16 : chunk8;
        auto input = chunk.GetDeviceSpan<float>().CpuSpan();
        for (size_t row = 0; row < step.profile_rows; ++row) {
          for (size_t col = 0; col < 3; ++col) {
            if (row < step.valid_rows) {
              const size_t logical_pos = step.pos + row;
              input[row * 3 + col] = static_cast<float>(logical_pos * 10 + col);
            } else {
              // Padding rows in partial R=8 tail chunks must never be copied.
              input[row * 3 + col] = -999.0f;
            }
          }
        }
        CopyNemotronBoltPrefillAuxRows(full, chunk, step.pos, step.valid_rows, 0);
        CopyNemotronBoltPrefillAuxRows(bounded, chunk, step.pos, step.valid_rows, aux_win.draft_start);
      }

      const auto full_cpu = full.GetDeviceSpan<float>().CopyDeviceToCpu();
      const auto bounded_cpu = bounded.GetDeviceSpan<float>().CopyDeviceToCpu();

      // 1. Physical rows [0..draft_rows) in prefill_aux must match logical [draft_start..count)
      //    and exclude any padding poison (-999.0f).
      for (size_t phys_row = 0; phys_row < aux_win.draft_rows; ++phys_row) {
        const size_t logical_pos = aux_win.draft_start + phys_row;
        for (size_t col = 0; col < 3; ++col) {
          const float actual_val = bounded_cpu[phys_row * 3 + col];
          const float expected_val = static_cast<float>(logical_pos * 10 + col);
          EXPECT_EQ(actual_val, expected_val)
              << "N=" << count << " allow_r16=" << allow_r16 << " phys_row=" << phys_row;
          EXPECT_EQ(actual_val, full_cpu[logical_pos * 3 + col]);
        }
      }

      // 2. Unused physical tail rows [draft_rows..max_buffer_rows) in prefill_aux must remain untouched (-1.0f).
      for (size_t phys_row = aux_win.draft_rows; phys_row < aux_win.max_buffer_rows; ++phys_row) {
        for (size_t col = 0; col < 3; ++col) {
          EXPECT_EQ(bounded_cpu[phys_row * 3 + col], -1.0f)
              << "N=" << count << " allow_r16=" << allow_r16 << " untouched phys_row=" << phys_row;
        }
      }

      // 3. Final suffix ingested after drafter trimming [ingest_offset..ingest_offset+ingest_rows)
      //    must equal the exact trailing min(count, 1032) rows of the prompt.
      const auto ingested_slice =
          bounded_cpu.subspan(aux_win.ingest_offset * 3, aux_win.ingest_rows * 3);
      const auto expected_suffix =
          full_cpu.subspan((count - aux_win.ingest_rows) * 3, aux_win.ingest_rows * 3);
      ASSERT_EQ(ingested_slice.size(), expected_suffix.size());
      for (size_t i = 0; i < ingested_slice.size(); ++i) {
        const size_t suffix_row = i / 3;
        const size_t suffix_col = i % 3;
        const size_t expected_logical_pos = aux_win.ingest_first_position + suffix_row;
        EXPECT_EQ(ingested_slice[i], expected_suffix[i])
            << "N=" << count << " allow_r16=" << allow_r16 << " suffix_row=" << suffix_row;
        EXPECT_EQ(ingested_slice[i], static_cast<float>(expected_logical_pos * 10 + suffix_col));
      }
    }

    // 4. R=8 and R=16 physical bounded auxiliary buffers must match identically across all rows.
    const auto b8 = bounded_r8.GetDeviceSpan<float>().CopyDeviceToCpu();
    const auto b16 = bounded_r16.GetDeviceSpan<float>().CopyDeviceToCpu();
    ASSERT_EQ(b8.size(), b16.size());
    for (size_t i = 0; i < b8.size(); ++i) {
      EXPECT_EQ(b16[i], b8[i]) << "N=" << count << " idx=" << i;
    }
  }
}

TEST(NemotronBolt, BoundedAuxiliaryKeepsFullAttentionAndRejectsInvalidRanges) {
  const auto full_attn = PlanNemotronBoltPrefillAuxWindow(4099, 0, 8);
  EXPECT_EQ(full_attn.draft_start, 0u);
  EXPECT_EQ(full_attn.draft_rows, 4099u);
  EXPECT_EQ(full_attn.ingest_offset, 0u);
  EXPECT_EQ(full_attn.ingest_rows, 4099u);
  EXPECT_EQ(full_attn.ingest_first_position, 0u);
  EXPECT_THROW(PlanNemotronBoltPrefillAuxWindow(0, 1032, 8), std::runtime_error);
  EXPECT_THROW(PlanNemotronBoltPrefillAuxWindow(4099, 1032, 0), std::runtime_error);
  EXPECT_THROW(PlanNemotronBoltPrefillAuxWindow(4099, std::numeric_limits<size_t>::max() - 8, 8),
               std::runtime_error);
  auto* device = GetDeviceInterface(DeviceType::CPU);
  Tensor chunk(device, Ort::TypeToTensorType<float>);
  Tensor bounded(device, Ort::TypeToTensorType<float>);
  chunk.CreateTensor(std::vector<int64_t>{8, 3});
  bounded.CreateTensor(std::vector<int64_t>{1032, 3});
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(bounded, chunk, 1033, 1, 1), std::runtime_error);
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(bounded, chunk, 0, 9, 1), std::runtime_error);
  EXPECT_THROW(CopyNemotronBoltPrefillAuxRows(bounded, chunk, 0, 0, 1), std::runtime_error);
}

TEST(NemotronBolt, TargetRowContractAcceptsFixedR8AndSymbolicRAndRejectsUnsupportedWidths) {
  const auto fixed = ValidateNemotronBoltTargetRowContract(
      {1, 8}, {"", ""}, {1, 8}, {"", ""}, {1, 8, 131072}, {"", "", ""},
      {1, 8, 24576}, {"", "", ""}, 131072);
  EXPECT_FALSE(fixed.dynamic_r);
  EXPECT_EQ(fixed.aux_width, 24576u);

  const auto dynamic = ValidateNemotronBoltTargetRowContract(
      {1, -1}, {"", "R"}, {1, -1}, {"", "R"}, {1, -1, 131072}, {"", "R", ""},
      {1, -1, 24576}, {"", "R", ""}, 131072);
  EXPECT_TRUE(dynamic.dynamic_r);
  EXPECT_EQ(dynamic.aux_width, 24576u);

  // Reject static 16 without symbolic R
  EXPECT_THROW(ValidateNemotronBoltTargetRowContract(
                   {1, 16}, {"", ""}, {1, 16}, {"", ""}, {1, 16, 131072}, {"", "", ""},
                   {1, 16, 24576}, {"", "", ""}, 131072),
               std::runtime_error);
  // Reject mismatched symbolic / static
  EXPECT_THROW(ValidateNemotronBoltTargetRowContract(
                   {1, -1}, {"", "R"}, {1, 8}, {"", ""}, {1, -1, 131072}, {"", "R", ""},
                   {1, -1, 24576}, {"", "R", ""}, 131072),
               std::runtime_error);
  // Reject wrong symbolic dimension name
  EXPECT_THROW(ValidateNemotronBoltTargetRowContract(
                   {1, -1}, {"", "seq"}, {1, -1}, {"", "seq"}, {1, -1, 131072}, {"", "seq", ""},
                   {1, -1, 24576}, {"", "seq", ""}, 131072),
               std::runtime_error);
  // Reject wrong vocab size
  EXPECT_THROW(ValidateNemotronBoltTargetRowContract(
                   {1, -1}, {"", "R"}, {1, -1}, {"", "R"}, {1, -1, 1000}, {"", "R", ""},
                   {1, -1, 24576}, {"", "R", ""}, 131072),
               std::runtime_error);
}

TEST(NemotronBolt, PlanPrefillChunksCoversR16AndR8TailsAndCapacityGuards) {
  std::vector<size_t> lengths{1, 7, 8, 9, 15, 16, 32, 33, 57, 1031, 1032, 1033, 4099, 32768};
  for (size_t rem = 1; rem <= 15; ++rem) lengths.push_back(16 + rem);
  for (size_t n : lengths) {
    const auto steps = PlanNemotronBoltPrefillChunks(n, 135168, true, 0, 0);
    ASSERT_FALSE(steps.empty());
    size_t expected_r16 = n / 16;
    size_t rem = n % 16;
    size_t expected_r8 = (rem == 0) ? 0 : (rem <= 8 ? 1 : 2);
    size_t actual_r16 = 0, actual_r8 = 0;
    size_t pos = 0;
    int32_t expected_state_in = 0;
    for (size_t i = 0; i < steps.size(); ++i) {
      const auto& s = steps[i];
      EXPECT_EQ(s.pos, pos);
      EXPECT_EQ(s.state_row_in, expected_state_in);
      if (s.profile_rows == 16) {
        ++actual_r16;
        EXPECT_EQ(s.valid_rows, 16u);
        EXPECT_EQ(s.seqlens_k, static_cast<int32_t>(pos + 15));
        EXPECT_EQ(s.state_row_out, 7);
      } else {
        ++actual_r8;
        EXPECT_EQ(s.profile_rows, 8u);
        EXPECT_GE(s.valid_rows, 1u);
        EXPECT_LE(s.valid_rows, 8u);
        EXPECT_EQ(s.seqlens_k, static_cast<int32_t>(pos + 7));
        EXPECT_EQ(s.state_row_out, static_cast<int32_t>(s.valid_rows - 1));
      }
      pos += s.valid_rows;
      EXPECT_EQ(s.length_after, pos);
      expected_state_in = s.state_row_out;
    }
    EXPECT_EQ(pos, n);
    EXPECT_EQ(actual_r16, expected_r16) << "n=" << n;
    EXPECT_EQ(actual_r8, expected_r8) << "n=" << n;
  }
  // Capacity guards
  EXPECT_THROW(PlanNemotronBoltPrefillChunks(16, 15, true, 0, 0), std::runtime_error);
  EXPECT_THROW(PlanNemotronBoltPrefillChunks(17, 23, true, 0, 0), std::runtime_error);
  EXPECT_NO_THROW(PlanNemotronBoltPrefillChunks(17, 24, true, 0, 0));
}

TEST(NemotronBolt, PlanPrefillChunksSupportsNonzeroPrefixSplitAndFinalStateMatch) {
  struct PrefixCase {
    size_t prefix_r8;
    int32_t expected_mid_state_row;
    size_t total_rows;
  };
  const std::vector<PrefixCase> cases{{9, 0, 25}, {12, 3, 28}, {8, 7, 24}};
  for (const auto& tc : cases) {
    const auto r16_steps = PlanNemotronBoltPrefillChunks(tc.total_rows, 135168, true, 0, 0, tc.prefix_r8);
    const auto r8_steps = PlanNemotronBoltPrefillChunks(tc.total_rows, 135168, false, 0, 0, tc.prefix_r8);
    ASSERT_GE(r16_steps.size(), 2u);
    EXPECT_EQ(r16_steps.back().profile_rows, 16u);
    EXPECT_EQ(r16_steps.back().pos, tc.prefix_r8);
    EXPECT_EQ(r16_steps.back().state_row_in, tc.expected_mid_state_row);
    EXPECT_EQ(r16_steps.back().state_row_out, 7);
    EXPECT_EQ(r16_steps.back().length_after, tc.total_rows);
    EXPECT_EQ(r8_steps.back().state_row_out, r16_steps.back().state_row_out);
    EXPECT_EQ(r8_steps.back().length_after, r16_steps.back().length_after);
  }
  EXPECT_THROW(PlanNemotronBoltPrefillChunks(16, 135168, true, 0, 0, 16), std::runtime_error);
}

struct RoundAcceptanceSnapshot {
  size_t rounds{};
  size_t draft_tokens_proposed{};
  size_t draft_tokens_evaluated{};
  size_t draft_tokens_accepted{};
  size_t full_accept_rounds{};
  size_t partial_accept_rounds{};
  size_t zero_accept_rounds{};
  std::array<size_t, 8> histogram{};

  bool operator==(const RoundAcceptanceSnapshot&) const = default;
};

struct BoltRunRecord {
  std::vector<int32_t> generated_tokens;
  std::vector<RoundAcceptanceSnapshot> acceptance_trace;
  uint64_t warmup_r8_calls{};
  uint64_t warmup_r16_calls{};
  uint64_t prefill_r8_calls{};
  uint64_t prefill_r16_calls{};
  uint64_t verify_r8_calls{};
  size_t post_prefill_length{};
  int32_t post_prefill_state_row{};
  int32_t post_prefill_anchor{};
  size_t last_prefill_draft_start{};
  size_t last_prefill_draft_rows{};
  std::vector<float> post_prefill_ssm0_slice;
  std::vector<float> post_prefill_logits_slice;
  SpeculativeStats post_prefill_stats{};
  SpeculativeStats final_stats{};
};

std::vector<int32_t> BuildBoltBasePrompt() {
  return {
      10, 25708, 1010, 11, 1010, 10, 3263, 1010, 18746, 1261, 17616, 2254, 1455,
      29196, 5627, 1261, 3310, 1395, 1261, 5796, 92145, 1044, 1454, 1261, 4958,
      17984, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13};
}

std::vector<int32_t> LoadBoltPromptTokens(size_t target_length) {
  if (const char* json_path = std::getenv("ORTGENAI_BOLT_PROMPT_JSON")) {
    std::ifstream in{json_path};
    if (!in) {
      throw std::runtime_error(std::string{"Failed to open ORTGENAI_BOLT_PROMPT_JSON: "} + json_path);
    }
    const auto parsed = nlohmann::json::parse(in);
    const auto& arr = parsed.is_object() ? parsed.at("input_ids") : parsed;
    std::vector<int32_t> tokens = arr.get<std::vector<int32_t>>();
    if (tokens.size() < target_length) {
      throw std::runtime_error("ORTGENAI_BOLT_PROMPT_JSON has fewer tokens than requested length");
    }
    tokens.resize(target_length);
    return tokens;
  }
  const auto base = BuildBoltBasePrompt();
  std::vector<int32_t> out;
  out.reserve(target_length);
  while (out.size() < target_length) {
    const size_t copy_n = std::min(base.size(), target_length - out.size());
    out.insert(out.end(), base.begin(), base.begin() + static_cast<std::ptrdiff_t>(copy_n));
  }
  return out;
}

BoltRunRecord ExecuteBoltPromptRun(Model& model, std::span<const int32_t> prompt,
                                   size_t max_new_tokens, bool disable_r16) {
  ScopedBoltEnvVar env("BOLT_DISABLE_R16_PREFILL",
                       disable_r16 ? std::optional<std::string>{"1"} : std::nullopt);
  auto params = std::make_shared<GeneratorParams>(model);
  params->use_graph_capture = true;
  params->search.max_length = static_cast<int>(prompt.size() + max_new_tokens);
  Generator generator{model, *params};
  auto& state = dynamic_cast<NemotronBoltState&>(*generator.state_);

  BoltRunRecord rec{};
  rec.warmup_r8_calls = NemotronBoltTestAccess::WarmupR8Calls(state);
  rec.warmup_r16_calls = NemotronBoltTestAccess::WarmupR16Calls(state);
  EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 0u);
  EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 0u);
  EXPECT_EQ(NemotronBoltTestAccess::VerifyR8Calls(state), 0u);

  generator.AppendTokens(cpu_span<const int32_t>{prompt.data(), prompt.size()});
  rec.prefill_r8_calls = NemotronBoltTestAccess::PrefillR8Calls(state);
  rec.prefill_r16_calls = NemotronBoltTestAccess::PrefillR16Calls(state);
  rec.post_prefill_length = NemotronBoltTestAccess::Length(state);
  rec.post_prefill_state_row = NemotronBoltTestAccess::StateRow(state);
  rec.post_prefill_anchor = NemotronBoltTestAccess::Anchor(state);
  rec.last_prefill_draft_start = NemotronBoltTestAccess::LastPrefillDraftStart(state);
  rec.last_prefill_draft_rows = NemotronBoltTestAccess::LastPrefillDraftRows(state);
  rec.post_prefill_ssm0_slice = NemotronBoltTestAccess::SelectedSsm0RowSlice(state, 32);
  rec.post_prefill_logits_slice =
      NemotronBoltTestAccess::SelectedLastLogitsSlice(state, prompt.size(), 32);
  rec.post_prefill_stats = generator.GetSpeculativeStats();

  size_t prev_round = 0;
  while (!generator.IsDone()) {
    generator.GenerateNextToken();
    const auto st = generator.GetSpeculativeStats();
    if (st.rounds != prev_round) {
      RoundAcceptanceSnapshot snap{
          .rounds = st.rounds,
          .draft_tokens_proposed = st.draft_tokens_proposed,
          .draft_tokens_evaluated = st.draft_tokens_evaluated,
          .draft_tokens_accepted = st.draft_tokens_accepted,
          .full_accept_rounds = st.full_accept_rounds,
          .partial_accept_rounds = st.partial_accept_rounds,
          .zero_accept_rounds = st.zero_accept_rounds};
      for (size_t i = 0; i < 8; ++i) {
        snap.histogram[i] = st.acceptance_length_histogram[i];
      }
      rec.acceptance_trace.push_back(snap);
      prev_round = st.rounds;
    }
  }
  rec.verify_r8_calls = NemotronBoltTestAccess::VerifyR8Calls(state);
  rec.final_stats = generator.GetSpeculativeStats();
  const auto seq = generator.GetSequence(0).CopyDeviceToCpu();
  rec.generated_tokens.assign(seq.begin() + static_cast<std::ptrdiff_t>(prompt.size()), seq.end());
  return rec;
}

bool RequireGpuQualificationEnv() {
  const char* req = std::getenv("ORTGENAI_BOLT_REQUIRE_GPU_QUALIFICATION");
  return req != nullptr && std::string_view{req} == "1";
}

constexpr uint8_t kSymbolicRTargetGraph[1048] = {0x08, 0x09, 0x3a, 0x8d, 0x08, 0x0a, 0x27, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x0a, 0x0c, 0x6e, 0x69, 0x6e, 0x65, 0x74, 0x65, 0x65, 0x6e, 0x5f, 0x69, 0x36, 0x34, 0x12, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x73, 0x75, 0x62, 0x22, 0x03, 0x53, 0x75, 0x62, 0x0a, 0x30, 0x0a, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x73, 0x75, 0x62, 0x0a, 0x0a, 0x65, 0x69, 0x67, 0x68, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x12, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x6d, 0x6f, 0x64, 0x22, 0x03, 0x4d, 0x6f, 0x64, 0x2a, 0x0b, 0x0a, 0x04, 0x66, 0x6d, 0x6f, 0x64, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x29, 0x0a, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x6d, 0x6f, 0x64, 0x0a, 0x0a, 0x74, 0x77, 0x65, 0x6e, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x12, 0x0d, 0x69, 0x64, 0x73, 0x5f, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x62, 0x61, 0x73, 0x65, 0x22, 0x03, 0x41, 0x64, 0x64, 0x0a, 0x23, 0x0a, 0x0d, 0x69, 0x64, 0x73, 0x5f, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x62, 0x61, 0x73, 0x65, 0x12, 0x08, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x69, 0x64, 0x73, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x0a, 0x1c, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x08, 0x69, 0x6e, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x05, 0x53, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x38, 0x0a, 0x08, 0x69, 0x6e, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x09, 0x76, 0x6f, 0x63, 0x61, 0x62, 0x5f, 0x64, 0x69, 0x6d, 0x12, 0x0c, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x06, 0x43, 0x6f, 0x6e, 0x63, 0x61, 0x74, 0x2a, 0x0b, 0x0a, 0x04, 0x61, 0x78, 0x69, 0x73, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x2b, 0x0a, 0x0c, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x12, 0x0a, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x22, 0x0f, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x4f, 0x66, 0x53, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x25, 0x0a, 0x0a, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x12, 0x06, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x10, 0xa0, 0x01, 0x02, 0x0a, 0x26, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x0a, 0x06, 0x74, 0x77, 0x6f, 0x5f, 0x31, 0x64, 0x12, 0x06, 0x69, 0x64, 0x73, 0x5f, 0x33, 0x64, 0x22, 0x09, 0x55, 0x6e, 0x73, 0x71, 0x75, 0x65, 0x65, 0x7a, 0x65, 0x0a, 0x22, 0x0a, 0x06, 0x69, 0x64, 0x73, 0x5f, 0x33, 0x64, 0x12, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x01, 0xa0, 0x01, 0x02, 0x0a, 0x33, 0x0a, 0x08, 0x69, 0x6e, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x64, 0x69, 0x6d, 0x12, 0x09, 0x61, 0x75, 0x78, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x06, 0x43, 0x6f, 0x6e, 0x63, 0x61, 0x74, 0x2a, 0x0b, 0x0a, 0x04, 0x61, 0x78, 0x69, 0x73, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x25, 0x0a, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x0a, 0x09, 0x61, 0x75, 0x78, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x12, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x66, 0x33, 0x32, 0x22, 0x06, 0x45, 0x78, 0x70, 0x61, 0x6e, 0x64, 0x0a, 0x2d, 0x0a, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x66, 0x33, 0x32, 0x12, 0x11, 0x61, 0x75, 0x78, 0x5f, 0x68, 0x69, 0x64, 0x64, 0x65, 0x6e, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x73, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x10, 0xa0, 0x01, 0x02, 0x0a, 0x25, 0x0a, 0x0a, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x12, 0x0d, 0x70, 0x72, 0x65, 0x73, 0x65, 0x6e, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x12, 0x10, 0x62, 0x6f, 0x6c, 0x74, 0x5f, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x74, 0x61, 0x72, 0x67, 0x65, 0x74, 0x2a, 0x15, 0x08, 0x01, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x14, 0x42, 0x0a, 0x74, 0x77, 0x65, 0x6e, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x2a, 0x17, 0x08, 0x01, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x13, 0x42, 0x0c, 0x6e, 0x69, 0x6e, 0x65, 0x74, 0x65, 0x65, 0x6e, 0x5f, 0x69, 0x36, 0x34, 0x2a, 0x15, 0x08, 0x01, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x50, 0x42, 0x0a, 0x65, 0x69, 0x67, 0x68, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x2a, 0x0f, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x02, 0x42, 0x06, 0x74, 0x77, 0x6f, 0x5f, 0x31, 0x64, 0x2a, 0x13, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x02, 0x80, 0x01, 0x42, 0x09, 0x76, 0x6f, 0x63, 0x61, 0x62, 0x5f, 0x64, 0x69, 0x6d, 0x2a, 0x10, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x04, 0x42, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x64, 0x69, 0x6d, 0x5a, 0x1c, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x0f, 0x0a, 0x0d, 0x08, 0x07, 0x12, 0x09, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x03, 0x12, 0x01, 0x52, 0x5a, 0x17, 0x0a, 0x09, 0x73, 0x65, 0x71, 0x6c, 0x65, 0x6e, 0x73, 0x5f, 0x6b, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x06, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x5a, 0x17, 0x0a, 0x09, 0x73, 0x74, 0x61, 0x74, 0x65, 0x5f, 0x72, 0x6f, 0x77, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x06, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x5a, 0x1f, 0x0a, 0x15, 0x74, 0x6f, 0x74, 0x61, 0x6c, 0x5f, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x5f, 0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x12, 0x06, 0x0a, 0x04, 0x08, 0x06, 0x12, 0x00, 0x5a, 0x24, 0x0a, 0x0a, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x12, 0x16, 0x0a, 0x14, 0x08, 0x01, 0x12, 0x10, 0x0a, 0x02, 0x08, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x62, 0x1b, 0x0a, 0x08, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x0f, 0x0a, 0x0d, 0x08, 0x07, 0x12, 0x09, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x03, 0x12, 0x01, 0x52, 0x62, 0x1e, 0x0a, 0x06, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x12, 0x14, 0x0a, 0x12, 0x08, 0x10, 0x12, 0x0e, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x03, 0x12, 0x01, 0x52, 0x0a, 0x03, 0x08, 0x80, 0x01, 0x62, 0x28, 0x0a, 0x11, 0x61, 0x75, 0x78, 0x5f, 0x68, 0x69, 0x64, 0x64, 0x65, 0x6e, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x73, 0x12, 0x13, 0x0a, 0x11, 0x08, 0x10, 0x12, 0x0d, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x03, 0x12, 0x01, 0x52, 0x0a, 0x02, 0x08, 0x04, 0x62, 0x27, 0x0a, 0x0d, 0x70, 0x72, 0x65, 0x73, 0x65, 0x6e, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x12, 0x16, 0x0a, 0x14, 0x08, 0x01, 0x12, 0x10, 0x0a, 0x02, 0x08, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x12};
constexpr uint8_t kFixedR8TargetGraph[1044] = {0x08, 0x09, 0x3a, 0x89, 0x08, 0x0a, 0x27, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x0a, 0x0c, 0x6e, 0x69, 0x6e, 0x65, 0x74, 0x65, 0x65, 0x6e, 0x5f, 0x69, 0x36, 0x34, 0x12, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x73, 0x75, 0x62, 0x22, 0x03, 0x53, 0x75, 0x62, 0x0a, 0x30, 0x0a, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x73, 0x75, 0x62, 0x0a, 0x0a, 0x65, 0x69, 0x67, 0x68, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x12, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x6d, 0x6f, 0x64, 0x22, 0x03, 0x4d, 0x6f, 0x64, 0x2a, 0x0b, 0x0a, 0x04, 0x66, 0x6d, 0x6f, 0x64, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x29, 0x0a, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x6d, 0x6f, 0x64, 0x0a, 0x0a, 0x74, 0x77, 0x65, 0x6e, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x12, 0x0d, 0x69, 0x64, 0x73, 0x5f, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x62, 0x61, 0x73, 0x65, 0x22, 0x03, 0x41, 0x64, 0x64, 0x0a, 0x23, 0x0a, 0x0d, 0x69, 0x64, 0x73, 0x5f, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x62, 0x61, 0x73, 0x65, 0x12, 0x08, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x69, 0x64, 0x73, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x0a, 0x1c, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x08, 0x69, 0x6e, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x05, 0x53, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x38, 0x0a, 0x08, 0x69, 0x6e, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x09, 0x76, 0x6f, 0x63, 0x61, 0x62, 0x5f, 0x64, 0x69, 0x6d, 0x12, 0x0c, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x06, 0x43, 0x6f, 0x6e, 0x63, 0x61, 0x74, 0x2a, 0x0b, 0x0a, 0x04, 0x61, 0x78, 0x69, 0x73, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x2b, 0x0a, 0x0c, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x12, 0x0a, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x22, 0x0f, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x4f, 0x66, 0x53, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x25, 0x0a, 0x0a, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x12, 0x06, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x10, 0xa0, 0x01, 0x02, 0x0a, 0x26, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x0a, 0x06, 0x74, 0x77, 0x6f, 0x5f, 0x31, 0x64, 0x12, 0x06, 0x69, 0x64, 0x73, 0x5f, 0x33, 0x64, 0x22, 0x09, 0x55, 0x6e, 0x73, 0x71, 0x75, 0x65, 0x65, 0x7a, 0x65, 0x0a, 0x22, 0x0a, 0x06, 0x69, 0x64, 0x73, 0x5f, 0x33, 0x64, 0x12, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x01, 0xa0, 0x01, 0x02, 0x0a, 0x33, 0x0a, 0x08, 0x69, 0x6e, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x64, 0x69, 0x6d, 0x12, 0x09, 0x61, 0x75, 0x78, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x06, 0x43, 0x6f, 0x6e, 0x63, 0x61, 0x74, 0x2a, 0x0b, 0x0a, 0x04, 0x61, 0x78, 0x69, 0x73, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x25, 0x0a, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x0a, 0x09, 0x61, 0x75, 0x78, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x12, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x66, 0x33, 0x32, 0x22, 0x06, 0x45, 0x78, 0x70, 0x61, 0x6e, 0x64, 0x0a, 0x2d, 0x0a, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x66, 0x33, 0x32, 0x12, 0x11, 0x61, 0x75, 0x78, 0x5f, 0x68, 0x69, 0x64, 0x64, 0x65, 0x6e, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x73, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x10, 0xa0, 0x01, 0x02, 0x0a, 0x25, 0x0a, 0x0a, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x12, 0x0d, 0x70, 0x72, 0x65, 0x73, 0x65, 0x6e, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x12, 0x10, 0x62, 0x6f, 0x6c, 0x74, 0x5f, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x74, 0x61, 0x72, 0x67, 0x65, 0x74, 0x2a, 0x15, 0x08, 0x01, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x14, 0x42, 0x0a, 0x74, 0x77, 0x65, 0x6e, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x2a, 0x17, 0x08, 0x01, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x13, 0x42, 0x0c, 0x6e, 0x69, 0x6e, 0x65, 0x74, 0x65, 0x65, 0x6e, 0x5f, 0x69, 0x36, 0x34, 0x2a, 0x15, 0x08, 0x01, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x50, 0x42, 0x0a, 0x65, 0x69, 0x67, 0x68, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x2a, 0x0f, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x02, 0x42, 0x06, 0x74, 0x77, 0x6f, 0x5f, 0x31, 0x64, 0x2a, 0x13, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x02, 0x80, 0x01, 0x42, 0x09, 0x76, 0x6f, 0x63, 0x61, 0x62, 0x5f, 0x64, 0x69, 0x6d, 0x2a, 0x10, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x04, 0x42, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x64, 0x69, 0x6d, 0x5a, 0x1b, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x07, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x08, 0x5a, 0x17, 0x0a, 0x09, 0x73, 0x65, 0x71, 0x6c, 0x65, 0x6e, 0x73, 0x5f, 0x6b, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x06, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x5a, 0x17, 0x0a, 0x09, 0x73, 0x74, 0x61, 0x74, 0x65, 0x5f, 0x72, 0x6f, 0x77, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x06, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x5a, 0x1f, 0x0a, 0x15, 0x74, 0x6f, 0x74, 0x61, 0x6c, 0x5f, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x5f, 0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x12, 0x06, 0x0a, 0x04, 0x08, 0x06, 0x12, 0x00, 0x5a, 0x24, 0x0a, 0x0a, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x12, 0x16, 0x0a, 0x14, 0x08, 0x01, 0x12, 0x10, 0x0a, 0x02, 0x08, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x62, 0x1a, 0x0a, 0x08, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x07, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x08, 0x62, 0x1d, 0x0a, 0x06, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x12, 0x13, 0x0a, 0x11, 0x08, 0x10, 0x12, 0x0d, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x08, 0x0a, 0x03, 0x08, 0x80, 0x01, 0x62, 0x27, 0x0a, 0x11, 0x61, 0x75, 0x78, 0x5f, 0x68, 0x69, 0x64, 0x64, 0x65, 0x6e, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x73, 0x12, 0x12, 0x0a, 0x10, 0x08, 0x10, 0x12, 0x0c, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x08, 0x0a, 0x02, 0x08, 0x04, 0x62, 0x27, 0x0a, 0x0d, 0x70, 0x72, 0x65, 0x73, 0x65, 0x6e, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x12, 0x16, 0x0a, 0x14, 0x08, 0x01, 0x12, 0x10, 0x0a, 0x02, 0x08, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x12};
constexpr uint8_t kSymbolicRRejectingR16TargetGraph[1330] = {0x08, 0x09, 0x3a, 0xa7, 0x0a, 0x0a, 0x27, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x0a, 0x0c, 0x6e, 0x69, 0x6e, 0x65, 0x74, 0x65, 0x65, 0x6e, 0x5f, 0x69, 0x36, 0x34, 0x12, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x73, 0x75, 0x62, 0x22, 0x03, 0x53, 0x75, 0x62, 0x0a, 0x30, 0x0a, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x73, 0x75, 0x62, 0x0a, 0x0a, 0x65, 0x69, 0x67, 0x68, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x12, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x6d, 0x6f, 0x64, 0x22, 0x03, 0x4d, 0x6f, 0x64, 0x2a, 0x0b, 0x0a, 0x04, 0x66, 0x6d, 0x6f, 0x64, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x29, 0x0a, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x6d, 0x6f, 0x64, 0x0a, 0x0a, 0x74, 0x77, 0x65, 0x6e, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x12, 0x0d, 0x69, 0x64, 0x73, 0x5f, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x62, 0x61, 0x73, 0x65, 0x22, 0x03, 0x41, 0x64, 0x64, 0x0a, 0x1f, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x0b, 0x72, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x5f, 0x76, 0x65, 0x63, 0x22, 0x05, 0x53, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x3c, 0x0a, 0x0b, 0x72, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x5f, 0x76, 0x65, 0x63, 0x0a, 0x0a, 0x6f, 0x6e, 0x65, 0x5f, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x12, 0x0c, 0x72, 0x5f, 0x6c, 0x65, 0x6e, 0x5f, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x22, 0x06, 0x47, 0x61, 0x74, 0x68, 0x65, 0x72, 0x2a, 0x0b, 0x0a, 0x04, 0x61, 0x78, 0x69, 0x73, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x39, 0x0a, 0x0b, 0x7a, 0x65, 0x72, 0x6f, 0x5f, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x0a, 0x0c, 0x72, 0x5f, 0x6c, 0x65, 0x6e, 0x5f, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x0a, 0x0a, 0x6f, 0x6e, 0x65, 0x5f, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x12, 0x09, 0x72, 0x5f, 0x69, 0x6e, 0x64, 0x69, 0x63, 0x65, 0x73, 0x22, 0x05, 0x52, 0x61, 0x6e, 0x67, 0x65, 0x0a, 0x36, 0x0a, 0x08, 0x72, 0x38, 0x5f, 0x74, 0x61, 0x62, 0x6c, 0x65, 0x0a, 0x09, 0x72, 0x5f, 0x69, 0x6e, 0x64, 0x69, 0x63, 0x65, 0x73, 0x12, 0x0a, 0x72, 0x5f, 0x67, 0x75, 0x61, 0x72, 0x64, 0x5f, 0x31, 0x64, 0x22, 0x06, 0x47, 0x61, 0x74, 0x68, 0x65, 0x72, 0x2a, 0x0b, 0x0a, 0x04, 0x61, 0x78, 0x69, 0x73, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x2a, 0x0a, 0x0d, 0x69, 0x64, 0x73, 0x5f, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x62, 0x61, 0x73, 0x65, 0x0a, 0x0a, 0x72, 0x5f, 0x67, 0x75, 0x61, 0x72, 0x64, 0x5f, 0x31, 0x64, 0x12, 0x08, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x69, 0x64, 0x73, 0x22, 0x03, 0x41, 0x64, 0x64, 0x0a, 0x1c, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x08, 0x69, 0x6e, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x05, 0x53, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x38, 0x0a, 0x08, 0x69, 0x6e, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x09, 0x76, 0x6f, 0x63, 0x61, 0x62, 0x5f, 0x64, 0x69, 0x6d, 0x12, 0x0c, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x06, 0x43, 0x6f, 0x6e, 0x63, 0x61, 0x74, 0x2a, 0x0b, 0x0a, 0x04, 0x61, 0x78, 0x69, 0x73, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x2b, 0x0a, 0x0c, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x12, 0x0a, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x22, 0x0f, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x4f, 0x66, 0x53, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x25, 0x0a, 0x0a, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x12, 0x06, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x10, 0xa0, 0x01, 0x02, 0x0a, 0x26, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x0a, 0x06, 0x74, 0x77, 0x6f, 0x5f, 0x31, 0x64, 0x12, 0x06, 0x69, 0x64, 0x73, 0x5f, 0x33, 0x64, 0x22, 0x09, 0x55, 0x6e, 0x73, 0x71, 0x75, 0x65, 0x65, 0x7a, 0x65, 0x0a, 0x22, 0x0a, 0x06, 0x69, 0x64, 0x73, 0x5f, 0x33, 0x64, 0x12, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x01, 0xa0, 0x01, 0x02, 0x0a, 0x33, 0x0a, 0x08, 0x69, 0x6e, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x0a, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x64, 0x69, 0x6d, 0x12, 0x09, 0x61, 0x75, 0x78, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x06, 0x43, 0x6f, 0x6e, 0x63, 0x61, 0x74, 0x2a, 0x0b, 0x0a, 0x04, 0x61, 0x78, 0x69, 0x73, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x25, 0x0a, 0x07, 0x69, 0x64, 0x73, 0x5f, 0x66, 0x33, 0x32, 0x0a, 0x09, 0x61, 0x75, 0x78, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x12, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x66, 0x33, 0x32, 0x22, 0x06, 0x45, 0x78, 0x70, 0x61, 0x6e, 0x64, 0x0a, 0x2d, 0x0a, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x66, 0x33, 0x32, 0x12, 0x11, 0x61, 0x75, 0x78, 0x5f, 0x68, 0x69, 0x64, 0x64, 0x65, 0x6e, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x73, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x10, 0xa0, 0x01, 0x02, 0x0a, 0x25, 0x0a, 0x0a, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x12, 0x0d, 0x70, 0x72, 0x65, 0x73, 0x65, 0x6e, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x12, 0x10, 0x62, 0x6f, 0x6c, 0x74, 0x5f, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x74, 0x61, 0x72, 0x67, 0x65, 0x74, 0x2a, 0x15, 0x08, 0x01, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x14, 0x42, 0x0a, 0x74, 0x77, 0x65, 0x6e, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x2a, 0x17, 0x08, 0x01, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x13, 0x42, 0x0c, 0x6e, 0x69, 0x6e, 0x65, 0x74, 0x65, 0x65, 0x6e, 0x5f, 0x69, 0x36, 0x34, 0x2a, 0x15, 0x08, 0x01, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x50, 0x42, 0x0a, 0x65, 0x69, 0x67, 0x68, 0x74, 0x79, 0x5f, 0x69, 0x36, 0x34, 0x2a, 0x0f, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x02, 0x42, 0x06, 0x74, 0x77, 0x6f, 0x5f, 0x31, 0x64, 0x2a, 0x13, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x02, 0x80, 0x01, 0x42, 0x09, 0x76, 0x6f, 0x63, 0x61, 0x62, 0x5f, 0x64, 0x69, 0x6d, 0x2a, 0x10, 0x08, 0x01, 0x10, 0x07, 0x3a, 0x01, 0x04, 0x42, 0x07, 0x61, 0x75, 0x78, 0x5f, 0x64, 0x69, 0x6d, 0x2a, 0x12, 0x10, 0x07, 0x3a, 0x01, 0x00, 0x42, 0x0b, 0x7a, 0x65, 0x72, 0x6f, 0x5f, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x2a, 0x11, 0x10, 0x07, 0x3a, 0x01, 0x01, 0x42, 0x0a, 0x6f, 0x6e, 0x65, 0x5f, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x2a, 0x18, 0x08, 0x08, 0x10, 0x07, 0x3a, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x42, 0x08, 0x72, 0x38, 0x5f, 0x74, 0x61, 0x62, 0x6c, 0x65, 0x5a, 0x1c, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x0f, 0x0a, 0x0d, 0x08, 0x07, 0x12, 0x09, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x03, 0x12, 0x01, 0x52, 0x5a, 0x17, 0x0a, 0x09, 0x73, 0x65, 0x71, 0x6c, 0x65, 0x6e, 0x73, 0x5f, 0x6b, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x06, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x5a, 0x17, 0x0a, 0x09, 0x73, 0x74, 0x61, 0x74, 0x65, 0x5f, 0x72, 0x6f, 0x77, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x06, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x5a, 0x1f, 0x0a, 0x15, 0x74, 0x6f, 0x74, 0x61, 0x6c, 0x5f, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x5f, 0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x12, 0x06, 0x0a, 0x04, 0x08, 0x06, 0x12, 0x00, 0x5a, 0x24, 0x0a, 0x0a, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x12, 0x16, 0x0a, 0x14, 0x08, 0x01, 0x12, 0x10, 0x0a, 0x02, 0x08, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x62, 0x1b, 0x0a, 0x08, 0x70, 0x72, 0x65, 0x64, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x0f, 0x0a, 0x0d, 0x08, 0x07, 0x12, 0x09, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x03, 0x12, 0x01, 0x52, 0x62, 0x1e, 0x0a, 0x06, 0x6c, 0x6f, 0x67, 0x69, 0x74, 0x73, 0x12, 0x14, 0x0a, 0x12, 0x08, 0x10, 0x12, 0x0e, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x03, 0x12, 0x01, 0x52, 0x0a, 0x03, 0x08, 0x80, 0x01, 0x62, 0x28, 0x0a, 0x11, 0x61, 0x75, 0x78, 0x5f, 0x68, 0x69, 0x64, 0x64, 0x65, 0x6e, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x73, 0x12, 0x13, 0x0a, 0x11, 0x08, 0x10, 0x12, 0x0d, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x03, 0x12, 0x01, 0x52, 0x0a, 0x02, 0x08, 0x04, 0x62, 0x27, 0x0a, 0x0d, 0x70, 0x72, 0x65, 0x73, 0x65, 0x6e, 0x74, 0x5f, 0x73, 0x73, 0x6d, 0x2e, 0x30, 0x12, 0x16, 0x0a, 0x14, 0x08, 0x01, 0x12, 0x10, 0x0a, 0x02, 0x08, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x12};

constexpr uint8_t kSyntheticDflash2Graph[1075] = {0x08, 0x09, 0x3a, 0xa8, 0x08, 0x0a, 0x31, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x0a, 0x04, 0x69, 0x64, 0x78, 0x30, 0x12, 0x09, 0x61, 0x6e, 0x63, 0x68, 0x6f, 0x72, 0x5f, 0x31, 0x64, 0x22, 0x06, 0x47, 0x61, 0x74, 0x68, 0x65, 0x72, 0x2a, 0x0b, 0x0a, 0x04, 0x61, 0x78, 0x69, 0x73, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x0a, 0x28, 0x0a, 0x09, 0x61, 0x6e, 0x63, 0x68, 0x6f, 0x72, 0x5f, 0x31, 0x64, 0x12, 0x0a, 0x61, 0x6e, 0x63, 0x68, 0x6f, 0x72, 0x5f, 0x69, 0x33, 0x32, 0x22, 0x04, 0x43, 0x61, 0x73, 0x74, 0x2a, 0x09, 0x0a, 0x02, 0x74, 0x6f, 0x18, 0x06, 0xa0, 0x01, 0x02, 0x0a, 0x35, 0x0a, 0x0a, 0x61, 0x6e, 0x63, 0x68, 0x6f, 0x72, 0x5f, 0x69, 0x33, 0x32, 0x0a, 0x0d, 0x64, 0x72, 0x61, 0x66, 0x74, 0x5f, 0x6f, 0x66, 0x66, 0x73, 0x65, 0x74, 0x73, 0x12, 0x13, 0x64, 0x72, 0x61, 0x66, 0x74, 0x5f, 0x63, 0x61, 0x6e, 0x64, 0x69, 0x64, 0x61, 0x74, 0x65, 0x5f, 0x69, 0x64, 0x73, 0x22, 0x03, 0x41, 0x64, 0x64, 0x0a, 0x25, 0x0a, 0x0b, 0x7a, 0x65, 0x72, 0x6f, 0x5f, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x12, 0x0c, 0x64, 0x72, 0x61, 0x66, 0x74, 0x5f, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x0a, 0x30, 0x0a, 0x15, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x6b, 0x65, 0x79, 0x5f, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x2e, 0x30, 0x2e, 0x6b, 0x65, 0x79, 0x12, 0x0d, 0x70, 0x72, 0x65, 0x73, 0x65, 0x6e, 0x74, 0x2e, 0x30, 0x2e, 0x6b, 0x65, 0x79, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x0a, 0x34, 0x0a, 0x17, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x6b, 0x65, 0x79, 0x5f, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x2e, 0x30, 0x2e, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x12, 0x0f, 0x70, 0x72, 0x65, 0x73, 0x65, 0x6e, 0x74, 0x2e, 0x30, 0x2e, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x12, 0x11, 0x62, 0x6f, 0x6c, 0x74, 0x5f, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x64, 0x66, 0x6c, 0x61, 0x73, 0x68, 0x32, 0x2a, 0x14, 0x08, 0x01, 0x10, 0x07, 0x42, 0x04, 0x69, 0x64, 0x78, 0x30, 0x4a, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2a, 0x35, 0x08, 0x01, 0x08, 0x07, 0x08, 0x01, 0x10, 0x06, 0x42, 0x0d, 0x64, 0x72, 0x61, 0x66, 0x74, 0x5f, 0x6f, 0x66, 0x66, 0x73, 0x65, 0x74, 0x73, 0x4a, 0x1c, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x65, 0x00, 0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00, 0x2a, 0x35, 0x08, 0x01, 0x08, 0x07, 0x08, 0x01, 0x08, 0x01, 0x10, 0x01, 0x42, 0x0b, 0x7a, 0x65, 0x72, 0x6f, 0x5f, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x4a, 0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5a, 0x24, 0x0a, 0x11, 0x61, 0x75, 0x78, 0x5f, 0x68, 0x69, 0x64, 0x64, 0x65, 0x6e, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x73, 0x12, 0x0f, 0x0a, 0x0d, 0x08, 0x10, 0x12, 0x09, 0x0a, 0x03, 0x12, 0x01, 0x6e, 0x0a, 0x02, 0x08, 0x04, 0x5a, 0x18, 0x0a, 0x09, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x12, 0x05, 0x0a, 0x03, 0x12, 0x01, 0x6d, 0x5a, 0x19, 0x0a, 0x09, 0x71, 0x5f, 0x72, 0x6f, 0x77, 0x5f, 0x6d, 0x61, 0x70, 0x12, 0x0c, 0x0a, 0x0a, 0x08, 0x06, 0x12, 0x06, 0x0a, 0x04, 0x12, 0x02, 0x71, 0x72, 0x5a, 0x1d, 0x0a, 0x0b, 0x71, 0x6b, 0x76, 0x5f, 0x72, 0x6f, 0x77, 0x5f, 0x6d, 0x61, 0x70, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x06, 0x12, 0x08, 0x0a, 0x06, 0x12, 0x04, 0x71, 0x6b, 0x76, 0x72, 0x5a, 0x1f, 0x0a, 0x0f, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x5f, 0x72, 0x6f, 0x77, 0x5f, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x12, 0x0c, 0x0a, 0x0a, 0x08, 0x06, 0x12, 0x06, 0x0a, 0x04, 0x12, 0x02, 0x62, 0x72, 0x5a, 0x1f, 0x0a, 0x10, 0x61, 0x6e, 0x63, 0x68, 0x6f, 0x72, 0x5f, 0x72, 0x6f, 0x77, 0x5f, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x12, 0x0b, 0x0a, 0x09, 0x08, 0x06, 0x12, 0x05, 0x0a, 0x03, 0x12, 0x01, 0x71, 0x5a, 0x2c, 0x0a, 0x1b, 0x63, 0x75, 0x6d, 0x75, 0x6c, 0x61, 0x74, 0x69, 0x76, 0x65, 0x5f, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x5f, 0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x73, 0x12, 0x0d, 0x0a, 0x0b, 0x08, 0x06, 0x12, 0x07, 0x0a, 0x05, 0x12, 0x03, 0x63, 0x73, 0x6c, 0x5a, 0x26, 0x0a, 0x15, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x5f, 0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x73, 0x12, 0x0d, 0x0a, 0x0b, 0x08, 0x06, 0x12, 0x07, 0x0a, 0x05, 0x12, 0x03, 0x70, 0x73, 0x6c, 0x5a, 0x23, 0x0a, 0x0b, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x5f, 0x74, 0x61, 0x62, 0x6c, 0x65, 0x12, 0x14, 0x0a, 0x12, 0x08, 0x06, 0x12, 0x0e, 0x0a, 0x05, 0x12, 0x03, 0x62, 0x74, 0x30, 0x0a, 0x05, 0x12, 0x03, 0x62, 0x74, 0x31, 0x5a, 0x20, 0x0a, 0x12, 0x61, 0x74, 0x74, 0x65, 0x6e, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x06, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x03, 0x5a, 0x31, 0x0a, 0x15, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x6b, 0x65, 0x79, 0x5f, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x2e, 0x30, 0x2e, 0x6b, 0x65, 0x79, 0x12, 0x18, 0x0a, 0x16, 0x08, 0x01, 0x12, 0x12, 0x0a, 0x03, 0x12, 0x01, 0x62, 0x0a, 0x03, 0x08, 0x80, 0x02, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x5a, 0x33, 0x0a, 0x17, 0x70, 0x61, 0x73, 0x74, 0x5f, 0x6b, 0x65, 0x79, 0x5f, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x2e, 0x30, 0x2e, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x12, 0x18, 0x0a, 0x16, 0x08, 0x01, 0x12, 0x12, 0x0a, 0x03, 0x12, 0x01, 0x62, 0x0a, 0x03, 0x08, 0x80, 0x02, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x62, 0x2a, 0x0a, 0x13, 0x64, 0x72, 0x61, 0x66, 0x74, 0x5f, 0x63, 0x61, 0x6e, 0x64, 0x69, 0x64, 0x61, 0x74, 0x65, 0x5f, 0x69, 0x64, 0x73, 0x12, 0x13, 0x0a, 0x11, 0x08, 0x06, 0x12, 0x0d, 0x0a, 0x03, 0x12, 0x01, 0x71, 0x0a, 0x02, 0x08, 0x07, 0x0a, 0x02, 0x08, 0x01, 0x62, 0x27, 0x0a, 0x0c, 0x64, 0x72, 0x61, 0x66, 0x74, 0x5f, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x12, 0x17, 0x0a, 0x15, 0x08, 0x01, 0x12, 0x11, 0x0a, 0x03, 0x12, 0x01, 0x71, 0x0a, 0x02, 0x08, 0x07, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x62, 0x29, 0x0a, 0x0d, 0x70, 0x72, 0x65, 0x73, 0x65, 0x6e, 0x74, 0x2e, 0x30, 0x2e, 0x6b, 0x65, 0x79, 0x12, 0x18, 0x0a, 0x16, 0x08, 0x01, 0x12, 0x12, 0x0a, 0x03, 0x12, 0x01, 0x62, 0x0a, 0x03, 0x08, 0x80, 0x02, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x62, 0x2b, 0x0a, 0x0f, 0x70, 0x72, 0x65, 0x73, 0x65, 0x6e, 0x74, 0x2e, 0x30, 0x2e, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x12, 0x18, 0x0a, 0x16, 0x08, 0x01, 0x12, 0x12, 0x0a, 0x03, 0x12, 0x01, 0x62, 0x0a, 0x03, 0x08, 0x80, 0x02, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x12};

TEST(NemotronBolt, GeneratorStateFrozenR16PolicyBoundaryAnd32KExecution) {
  const std::vector<int32_t> prompt35 = BuildBoltBasePrompt();
  std::vector<int32_t> synthetic_prompt35(prompt35.size());
  for (size_t i = 0; i < prompt35.size(); ++i) {
    synthetic_prompt35[i] = static_cast<int32_t>(20 + (i % 70));
  }

  // 1. Symbolic-R graph whose operator rejects R=16:
  //    - Without BOLT_DISABLE_R16_PREFILL=1, Generator construction fails during R=16 warmup.
  //    - With BOLT_DISABLE_R16_PREFILL=1, Generator construction and AppendTokens bypass R=16
  //      completely (0 R=16 warmup calls, 0 R=16 prefill calls) and remain stable even if the
  //      environment variable is unset after construction or across RewindToLength(0).
  auto r8_only_sym_model = NemotronBoltTestAccess::CreateSyntheticBoltModel(
      kSymbolicRRejectingR16TargetGraph, sizeof(kSymbolicRRejectingR16TargetGraph),
      kSyntheticDflash2Graph, sizeof(kSyntheticDflash2Graph));
  {
    ScopedBoltEnvVar unset_env("BOLT_DISABLE_R16_PREFILL", std::nullopt);
    auto params = std::make_shared<GeneratorParams>(*r8_only_sym_model);
    params->use_graph_capture = true;
    params->search.max_length = static_cast<int>(synthetic_prompt35.size() + 16);
    EXPECT_THROW((Generator{*r8_only_sym_model, *params}), std::exception);
  }
  std::vector<int32_t> baseline35;
  {
    ScopedBoltEnvVar disable_env("BOLT_DISABLE_R16_PREFILL", std::string{"1"});
    auto params = std::make_shared<GeneratorParams>(*r8_only_sym_model);
    params->use_graph_capture = true;
    params->search.max_length = static_cast<int>(synthetic_prompt35.size() + 16);
    Generator disabled_gen{*r8_only_sym_model, *params};
    auto& state = dynamic_cast<NemotronBoltState&>(*disabled_gen.state_);
    EXPECT_TRUE(NemotronBoltTestAccess::ModelSupportsR16(state));
    EXPECT_FALSE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_FALSE(NemotronBoltTestAccess::HasR16ProfileTensors(state));
    EXPECT_EQ(NemotronBoltTestAccess::VerifyInputsSpanSize(state), 2 * kBoltVerifyRows + 8);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR8Calls(state), 4u);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR16Calls(state), 0u);
    const std::vector<int32_t> dummy16(kBoltPrefillRows16, 20);
    EXPECT_THROW(NemotronBoltTestAccess::InvokePrefillChunk16ForTest(state, dummy16), std::runtime_error);

    // Mutate environment after construction: existing generator must keep its frozen disabled policy.
    ScopedBoltEnvVar::Apply("BOLT_DISABLE_R16_PREFILL", std::nullopt);
    disabled_gen.AppendTokens(synthetic_prompt35);
    EXPECT_FALSE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR16Calls(state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 5u);
    EXPECT_EQ(NemotronBoltTestAccess::Length(state), synthetic_prompt35.size());
    EXPECT_EQ(NemotronBoltTestAccess::StateRow(state), 2);
    while (!disabled_gen.IsDone()) disabled_gen.GenerateNextToken();
    const auto seq = disabled_gen.GetSequence(0).CopyDeviceToCpu();
    baseline35.assign(seq.begin(), seq.end());
    ASSERT_EQ(baseline35.size(), synthetic_prompt35.size() + 16);

    disabled_gen.RewindToLength(0);
    EXPECT_FALSE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 0u);
    disabled_gen.AppendTokens(synthetic_prompt35);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 5u);
    while (!disabled_gen.IsDone()) disabled_gen.GenerateNextToken();
    const auto seq_reset = disabled_gen.GetSequence(0).CopyDeviceToCpu();
    EXPECT_EQ(std::vector<int32_t>(seq_reset.begin(), seq_reset.end()), baseline35);
  }

  // 2. Symbolic-R model supporting both R=8 and R=16:
  //    Verify enabled construction, post-construction env change stability ("1" set after construction),
  //    reset preservation, and recreation with new env.
  auto sym_model = NemotronBoltTestAccess::CreateSyntheticBoltModel(
      kSymbolicRTargetGraph, sizeof(kSymbolicRTargetGraph),
      kSyntheticDflash2Graph, sizeof(kSyntheticDflash2Graph));
  {
    ScopedBoltEnvVar enable_env("BOLT_DISABLE_R16_PREFILL", std::nullopt);
    auto params = std::make_shared<GeneratorParams>(*sym_model);
    params->use_graph_capture = true;
    params->search.max_length = static_cast<int>(synthetic_prompt35.size() + 16);
    Generator enabled_gen{*sym_model, *params};
    auto& state = dynamic_cast<NemotronBoltState&>(*enabled_gen.state_);
    EXPECT_TRUE(NemotronBoltTestAccess::ModelSupportsR16(state));
    EXPECT_TRUE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_TRUE(NemotronBoltTestAccess::HasR16ProfileTensors(state));
    EXPECT_EQ(NemotronBoltTestAccess::VerifyInputsSpanSize(state), 2 * kBoltPrefillRows16 + 8);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR8Calls(state), 4u);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR16Calls(state), 4u);

    // Mutate environment to "1" after construction: existing generator keeps R=16 enabled.
    ScopedBoltEnvVar::Apply("BOLT_DISABLE_R16_PREFILL", std::string{"1"});
    enabled_gen.AppendTokens(synthetic_prompt35);
    EXPECT_TRUE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 2u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 1u);
    EXPECT_EQ(NemotronBoltTestAccess::StateRow(state), 2);
    while (!enabled_gen.IsDone()) enabled_gen.GenerateNextToken();
    const auto seq = enabled_gen.GetSequence(0).CopyDeviceToCpu();
    EXPECT_EQ(std::vector<int32_t>(seq.begin(), seq.end()), baseline35);

    enabled_gen.RewindToLength(0);
    enabled_gen.AppendTokens(synthetic_prompt35);
    EXPECT_TRUE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 2u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 1u);

    // A newly constructed generator while BOLT_DISABLE_R16_PREFILL=1 is active captures disabled policy.
    Generator recreated_disabled{*sym_model, *params};
    auto& rec_state = dynamic_cast<NemotronBoltState&>(*recreated_disabled.state_);
    EXPECT_FALSE(NemotronBoltTestAccess::R16Enabled(rec_state));
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR16Calls(rec_state), 0u);
    recreated_disabled.AppendTokens(synthetic_prompt35);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(rec_state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(rec_state), 5u);
  }

  // 3. Fixed-R=8 model compatibility: never allocates or executes R=16 even when switch is unset.
  auto fixed_model = NemotronBoltTestAccess::CreateSyntheticBoltModel(
      kFixedR8TargetGraph, sizeof(kFixedR8TargetGraph),
      kSyntheticDflash2Graph, sizeof(kSyntheticDflash2Graph));
  {
    ScopedBoltEnvVar enable_env("BOLT_DISABLE_R16_PREFILL", std::nullopt);
    auto params = std::make_shared<GeneratorParams>(*fixed_model);
    params->use_graph_capture = true;
    params->search.max_length = static_cast<int>(synthetic_prompt35.size() + 16);
    Generator fixed_gen{*fixed_model, *params};
    auto& state = dynamic_cast<NemotronBoltState&>(*fixed_gen.state_);
    EXPECT_FALSE(NemotronBoltTestAccess::ModelSupportsR16(state));
    EXPECT_FALSE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_FALSE(NemotronBoltTestAccess::HasR16ProfileTensors(state));
    EXPECT_EQ(NemotronBoltTestAccess::VerifyInputsSpanSize(state), 2 * kBoltVerifyRows + 8);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR8Calls(state), 4u);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR16Calls(state), 0u);
    fixed_gen.AppendTokens(synthetic_prompt35);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 5u);
    while (!fixed_gen.IsDone()) fixed_gen.GenerateNextToken();
    const auto seq = fixed_gen.GetSequence(0).CopyDeviceToCpu();
    EXPECT_EQ(std::vector<int32_t>(seq.begin(), seq.end()), baseline35);
  }

  // 4. Boundary lengths (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1031, 1032, 1033) plus 32,768 tokens
  //    through Generator::AppendTokens comparing disabled R=8 vs enabled R=16.
  std::vector<int32_t> long_source(32768);
  for (size_t i = 0; i < long_source.size(); ++i) {
    long_source[i] = static_cast<int32_t>(20 + (i % 70));
  }
  for (size_t len : {size_t{1}, size_t{7}, size_t{8}, size_t{9}, size_t{15}, size_t{16},
                     size_t{17}, size_t{31}, size_t{32}, size_t{33}, size_t{1031},
                     size_t{1032}, size_t{1033}, size_t{32768}}) {
    const std::span<const int32_t> sub_prompt(long_source.data(), len);
    const auto r8_rec = ExecuteBoltPromptRun(*sym_model, sub_prompt, 24, /*disable_r16=*/true);
    const auto r16_rec = ExecuteBoltPromptRun(*sym_model, sub_prompt, 24, /*disable_r16=*/false);

    const size_t expected_r8_prefill = (len + 7) / 8;
    const size_t expected_r16_chunks = len / 16;
    const size_t rem = len % 16;
    const size_t expected_r16_tail_r8 = (rem == 0) ? 0 : (rem <= 8 ? 1 : 2);
    const int32_t expected_state_row = (rem == 0) ? 7 : static_cast<int32_t>((rem - 1) % 8);
    const auto expected_aux_win = PlanNemotronBoltPrefillAuxWindow(len, 1032, 8);

    EXPECT_EQ(r8_rec.warmup_r8_calls, 4u) << "len=" << len;
    EXPECT_EQ(r8_rec.warmup_r16_calls, 0u) << "len=" << len;
    EXPECT_EQ(r8_rec.prefill_r8_calls, expected_r8_prefill) << "len=" << len;
    EXPECT_EQ(r8_rec.prefill_r16_calls, 0u) << "len=" << len;

    EXPECT_EQ(r16_rec.warmup_r8_calls, 4u) << "len=" << len;
    EXPECT_EQ(r16_rec.warmup_r16_calls, 4u) << "len=" << len;
    EXPECT_EQ(r16_rec.prefill_r16_calls, expected_r16_chunks) << "len=" << len;
    EXPECT_EQ(r16_rec.prefill_r8_calls, expected_r16_tail_r8) << "len=" << len;

    if (len == 32768) {
      EXPECT_EQ(r8_rec.prefill_r8_calls, 4096u);
      EXPECT_EQ(r16_rec.prefill_r16_calls, 2048u);
      EXPECT_EQ(r16_rec.prefill_r8_calls, 0u);
    }

    EXPECT_EQ(r8_rec.post_prefill_length, len) << "len=" << len;
    EXPECT_EQ(r16_rec.post_prefill_length, len) << "len=" << len;
    EXPECT_EQ(r8_rec.post_prefill_state_row, expected_state_row) << "len=" << len;
    EXPECT_EQ(r16_rec.post_prefill_state_row, expected_state_row) << "len=" << len;
    EXPECT_EQ(r16_rec.post_prefill_anchor, r8_rec.post_prefill_anchor) << "len=" << len;
    EXPECT_EQ(r8_rec.last_prefill_draft_start, expected_aux_win.draft_start) << "len=" << len;
    EXPECT_EQ(r8_rec.last_prefill_draft_rows, expected_aux_win.draft_rows) << "len=" << len;
    EXPECT_EQ(r16_rec.last_prefill_draft_start, expected_aux_win.draft_start) << "len=" << len;
    EXPECT_EQ(r16_rec.last_prefill_draft_rows, expected_aux_win.draft_rows) << "len=" << len;
    EXPECT_EQ(r16_rec.generated_tokens, r8_rec.generated_tokens) << "len=" << len;
    EXPECT_EQ(r16_rec.acceptance_trace, r8_rec.acceptance_trace) << "len=" << len;
  }
}

// Optional real-graph correctness coverage. No models or GPU are needed for the tests above.
TEST(NemotronBoltGPU, SequentialGeneratorsResetAndPublicBoundaries) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) {
    if (RequireGpuQualificationEnv()) {
      FAIL() << "ORTGENAI_BOLT_TEST_MODEL is required when ORTGENAI_BOLT_REQUIRE_GPU_QUALIFICATION=1";
    }
    GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native symbolic-R Bolt package";
  }
  auto model = CreateModel(GetOrtEnv(), path);
  const auto prompt = BuildBoltBasePrompt();
  std::vector<int32_t> baseline;

  // 1. Verify disabled construction (BOLT_DISABLE_R16_PREFILL=1) allocates no R=16 tensors,
  //    runs 0 R=16 warmup calls during construction, uses coherent R=8 input buffer layout,
  //    and preserves its disabled policy even if the environment variable is unset before AppendTokens
  //    or before RewindToLength(0) + AppendTokens.
  {
    ScopedBoltEnvVar disable_env("BOLT_DISABLE_R16_PREFILL", std::string{"1"});
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size() + 32);
    Generator disabled_gen{*model, *params};
    auto& state = dynamic_cast<NemotronBoltState&>(*disabled_gen.state_);
    ASSERT_TRUE(NemotronBoltTestAccess::ModelSupportsR16(state))
        << "ORTGENAI_BOLT_TEST_MODEL must be a symbolic-R target package";
    EXPECT_FALSE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_FALSE(NemotronBoltTestAccess::HasR16ProfileTensors(state));
    EXPECT_EQ(NemotronBoltTestAccess::VerifyInputsSpanSize(state), 2 * kBoltVerifyRows + 8);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR8Calls(state), 4u);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR16Calls(state), 0u);
    const std::vector<int32_t> dummy16(kBoltPrefillRows16, 1);
    EXPECT_THROW(NemotronBoltTestAccess::InvokePrefillChunk16ForTest(state, dummy16), std::runtime_error);

    // Mutate environment after construction: existing generator must keep its frozen disabled policy.
    ScopedBoltEnvVar::Apply("BOLT_DISABLE_R16_PREFILL", std::nullopt);
    disabled_gen.AppendTokens(prompt);
    EXPECT_FALSE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR16Calls(state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 5u);
    EXPECT_EQ(NemotronBoltTestAccess::Length(state), prompt.size());
    EXPECT_EQ(NemotronBoltTestAccess::StateRow(state), 2);
    while (!disabled_gen.IsDone()) disabled_gen.GenerateNextToken();
    const auto seq = disabled_gen.GetSequence(0).CopyDeviceToCpu();
    baseline.assign(seq.begin(), seq.end());
    ASSERT_EQ(baseline.size(), prompt.size() + 32);

    // Reset must preserve the generator's frozen disabled policy even while env remains unset.
    disabled_gen.RewindToLength(0);
    EXPECT_FALSE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 0u);
    disabled_gen.AppendTokens(prompt);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 5u);
    while (!disabled_gen.IsDone()) disabled_gen.GenerateNextToken();
    const auto seq_after_reset = disabled_gen.GetSequence(0).CopyDeviceToCpu();
    EXPECT_EQ(std::vector<int32_t>(seq_after_reset.begin(), seq_after_reset.end()), baseline);
  }

  // 2. Verify enabled construction (BOLT_DISABLE_R16_PREFILL unset) warms R=16 4 times,
  //    executes 2 R=16 chunks + 1 R=8 tail for length 35, and preserves its enabled policy
  //    even if BOLT_DISABLE_R16_PREFILL=1 is set after construction or across RewindToLength(0).
  for (int request = 0; request < 2; ++request) {
    ScopedBoltEnvVar enable_env("BOLT_DISABLE_R16_PREFILL", std::nullopt);
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size() + 32);
    Generator generator{*model, *params};
    auto& state = dynamic_cast<NemotronBoltState&>(*generator.state_);
    ASSERT_TRUE(NemotronBoltTestAccess::ModelSupportsR16(state));
    EXPECT_TRUE(NemotronBoltTestAccess::R16Enabled(state));
    EXPECT_TRUE(NemotronBoltTestAccess::HasR16ProfileTensors(state));
    EXPECT_EQ(NemotronBoltTestAccess::VerifyInputsSpanSize(state), 2 * kBoltPrefillRows16 + 8);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR8Calls(state), 4u);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR16Calls(state), 4u);

    // Mutate environment after construction to "1": existing enabled generator must keep R=16 policy.
    ScopedBoltEnvVar::Apply("BOLT_DISABLE_R16_PREFILL", std::string{"1"});

    for (int reset = 0; reset < 2; ++reset) {
      if (reset == 1) {
        generator.AppendTokens(prompt);
        generator.GenerateNextToken();
        generator.GenerateNextToken();
        EXPECT_GT(generator.GetSpeculativeStats().tokens_buffered, 0);
        generator.RewindToLength(0);
      }
      generator.AppendTokens(prompt);
      EXPECT_TRUE(NemotronBoltTestAccess::R16Enabled(state));
      EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(state), 2u);
      EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(state), 1u);
      EXPECT_EQ(NemotronBoltTestAccess::Length(state), prompt.size());
      EXPECT_EQ(NemotronBoltTestAccess::StateRow(state), 2);
      EXPECT_THROW(generator.AppendTokens(prompt), std::runtime_error);
      EXPECT_THROW(generator.GetLogits(), std::runtime_error);
      EXPECT_THROW(generator.RewindToLength(1), std::runtime_error);
      EXPECT_EQ(generator.GetSequence(0).CopyDeviceToCpu().size(), prompt.size());
      while (!generator.IsDone()) {
        const size_t before = generator.TokenCount();
        generator.GenerateNextToken();
        EXPECT_EQ(generator.TokenCount(), before + 1);
      }
      auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
      const std::vector<int32_t> result(sequence.begin(), sequence.end());
      EXPECT_EQ(result, baseline);
      EXPECT_EQ(result.size(), prompt.size() + 32);
      const auto stats = generator.GetSpeculativeStats();
      EXPECT_EQ(stats.tokens_emitted, 32);
      EXPECT_EQ(stats.tokens_queued, 32);
      EXPECT_EQ(stats.tokens_buffered, 0);
      EXPECT_GT(stats.full_accept_rounds, 0);
      EXPECT_GT(stats.partial_accept_rounds, 0);
      EXPECT_EQ(NemotronBoltTestAccess::VerifyR8Calls(state), stats.target_verify_forward_passes);
      generator.RewindToLength(0);
      EXPECT_EQ(generator.TokenCount(), 0);
    }
  }
  for (int budget : {0, 1, 2, 3, 7, 8, 9}) {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size()) + budget;
    Generator generator{*model, *params};
    EXPECT_THROW(generator.AppendTokens(std::vector<int32_t>{-1}), std::runtime_error);
    EXPECT_EQ(generator.TokenCount(), 0);
    generator.AppendTokens(prompt);
    while (!generator.IsDone()) generator.GenerateNextToken();
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    EXPECT_EQ(sequence.size(), prompt.size() + budget);
    EXPECT_TRUE(std::equal(sequence.begin(), sequence.end(), baseline.begin()));
    EXPECT_THROW(generator.GenerateNextToken(), std::runtime_error);
  }
  {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size()) + 32;
    {
      Generator serial{*model, *params};
      serial.AppendTokens(prompt);
      const auto target_only = NemotronBoltTestAccess::TargetOnly(
          dynamic_cast<NemotronBoltState&>(*serial.state_), 32);
      EXPECT_TRUE(std::equal(target_only.begin(), target_only.end(), baseline.begin() + prompt.size()));
    }
    {
      Generator generator{*model, *params};
      generator.AppendTokens(prompt);
      generator.GenerateNextToken();
      NemotronBoltTestAccess::ForceRejectedDrafts(dynamic_cast<NemotronBoltState&>(*generator.state_));
      generator.GenerateNextToken();
      EXPECT_EQ(generator.GetSpeculativeStats().zero_accept_rounds, 1);
      while (!generator.IsDone()) generator.GenerateNextToken();
      const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
      EXPECT_TRUE(std::equal(sequence.begin(), sequence.end(), baseline.begin()));
    }
  }

  nlohmann::json raw_records_json = nlohmann::json::object();
  raw_records_json["boundary_runs"] = nlohmann::json::array();

  auto record_to_json = [](size_t len, const BoltRunRecord& r8, const BoltRunRecord& r16) {
    auto serialize_one = [](const BoltRunRecord& r) {
      nlohmann::json trace = nlohmann::json::array();
      for (const auto& s : r.acceptance_trace) {
        trace.push_back({
            {"rounds", s.rounds},
            {"draft_tokens_proposed", s.draft_tokens_proposed},
            {"draft_tokens_evaluated", s.draft_tokens_evaluated},
            {"draft_tokens_accepted", s.draft_tokens_accepted},
            {"full_accept_rounds", s.full_accept_rounds},
            {"partial_accept_rounds", s.partial_accept_rounds},
            {"zero_accept_rounds", s.zero_accept_rounds},
            {"histogram", s.histogram},
        });
      }
      return nlohmann::json{
          {"warmup_r8_calls", r.warmup_r8_calls},
          {"warmup_r16_calls", r.warmup_r16_calls},
          {"prefill_r8_calls", r.prefill_r8_calls},
          {"prefill_r16_calls", r.prefill_r16_calls},
          {"verify_r8_calls", r.verify_r8_calls},
          {"post_prefill_length", r.post_prefill_length},
          {"post_prefill_state_row", r.post_prefill_state_row},
          {"post_prefill_anchor", r.post_prefill_anchor},
          {"last_prefill_draft_start", r.last_prefill_draft_start},
          {"last_prefill_draft_rows", r.last_prefill_draft_rows},
          {"post_prefill_ssm0_slice", r.post_prefill_ssm0_slice},
          {"post_prefill_logits_slice", r.post_prefill_logits_slice},
          {"generated_tokens", r.generated_tokens},
          {"acceptance_trace", trace},
      };
    };
    float max_ssm_abs_diff = 0.0f;
    for (size_t i = 0; i < std::min(r8.post_prefill_ssm0_slice.size(), r16.post_prefill_ssm0_slice.size()); ++i) {
      max_ssm_abs_diff = std::max(max_ssm_abs_diff,
                                  std::abs(r8.post_prefill_ssm0_slice[i] - r16.post_prefill_ssm0_slice[i]));
    }
    float max_logits_abs_diff = 0.0f;
    for (size_t i = 0; i < std::min(r8.post_prefill_logits_slice.size(), r16.post_prefill_logits_slice.size()); ++i) {
      max_logits_abs_diff = std::max(
          max_logits_abs_diff, std::abs(r8.post_prefill_logits_slice[i] - r16.post_prefill_logits_slice[i]));
    }
    return nlohmann::json{
        {"prompt_length", len},
        {"tokens_equal", r8.generated_tokens == r16.generated_tokens},
        {"acceptance_trace_equal", r8.acceptance_trace == r16.acceptance_trace},
        {"post_prefill_anchor_equal", r8.post_prefill_anchor == r16.post_prefill_anchor},
        {"max_ssm0_selected_row_abs_diff", max_ssm_abs_diff},
        {"max_last_logits_abs_diff", max_logits_abs_diff},
        {"r8", serialize_one(r8)},
        {"r16", serialize_one(r16)},
    };
  };

  // 3. Runnable boundary parity regression across 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1031, 1032, 1033
  //    comparing disabled R=8 and enabled R=16 through Generator::AppendTokens on the same model.
  const std::vector<size_t> boundary_lengths{1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1031, 1032, 1033};
  const auto boundary_source = LoadBoltPromptTokens(boundary_lengths.back());
  for (size_t len : boundary_lengths) {
    const std::span<const int32_t> sub_prompt(boundary_source.data(), len);
    const auto r8_rec = ExecuteBoltPromptRun(*model, sub_prompt, 32, /*disable_r16=*/true);
    const auto r16_rec = ExecuteBoltPromptRun(*model, sub_prompt, 32, /*disable_r16=*/false);

    const size_t expected_r8_prefill = (len + 7) / 8;
    const size_t expected_r16_chunks = len / 16;
    const size_t rem = len % 16;
    const size_t expected_r16_tail_r8 = (rem == 0) ? 0 : (rem <= 8 ? 1 : 2);
    const int32_t expected_state_row =
        (rem == 0) ? 7 : static_cast<int32_t>((rem - 1) % 8);
    const auto expected_aux_win = PlanNemotronBoltPrefillAuxWindow(len, 1032, 8);

    EXPECT_EQ(r8_rec.warmup_r8_calls, 4u) << "len=" << len;
    EXPECT_EQ(r8_rec.warmup_r16_calls, 0u) << "len=" << len;
    EXPECT_EQ(r8_rec.prefill_r8_calls, expected_r8_prefill) << "len=" << len;
    EXPECT_EQ(r8_rec.prefill_r16_calls, 0u) << "len=" << len;
    EXPECT_EQ(r8_rec.post_prefill_stats.target_forward_passes, expected_r8_prefill) << "len=" << len;

    EXPECT_EQ(r16_rec.warmup_r8_calls, 4u) << "len=" << len;
    EXPECT_EQ(r16_rec.warmup_r16_calls, 4u) << "len=" << len;
    EXPECT_EQ(r16_rec.prefill_r16_calls, expected_r16_chunks) << "len=" << len;
    EXPECT_EQ(r16_rec.prefill_r8_calls, expected_r16_tail_r8) << "len=" << len;
    EXPECT_EQ(r16_rec.post_prefill_stats.target_forward_passes,
              expected_r16_chunks + expected_r16_tail_r8)
        << "len=" << len;

    EXPECT_EQ(r8_rec.post_prefill_length, len) << "len=" << len;
    EXPECT_EQ(r16_rec.post_prefill_length, len) << "len=" << len;
    EXPECT_EQ(r8_rec.post_prefill_state_row, expected_state_row) << "len=" << len;
    EXPECT_EQ(r16_rec.post_prefill_state_row, expected_state_row) << "len=" << len;
    EXPECT_EQ(r16_rec.post_prefill_anchor, r8_rec.post_prefill_anchor) << "len=" << len;

    ASSERT_EQ(r8_rec.post_prefill_ssm0_slice.size(), 32u);
    ASSERT_EQ(r16_rec.post_prefill_ssm0_slice.size(), 32u);
    ASSERT_EQ(r8_rec.post_prefill_logits_slice.size(), 32u);
    ASSERT_EQ(r16_rec.post_prefill_logits_slice.size(), 32u);
    for (size_t i = 0; i < 32; ++i) {
      EXPECT_TRUE(std::isfinite(r8_rec.post_prefill_ssm0_slice[i]));
      EXPECT_TRUE(std::isfinite(r16_rec.post_prefill_ssm0_slice[i]));
      EXPECT_NEAR(r16_rec.post_prefill_ssm0_slice[i], r8_rec.post_prefill_ssm0_slice[i], 5e-2f)
          << "len=" << len << " ssm0 idx=" << i;
      EXPECT_TRUE(std::isfinite(r8_rec.post_prefill_logits_slice[i]));
      EXPECT_TRUE(std::isfinite(r16_rec.post_prefill_logits_slice[i]));
      EXPECT_NEAR(r16_rec.post_prefill_logits_slice[i], r8_rec.post_prefill_logits_slice[i], 5e-1f)
          << "len=" << len << " logits idx=" << i;
    }

    EXPECT_EQ(r8_rec.last_prefill_draft_start, expected_aux_win.draft_start) << "len=" << len;
    EXPECT_EQ(r8_rec.last_prefill_draft_rows, expected_aux_win.draft_rows) << "len=" << len;
    EXPECT_EQ(r16_rec.last_prefill_draft_start, expected_aux_win.draft_start) << "len=" << len;
    EXPECT_EQ(r16_rec.last_prefill_draft_rows, expected_aux_win.draft_rows) << "len=" << len;

    EXPECT_EQ(r8_rec.verify_r8_calls, r8_rec.final_stats.target_verify_forward_passes) << "len=" << len;
    EXPECT_EQ(r16_rec.verify_r8_calls, r16_rec.final_stats.target_verify_forward_passes) << "len=" << len;
    EXPECT_EQ(r16_rec.generated_tokens, r8_rec.generated_tokens) << "len=" << len;
    EXPECT_EQ(r16_rec.acceptance_trace, r8_rec.acceptance_trace) << "len=" << len;
    raw_records_json["boundary_runs"].push_back(record_to_json(len, r8_rec, r16_rec));
  }

  // 4. Optional selectable 32,768-token parity and forward-count gate (4,096 R8 vs 2,048 R16 prompt prefill calls).
  if (const char* run_32k = std::getenv("ORTGENAI_BOLT_RUN_32K_TEST");
      run_32k != nullptr && std::string_view{run_32k} == "1") {
    const auto prompt_32k = LoadBoltPromptTokens(32768);
    const auto r8_32k = ExecuteBoltPromptRun(*model, prompt_32k, 32, /*disable_r16=*/true);
    const auto r16_32k = ExecuteBoltPromptRun(*model, prompt_32k, 32, /*disable_r16=*/false);

    EXPECT_EQ(r8_32k.warmup_r8_calls, 4u);
    EXPECT_EQ(r8_32k.warmup_r16_calls, 0u);
    EXPECT_EQ(r8_32k.prefill_r8_calls, 4096u);
    EXPECT_EQ(r8_32k.prefill_r16_calls, 0u);
    EXPECT_EQ(r8_32k.post_prefill_stats.target_forward_passes, 4096u);
    EXPECT_EQ(r8_32k.post_prefill_state_row, 7);

    EXPECT_EQ(r16_32k.warmup_r8_calls, 4u);
    EXPECT_EQ(r16_32k.warmup_r16_calls, 4u);
    EXPECT_EQ(r16_32k.prefill_r8_calls, 0u);
    EXPECT_EQ(r16_32k.prefill_r16_calls, 2048u);
    EXPECT_EQ(r16_32k.post_prefill_stats.target_forward_passes, 2048u);
    EXPECT_EQ(r16_32k.post_prefill_state_row, 7);

    EXPECT_EQ(r16_32k.post_prefill_anchor, r8_32k.post_prefill_anchor);
    EXPECT_EQ(r16_32k.generated_tokens, r8_32k.generated_tokens);
    EXPECT_EQ(r16_32k.acceptance_trace, r8_32k.acceptance_trace);
    raw_records_json["run_32768"] = record_to_json(32768, r8_32k, r16_32k);
  }

  // 5. Optional genuine fixed-R=8 package compatibility gate (ORTGENAI_BOLT_TEST_MODEL_FIXED_R8).
  //    Release the symbolic-R model first so two 45 GB models are never resident simultaneously on one A100.
  model.reset();
  if (const char* fixed_r8_path = std::getenv("ORTGENAI_BOLT_TEST_MODEL_FIXED_R8")) {
    ScopedBoltEnvVar enable_env("BOLT_DISABLE_R16_PREFILL", std::nullopt);
    auto fixed_model = CreateModel(GetOrtEnv(), fixed_r8_path);
    auto fixed_params = std::make_shared<GeneratorParams>(*fixed_model);
    fixed_params->search.max_length = static_cast<int>(prompt.size() + 32);
    Generator fixed_gen{*fixed_model, *fixed_params};
    auto& fixed_state = dynamic_cast<NemotronBoltState&>(*fixed_gen.state_);
    EXPECT_FALSE(NemotronBoltTestAccess::ModelSupportsR16(fixed_state));
    EXPECT_FALSE(NemotronBoltTestAccess::R16Enabled(fixed_state));
    EXPECT_FALSE(NemotronBoltTestAccess::HasR16ProfileTensors(fixed_state));
    EXPECT_EQ(NemotronBoltTestAccess::VerifyInputsSpanSize(fixed_state), 2 * kBoltVerifyRows + 8);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR8Calls(fixed_state), 4u);
    EXPECT_EQ(NemotronBoltTestAccess::WarmupR16Calls(fixed_state), 0u);

    fixed_gen.AppendTokens(prompt);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR16Calls(fixed_state), 0u);
    EXPECT_EQ(NemotronBoltTestAccess::PrefillR8Calls(fixed_state), 5u);
    while (!fixed_gen.IsDone()) fixed_gen.GenerateNextToken();
    const auto fixed_seq = fixed_gen.GetSequence(0).CopyDeviceToCpu();
    const std::vector<int32_t> fixed_vec(fixed_seq.begin(), fixed_seq.end());
    EXPECT_EQ(fixed_vec, baseline);
    raw_records_json["fixed_r8_package_check"] = {
        {"model_supports_r16", NemotronBoltTestAccess::ModelSupportsR16(fixed_state)},
        {"r16_enabled", NemotronBoltTestAccess::R16Enabled(fixed_state)},
        {"warmup_r8_calls", NemotronBoltTestAccess::WarmupR8Calls(fixed_state)},
        {"warmup_r16_calls", NemotronBoltTestAccess::WarmupR16Calls(fixed_state)},
        {"prefill_r8_calls", NemotronBoltTestAccess::PrefillR8Calls(fixed_state)},
        {"prefill_r16_calls", NemotronBoltTestAccess::PrefillR16Calls(fixed_state)},
        {"matches_baseline_35", fixed_vec == baseline},
    };
  } else if (RequireGpuQualificationEnv()) {
    FAIL() << "ORTGENAI_BOLT_TEST_MODEL_FIXED_R8 is required when ORTGENAI_BOLT_REQUIRE_GPU_QUALIFICATION=1";
  }

  if (const char* out_json = std::getenv("ORTGENAI_BOLT_RAW_RECORDS_OUT")) {
    std::ofstream out{out_json};
    out << raw_records_json.dump(2) << "\n";
  }
}

}  // namespace
}  // namespace Generators::test
