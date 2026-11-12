// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <cstring>
#include <cmath>

#include <nlohmann/json.hpp>

#include "dflash2_drafter.h"
#include "models/nemotron_bolt.h"
#include "search.h"

namespace Generators::test {

struct NemotronBoltTestAccess {
  static std::unique_ptr<Tensor> SaveVerifyPrefix(Tensor& tensor, size_t rows) {
    return NemotronBoltState::SaveVerifyPrefix(tensor, rows);
  }

  static void RestoreVerifyPrefix(Tensor& tensor, Tensor& prefix) {
    NemotronBoltState::RestoreVerifyPrefix(tensor, prefix);
  }

  static std::vector<uint8_t> Bytes(DeviceSpan<uint8_t> span) {
    auto selected = GetDeviceInterface(DeviceType::CUDA)->WrapMemory(span.Span());
    const auto cpu = selected.CopyDeviceToCpu();
    return {cpu.begin(), cpu.end()};
  }

  static size_t Length(const NemotronBoltState& state) { return state.length_; }
  static size_t SelectedRow(const NemotronBoltState& state) { return static_cast<size_t>(state.state_row_); }

  static nlohmann::json CompareLiveBoundary(NemotronBoltState& ordinary,
                                            NemotronBoltState& speculative, size_t row) {
    nlohmann::json differences = nlohmann::json::array();
    size_t bytes_compared = 0;
    for (const auto& [name, tensor] : ordinary.tensors_) {
      if (name.rfind("past_", 0) != 0 && name != "logits" &&
          name != "aux_hidden_states" && name != "pred_ids") continue;
      auto expected = tensor->GetByteSpan();
      auto actual = speculative.tensors_.at(name)->GetByteSpan();
      const auto compare = [&](DeviceSpan<uint8_t> lhs, DeviceSpan<uint8_t> rhs) {
        const auto a = Bytes(lhs), b = Bytes(rhs);
        bytes_compared += a.size();
        if (std::memcmp(a.data(), b.data(), a.size()) != 0) {
          const auto first = std::mismatch(a.begin(), a.end(), b.begin());
          differences.push_back({{"name", name}, {"first_byte", first.first - a.begin()}});
        }
        if (name == "pred_ids") return;
        const bool fp32 = name.rfind("past_ssm.", 0) == 0;
        const size_t item_bytes = fp32 ? sizeof(uint32_t) : sizeof(uint16_t);
        for (size_t i = 0; i < b.size(); i += item_bytes) {
          uint32_t bits = 0;
          std::memcpy(&bits, b.data() + i, item_bytes);
          if ((fp32 && (bits & 0x7f800000) == 0x7f800000) ||
              (!fp32 && (bits & 0x7f80) == 0x7f80)) {
            differences.push_back({{"name", name}, {"nonfinite_byte", i}});
            break;
          }
        }
      };
      if (name.rfind("past_key.", 0) == 0 || name.rfind("past_value.", 0) == 0) {
        const auto shape = tensor->GetShape();
        const size_t head_bytes = expected.size() / static_cast<size_t>(shape[1]);
        const size_t valid_bytes = ordinary.length_ * static_cast<size_t>(shape[3]) * sizeof(uint16_t);
        for (size_t head = 0; head < static_cast<size_t>(shape[1]); ++head)
          compare(expected.subspan(head * head_bytes, valid_bytes), actual.subspan(head * head_bytes, valid_bytes));
      } else {
        const bool recurrent = name.rfind("past_", 0) == 0;
        const auto shape = tensor->GetShape();
        const auto actual_shape = speculative.tensors_.at(name)->GetShape();
        const size_t row_bytes = expected.size() / static_cast<size_t>(shape[recurrent ? 0 : 1]);
        const size_t actual_row_bytes = actual.size() / static_cast<size_t>(actual_shape[recurrent ? 0 : 1]);
        if (row_bytes != actual_row_bytes) {
          differences.push_back({{"name", name}, {"row_bytes_mismatch", true}});
          continue;
        }
        compare(expected.subspan(static_cast<size_t>(ordinary.state_row_) * row_bytes, row_bytes),
                actual.subspan(row * row_bytes, row_bytes));
      }
    }
    return {{"logical_input_prefix", ordinary.length_}, {"snapshot_row", row}, {"complete_bytes_compared", bytes_compared}, {"differences", differences}};
  }

  // Replay identical candidate IDs from the same committed state, without feeding
  // reference activations/routes into the live generator.
  static nlohmann::json CompareCandidatePrefix(NemotronBoltState& state) {
    const size_t length = state.length_;
    const int32_t selected = state.state_row_;
    const auto stats = state.stats_;
    std::vector<int32_t> candidates{state.anchor_};
    candidates.insert(candidates.end(), state.drafts_.begin(), state.drafts_.end());
    struct SavedCache {
      Tensor* tensor;
      std::string name;
      bool recurrent;
      size_t row_bytes;
      std::vector<uint8_t> initial;
      std::vector<uint8_t> block;
    };
    std::vector<SavedCache> caches;
    for (auto* tensor : state.caches_) {
      const auto shape = tensor->GetShape();
      const bool recurrent = shape[0] == 8;
      const size_t row_bytes = recurrent ? tensor->GetByteSpan().size() / 8 : 0;
      auto span = tensor->GetByteSpan();
      const auto initial = recurrent
                               ? Bytes(span.subspan(static_cast<size_t>(selected) * row_bytes, row_bytes))
                               : Bytes(span);
      std::string name;
      for (const auto& entry : state.tensors_)
        if (entry.second.get() == tensor) name = entry.first;
      caches.push_back({tensor, name, recurrent, row_bytes, initial, {}});
    }
    const auto restore = [&] {
      for (auto& cache : caches) {
        auto destination = cache.tensor->GetByteSpan();
        if (cache.recurrent)
          destination = destination.subspan(static_cast<size_t>(selected) * cache.row_bytes, cache.row_bytes);
        const auto cpu = destination.CpuSpan();
        std::copy(cache.initial.begin(), cache.initial.end(), cpu.begin());
        destination.CopyCpuRangeToDevice();
      }
      state.length_ = length;
      state.state_row_ = selected;
      state.stats_ = stats;
    };
    const auto predictions = state.Verify(candidates);
    const auto block_predictions = Bytes(state.tensors_.at("pred_ids")->GetByteSpan());
    const auto block_logits = Bytes(state.tensors_.at("logits")->GetByteSpan());
    for (auto& cache : caches)
      cache.block = Bytes(cache.tensor->GetByteSpan());
    restore();
    for (const auto& cache : caches) {
      auto span = cache.tensor->GetByteSpan();
      if (cache.recurrent)
        span = span.subspan(static_cast<size_t>(selected) * cache.row_bytes, cache.row_bytes);
      RequireRestored(Bytes(span) == cache.initial);
    }
    state.Verify(candidates, false);
    const bool discarded_predictions_match =
        Bytes(state.tensors_.at("pred_ids")->GetByteSpan()) == block_predictions;
    restore();
    nlohmann::json result{
        {"length", length}, {"state_row", selected}, {"candidates", candidates}, {"predictions", predictions}, {"discarded_predictions_match", discarded_predictions_match}, {"accepted", NemotronBoltAcceptedPrefix(state.drafts_, predictions)}, {"rows", nlohmann::json::array()}};
    const size_t logit_bytes = block_logits.size() / 8;
    for (size_t row = 0; row < candidates.size(); ++row) {
      const auto sequential = state.Verify(std::vector<int32_t>(8, candidates[row]));
      const auto logits = Bytes(state.tensors_.at("logits")->GetByteSpan().subspan(0, logit_bytes));
      nlohmann::json differences = nlohmann::json::array();
      for (const auto& cache : caches) {
        if (!cache.recurrent) {
          const auto shape = cache.tensor->GetShape();
          const size_t head_bytes = cache.tensor->GetByteSpan().size() / static_cast<size_t>(shape[1]);
          const size_t valid_bytes = (length + row + 1) * static_cast<size_t>(shape[3]) * sizeof(uint16_t);
          const auto bytes = Bytes(cache.tensor->GetByteSpan());
          for (size_t head = 0; head < static_cast<size_t>(shape[1]); ++head) {
            const auto begin = bytes.begin() + head * head_bytes;
            const auto end = begin + valid_bytes;
            const auto first = std::mismatch(begin, end, cache.block.begin() + head * head_bytes);
            if (first.first != end)
              differences.push_back({{"name", cache.name}, {"head", head}, {"first_byte", first.first - begin}});
          }
          continue;
        }
        const auto bytes = Bytes(cache.tensor->GetByteSpan().subspan(0, cache.row_bytes));
        const auto first = std::mismatch(bytes.begin(), bytes.end(), cache.block.begin() + row * cache.row_bytes);
        if (first.first != bytes.end())
          differences.push_back({{"name", cache.name},
                                 {"cache_index", &cache - caches.data()},
                                 {"first_byte", first.first - bytes.begin()}});
      }
      const auto first = std::mismatch(logits.begin(), logits.end(), block_logits.begin() + row * logit_bytes);
      const auto scores = [](const uint8_t* data, size_t bytes) {
        std::array<std::pair<float, size_t>, 2> best{{{-INFINITY, 0}, {-INFINITY, 0}}};
        for (size_t i = 0; i < bytes / sizeof(uint16_t); ++i) {
          uint16_t stored;
          std::memcpy(&stored, data + i * sizeof(stored), sizeof(stored));
          const uint32_t bits = static_cast<uint32_t>(stored) << 16;
          float value;
          std::memcpy(&value, &bits, sizeof(value));
          if (value > best[0].first) {
            best[1] = best[0];
            best[0] = {value, i};
          } else if (value > best[1].first) {
            best[1] = {value, i};
          }
        }
        return nlohmann::json{{"top_id", best[0].second}, {"top", best[0].first}, {"second_id", best[1].second}, {"second", best[1].first}, {"margin", best[0].first - best[1].first}};
      };
      const char* output = std::getenv("ORTGENAI_BOLT_OUTPUT");
      if (output && first.first != logits.end()) {
        const auto prefix = std::string{output} + "-L" + std::to_string(length) + "-R" + std::to_string(row);
        std::ofstream serial{prefix + "-sequential.bf16", std::ios::binary};
        std::ofstream block{prefix + "-block.bf16", std::ios::binary};
        serial.write(reinterpret_cast<const char*>(logits.data()), logits.size());
        block.write(reinterpret_cast<const char*>(block_logits.data() + row * logit_bytes), logit_bytes);
      }
      result["rows"].push_back({{"row", row}, {"sequential_prediction", sequential[0]}, {"logits_bitwise", first.first == logits.end()}, {"cache_differences", differences}, {"sequential_raw_processed_scores", scores(logits.data(), logit_bytes)}, {"block_raw_processed_scores", scores(block_logits.data() + row * logit_bytes, logit_bytes)}});
      ++state.length_;
      state.state_row_ = 0;
    }
    restore();
    return result;
  }

  static bool ReadyForRound(const NemotronBoltState& state) {
    const char* round = std::getenv("ORTGENAI_BOLT_PREFIX_ROUND");
    return state.pending_.empty() &&
           (!round || state.stats_.rounds + 1 == static_cast<uint64_t>(std::stoul(round)));
  }

  static void RequireRestored(bool restored) {
    if (!restored) throw std::runtime_error("identical-prefix observer failed to restore prior state");
  }

  static std::vector<uint8_t> SelectedLogits(NemotronBoltState& state) {
    const size_t row_bytes = static_cast<size_t>(state.bolt_.config_->model.vocab_size) * sizeof(uint16_t);
    // Keep the wrapped buffer's host mirror alive until the vector has been copied.
    auto selected = state.tensors_.at("logits")->GetByteSpan().subspan(static_cast<size_t>(state.state_row_) * row_bytes, row_bytes);
    const auto bytes = selected.CopyDeviceToCpu();
    return {bytes.begin(), bytes.end()};
  }

  static void ForceRejectedDrafts(NemotronBoltState& state) {
    state.drafts_.assign(7, 0);
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

std::unique_ptr<Config> BoltTestConfig(const char* path);

void CheckVerificationPrefixMerge(DeviceInterface* device) {
  const std::vector<std::pair<ONNXTensorElementDataType, std::vector<int64_t>>> cases{
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {8, 128, 128, 64}},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, {8, 5, 3}},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, {1, 8, 7}},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {1, 8}}};
  for (const auto& [type, shape] : cases) {
    Tensor tensor{device, type};
    tensor.CreateTensor(shape, true);
    const auto address = tensor.GetMutableRawData();
    const size_t bytes = tensor.GetByteSpan().size();
    const size_t row_bytes = bytes / 8;
    std::vector<uint8_t> first(bytes), second(bytes), expected(bytes);
    for (size_t row = 0; row < 8; ++row) {
      std::fill_n(first.begin() + row * row_bytes, row_bytes, static_cast<uint8_t>(row + 1));
      std::fill_n(second.begin() + row * row_bytes, row_bytes, static_cast<uint8_t>(row + 17));
    }
    Tensor tail{device, type};
    tail.CreateTensor(shape, true);
    tail.GetByteSpan().CopyFromCpu(second);
    for (size_t rows = 1; rows < 8; ++rows) {
      SCOPED_TRACE(::testing::Message() << "type=" << type << " prefix_rows=" << rows);
      tensor.GetByteSpan().CopyFromCpu(first);
      auto prefix = NemotronBoltTestAccess::SaveVerifyPrefix(tensor, rows);
      EXPECT_EQ(prefix->GetByteSpan().size(), rows * row_bytes);
      EXPECT_LT(prefix->GetByteSpan().size(), bytes);
      tensor.GetByteSpan().CopyFrom(tail.GetByteSpan());
      NemotronBoltTestAccess::RestoreVerifyPrefix(tensor, *prefix);
      std::copy_n(first.begin(), rows * row_bytes, expected.begin());
      std::copy_n(second.begin(), bytes - rows * row_bytes, expected.begin() + rows * row_bytes);
      auto span = tensor.GetByteSpan();
      const auto actual = span.CopyDeviceToCpu();
      EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin()));
      EXPECT_EQ(tensor.GetMutableRawData(), address);
      EXPECT_EQ(tensor.GetShape(), shape);
    }
  }
}

TEST(NemotronBolt, VerificationPrefixMergePreservesAllRowsWithCompactScratch) {
  CheckVerificationPrefixMerge(GetDeviceInterface(DeviceType::CPU));
}

TEST(NemotronBoltGPU, VerificationPrefixMergePreservesAllRowsWithCompactScratch) {
  if (!std::getenv("ORTGENAI_TEST_CUDA_PREFIX_MERGE"))
    GTEST_SKIP() << "Set ORTGENAI_TEST_CUDA_PREFIX_MERGE with an admitted CUDA device";
  auto config = std::make_unique<Config>(
      fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, "{}");
  SetProviderOption(*config, "cuda", "enable_cuda_graph", "0");
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  ASSERT_EQ(model->p_device_->GetType(), DeviceType::CUDA);
  CheckVerificationPrefixMerge(model->p_device_);
}

TEST(NemotronBolt, VerificationKeepsQueryEndpointsInOneKeyTile) {
  for (size_t position : {size_t{0}, size_t{500}, size_t{504}, size_t{505},
                          size_t{632}, size_t{633}, size_t{135158}}) {
    const size_t rows = NemotronBoltVerifyPrefixRows(position);
    ASSERT_GE(rows, 1);
    ASSERT_LE(rows, 8);
    for (size_t row = 0; row < rows; ++row)
      EXPECT_EQ((position + 7) / 128, (position + row + 7) / 128);
    if (rows < 8)
      EXPECT_NE((position + 7) / 128, (position + rows + 7) / 128);
  }
}

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

TEST(NemotronBoltGPU, IdenticalCandidatePrefixReplay) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  const char* cases_path = std::getenv("ORTGENAI_BOLT_TEST_CASES");
  const char* output = std::getenv("ORTGENAI_BOLT_OUTPUT");
  if (!path || !cases_path || !output)
    GTEST_SKIP() << "Set model, exact token fixtures and owned output path";
  std::ifstream input{cases_path};
  nlohmann::json fixtures;
  input >> fixtures;
  auto model = CreateModel(GetOrtEnv(), BoltTestConfig(path));
  std::ofstream report{std::string{output} + "-prefix.jsonl"};
  ASSERT_TRUE(report);
  for (const auto& fixture : fixtures.at("cases")) {
    const auto prompt = fixture.at("prompt_ids").get<std::vector<int32_t>>();
    auto params = std::make_shared<GeneratorParams>(*model);
    params->speculative.max_draft_tokens = 7;
    params->use_graph_capture = true;
    params->search.max_length = static_cast<int>(prompt.size() + 60);
    Generator generator{*model, *params};
    generator.AppendTokens(prompt);
    while (!generator.IsDone()) {
      auto& state = dynamic_cast<NemotronBoltState&>(*generator.state_);
      if (NemotronBoltTestAccess::ReadyForRound(state)) {
        const auto comparison = NemotronBoltTestAccess::CompareCandidatePrefix(state);
        report << comparison.dump() << std::endl;
        EXPECT_TRUE(comparison.at("discarded_predictions_match").get<bool>()) << comparison.dump();
        for (const auto& row : comparison.at("rows")) {
          EXPECT_TRUE(row.at("logits_bitwise").get<bool>()) << comparison.dump();
          EXPECT_TRUE(row.at("cache_differences").empty()) << comparison.dump();
        }
      }
      generator.GenerateNextToken();
    }
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    if (fixture.contains("expected_speculative")) {
      const auto expected = fixture.at("expected_speculative").get<std::vector<int32_t>>();
      EXPECT_EQ(std::vector<int32_t>(sequence.begin() + prompt.size(), sequence.end()), expected)
          << "Deep observer changed live generation";
    }
    report << nlohmann::json{{"live_generated_ids", std::vector<int32_t>(
                                                        sequence.begin() + prompt.size(), sequence.end())}}
                  .dump()
           << std::endl;
  }
}

TEST(NemotronBoltGPU, LiveOrdinarySpeculativeBoundaryStates) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  const char* cases_path = std::getenv("ORTGENAI_BOLT_TEST_CASES");
  const char* output = std::getenv("ORTGENAI_BOLT_OUTPUT");
  if (!path || !cases_path || !output)
    GTEST_SKIP() << "Set model, exact token fixtures and owned output path";
  std::ifstream input{cases_path};
  nlohmann::json fixtures;
  input >> fixtures;
  auto model = CreateModel(GetOrtEnv(), BoltTestConfig(path));
  std::ofstream report{std::string{output} + "-live-states.jsonl"};
  ASSERT_TRUE(report);
  for (const auto& fixture : fixtures.at("cases")) {
    const auto prompt = fixture.at("prompt_ids").get<std::vector<int32_t>>();
    const auto budget = fixture.at("max_new_tokens").get<int>();
    const auto expected = fixture.at("expected_ordinary").get<std::vector<int32_t>>();
    auto ordinary_params = std::make_shared<GeneratorParams>(*model);
    ordinary_params->speculative.max_draft_tokens = 0;
    ordinary_params->search.max_length = static_cast<int>(prompt.size()) + budget;
    auto speculative_params = std::make_shared<GeneratorParams>(*model);
    speculative_params->speculative.max_draft_tokens = 7;
    speculative_params->search.max_length = ordinary_params->search.max_length;
    Generator ordinary{*model, *ordinary_params};
    Generator speculative{*model, *speculative_params};
    auto& ordinary_state = dynamic_cast<NemotronBoltState&>(*ordinary.state_);
    auto& speculative_state = dynamic_cast<NemotronBoltState&>(*speculative.state_);
    for (int repeat = 0; repeat < 2; ++repeat) {
      ordinary.AppendTokens(prompt);
      speculative.AppendTokens(prompt);
      size_t boundaries = 0;
      size_t bytes_compared = 0;
      const auto prefill = NemotronBoltTestAccess::CompareLiveBoundary(
          ordinary_state, speculative_state, NemotronBoltTestAccess::SelectedRow(speculative_state));
      ASSERT_TRUE(prefill.at("differences").empty()) << prefill.dump();
      bytes_compared += prefill.at("complete_bytes_compared").get<size_t>();
      while (!speculative.IsDone()) {
        const auto before = speculative.GetSpeculativeStats();
        const size_t position = NemotronBoltTestAccess::Length(speculative_state);
        speculative.GenerateNextToken();
        const auto after = speculative.GetSpeculativeStats();
        if (after.rounds == before.rounds) continue;
        const size_t accepted = static_cast<size_t>(after.draft_tokens_accepted - before.draft_tokens_accepted);
        for (size_t row = 0; row <= accepted; ++row) {
          const size_t logical_prefix = position + row + 1;
          while (!ordinary.IsDone() && NemotronBoltTestAccess::Length(ordinary_state) < logical_prefix)
            ordinary.GenerateNextToken();
          if (NemotronBoltTestAccess::Length(ordinary_state) != logical_prefix) break;
          auto comparison = NemotronBoltTestAccess::CompareLiveBoundary(ordinary_state, speculative_state, row);
          ++boundaries;
          bytes_compared += comparison.at("complete_bytes_compared").get<size_t>();
          if (!comparison.at("differences").empty()) {
            comparison["case_id"] = fixture.at("case_id");
            comparison["repeat"] = repeat;
            report << comparison.dump() << std::endl;
          }
          ASSERT_TRUE(comparison.at("differences").empty()) << comparison.dump();
        }
      }
      while (!ordinary.IsDone()) ordinary.GenerateNextToken();
      const auto ordinary_sequence = ordinary.GetSequence(0).CopyDeviceToCpu();
      const auto speculative_sequence = speculative.GetSequence(0).CopyDeviceToCpu();
      const std::vector<int32_t> ordinary_tokens(ordinary_sequence.begin() + prompt.size(), ordinary_sequence.end());
      const std::vector<int32_t> speculative_tokens(speculative_sequence.begin() + prompt.size(), speculative_sequence.end());
      EXPECT_EQ(ordinary_tokens, expected);
      EXPECT_EQ(speculative_tokens, expected);
      const auto stats = speculative.GetSpeculativeStats();
      EXPECT_EQ(stats.tokens_emitted, expected.size());
      EXPECT_EQ(stats.tokens_buffered, 0);
      EXPECT_EQ(stats.dflash2_failures, 0);
      EXPECT_EQ(stats.dflash2_disables, 0);
      EXPECT_GE(stats.target_verify_forward_passes, stats.rounds);
      report << nlohmann::json{{"case_id", fixture.at("case_id")}, {"repeat", repeat}, {"complete_boundaries", boundaries}, {"complete_bytes_compared", bytes_compared}, {"ordinary_tokens", ordinary_tokens}, {"speculative_tokens", speculative_tokens}, {"target_verify_passes", stats.target_verify_forward_passes}, {"rounds", stats.rounds}}.dump()
             << std::endl;
      ordinary.RewindToLength(0);
      speculative.RewindToLength(0);
      EXPECT_EQ(ordinary.TokenCount(), 0);
      EXPECT_EQ(speculative.TokenCount(), 0);
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

TEST(NemotronBolt, HonorsGreedyEquivalentSamplingOptions) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.vocab_size = 128;
  config.model.context_length = 16;
  config.search.max_length = 16;
  config.speculative.max_draft_tokens = 7;
  struct Case {
    float temperature;
    int top_k;
    bool greedy;
  };
  for (const auto& test : std::array<Case, 8>{{
           {0.0f, 0, true},
           {0.0f, 1, true},
           {0.0f, 50, true},
           {0.7f, 1, true},
           {1.0f, 1, true},
           {0.7f, 50, false},
           {1.0f, 0, false},
           {1.0f, 50, false},
       }}) {
    SCOPED_TRACE(::testing::Message() << "temperature=" << test.temperature << ", top_k=" << test.top_k);
    auto params = std::make_shared<GeneratorParams>(config);
    params->use_graph_capture = true;
    params->search.do_sample = true;
    params->search.temperature = test.temperature;
    params->search.top_k = test.top_k;
    if (test.greedy) {
      EXPECT_NO_THROW(ValidateNemotronBoltSearch(*params));
    } else {
      EXPECT_THROW(ValidateNemotronBoltSearch(*params), std::runtime_error);
    }
  }
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

TEST(NemotronBolt, TargetOnlySearchDoesNotRequireCapture) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.speculative.max_draft_tokens = 0;
  auto params = std::make_shared<GeneratorParams>(config);
  for (bool capture : {false, true}) {
    params->use_graph_capture = capture;
    EXPECT_NO_THROW(ValidateNemotronBoltSearch(*params));
  }
  params->speculative.max_draft_tokens = 7;
  params->use_graph_capture = false;
  EXPECT_THROW(ValidateNemotronBoltSearch(*params), std::runtime_error);
}

TEST(NemotronBolt, RejectsUnsupportedPackageDraftCountBeforeLoadingSessions) {
  for (int draft_tokens : {Config{}.speculative.max_draft_tokens, -1, 1, 6, 8}) {
    auto config = std::make_unique<Config>();
    config->model.type = "nemotron_bolt";
    config->model.decoder.filename = "must-not-load.onnx";
    config->speculative.max_draft_tokens = draft_tokens;
    try {
      NemotronBoltModel model{std::move(config), GetOrtEnv()};
      FAIL() << "Unsupported package draft count " << draft_tokens << " was accepted";
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), "nemotron_bolt: speculative.max_draft_tokens must be 0 or 7");
    }
  }
}

TEST(NemotronBolt, ParsesTargetOnlyRegardlessOfJsonMemberOrder) {
  for (const auto document : {
           R"({"speculative":{"max_draft_tokens":0},"model":{"type":"nemotron_bolt","context_length":16,"vocab_size":128}})",
           R"({"model":{"type":"nemotron_bolt","context_length":16,"vocab_size":128},"speculative":{"max_draft_tokens":0}})"}) {
    Config config{fs::path{}, {}, document};
    EXPECT_EQ(config.speculative.max_draft_tokens, 0);
    EXPECT_NO_THROW(OverlayConfig(config, R"({"speculative":{"max_draft_tokens":0}})"));
  }
  EXPECT_THROW((Config{fs::path{}, {}, R"({"speculative":{"max_draft_tokens":0},"model":{"type":"llama","context_length":16}})"}),
               std::runtime_error);
}

TEST(NemotronBolt, PublicSpeculativeSetterAllowsZeroOnlyForBolt) {
  for (const char* type : {"nemotron_bolt", "llama"}) {
    Config config;
    config.model.type = type;
    GeneratorParams params{config};
    ASSERT_NO_THROW(params.SetSpeculativeNumber("max_draft_tokens", 7));
    if (config.model.type == "nemotron_bolt") {
      EXPECT_NO_THROW(params.SetSpeculativeNumber("max_draft_tokens", 0));
      EXPECT_EQ(params.GetSpeculativeNumber("max_draft_tokens"), 0);
    } else {
      EXPECT_THROW(params.SetSpeculativeNumber("max_draft_tokens", 0), std::exception);
      EXPECT_EQ(params.GetSpeculativeNumber("max_draft_tokens"), 7);
    }
    for (double invalid : {-1.0, 0.5, 17.0}) {
      EXPECT_THROW(params.SetSpeculativeNumber("max_draft_tokens", invalid), std::exception);
    }
  }
}

TEST(NemotronBolt, ValidatesTargetOnlyAfterMergingConfiguration) {
  const std::string base = R"({"speculative":{"max_draft_tokens":0},"model":{"type":"llama","context_length":16,"vocab_size":128}})";
  const std::string bolt = R"({"model":{"type":"nemotron_bolt"}})";
  Config config{fs::path{}, bolt, base};
  EXPECT_EQ(config.speculative.max_draft_tokens, 0);
  EXPECT_EQ(config.model.type, "nemotron_bolt");
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"type":"llama"}})"), std::runtime_error);
  EXPECT_EQ(config.model.type, "nemotron_bolt");
  EXPECT_NO_THROW(OverlayConfig(config, R"({"model":{"type":"llama"},"speculative":{"max_draft_tokens":7}})"));
  EXPECT_EQ(config.model.type, "llama");
  EXPECT_EQ(config.speculative.max_draft_tokens, 7);
  EXPECT_THROW((Config{fs::path{}, R"({"model":{"type":"llama"}})",
                       R"({"model":{"type":"nemotron_bolt","context_length":16},"speculative":{"max_draft_tokens":0}})"}),
               std::runtime_error);
  try {
    OverlayConfig(config, R"({"speculative":{"max_draft_tokens":-1}})");
    FAIL() << "Negative draft width was accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("must be between 0 and 16"), std::string::npos);
  }
}

TEST(NemotronBolt, PreservesDrafterOptimizerExclusions) {
  const std::string fusion = "GatherSliceToSplitFusion";
  for (const auto& [existing, expected] :
       std::vector<std::pair<std::optional<std::string>, std::string>>{
           {std::nullopt, fusion},
           {"", fusion},
           {"OtherFusion", "OtherFusion;" + fusion},
           {"OtherFusion;", "OtherFusion;" + fusion},
           {fusion, fusion},
           {fusion + ";OtherFusion", fusion + ";OtherFusion"},
           {"OtherFusion;" + fusion + ";", "OtherFusion;" + fusion + ";"},
           {"GatherSliceToSplitFusionSuffix", "GatherSliceToSplitFusionSuffix;" + fusion}}) {
    Config config;
    config.model.vocab_size = 128;
    auto& drafter = config.model.dflash2;
    drafter.filename = "drafter.onnx";
    drafter.num_hidden_layers = drafter.num_key_value_heads = drafter.head_size = 1;
    drafter.block_size = 8;
    drafter.num_draft_tokens = 7;
    drafter.selector_top_k = 1;
    drafter.sliding_window = 8;
    drafter.session_options.emplace();
    auto& entries = drafter.session_options->config_entries;
    if (existing) {
      entries.emplace_back("optimization.disable_specified_optimizers", "SupersededFusion");
      entries.emplace_back("optimization.disable_specified_optimizers", *existing);
    }
    const auto projected = CreateNemotronBoltDrafterConfig(config);
    const auto& merged = projected->model.decoder.session_options.config_entries;
    if (existing) EXPECT_EQ(merged.front(), entries.front());
    const auto exclusion = std::find_if(merged.rbegin(), merged.rend(), [](const auto& option) {
      return option.first == "optimization.disable_specified_optimizers";
    });
    ASSERT_NE(exclusion, merged.rend());
    EXPECT_EQ(exclusion->second, expected);
  }
}

class NemotronBoltRuntimePolicyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Ort::InitApi();
    saved_version_ = Ort::runtime_api_version;
    saved_api_version_ = Ort::api_version;
  }
  void TearDown() override {
    Ort::runtime_api_version = saved_version_;
    Ort::api_version = saved_api_version_;
  }

 private:
  uint32_t saved_version_{};
  uint32_t saved_api_version_{};
};

#if ORT_API_VERSION >= 27
TEST_F(NemotronBoltRuntimePolicyTest, ReleaseWrapperRejectsOlderNegotiatedTables) {
  Config config{fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"}, "{}"};
  const auto path = config.ResolvePath(config.model.decoder.filename);
  auto session = OrtSession::Create(GetOrtEnv(), path.c_str(), nullptr);
  Ort::api_version = 26;
  try {
    session->ReleaseCapturedGraph(0);
    FAIL() << "Release used an unsupported API table";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "ONNX Runtime API 27 or later is required to release captured graphs");
  }
}
#endif

TEST_F(NemotronBoltRuntimePolicyTest, RequiresSafeCaptureInBothProjectedSessions) {
  Config config;
  config.model.vocab_size = 128;
  auto& drafter = config.model.dflash2;
  drafter.filename = "drafter.onnx";
  drafter.num_hidden_layers = drafter.num_key_value_heads = drafter.head_size = 1;
  drafter.block_size = 8;
  drafter.num_draft_tokens = 7;
  drafter.selector_top_k = 1;
  drafter.sliding_window = 8;
  SetProviderOption(config, "cuda", "enable_cuda_graph", "1");
  for (uint32_t version : {18u, 26u, 27u, 31u}) {
    Ort::api_version = version;
    for (const std::string override : {"", "0", "1"}) {
      drafter.session_options.reset();
      if (!override.empty()) {
        drafter.session_options.emplace();
        drafter.session_options->provider_options.push_back({"cuda", {{"enable_cuda_graph", override}}});
      }
      const auto projected = CreateNemotronBoltDrafterConfig(config);
      const auto& target_options = config.model.decoder.session_options;
      const auto& draft_options = projected->model.decoder.session_options;
#if ORT_API_VERSION >= 27
      if (version >= 27) {
        EXPECT_NO_THROW(ValidateNemotronBoltGraphCapture(target_options, true));
        if (override != "0") {
          EXPECT_NO_THROW(ValidateNemotronBoltGraphCapture(draft_options, true));
        } else {
          EXPECT_THROW(ValidateNemotronBoltGraphCapture(draft_options, true), std::runtime_error);
        }
        continue;
      }
#endif
      EXPECT_THROW(ValidateNemotronBoltGraphCapture(target_options, true), std::runtime_error);
      EXPECT_THROW(ValidateNemotronBoltGraphCapture(draft_options, true), std::runtime_error);
    }
    Config::SessionOptions eager;
    EXPECT_NO_THROW(ValidateNemotronBoltGraphCapture(eager, false));
    EXPECT_THROW(ValidateNemotronBoltGraphCapture(eager, true), std::runtime_error);
  }
}

TEST_F(NemotronBoltRuntimePolicyTest, PreservesLogicalDrafterLengthsAndExplicitSanitizerPolicy) {
  for (uint32_t version : {0u, 31u, 32u, 33u}) {
    SCOPED_TRACE(version);
    Ort::runtime_api_version = version;
    for (const std::string policy : {"", "0", "1"}) {
      SCOPED_TRACE(policy);
      Config config;
      config.model.vocab_size = 128;
      auto& drafter = config.model.dflash2;
      drafter.filename = "drafter.onnx";
      drafter.num_hidden_layers = drafter.num_key_value_heads = drafter.head_size = 1;
      drafter.block_size = 8;
      drafter.num_draft_tokens = 7;
      drafter.selector_top_k = 1;
      drafter.sliding_window = 8;
      drafter.session_options.emplace();
      constexpr auto key = "session.paged_attention.skip_input_sanitizer";
      if (!policy.empty()) {
        drafter.session_options->config_entries.emplace_back(key, policy);
      }
      const auto projected = CreateNemotronBoltDrafterConfig(config);
      const auto& entries = projected->model.decoder.session_options.config_entries;
      const auto length_policy = std::find_if(entries.begin(), entries.end(), [](const auto& option) {
        return option.first == key;
      });
      if (version < 32 && policy.empty()) {
        EXPECT_EQ(length_policy, entries.end());
      } else {
        ASSERT_NE(length_policy, entries.end());
        EXPECT_EQ(length_policy->second, policy.empty() ? "1" : policy);
        EXPECT_EQ(std::count_if(entries.begin(), entries.end(), [](const auto& option) {
                    return option.first == key;
                  }),
                  1);
      }
      EXPECT_EQ(drafter.session_options->config_entries.size(), policy.empty() ? 0u : 1u);
    }
  }
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

std::unique_ptr<Config> BoltTestConfig(const char* path) {
  auto config = std::make_unique<Config>(fs::path{path}, "{}");
  for (auto* session : {&config->model.decoder.session_options,
                        config->model.dflash2.session_options
                            ? &*config->model.dflash2.session_options
                            : nullptr}) {
    if (!session) continue;
    session->intra_op_num_threads = 1;
    session->inter_op_num_threads = 1;
    for (auto& provider : session->provider_options) {
      if (provider.name != "cuda") continue;
      auto warmup = std::find_if(provider.options.begin(), provider.options.end(),
                                 [](const auto& option) {
                                   return option.first == "min_num_runs_before_cuda_graph_capture";
                                 });
      if (warmup != provider.options.end()) {
        if (warmup->second != "2")
          throw std::runtime_error("Bolt diagnostic requires exactly two CUDA graph warmups");
        // The built-in EP fixes this count at two; only the plugin EP accepts the option.
        provider.options.erase(warmup);
      }
    }
    if (std::getenv("ORTGENAI_BOLT_BUILTIN")) {
      session->custom_ops_library.reset();
      session->custom_ops_libraries.clear();
    }
  }
  return config;
}

// Optional real-graph correctness coverage. No models or GPU are needed for the tests above.
TEST(NemotronBoltGPU, TargetOnlyCaptureAndReset) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto config = BoltTestConfig(path);
  config->speculative.max_draft_tokens = 0;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto& bolt = dynamic_cast<NemotronBoltModel&>(*model);
  ASSERT_FALSE(bolt.drafter_model_);
  const std::vector<int32_t> prompt{
      10, 25708, 1010, 11, 1010, 10, 3263, 1010, 18746, 1261, 17616, 2254, 1455,
      29196, 5627, 1261, 3310, 1395, 1261, 5796, 92145, 1044, 1454, 1261, 4958,
      17984, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13};
  std::vector<int32_t> expected;
  std::vector<uint8_t> expected_logits;
  for (bool capture : {false, true}) {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->use_graph_capture = capture;
    params->search.max_length = static_cast<int>(prompt.size()) + 32;
    Generator generator{*model, *params};
    for (int request = 0; request < 2; ++request) {
      std::vector<uint8_t> logits;
      generator.AppendTokens(prompt);
      while (!generator.IsDone()) {
        generator.GenerateNextToken();
        const auto row = NemotronBoltTestAccess::SelectedLogits(
            dynamic_cast<NemotronBoltState&>(*generator.state_));
        logits.insert(logits.end(), row.begin(), row.end());
      }
      const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
      const std::vector<int32_t> result(sequence.begin(), sequence.end());
      if (expected.empty()) {
        expected = result;
        expected_logits = logits;
      }
      EXPECT_EQ(result, expected);
      EXPECT_EQ(logits.size(), expected_logits.size());
      EXPECT_TRUE(logits == expected_logits);
      const auto stats = generator.GetSpeculativeStats();
      EXPECT_EQ(stats.draft_forward_passes, 0);
      EXPECT_EQ(stats.rounds, 0);
      EXPECT_EQ(stats.effective_k, 0);
      EXPECT_EQ(stats.tokens_emitted, result.size() - prompt.size());
      generator.RewindToLength(0);
      EXPECT_EQ(generator.TokenCount(), 0);
    }
  }
  if (const char* output = std::getenv("ORTGENAI_BOLT_OUTPUT")) {
    std::ofstream tokens{std::string{output} + ".tokens"};
    for (int32_t token : expected) tokens << token << "\n";
    std::ofstream logits{std::string{output} + ".logits", std::ios::binary};
    logits.write(reinterpret_cast<const char*>(expected_logits.data()),
                 static_cast<std::streamsize>(expected_logits.size()));
    ASSERT_TRUE(tokens && logits);
  }
#if defined(__linux__)
  if (std::getenv("ORTGENAI_BOLT_BUILTIN")) {
    std::ifstream maps{"/proc/self/maps"};
    ASSERT_TRUE(maps);
    const std::string libraries{std::istreambuf_iterator<char>{maps}, {}};
    EXPECT_EQ(libraries.find("libbolt"), std::string::npos);
  }
#endif
}

TEST(NemotronBoltGPU, OptInOneTokenMatchesEightRowCommittedStates) {
  const char* reference_path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  const char* candidate_path = std::getenv("ORTGENAI_BOLT_R1_MODEL");
  if (!reference_path || !candidate_path)
    GTEST_SKIP() << "Set native R8 and opt-in attention-padded R1 packages";
  auto reference_config = BoltTestConfig(reference_path);
  auto candidate_config = BoltTestConfig(candidate_path);
  reference_config->speculative.max_draft_tokens = 0;
  candidate_config->speculative.max_draft_tokens = 0;
  auto reference = CreateModel(GetOrtEnv(), std::move(reference_config));
  auto candidate = CreateModel(GetOrtEnv(), std::move(candidate_config));
  ASSERT_TRUE(dynamic_cast<NemotronBoltModel&>(*candidate).single_token_decode_);
  const std::vector<int32_t> prompt{
      10, 25708, 1010, 11, 1010, 10, 3263, 1010, 18746, 1261, 17616, 2254, 1455,
      29196, 5627, 1261, 3310, 1395, 1261, 5796, 92145, 1044, 1454, 1261, 4958,
      17984, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13};
  std::ofstream state_report;
  if (const char* output = std::getenv("ORTGENAI_BOLT_OUTPUT"))
    state_report.open(std::string{output} + "-r1-live-states.jsonl");
  for (const bool capture : {false, true}) {
    auto reference_params = std::make_shared<GeneratorParams>(*reference);
    auto candidate_params = std::make_shared<GeneratorParams>(*candidate);
    reference_params->use_graph_capture = candidate_params->use_graph_capture = capture;
    reference_params->search.max_length = candidate_params->search.max_length =
        static_cast<int>(prompt.size()) + 96;
    Generator baseline{*reference, *reference_params};
    Generator optimized{*candidate, *candidate_params};
    auto& baseline_state = dynamic_cast<NemotronBoltState&>(*baseline.state_);
    auto& optimized_state = dynamic_cast<NemotronBoltState&>(*optimized.state_);
    for (int repeat = 0; repeat < 2; ++repeat) {
      baseline.AppendTokens(prompt);
      optimized.AppendTokens(prompt);
      do {
        auto comparison = NemotronBoltTestAccess::CompareLiveBoundary(
            baseline_state, optimized_state, NemotronBoltTestAccess::SelectedRow(optimized_state));
        comparison["capture"] = capture;
        comparison["repeat"] = repeat;
        if (state_report.is_open()) state_report << comparison.dump() << std::endl;
        ASSERT_TRUE(comparison.at("differences").empty()) << comparison.dump();
        ASSERT_EQ(baseline.IsDone(), optimized.IsDone());
        if (baseline.IsDone()) break;
        baseline.GenerateNextToken();
        optimized.GenerateNextToken();
      } while (true);
      const auto a = baseline.GetSequence(0).CopyDeviceToCpu();
      const auto b = optimized.GetSequence(0).CopyDeviceToCpu();
      EXPECT_EQ(std::vector<int32_t>(a.begin(), a.end()), std::vector<int32_t>(b.begin(), b.end()));
      if (state_report.is_open())
        state_report << nlohmann::json{{"capture", capture}, {"repeat", repeat}, {"reference_tokens", std::vector<int32_t>(a.begin(), a.end())}, {"candidate_tokens", std::vector<int32_t>(b.begin(), b.end())}}.dump()
                     << std::endl;
      EXPECT_EQ(baseline.GetSpeculativeStats().tokens_emitted,
                optimized.GetSpeculativeStats().tokens_emitted);
      EXPECT_EQ(baseline.GetSpeculativeStats().target_forward_passes,
                optimized.GetSpeculativeStats().target_forward_passes);
      baseline.RewindToLength(0);
      optimized.RewindToLength(0);
    }
  }
}

TEST(NemotronBoltGPU, SequentialGeneratorsResetAndPublicBoundaries) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto model = CreateModel(GetOrtEnv(), BoltTestConfig(path));
  auto& bolt = dynamic_cast<NemotronBoltModel&>(*model);
  ASSERT_TRUE(bolt.drafter_model_);
  ASSERT_FALSE(bolt.config_->model.decoder.shared_initializers.empty());
  ASSERT_EQ(bolt.shared_initializer_entries_.size(),
            bolt.config_->model.decoder.shared_initializers.size());
  ASSERT_EQ(bolt.drafter_model_->shared_initializer_entries_.size(),
            bolt.drafter_model_->config_->model.decoder.shared_initializers.size());
  ASSERT_EQ(bolt.shared_initializer_entries_.size(),
            bolt.drafter_model_->shared_initializer_entries_.size());
  ASSERT_EQ(bolt.shared_initializer_entries_.size(),
            bolt.drafter_model_->shared_initializer_entries_.size());
  for (size_t i = 0; i < bolt.shared_initializer_entries_.size(); ++i) {
    const auto& target = bolt.config_->model.decoder.shared_initializers[i];
    const auto& draft = bolt.drafter_model_->config_->model.decoder.shared_initializers[i];
    const auto target_path = bolt.config_->ResolvePath(target.data_file);
    const auto draft_path = bolt.drafter_model_->config_->ResolvePath(draft.data_file);
#ifdef _WIN32
    const bool same_file = target_path.string() == draft_path.string();
#else
    const bool same_file = std::filesystem::equivalent(target_path.string(), draft_path.string());
#endif
    const auto offset = [](const std::string& value) {
      return value.empty() ? uint64_t{0} : std::stoull(value);
    };
    const auto& target_memory = bolt.p_device_->GetAllocator().GetInfo();
    const auto& draft_memory = bolt.drafter_model_->p_device_->GetAllocator().GetInfo();
    const bool same_region = same_file && offset(target.offset) == offset(draft.offset) &&
                             std::stoull(target.length) == std::stoull(draft.length) &&
                             target.data_type == draft.data_type && target.shape == draft.shape &&
                             target_memory.GetDeviceType() == draft_memory.GetDeviceType() &&
                             target_memory.GetDeviceId() == draft_memory.GetDeviceId();
    // Split data files contain copies; registry sharing follows the source region, not equal values.
    if (same_region)
      EXPECT_EQ(bolt.shared_initializer_entries_[i].get(),
                bolt.drafter_model_->shared_initializer_entries_[i].get());
    else
      EXPECT_NE(bolt.shared_initializer_entries_[i].get(),
                bolt.drafter_model_->shared_initializer_entries_[i].get());
  }
  // A real, non-thinking chat asking for a palindrome function. Length 35 also exercises a
  // three-row final prefill chunk and its nonzero recurrent/anchor row.
  const std::vector<int32_t> prompt{
      10, 25708, 1010, 11, 1010, 10, 3263, 1010, 18746, 1261, 17616, 2254, 1455,
      29196, 5627, 1261, 3310, 1395, 1261, 5796, 92145, 1044, 1454, 1261, 4958,
      17984, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13};
  std::vector<int32_t> baseline;
  for (int request = 0; request < 3; ++request) {
    SCOPED_TRACE(request);
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size() + 32);
    params->search.do_sample = request != 0;
    params->search.temperature = request == 1 ? 0.0f : 1.0f;
    params->search.top_k = request == 2 ? 1 : 50;
    Generator generator{*model, *params};
    for (int reset = 0; reset < 2; ++reset) {
      if (reset == 1) {
        generator.AppendTokens(prompt);
        generator.GenerateNextToken();
        generator.GenerateNextToken();
        EXPECT_GT(generator.GetSpeculativeStats().tokens_buffered, 0);
        generator.RewindToLength(0);
      }
      generator.AppendTokens(prompt);
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
      if (baseline.empty()) baseline = result;
      EXPECT_EQ(result, baseline);
      EXPECT_EQ(result.size(), prompt.size() + 32);
      const auto stats = generator.GetSpeculativeStats();
      EXPECT_EQ(stats.tokens_emitted, 32);
      EXPECT_EQ(stats.tokens_queued, 32);
      EXPECT_EQ(stats.tokens_buffered, 0);
      EXPECT_GT(stats.full_accept_rounds, 0);
      EXPECT_GT(stats.partial_accept_rounds, 0);
      EXPECT_EQ(stats.zero_accept_rounds + stats.partial_accept_rounds + stats.full_accept_rounds,
                stats.rounds);
      generator.RewindToLength(0);
      EXPECT_EQ(generator.TokenCount(), 0);
    }
  }
  {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.do_sample = true;
    params->search.temperature = 1.0f;
    params->search.top_k = 50;
    EXPECT_THROW(Generator(*model, *params), std::runtime_error);
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
  auto params = std::make_shared<GeneratorParams>(*model);
  params->search.max_length = static_cast<int>(prompt.size()) + 32;
  Generator serial{*model, *params};
  serial.AppendTokens(prompt);
  const auto target_only = NemotronBoltTestAccess::TargetOnly(
      dynamic_cast<NemotronBoltState&>(*serial.state_), 32);
  EXPECT_TRUE(std::equal(target_only.begin(), target_only.end(), baseline.begin() + prompt.size()));
  for (bool capture : {false, true}) {
    auto ordinary_params = std::make_shared<GeneratorParams>(*model);
    ordinary_params->speculative.max_draft_tokens = 0;
    ordinary_params->use_graph_capture = capture;
    ordinary_params->search.max_length = static_cast<int>(prompt.size()) + 32;
    Generator ordinary{*model, *ordinary_params};
    for (int request = 0; request < 2; ++request) {
      ordinary.AppendTokens(prompt);
      while (!ordinary.IsDone()) ordinary.GenerateNextToken();
      const auto sequence = ordinary.GetSequence(0).CopyDeviceToCpu();
      EXPECT_TRUE(std::equal(sequence.begin(), sequence.end(), baseline.begin(), baseline.end()));
      const auto stats = ordinary.GetSpeculativeStats();
      EXPECT_EQ(stats.draft_forward_passes, 0);
      EXPECT_EQ(stats.rounds, 0);
      EXPECT_EQ(stats.effective_k, 0);
      EXPECT_EQ(stats.tokens_emitted, 32);
      EXPECT_EQ(stats.tokens_buffered, 0);
      ordinary.RewindToLength(0);
    }
  }
  {
    Generator generator{*model, *params};
    generator.AppendTokens(prompt);
    generator.GenerateNextToken();
    NemotronBoltTestAccess::ForceRejectedDrafts(dynamic_cast<NemotronBoltState&>(*generator.state_));
    generator.GenerateNextToken();
    const auto stats = generator.GetSpeculativeStats();
    EXPECT_EQ(stats.zero_accept_rounds, 1);
    EXPECT_EQ(stats.partial_accept_rounds, 0);
    EXPECT_EQ(stats.full_accept_rounds, 0);
    EXPECT_EQ(stats.rounds, 1);
    while (!generator.IsDone()) generator.GenerateNextToken();
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    EXPECT_TRUE(std::equal(sequence.begin(), sequence.end(), baseline.begin()));
  }

  // Cross multiple paged-cache blocks and wrap the drafter's 1024-position sliding-window ring.
  std::vector<int32_t> long_prompt;
  for (int i = 0; i < 32; ++i) long_prompt.insert(long_prompt.end(), prompt.begin(), prompt.end());
  auto long_params = std::make_shared<GeneratorParams>(*model);
  long_params->search.max_length = static_cast<int>(long_prompt.size() + 16);
  std::vector<int32_t> long_result;
  {
    Generator generator{*model, *long_params};
    generator.AppendTokens(long_prompt);
    while (!generator.IsDone()) generator.GenerateNextToken();
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    long_result.assign(sequence.begin() + long_prompt.size(), sequence.end());
  }
  ASSERT_FALSE(long_result.empty());
  {
    Generator generator{*model, *long_params};
    generator.AppendTokens(long_prompt);
    const auto expected = NemotronBoltTestAccess::TargetOnly(
        dynamic_cast<NemotronBoltState&>(*generator.state_), long_result.size());
    EXPECT_EQ(long_result, expected);
  }
#if defined(__linux__)
  if (std::getenv("ORTGENAI_BOLT_BUILTIN")) {
    std::ifstream maps{"/proc/self/maps"};
    ASSERT_TRUE(maps);
    const std::string libraries{std::istreambuf_iterator<char>{maps}, {}};
    EXPECT_EQ(libraries.find("libbolt"), std::string::npos);
  }
#endif
}

TEST(NemotronBoltGPU, RealPromptOrdinaryAndSpeculative) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  const char* cases_path = std::getenv("ORTGENAI_BOLT_TEST_CASES");
  const char* output = std::getenv("ORTGENAI_BOLT_OUTPUT");
  if (!path || !cases_path || !output)
    GTEST_SKIP() << "Set model, exact token fixtures and owned output path";
  std::ifstream input{cases_path};
  ASSERT_TRUE(input);
  nlohmann::json fixtures;
  input >> fixtures;
  ASSERT_FALSE(fixtures.at("cases").empty());
  std::ofstream report{std::string{output} + ".jsonl"};
  ASSERT_TRUE(report);
  auto model = CreateModel(GetOrtEnv(), BoltTestConfig(path));
  for (const auto& fixture : fixtures.at("cases")) {
    const auto case_id = fixture.at("case_id").get<std::string>();
    SCOPED_TRACE(case_id);
    const auto prompt = fixture.at("prompt_ids").get<std::vector<int32_t>>();
    const auto budget = fixture.at("max_new_tokens").get<size_t>();
    ASSERT_FALSE(prompt.empty());
    ASSERT_GT(budget, 0);
    ASSERT_LE(budget, 512);
    std::array<std::vector<int32_t>, 2> generated;
    nlohmann::json modes = nlohmann::json::array();
    for (size_t mode = 0; mode < generated.size(); ++mode) {
      auto params = std::make_shared<GeneratorParams>(*model);
      params->speculative.max_draft_tokens = mode == 0 ? 0 : 7;
      params->use_graph_capture = true;
      params->search.max_length = static_cast<int>(prompt.size() + budget);
      Generator generator{*model, *params};
      generator.AppendTokens(prompt);
      while (!generator.IsDone()) generator.GenerateNextToken();
      const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
      ASSERT_GE(sequence.size(), prompt.size());
      EXPECT_TRUE(std::equal(prompt.begin(), prompt.end(), sequence.begin()));
      generated[mode].assign(sequence.begin() + prompt.size(), sequence.end());
      ASSERT_FALSE(generated[mode].empty());
      EXPECT_LE(generated[mode].size(), budget);
      const auto& eos = model->config_->model.eos_token_id;
      const bool ended_with_eos = std::find(eos.begin(), eos.end(), generated[mode].back()) != eos.end();
      EXPECT_TRUE(ended_with_eos || generated[mode].size() == budget);
      const auto stats = generator.GetSpeculativeStats();
      EXPECT_EQ(stats.tokens_emitted, generated[mode].size());
      EXPECT_EQ(stats.dflash2_failures, 0);
      EXPECT_EQ(stats.dflash2_disables, 0);
      if (mode == 0) {
        EXPECT_EQ(stats.rounds, 0);
        EXPECT_EQ(stats.draft_forward_passes, 0);
      } else if (generated[mode].size() > 1) {
        EXPECT_GT(stats.rounds, 0);
        EXPECT_GT(stats.draft_forward_passes, 0);
      }
      modes.push_back({
          {"k", params->speculative.max_draft_tokens},
          {"generated_ids", generated[mode]},
          {"ended_with_eos", ended_with_eos},
          {"reached_output_cap", generated[mode].size() == budget},
          {"rounds", stats.rounds},
          {"draft_forward_passes", stats.draft_forward_passes},
          {"acceptance_length_histogram", stats.acceptance_length_histogram},
          {"dflash2_failures", stats.dflash2_failures},
          {"dflash2_disables", stats.dflash2_disables},
          {"tokens_emitted", stats.tokens_emitted},
          {"tokens_buffered", stats.tokens_buffered},
      });
    }
    report << nlohmann::json{
                  {"case_id", case_id},
                  {"prompt_tokens", prompt.size()},
                  {"same_generated_ids", generated[0] == generated[1]},
                  {"modes", modes}}
                  .dump()
           << "\n";
    report.flush();
    ASSERT_TRUE(report);
    EXPECT_EQ(generated[0], generated[1]);
  }
#if defined(__linux__)
  if (std::getenv("ORTGENAI_BOLT_BUILTIN")) {
    std::ifstream maps{"/proc/self/maps"};
    ASSERT_TRUE(maps);
    const std::string libraries{std::istreambuf_iterator<char>{maps}, {}};
    EXPECT_EQ(libraries.find("libbolt"), std::string::npos);
  }
#endif
}

TEST(NemotronBolt, GgufInitializersConfigParsingAndPathValidation) {
  namespace fs_std = std::filesystem;
  const auto tmp_dir = fs_std::temp_directory_path() / "ortgenai_gguf_init_test";
  std::error_code ec;
  fs_std::remove_all(tmp_dir, ec);
  fs_std::create_directories(tmp_dir);

  auto write_file = [&](const fs_std::path& p, const std::string& content) {
    std::ofstream out(p, std::ios::binary);
    out << content;
  };

  // 1. String form for decoder, inline object form for dflash2
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "nemotron_bolt",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "model.onnx",
        "gguf_initializers": "gguf_initializers.json"
      },
      "dflash2": {
        "filename": "dflash2.onnx",
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "head_size": 64,
        "block_size": 8,
        "num_draft_tokens": 7,
        "selector_top_k": 8,
        "mask_token_id": 100,
        "sliding_window": 1024,
        "gguf_initializers": {
          "files": ["weights.gguf"],
          "share": true,
          "tensors": {
            "embed_w": "token_embd.weight",
            "lm_head": {
              "gguf_name": "output.weight",
              "data_type": 2,
              "shape": [128, 34],
              "share": false
            }
          }
        }
      }
    },
    "search": {}
  })");

  Config cfg(fs::path{tmp_dir.string()}, {});
  EXPECT_EQ(cfg.model.decoder.gguf_initializers.config_file, "gguf_initializers.json");
  ASSERT_EQ(cfg.model.dflash2.gguf_initializers.files.size(), 1u);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.files[0], "weights.gguf");
  EXPECT_TRUE(cfg.model.dflash2.gguf_initializers.share);
  EXPECT_TRUE(cfg.model.dflash2.gguf_initializers.has_share);
  ASSERT_EQ(cfg.model.dflash2.gguf_initializers.tensors.size(), 2u);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[0].onnx_name, "embed_w");
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[0].gguf_name, "token_embd.weight");
  EXPECT_FALSE(cfg.model.dflash2.gguf_initializers.tensors[0].has_share);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].onnx_name, "lm_head");
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].gguf_name, "output.weight");
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].data_type, 2);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].shape, (std::vector<int64_t>{128, 34}));
  EXPECT_TRUE(cfg.model.dflash2.gguf_initializers.tensors[1].has_share);
  EXPECT_FALSE(cfg.model.dflash2.gguf_initializers.tensors[1].share);

  auto dflash2_cfg = CreateDflash2Config(cfg);
  EXPECT_EQ(dflash2_cfg->model.decoder.gguf_initializers.files, cfg.model.dflash2.gguf_initializers.files);
  EXPECT_EQ(dflash2_cfg->model.decoder.gguf_initializers.tensors.size(), 2u);

  // Per-tensor file and data type overrides for mtp
  Config mtp_cfg;
  OverlayConfig(mtp_cfg, R"({
    "model": {
      "mtp": {
        "filename": "mtp.onnx",
        "gguf_initializers": {
          "files": ["mtp.gguf"],
          "tensors": {
            "w0": {
              "gguf_name": "blk.0.weight",
              "file": "shard0.gguf",
              "data_type": 1
            }
          }
        }
      }
    }
  })");
  ASSERT_EQ(mtp_cfg.model.mtp.gguf_initializers.files.size(), 1u);
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.files[0], "mtp.gguf");
  ASSERT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors.size(), 1u);
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].onnx_name, "w0");
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].gguf_name, "blk.0.weight");
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].file, "shard0.gguf");
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].data_type, 1);

  // 2. LoadGgufInitializersConfig from external JSON file and reject path traversal
  write_file(tmp_dir / "gguf_initializers.json", R"({
    "files": ["model.gguf"],
    "tensors": {
      "embed": {
        "gguf_name": "token_embd.weight",
        "share": true
      }
    }
  })");
  Config::Model::GgufInitializers ext_loaded;
  LoadGgufInitializersConfig(fs::path{(tmp_dir / "gguf_initializers.json").string()}, ext_loaded);
  ASSERT_EQ(ext_loaded.files.size(), 1u);
  EXPECT_EQ(ext_loaded.files[0], "model.gguf");
  ASSERT_EQ(ext_loaded.tensors.size(), 1u);
  EXPECT_EQ(ext_loaded.tensors[0].onnx_name, "embed");
  EXPECT_TRUE(ext_loaded.tensors[0].share);

  // Path traversal in external gguf_initializers.json must be rejected
  write_file(tmp_dir / "bad_gguf_initializers.json", R"({
    "files": ["../outside.gguf"],
    "tensors": {}
  })");
  Config::Model::GgufInitializers bad_loaded;
  EXPECT_THROW(LoadGgufInitializersConfig(fs::path{(tmp_dir / "bad_gguf_initializers.json").string()}, bad_loaded),
               std::runtime_error);

  // Path traversal in inline genai_config.json gguf_initializers must be rejected
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "nemotron_bolt",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "model.onnx",
        "gguf_initializers": "../outside.json"
      }
    },
    "search": {}
  })");
  EXPECT_THROW(Config(fs::path{tmp_dir.string()}, {}), std::runtime_error);

  // 3. Synthetic GGUF v3 binary header parsing and AddGgufInitializers validation
  auto write_u32 = [](std::string& buf, uint32_t v) {
    buf.append(reinterpret_cast<const char*>(&v), 4);
  };
  auto write_u64 = [](std::string& buf, uint64_t v) {
    buf.append(reinterpret_cast<const char*>(&v), 8);
  };
  auto write_gguf_str = [&](std::string& buf, std::string_view s) {
    write_u64(buf, s.size());
    buf.append(s.data(), s.size());
  };

  std::string gguf_bytes;
  gguf_bytes.append("GGUF", 4);
  write_u32(gguf_bytes, 3);  // version 3
  write_u64(gguf_bytes, 2);  // tensor_count = 2
  write_u64(gguf_bytes, 1);  // kv_count = 1
  write_gguf_str(gguf_bytes, "general.alignment");
  write_u32(gguf_bytes, 4);  // UINT32
  write_u32(gguf_bytes, 32);

  // Tensor 0: "weight_f32", dims=[4, 2] -> ONNX default shape [2, 4] FLOAT (32 bytes), offset 0
  write_gguf_str(gguf_bytes, "weight_f32");
  write_u32(gguf_bytes, 2);
  write_u64(gguf_bytes, 4);
  write_u64(gguf_bytes, 2);
  write_u32(gguf_bytes, 0);  // F32
  write_u64(gguf_bytes, 0);

  // Tensor 1: "weight_q8", dims=[32, 2] -> ONNX default shape [2, 34] UINT8 (68 bytes), offset 32
  write_gguf_str(gguf_bytes, "weight_q8");
  write_u32(gguf_bytes, 2);
  write_u64(gguf_bytes, 32);
  write_u64(gguf_bytes, 2);
  write_u32(gguf_bytes, 8);  // Q8_0
  write_u64(gguf_bytes, 32);

  while (gguf_bytes.size() % 32 != 0) gguf_bytes.push_back('\0');
  gguf_bytes.append(32 + 68, '\x01');
  write_file(tmp_dir / "synthetic.gguf", gguf_bytes);

  // Missing GGUF tensor name throws from AddGgufInitializers before CreateSession
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "gpt2",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "nonexistent_model.onnx",
        "gguf_initializers": {
          "files": ["synthetic.gguf"],
          "tensors": {
            "w_missing": "does_not_exist"
          }
        }
      }
    },
    "search": {}
  })");
  try {
    CreateModel(GetOrtEnv(), tmp_dir.string().c_str());
    FAIL() << "Expected missing GGUF tensor to throw";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("does_not_exist"), std::string::npos) << e.what();
  }

  // Byte size mismatch throws from AddGgufInitializers
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "gpt2",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "nonexistent_model.onnx",
        "gguf_initializers": {
          "files": ["synthetic.gguf"],
          "tensors": {
            "w0": {
              "gguf_name": "weight_f32",
              "shape": [2, 5]
            }
          }
        }
      }
    },
    "search": {}
  })");
  try {
    CreateModel(GetOrtEnv(), tmp_dir.string().c_str());
    FAIL() << "Expected byte size mismatch to throw";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("Byte size mismatch"), std::string::npos) << e.what();
  }

  // Valid GGUF tensor mappings succeed in AddGgufInitializers and reach CreateSession(nonexistent_model.onnx)
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "gpt2",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "nonexistent_model.onnx",
        "gguf_initializers": {
          "files": ["synthetic.gguf"],
          "tensors": {
            "w0": "weight_f32",
            "w1": "weight_q8"
          }
        }
      }
    },
    "search": {}
  })");
  try {
    CreateModel(GetOrtEnv(), tmp_dir.string().c_str());
    FAIL() << "Expected nonexistent_model.onnx to throw after AddGgufInitializers succeeds";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("nonexistent_model.onnx"), std::string::npos) << e.what();
  }

  fs_std::remove_all(tmp_dir, ec);
}

}  // namespace
}  // namespace Generators::test
