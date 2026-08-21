// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "generators.h"
#include "ort_genai.h"
#include "telemetry_test_environment.h"

namespace {

std::unique_ptr<OgaModel> CreateCudaModel() {
  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  config->ClearProviders();
  config->AppendProvider("cuda");
  return OgaModel::Create(*config);
}

TEST(SamplingTests, SchedulerOwnedSamplerHandlesHeterogeneousRowsCuda) {
  constexpr int vocab_size = 5;
  [[maybe_unused]] auto model = CreateCudaModel();
  auto* device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  auto sampler = device->CreateBatchedSampler(3, vocab_size);
  ASSERT_NE(sampler, nullptr);

  const std::array<std::array<float, vocab_size>, 3> logits{{
      {{0.0f, 5.0f, 1.0f, 2.0f, 3.0f}},
      {{0.0f, 1.0f, 6.0f, 2.0f, 3.0f}},
      {{0.0f, 1.0f, 2.0f, 7.0f, 3.0f}},
  }};
  std::vector<Generators::DeviceSpan<float>> rows;
  for (const auto& row_logits : logits) {
    auto row = device->Allocate<float>(vocab_size);
    std::copy(row_logits.begin(), row_logits.end(), row.CpuSpan().begin());
    row.CopyCpuToDevice();
    rows.push_back(std::move(row));
  }

  const std::array<Generators::BatchedSamplingParams, 3> params{{
      {1, 0.0f, 1.0f},
      {2, 0.01f, 0.7f},
      {-1, 0.01f, 1.3f},
  }};
  std::array<std::unique_ptr<Generators::BatchedSamplerState>, 3> owned_states;
  std::array<Generators::BatchedSamplerState*, 3> states;
  for (size_t row_index = 0; row_index < states.size(); ++row_index) {
    owned_states[row_index] = sampler->CreateState(static_cast<int>(10 + row_index));
    states[row_index] = owned_states[row_index].get();
  }

  auto tokens = sampler->Sample(rows, params, states, vocab_size).CopyDeviceToCpu();
  EXPECT_EQ(tokens[0], 1);
  EXPECT_EQ(tokens[1], 2);
  EXPECT_EQ(tokens[2], 3);
}

TEST(LogitsMaskTests, UsesPaddedRowStrideForNonAlignedVocabularyCuda) {
  constexpr int batch_size = 2;
  constexpr int vocab_size = 33;
  constexpr size_t words_per_row = 2;
  [[maybe_unused]] auto model = CreateCudaModel();
  auto* device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);

  auto logits = device->Allocate<float>(batch_size * vocab_size);
  std::fill(logits.CpuSpan().begin(), logits.CpuSpan().end(), 1.0f);
  logits.CopyCpuToDevice();

  const std::array<uint32_t, batch_size * words_per_row> host_mask{{
      uint32_t{1} << 0, uint32_t{1} << 0,
      uint32_t{1} << 1, uint32_t{1} << 0,
  }};
  auto mask = device->Allocate<uint32_t>(host_mask.size());
  std::copy(host_mask.begin(), host_mask.end(), mask.CpuSpan().begin());
  mask.CopyCpuToDevice();

  device->LaunchAddLogitsMask(
      logits.Span().data(), batch_size, vocab_size, mask.Span().data());
  const auto result = logits.CopyDeviceToCpu();

  for (int row = 0; row < batch_size; ++row) {
    for (int token = 0; token < vocab_size; ++token) {
      const bool allowed = token == 32 || token == row;
      EXPECT_EQ(result[row * vocab_size + token],
                allowed ? 1.0f : std::numeric_limits<float>::lowest())
          << "row=" << row << " token=" << token;
    }
  }
}

TEST(SamplingTests, SchedulerOwnedSamplerPreservesRngAcrossBatchReorderCuda) {
  constexpr int vocab_size = 8;
  [[maybe_unused]] auto model = CreateCudaModel();
  auto* device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  auto sampler_ab = device->CreateBatchedSampler(2, vocab_size);
  auto sampler_ba = device->CreateBatchedSampler(2, vocab_size);
  ASSERT_NE(sampler_ab, nullptr);
  ASSERT_NE(sampler_ba, nullptr);

  std::array<Generators::DeviceSpan<float>, 2> rows;
  const std::array<float, vocab_size> logits{{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f}};
  for (auto& row : rows) {
    row = device->Allocate<float>(vocab_size);
    std::copy(logits.begin(), logits.end(), row.CpuSpan().begin());
    row.CopyCpuToDevice();
  }

  const std::array<Generators::BatchedSamplingParams, 2> params{{
      {4, 1.0f, 1.0f},
      {4, 1.0f, 1.0f},
  }};
  auto state_a_ab = sampler_ab->CreateState(11);
  auto state_b_ab = sampler_ab->CreateState(29);
  auto state_b_ba = sampler_ba->CreateState(29);
  auto state_a_ba = sampler_ba->CreateState(11);
  std::array<Generators::BatchedSamplerState*, 2> states_ab{{state_a_ab.get(), state_b_ab.get()}};
  std::array<Generators::BatchedSamplerState*, 2> states_ba{{state_b_ba.get(), state_a_ba.get()}};
  std::array<Generators::DeviceSpan<float>, 2> rows_ba{{rows[1], rows[0]}};

  for (int step = 0; step < 8; ++step) {
    auto tokens_ab = sampler_ab->Sample(rows, params, states_ab, vocab_size).CopyDeviceToCpu();
    auto tokens_ba = sampler_ba->Sample(rows_ba, params, states_ba, vocab_size).CopyDeviceToCpu();
    EXPECT_EQ(tokens_ab[0], tokens_ba[1]);
    EXPECT_EQ(tokens_ab[1], tokens_ba[0]);
  }
}

// A batch where every row shares the same sampling parameters and the same allocation takes the
// contiguous fast path, which samples in place and therefore requires the packed RNG state indices
// to stay in original row order. The batch is deliberately larger than 16 rows because that is where
// libstdc++ introsort starts reordering elements that compare equal.
TEST(SamplingTests, SchedulerOwnedSamplerKeepsRngRowsAlignedForContiguousBatchCuda) {
  constexpr int batch_size = 20;
  constexpr int vocab_size = 8;
  [[maybe_unused]] auto model = CreateCudaModel();
  auto* device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  auto contiguous_sampler = device->CreateBatchedSampler(batch_size, vocab_size);
  auto separate_sampler = device->CreateBatchedSampler(batch_size, vocab_size);
  ASSERT_NE(contiguous_sampler, nullptr);
  ASSERT_NE(separate_sampler, nullptr);

  const std::array<float, vocab_size> logits{{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f}};

  // One allocation covering the whole batch, so rows are views into contiguous device memory.
  auto packed = device->Allocate<float>(static_cast<size_t>(batch_size) * vocab_size);
  auto packed_cpu = packed.CpuSpan();
  for (int row = 0; row < batch_size; ++row)
    std::copy(logits.begin(), logits.end(), packed_cpu.begin() + static_cast<size_t>(row) * vocab_size);
  packed.CopyCpuToDevice();

  std::vector<Generators::DeviceSpan<float>> contiguous_rows;
  std::vector<Generators::DeviceSpan<float>> separate_rows;
  for (int row = 0; row < batch_size; ++row) {
    contiguous_rows.push_back(packed.subspan(static_cast<size_t>(row) * vocab_size, vocab_size));
    auto separate_row = device->Allocate<float>(vocab_size);
    std::copy(logits.begin(), logits.end(), separate_row.CpuSpan().begin());
    separate_row.CopyCpuToDevice();
    separate_rows.push_back(std::move(separate_row));
  }

  const std::vector<Generators::BatchedSamplingParams> params(
      batch_size, Generators::BatchedSamplingParams{4, 1.0f, 1.0f});

  std::vector<std::unique_ptr<Generators::BatchedSamplerState>> contiguous_owned, separate_owned;
  std::vector<Generators::BatchedSamplerState*> contiguous_states, separate_states;
  for (int row = 0; row < batch_size; ++row) {
    const int seed = 1000 + 7 * row;
    contiguous_owned.push_back(contiguous_sampler->CreateState(seed));
    separate_owned.push_back(separate_sampler->CreateState(seed));
    contiguous_states.push_back(contiguous_owned.back().get());
    separate_states.push_back(separate_owned.back().get());
  }

  bool saw_distinct_tokens = false;
  for (int step = 0; step < 4; ++step) {
    auto sampled = contiguous_sampler->Sample(contiguous_rows, params, contiguous_states, vocab_size).CopyDeviceToCpu();
    const std::vector<int32_t> contiguous_tokens(sampled.begin(), sampled.end());
    auto separate_tokens = separate_sampler->Sample(separate_rows, params, separate_states, vocab_size).CopyDeviceToCpu();
    for (int row = 0; row < batch_size; ++row) {
      EXPECT_EQ(contiguous_tokens[row], separate_tokens[row])
          << "RNG stream leaked across rows at step " << step << ", row " << row;
      saw_distinct_tokens = saw_distinct_tokens || contiguous_tokens[row] != contiguous_tokens[0];
    }
  }
  // Guards against the comparison passing trivially because every seed picked the same token.
  EXPECT_TRUE(saw_distinct_tokens) << "Sampling was degenerate, so the row/RNG alignment was not exercised.";
}

TEST(SamplingTests, SchedulerOwnedSamplerCachesTopKPerBucketSizeCuda) {
  constexpr int batch_size = 3;
  constexpr int vocab_size = 100001;
  constexpr int top_k = 25;
  [[maybe_unused]] auto model = CreateCudaModel();
  auto* device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  auto sampler = device->CreateBatchedSampler(batch_size, vocab_size);
  ASSERT_NE(sampler, nullptr);

  std::array<Generators::DeviceSpan<float>, batch_size> rows;
  for (int row_index = 0; row_index < batch_size; ++row_index) {
    rows[row_index] = device->Allocate<float>(vocab_size);
    auto logits = rows[row_index].CpuSpan();
    std::fill(logits.begin(), logits.end(), 0.0f);
    logits[row_index + 1] = 100.0f;
    rows[row_index].CopyCpuToDevice();
  }

  const std::array<Generators::BatchedSamplingParams, batch_size> params{{
      {top_k, 0.01f, 1.0f},
      {top_k, 0.02f, 1.0f},
      {top_k, 0.02f, 1.0f},
  }};
  std::array<std::unique_ptr<Generators::BatchedSamplerState>, batch_size> owned_states;
  std::array<Generators::BatchedSamplerState*, batch_size> states;
  for (int row_index = 0; row_index < batch_size; ++row_index) {
    owned_states[row_index] = sampler->CreateState(row_index);
    states[row_index] = owned_states[row_index].get();
  }

  auto tokens = sampler->Sample(rows, params, states, vocab_size).CopyDeviceToCpu();
  for (int row_index = 0; row_index < batch_size; ++row_index)
    EXPECT_EQ(tokens[row_index], row_index + 1);
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  OgaShutdown();
  return result;
}