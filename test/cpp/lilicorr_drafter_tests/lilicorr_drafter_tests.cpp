// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Runs the real Dflash2Drafter (paged K/V ring, packed row maps, anchor_row_index, optional CUDA graph) on a LiLiCorr
// drafter exported by builders/lilicorr.py and compares its greedy lattice walk with the Python reference steps.
//
//   LILICORR_TEST_DIR   directory with genai_config.json, dflash2.onnx[.data] and steps.bin
//   LILICORR_EP_LIBRARY (optional) CUDA plugin EP library to register as "CUDAExecutionProvider"
//
// steps.bin (little endian): int32 steps; per step int32 rows, anchor, first_position; rows * aux_width bf16;
// 7 int32 expected draft tokens.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "dflash2_drafter.h"
#include "generator/generators.h"
#include "ort_genai.h"

namespace {

namespace fs_std = std::filesystem;

struct Step {
  int32_t rows{};
  int32_t anchor{};
  int32_t first_position{};
  std::vector<uint16_t> aux;
  std::vector<int32_t> expected;
};

std::vector<Step> LoadSteps(const fs_std::path& path, size_t aux_width, size_t num_drafts) {
  std::ifstream in(path, std::ios::binary);
  EXPECT_TRUE(in.good()) << path;
  int32_t count = 0;
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  std::vector<Step> steps(static_cast<size_t>(count));
  for (auto& step : steps) {
    in.read(reinterpret_cast<char*>(&step.rows), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&step.anchor), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&step.first_position), sizeof(int32_t));
    step.aux.resize(static_cast<size_t>(step.rows) * aux_width);
    in.read(reinterpret_cast<char*>(step.aux.data()), static_cast<std::streamsize>(step.aux.size() * sizeof(uint16_t)));
    step.expected.resize(num_drafts);
    in.read(reinterpret_cast<char*>(step.expected.data()), static_cast<std::streamsize>(num_drafts * sizeof(int32_t)));
  }
  return steps;
}

TEST(LiLiCorrDrafterTests, ProposesTheReferenceWalkCuda) {
  const char* dir_env = std::getenv("LILICORR_TEST_DIR");
  if (dir_env == nullptr) {
    GTEST_SKIP() << "LILICORR_TEST_DIR is not set";
  }
  const fs_std::path dir{dir_env};

  Generators::Config config(fs::path{dir.string()}, "");
  const auto& dflash2 = config.model.dflash2;
  ASSERT_FALSE(dflash2.inputs.anchor_row_index.empty());
  const size_t aux_width = static_cast<size_t>(dflash2.aux_hidden_state_layers.size()) * 4096;
  const size_t num_drafts = static_cast<size_t>(dflash2.num_draft_tokens);
  const auto steps = LoadSteps(dir / "steps.bin", aux_width, num_drafts);
  ASSERT_FALSE(steps.empty());

  auto model = std::make_shared<Generators::Dflash2Model>(Generators::CreateDflash2Config(config), Generators::GetOrtEnv());
  constexpr size_t kBlockSize = 256;
  Generators::Dflash2Drafter drafter(model, kBlockSize, Generators::Dflash2Drafter::PoolBlocks(config, kBlockSize, 1), 1);

  auto* device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  // The drafter keys its per-request state by pointer and never dereferences it.
  auto* request = reinterpret_cast<Generators::Request*>(0x1000);
  size_t matches = 0;
  for (size_t i = 0; i < steps.size(); ++i) {
    const auto& step = steps[i];
    Generators::Tensor aux(device, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    aux.CreateTensor(std::vector<int64_t>{step.rows, static_cast<int64_t>(aux_width)});
    auto bytes = aux.GetByteSpan();
    std::memcpy(bytes.CpuSpan().data(), step.aux.data(), step.aux.size() * sizeof(uint16_t));
    bytes.CopyCpuToDevice();

    Generators::Dflash2Drafter::Feed feed;
    feed.request = request;
    feed.aux_row_begin = 0;
    feed.aux_row_count = static_cast<size_t>(step.rows);
    feed.first_position = static_cast<size_t>(step.first_position);
    feed.anchor_token = step.anchor;
    feed.draft_eligible = true;
    feed.wants_drafts = true;
    std::vector<std::vector<int32_t>> drafts;
    ASSERT_TRUE(drafter.Propose(aux, std::span<const Generators::Dflash2Drafter::Feed>(&feed, 1), drafts));
    ASSERT_EQ(drafts.size(), 1u);
    ASSERT_EQ(drafts[0].size(), num_drafts) << "step " << i;
    const bool same = drafts[0] == step.expected;
    matches += same;
    if (!same) {
      std::string got, want;
      for (size_t j = 0; j < num_drafts; ++j) {
        got += std::to_string(drafts[0][j]) + " ";
        want += std::to_string(step.expected[j]) + " ";
      }
      ADD_FAILURE() << "step " << i << " rows " << step.rows << ": drafts " << got << "!= reference " << want;
    }
  }
  EXPECT_EQ(matches, steps.size());

  // Optional: steady-state latency of a 8-row-context step (what a decode step looks like after a full-acceptance verify).
  if (std::getenv("LILICORR_BENCH") != nullptr) {
    const Step* source = nullptr;
    for (const auto& step : steps) {
      if (step.rows == 8) source = &step;
    }
    ASSERT_NE(source, nullptr);
    size_t position = steps.back().first_position + steps.back().rows;
    const int warmup = 5, iterations = 50;
    double total_ms = 0;
    for (int it = 0; it < warmup + iterations; ++it) {
      Generators::Tensor aux(device, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
      aux.CreateTensor(std::vector<int64_t>{8, static_cast<int64_t>(aux_width)});
      auto bytes = aux.GetByteSpan();
      std::memcpy(bytes.CpuSpan().data(), source->aux.data(), source->aux.size() * sizeof(uint16_t));
      bytes.CopyCpuToDevice();
      Generators::Dflash2Drafter::Feed feed;
      feed.request = request;
      feed.aux_row_count = 8;
      feed.first_position = position;
      feed.anchor_token = source->anchor;
      feed.draft_eligible = feed.wants_drafts = true;
      std::vector<std::vector<int32_t>> drafts;
      const auto t0 = std::chrono::steady_clock::now();
      ASSERT_TRUE(drafter.Propose(aux, std::span<const Generators::Dflash2Drafter::Feed>(&feed, 1), drafts));
      const auto t1 = std::chrono::steady_clock::now();
      if (it >= warmup) total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      position += 8;
    }
    std::cout << "[ bench ] drafter step (8 context rows): " << total_ms / iterations << " ms" << std::endl;
  }
  drafter.Release(request);
}

}  // namespace

int main(int argc, char** argv) {
  // Creating any CUDA model initializes the CUDA device interface and the add-on library.
  ::testing::InitGoogleTest(&argc, argv);
  if (std::getenv("LILICORR_TEST_DIR") != nullptr) {
    if (const char* library = std::getenv("LILICORR_EP_LIBRARY")) {
      OgaRegisterExecutionProviderLibrary("CUDAExecutionProvider", library);
    }
    auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
    config->ClearProviders();
    config->AppendProvider("cuda");
    static auto model = OgaModel::Create(*config);
  }
  return RUN_ALL_TESTS();
}
