// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <array>
#include <cstdlib>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "models/model.h"

namespace Generators::test {
namespace {

std::shared_ptr<Model> CreateSharedInitializerTestModel(bool shared, bool cuda) {
  auto config = std::make_unique<Config>(
      fs::path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32"},
      R"({"model":{"decoder":{"filename":"external-data/past.onnx",)"
      R"("session_options":{"intra_op_num_threads":1,"inter_op_num_threads":1}}}})");
  if (cuda) {
    SetProviderOption(*config, "cuda", "enable_cuda_graph", "0");
  }
  if (shared) {
    OverlayConfig(*config,
                  R"({"model":{"decoder":{"shared_initializers":[{)"
                  R"("name":"transformer.wte.weight","data_file":"external-data/weights.data",)"
                  R"("offset":"0","length":"128000","data_type":1,"shape":[1000,32])"
                  R"(}]}}})");
  }
  return CreateModel(GetOrtEnv(), std::move(config));
}

std::vector<int32_t> GenerateSharedInitializerTestTokens(const Model& model) {
  auto params = std::make_shared<GeneratorParams>(model);
  params->search.max_length = 8;
  params->search.do_sample = false;
  Generator generator{model, *params};
  generator.AppendTokens(std::array<int32_t, 3>{0, 52, 195});
  while (!generator.IsDone())
    generator.GenerateNextToken();
  const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
  return {sequence.begin(), sequence.end()};
}

void CheckSharedInitializerLifetime(bool cuda) {
  const auto reference = CreateSharedInitializerTestModel(false, cuda);
  ASSERT_EQ(reference->p_device_->GetType(), cuda ? DeviceType::CUDA : DeviceType::CPU);
  const auto expected = GenerateSharedInitializerTestTokens(*reference);
  for (size_t destroyed_first : {0, 1}) {
    std::array<std::shared_ptr<Model>, 2> models{
        CreateSharedInitializerTestModel(true, cuda), CreateSharedInitializerTestModel(true, cuda)};
    ASSERT_EQ(models[0]->p_device_->GetType(), reference->p_device_->GetType());
    ASSERT_EQ(models[1]->p_device_->GetType(), reference->p_device_->GetType());
    ASSERT_EQ(models[0]->shared_initializer_entries_.size(), 1);
    ASSERT_EQ(models[1]->shared_initializer_entries_.size(), 1);
    EXPECT_EQ(models[0]->shared_initializer_entries_[0].get(),
              models[1]->shared_initializer_entries_[0].get());
    std::weak_ptr<void> allocation = models[0]->shared_initializer_entries_[0];
    EXPECT_EQ(GenerateSharedInitializerTestTokens(*models[0]), expected);
    EXPECT_EQ(GenerateSharedInitializerTestTokens(*models[1]), expected);

    models[destroyed_first].reset();
    ASSERT_FALSE(allocation.expired());
    const auto& survivor = models[1 - destroyed_first];
    EXPECT_EQ(GenerateSharedInitializerTestTokens(*survivor), expected);
    models[destroyed_first] = CreateSharedInitializerTestModel(true, cuda);
    EXPECT_EQ(models[destroyed_first]->shared_initializer_entries_[0].get(),
              survivor->shared_initializer_entries_[0].get());
    EXPECT_EQ(GenerateSharedInitializerTestTokens(*models[destroyed_first]), expected);
    models[0].reset();
    models[1].reset();
    EXPECT_TRUE(allocation.expired());
  }
}

TEST(SharedInitializers, AllocationOutlivesEitherSessionAndExpiresAfterBoth) {
  CheckSharedInitializerLifetime(false);
}

TEST(SharedInitializersGPU, AllocationOutlivesEitherCudaSessionAndExpiresAfterBoth) {
  if (!std::getenv("ORTGENAI_TEST_CUDA_SHARING"))
    GTEST_SKIP() << "Set ORTGENAI_TEST_CUDA_SHARING with an admitted CUDA device";
  CheckSharedInitializerLifetime(true);
}

}  // namespace
}  // namespace Generators::test
