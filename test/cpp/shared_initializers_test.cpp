// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// shared_initializers: sessions naming the same file range read one device copy of it, each through a
// view with the shape its own graph declares (e.g. a multimodal embedding table tied to the LM head).

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "models/model.h"

namespace {

struct SharedInitializerTestModel : Generators::Model {
  explicit SharedInitializerTestModel(std::unique_ptr<Generators::Config> config) : Model{std::move(config)} {}

  std::unique_ptr<Generators::State> CreateState(
      Generators::DeviceSpan<int32_t>, const Generators::GeneratorParams&) const override {
    return nullptr;
  }

  using Model::AddSharedInitializers;
};

std::unique_ptr<Generators::Config> ConfigFor(const char* provider) {
  auto config = std::make_unique<Generators::Config>();
  if (provider) {
    config->model.decoder.session_options.provider_options.push_back({provider, {}, {}});
    config->model.decoder.session_options.providers.push_back(provider);
  }
  return config;
}

// Session options with the model's providers, as the model's own sessions get them.
std::unique_ptr<OrtSessionOptions> OptionsFor(Generators::Model& model) {
  auto options = OrtSessionOptions::Create();
  model.CreateSessionOptionsFromConfig(model.config_->model.decoder.session_options, *options, true);
  return options;
}

bool HasWebGpu() {
  Generators::GetOrtEnv();
  const auto providers = Ort::GetAvailableProviders();
  return std::find(providers.begin(), providers.end(), "WebGpuExecutionProvider") != providers.end();
}

// rows.onnx and flat.onnx output their only initializer times input x ("table" float [2, 3] and "head"
// float [6], both stored as zeros in the graph).
std::vector<float> RunGraph(Generators::Model& model, const char* file, OrtSessionOptions& options,
                            const std::vector<int64_t>& expected_shape) {
  auto session = model.CreateSession(Generators::GetOrtEnv(), std::string{MODEL_PATH "shared-initializers/"} + file,
                                     &options);
  float one = 1.0f;
  const int64_t one_shape = 1;
  auto memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  auto x = OrtValue::CreateTensor<float>(*memory_info, std::span<float>{&one, 1}, std::span<const int64_t>{&one_shape, 1});
  const char* input_name = "x";
  const OrtValue* inputs[] = {x.get()};
  const char* output_name = "out";
  auto outputs = session->Run(nullptr, &input_name, inputs, 1, &output_name, 1);
  auto info = outputs[0]->GetTensorTypeAndShapeInfo();
  EXPECT_EQ(info->GetShape(), expected_shape);
  const float* data = outputs[0]->GetTensorData<float>();
  return {data, data + info->GetElementCount()};
}

struct ScopedWeights {
  ScopedWeights()
      : dir{std::filesystem::temp_directory_path() /
            ("ortgenai_shared_initializers_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))} {
    std::filesystem::create_directories(dir);
    std::vector<float> values(8);
    std::iota(values.begin(), values.end(), 0.0f);
    std::ofstream out(dir / "weights.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
  }
  ~ScopedWeights() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
  // values[2..7] of weights.bin (0, 1, ..., 7), viewed as [2, 3] by "table" and [6] by "head".
  Generators::Config::Model::SharedInitializer Range(const char* name, std::vector<int64_t> shape,
                                                     const char* offset = "8") const {
    return {name, (dir / "weights.bin").string(), offset, "24", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::move(shape)};
  }
  std::filesystem::path dir;
};

// Two sessions read one range as [2, 3] and [6]: one copy on the model's device, a view per session, and both graphs
// see the file's bytes. A different range gets its own copy.
void ExpectOneCopyViewedWithEachShape(SharedInitializerTestModel& model, const ScopedWeights& weights) {
  auto table_options = OptionsFor(model);
  auto head_options = OptionsFor(model);
  model.AddSharedInitializers({weights.Range("table", {2, 3})}, *table_options, *model.p_device_);
  model.AddSharedInitializers({weights.Range("head", {6})}, *head_options, *model.p_device_);

  // One buffer (the entry of each call) and a view per session.
  ASSERT_EQ(model.shared_initializer_entries_.size(), 4u);
  EXPECT_EQ(model.shared_initializer_entries_[0], model.shared_initializer_entries_[2]);
  EXPECT_NE(model.shared_initializer_entries_[1], model.shared_initializer_entries_[3]);

  const std::vector<float> expected{2, 3, 4, 5, 6, 7};
  EXPECT_EQ(RunGraph(model, "rows.onnx", *table_options, {2, 3}), expected);
  EXPECT_EQ(RunGraph(model, "flat.onnx", *head_options, {6}), expected);

  auto other_options = OptionsFor(model);
  model.AddSharedInitializers({weights.Range("head", {6}, "4")}, *other_options, *model.p_device_);
  ASSERT_EQ(model.shared_initializer_entries_.size(), 6u);
  EXPECT_NE(model.shared_initializer_entries_[4], model.shared_initializer_entries_[0]);
  EXPECT_EQ(RunGraph(model, "flat.onnx", *other_options, {6}), (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(SharedInitializerTests, CpuSessionsViewOneCopyOfAFileRangeWithTheirOwnShapes) {
  ScopedWeights weights;
  SharedInitializerTestModel model{ConfigFor(nullptr)};
  ASSERT_EQ(model.p_device_->GetType(), Generators::DeviceType::CPU);
  ExpectOneCopyViewedWithEachShape(model, weights);
}

TEST(SharedInitializerTests, WebGpuSessionsViewOneCopyOfAFileRangeWithTheirOwnShapes) {
  if (!HasWebGpu()) {
    GTEST_SKIP() << "WebGPU is not available.";
  }
  ScopedWeights weights;
  SharedInitializerTestModel model{ConfigFor("WebGPU")};
  ASSERT_EQ(model.p_device_->GetType(), Generators::DeviceType::WEBGPU);
  ExpectOneCopyViewedWithEachShape(model, weights);
}

TEST(SharedInitializerTests, EmbeddingAcceptsSharedInitializers) {
  const auto dir = std::filesystem::temp_directory_path() / "ortgenai_embedding_shared_initializers_config";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "genai_config.json")
      << R"({"model": {"type": "phi3v", "vocab_size": 16, "context_length": 32,
             "decoder": {"filename": "decoder.onnx"},
             "embedding": {"filename": "embedding.onnx", "shared_initializers": [
               {"name": "table", "data_file": "decoder.onnx.data", "offset": "64", "length": "1024",
                "data_type": 2, "shape": [16, 64]}]}},
           "search": {}})";
  const Generators::Config config{fs::path{dir.string()}, {}};
  ASSERT_EQ(config.model.embedding.shared_initializers.size(), 1u);
  const auto& table = config.model.embedding.shared_initializers[0];
  EXPECT_EQ(table.name, "table");
  EXPECT_EQ(table.offset, "64");
  EXPECT_EQ(table.length, "1024");
  EXPECT_EQ(table.data_type, 2);
  EXPECT_EQ(table.shape, (std::vector<int64_t>{16, 64}));
  std::filesystem::remove_all(dir);
}

}  // namespace
