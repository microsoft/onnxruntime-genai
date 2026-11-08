// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "models/model.h"

namespace Generators::test {
namespace {

struct GgufLoaderModel : Model {
  using Model::AddGgufInitializers;
  using Model::Model;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams&) const override {
    throw std::logic_error("The GGUF loader fixture does not implement generation.");
  }
};

class GgufInitializersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto candidate = std::filesystem::temp_directory_path() /
                           ("ortgenai_gguf_regressions_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(candidate));
    directory_ = candidate;
    for (const char* filename : {"model.onnx", "scalar.onnx"}) {
      std::filesystem::copy_file(std::filesystem::path{MODEL_PATH "gguf-initializers"} / filename,
                                 directory_ / filename);
    }
    WriteGguf("weights.gguf", 1.5f, {2.0f, 3.0f});
  }

  void TearDown() override {
    if (directory_.empty()) return;
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }

  void WriteFile(const std::string& filename, std::string_view bytes) const {
    auto output = fs::path{directory_.string()}.join(filename).open_for_write(std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
  }

  void WriteGguf(const std::string& filename, float weight, std::array<float, 2> other) const {
    std::string bytes{"GGUF"};
    auto append_integer = [&](auto value) {
      for (size_t i = 0; i < sizeof(value); ++i) {
        bytes.push_back(static_cast<char>(value & 0xff));
        value >>= 8;
      }
    };
    auto append_string = [&](std::string_view value) {
      append_integer(static_cast<uint64_t>(value.size()));
      bytes.append(value);
    };
    append_integer(uint32_t{3});
    append_integer(uint64_t{2});
    append_integer(uint64_t{0});
    for (const auto& [name, count, offset] :
         {std::tuple{"weight", uint64_t{1}, uint64_t{0}},
          std::tuple{"other_weight", uint64_t{2}, uint64_t{32}}}) {
      append_string(name);
      append_integer(uint32_t{1});
      append_integer(count);
      append_integer(uint32_t{0});
      append_integer(offset);
    }
    while (bytes.size() % 32 != 0) bytes.push_back('\0');
    append_integer(std::bit_cast<uint32_t>(weight));
    bytes.append(28, '\0');
    for (float value : other) append_integer(std::bit_cast<uint32_t>(value));
    WriteFile(filename, bytes);
  }

  void WriteConfig(std::string_view gguf, std::string_view filename = "model.onnx",
                   std::string_view draft = {}) const {
    std::string json = R"({"model":{"type":"gpt2","vocab_size":4,"context_length":32,"decoder":{"filename":")";
    json.append(filename);
    json += R"(","session_options":{"intra_op_num_threads":1},"gguf_initializers":)";
    json.append(gguf);
    json += "}";
    if (!draft.empty()) {
      json += R"(,"draft":{"filename":"draft.onnx","gguf_initializers":)";
      json.append(draft);
      json += "}";
    }
    json += R"(},"search":{}})";
    WriteFile("genai_config.json", json);
  }

  std::unique_ptr<Config> LoadConfig() const {
    return std::make_unique<Config>(fs::path{directory_.string()}, std::string_view{});
  }

  std::unique_ptr<GgufLoaderModel> LoadModel() const {
    return std::make_unique<GgufLoaderModel>(LoadConfig());
  }

  void ExpectModelError(std::string_view message) const {
    try {
      LoadModel();
      FAIL() << "Expected GGUF loading to fail";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string_view{error.what()}.find(message), std::string_view::npos) << error.what();
    }
  }

  void CheckOutputs(GgufLoaderModel& model, OrtSessionOptions& options,
                    const std::vector<int64_t>& shape = {1}, float weight = 1.5f) const {
    auto session = model.CreateSession(GetOrtEnv(), model.config_->model.decoder.filename, &options);
    const std::array names{"weight_out", "other_out"};
    std::array<OrtValue*, 2> outputs{};
    session->Run(nullptr, nullptr, nullptr, 0, names.data(), outputs.data(), outputs.size());
    std::unique_ptr<OrtValue> weight_output{outputs[0]};
    std::unique_ptr<OrtValue> other_output{outputs[1]};
    EXPECT_EQ(weight_output->GetTensorTypeAndShapeInfo()->GetShape(), shape);
    EXPECT_FLOAT_EQ(weight_output->GetTensorData<float>()[0], weight);
    EXPECT_EQ(other_output->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{2}));
    EXPECT_FLOAT_EQ(other_output->GetTensorData<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(other_output->GetTensorData<float>()[1], 3.0f);
    EXPECT_FALSE(std::filesystem::exists(directory_ / "missing.data"));
  }

  static constexpr std::string_view kDefaultMapping =
      R"({"files":["weights.gguf"],"tensors":{"weight":"weight","other_weight":"other_weight"}})";
  std::filesystem::path directory_;
};

TEST_F(GgufInitializersTest, DefaultShapesLoadWithoutExternalData) {
  WriteConfig(kDefaultMapping);
  auto model = LoadModel();
  EXPECT_FALSE(model->config_->model.decoder.gguf_initializers.tensors[0].has_shape);
  CheckOutputs(*model, *model->session_options_);
}

TEST_F(GgufInitializersTest, OverridesDoNotChangeDefaultSearchFiles) {
  WriteGguf("overrides.gguf", 4.0f, {5.0f, 6.0f});
  const std::string override_mapping =
      R"({"onnx_name":"weight","gguf_name":"weight","file":"overrides.gguf"})";
  const std::string default_mapping = R"({"onnx_name":"other_weight","gguf_name":"other_weight"})";
  for (bool override_first : {true, false}) {
    SCOPED_TRACE(override_first);
    WriteConfig(R"({"files":["weights.gguf"],"tensors":[)" +
                (override_first ? override_mapping + "," + default_mapping
                                : default_mapping + "," + override_mapping) +
                "]}");
    auto model = LoadModel();
    CheckOutputs(*model, *model->session_options_, {1}, 4.0f);
  }
}

TEST_F(GgufInitializersTest, FileAliasesAreNotAmbiguous) {
  WriteConfig(R"({"files":["weights.gguf","./weights.gguf","weights.gguf"],
                  "tensors":{"weight":"weight","other_weight":"other_weight"}})");
  auto model = LoadModel();
  CheckOutputs(*model, *model->session_options_);
}

TEST_F(GgufInitializersTest, DistinctDefaultFilesRemainAmbiguous) {
  WriteGguf("other.gguf", 4.0f, {5.0f, 6.0f});
  WriteConfig(R"({"files":["weights.gguf","other.gguf"],"tensors":{"weight":"weight"}})");
  ExpectModelError("Ambiguous GGUF tensor 'weight'");
}

TEST_F(GgufInitializersTest, OverridesDoNotProvideImplicitDefaults) {
  WriteConfig(R"({"tensors":[
    {"onnx_name":"weight","gguf_name":"weight","file":"weights.gguf"},
    {"onnx_name":"other_weight","gguf_name":"other_weight"}]})");
  ExpectModelError("GGUF tensor 'other_weight'");
}

TEST_F(GgufInitializersTest, ExplicitScalarShapeLoadsAsRankZero) {
  WriteConfig(R"({"files":["weights.gguf"],"tensors":{
    "weight":{"gguf_name":"weight","shape":[]},"other_weight":"other_weight"}})",
              "scalar.onnx");
  auto model = LoadModel();
  EXPECT_TRUE(model->config_->model.decoder.gguf_initializers.tensors[0].has_shape);
  CheckOutputs(*model, *model->session_options_, {});
}

TEST_F(GgufInitializersTest, ExternalMappingRetainsExplicitScalarShape) {
  WriteFile("mapping.json", R"({"files":["weights.gguf"],"tensors":{
    "weight":{"gguf_name":"weight","shape":[]},"other_weight":"other_weight"}})");
  WriteConfig(R"("mapping.json")", "scalar.onnx");
  auto model = LoadModel();
  CheckOutputs(*model, *model->session_options_, {});
}

TEST_F(GgufInitializersTest, ScalarOverrideRequiresExactlyOneElement) {
  WriteConfig(R"({"files":["weights.gguf"],"tensors":{
    "weight":"weight","other_weight":{"gguf_name":"other_weight","shape":[]}}})");
  ExpectModelError("Byte size mismatch");
}

TEST_F(GgufInitializersTest, ProgrammaticNonemptyShapeOverridesArePreserved) {
  WriteConfig(kDefaultMapping);
  auto config = LoadConfig();
  config->model.decoder.gguf_initializers.tensors[0].shape = {1, 1};
  GgufLoaderModel model{std::move(config)};
  ASSERT_EQ(model.gguf_external_values_.size(), 2u);
  EXPECT_EQ(model.gguf_external_values_[0]->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{1, 1}));
}

TEST_F(GgufInitializersTest, AlternateDecoderOptionsRegisterIndependentMappings) {
  WriteConfig(kDefaultMapping);
  auto model = LoadModel();
  for (size_t i = 0; i < 2; ++i) {
    auto options = OrtSessionOptions::Create();
    model->CreateSessionOptionsFromConfig(model->config_->model.decoder.session_options, *options, true);
    model->AddGgufInitializers(*options);
    CheckOutputs(*model, *options);
  }
  CheckOutputs(*model, *model->session_options_);
}

TEST_F(GgufInitializersTest, DraftGgufPathsAreValidated) {
  for (std::string_view draft : {
           R"("../outside.json")",
           R"({"config_file":"../outside.json"})",
           R"({"files":["../outside.gguf"]})",
           R"({"tensors":{"weight":{"gguf_name":"weight","file":"../outside.gguf"}}})",
       }) {
    SCOPED_TRACE(draft);
    WriteConfig("{}", "model.onnx", draft);
    try {
      LoadConfig();
      FAIL() << "Expected draft GGUF path validation to fail";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string_view{error.what()}.find("model.draft.gguf_initializers"),
                std::string_view::npos)
          << error.what();
    }
  }
  WriteConfig("{}", "model.onnx", kDefaultMapping);
  EXPECT_NO_THROW(LoadConfig());
}

TEST_F(GgufInitializersTest, Utf8GgufPathsLoad) {
  const std::string filename = "weights_\xe4\xb8\xad.gguf";
  WriteGguf(filename, 1.5f, {2.0f, 3.0f});
  WriteConfig(R"({"files":[")" + filename +
              R"("],"tensors":{"weight":"weight","other_weight":"other_weight"}})");
  auto model = LoadModel();
  CheckOutputs(*model, *model->session_options_);
}

TEST_F(GgufInitializersTest, PipelinedDecodersRejectMappingsExplicitly) {
  WriteConfig(kDefaultMapping);
  auto config = LoadConfig();
  config->model.decoder.pipeline.emplace_back();
  try {
    GgufLoaderModel model{std::move(config)};
    FAIL() << "Expected pipelined decoder GGUF mappings to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string_view{error.what()}.find("pipelined decoders"), std::string_view::npos) << error.what();
  }
}

TEST_F(GgufInitializersTest, InvalidHeaderDoesNotPoisonFileCache) {
  WriteFile("weights.gguf", "bad");
  WriteConfig(kDefaultMapping);
  ExpectModelError("Truncated GGUF header");
  WriteGguf("weights.gguf", 1.5f, {2.0f, 3.0f});
  auto model = LoadModel();
  CheckOutputs(*model, *model->session_options_);
}

}  // namespace
}  // namespace Generators::test
