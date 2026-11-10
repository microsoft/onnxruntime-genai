// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "models/model.h"

namespace Generators::test {
namespace {

// std::bit_cast requires C++20, which the Android and GCC < 10 test configurations do not use.
uint32_t FloatBits(float value) {
  static_assert(sizeof(uint32_t) == sizeof(float));
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct GgufLoaderModel : Model {
  using Model::AddGgufInitializers;
  using Model::Model;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams&) const override {
    throw std::logic_error("The GGUF loader fixture does not implement generation.");
  }
};

// Reports DML to device-type checks; any other device call means the DML rejection happened too late.
struct DmlTypeOnlyDevice final : DeviceInterface {
  DeviceType GetType() const override { return DeviceType::DML; }
  void InitOrt(const OrtApi&, Ort::Allocator&) override { throw Unexpected(); }
  Ort::Allocator& GetAllocator() override { throw Unexpected(); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override { throw Unexpected(); }
  std::string GetExecutionProviderName() const override { throw Unexpected(); }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t) override { throw Unexpected(); }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void*, size_t) override { throw Unexpected(); }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams&) override { throw Unexpected(); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams&) override { throw Unexpected(); }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State&) override { throw Unexpected(); }
  void Synchronize() override { throw Unexpected(); }

 private:
  static std::logic_error Unexpected() { return std::logic_error{"Unexpected DML test device call."}; }
};

struct F32Tensor {
  std::string name;
  std::vector<float> values;
};

class GgufInitializersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto candidate = std::filesystem::temp_directory_path() /
                           ("ortgenai_gguf_regressions_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(candidate));
    directory_ = candidate;
    for (const char* filename : {"model.onnx", "scalar.onnx", "large.onnx"}) {
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

  // Writes a GGUF v3 file of one-dimensional F32 tensors, each starting on a 32-byte boundary.
  void WriteF32Gguf(const std::string& filename, const std::vector<F32Tensor>& tensors) const {
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
    append_integer(static_cast<uint64_t>(tensors.size()));
    append_integer(uint64_t{0});
    uint64_t offset = 0;
    for (const auto& tensor : tensors) {
      append_string(tensor.name);
      append_integer(uint32_t{1});
      append_integer(static_cast<uint64_t>(tensor.values.size()));
      append_integer(uint32_t{0});
      append_integer(offset);
      offset += (tensor.values.size() * sizeof(float) + 31) / 32 * 32;
    }
    for (const auto& tensor : tensors) {
      while (bytes.size() % 32 != 0) bytes.push_back('\0');
      for (float value : tensor.values) append_integer(FloatBits(value));
    }
    WriteFile(filename, bytes);
  }

  void WriteGguf(const std::string& filename, float weight, std::array<float, 2> other) const {
    WriteF32Gguf(filename, {{"weight", {weight}}, {"other_weight", {other[0], other[1]}}});
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

  // Runs large.onnx (y = x + large_weight) with x = 1 and checks y against the GGUF tensor values.
  static void CheckLargeOutput(OrtSession& session, const std::vector<float>& weight) {
    std::vector<float> input(weight.size(), 1.0f);
    const std::vector<int64_t> shape{static_cast<int64_t>(input.size())};
    auto memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto x = OrtValue::CreateTensor(*memory_info, input.data(), input.size() * sizeof(float), shape,
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const char* input_name = "x";
    const char* output_name = "y";
    const OrtValue* inputs[] = {x.get()};
    OrtValue* output = nullptr;
    session.Run(nullptr, &input_name, inputs, 1, &output_name, &output, 1);
    std::unique_ptr<OrtValue> y{output};
    ASSERT_EQ(y->GetTensorTypeAndShapeInfo()->GetShape(), shape);
    const float* values = y->GetTensorData<float>();
    for (size_t i = 0; i < weight.size(); ++i) {
      EXPECT_FLOAT_EQ(values[i], weight[i] + 1.0f) << i;
    }
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

TEST_F(GgufInitializersTest, ExternalMappingRejectsNestedConfigFile) {
  WriteFile("nested.json", kDefaultMapping);
  WriteFile("mapping.json", R"({"config_file":"nested.json"})");
  WriteConfig(R"("mapping.json")");
  ExpectModelError("nested mapping files are not supported");
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

TEST_F(GgufInitializersTest, StreamingEncoderDecoderAsrRejectsMappingsExplicitly) {
  WriteConfig(kDefaultMapping);
  auto config = LoadConfig();
  config->model.type = "streaming_enc_dec_asr";
  try {
    GgufLoaderModel model{std::move(config)};
    FAIL() << "Expected streaming_enc_dec_asr GGUF mappings to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string_view{error.what()}.find("streaming_enc_dec_asr"), std::string_view::npos) << error.what();
  }
}

TEST_F(GgufInitializersTest, DmlRejectsSharedTensors) {
  WriteConfig(R"({"files":["weights.gguf"],"share":true,"tensors":{"weight":"weight","other_weight":"other_weight"}})");
  auto model = LoadModel();
  DmlTypeOnlyDevice dml;
  // Restores the CPU device before the model is destroyed, even when an assertion ends the test early.
  struct RestoreDevice {
    Model* model;
    DeviceInterface* device;
    ~RestoreDevice() { model->p_device_ = device; }
  } restore{model.get(), std::exchange(model->p_device_, &dml)};
  auto options = OrtSessionOptions::Create();
  try {
    model->AddGgufInitializers(*options);
    FAIL() << "Expected GGUF tensor sharing on DML to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string_view{error.what()}.find("not supported on DML"), std::string_view::npos) << error.what();
  }
}

TEST_F(GgufInitializersTest, LargeInitializerMappingOutlivesReleasedModels) {
  // 256 naturally aligned bytes: large enough for ONNX Runtime to borrow the mapped data instead of copying it.
  std::vector<float> weight(64);
  for (size_t i = 0; i < weight.size(); ++i) weight[i] = 0.25f * static_cast<float>(i) - 4.0f;
  WriteF32Gguf("large.gguf", {{"large_weight", weight}});
  WriteConfig(R"({"files":["large.gguf"],"tensors":{"large_weight":"large_weight"}})", "large.onnx");
  auto create_session = [](GgufLoaderModel& model) {
    return model.CreateSession(GetOrtEnv(), model.config_->model.decoder.filename, model.session_options_.get());
  };

  auto first = LoadModel();
  auto second = LoadModel();
  ASSERT_EQ(first->gguf_external_values_.size(), 1u);
  ASSERT_EQ(second->gguf_external_values_.size(), 1u);
  const void* data = first->gguf_external_values_[0]->GetTensorRawData();
  EXPECT_EQ(reinterpret_cast<uintptr_t>(data) % sizeof(float), 0u);
  EXPECT_EQ(second->gguf_external_values_[0]->GetTensorRawData(), data);
  auto first_session = create_session(*first);
  auto second_session = create_session(*second);
  CheckLargeOutput(*first_session, weight);
  CheckLargeOutput(*second_session, weight);

  // The cached mapping must stay valid for live models and sessions after another model using it is released.
  first_session.reset();
  first.reset();
  const float* mapped = second->gguf_external_values_[0]->GetTensorData<float>();
  EXPECT_EQ(std::vector<float>(mapped, mapped + weight.size()), weight);
  CheckLargeOutput(*second_session, weight);

  auto third = LoadModel();
  ASSERT_EQ(third->gguf_external_values_.size(), 1u);
  EXPECT_EQ(third->gguf_external_values_[0]->GetTensorRawData(), data);
  auto third_session = create_session(*third);
  second_session.reset();
  second.reset();
  CheckLargeOutput(*third_session, weight);
}

TEST_F(GgufInitializersTest, EmptyFilesAreRejected) {
  WriteFile("weights.gguf", "");
  WriteConfig(kDefaultMapping);
  ExpectModelError("Empty GGUF file");
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
