// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <gtest/gtest.h>

#include "models/model.h"
#include "models/session_options.h"

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

#if USE_CUDA
// Forwards the calls that initializer loading makes to a real device and counts its allocations. A `device_id` makes it
// report another device identity, which the shared-initializer cache must not match with the real device's allocations.
struct ForwardingDevice final : DeviceInterface {
  explicit ForwardingDevice(DeviceInterface& device, std::optional<int> device_id = {})
      : device_{device}, device_id_{device_id} {}

  DeviceType GetType() const override { return device_.GetType(); }
  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override { device_.InitOrt(api, allocator); }
  Ort::Allocator& GetAllocator() override { return device_.GetAllocator(); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    auto info = device_.GetMemoryInfo();
    if (!device_id_) return info;
    return OrtMemoryInfo::Create(info->GetAllocatorName().c_str(), info->GetAllocatorType(), *device_id_,
                                 info->GetMemoryType());
  }
  std::string GetExecutionProviderName() const override { return device_.GetExecutionProviderName(); }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    ++allocations;
    return device_.AllocateBase(size);
  }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override { return device_.WrapMemoryBase(p, size); }
  // Loading initializers never creates searches or caches.
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams&) override { throw Unexpected(); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams&) override { throw Unexpected(); }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State&) override { throw Unexpected(); }
  void Synchronize() override { device_.Synchronize(); }

  size_t allocations{};

 private:
  static std::logic_error Unexpected() { return std::logic_error{"Unexpected forwarding test device call."}; }

  DeviceInterface& device_;
  std::optional<int> device_id_;
};
#endif

// Rewrites bytes of a file through a writable shared mapping, which every other mapping of the file observes.
void OverwriteMappedBytes(const std::filesystem::path& path, size_t offset, const void* data, size_t size) {
#ifdef _WIN32
  HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  ASSERT_NE(file, INVALID_HANDLE_VALUE) << ::GetLastError();
  HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
  ::CloseHandle(file);
  ASSERT_NE(mapping, nullptr) << ::GetLastError();
  void* view = ::MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, 0);
  ::CloseHandle(mapping);
  ASSERT_NE(view, nullptr) << ::GetLastError();
  std::memcpy(static_cast<uint8_t*>(view) + offset, data, size);
  ::UnmapViewOfFile(view);
#else
  const int fd = ::open(path.c_str(), O_RDWR);
  ASSERT_GE(fd, 0) << path;
  struct stat file_info = {};
  ASSERT_EQ(::fstat(fd, &file_info), 0) << path;
  const auto file_size = static_cast<size_t>(file_info.st_size);
  void* view = ::mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  ASSERT_NE(view, MAP_FAILED) << path;
  std::memcpy(static_cast<uint8_t*>(view) + offset, data, size);
  ::munmap(view, file_size);
#endif
}

struct F32Tensor {
  std::string name;
  std::vector<float> values;
};

// A GGUF tensor with GGML dimensions (innermost first) and its raw data bytes.
struct GgufTensor {
  std::string name;
  uint32_t ggml_type{};
  std::vector<uint64_t> dims;
  std::string data;
};

constexpr uint32_t kGgmlF32 = 0;
constexpr uint32_t kUnsupportedGgmlType = 43;  // Not in the loader's GGML type table.

GgufTensor F32GgufTensor(const F32Tensor& tensor) {
  std::string data;
  for (float value : tensor.values) {
    uint32_t bits = FloatBits(value);
    for (size_t i = 0; i < sizeof(bits); ++i, bits >>= 8) data.push_back(static_cast<char>(bits & 0xff));
  }
  return {tensor.name, kGgmlF32, {tensor.values.size()}, std::move(data)};
}

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

  // Serializes a GGUF v3 file whose tensor data each start on a 32-byte boundary, and stores the file offset of each
  // tensor's data in `data_offsets` when given.
  static std::string SerializeGguf(const std::vector<GgufTensor>& tensors,
                                   std::vector<size_t>* data_offsets = nullptr) {
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
      append_integer(static_cast<uint32_t>(tensor.dims.size()));
      for (uint64_t dim : tensor.dims) append_integer(dim);
      append_integer(tensor.ggml_type);
      append_integer(offset);
      offset += (tensor.data.size() + 31) / 32 * 32;
    }
    for (const auto& tensor : tensors) {
      while (bytes.size() % 32 != 0) bytes.push_back('\0');
      if (data_offsets) data_offsets->push_back(bytes.size());
      bytes.append(tensor.data);
    }
    return bytes;
  }

  // Writes one-dimensional F32 tensors and returns the file offset of each tensor's data.
  std::vector<size_t> WriteF32Gguf(const std::string& filename, const std::vector<F32Tensor>& tensors) const {
    std::vector<GgufTensor> gguf_tensors;
    for (const auto& tensor : tensors) gguf_tensors.push_back(F32GgufTensor(tensor));
    std::vector<size_t> data_offsets;
    WriteFile(filename, SerializeGguf(gguf_tensors, &data_offsets));
    return data_offsets;
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

  // Loads the model with `provider` as its only execution provider.
  std::unique_ptr<GgufLoaderModel> LoadModel(std::string_view provider, std::string_view option = {},
                                             std::string_view value = {}) const {
    auto config = LoadConfig();
    ClearProviders(*config);
    SetProviderOption(*config, provider, option, value);
    return std::make_unique<GgufLoaderModel>(std::move(config));
  }

  // The shared cache keeps a tensor's device buffer alive for as long as any model uses the weights, so the upload must
  // not leave a host staging copy in it. Also checks that the upload produced the GGUF tensor bytes.
  static void ExpectUploadedWithoutHostCopy(DeviceSpan<uint8_t> data, const std::vector<float>& weight) {
    const auto& buffer = data.GetBuffer();
    EXPECT_TRUE(buffer.p_cpu_ == nullptr || buffer.p_cpu_ == buffer.p_device_);
    ASSERT_EQ(data.size(), weight.size() * sizeof(float));
    const auto bytes = data.CopyDeviceToCpu();
    EXPECT_EQ(std::memcmp(bytes.data(), weight.data(), bytes.size()), 0);
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

  // Runs large.onnx (y = x + large_weight) with x = 1 and returns y.
  static std::vector<float> RunLargeModel(OrtSession& session, size_t count) {
    std::vector<float> input(count, 1.0f);
    const std::vector<int64_t> shape{static_cast<int64_t>(input.size())};
    auto memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto x = OrtValue::CreateTensor(*memory_info, input.data(), input.size() * sizeof(float), shape,
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const char* input_name = "x";
    const char* output_name = "y";
    const OrtValue* inputs[] = {x.get()};
    OrtValue* output = nullptr;
    session.Run(nullptr, &input_name, inputs, 1, &output_name, &output, 1);
    std::unique_ptr<OrtValue> y{output};
    const auto y_info = y->GetTensorTypeAndShapeInfo();
    EXPECT_EQ(y_info->GetShape(), shape);
    const float* values = y->GetTensorData<float>();
    return std::vector<float>(values, values + y_info->GetElementCount());
  }

  // Checks large.onnx outputs against the GGUF tensor values.
  static void CheckLargeOutput(OrtSession& session, const std::vector<float>& weight) {
    const auto values = RunLargeModel(session, weight.size());
    ASSERT_EQ(values.size(), weight.size());
    for (size_t i = 0; i < weight.size(); ++i) {
      EXPECT_FLOAT_EQ(values[i], weight[i] + 1.0f) << i;
    }
  }

  // Whether this ONNX Runtime reads AddExternalInitializers buffers in place when
  // session.use_external_initializer_file_buffers_directly is set (microsoft/onnxruntime#33877), rather than copying
  // them during session creation. Changing a borrowed buffer after session creation changes the next Run's output.
  bool RuntimeBorrowsExternalInitializers() const {
    std::vector<float> weight(64, 1.0f);
    const std::vector<int64_t> shape{static_cast<int64_t>(weight.size())};
    auto memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<std::unique_ptr<OrtValue>> values;
    values.push_back(OrtValue::CreateTensor(*memory_info, weight.data(), weight.size() * sizeof(float), shape,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    auto options = OrtSessionOptions::Create();
    options->SetIntraOpNumThreads(1);
    options->AddConfigEntry("session.use_external_initializer_file_buffers_directly", "1");
    options->AddExternalInitializers({"large_weight"}, values);
    const auto model_path = directory_ / "large.onnx";
    auto session = OrtSession::Create(GetOrtEnv(), model_path.c_str(), options.get());
    std::fill(weight.begin(), weight.end(), 2.0f);
    const auto y = RunLargeModel(*session, weight.size());
    return !y.empty() && y[0] == 3.0f;
  }

  static constexpr std::string_view kDefaultMapping =
      R"({"files":["weights.gguf"],"tensors":{"weight":"weight","other_weight":"other_weight"}})";
  std::filesystem::path directory_;
};

TEST_F(GgufInitializersTest, DefaultShapesLoadWithoutExternalData) {
  WriteConfig(kDefaultMapping);
  auto model = LoadModel();
  EXPECT_FALSE(model->config_->model.decoder.gguf_initializers.tensors[0].has_shape);
  ASSERT_EQ(model->gguf_external_values_.size(), 2u);
  for (const auto& value : model->gguf_external_values_) {
    EXPECT_EQ(value->GetTensorMemoryInfo().GetMemoryType(), OrtMemTypeDefault);
    EXPECT_EQ(value->GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_CPU);
  }
  CheckOutputs(*model, *model->session_options_);
}

TEST_F(GgufInitializersTest, OverridesDoNotChangeDefaultSearchFiles) {
  WriteGguf("overrides.gguf", 4.0f, {5.0f, 6.0f});
  const std::string override_mapping = R"("weight":{"gguf_name":"weight","file":"overrides.gguf"})";
  const std::string default_mapping = R"("other_weight":"other_weight")";
  for (bool override_first : {true, false}) {
    SCOPED_TRACE(override_first);
    WriteConfig(R"({"files":["weights.gguf"],"tensors":{)" +
                (override_first ? override_mapping + "," + default_mapping
                                : default_mapping + "," + override_mapping) +
                "}}");
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
  WriteConfig(R"({"tensors":{
    "weight":{"gguf_name":"weight","file":"weights.gguf"},
    "other_weight":"other_weight"}})");
  ExpectModelError("GGUF tensor 'other_weight'");
}

TEST_F(GgufInitializersTest, MappingFieldsHaveOneSpelling) {
  for (std::string_view mapping : {
           R"({"file":"weights.gguf","tensors":{"weight":"weight"}})",
           R"({"files":["weights.gguf"],"tensors":[{"onnx_name":"weight","gguf_name":"weight"}]})",
           R"({"files":["weights.gguf"],"tensors":{"weight":{"onnx_name":"weight","gguf_name":"weight"}}})",
           R"({"files":["weights.gguf"],"tensors":{"weight":{"tensor":"weight"}}})",
           R"({"files":["weights.gguf"],"tensors":{"weight":{"gguf_name":"weight","data_file":"weights.gguf"}}})",
           R"({"files":["weights.gguf"],"tensors":{"weight":{"gguf_name":"weight","dtype":1}}})",
       }) {
    SCOPED_TRACE(mapping);
    WriteConfig(mapping);
    try {
      LoadConfig();
      FAIL() << "Expected an alternative GGUF mapping spelling to be rejected";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string_view{error.what()}.find("Unknown value"), std::string_view::npos) << error.what();
    }
  }
}

TEST_F(GgufInitializersTest, OnlySelectedTensorsAreValidated) {
  // Besides the mapped tensors, the file holds a tensor of an unsupported type and one whose data the file cuts short.
  auto bytes = SerializeGguf({F32GgufTensor({"weight", {1.5f}}), F32GgufTensor({"other_weight", {2.0f, 3.0f}}), {"unsupported", kUnsupportedGgmlType, {32}, std::string(32, '\x01')}, F32GgufTensor({"truncated", std::vector<float>(16, 1.0f)})});
  bytes.resize(bytes.size() - 32);
  WriteFile("weights.gguf", bytes);
  WriteConfig(kDefaultMapping);
  {
    auto model = LoadModel();
    CheckOutputs(*model, *model->session_options_);
  }

  WriteConfig(R"({"files":["weights.gguf"],"tensors":{"weight":"unsupported"}})");
  ExpectModelError("Unsupported GGUF tensor type 43 for tensor 'unsupported'");
  WriteConfig(R"({"files":["weights.gguf"],"tensors":{"weight":"truncated"}})");
  ExpectModelError("GGUF tensor 'truncated' exceeds file bounds");
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

// A session that borrows the mapped GGUF data reads it during Run, so bytes rewritten after session creation change the
// output. A session holding a private copy keeps returning the original values, as it would if the direct-buffer option
// were not set.
TEST_F(GgufInitializersTest, SessionsReadMappedTensorsWithoutCopying) {
  if (!RuntimeBorrowsExternalInitializers()) {
    GTEST_SKIP() << "This ONNX Runtime copies AddExternalInitializers buffers; borrowing requires "
                    "microsoft/onnxruntime#33877.";
  }
  std::vector<float> weight(64);
  for (size_t i = 0; i < weight.size(); ++i) weight[i] = 0.25f * static_cast<float>(i) - 4.0f;
  const auto data_offsets = WriteF32Gguf("large.gguf", {{"large_weight", weight}});
  WriteConfig(R"({"files":["large.gguf"],"tensors":{"large_weight":"large_weight"}})", "large.onnx");
  auto model = LoadModel();
  auto session = model->CreateSession(GetOrtEnv(), model->config_->model.decoder.filename,
                                      model->session_options_.get());
  CheckLargeOutput(*session, weight);

  for (float& value : weight) value += 100.0f;
  ASSERT_NO_FATAL_FAILURE(OverwriteMappedBytes(directory_ / "large.gguf", data_offsets[0], weight.data(),
                                               weight.size() * sizeof(float)));
  CheckLargeOutput(*session, weight);
}

#if USE_CUDA
TEST_F(GgufInitializersTest, CudaSharedTensorsUseOneDeviceAllocation) {
  if (!GetDeviceInterface(DeviceType::CUDA)) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  std::vector<float> weight(64);
  for (size_t i = 0; i < weight.size(); ++i) weight[i] = 0.5f * static_cast<float>(i) - 8.0f;
  WriteF32Gguf("large.gguf", {{"large_weight", weight}});
  WriteConfig(R"({"files":["large.gguf"],"share":true,"tensors":{"large_weight":"large_weight"}})", "large.onnx");
  auto create_session = [](GgufLoaderModel& model) {
    return model.CreateSession(GetOrtEnv(), model.config_->model.decoder.filename, model.session_options_.get());
  };

  // A target and a drafter mapping the same tensor bind one device allocation, which keeps no host staging copy.
  auto first = LoadModel("cuda", "enable_cuda_graph", "0");
  auto second = LoadModel("cuda", "enable_cuda_graph", "0");
  ASSERT_EQ(first->p_device_->GetType(), DeviceType::CUDA);
  ASSERT_EQ(first->gguf_shared_device_data_.size(), 1u);
  ASSERT_EQ(second->gguf_shared_device_data_.size(), 1u);
  const DeviceSpan<uint8_t> shared = first->gguf_shared_device_data_[0];
  EXPECT_TRUE(shared.SameBufferAs(second->gguf_shared_device_data_[0]));
  ExpectUploadedWithoutHostCopy(shared, weight);

  auto first_session = create_session(*first);
  auto second_session = create_session(*second);
  CheckLargeOutput(*first_session, weight);
  CheckLargeOutput(*second_session, weight);

  // The surviving model keeps the shared allocation valid after the other model and its session are released.
  first_session.reset();
  first.reset();
  CheckLargeOutput(*second_session, weight);

  // Registers the mapping again through `device` and returns the allocation it bound.
  DeviceInterface* const cuda = second->p_device_;
  auto register_through = [&](ForwardingDevice& device) {
    auto options = OrtSessionOptions::Create();
    second->p_device_ = &device;
    try {
      second->AddGgufInitializers(*options);
    } catch (...) {
      second->p_device_ = cuda;
      throw;
    }
    second->p_device_ = cuda;
    return second->gguf_shared_device_data_.back();
  };
  auto& mapping = second->config_->model.decoder.gguf_initializers.tensors[0];

  ForwardingDevice same_device{*cuda};
  EXPECT_TRUE(register_through(same_device).SameBufferAs(shared));
  EXPECT_EQ(same_device.allocations, 0u);

  // Other layouts of the same bytes, and other device identities, get their own allocations.
  mapping.has_shape = true;
  mapping.shape = {8, 8};
  EXPECT_FALSE(register_through(same_device).SameBufferAs(shared));
  mapping.shape = {64};
  mapping.data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  EXPECT_FALSE(register_through(same_device).SameBufferAs(shared));
  EXPECT_EQ(same_device.allocations, 2u);

  mapping.has_shape = false;
  mapping.shape.clear();
  mapping.data_type = 0;
  ForwardingDevice other_device{*cuda, cuda->GetMemoryInfo()->GetDeviceId() + 1};
  EXPECT_FALSE(register_through(other_device).SameBufferAs(shared));
  EXPECT_EQ(other_device.allocations, 1u);

  CheckLargeOutput(*second_session, weight);
}
#endif

TEST_F(GgufInitializersTest, WebGpuSharedUploadsKeepNoHostCopy) {
  GetOrtEnv();
  const auto providers = Ort::GetAvailableProviders();
  if (std::find(providers.begin(), providers.end(), "WebGpuExecutionProvider") == providers.end() &&
      FindRegisteredEpDevices("WebGpuExecutionProvider").empty()) {
    GTEST_SKIP() << "No native or registered WebGPU EP is available.";
  }
  std::vector<float> weight(64);
  for (size_t i = 0; i < weight.size(); ++i) weight[i] = 0.5f * static_cast<float>(i) - 8.0f;
  WriteF32Gguf("large.gguf", {{"large_weight", weight}});
  WriteConfig(R"({"files":["large.gguf"],"share":true,"tensors":{"large_weight":"large_weight"}})", "large.onnx");
  auto model = LoadModel("webgpu");
  ASSERT_EQ(model->gguf_shared_device_data_.size(), 1u);
  ExpectUploadedWithoutHostCopy(model->gguf_shared_device_data_[0], weight);
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
