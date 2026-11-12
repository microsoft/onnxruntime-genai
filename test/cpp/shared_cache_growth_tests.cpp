// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <fstream>
#include <future>
#include <numeric>
#include <type_traits>
#include "models/graph_builder.h"
#include "models/io/shared_kv_cache.h"
#include "models/decoder_only.h"
#include "ep/webgpu/kv_cache.h"

namespace Generators {
namespace {

using WebGPU::SharedCacheGrowth;

bool GrowthAvailable(bool capture) {
#if ORT_API_VERSION < 27
  if (capture) return false;
#endif
  return Ort::api_version >= 22 && (!capture || Ort::api_version >= 27) && Ort::api->GetModelEditorApi() != nullptr;
}

class WebGpuGrowthTest : public ::testing::Test {
  void SetUp() override {
    if (!GrowthAvailable(false)) GTEST_SKIP() << "KV growth requires ORT model-editor support (API 22+).";
  }
};

std::unique_ptr<Config> MakeConfig(bool capture = true) {
  auto config = std::make_unique<Config>();
  config->model.type = "qwen3";
  config->model.context_length = 4096;
  config->model.vocab_size = 32;
  config->model.decoder.num_hidden_layers = 1;
  config->model.decoder.num_attention_heads = 3;
  config->model.decoder.num_key_value_heads = 3;
  config->model.decoder.head_size = 4;
  config->model.decoder.hidden_size = 12;
  config->model.decoder.session_options.providers = {"WebGPU"};
  config->model.decoder.session_options.provider_options.push_back(
      {"WebGPU", {{"enableGraphCapture", capture ? "1" : "0"}, {"validationMode", "basic"}}});
  config->search.max_length = 4093;
  config->search.batch_size = 2;
  config->search.past_present_share_buffer = true;
  return config;
}

struct TestModel : Model {
  TestModel(int64_t fixed_capacity = -1, const char* op = "Identity",
            ONNXTensorElementDataType cache_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) : Model{MakeConfig(GrowthAvailable(true))} {
    for (const char* suffix : {"key", "value"}) {
      ModelConfig config(op);
      config.inputs.emplace_back(std::string{"past_key_values.0."} + suffix, cache_type,
                                 std::vector<int64_t>{-1, 3, fixed_capacity, 4});
      config.outputs.emplace_back(std::string{"present.0."} + suffix, cache_type,
                                  std::vector<int64_t>{-1, 3, fixed_capacity, 4});
      auto graph = GraphBuilder::Build(config);
      OrtSession* session = nullptr;
      Ort::ThrowOnError(Ort::GetModelEditorApi().CreateSessionFromModel(
          &GetOrtEnv(), graph.get(), session_options_.get(), &session));
      sessions.emplace_back(session);
      session_info_.Add(*session);
    }
  }
  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams&) const override { return nullptr; }
  std::vector<std::unique_ptr<OrtSession>> sessions;
};

struct TestState : State {
  TestState(const Model& model, const GeneratorParams& params) : State{params, model} {}
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override { return {}; }
  void Execute(OrtSession& session, int length = 1, int variant = 0) { State::Run(session, true, length, variant); }
  void ExecuteUncaptured(OrtSession& session) { State::Run(session, false); }
  int CaptureId(int length = 1, int variant = 0) { return GraphIdForLength(length, variant); }
};

// Load a real GQA graph with sparse layer indices and custom cache names.
struct TestGqaModel : Model {
  explicit TestGqaModel(bool capture = true) : Model{MakeConfig(capture && GrowthAvailable(true))} {
    config_->model.decoder.num_key_value_heads = 1;
    config_->model.decoder.num_attention_heads = 2;
    config_->model.decoder.head_size = 64;
    config_->model.decoder.inputs.past_key_names = "kv.%d.k";
    config_->model.decoder.inputs.past_value_names = "kv.%d.v";
    config_->model.decoder.outputs.present_key_names = "out.%d.k";
    config_->model.decoder.outputs.present_value_names = "out.%d.v";
    config_->config_path = fs::path{ORTGENAI_TEST_WEBGPU_FALLBACK_MODEL_DIR};
    config_->model.decoder.filename = "gqa.onnx";
    session_options_->AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    session = CreateSession(GetOrtEnv(), config_->model.decoder.filename, session_options_.get());
    session_info_.Add(*session);
  }
  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams&) const override { return nullptr; }
  std::unique_ptr<OrtSession> session;
};

std::vector<float> Read(DeviceInterface& device, OrtValue& value) {
  auto wrapped = WrapTensor<float>(device, value);
  auto data = wrapped.CopyDeviceToCpu();
  return {data.begin(), data.end()};
}

void CheckGrowth(const std::shared_ptr<TestModel>& model) {
  auto params = CreateGeneratorParams(*model);
  TestState state{*model, *params};
  ASSERT_EQ(model->p_device_kvcache_->GetType(), DeviceType::WEBGPU);
  ASSERT_TRUE(params->IsPastPresentShareBufferEnabled(model->config_->model.type));
  ASSERT_FALSE(model->config_->model.decoder.sliding_window.has_value());
  ASSERT_EQ(model->session_info_.GetInputShape("past_key_values.0.key")[2], -1);
  ASSERT_EQ(model->session_info_.GetOutputShape("present.0.key")[2], -1);
  ASSERT_TRUE(SharedCacheGrowth::IsSupported(
      state, std::vector<std::string>{"past_key_values.0.key", "past_key_values.0.value"},
      std::vector<std::string>{"present.0.key", "present.0.value"}));
  WebGPU::SharedKeyValueCache cache{state};
  cache.Add();
  ASSERT_EQ(cache.GetShape()[2], 1);
  cache.Update({}, 3);
  ASSERT_EQ(cache.GetShape()[2], 512);
  auto& device = *model->p_device_kvcache_;
  auto before = state.inputs_[0];
  std::vector<float> values(2 * 3 * 512 * 4);
  std::iota(values.begin(), values.end(), 1.0f);
  auto span = WrapTensor<float>(device, *state.outputs_[0]);
  std::copy(values.begin(), values.end(), span.CpuSpan().begin());
  span.CopyCpuToDevice();
  cache.Update({}, 4);
  EXPECT_EQ(before, state.inputs_[0]);
  EXPECT_EQ(state.inputs_[0], state.outputs_[0]);
  cache.Update({}, 1025);  // A multi-token append crosses several capacity boundaries.
  ASSERT_EQ(cache.GetShape()[2], 2048);
  EXPECT_NE(before, state.inputs_[0]);
  EXPECT_EQ(state.inputs_[0], state.outputs_[0]);
  auto migrated = Read(device, *state.inputs_[0]);
  for (size_t head = 0; head < 6; ++head) {
    for (size_t i = 0; i < 2048; ++i) EXPECT_EQ(migrated[head * 8192 + i], values[head * 2048 + i]);
    for (size_t i = 2048; i < 8192; ++i) EXPECT_EQ(migrated[head * 8192 + i], 0);
  }
  cache.RewindTo(2);
  cache.Update({}, 5);
  EXPECT_EQ(cache.GetShape()[2], 2048);
  cache.Update({}, 4090);
  EXPECT_EQ(cache.GetShape()[2], 4093);  // Clamp doubling to a non-power-of-two limit.
  EXPECT_THROW(cache.Update({}, 4094), std::runtime_error);
  EXPECT_EQ(cache.GetShape()[2], 4093);
}

TEST_F(WebGpuGrowthTest, MigratesBatchesAndHeadsAndRetainsAliasing) {
  CheckGrowth(std::make_shared<TestModel>());
}

std::unique_ptr<Config> FallbackConfig(const char* filename, bool capture, int max_length = 1024) {
  auto config = MakeConfig();
  config->config_path = fs::path{ORTGENAI_TEST_WEBGPU_FALLBACK_MODEL_DIR};
  config->model.decoder.filename = filename;
  config->model.vocab_size = 16;
  config->model.decoder.num_attention_heads = 1;
  config->model.decoder.num_key_value_heads = 1;
  config->model.decoder.head_size = 64;
  config->model.decoder.hidden_size = 64;
  config->search.batch_size = 1;
  config->search.num_beams = 1;
  config->search.max_length = max_length;
  config->search.past_present_share_buffer = true;
  SetProviderOption(*config, "WebGPU", "enableGraphCapture", capture ? "1" : "0");
  config->model.decoder.inputs.past_key_names = "kv.%d.k";
  config->model.decoder.inputs.past_value_names = "kv.%d.v";
  config->model.decoder.outputs.present_key_names = "out.%d.k";
  config->model.decoder.outputs.present_value_names = "out.%d.v";
  return config;
}

void CheckFallbackGeneration(const char* filename, bool memory_model = false, int max_length = 1024,
                             bool cross_growth = false, bool expected_capture = false, bool omit_capture = false) {
  auto config = FallbackConfig(filename, true, max_length);
  if (omit_capture) {
    for (auto& provider : config->model.decoder.session_options.provider_options) {
      auto& options = provider.options;
      options.erase(std::remove_if(options.begin(), options.end(),
                                   [](const auto& option) { return option.first == "enableGraphCapture"; }),
                    options.end());
    }
  }
  std::vector<std::byte> bytes;
  if (memory_model) {
    std::ifstream input((config->config_path / fs::path{filename}).string(), std::ios::binary | std::ios::ate);
    ASSERT_TRUE(input.good());
    bytes.resize(static_cast<size_t>(input.tellg()));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    config->model_data_spans_[filename] = bytes;
  }
  const int initial_models = LeakChecked<Model>::Count();
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  // The failed Model is gone; no retained GC model or eager variant remains.
  EXPECT_EQ(LeakChecked<Model>::Count(), initial_models + 1);
  std::fill(bytes.begin(), bytes.end(), std::byte{0});
  ASSERT_EQ(model->p_device_->GetType(), DeviceType::WEBGPU);
  EXPECT_EQ(IsGraphCaptureEnabled(model->config_->model.decoder.session_options), expected_capture);
  EXPECT_TRUE(model->config_->search.past_present_share_buffer);
  EXPECT_EQ(model->config_->search.num_beams, 1);
  EXPECT_EQ(model->config_->search.max_length, max_length);
  EXPECT_EQ(model->p_device_inputs_->GetType(), expected_capture ? DeviceType::WEBGPU : DeviceType::CPU);
  const auto& decoder = model->config_->model.decoder;
  const auto key = ComposeKeyValueName(decoder.inputs.past_key_names, 3);
  const auto declared_capacity = model->session_info_.GetInputShape(key)[2];
  ASSERT_EQ(declared_capacity, std::string_view{filename}.compare(0, 6, "fixed-") == 0 ? 16 : -1);
  auto params = CreateGeneratorParams(*model);
  EXPECT_EQ(params->use_graph_capture, expected_capture);
  auto generator = CreateGenerator(*model, *params);
  EXPECT_EQ(generator->model_.get(), model.get());
  auto reference = CreateModel(GetOrtEnv(), FallbackConfig(filename, false, max_length));
  EXPECT_FALSE(IsGraphCaptureEnabled(reference->config_->model.decoder.session_options));
  auto reference_params = CreateGeneratorParams(*reference);
  auto eager = CreateGenerator(*reference, *reference_params);
  std::vector<int32_t> prompt(cross_growth ? 510 : 2, 1);
  prompt.back() = 2;
  generator->AppendTokens(prompt);
  eager->AppendTokens(prompt);
  for (int step = 0; step < 6; ++step) {
    const auto scores = generator->GetLogits().CopyDeviceToCpu();
    const auto expected = eager->GetLogits().CopyDeviceToCpu();
    ASSERT_EQ(scores.size(), expected.size());
    EXPECT_TRUE(std::equal(scores.begin(), scores.end(), expected.begin()));
    EXPECT_EQ(std::max_element(scores.begin(), scores.end()) - scores.begin(), 3 + step);
    generator->GenerateNextToken();
    eager->GenerateNextToken();
  }
  const auto sequence = generator->GetSequence(0).CopyDeviceToCpu();
  const auto expected_sequence = eager->GetSequence(0).CopyDeviceToCpu();
  EXPECT_TRUE(std::equal(sequence.begin(), sequence.end(), expected_sequence.begin(), expected_sequence.end()));
  auto& state = *generator->state_;
  size_t cache_count = 0;
  for (size_t i = 0; i < state.output_names_.size(); ++i) {
    const std::string_view name{state.output_names_[i]};
    if (name.compare(0, 4, "out.") != 0) continue;
    ++cache_count;
    const auto cache = Read(*model->p_device_kvcache_, *state.outputs_[i]);
    const auto expected_cache = Read(*reference->p_device_kvcache_, *eager->state_->outputs_[i]);
    const auto live_elements = (prompt.size() + 5) * 64;
    ASSERT_GE(cache.size(), live_elements);
    ASSERT_GE(expected_cache.size(), live_elements);
    EXPECT_TRUE(std::equal(cache.begin(), cache.begin() + live_elements, expected_cache.begin()));
    const auto expected_capacity = declared_capacity > 0               ? declared_capacity
                                   : GrowthAvailable(expected_capture) ? (cross_growth ? 1024 : std::min(max_length, 512))
                                                                       : max_length;
    EXPECT_EQ(state.outputs_[i]->GetTensorTypeAndShapeInfo()->GetShape()[2], expected_capacity);
    EXPECT_NE(std::find(state.inputs_.begin(), state.inputs_.end(), state.outputs_[i]), state.inputs_.end());
    if (name.size() >= 2 && name.compare(name.size() - 2, 2, ".v") == 0) {
      for (size_t token = 0; token < prompt.size() + 5; ++token) {
        const float expected = token < prompt.size() ? static_cast<float>(prompt[token]) : static_cast<float>(3 + token - prompt.size());
        for (size_t dim = 0; dim < 64; ++dim) EXPECT_FLOAT_EQ(cache[token * 64 + dim], expected);
      }
    }
  }
  EXPECT_EQ(cache_count, 2);
}

TEST(WebGpuCaptureFallback, SharedDynamicPreservesMemoryModelAndGrows) {
  CheckFallbackGeneration("control-flow.onnx", true, 4093, true);
}

TEST(WebGpuCaptureFallback, SharedDynamicCpuComputeRebuildsEagerSession) {
  CheckFallbackGeneration("cpu-compute.onnx");
}

TEST(WebGpuCaptureFallback, SharedStaticPreservesDeclaredCapacity) {
  // A smaller search limit must not shrink the model's fixed 16-slot tensors.
  CheckFallbackGeneration("fixed-control-flow.onnx", false, 8);
}

TEST(WebGpuCaptureFallback, CompatibleSharedDynamicKeepsCaptureThroughGrowth) {
  CheckFallbackGeneration("shared.onnx", false, 1024, true, true);
}

TEST(WebGpuCaptureFallback, MissingCaptureOptionDefaultsOnWithEagerFallback) {
  CheckFallbackGeneration("shared.onnx", false, 1024, true, true, true);
  CheckFallbackGeneration("control-flow.onnx", false, 1024, true, false, true);
  CheckFallbackGeneration("fixed-control-flow.onnx", false, 8, false, false, true);
  auto config = FallbackConfig("shared.onnx", false);
  config->model.decoder.session_options.provider_options.front().options.clear();
  config->search.past_present_share_buffer = false;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  EXPECT_FALSE(IsGraphCaptureEnabled(model->config_->model.decoder.session_options));
  EXPECT_FALSE(model->config_->search.past_present_share_buffer);
  config = FallbackConfig("shared.onnx", false);
  config->model.decoder.session_options.provider_options.front().options.clear();
  config->search.num_beams = 2;
  model = CreateModel(GetOrtEnv(), std::move(config));
  EXPECT_FALSE(IsGraphCaptureEnabled(model->config_->model.decoder.session_options));
  config = FallbackConfig("shared.onnx", false);
  config->model.decoder.session_options.provider_options.clear();
  model = CreateModel(GetOrtEnv(), std::move(config));
  EXPECT_TRUE(IsGraphCaptureEnabled(model->config_->model.decoder.session_options));
  config = FallbackConfig("shared.onnx", false);
  ClearProviders(*config);
  SetProviderOption(*config, "CPU", {}, {});
  // Stale WebGPU options must not change the selected CPU provider's behavior.
  model = CreateModel(GetOrtEnv(), std::move(config));
  EXPECT_EQ(model->p_device_->GetType(), DeviceType::CPU);
  EXPECT_FALSE(IsGraphCaptureEnabled(model->config_->model.decoder.session_options));
}

TEST(WebGpuCaptureFallback, RepeatedGeneratorsUseOneEagerModel) {
  const int initial_models = LeakChecked<Model>::Count();
  for (const char* filename : {"control-flow.onnx", "fixed-control-flow.onnx"}) {
    SCOPED_TRACE(filename);
    auto model = CreateModel(GetOrtEnv(), FallbackConfig(filename, true));
    auto* decoder = dynamic_cast<DecoderOnly_Model*>(model.get());
    ASSERT_NE(decoder, nullptr);
    auto* session = decoder->session_decoder_.get();
    for (int attempt = 0; attempt < 2; ++attempt) {
      auto params = CreateGeneratorParams(*model);
      auto first = CreateGenerator(*model, *params);
      auto second = CreateGenerator(*model, *params);
      EXPECT_FALSE(params->use_graph_capture);
      EXPECT_EQ(first->model_.get(), model.get());
      EXPECT_EQ(second->model_.get(), model.get());
      EXPECT_EQ(decoder->session_decoder_.get(), session);
      EXPECT_EQ(LeakChecked<Model>::Count(), initial_models + 1);
      // Reusing the model must also work after a previous generator ran.
      const std::vector<int32_t> prompt{1, 2};
      for (auto* generator : {first.get(), second.get()}) {
        generator->AppendTokens(prompt);
        generator->GenerateNextToken();
        EXPECT_EQ(generator->GetSequence(0).CopyDeviceToCpu().back(), 3);
      }
    }
  }
  EXPECT_EQ(LeakChecked<Model>::Count(), initial_models);
}

TEST(WebGpuCaptureFallback, OutOfScopeRetainsCaptureError) {
  for (const char* filename : {"control-flow.onnx", "fixed-control-flow.onnx"}) {
    for (int restriction = 0; restriction < 3; ++restriction) {
      SCOPED_TRACE(filename);
      SCOPED_TRACE(restriction);
      auto config = FallbackConfig(filename, true);
      if (restriction == 0) config->search.past_present_share_buffer = false;
      if (restriction == 1) config->search.num_beams = 2;
      if (restriction == 2) config->model.decoder.inputs.past_key_names = "absent.%d.key";
      const int initial_models = LeakChecked<Model>::Count();
      EXPECT_THROW(CreateModel(GetOrtEnv(), std::move(config)), Ort::Exception);
      EXPECT_EQ(LeakChecked<Model>::Count(), initial_models);
    }
  }
}

TEST(WebGpuCaptureFallback, MissingModelsStillFail) {
  EXPECT_THROW(CreateModel(GetOrtEnv(), FallbackConfig("missing.onnx", true)), Ort::Exception);
}

TEST(WebGpuCaptureFallback, EagerFailureStillPropagates) {
  auto config = FallbackConfig("control-flow.onnx", true);
  config->model.decoder.session_options.config_entries.emplace_back("session.disable_cpu_ep_fallback", "1");
  EXPECT_THROW(CreateModel(GetOrtEnv(), std::move(config)), Ort::Exception);
}

TEST_F(WebGpuGrowthTest, SymbolicShapeAloneDoesNotEnableSharing) {
  auto model = std::make_shared<TestModel>();
  auto params = CreateGeneratorParams(*model);
  params->search.past_present_share_buffer = false;
  EXPECT_FALSE(params->IsPastPresentShareBufferEnabled(model->config_->model.type));
  TestState state{*model, *params};
  if (params->use_graph_capture) {
    EXPECT_THROW(WebGPU::SharedKeyValueCache{state}, std::runtime_error);
  }
}

TEST_F(WebGpuGrowthTest, SharesSparseGqaBindingsOnlyWhenRequested) {
  auto model = std::make_shared<TestGqaModel>();
  auto params = CreateGeneratorParams(*model);
  ASSERT_TRUE(params->search.past_present_share_buffer);
  ASSERT_TRUE(params->IsPastPresentShareBufferEnabled(model->config_->model.type));
  TestState state{*model, *params};
  WebGPU::SharedKeyValueCache cache{state};
  cache.Add();
  cache.Update({}, 128);
  auto* original = state.inputs_[0];
  EXPECT_EQ(state.inputs_[0], state.outputs_[0]);
  cache.Update({}, 320);
  EXPECT_EQ(state.inputs_[0], original);
  cache.Update({}, 513);
  EXPECT_EQ(cache.GetShape()[2], 1024);
  EXPECT_EQ(state.inputs_[0], state.outputs_[0]);
  EXPECT_NE(state.inputs_[0], original);
  EXPECT_TRUE(params->search.past_present_share_buffer);
  params->search.num_beams = 2;
  EXPECT_FALSE(params->IsPastPresentShareBufferEnabled(model->config_->model.type));
  params->search.num_beams = 1;
  params->search.past_present_share_buffer = false;
  EXPECT_FALSE(params->IsPastPresentShareBufferEnabled(model->config_->model.type));
  EXPECT_FALSE(SharedCacheGrowth::IsSupported(
      state, std::vector<std::string>{"kv.3.k", "kv.3.v"}, std::vector<std::string>{"out.3.k", "out.3.v"}));
  TestState unshared_state{*model, *params};
  if (params->use_graph_capture) {
    EXPECT_THROW(CreateStandardKeyValueCache(unshared_state), std::runtime_error);
  }
  params->use_graph_capture = false;
  auto unshared = CreateStandardKeyValueCache(unshared_state);
  ASSERT_NE(unshared, nullptr);
  unshared->Add();
  unshared->Update({}, 128);
  EXPECT_NE(unshared_state.inputs_[0], unshared_state.outputs_[0]);
  EXPECT_EQ(unshared_state.outputs_[0]->GetTensorTypeAndShapeInfo()->GetShape()[2], 128);
  unshared->Update({}, 320);
  EXPECT_EQ(unshared_state.outputs_[0]->GetTensorTypeAndShapeInfo()->GetShape()[2], 320);
}

TEST_F(WebGpuGrowthTest, OtherDevicesRetainSharedMaxLengthPolicy) {
  auto model = std::make_shared<TestModel>();
  model->p_device_kvcache_ = GetDeviceInterface(DeviceType::CPU);
  auto params = CreateGeneratorParams(*model);
  ASSERT_TRUE(params->search.past_present_share_buffer);
  ASSERT_TRUE(params->IsPastPresentShareBufferEnabled(model->config_->model.type));
  TestState state{*model, *params};
  EXPECT_FALSE(SharedCacheGrowth::IsSupported(
      state, std::vector<std::string>{"past_key_values.0.key", "past_key_values.0.value"},
      std::vector<std::string>{"present.0.key", "present.0.value"}));
  SharedKeyValueCache cache{state};
  EXPECT_EQ(cache.GetShape()[2], params->search.max_length);
}

TEST_F(WebGpuGrowthTest, ChecksDeviceAndRuntimeBeforeProbingOptionalApi) {
  auto model = std::make_shared<TestModel>();
  auto params = CreateGeneratorParams(*model);
  TestState state{*model, *params};
  const std::vector<std::string> inputs{"past_key_values.0.key"}, outputs{"present.0.key"};
  const auto* original_api = Ort::api;
  const auto original_version = Ort::api_version;
  OrtApi unavailable_api{};
  // A null probe would fail if either guard tried to access the optional entry.
  Ort::api = &unavailable_api;
  Ort::api_version = 21;
  EXPECT_FALSE(SharedCacheGrowth::IsSupported(state, inputs, outputs));
  Ort::api_version = 26;
  params->use_graph_capture = true;
  EXPECT_FALSE(SharedCacheGrowth::IsSupported(state, inputs, outputs));
  model->p_device_kvcache_ = GetDeviceInterface(DeviceType::CPU);
  Ort::api_version = original_version;
  EXPECT_FALSE(SharedCacheGrowth::IsSupported(state, inputs, outputs));
  Ort::api = original_api;
  Ort::api_version = original_version;
}

TEST_F(WebGpuGrowthTest, WebGpuFactoryGrowsAttentionInConvCache) {
  auto model = std::make_shared<TestModel>();
  model->config_->model.decoder.layer_types = {"full_attention"};
  auto params = CreateGeneratorParams(*model);
  TestState state{*model, *params};
  auto cache = CreateStandardKeyValueCache(state);
  ASSERT_NE(dynamic_cast<WebGPU::ConvKeyValueCache*>(cache.get()), nullptr);
  cache->Add();
  cache->Update({}, 128);
  ASSERT_EQ(state.outputs_[0]->GetTensorTypeAndShapeInfo()->GetShape()[2], 512);
  auto& device = *model->p_device_kvcache_;
  auto values = WrapTensor<float>(device, *state.outputs_[0]);
  std::fill(values.CpuSpan().begin(), values.CpuSpan().end(), 2.0f);
  values.CopyCpuToDevice();
  cache->Update({}, 513);
  ASSERT_EQ(state.outputs_[0]->GetTensorTypeAndShapeInfo()->GetShape()[2], 1024);
  EXPECT_EQ(state.inputs_[0], state.outputs_[0]);
  const auto migrated = Read(device, *state.outputs_[0]);
  for (size_t head = 0; head < 6; ++head) {
    EXPECT_EQ(migrated[head * 4096], 2.0f);
    EXPECT_EQ(migrated[head * 4096 + 2048], 0.0f);
  }
  model->p_device_kvcache_ = GetDeviceInterface(DeviceType::CPU);
  TestState cpu_state{*model, *params};
  auto cpu_cache = CreateStandardKeyValueCache(cpu_state);
  EXPECT_EQ(dynamic_cast<WebGPU::ConvKeyValueCache*>(cpu_cache.get()), nullptr);
  cpu_cache->Add();
  EXPECT_EQ(cpu_state.outputs_[0]->GetTensorTypeAndShapeInfo()->GetShape()[2], params->search.max_length);
}

TEST_F(WebGpuGrowthTest, MigratesFloat16BatchesAndHeadsWithoutChangingBits) {
  auto model = std::make_shared<TestModel>(-1, "Identity", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  auto params = CreateGeneratorParams(*model);
  TestState state{*model, *params};
  WebGPU::SharedKeyValueCache cache{state};
  cache.Add();
  cache.Update({}, 3);
  ASSERT_EQ(cache.GetShape()[2], 512);
  auto& device = *model->p_device_kvcache_;
  std::vector<Ort::Float16_t> values(2 * 3 * 512 * 4);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = Ort::Float16_t{static_cast<uint16_t>(0x3000 + i % 4096)};
  }
  for (auto* tensor : state.inputs_) {
    auto span = WrapTensor<Ort::Float16_t>(device, *tensor);
    std::copy(values.begin(), values.end(), span.CpuSpan().begin());
    span.CopyCpuToDevice();
  }
  cache.Update({}, 513);
  ASSERT_EQ(cache.GetShape()[2], 1024);
  for (size_t tensor = 0; tensor < state.inputs_.size(); ++tensor) {
    EXPECT_EQ(state.inputs_[tensor], state.outputs_[tensor]);
    auto span = WrapTensor<Ort::Float16_t>(device, *state.inputs_[tensor]);
    const auto migrated = span.CopyDeviceToCpu();
    for (size_t head = 0; head < 6; ++head) {
      for (size_t i = 0; i < 2048; ++i) EXPECT_EQ(migrated[head * 4096 + i].value, values[head * 2048 + i].value);
      for (size_t i = 2048; i < 4096; ++i) EXPECT_EQ(migrated[head * 4096 + i].value, 0);
    }
  }
}

void CheckGqaDecodeThroughGrowth(bool capture) {
  auto model = std::make_shared<TestGqaModel>(capture);
  auto params = CreateGeneratorParams(*model);
  ASSERT_EQ(params->use_graph_capture, capture);
  ASSERT_TRUE(params->IsPastPresentShareBufferEnabled(model->config_->model.type));
  TestState state{*model, *params};
  WebGPU::SharedKeyValueCache cache{state};
  cache.Add();
  cache.Update({}, 510);
  ASSERT_EQ(cache.GetShape()[2], 512);
  auto& device = *model->p_device_kvcache_;
  std::vector<float> history(510);
  std::iota(history.begin(), history.end(), 1.0f);
  auto write = [&](OrtValue& tensor, const auto& values) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    auto span = WrapTensor<T>(device, tensor);
    std::copy(values.begin(), values.end(), span.CpuSpan().begin());
    span.CopyCpuToDevice();
  };
  write(*state.inputs_[0], std::vector<float>(2 * 512 * 64, 0.0f));
  std::vector<float> values(2 * 512 * 64, 100000.0f);  // Poison spare capacity.
  for (size_t batch = 0; batch < 2; ++batch) {
    for (size_t token = 0; token < history.size(); ++token) {
      for (size_t dim = 0; dim < 64; ++dim) {
        values[(batch * 512 + token) * 64 + dim] = history[token] + batch * 128 + dim / 8.0f;
      }
    }
  }
  write(*state.inputs_[1], values);
  auto query = OrtValue::CreateTensor<float>(device.GetAllocator(), std::array<int64_t, 3>{2, 1, 128});
  auto key = OrtValue::CreateTensor<float>(device.GetAllocator(), std::array<int64_t, 3>{2, 1, 64});
  auto value = OrtValue::CreateTensor<float>(device.GetAllocator(), std::array<int64_t, 3>{2, 1, 64});
  auto seqlens = OrtValue::CreateTensor<int32_t>(device.GetAllocator(), std::array<int64_t, 1>{2});
  auto total = OrtValue::CreateTensor<int32_t>(device.GetAllocator(), std::array<int64_t, 1>{1});
  auto attention = OrtValue::CreateTensor<float>(device.GetAllocator(), std::array<int64_t, 3>{2, 1, 128});
  write(*query, std::vector<float>(256, 0.0f));
  write(*key, std::vector<float>(128, 0.0f));
  for (const char* name : {"q", "k", "v", "seqlens_k", "total_sequence_length"}) state.input_names_.push_back(name);
  for (auto* tensor : {query.get(), key.get(), value.get(), seqlens.get(), total.get()}) {
    state.inputs_.push_back(tensor);
  }
  state.output_names_.push_back("attention");
  state.outputs_.push_back(attention.get());
  int ordinal = 0;
  auto decode = [&] {
    const auto past_length = static_cast<int32_t>(history.size());
    cache.Update({}, past_length + 1);
    history.push_back(1000.0f + ++ordinal * 16);
    std::vector<float> next(128);
    for (size_t batch = 0; batch < 2; ++batch) {
      for (size_t dim = 0; dim < 64; ++dim) next[batch * 64 + dim] = history.back() + batch * 128 + dim / 8.0f;
    }
    write(*value, next);
    write(*seqlens, std::vector<int32_t>(2, past_length));
    write(*total, std::vector<int32_t>{past_length + 1});
    if (capture) {
      state.Execute(*model->session);
    } else {
      state.ExecuteUncaptured(*model->session);
    }
    EXPECT_EQ(state.inputs_[0], state.outputs_[0]);
    EXPECT_EQ(state.inputs_[1], state.outputs_[1]);
    // Zero Q/K gives uniform attention over the live tokens, independently
    // of cache capacity, poisoned tails, or whether this run was replayed.
    const float mean = std::accumulate(history.begin(), history.end(), 0.0f) / history.size();
    const auto output = Read(device, *attention);
    for (size_t batch = 0; batch < 2; ++batch) {
      for (size_t dim = 0; dim < 128; ++dim) {
        EXPECT_NEAR(output[batch * 128 + dim], mean + batch * 128 + (dim % 64) / 8.0f, 0.005f);
      }
    }
  };
  const auto capture_id = state.CaptureId();
  decode();  // Record at length 511, then replay with changing values/lengths.
  decode();
  EXPECT_EQ(state.CaptureId(), capture_id);
  decode();  // Grow to 1024, migrate and recapture at length 513.
  decode();
  ASSERT_EQ(cache.GetShape()[2], 1024);
  EXPECT_NE(state.CaptureId(), capture_id);
  for (size_t length : {size_t{2}, size_t{0}}) {
    cache.RewindTo(length);
    history.resize(length);
    decode();  // Old live tokens become masked tails; overwrite the new token.
    decode();
    EXPECT_EQ(cache.GetShape()[2], 1024);
    EXPECT_NE(state.CaptureId(), capture_id);
  }
}

TEST_F(WebGpuGrowthTest, GqaUsesLogicalLengthThroughGrowthAndRewind) {
  CheckGqaDecodeThroughGrowth(false);
}

TEST_F(WebGpuGrowthTest, GqaReplayUsesLogicalLengthThroughGrowthAndRewind) {
  if (!GrowthAvailable(true)) GTEST_SKIP() << "Captured growth requires ORT API 27+ and model-editor support.";
  CheckGqaDecodeThroughGrowth(true);
}

TEST_F(WebGpuGrowthTest, PrefillReservesFirstDecodeToken) {
  auto model = std::make_shared<TestModel>();
  auto params = CreateGeneratorParams(*model);
  TestState state{*model, *params};
  WebGPU::SharedKeyValueCache cache{state};
  cache.Add();
  cache.Update({}, 512);
  EXPECT_EQ(cache.GetShape()[2], 1024);
  const auto* original = state.inputs_[0];
  cache.Update({}, 513);
  EXPECT_EQ(state.inputs_[0], original);
}

TEST_F(WebGpuGrowthTest, InitialReserveAvoidsShortDecodeMigration) {
  auto model = std::make_shared<TestModel>();
  auto params = CreateGeneratorParams(*model);
  TestState state{*model, *params};
  WebGPU::SharedKeyValueCache cache{state};
  cache.Add();
  EXPECT_EQ(cache.GetShape()[2], 1);  // Defer the reserve until prefill.
  cache.Update({}, 128);
  ASSERT_EQ(cache.GetShape()[2], 512);
  const auto* original = state.inputs_[0];
  const auto capture_id = state.CaptureId();
  cache.Update({}, 320);
  EXPECT_EQ(state.inputs_[0], original);
  EXPECT_EQ(state.inputs_[0], state.outputs_[0]);
  EXPECT_EQ(state.CaptureId(), capture_id);
}

TEST_F(WebGpuGrowthTest, InitialReserveRespectsSmallSearchLimit) {
  auto model = std::make_shared<TestModel>();
  auto params = CreateGeneratorParams(*model);
  params->search.max_length = 257;
  TestState state{*model, *params};
  WebGPU::SharedKeyValueCache cache{state};
  cache.Add();
  cache.Update({}, 128);
  ASSERT_EQ(cache.GetShape()[2], 257);
  const auto* original = state.inputs_[0];
  cache.Update({}, 257);
  EXPECT_EQ(state.inputs_[0], original);
  EXPECT_THROW(cache.Update({}, 258), std::runtime_error);
}

TEST_F(WebGpuGrowthTest, DeclaredZeroDimensionIsNotGrowable) {
  auto model = std::make_shared<TestModel>(0);
  auto params = CreateGeneratorParams(*model);
  TestState state{*model, *params};
  EXPECT_FALSE(SharedCacheGrowth::IsSupported(state, std::vector<std::string>{"past_key_values.0.key"},
                                              std::vector<std::string>{"present.0.key"}));
}

TEST_F(WebGpuGrowthTest, FixedShapePreservesDeclaredCapacity) {
  auto model = std::make_shared<TestModel>(32);
  auto params = CreateGeneratorParams(*model);
  params->search.past_present_share_buffer = false;
  EXPECT_FALSE(params->IsPastPresentShareBufferEnabled(model->config_->model.type));
  TestState state{*model, *params};
  WebGPU::SharedKeyValueCache cache{state};
  cache.Add();
  EXPECT_EQ(cache.GetShape()[2], 32);
  const auto* original = state.inputs_[0];
  cache.Update({}, 5);
  EXPECT_EQ(state.inputs_[0], original);
  EXPECT_FALSE(params->search.past_present_share_buffer);
}

TEST_F(WebGpuGrowthTest, ParallelSessionsHaveIndependentCapacityAndData) {
  auto first = std::make_shared<TestModel>();
  auto second = std::make_shared<TestModel>();
  auto run = std::async(std::launch::async, [&] { CheckGrowth(first); });
  CheckGrowth(second);
  run.get();
}

TEST_F(WebGpuGrowthTest, GrowthRecapturesAllVariantsAndPreservesAnotherGenerator) {
  if (!GrowthAvailable(true)) GTEST_SKIP() << "Captured growth requires ORT API 27+ and model-editor support.";
  auto model = std::make_shared<TestModel>(-1, "Neg");
  auto params = CreateGeneratorParams(*model);
  TestState first{*model, *params}, second{*model, *params};
  auto& device = *model->p_device_kvcache_;
  auto bind = [&](TestState& state, std::unique_ptr<OrtValue>& output) {
    std::vector<std::unique_ptr<OrtValue>> tensors;
    tensors.push_back(OrtValue::CreateTensor<float>(device.GetAllocator(), std::array<int64_t, 4>{2, 3, 4, 4}));
    auto span = WrapTensor<float>(device, *tensors.front());
    std::fill(span.CpuSpan().begin(), span.CpuSpan().end(), 2.0f);
    span.CopyCpuToDevice();
    state.input_names_ = {"past_key_values.0.key"};
    state.output_names_ = {"present.0.key"};
    state.inputs_ = {tensors.front().get()};
    output = OrtValue::CreateTensor<float>(device.GetAllocator(), std::array<int64_t, 4>{2, 3, 4, 4});
    state.outputs_ = {output.get()};
    return tensors;
  };
  std::unique_ptr<OrtValue> first_output, second_output;
  auto first_tensors = bind(first, first_output);
  auto second_tensors = bind(second, second_output);
  auto& session = *model->sessions.front();
  first.Execute(session);
  first.Execute(session, 2, 1);
  second.Execute(session);
  const auto second_before = Read(device, *second.inputs_[0]);
  const auto first_id = first.CaptureId();
  const auto second_id = second.CaptureId();
  SharedCacheGrowth growth{first};
  EXPECT_EQ(growth.EnsureCapacity(9, true, first_tensors, 0, 0), 16);
  first_output = OrtValue::CreateTensor<float>(device.GetAllocator(), std::array<int64_t, 4>{2, 3, 16, 4});
  first.outputs_[0] = first_output.get();
  EXPECT_NE(first.CaptureId(), first_id);
  EXPECT_EQ(second.CaptureId(), second_id);
  second.Execute(session);
  const auto second_after = Read(device, *second.outputs_[0]);
  for (size_t i = 0; i < second_before.size(); ++i) EXPECT_EQ(second_after[i], -second_before[i]);
  for (auto [length, variant] : {std::pair{1, 0}, std::pair{2, 1}}) {
    first.Execute(session, length, variant);
    const auto before = Read(device, *first.inputs_[0]);
    first.Execute(session, length, variant);
    const auto after = Read(device, *first.outputs_[0]);
    for (size_t i = 0; i < before.size(); ++i) EXPECT_EQ(after[i], -before[i]);
  }
}

}  // namespace
}  // namespace Generators
