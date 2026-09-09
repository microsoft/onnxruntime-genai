// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "search.h"
#include "models/model.h"
#include "models/multi_modal.h"
#include "models/io/dynamic_kv_cache.h"
#include "models/io/shared_kv_cache.h"
#include "json.h"
#include "ep_registration.h"
#include "telemetry_test_environment.h"

namespace {
using namespace Generators;

// Explicit-instantiation access is confined to this white-box binary. It leaves
// production declarations/layout unchanged (unlike redefining private as public).
template <typename Tag, auto Member>
struct InspectField {
  friend typename Tag::type Field(Tag) { return Member; }
};
#define INSPECT_FIELD(Tag, Owner, Member, Type) \
  struct Tag {                                  \
    using type = Type Owner::*;                 \
    friend type Field(Tag);                     \
  };                                            \
  template struct InspectField<Tag, &Owner::Member>
INSPECT_FIELD(Decoder, MultiModalPipelineState, decoder_state_, std::unique_ptr<DecoderState>);
INSPECT_FIELD(Vision, MultiModalPipelineState, vision_state_, std::unique_ptr<VisionState>);
INSPECT_FIELD(Embedding, MultiModalPipelineState, embedding_state_, std::unique_ptr<EmbeddingState>);
INSPECT_FIELD(TurnInputs, MultiModalPipelineState, turn_extra_inputs_, std::vector<ExtraInput>);
INSPECT_FIELD(Boundary, MultiModalPipelineState, multimodal_prompt_length_, size_t);
INSPECT_FIELD(Failed, MultiModalPipelineState, execution_failed_, bool);
INSPECT_FIELD(Cache, DecoderState, kv_cache_, std::unique_ptr<KeyValueCache>);
INSPECT_FIELD(Recurrent, DecoderState, recurrent_state_, std::unique_ptr<RecurrentState>);
INSPECT_FIELD(Length, DefaultKeyValueCacheBase, current_length_, int);
INSPECT_FIELD(CaptureSession, State, graph_capture_session_, OrtSession*);
INSPECT_FIELD(VisionFeatures, VisionState, image_features_, std::unique_ptr<MultiModalFeatures>);
INSPECT_FIELD(ImageTemporaries, VisionState, per_image_tensors_, std::vector<std::unique_ptr<OrtValue>>);
INSPECT_FIELD(EmbeddingFeatures, EmbeddingState, image_features_, std::unique_ptr<MultiModalFeatures>);
#undef INSPECT_FIELD

struct GraphIds {
  friend std::map<int, int> Captures(State&, GraphIds);
};
template <auto Member>
struct InspectCaptures {
  friend std::map<int, int> Captures(State& state, GraphIds) {
    std::map<int, int> result;
    for (const auto& [shape, id] : state.*Member) result.emplace(shape, id.value);
    return result;
  }
};
template struct InspectCaptures<&State::graph_ids_>;

std::filesystem::path model_root = MULTIMODAL_GQA_MODEL_PATH;
std::vector<std::string> devices{"cpu"};

struct Case {
  std::string device, dtype, family;
  std::string Name() const { return device + "-" + dtype + "-" + family; }
  std::filesystem::path Directory(bool capture = false) const {
    return model_root / (Name() + (capture ? "-capture" : ""));
  }
  bool Gpu() const { return device == "cuda" || device == "webgpu"; }
};

std::vector<Case> Cases() {
  std::vector<Case> result;
  for (const auto& device : devices) {
    const std::vector<std::string> dtypes =
        device == "cuda"     ? std::vector<std::string>{"fp16"}
        : device == "webgpu" ? std::vector<std::string>{"fp32", "fp16"}
                             : std::vector<std::string>{"fp32"};
    for (const auto& dtype : dtypes)
      for (const auto* family : {"phi3v", "qwen2_5_vl", "mistral3"})
        result.push_back({device, dtype, family});
  }
  return result;
}

std::shared_ptr<MultiModalLanguageModel> Load(const Case& test, bool capture = false) {
  const auto directory = test.Directory(capture);
  if (!std::filesystem::exists(directory / "genai_config.json"))
    throw std::runtime_error("Missing GQA fixture: " + directory.string() +
                             ". Generate create_multimodal_gqa_model --suite first.");
  auto model = std::dynamic_pointer_cast<MultiModalLanguageModel>(
      CreateModel(GetOrtEnv(), directory.string().c_str()));
  if (!model)
    throw std::runtime_error("Fixture did not create a multimodal model");
  const auto expected = test.device == "cpu"    ? DeviceType::CPU
                        : test.device == "cuda" ? DeviceType::CUDA
                                                : DeviceType::WEBGPU;
  EXPECT_EQ(model->p_device_->GetType(), expected);
  return model;
}

std::unique_ptr<Generator> MakeGenerator(const Model& model, bool shared) {
  auto params = std::make_shared<GeneratorParams>(model);
  params->search.past_present_share_buffer = shared;
  return std::make_unique<Generator>(model, *params);
}

template <typename T>
std::shared_ptr<Tensor> CpuTensor(const std::vector<int64_t>& shape, const std::vector<T>& values) {
  auto value = OrtValue::CreateTensor<T>(GetDeviceInterface(DeviceType::CPU)->GetAllocator(), shape);
  std::copy(values.begin(), values.end(), value->GetTensorMutableData<T>());
  return std::make_shared<Tensor>(std::move(value));
}

NamedTensors ImageTurn(const Case& test, float start, bool unequal = false) {
  const bool qwen = test.family == "qwen2_5_vl";
  const bool pixtral = test.family == "mistral3";
  const std::vector<int64_t> counts = unequal && (qwen || pixtral) ? std::vector<int64_t>{2, 6}
                                      : qwen                       ? std::vector<int64_t>{4}
                                                                   : std::vector<int64_t>{1};
  std::vector<int32_t> ids{2};
  for (auto count : counts) {
    if (qwen) ids.push_back(28);
    ids.insert(ids.end(), count, qwen || pixtral ? 29 : -1);
    if (qwen) ids.push_back(27);
  }
  ids.push_back(3);
  const int64_t patches = std::accumulate(counts.begin(), counts.end(), int64_t{0});
  std::vector<float> pixels(patches * 3);
  std::iota(pixels.begin(), pixels.end(), start);
  std::vector<int64_t> pixel_shape = qwen ? std::vector<int64_t>{patches, 3}
                                          : std::vector<int64_t>{1, 1, 3, 1, 1};
  const std::vector<int64_t> sizes = unequal && pixtral ? std::vector<int64_t>{1, 2, 2, 3}
                                                        : std::vector<int64_t>{1, 1};
  if (pixtral) {
    const int64_t height = unequal ? 2 : 1, width = unequal ? 3 : 1;
    pixel_shape = {static_cast<int64_t>(counts.size()), 3, height, width};
    std::vector<float> padded(counts.size() * 3 * height * width);
    int64_t patch = 0;
    for (size_t image = 0; image < counts.size(); ++image) {
      for (int64_t row = 0; row < sizes[image * 2]; ++row)
        for (int64_t column = 0; column < sizes[image * 2 + 1]; ++column, ++patch)
          for (int64_t channel = 0; channel < 3; ++channel)
            padded[((image * 3 + channel) * height + row) * width + column] = pixels[patch * 3 + channel];
    }
    pixels = std::move(padded);
  }
  auto pixel_tensor = CpuTensor<float>(pixel_shape, pixels);
  if (test.dtype == "fp16") {
    std::unique_ptr<OrtValue> half;
    Cast(*pixel_tensor->ort_tensor_, half, *GetDeviceInterface(DeviceType::CPU), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    pixel_tensor = std::make_shared<Tensor>(std::move(half));
  }
  NamedTensors result{
      {"input_ids", CpuTensor<int32_t>({1, static_cast<int64_t>(ids.size())}, ids)},
      {"pixel_values", pixel_tensor},
      {"num_image_tokens", CpuTensor<int64_t>({static_cast<int64_t>(counts.size())}, counts)}};
  if (qwen) {
    result["image_grid_thw"] = CpuTensor<int64_t>(
        {static_cast<int64_t>(counts.size()), 3},
        unequal ? std::vector<int64_t>{1, 1, 2, 1, 2, 3} : std::vector<int64_t>{1, 2, 2});
  } else {
    result["image_sizes"] = CpuTensor<int64_t>({static_cast<int64_t>(counts.size()), 2}, sizes);
  }
  return result;
}

MultiModalPipelineState& Pipeline(Generator& generator) {
  return dynamic_cast<MultiModalPipelineState&>(*generator.state_);
}

DecoderState& DecoderOf(Generator& generator) {
  return *(Pipeline(generator).*Field(Decoder{}));
}

DefaultKeyValueCacheBase& CacheOf(Generator& generator) {
  return dynamic_cast<DefaultKeyValueCacheBase&>(*(DecoderOf(generator).*Field(Cache{})));
}

template <typename T>
std::vector<T> Read(DeviceSpan<T> span) {
  auto host = span.CopyDeviceToCpu();
  return {host.begin(), host.end()};
}

std::vector<uint8_t> Bytes(OrtValue& value, DeviceInterface& device) {
  return Read(ByteWrapTensor(device, value));
}

void CheckLogits(const std::vector<float>& actual, const std::vector<float>& expected, const Case& test) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index)
    EXPECT_NEAR(actual[index], expected[index], test.dtype == "fp16" ? 0.004f : 0.00002f);
}

std::vector<uint8_t> CachePrefix(Generator& generator, const char* name, size_t length) {
  auto* value = generator.state_->GetOutput(name);
  const auto shape = value->GetTensorTypeAndShapeInfo()->GetShape();
  const size_t row_bytes = static_cast<size_t>(shape[3]) *
                           Ort::SizeOf(value->GetTensorTypeAndShapeInfo()->GetElementType());
  const auto bytes = Bytes(*value, *generator.model_->p_device_kvcache_);
  std::vector<uint8_t> prefix;
  for (int64_t head = 0; head < shape[1]; ++head) {
    const auto begin = bytes.begin() + head * shape[2] * row_bytes;
    prefix.insert(prefix.end(), begin, begin + length * row_bytes);
  }
  return prefix;
}

void CheckPlacement(OrtValue& value, DeviceInterface& device, OrtMemoryInfoDeviceType kind) {
  const auto& actual = value.GetTensorMemoryInfo();
  const auto& expected = device.GetAllocator().GetInfo();
  EXPECT_EQ(actual.GetDeviceType(), kind);
  EXPECT_EQ(actual.GetDeviceId(), expected.GetDeviceId());
  EXPECT_EQ(actual.GetMemoryType(), expected.GetMemoryType());
  EXPECT_EQ(actual.GetAllocatorName(), expected.GetAllocatorName());
}

// Profiling permits only GQA's exact host length chain; no numerical fallback.
struct IgnoreJson : JSON::Element {
  void OnValue(std::string_view, JSON::Value) override {}
  JSON::Element& OnArray(std::string_view) override { return *this; }
  JSON::Element& OnObject(std::string_view) override { return *this; }
};

struct Profile : IgnoreJson {
  struct Event : IgnoreJson {
    struct Args : IgnoreJson {
      std::string provider, op;
      void OnValue(std::string_view name, JSON::Value value) override {
        if (name == "provider") provider = JSON::Get<std::string_view>(value);
        if (name == "op_name") op = JSON::Get<std::string_view>(value);
      }
    } args;
    std::string name, category;
    std::vector<Event>* events{};
    JSON::Element& OnObject(std::string_view key) override {
      return key == "args" ? static_cast<JSON::Element&>(args) : IgnoreJson::OnObject(key);
    }
    void OnValue(std::string_view key, JSON::Value value) override {
      if (key == "name") name = JSON::Get<std::string_view>(value);
      if (key == "cat") category = JSON::Get<std::string_view>(value);
    }
    void OnComplete(bool) override {
      if (category == "Node" && !args.provider.empty()) events->push_back(*this);
    }
  } event;
  std::vector<Event> events;
  JSON::Element& OnObject(std::string_view) override {
    event = Event{};
    event.events = &events;
    return event;
  }
};

void CheckProfile(OrtSession& session, const Case& test, const std::string& role) {
  std::ifstream file(session.EndProfiling());
  ASSERT_TRUE(file.good());
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  Profile profile;
  JSON::Parse(profile, text);
  const std::string provider = test.device == "cpu"    ? "CPUExecutionProvider"
                               : test.device == "cuda" ? "CUDAExecutionProvider"
                                                       : "WebGpuExecutionProvider";
  const std::map<std::string, std::string> metadata{
      {"metadata.mask_i32", "Cast"}, {"metadata.length_plus_one", "ReduceSum"}, {"metadata.seqlens_k", "Sub"}, {"metadata.mask_shape", "Shape"}, {"metadata.total_i64", "Gather"}, {"metadata.total_sl", "Cast"}};
  size_t numerical = 0, attention = 0;
  for (const auto& event : profile.events) {
    if (event.args.op == "MemcpyFromHost" || event.args.op == "MemcpyToHost") continue;
    auto name = event.name;
    const std::string suffix{"_kernel_time"};
    if (name.ends_with(suffix)) name.resize(name.size() - suffix.size());
    const auto found = metadata.find(name);
    if (role == "decoder" && found != metadata.end() && found->second == event.args.op) {
      EXPECT_TRUE(event.args.provider == provider || event.args.provider == "CPUExecutionProvider");
      continue;
    }
    EXPECT_EQ(event.args.provider, provider) << role << ": " << event.name << " " << event.args.op;
    ++numerical;
    attention += event.args.op == "GroupQueryAttention";
  }
  EXPECT_GT(numerical, 0u) << role;
  if (role == "decoder") EXPECT_GT(attention, 0u);
}

void CheckProfiles(MultiModalLanguageModel& model, const Case& test) {
  CheckProfile(*model.vision_session_, test, "vision");
  CheckProfile(*model.embedding_session_, test, "embedding");
  CheckProfile(*model.decoder_session_, test, "decoder");
}

// Only the model's execution-device pointer is wrapped; input/scoring/cache
// devices and every allocation/copy still delegate to the actual requested EP.
struct MediaFenceProbe : DeviceInterface {
  explicit MediaFenceProbe(Model& model) : model_{model}, device_{*model.p_device_} { model_.p_device_ = this; }
  ~MediaFenceProbe() override { model_.p_device_ = &device_; }
  DeviceType GetType() const override { return device_.GetType(); }
  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override { device_.InitOrt(api, allocator); }
  Ort::Allocator& GetAllocator() override { return device_.GetAllocator(); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override { return device_.GetMemoryInfo(); }
  std::string GetExecutionProviderName() const override { return device_.GetExecutionProviderName(); }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override { return device_.AllocateBase(size); }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override {
    return device_.WrapMemoryBase(memory, size);
  }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return device_.CreateGreedy(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return device_.CreateBeam(params); }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override { return device_.CreateKeyValueCache(state); }
  void Synchronize() override {
    ++fences;
    if (inspect) inspect();
    device_.Synchronize();
  }
  Model& model_;
  DeviceInterface& device_;
  std::function<void()> inspect;
  size_t fences{};
};

TEST(MultimodalDevice, MediaFencePrecedesOwnerReleaseAndNotDecode) {
  for (const auto& test : Cases()) {
    SCOPED_TRACE(test.Name());
    auto model = Load(test);
    const Config::RunOptions asynchronous{{"disable_synchronize_execution_providers", "1"}};
    model->config_->model.vision.run_options = asynchronous;
    model->config_->model.embedding.run_options = asynchronous;
    model->config_->model.decoder.run_options = asynchronous;
    auto generator = MakeGenerator(*model, true);
    auto& pipeline = Pipeline(*generator);
    MediaFenceProbe probe(*model);
    std::weak_ptr<Tensor> caller_pixels;
    probe.inspect = [&] {
      auto* vision = (pipeline.*Field(Vision{})).get();
      ASSERT_NE(vision, nullptr);
      EXPECT_FALSE(caller_pixels.expired());
      EXPECT_FALSE((pipeline.*Field(TurnInputs{})).empty());
      EXPECT_EQ(vision->GetOutput("image_features"), (pipeline.*Field(Embedding{}))->GetInput("image_features"));
      if (test.family != "phi3v") {
        const auto& owners = vision->*Field(ImageTemporaries{});
        EXPECT_EQ(owners.size(), test.family == "qwen2_5_vl" ? 6u : 4u);
        for (const auto& value : owners) EXPECT_NE(value, nullptr);
      }
    };
    for (int turn = 0; turn < 3; ++turn) {
      auto inputs = ImageTurn(test, static_cast<float>(turn + 1), test.family != "phi3v");
      caller_pixels = inputs.at("pixel_values");
      const auto before = probe.fences;
      generator->SetInputs(inputs);
      EXPECT_EQ(probe.fences, before + 1);
      inputs.clear();
      EXPECT_TRUE(caller_pixels.expired());
      EXPECT_FALSE(pipeline.*Field(Vision{}));
      EXPECT_TRUE((pipeline.*Field(TurnInputs{})).empty());
      for (int token = 0; token < 4; ++token) generator->GenerateNextToken();
      EXPECT_EQ(probe.fences, before + 1);
    }
    generator.reset();
    CheckProfiles(*model, test);
  }
}

TEST(MultimodalDevice, UnsupportedContinuationDeviceGuard) {
  for (auto device : {DeviceType::DML, DeviceType::QnnHtp, DeviceType::QnnGpu, DeviceType::AMDGPU})
    EXPECT_FALSE(SupportsContinuousDecoding(device));
}

TEST(MultimodalDevice, PortableValidationRejectsBeforeMutation) {
  for (const auto& test : Cases()) {
    for (bool shared : {false, true}) {
      SCOPED_TRACE(test.Name() + (shared ? "-shared" : "-dynamic"));
      auto model = Load(test);
      auto invalid_params = std::make_shared<GeneratorParams>(*model);
      invalid_params->search.max_length = 0;
      EXPECT_THROW((void)std::make_unique<Generator>(*model, *invalid_params), std::runtime_error);
      invalid_params->search.max_length = model->config_->model.context_length + 1;
      EXPECT_THROW((void)std::make_unique<Generator>(*model, *invalid_params), std::runtime_error);
      auto generator = MakeGenerator(*model, shared);
      auto reference = MakeGenerator(*model, shared);
      for (auto* current : {generator.get(), reference.get()}) {
        current->SetInputs(ImageTurn(test, 1));
        for (int token = 0; token < 3; ++token) current->GenerateNextToken();
      }
      const auto logits = Read(generator->GetLogits());
      const auto sequence = Read(generator->GetSequence(0));
      auto& pipeline = Pipeline(*generator);
      auto* decoder = &DecoderOf(*generator);
      auto* cache = &CacheOf(*generator);
      const auto boundary = pipeline.*Field(Boundary{});
      const auto length = cache->*Field(Length{});
      const auto key = CachePrefix(*generator, "present.0.key", length);
      const auto value = CachePrefix(*generator, "present.0.value", length);
      auto* key_pointer = generator->state_->GetOutput("present.0.key")->GetTensorMutableRawData();
      auto* embedding_pointer = generator->state_->GetInput("inputs_embeds")->GetTensorMutableRawData();
      const auto unchanged = [&] {
        EXPECT_EQ(Read(generator->GetSequence(0)), sequence);
        EXPECT_EQ(Read(generator->GetLogits()), logits);
        EXPECT_EQ(pipeline.*Field(Boundary{}), boundary);
        EXPECT_FALSE(pipeline.*Field(Failed{}));
        EXPECT_EQ(&DecoderOf(*generator), decoder);
        EXPECT_EQ(&CacheOf(*generator), cache);
        EXPECT_EQ(cache->*Field(Length{}), length);
        EXPECT_EQ(CachePrefix(*generator, "present.0.key", length), key);
        EXPECT_EQ(CachePrefix(*generator, "present.0.value", length), value);
        EXPECT_EQ(generator->state_->GetOutput("present.0.key")->GetTensorMutableRawData(), key_pointer);
        EXPECT_EQ(generator->state_->GetInput("inputs_embeds")->GetTensorMutableRawData(), embedding_pointer);
      };
      // These errors are validated on the host before any vision/decoder Run.
      // Do not inject invalid GPU Gather indices: those can poison the EP context.
      auto invalid = ImageTurn(test, 7);
      invalid.erase("input_ids");
      EXPECT_THROW(generator->SetInputs(invalid), std::runtime_error);
      unchanged();
      invalid = ImageTurn(test, 7);
      invalid["num_image_tokens"] = CpuTensor<int32_t>({1}, {1});
      EXPECT_THROW(generator->SetInputs(invalid), std::runtime_error);
      unchanged();
      invalid = ImageTurn(test, 7);
      const auto capacity = model->config_->search.max_length;
      invalid["input_ids"] = CpuTensor<int32_t>({1, static_cast<int64_t>(capacity + 1)},
                                                std::vector<int32_t>(capacity + 1, 2));
      EXPECT_THROW(generator->SetInputs(invalid), std::runtime_error);
      unchanged();
      EXPECT_THROW(generator->RewindToLength(boundary), std::runtime_error);
      unchanged();
      for (auto* current : {generator.get(), reference.get()}) {
        current->SetInputs(ImageTurn(test, 7));
        for (int token = 0; token < 3; ++token) current->GenerateNextToken();
      }
      EXPECT_EQ(Read(generator->GetSequence(0)), Read(reference->GetSequence(0)));
      CheckLogits(Read(generator->GetLogits()), Read(reference->GetLogits()), test);
      generator.reset();
      reference.reset();
      invalid_params.reset();
      CheckProfiles(*model, test);
    }
  }
}

TEST(MultimodalDevice, RetainsDecoderCacheAndReleasesTurnOwners) {
  for (const auto& test : Cases()) {
    for (bool shared : {false, true}) {
      SCOPED_TRACE(test.Name() + (shared ? "-shared" : "-dynamic"));
      auto model = Load(test);
      auto generator = MakeGenerator(*model, shared);
      auto& pipeline = Pipeline(*generator);
      auto* decoder = &DecoderOf(*generator);
      auto* cache = &CacheOf(*generator);
      auto* session = model->decoder_session_.get();
      auto* vision_session = model->vision_session_.get();
      auto* embedding_session = model->embedding_session_.get();
      EXPECT_FALSE(decoder->*Field(Recurrent{}));
      EXPECT_EQ(dynamic_cast<SharedKeyValueCache*>(cache) != nullptr, shared);
      EXPECT_EQ(dynamic_cast<DynamicKeyValueCache*>(cache) != nullptr, !shared);
      void* shared_key{};
      size_t retained_length = 0;
      std::vector<uint8_t> key_prefix, value_prefix;
      for (int turn = 0; turn < 3; ++turn) {
        auto inputs = ImageTurn(test, 1.0f + turn * 6, turn == 1 && test.family != "phi3v");
        std::weak_ptr<Tensor> caller_pixels = inputs.at("pixel_values");
        generator->SetInputs(inputs);
        inputs.clear();
        EXPECT_TRUE(caller_pixels.expired());
        EXPECT_TRUE(generator->extra_inputs_.empty());
        EXPECT_TRUE((pipeline.*Field(TurnInputs{})).empty());
        EXPECT_FALSE(pipeline.*Field(Vision{}));
        EXPECT_FALSE(pipeline.*Field(Failed{}));
        EXPECT_EQ(&DecoderOf(*generator), decoder);
        EXPECT_EQ(&CacheOf(*generator), cache);
        EXPECT_EQ(model->decoder_session_.get(), session);
        EXPECT_EQ(model->vision_session_.get(), vision_session);
        EXPECT_EQ(model->embedding_session_.get(), embedding_session);
        EXPECT_EQ(pipeline.*Field(Boundary{}), generator->TokenCount());
        auto* features = generator->state_->GetInput("image_features");
        ASSERT_NE(features, nullptr);
        const auto device_kind = test.Gpu() ? OrtMemoryInfoDeviceType_GPU : OrtMemoryInfoDeviceType_CPU;
        const auto input_kind = test.device == "cuda" ? OrtMemoryInfoDeviceType_GPU : OrtMemoryInfoDeviceType_CPU;
        CheckPlacement(*features, *model->p_device_, device_kind);
        CheckPlacement(*generator->state_->GetInput("inputs_embeds"), *model->p_device_inputs_, input_kind);
        CheckPlacement(*generator->state_->GetOutput("logits"), *model->p_device_logits_, input_kind);
        EXPECT_EQ(generator->state_->GetInput("inputs_embeds"), generator->state_->GetOutput("inputs_embeds"));
        for (const auto* kind : {"key", "value"}) {
          auto* past = generator->state_->GetInput((std::string("past_key_values.0.") + kind).c_str());
          auto* present = generator->state_->GetOutput((std::string("present.0.") + kind).c_str());
          CheckPlacement(*present, *model->p_device_kvcache_, device_kind);
          EXPECT_EQ(past->GetTensorMutableRawData() == present->GetTensorMutableRawData(), shared);
        }
        auto* key = generator->state_->GetOutput("present.0.key");
        if (shared && shared_key) EXPECT_EQ(key->GetTensorMutableRawData(), shared_key);
        shared_key = key->GetTensorMutableRawData();
        if (retained_length) {
          EXPECT_EQ(CachePrefix(*generator, "present.0.key", retained_length), key_prefix);
          EXPECT_EQ(CachePrefix(*generator, "present.0.value", retained_length), value_prefix);
        }
        retained_length = cache->*Field(Length{});
        EXPECT_EQ(retained_length, generator->TokenCount());
        key_prefix = CachePrefix(*generator, "present.0.key", retained_length);
        value_prefix = CachePrefix(*generator, "present.0.value", retained_length);
        for (int token = 0; token < 4; ++token) generator->GenerateNextToken();
        const std::vector<int32_t> text{5, 9};
        generator->AppendTokens(text);
      }
      generator.reset();
      CheckProfiles(*model, test);
    }
  }
}

TEST(MultimodalDevice, PerImageOutputsTransferOwnershipWithoutHostViews) {
  for (const auto& test : Cases()) {
    if (test.family == "phi3v") continue;
    SCOPED_TRACE(test.Name());
    auto model = Load(test);
    auto params = std::make_shared<GeneratorParams>(*model);
    auto vision = CreateVisionState(*model, *params);
    auto inputs = ImageTurn(test, 1, true);
    std::vector<ExtraInput> extras;
    for (const auto& [name, value] : inputs)
      if (name != "input_ids") extras.push_back({name, value});
    vision->SetExtraInputs(extras, 2, 8);
    DeviceSpan<int32_t> tokens;
    vision->Run(14, tokens);
    auto& owned = vision.get()->*Field(ImageTemporaries{});
    const bool qwen = test.family == "qwen2_5_vl";
    ASSERT_EQ(owned.size(), qwen ? 6u : 4u);
    auto* combined = vision->GetOutput("image_features");
    for (size_t index : qwen ? std::vector<size_t>{2, 5} : std::vector<size_t>{1, 3}) {
      CheckPlacement(*owned[index], *model->p_device_,
                     test.Gpu() ? OrtMemoryInfoDeviceType_GPU : OrtMemoryInfoDeviceType_CPU);
      EXPECT_NE(owned[index]->GetTensorMutableRawData(), combined->GetTensorMutableRawData());
    }
    if (!qwen) {
      for (size_t index : {0u, 2u}) {
        EXPECT_EQ(owned[index]->GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_CPU);
        EXPECT_NE(owned[index]->GetTensorMutableRawData(), inputs.at("pixel_values")->GetMutableRawData());
      }
    }
    auto embedding = std::make_unique<EmbeddingState>(*model, *params);
    embedding->SetExtraInputs(2, 8, 0);
    auto& source = *(vision.get()->*Field(VisionFeatures{}));
    auto& target = *(embedding.get()->*Field(EmbeddingFeatures{}));
    target.ReuseFeaturesBuffer(source);
    EXPECT_EQ(source.Get(), nullptr);
    EXPECT_EQ(target.Get(), combined);
    EXPECT_EQ(embedding->GetInput("image_features"), combined);
    // This is the production ownership boundary, not a per-token stress readback.
    model->p_device_->Synchronize();
    const auto expected = Bytes(*combined, *model->p_device_);
    vision.reset();
    inputs.clear();
    extras.clear();
    EXPECT_EQ(Bytes(*embedding->GetInput("image_features"), *model->p_device_), expected);
    embedding.reset();
    // This isolated test executes vision only; full pipeline tests audit all roles.
    CheckProfile(*model->vision_session_, test, "vision");
  }
}

TEST(MultimodalDevice, UnequalGridNoReadbackAllocationStress) {
  for (const auto& test : Cases()) {
    if (test.family == "phi3v") continue;
    for (bool shared : {false, true}) {
      SCOPED_TRACE(test.Name() + (shared ? "-shared" : "-dynamic"));
      auto model = Load(test);
      const Config::RunOptions asynchronous{{"disable_synchronize_execution_providers", "1"}};
      model->config_->model.vision.run_options = asynchronous;
      model->config_->model.embedding.run_options = asynchronous;
      model->config_->model.decoder.run_options = asynchronous;
      auto generator = MakeGenerator(*model, shared);
      auto* decoder = &DecoderOf(*generator);
      std::vector<size_t> prompts;
      for (int turn = 0; turn < 8; ++turn) {
        auto inputs = ImageTurn(test, static_cast<float>(turn + 1), turn % 2 == 0);
        prompts.push_back(inputs.at("input_ids")->GetElementCount());
        generator->SetInputs(inputs);
        inputs.clear();
        std::vector<std::unique_ptr<Generator>> churn;
        for (int allocation = 0; allocation < 3; ++allocation) {
          auto other = MakeGenerator(*model, shared);
          other->AppendTokens(std::vector<int32_t>{4, 6, 8});
          other->GenerateNextToken();
          churn.push_back(std::move(other));
        }
        churn.clear();
        for (int token = 0; token < 4; ++token) generator->GenerateNextToken();
      }
      // No logits/sequence/cache/profile readback occurs until the turn loop ends.
      EXPECT_EQ(&DecoderOf(*generator), decoder);
      const auto sequence = Read(generator->GetSequence(0));
      const auto actual = Read(generator->GetLogits());
      auto reference = MakeGenerator(*model, !shared);
      size_t offset = 0;
      for (int turn = 0; turn < 8; ++turn) {
        reference->SetInputs(ImageTurn(test, static_cast<float>(turn + 1), turn % 2 == 0));
        offset += prompts[turn];
        reference->AppendTokens(cpu_span<const int32_t>{sequence.data() + offset, 4});
        offset += 4;
      }
      EXPECT_EQ(Read(reference->GetSequence(0)), sequence);
      CheckLogits(actual, Read(reference->GetLogits()), test);
      generator.reset();
      reference.reset();
      CheckProfiles(*model, test);
    }
  }
}

TEST(MultimodalDevice, ActualCapturedDecodeRejectsImageBeforeMutation) {
  bool exercised = false;
  for (const auto& test : Cases()) {
    if (!test.Gpu() || test.family != "phi3v") continue;
    exercised = true;
    SCOPED_TRACE(test.Name());
    auto model = Load(test, true);
    auto eager_model = Load(test);
    auto generator = MakeGenerator(*model, true);
    auto reference = MakeGenerator(*eager_model, true);
    auto& pipeline = Pipeline(*generator);
    auto& decoder = DecoderOf(*generator);
    ASSERT_TRUE(decoder.params_->use_graph_capture);
    for (auto* current : {generator.get(), reference.get()}) {
      current->SetInputs(ImageTurn(test, 1));
      for (int token = 0; token < 8; ++token) current->GenerateNextToken();
    }
    ASSERT_FALSE(Captures(decoder, GraphIds{}).empty());
    ASSERT_EQ(decoder.*Field(CaptureSession{}), model->decoder_session_.get());
    const auto graph_ids = Captures(decoder, GraphIds{});
    const auto boundary = pipeline.*Field(Boundary{});
    const auto logits = Read(generator->GetLogits());
    auto* cache = &CacheOf(*generator);
    const auto length = cache->*Field(Length{});
    const auto sequence = Read(generator->GetSequence(0));
    const auto key = Bytes(*generator->state_->GetOutput("present.0.key"), *model->p_device_kvcache_);
    const auto value = Bytes(*generator->state_->GetOutput("present.0.value"), *model->p_device_kvcache_);
    auto* key_pointer = generator->state_->GetOutput("present.0.key")->GetTensorMutableRawData();
    auto* embedding_pointer = generator->state_->GetInput("inputs_embeds")->GetTensorMutableRawData();
    try {
      generator->SetInputs(ImageTurn(test, 7));
      FAIL() << "Captured decoder accepted a later image";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string(error.what()).find("graph capture"), std::string::npos);
    }
    EXPECT_EQ(pipeline.*Field(Boundary{}), boundary);
    EXPECT_EQ(&CacheOf(*generator), cache);
    EXPECT_EQ(cache->*Field(Length{}), length);
    EXPECT_EQ(&DecoderOf(*generator), &decoder);
    EXPECT_FALSE(pipeline.*Field(Failed{}));
    EXPECT_EQ(Read(generator->GetSequence(0)), sequence);
    EXPECT_EQ(Read(generator->GetLogits()), logits);
    EXPECT_EQ(Bytes(*generator->state_->GetOutput("present.0.key"), *model->p_device_kvcache_), key);
    EXPECT_EQ(Bytes(*generator->state_->GetOutput("present.0.value"), *model->p_device_kvcache_), value);
    EXPECT_EQ(generator->state_->GetOutput("present.0.key")->GetTensorMutableRawData(), key_pointer);
    EXPECT_EQ(generator->state_->GetInput("inputs_embeds")->GetTensorMutableRawData(), embedding_pointer);
    EXPECT_EQ(Captures(decoder, GraphIds{}), graph_ids);
    for (int token = 0; token < 4; ++token) {
      generator->GenerateNextToken();
      reference->GenerateNextToken();
    }
    EXPECT_EQ(Read(generator->GetSequence(0)), Read(reference->GetSequence(0)));
    CheckLogits(Read(generator->GetLogits()), Read(reference->GetLogits()), test);
    generator.reset();
    reference.reset();
    CheckProfiles(*model, test);
    CheckProfiles(*eager_model, test);
  }
  if (!exercised) GTEST_SKIP() << "CPU guard is not actual CUDA/WebGPU graph capture coverage";
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  std::filesystem::path ep_dir;
  std::string requested = std::getenv("ORTGENAI_MULTIMODAL_TEST_EPS") ? std::getenv("ORTGENAI_MULTIMODAL_TEST_EPS") : "cpu";
  if (const char* root = std::getenv("ORTGENAI_MULTIMODAL_GQA_MODELS")) model_root = root;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--ep_dir" && index + 1 < argc) ep_dir = argv[++index];
    if (argument == "--model_root" && index + 1 < argc) model_root = argv[++index];
    if (argument == "--devices" && index + 1 < argc) requested = argv[++index];
  }
  std::stringstream list(requested);
  std::string device;
  while (std::getline(list, device, ',')) {
    if (device != "cpu" && device != "cuda" && device != "webgpu") {
      std::cerr << "GQA fixture has no established kernel contract for EP: " << device << '\n';
      return 2;
    }
    if (std::find(devices.begin(), devices.end(), device) == devices.end()) devices.push_back(device);
  }
  test_ep::EpRegistrar registrar;
  registrar.DiscoverFromDirectory(ep_dir);
  registrar.RegisterAll();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
