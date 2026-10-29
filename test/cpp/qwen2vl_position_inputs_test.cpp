// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "models/io/default_position_inputs.h"
#include "models/io/kv_cache.h"
#include "models/io/position_inputs.h"
#include "models/io/qwen_vl_position_inputs.h"
#include "models/io/windowed_position_inputs.h"
#include "search.h"
#include "test_utils.h"

namespace Generators::test {
namespace {

template <typename Fn>
std::string CaptureThrowMessage(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

struct PositionInputsTestState : State {
  using State::State;

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override {
    return {};
  }
};

class StandardPositionInputsTest : public testing::Test {
 protected:
  void SetUp() override {
    model_ = CreateModel(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    params_ = std::make_shared<GeneratorParams>(*model_);
    state_ = std::make_unique<PositionInputsTestState>(*params_, *model_);
    sequence_lengths_ = model_->p_device_->Allocate<int32_t>(params_->search.batch_size);
  }

  void Use3DPositionIds() {
    session_ = model_->CreateSession(GetOrtEnv(), "../../qwen3-vl/dummy_text.onnx", model_->session_options_.get());
    model_->session_info_ = SessionInfo{};
    model_->session_info_.Add(*session_);
    ASSERT_EQ(model_->session_info_.GetInputShape("position_ids").size(), 3u);
  }

  std::unique_ptr<PositionInputs> CreatePositionInputs() {
    return CreateStandardPositionInputs(*state_, sequence_lengths_, model_->config_->model.decoder.inputs.attention_mask);
  }

  static constexpr std::array<const char*, 5> qwen_model_types_{
      "qwen2_5_vl", "qwen3_vl", "qwen3_5", "qwen3_5_text", "qwen3_5_moe_text"};
  std::shared_ptr<Model> model_;
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<PositionInputsTestState> state_;
  DeviceSpan<int32_t> sequence_lengths_;
  std::unique_ptr<OrtSession> session_;
};

// Exercise provider policy and CPU fallback with CPU-backed storage. WebGPU's
// staged policy must never call the device's unpadded-prefix mask update.
struct PositionInputsTestDevice : DeviceInterface {
  PositionInputsTestDevice(Model& model, DeviceType type, bool static_shared_mask)
      : model_{model}, inner_{model.p_device_inputs_}, type_{type}, static_shared_mask_{static_shared_mask} {
    model_.p_device_inputs_ = this;
  }
  ~PositionInputsTestDevice() override { model_.p_device_inputs_ = inner_; }
  DeviceType GetType() const override { return type_; }
  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override { inner_->InitOrt(api, allocator); }
  Ort::Allocator& GetAllocator() override { return inner_->GetAllocator(); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override { return inner_->GetMemoryInfo(); }
  std::string GetExecutionProviderName() const override { return inner_->GetExecutionProviderName(); }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override { return inner_->AllocateBase(size); }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override { return inner_->WrapMemoryBase(memory, size); }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return inner_->CreateGreedy(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return inner_->CreateBeam(params); }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override { return inner_->CreateKeyValueCache(state); }
  void Synchronize() override { inner_->Synchronize(); }
  bool ShouldUseStaticPositionInputsForSharedBuffers(const Config::Model&) const override { return static_shared_mask_; }
  bool UpdateAttentionMask(void*, void*, int, int, int, int, bool, ONNXTensorElementDataType) override {
    ++mask_updates;
    return false;
  }

  int mask_updates{};
  Model& model_;
  DeviceInterface* inner_;
  DeviceType type_;
  bool static_shared_mask_;
};

template <typename T>
class PositionInputBuffersTest : public StandardPositionInputsTest {};
using PositionInputTypes = testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(PositionInputBuffersTest, PositionInputTypes);

TYPED_TEST(PositionInputBuffersTest, PositionStorageHandlesRanksBeamsAndCaptureLengths) {
  const auto type = Ort::TypeToTensorType<TypeParam>;
  this->params_->max_graph_capture_length = 2;
  for (bool capture : {false, true}) {
    this->params_->use_graph_capture = capture;
    for (int rank : {2, 3}) {
      for (int beams : {1, 2}) {
        SCOPED_TRACE(testing::Message() << capture << ", rank=" << rank << ", beams=" << beams);
        auto buffer = CreatePositionIdsBuffer(*this->params_);
        Tensor tensor{this->model_->p_device_inputs_, type};
        const void* captured_address = nullptr;
        for (int length : {1, 2, 5, 1, 2}) {
          std::vector<int64_t> shape = rank == 2 ? std::vector<int64_t>{2, length} : std::vector<int64_t>{3, 2, length};
          auto values = OrtValue::CreateTensor(this->model_->allocator_cpu_, shape, type);
          const int rows = rank == 2 ? 2 : 6;
          auto* data = values->template GetTensorMutableData<TypeParam>();
          for (int row = 0; row < rows; ++row)
            for (int col = 0; col < length; ++col)
              data[row * length + col] = static_cast<TypeParam>(row * 10 + col);
          buffer->Assign(tensor, *this->model_, values, beams);
          shape[rank - 2] *= beams;
          EXPECT_EQ(tensor.GetShape(), shape);
          for (int row = 0; row < rows; ++row)
            for (int beam = 0; beam < beams; ++beam)
              for (int col = 0; col < length; ++col)
                EXPECT_EQ(tensor.GetData<TypeParam>()[(row * beams + beam) * length + col], row * 10 + col);
          if (capture && length <= 2) {
            if (!captured_address)
              captured_address = tensor.GetRawData();
            EXPECT_EQ(tensor.GetRawData(), captured_address);
          }
        }
      }
    }
  }
}

TYPED_TEST(PositionInputBuffersTest, StaticMasksPreservePaddingAcrossUpdatesRewindAndReset) {
  const auto type = Ort::TypeToTensorType<TypeParam>;
  this->params_->search.max_length = 8;
  for (DeviceType device_type : {DeviceType::CPU, DeviceType::WEBGPU}) {
    PositionInputsTestDevice device{*this->model_, device_type, true};
    for (bool capture : {false, true}) {
      SCOPED_TRACE(testing::Message() << static_cast<int>(device_type) << ", capture=" << capture);
      this->params_->use_graph_capture = capture;
      this->params_->search.past_present_share_buffer = !capture;
      auto buffer = CreateAttentionMaskBuffer(*this->model_, *this->params_);
      Tensor tensor{&device, type};
      const void* address = nullptr;
      for (int run = 0; run < 2; ++run) {
        std::array<int64_t, 2> shape{2, 3};
        auto values = OrtValue::CreateTensor(this->model_->allocator_cpu_, shape, type);
        const std::array<TypeParam, 6> prompt{static_cast<TypeParam>(run), 1, 1, 1, 1, 0};
        std::copy(prompt.begin(), prompt.end(), values->template GetTensorMutableData<TypeParam>());
        buffer->Initialize(tensor, values, 2, shape);
        EXPECT_EQ(shape, (std::array<int64_t, 2>{4, 8}));
        if (!address)
          address = tensor.GetRawData();
        auto check = [&](int length) {
          EXPECT_EQ(tensor.GetRawData(), address);
          const auto* data = tensor.GetData<TypeParam>();
          for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 8; ++col)
              EXPECT_EQ(data[row * 8 + col], col >= length ? 0 : col < 3 ? prompt[(row / 2) * 3 + col]
                                                                         : 1);
        };
        check(3);
        buffer->Update(tensor, shape, 5, 2);
        check(5);
        buffer->Update(tensor, shape, 6, 1);
        check(6);
        buffer->Rewind(tensor, shape, 3);
        check(3);
        buffer->Update(tensor, shape, 4, 1);
        check(4);
        EXPECT_THROW(buffer->Update(tensor, shape, 9, 1), std::runtime_error);
        EXPECT_THROW(buffer->Update(tensor, shape, 4, 0), std::runtime_error);
        EXPECT_THROW(buffer->Rewind(tensor, shape, 9), std::runtime_error);
        check(4);
      }
      if (device_type == DeviceType::WEBGPU)
        EXPECT_EQ(device.mask_updates, 0);
      else
        EXPECT_GT(device.mask_updates, 0);
      std::array<int64_t, 2> oversized_shape{2, 9};
      auto oversized = OrtValue::CreateTensor(this->model_->allocator_cpu_, oversized_shape, type);
      EXPECT_THROW(buffer->Initialize(tensor, oversized, 2, oversized_shape), std::runtime_error);
    }
  }
}

TYPED_TEST(PositionInputBuffersTest, DynamicMasksRetainDecoderSpecificPaddingBehavior) {
  const auto type = Ort::TypeToTensorType<TypeParam>;
  this->params_->use_graph_capture = false;
  for (bool shared : {false, true}) {
    // Shared KV alone does not require a static mask unless the provider requests it.
    this->params_->search.past_present_share_buffer = shared;
    PositionInputsTestDevice device{*this->model_, DeviceType::CPU, false};
    for (auto update : {DynamicMaskUpdate::PreservePadding, DynamicMaskUpdate::FillOnes}) {
      auto buffer = CreateAttentionMaskBuffer(*this->model_, *this->params_, update);
      Tensor tensor{&device, type};
      std::array<int64_t, 2> shape{2, 3};
      auto values = OrtValue::CreateTensor(this->model_->allocator_cpu_, shape, type);
      const std::array<TypeParam, 6> prompt{1, 1, 0, 1, 1, 1};
      std::copy(prompt.begin(), prompt.end(), values->template GetTensorMutableData<TypeParam>());
      buffer->Initialize(tensor, values, 2, shape);
      EXPECT_EQ(shape, (std::array<int64_t, 2>{4, 3}));
      for (int length : {4, 5}) {
        buffer->Update(tensor, shape, length, 1);
        EXPECT_EQ(tensor.GetShape(), (std::vector<int64_t>{4, length}));
        for (int row = 0; row < 4; ++row)
          for (int col = 0; col < length; ++col)
            EXPECT_EQ(tensor.GetData<TypeParam>()[row * length + col],
                      update == DynamicMaskUpdate::PreservePadding && row < 2 && col == 2 ? 0 : 1);
      }
    }
  }
}

class QwenPositionValuesTest : public StandardPositionInputsTest, public testing::WithParamInterface<int64_t> {
 protected:
  void SetUp() override {
    StandardPositionInputsTest::SetUp();
    if (GetParam() == 4) {
      session_ = model_->CreateSession(GetOrtEnv(), "../../qwen-position-inputs/four-row.onnx", model_->session_options_.get());
      model_->session_info_ = SessionInfo{};
      model_->session_info_.Add(*session_);
    } else {
      ASSERT_NO_FATAL_FAILURE(Use3DPositionIds());
    }
    ASSERT_EQ(model_->session_info_.GetInputShape("position_ids"),
              (std::vector<int64_t>{GetParam(), -1, -1}));
    ASSERT_EQ(model_->session_info_.GetInputDataType("position_ids"), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    ASSERT_EQ(model_->session_info_.GetInputDataType("attention_mask"), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    model_->config_->model.type = GetParam() == 4 ? "qwen3_5" : "qwen3_vl";
    model_->config_->model.pad_token_id = 0;
    model_->config_->model.image_token_id = 100;
    model_->config_->model.video_token_id = 101;
    model_->config_->model.vision_start_token_id = 102;
    model_->config_->model.vision.spatial_merge_size = 2;
    params_->search.batch_size = 1;
    params_->search.num_beams = 1;
  }

  DeviceSpan<int32_t> UploadTokens(std::span<const int32_t> values) {
    auto tokens = model_->p_device_->Allocate<int32_t>(values.size());
    std::copy(values.begin(), values.end(), tokens.CpuSpan().begin());
    tokens.CopyCpuToDevice();
    return tokens;
  }

  void ExpectPositionIds(int64_t sequence_length, const std::vector<int64_t>& expected) {
    const auto* tensor = state_->GetInput(model_->config_->model.decoder.inputs.position_ids.c_str());
    ASSERT_NE(tensor, nullptr);
    const auto info = tensor->GetTensorTypeAndShapeInfo();
    ASSERT_EQ(info->GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    ASSERT_EQ(info->GetShape(), (std::vector<int64_t>{GetParam(), 1, sequence_length}));
    ASSERT_EQ(info->GetElementCount(), expected.size());
    const auto* data = tensor->GetTensorData<int64_t>();
    EXPECT_EQ(std::vector<int64_t>(data, data + expected.size()), expected);
  }
};

}  // namespace

TEST_P(QwenPositionValuesTest, ImagePrefillAndSuccessiveDecode) {
  auto inputs = CreatePositionInputs();
  auto* qwen_inputs = dynamic_cast<Qwen2VLPositionInputs*>(inputs.get());
  ASSERT_NE(qwen_inputs, nullptr);
  auto grid = OrtValue::CreateTensor(model_->allocator_cpu_, std::array<int64_t, 2>{1, 3},
                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  const std::array<int64_t, 3> grid_values{1, 4, 6};
  std::copy(grid_values.begin(), grid_values.end(), grid->GetTensorMutableData<int64_t>());
  qwen_inputs->SetGridTensors(std::make_shared<Tensor>(std::move(grid)), nullptr, nullptr);
  inputs->Add();

  const std::array<int32_t, 10> prompt{10, 102, 100, 100, 100, 100, 100, 100, 103, 11};
  auto tokens = UploadTokens(prompt);
  inputs->Update(tokens, 10, 10);
  std::vector<int64_t> expected_prefill{
      0, 1, 2, 2, 2, 2, 2, 2, 5, 6,
      0, 1, 2, 2, 2, 3, 3, 3, 5, 6,
      0, 1, 2, 3, 4, 2, 3, 4, 5, 6};
  if (GetParam() == 4) {
    expected_prefill.insert(expected_prefill.begin(), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  }
  {
    SCOPED_TRACE("image prefill");
    ASSERT_NO_FATAL_FAILURE(ExpectPositionIds(10, expected_prefill));
  }

  tokens = UploadTokens(std::array<int32_t, 1>{12});
  inputs->Update(tokens, 11, 1);
  {
    SCOPED_TRACE("first decode, cached RoPE delta -3");
    ASSERT_NO_FATAL_FAILURE(ExpectPositionIds(1, GetParam() == 4 ? std::vector<int64_t>{10, 7, 7, 7}
                                                                 : std::vector<int64_t>{7, 7, 7}));
  }
  tokens = UploadTokens(std::array<int32_t, 1>{13});
  inputs->Update(tokens, 12, 1);
  {
    SCOPED_TRACE("second decode, cached RoPE delta -3");
    ASSERT_NO_FATAL_FAILURE(ExpectPositionIds(1, GetParam() == 4 ? std::vector<int64_t>{11, 8, 8, 8}
                                                                 : std::vector<int64_t>{8, 8, 8}));
  }
}

INSTANTIATE_TEST_SUITE_P(RopeRows, QwenPositionValuesTest, testing::Values<int64_t>(3, 4));

TEST_F(StandardPositionInputsTest, QwenVLWith3DPositionIdsUsesMrope) {
  Use3DPositionIds();
  for (const auto* model_type : qwen_model_types_) {
    SCOPED_TRACE(model_type);
    model_->config_->model.type = model_type;
    auto inputs = CreatePositionInputs();
    EXPECT_NE(dynamic_cast<Qwen2VLPositionInputs*>(inputs.get()), nullptr);
  }
}

TEST_F(StandardPositionInputsTest, QwenVLWith2DPositionIdsUsesDefault) {
  ASSERT_EQ(model_->session_info_.GetInputShape("position_ids").size(), 2u);
  for (const auto* model_type : qwen_model_types_) {
    SCOPED_TRACE(model_type);
    model_->config_->model.type = model_type;
    auto inputs = CreatePositionInputs();
    EXPECT_NE(dynamic_cast<DefaultPositionInputs*>(inputs.get()), nullptr);
  }
}

TEST_F(StandardPositionInputsTest, QwenVLWithoutConfiguredPositionIdsUsesDefault) {
  model_->config_->model.decoder.inputs.position_ids = "missing_position_ids";
  ASSERT_FALSE(model_->session_info_.HasInput("missing_position_ids"));
  for (const auto* model_type : qwen_model_types_) {
    SCOPED_TRACE(model_type);
    model_->config_->model.type = model_type;
    auto inputs = CreatePositionInputs();
    EXPECT_NE(dynamic_cast<DefaultPositionInputs*>(inputs.get()), nullptr);
  }
}

TEST_F(StandardPositionInputsTest, QwenVLWithEmptyPositionIdsNameUsesDefault) {
  model_->config_->model.type = "qwen3_5_text";
  model_->config_->model.decoder.inputs.position_ids.clear();
  auto inputs = CreatePositionInputs();
  EXPECT_NE(dynamic_cast<DefaultPositionInputs*>(inputs.get()), nullptr);
}

TEST_F(StandardPositionInputsTest, UsesConfiguredPositionIdsNameForRankCheck) {
  Use3DPositionIds();
  model_->config_->model.type = "qwen3_5_text";
  model_->config_->model.decoder.inputs.position_ids = "attention_mask";
  ASSERT_EQ(model_->session_info_.GetInputShape("attention_mask").size(), 2u);
  auto inputs = CreatePositionInputs();
  EXPECT_NE(dynamic_cast<DefaultPositionInputs*>(inputs.get()), nullptr);
}

TEST_F(StandardPositionInputsTest, NonQwenModelWith3DPositionIdsUsesDefault) {
  Use3DPositionIds();
  model_->config_->model.type = "llama";
  auto inputs = CreatePositionInputs();
  EXPECT_NE(dynamic_cast<DefaultPositionInputs*>(inputs.get()), nullptr);
}

TEST_F(StandardPositionInputsTest, QwenVLWith2DPositionIdsUsesSlidingWindowWhenEnabled) {
  model_->config_->model.type = "qwen3_5_text";
  auto& sliding_window = model_->config_->model.decoder.sliding_window.emplace();
  sliding_window.window_size = 8;
  sliding_window.slide_inputs = true;
  auto inputs = CreatePositionInputs();
  EXPECT_NE(dynamic_cast<WindowedPositionInputs*>(inputs.get()), nullptr);
}

TEST_F(StandardPositionInputsTest, QwenVLWith3DPositionIdsPrefersMropeOverSlidingWindow) {
  Use3DPositionIds();
  model_->config_->model.type = "qwen3_5_text";
  auto& sliding_window = model_->config_->model.decoder.sliding_window.emplace();
  sliding_window.window_size = 8;
  sliding_window.slide_inputs = true;
  auto inputs = CreatePositionInputs();
  EXPECT_NE(dynamic_cast<Qwen2VLPositionInputs*>(inputs.get()), nullptr);
}

TEST_F(StandardPositionInputsTest, CapturedMropeInputsStayStableAcrossDecodeAndRewind) {
  Use3DPositionIds();
  model_->config_->model.type = "qwen3_5_text";
  model_->config_->model.pad_token_id = 0;
  params_->use_graph_capture = true;
  params_->max_graph_capture_length = 2;
  params_->search.max_length = 16;
  auto inputs = CreatePositionInputs();
  inputs->Add();
  auto tokens = model_->p_device_->Allocate<int32_t>(3);
  std::copy_n(std::array<int32_t, 3>{0, 10, 20}.data(), 3, tokens.CpuSpan().data());
  inputs->Update(tokens, 3, 3);

  auto* mask = state_->GetInput("attention_mask");
  ASSERT_EQ(mask->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{1, 16}));
  const void* mask_address = mask->GetTensorRawData();
  EXPECT_EQ(mask->GetTensorData<int64_t>()[0], 0);
  EXPECT_EQ(mask->GetTensorData<int64_t>()[2], 1);
  EXPECT_EQ(mask->GetTensorData<int64_t>()[3], 0);

  auto next_token = tokens.subspan(1, 1);
  inputs->Update(next_token, 4, 1);
  const void* position_address = state_->GetInput("position_ids")->GetTensorRawData();
  auto expect_inputs = [&](int sequence_length, int first_position, int valid_length) {
    auto* positions = state_->GetInput("position_ids");
    EXPECT_EQ(positions->GetTensorRawData(), position_address);
    EXPECT_EQ(positions->GetTensorTypeAndShapeInfo()->GetShape(),
              (std::vector<int64_t>{3, 1, sequence_length}));
    for (int dim = 0; dim < 3; ++dim) {
      for (int s = 0; s < sequence_length; ++s) {
        EXPECT_EQ(positions->GetTensorData<int64_t>()[dim * sequence_length + s], first_position + s);
      }
    }
    auto* current_mask = state_->GetInput("attention_mask");
    EXPECT_EQ(current_mask->GetTensorRawData(), mask_address);
    ASSERT_EQ(current_mask->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{1, 16}));
    for (int s = 0; s < 16; ++s) {
      EXPECT_EQ(current_mask->GetTensorData<int64_t>()[s], s > 0 && s < valid_length ? 1 : 0);
    }
  };
  expect_inputs(1, 2, 4);
  inputs->Update(tokens.subspan(1, 2), 6, 2);
  expect_inputs(2, 3, 6);
  inputs->Update(next_token, 7, 1);
  expect_inputs(1, 5, 7);
  inputs->RewindTo(4);
  inputs->Update(next_token, 5, 1);
  expect_inputs(1, 3, 5);

  inputs->RewindTo(0);
  inputs->Update(tokens, 3, 3);
  inputs->Update(next_token, 4, 1);
  expect_inputs(1, 2, 4);

  // A short new prompt can itself be captured; it must reuse the same allocation.
  inputs->RewindTo(0);
  inputs->Update(next_token, 1, 1);
  EXPECT_EQ(state_->GetInput("position_ids")->GetTensorRawData(), position_address);
  EXPECT_EQ(state_->GetInput("attention_mask")->GetTensorRawData(), mask_address);
  EXPECT_EQ(state_->GetInput("position_ids")->GetTensorData<int64_t>()[0], 0);
  EXPECT_EQ(state_->GetInput("attention_mask")->GetTensorData<int64_t>()[0], 1);
  EXPECT_EQ(state_->GetInput("attention_mask")->GetTensorData<int64_t>()[1], 0);
  EXPECT_THROW(inputs->RewindTo(17), std::runtime_error);
}

TEST_F(StandardPositionInputsTest, MropeFullRewindReinitializesPromptPositionsAndPadding) {
  Use3DPositionIds();
  model_->config_->model.type = "qwen3_5_text";
  model_->config_->model.pad_token_id = 0;
  params_->search.max_length = 16;
  for (bool capture : {false, true}) {
    SCOPED_TRACE(capture);
    params_->use_graph_capture = capture;
    state_ = std::make_unique<PositionInputsTestState>(*params_, *model_);
    auto inputs = CreatePositionInputs();
    inputs->Add();
    auto tokens = model_->p_device_->Allocate<int32_t>(3);
    std::fill_n(tokens.CpuSpan().data(), 3, 10);
    inputs->Update(tokens, 3, 3);
    inputs->Update(tokens.subspan(0, 1), 4, 1);
    inputs->RewindTo(0);
    tokens.CpuSpan()[0] = 0;
    inputs->Update(tokens, 3, 3);
    const auto* positions = state_->GetInput("position_ids")->GetTensorData<int64_t>();
    for (int dim = 0; dim < 3; ++dim) {
      EXPECT_EQ(positions[dim * 3], 0);
      EXPECT_EQ(positions[dim * 3 + 1], 0);
      EXPECT_EQ(positions[dim * 3 + 2], 1);
    }
    const auto* mask = state_->GetInput("attention_mask")->GetTensorData<int64_t>();
    EXPECT_EQ(mask[0], 0);
    EXPECT_EQ(mask[1], 1);
    EXPECT_EQ(mask[2], 1);
  }
}

TEST_F(StandardPositionInputsTest, DefaultCapturedInputsReuseBuffersAcrossBatchedBeamPrompts) {
  model_->config_->model.pad_token_id = 0;
  params_->search.batch_size = 2;
  params_->search.num_beams = 2;
  params_->search.max_length = 8;
  params_->use_graph_capture = true;
  params_->max_graph_capture_length = 2;
  sequence_lengths_ = model_->p_device_->Allocate<int32_t>(4);
  auto inputs = CreatePositionInputs();
  inputs->Add();
  auto tokens = model_->p_device_->Allocate<int32_t>(6);
  const void* position_address = nullptr;
  const void* mask_address = nullptr;
  // A longer prefill between captured prompts must retain the captured storage.
  for (int length : {2, 3, 2}) {
    std::fill_n(tokens.CpuSpan().data(), 6, 10);
    tokens.CpuSpan()[length - 1] = 0;
    inputs->Update(tokens.subspan(0, 2 * length), length, length);
    auto* positions = state_->GetInput("position_ids");
    if (length == 2) {
      if (!position_address)
        position_address = positions->GetTensorRawData();
      EXPECT_EQ(positions->GetTensorRawData(), position_address);
    }
    auto* mask = state_->GetInput("attention_mask");
    if (!mask_address)
      mask_address = mask->GetTensorRawData();
    EXPECT_EQ(mask->GetTensorRawData(), mask_address);
    EXPECT_EQ(mask->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{4, 8}));
    inputs->Update(tokens.subspan(0, 4), length + 1, 1);
    positions = state_->GetInput("position_ids");
    EXPECT_EQ(positions->GetTensorRawData(), position_address);
    EXPECT_EQ(positions->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{4, 1}));
    mask = state_->GetInput("attention_mask");
    for (int row = 0; row < 4; ++row) {
      EXPECT_EQ(positions->GetTensorData<int64_t>()[row], row < 2 ? length - 1 : length);
      for (int col = 0; col < 8; ++col)
        EXPECT_EQ(mask->GetTensorData<int64_t>()[row * 8 + col], col <= length && !(row < 2 && col == length - 1) ? 1 : 0);
    }
    inputs->RewindTo(0);
  }
}

TEST_F(StandardPositionInputsTest, CapturedVisionPositionsPreserveMropeDeltasAndExpandBeams) {
  Use3DPositionIds();
  model_->config_->model.type = "qwen3_vl";
  model_->config_->model.vision_start_token_id = 40;
  model_->config_->model.image_token_id = 41;
  model_->config_->model.video_token_id = 42;
  model_->config_->model.vision.spatial_merge_size = 2;
  model_->config_->model.vision.tokens_per_second = 2;
  params_->search.num_beams = 2;
  params_->search.max_length = 12;
  params_->use_graph_capture = true;
  params_->max_graph_capture_length = 2;
  for (bool video : {false, true}) {
    SCOPED_TRACE(video);
    state_ = std::make_unique<PositionInputsTestState>(*params_, *model_);
    auto inputs = CreatePositionInputs();
    auto* qwen = dynamic_cast<Qwen2VLPositionInputs*>(inputs.get());
    ASSERT_NE(qwen, nullptr);
    auto grid = OrtValue::CreateTensor(model_->allocator_cpu_, std::array<int64_t, 2>{1, 3}, Ort::TypeToTensorType<int64_t>);
    const std::array<int64_t, 3> dimensions = video ? std::array<int64_t, 3>{2, 2, 4} : std::array<int64_t, 3>{1, 4, 4};
    std::copy(dimensions.begin(), dimensions.end(), grid->GetTensorMutableData<int64_t>());
    auto grid_tensor = std::make_shared<Tensor>(std::move(grid));
    qwen->SetGridTensors(video ? nullptr : grid_tensor, video ? grid_tensor : nullptr, nullptr);
    inputs->Add();
    auto tokens = model_->p_device_->Allocate<int32_t>(6);
    std::fill_n(tokens.CpuSpan().data(), 6, video ? 42 : 41);
    tokens.CpuSpan()[0] = 40;
    tokens.CpuSpan()[5] = 10;
    inputs->Update(tokens, 6, 6);
    auto* positions = state_->GetInput("position_ids");
    EXPECT_EQ(positions->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{3, 2, 6}));
    const std::array<int64_t, 18> expected = video
                                                 ? std::array<int64_t, 18>{0, 1, 1, 3, 3, 4, 0, 1, 1, 1, 1, 4, 0, 1, 2, 1, 2, 4}
                                                 : std::array<int64_t, 18>{0, 1, 1, 1, 1, 3, 0, 1, 1, 2, 2, 3, 0, 1, 2, 1, 2, 3};
    for (int dim = 0; dim < 3; ++dim)
      for (int beam = 0; beam < 2; ++beam)
        for (int col = 0; col < 6; ++col)
          EXPECT_EQ(positions->GetTensorData<int64_t>()[(dim * 2 + beam) * 6 + col], expected[dim * 6 + col]);
    inputs->Update(tokens.subspan(0, 4), 8, 2);
    positions = state_->GetInput("position_ids");
    const void* address = positions->GetTensorRawData();
    for (int row = 0; row < 6; ++row)
      for (int col = 0; col < 2; ++col)
        EXPECT_EQ(positions->GetTensorData<int64_t>()[row * 2 + col], (video ? 5 : 4) + col);
    inputs->RewindTo(7);
    inputs->Update(tokens.subspan(0, 2), 8, 1);
    positions = state_->GetInput("position_ids");
    EXPECT_EQ(positions->GetTensorRawData(), address);
    for (int row = 0; row < 6; ++row)
      EXPECT_EQ(positions->GetTensorData<int64_t>()[row], video ? 6 : 5);
    inputs->RewindTo(0);
    qwen->SetGridTensors(nullptr, nullptr, nullptr);
    std::fill_n(tokens.CpuSpan().data(), 2, 10);
    inputs->Update(tokens.subspan(0, 2), 2, 2);
    positions = state_->GetInput("position_ids");
    EXPECT_EQ(positions->GetTensorRawData(), address);
    EXPECT_EQ(positions->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{3, 2, 2}));
    for (int row = 0; row < 6; ++row)
      for (int col = 0; col < 2; ++col)
        EXPECT_EQ(positions->GetTensorData<int64_t>()[row * 2 + col], col);
  }
}

TEST(Qwen2VLPositionInputsTest, RejectsNegativeGridDimensions) {
  const std::vector<int64_t> grid{1, -1, 2};
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "image_grid_thw");
  });
  EXPECT_NE(message.find("non-negative"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, RejectsExcessiveGridDimensions) {
  const std::vector<int64_t> grid{1, 1, 20000};
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "video_grid_thw");
  });
  EXPECT_NE(message.find("<= 16384"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, RejectsVisionLenExceedingSequenceLength) {
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLVisionLengthFitsSequence(8, 8, 8, 5, 100);
  });
  EXPECT_NE(message.find("positions available in sequence"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, RejectsGridWithIncorrectElementCount) {
  const std::vector<int64_t> grid{1, 2};
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "image_grid_thw");
  });
  EXPECT_NE(message.find("divisible by 3"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, AcceptsValidGridDimensions) {
  const std::vector<int64_t> grid{1, 8, 8, 2, 4, 4};
  EXPECT_NO_THROW(ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "image_grid_thw"));
  EXPECT_NO_THROW(ValidateQwen2VLVisionLengthFitsSequence(2, 2, 2, 10, 100));
}

}  // namespace Generators::test
