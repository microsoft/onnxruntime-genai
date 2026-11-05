// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "models/io/static_kv_cache.h"
#include "models/io/logits.h"
#include "telemetry_test_environment.h"

namespace {

std::unique_ptr<Generators::Config> MakeConfig(bool is_pipeline) {
  auto config = std::make_unique<Generators::Config>();
  config->model.decoder.sliding_window.emplace();
  config->model.decoder.sliding_window->window_size = 64;
  if (is_pipeline)
    config->model.decoder.pipeline.emplace_back();
  return config;
}

struct CacheTestModel : Generators::Model {
  explicit CacheTestModel(bool is_pipeline) : Model{MakeConfig(is_pipeline)} {}

  std::unique_ptr<Generators::State> CreateState(
      Generators::DeviceSpan<int32_t>, const Generators::GeneratorParams&) const override {
    return nullptr;
  }
};

TEST(KvCacheTests, UsesDeviceWindowSizeForSingleSessionModel) {
  CacheTestModel model{false};
  Generators::Config::Search search;
  EXPECT_TRUE(Generators::UsesNonRewindableWindowedKeyValueCache(
      model, model.config_->model.decoder));
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 4096), 80);
}

TEST(KvCacheTests, IgnoresTopLevelDeviceWindowSizeForPipelineModel) {
  CacheTestModel model{true};
  Generators::Config::Search search;
  EXPECT_FALSE(Generators::UsesNonRewindableWindowedKeyValueCache(
      model, model.config_->model.decoder));
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 4096), 0);
}

TEST(KvCacheTests, StorageSupportsAdditionalFixedWidthTypes) {
  EXPECT_EQ(Ort::SizeOf(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16), sizeof(uint16_t));
  EXPECT_EQ(Ort::SizeOf(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2), sizeof(uint8_t));
}

TEST(KvCacheTests, BFloat16BeamReorderPreservesValues) {
  auto& device = *Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  const std::array<int64_t, 4> shape{2, 1, 1, 4};
  auto source = OrtValue::CreateTensor(
      device.GetAllocator(), shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  auto destination = OrtValue::CreateTensor(
      device.GetAllocator(), shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  auto* source_data = source->GetTensorMutableData<Ort::BFloat16_t>();
  for (uint16_t i = 0; i < 8; ++i) source_data[i] = Ort::BFloat16_t{i};

  auto source_span = Generators::KeyValueCacheDetail::WrapKvCacheTensor<Ort::BFloat16_t>(
      device, *source, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  auto destination_span = Generators::KeyValueCacheDetail::WrapKvCacheTensor<Ort::BFloat16_t>(
      device, *destination, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  const std::array<int32_t, 2> indices{1, 0};
  Generators::KeyValueCacheDetail::CopyReorderedBlocks(
      destination_span, source_span, indices, 4);

  const auto* reordered = destination->GetTensorData<Ort::BFloat16_t>();
  for (uint16_t i = 0; i < 4; ++i) {
    EXPECT_EQ(reordered[i], Ort::BFloat16_t{i + 4});
    EXPECT_EQ(reordered[i + 4], Ort::BFloat16_t{i});
  }
}

TEST(KvCacheTests, BFloat16RewindCopiesUniformAndPerLayerRows) {
  auto& device = *Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  const std::array<int64_t, 1> source_shape{12};
  auto source = OrtValue::CreateTensor(
      device.GetAllocator(), source_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  auto* source_data = source->GetTensorMutableData<Ort::BFloat16_t>();
  for (uint16_t i = 0; i < 12; ++i) source_data[i] = Ort::BFloat16_t{i + 1};
  auto source_span = Generators::KeyValueCacheDetail::WrapKvCacheTensor<Ort::BFloat16_t>(
      device, *source, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);

  auto uniform = OrtValue::CreateTensor(
      device.GetAllocator(), std::array<int64_t, 1>{6}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  auto uniform_span = Generators::KeyValueCacheDetail::WrapKvCacheTensor<Ort::BFloat16_t>(
      device, *uniform, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  Generators::KeyValueCacheDetail::CopyRows(uniform_span, source_span, 2, 6, 3);
  const auto* uniform_data = uniform->GetTensorData<Ort::BFloat16_t>();
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(uniform_data[i], source_data[i]);
    EXPECT_EQ(uniform_data[i + 3], source_data[i + 6]);
  }

  auto per_layer = OrtValue::CreateTensor(
      device.GetAllocator(), std::array<int64_t, 1>{4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  auto per_layer_span = Generators::KeyValueCacheDetail::WrapKvCacheTensor<Ort::BFloat16_t>(
      device, *per_layer, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  Generators::KeyValueCacheDetail::CopyRows(per_layer_span, source_span, 2, 4, 2);
  const auto* per_layer_data = per_layer->GetTensorData<Ort::BFloat16_t>();
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(per_layer_data[i], source_data[i]);
    EXPECT_EQ(per_layer_data[i + 2], source_data[i + 4]);
  }
}

TEST(LogitsTests, ValidatesSupportedElementTypes) {
  EXPECT_NO_THROW(Generators::ValidateLogitsType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  EXPECT_NO_THROW(Generators::ValidateLogitsType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  EXPECT_NO_THROW(Generators::ValidateLogitsType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16));
  EXPECT_THROW(Generators::ValidateLogitsType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8), std::runtime_error);
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
