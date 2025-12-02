// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "qwen2_5_vl_image_processor.h"
#include <numeric>

namespace Generators {

Qwen2_5VLImageProcessor::Qwen2_5VLImageProcessor(Config& config, const SessionInfo& session_info) {
  const auto processor_config = (config.config_path / fs::path("processor_config.json")).string();
  if (!fs::exists(config.config_path / fs::path("processor_config.json"))) {
    throw std::runtime_error("processor_config.json not found at: " + processor_config);
  }

  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  auto input_names = session_info.GetInputNames();
  for (const auto& input_name : input_names) {
    if (input_name.find("pixel_values") != std::string::npos) {
      pixel_values_name_ = input_name;
    } else if (input_name.find("image_grid_thw") != std::string::npos) {
      image_grid_thw_name_ = input_name;
    }
  }
}

std::unique_ptr<NamedTensors> Qwen2_5VLImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  if (!payload.images) {
    throw std::runtime_error("No images provided to Qwen2.5VLImageProcessor");
  }

  std::string prompt = std::string(payload.prompt);
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  const std::vector<int32_t> input_ids = tokenizer.Encode(prompt.c_str());
  std::unique_ptr<OrtValue> input_ids_value = OrtValue::CreateTensor<int32_t>(
      allocator, std::vector<int64_t>{1, static_cast<int64_t>(input_ids.size())});
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());
  named_tensors->emplace(Config::Defaults::InputIdsName, std::make_shared<Tensor>(std::move(input_ids_value)));

  // Run image preprocessing using onnxruntime-extensions
  // This will execute the full pipeline from processor_config.json:
  // DecodeImage -> ConvertRGB -> Resize (smart_resize) -> Rescale -> Normalize -> PatchImage
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), payload.images->images_.get(), result.ToBeAssigned()));

  OrtxTensor* pixel_values = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, &pixel_values));

  named_tensors->emplace(pixel_values_name_, std::make_shared<Tensor>(ProcessTensor<float>(pixel_values, allocator)));

  const void* pixel_values_data{};
  const int64_t* pixel_values_shape{};
  size_t pixel_values_dims{};
  CheckResult(OrtxGetTensorData(pixel_values, &pixel_values_data, &pixel_values_shape, &pixel_values_dims));
  
  if (pixel_values_dims >= 2) {
    int64_t batch_size = pixel_values_shape[0];
    int64_t num_patches = pixel_values_shape[1];

    int64_t grid_t = 1;  // Single frame
    int64_t grid_h = static_cast<int64_t>(std::sqrt(num_patches));
    int64_t grid_w = num_patches / grid_h;
    
    std::vector<int64_t> grid_thw_shape = {batch_size, 3};
    auto grid_thw_tensor = OrtValue::CreateTensor<int64_t>(allocator, grid_thw_shape);
    
    auto* dst = grid_thw_tensor->GetTensorMutableData<int64_t>();
    dst[0] = grid_t;
    dst[1] = grid_h;
    dst[2] = grid_w;
    
    named_tensors->emplace(image_grid_thw_name_, std::make_shared<Tensor>(std::move(grid_thw_tensor)));
  }

  return named_tensors;
}

}  // namespace Generators
