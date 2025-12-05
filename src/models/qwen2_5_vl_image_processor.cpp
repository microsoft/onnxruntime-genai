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

  auto pixel_values_ortvalue = ProcessTensor<float>(pixel_values, allocator);
  auto pixel_values_type_shape_info = pixel_values_ortvalue->GetTensorTypeAndShapeInfo();
  auto pixel_values_shape = pixel_values_type_shape_info->GetShape();
  named_tensors->emplace(pixel_values_name_, std::make_shared<Tensor>(std::move(pixel_values_ortvalue)));

  // Check if processor returns grid_thw as second output
  OrtxTensor* grid_thw_tensor = nullptr;
  auto grid_thw_result = OrtxTensorResultGetAt(result.get(), 1, &grid_thw_tensor);

  if (grid_thw_result == extError_t::kOrtxOK && grid_thw_tensor != nullptr) {
    named_tensors->emplace(image_grid_thw_name_, std::make_shared<Tensor>(ProcessTensor<int64_t>(grid_thw_tensor, allocator)));
  } else {
    // Fallback: calculate grid_thw from pixel_values shape
    if (pixel_values_shape.size() >= 2) {
      int64_t batch_size = pixel_values_shape[0];
      int64_t num_patches = pixel_values_shape[1];

      int64_t grid_t = 1;  // Single frame for static images
      int64_t grid_h, grid_w;

      grid_h = static_cast<int64_t>(std::sqrt(static_cast<double>(num_patches)));
      while (grid_h > 0 && num_patches % grid_h != 0) {
        grid_h--;
      }

      if (grid_h == 0) {
        throw std::runtime_error("Failed to factorize num_patches for grid calculation");
      }

      grid_w = num_patches / grid_h;

      std::vector<int64_t> grid_thw_shape = {batch_size, 3};
      auto grid_thw_output = OrtValue::CreateTensor<int64_t>(allocator, grid_thw_shape);

      auto* dst = grid_thw_output->GetTensorMutableData<int64_t>();
      dst[0] = grid_t;
      dst[1] = grid_h;
      dst[2] = grid_w;

      named_tensors->emplace(image_grid_thw_name_, std::make_shared<Tensor>(std::move(grid_thw_output)));
    }
  }

  return named_tensors;
}

}  // namespace Generators
