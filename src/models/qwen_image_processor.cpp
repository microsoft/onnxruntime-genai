// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include "../generators.h"
#include "model.h"

namespace Generators {

QwenImageProcessor::QwenImageProcessor(Config& config, const SessionInfo& session_info)
    : config_{config},
      pixel_values_type_{session_info.HasInput(config.model.vision.inputs.pixel_values)
                             ? session_info.GetInputDataType(config.model.vision.inputs.pixel_values)
                             : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
      image_grid_thw_type_{session_info.HasInput(config.model.vision.inputs.image_grid_thw)
                               ? session_info.GetInputDataType(config.model.vision.inputs.image_grid_thw)
                               : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64} {
  auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  
  if (!fs::exists(processor_config)) {
    throw std::runtime_error("Qwen processor config not found at: " + processor_config);
  }

  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  // Add name mappings for Qwen vision inputs
  config.AddMapping(config.model.vision.inputs.pixel_values, config.model.vision.inputs.pixel_values);
  config.AddMapping(config.model.vision.inputs.image_grid_thw, config.model.vision.inputs.image_grid_thw);
}

std::unique_ptr<NamedTensors> QwenImageProcessor::Process(const Tokenizer& tokenizer,
                                                           const Payload& payload) const {
  if (!payload.images || payload.images->num_images_ == 0) {
    throw std::runtime_error("Qwen processor requires at least one image");
  }

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Process images through onnxruntime-extensions processor
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), payload.images->images_.get(), result.ToBeAssigned()));

  // Extract pixel_values (first tensor output)
  OrtxTensor* pixel_values = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, &pixel_values));
  
  if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    named_tensors->emplace(config_.model.vision.inputs.pixel_values,
                           std::make_shared<Tensor>(ProcessTensor<float>(pixel_values, allocator)));
  } else if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    named_tensors->emplace(config_.model.vision.inputs.pixel_values,
                           std::make_shared<Tensor>(ProcessTensor<Ort::BFloat16_t>(pixel_values, allocator)));
  } else {
    named_tensors->emplace(config_.model.vision.inputs.pixel_values,
                           std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(pixel_values, allocator)));
  }

  // Try to extract image_grid_thw (spatial dimensions tensor - second output)
  // This is optional as some processor configs may not produce it
  OrtxTensor* image_grid_thw_tensor = nullptr;
  // Don't use CheckResult here since failure is expected if processor doesn't produce this tensor
  OrtxTensorResultGetAt(result.get(), 1, &image_grid_thw_tensor);
  
  if (image_grid_thw_tensor != nullptr) {
    // Successfully got image_grid_thw from processor
    named_tensors->emplace(config_.model.vision.inputs.image_grid_thw,
                           std::make_shared<Tensor>(ProcessTensor<int64_t>(image_grid_thw_tensor, allocator)));
  } else {
    // Processor didn't produce image_grid_thw, compute it from pixel_values shape
    // pixel_values shape: [batch_size, num_patches, channels * temporal_patch * patch * patch]
    // We need to infer grid_t, grid_h, grid_w from num_patches
    
    // Get the pixel_values tensor we just created
    auto& pixel_values_ort = named_tensors->at(config_.model.vision.inputs.pixel_values)->ort_tensor_;
    auto shape = pixel_values_ort->GetTensorTypeAndShapeInfo()->GetShape();
    
    if (shape.size() < 2) {
      throw std::runtime_error("pixel_values tensor must have at least 2 dimensions [batch, num_patches, ...]");
    }
    
    int64_t batch_size = shape[0];
    int64_t num_patches = shape[1];
    
    // For Qwen 2.5 VL with patch_size=14, merge_size=2, temporal_patch_size=2
    // The number of patches = grid_t * grid_h * grid_w
    // where grid_h and grid_w depend on the image resolution after smart_resize
    
    // Since we don't have the original image dimensions here, we need to make an assumption
    // or compute from num_patches. For now, assume square-ish layout with temporal=1
    int64_t grid_t = 1;  // Assume single frame (not video)
    
    // For square-ish images: grid_h ≈ grid_w, so num_patches ≈ grid_h * grid_w
    int64_t grid_hw = static_cast<int64_t>(std::sqrt(static_cast<double>(num_patches)));
    
    // Adjust to find exact factors
    int64_t grid_h = grid_hw;
    int64_t grid_w = num_patches / grid_hw;
    
    // Verify the factorization
    if (grid_t * grid_h * grid_w != num_patches) {
      // Try to find better factorization
      for (int64_t h = grid_hw; h > 0; h--) {
        if (num_patches % h == 0) {
          grid_h = h;
          grid_w = num_patches / h;
          break;
        }
      }
    }
    
    // Create image_grid_thw tensor with shape [batch_size, 3] containing [grid_t, grid_h, grid_w]
    std::vector<int64_t> grid_shape = {batch_size, 3};
    auto image_grid_thw = OrtValue::CreateTensor<int64_t>(allocator, std::span<const int64_t>(grid_shape.data(), grid_shape.size()));
    
    auto* grid_data = image_grid_thw->GetTensorMutableData<int64_t>();
    for (int64_t i = 0; i < batch_size; i++) {
      grid_data[i * 3 + 0] = grid_t;
      grid_data[i * 3 + 1] = grid_h;
      grid_data[i * 3 + 2] = grid_w;
    }
    
    named_tensors->emplace(config_.model.vision.inputs.image_grid_thw,
                           std::make_shared<Tensor>(std::move(image_grid_thw)));
  }

  // Note: The actual vision model execution (patch_embed, vision_attn, patch_merger) 
  // will be handled separately in the vision pipeline state manager

  return named_tensors;
}

}  // namespace Generators
