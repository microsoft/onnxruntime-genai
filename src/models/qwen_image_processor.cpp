// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <regex>

namespace Generators {

namespace {

constexpr int64_t kMergeSize = 2;  // Qwen2-VL merge size for vision tokens

std::tuple<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>>
ProcessImagePrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                   OrtxTensor* pixel_values, OrtxTensor* image_grid_thw, 
                   const int64_t* computed_grid_data, int64_t computed_grid_num_images,
                   Ort::Allocator& allocator) {
  constexpr char vision_start_token[] = "<|vision_start|>";
  constexpr char vision_end_token[] = "<|vision_end|>";
  constexpr char image_pad_token[] = "<|image_pad|>";

  int64_t num_images = 0;
  int64_t total_image_tokens = 0;
  const int64_t* image_grid_thw_data = nullptr;
  
  if (pixel_values) {
    const float* pixel_values_data{};
    const int64_t* pixel_values_shape{};
    size_t pixel_values_num_dims;
    CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&pixel_values_data),
                                  &pixel_values_shape, &pixel_values_num_dims));
    
    // Get image_grid_thw data from either processor output or computed value
    if (image_grid_thw) {
      const int64_t* image_grid_thw_shape{};
      size_t image_grid_thw_num_dims;
      CheckResult(OrtxGetTensorData(image_grid_thw, reinterpret_cast<const void**>(&image_grid_thw_data),
                                    &image_grid_thw_shape, &image_grid_thw_num_dims));
      num_images = image_grid_thw_shape[0];
    } else if (computed_grid_data) {
      image_grid_thw_data = computed_grid_data;
      num_images = computed_grid_num_images;
    }
    
    // Calculate total image tokens based on grid dimensions
    // For each image: (temporal * height * width) / (merge_size^2)
    for (int64_t i = 0; i < num_images; ++i) {
      int64_t t = image_grid_thw_data[i * 3 + 0];
      int64_t h = image_grid_thw_data[i * 3 + 1];
      int64_t w = image_grid_thw_data[i * 3 + 2];
      total_image_tokens += (t * h * w) / (kMergeSize * kMergeSize);
    }
  }

  // Generate input_ids with vision tokens
  std::string text = prompt;
  
  // If prompt is empty, add vision markers for each image
  if (text.empty()) {
    for (int64_t i = 0; i < num_images; ++i) {
      text += std::string(vision_start_token) + " " + std::string(vision_end_token);
      if (i < num_images - 1) {
        text += " ";
      }
    }
  }

  // Count the number of vision_start tokens and make sure it matches the number of images
  // Need to escape special regex characters in the token
  const std::regex vision_start_regex{R"(<\|vision_start\|>)"};
  const auto vision_start_begin = std::sregex_iterator(text.begin(), text.end(), vision_start_regex);
  const auto vision_start_end = std::sregex_iterator();
  const auto vision_start_tokens = std::distance(vision_start_begin, vision_start_end);
  
  if (num_images != vision_start_tokens) {
    throw std::runtime_error("Prompt contained " + std::to_string(vision_start_tokens) + 
                           " vision_start tokens but received " + std::to_string(num_images) + " images.");
  }

  // For Qwen2-VL, we need to replace vision markers with image_pad tokens
  // The number of image_pad tokens for each image depends on the image dimensions
  if (num_images > 0 && image_grid_thw_data) {
    std::string modified_text;
    size_t last_pos = 0;
    size_t image_idx = 0;
    
    std::smatch match;
    std::string temp_text = text;
    while (std::regex_search(temp_text, match, vision_start_regex)) {
      // Add text before the vision_start token
      modified_text += text.substr(last_pos, match.position() - (last_pos - (text.size() - temp_text.size())));
      
      // Calculate number of image_pad tokens for this image
      int64_t t = image_grid_thw_data[image_idx * 3 + 0];
      int64_t h = image_grid_thw_data[image_idx * 3 + 1];
      int64_t w = image_grid_thw_data[image_idx * 3 + 2];
      int64_t num_pads = (t * h * w) / (kMergeSize * kMergeSize);
      
      // Add vision_start, image_pad tokens, and vision_end
      modified_text += vision_start_token;
      for (int64_t i = 0; i < num_pads; ++i) {
        modified_text += image_pad_token;
      }
      modified_text += vision_end_token;
      
      last_pos = match.position() + match.length() + (text.size() - temp_text.size());
      
      // Find and skip vision_end token
      size_t vision_end_pos = text.find(vision_end_token, last_pos);
      if (vision_end_pos != std::string::npos) {
        last_pos = vision_end_pos + strlen(vision_end_token);
      }
      
      temp_text = match.suffix();
      image_idx++;
    }
    modified_text += text.substr(last_pos);
    text = modified_text;
  }

  const std::vector<int32_t> input_ids = tokenizer.Encode(text.c_str());

  std::unique_ptr<OrtValue> input_ids_value = OrtValue::CreateTensor<int32_t>(
      allocator, std::vector<int64_t>{1, static_cast<int64_t>(input_ids.size())});
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());

  std::unique_ptr<OrtValue> num_img_tokens = OrtValue::CreateTensor<int64_t>(
      allocator, std::vector<int64_t>{1});
  num_img_tokens->GetTensorMutableData<int64_t>()[0] = total_image_tokens;

  return {std::move(input_ids_value), std::move(num_img_tokens)};
}

}  // namespace

QwenImageProcessor::QwenImageProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)} {
  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
}

std::unique_ptr<NamedTensors> QwenImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  std::string prompt = std::string(payload.prompt);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!images) {
    [[maybe_unused]] auto [input_ids, num_img_tokens] = ProcessImagePrompt(tokenizer, prompt, nullptr, nullptr, nullptr, 0, allocator);
    named_tensors->emplace(Config::Defaults::InputIdsName, std::make_shared<Tensor>(std::move(input_ids)));
    return named_tensors;
  }

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), images->images_.get(), result.ToBeAssigned()));

  OrtxTensor* pixel_values = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, &pixel_values));

  OrtxTensor* image_grid_thw = nullptr;
  // Try to get image_grid_thw from processor (second output)
  auto status = OrtxTensorResultGetAt(result.get(), 1, &image_grid_thw);
  
  // Get pixel_values data and shape
  const float* pixel_values_data{};
  const int64_t* pixel_values_shape{};
  size_t pixel_values_num_dims;
  CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&pixel_values_data),
                                &pixel_values_shape, &pixel_values_num_dims));
  
  // If processor doesn't provide image_grid_thw or patched pixel_values, compute them
  std::unique_ptr<OrtValue> computed_image_grid_thw;
  std::unique_ptr<OrtValue> patched_pixel_values;
  const int64_t* computed_grid_data = nullptr;
  int64_t computed_grid_num_images = 0;
  
  // Check if pixel_values is already patched (3D: [batch, num_patches, patch_dim])
  if (pixel_values_num_dims == 3 && (status != kOrtxOK || !image_grid_thw)) {
    constexpr int64_t kPatchSize = 14;
    constexpr int64_t kTemporalPatchSize = 2;
    constexpr int64_t kChannels = 3;
    
    int64_t batch = pixel_values_shape[0];
    int64_t num_patches = pixel_values_shape[1];
    int64_t patch_dim = pixel_values_shape[2];
    
    // Verify patch_dim matches expected value (C * temporal_patch_size * patch_size * patch_size)
    int64_t expected_patch_dim = kChannels * kTemporalPatchSize * kPatchSize * kPatchSize;
    if (patch_dim != expected_patch_dim) {
      throw std::runtime_error(
          "Unexpected patch dimension " + std::to_string(patch_dim) + 
          ", expected " + std::to_string(expected_patch_dim));
    }
    
    // For patched data, we need to infer grid dimensions from num_patches
    // num_patches = grid_t * grid_h * grid_w
    // For single images: grid_t = 1, so num_patches = grid_h * grid_w
    // We need to factor num_patches into grid_h and grid_w
    // The grid dimensions must be divisible by merge_size (2)
    
    // Try to find the grid dimensions by factoring num_patches
    // Prefer square-ish grids
    int64_t grid_h = 0;
    int64_t grid_w = 0;
    int64_t grid_t = 1;  // Single image = 1 temporal frame
    
    // Find factors of num_patches that are both divisible by merge_size
    for (int64_t h = 1; h * h <= num_patches; ++h) {
      if (num_patches % h == 0) {
        int64_t w = num_patches / h;
        if (h % kMergeSize == 0 && w % kMergeSize == 0) {
          grid_h = h;
          grid_w = w;
        }
      }
    }
    
    if (grid_h == 0 || grid_w == 0) {
      throw std::runtime_error(
          "Could not determine valid grid dimensions from " + std::to_string(num_patches) + 
          " patches. Grid dimensions must be divisible by merge_size (" + std::to_string(kMergeSize) + ")");
    }
    
    // Create image_grid_thw: [batch, 3]
    computed_image_grid_thw = OrtValue::CreateTensor<int64_t>(
        allocator, std::vector<int64_t>{batch, 3});
    auto* grid_data = computed_image_grid_thw->GetTensorMutableData<int64_t>();
    
    for (int64_t b = 0; b < batch; ++b) {
      grid_data[b * 3 + 0] = grid_t;
      grid_data[b * 3 + 1] = grid_h;
      grid_data[b * 3 + 2] = grid_w;
    }
    
    computed_grid_data = grid_data;
    computed_grid_num_images = batch;
  }
  // Check if pixel_values needs patching (shape should be [1, height, width, channels] in HWC format)
  else if (pixel_values_num_dims == 4 && pixel_values_shape[0] == 1) {
    constexpr int64_t kPatchSize = 14;
    constexpr int64_t kTemporalPatchSize = 2;
    constexpr int64_t kChannels = 3;
    
    int64_t height = pixel_values_shape[1];      // HWC: [batch, height, width, channels]
    int64_t width = pixel_values_shape[2];
    int64_t channels = pixel_values_shape[3];
    
    int64_t height_patches = height / kPatchSize;
    int64_t width_patches = width / kPatchSize;
    
    // Validate that patch dimensions are compatible with 2x2 merging
    if (height_patches % kMergeSize != 0 || width_patches % kMergeSize != 0) {
      throw std::runtime_error(
          "Image dimensions " + std::to_string(width) + "x" + std::to_string(height) + 
          " produce patch grid " + std::to_string(width_patches) + "x" + std::to_string(height_patches) + 
          " which is not compatible with " + std::to_string(kMergeSize) + "x" + std::to_string(kMergeSize) + " merging. " +
          "Both dimensions must be divisible by " + std::to_string(kPatchSize * kMergeSize) + " (patch_size * merge_size). " +
          "Please ensure your image processor resizes images to compatible dimensions.");
    }
    
    int64_t total_patches = height_patches * width_patches;
    int64_t patch_dim = channels * kTemporalPatchSize * kPatchSize * kPatchSize;
    
    // Create patched pixel_values: [total_patches, patch_dim]
    patched_pixel_values = OrtValue::CreateTensor<float>(
        allocator, std::vector<int64_t>{total_patches, patch_dim});
    auto* patched_data = patched_pixel_values->GetTensorMutableData<float>();
    
    // Extract patches from single image in HWC format
    // Each spatial patch is replicated kTemporalPatchSize times
    int64_t patch_idx = 0;
    for (int64_t ph = 0; ph < height_patches; ++ph) {
      for (int64_t pw = 0; pw < width_patches; ++pw) {
        int64_t h_start = ph * kPatchSize;
        int64_t w_start = pw * kPatchSize;
        
        int64_t write_idx = patch_idx * patch_dim;
        
        // Repeat the same spatial patch kTemporalPatchSize times
        // Output: [temporal, channels, patch_h, patch_w]
        for (int64_t t = 0; t < kTemporalPatchSize; ++t) {
          for (int64_t c = 0; c < channels; ++c) {
            for (int64_t h = 0; h < kPatchSize; ++h) {
              for (int64_t w = 0; w < kPatchSize; ++w) {
                // HWC format: pixel_values[height][width][channels]
                int64_t src_idx = (h_start + h) * width * channels + (w_start + w) * channels + c;
                patched_data[write_idx++] = pixel_values_data[src_idx];
              }
            }
          }
        }
        patch_idx++;
      }
    }
    
    // Create image_grid_thw: [1, 3] for single image
    if (status != kOrtxOK || !image_grid_thw) {
      computed_image_grid_thw = OrtValue::CreateTensor<int64_t>(
          allocator, std::vector<int64_t>{1, 3});
      auto* grid_data = computed_image_grid_thw->GetTensorMutableData<int64_t>();
      
      // For a single image: T=1 (one frame), H=height_patches, W=width_patches
      // The kTemporalPatchSize is embedded in the patch dimension
      grid_data[0] = 1;  // Single temporal frame for images
      grid_data[1] = height_patches;
      grid_data[2] = width_patches;
      
      computed_grid_data = grid_data;
      computed_grid_num_images = 1;
    }
  }

  auto [input_ids, num_img_tokens] = ProcessImagePrompt(tokenizer, prompt, pixel_values, 
                                                          image_grid_thw, computed_grid_data, computed_grid_num_images, allocator);
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName), std::make_shared<Tensor>(std::move(input_ids)));

  // Use patched pixel_values if we computed it, otherwise use processor output
  if (patched_pixel_values) {
    // Convert to the correct type if needed
    if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      // Convert float to bfloat16
      auto shape_vec = patched_pixel_values->GetTensorTypeAndShapeInfo()->GetShape();
      auto bf16_tensor = OrtValue::CreateTensor<Ort::BFloat16_t>(allocator, shape_vec);
      const float* src = patched_pixel_values->GetTensorData<float>();
      auto* dst = static_cast<uint16_t*>(bf16_tensor->GetTensorMutableData<void>());
      size_t count = patched_pixel_values->GetTensorTypeAndShapeInfo()->GetElementCount();
      for (size_t i = 0; i < count; ++i) {
        dst[i] = Float32ToBFloat16(src[i]);
      }
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(std::move(bf16_tensor)));
    } else if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      // Convert float to float16
      auto shape_vec = patched_pixel_values->GetTensorTypeAndShapeInfo()->GetShape();
      auto fp16_tensor = OrtValue::CreateTensor<Ort::Float16_t>(allocator, shape_vec);
      const float* src = patched_pixel_values->GetTensorData<float>();
      auto* dst = static_cast<uint16_t*>(fp16_tensor->GetTensorMutableData<void>());
      size_t count = patched_pixel_values->GetTensorTypeAndShapeInfo()->GetElementCount();
      for (size_t i = 0; i < count; ++i) {
        dst[i] = FastFloat32ToFloat16(src[i]);
      }
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(std::move(fp16_tensor)));
    } else {
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(std::move(patched_pixel_values)));
    }
  } else if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    // For Qwen, pixel_values from PatchImage may have shape [1, num_patches, patch_dim]
    // but the model expects [num_patches, patch_dim], so squeeze batch dim if present
    if (pixel_values_num_dims == 3 && pixel_values_shape[0] == 1) {
      // Squeeze the batch dimension
      std::vector<int64_t> squeezed_shape{pixel_values_shape[1], pixel_values_shape[2]};
      auto squeezed_tensor = OrtValue::CreateTensor<float>(allocator, squeezed_shape);
      
      const float* src_data{};
      CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&src_data), 
                                    &pixel_values_shape, &pixel_values_num_dims));
      
      size_t total_elements = squeezed_shape[0] * squeezed_shape[1];
      std::copy(src_data, src_data + total_elements, squeezed_tensor->GetTensorMutableData<float>());
      
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(std::move(squeezed_tensor)));
    } else {
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(ProcessTensor<float>(pixel_values, allocator)));
    }
  } else if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    // For Qwen, pixel_values from PatchImage may have shape [1, num_patches, patch_dim]
    // but the model expects [num_patches, patch_dim], so squeeze batch dim if present
    if (pixel_values_num_dims == 3 && pixel_values_shape[0] == 1) {
      // Squeeze the batch dimension and convert to bfloat16
      std::vector<int64_t> squeezed_shape{pixel_values_shape[1], pixel_values_shape[2]};
      auto squeezed_tensor = OrtValue::CreateTensor<Ort::BFloat16_t>(allocator, squeezed_shape);
      
      const float* src_data{};
      CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&src_data), 
                                    &pixel_values_shape, &pixel_values_num_dims));
      
      auto* dst = static_cast<uint16_t*>(squeezed_tensor->GetTensorMutableData<void>());
      size_t total_elements = squeezed_shape[0] * squeezed_shape[1];
      for (size_t i = 0; i < total_elements; ++i) {
        dst[i] = Float32ToBFloat16(src_data[i]);
      }
      
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(std::move(squeezed_tensor)));
    } else {
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(ProcessTensor<Ort::BFloat16_t>(pixel_values, allocator)));
    }
  } else {
    // Float16 case
    if (pixel_values_num_dims == 3 && pixel_values_shape[0] == 1) {
      // Squeeze the batch dimension and convert to float16
      std::vector<int64_t> squeezed_shape{pixel_values_shape[1], pixel_values_shape[2]};
      auto squeezed_tensor = OrtValue::CreateTensor<Ort::Float16_t>(allocator, squeezed_shape);
      
      const float* src_data{};
      CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&src_data), 
                                    &pixel_values_shape, &pixel_values_num_dims));
      
      auto* dst = static_cast<uint16_t*>(squeezed_tensor->GetTensorMutableData<void>());
      size_t total_elements = squeezed_shape[0] * squeezed_shape[1];
      for (size_t i = 0; i < total_elements; ++i) {
        dst[i] = FastFloat32ToFloat16(src_data[i]);
      }
      
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(std::move(squeezed_tensor)));
    } else {
      named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                             std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(pixel_values, allocator)));
    }
  }

  // Add image_grid_thw tensor (either from processor or computed)
  // Model expects rank 2 shape [num_images, 3] where each row is [temporal, height, width]
  if (image_grid_thw) {
    // Get shape info from image_grid_thw
    const int64_t* grid_thw_data{};
    const int64_t* grid_thw_shape{};
    size_t grid_thw_num_dims;
    CheckResult(OrtxGetTensorData(image_grid_thw, reinterpret_cast<const void**>(&grid_thw_data),
                                  &grid_thw_shape, &grid_thw_num_dims));
    
    // If rank is 3, reshape to rank 2 by removing singleton dimensions appropriately
    if (grid_thw_num_dims == 3) {
      // Calculate the target shape [num_images, 3]
      // The last dimension should be 3 (for t, h, w)
      // Find which dimension is 3 and which is the batch/image dimension
      std::vector<int64_t> target_shape;
      
      // Expected output: [num_images, 3]
      // Common input shapes: [1, 1, 3], [1, 3, 1], [num_images, 1, 3], etc.
      int64_t num_images = 1;
      int64_t feature_dim = 3;
      
      // Find dimension with value 3 (the feature dimension)
      int feature_dim_idx = -1;
      for (size_t i = 0; i < grid_thw_num_dims; ++i) {
        if (grid_thw_shape[i] == 3) {
          feature_dim_idx = i;
          break;
        }
      }
      
      if (feature_dim_idx == -1) {
        throw std::runtime_error("image_grid_thw tensor must have a dimension of size 3 for [t, h, w]");
      }
      
      // Calculate num_images from non-feature, non-singleton dimensions
      for (size_t i = 0; i < grid_thw_num_dims; ++i) {
        if (i != static_cast<size_t>(feature_dim_idx) && grid_thw_shape[i] > 1) {
          num_images *= grid_thw_shape[i];
        }
      }
      
      target_shape = {num_images, feature_dim};
      
      // Create reshaped tensor
      auto reshaped_tensor = OrtValue::CreateTensor<int64_t>(allocator, target_shape);
      int64_t* dst_data = reshaped_tensor->GetTensorMutableData<int64_t>();
      
      // Copy data (total elements remain the same)
      size_t total_elements = num_images * feature_dim;
      std::copy(grid_thw_data, grid_thw_data + total_elements, dst_data);
      
      named_tensors->emplace("image_grid_thw",
                             std::make_shared<Tensor>(std::move(reshaped_tensor)));
    } else if (grid_thw_num_dims == 2) {
      // Already rank 2, use as-is
      named_tensors->emplace("image_grid_thw",
                             std::make_shared<Tensor>(ProcessTensor<int64_t>(image_grid_thw, allocator)));
    } else {
      throw std::runtime_error("image_grid_thw tensor has unexpected rank: " + std::to_string(grid_thw_num_dims));
    }
  } else if (computed_image_grid_thw) {
    named_tensors->emplace("image_grid_thw",
                           std::make_shared<Tensor>(std::move(computed_image_grid_thw)));
  }

  named_tensors->emplace(std::string(Config::Defaults::NumImageTokens), std::make_shared<Tensor>(std::move(num_img_tokens)));

  return named_tensors;
}

}  // namespace Generators
