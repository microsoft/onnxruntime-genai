// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "qwen2_5_vl_image_processor.h"
#include <numeric>
#include <regex>

namespace Generators {

namespace {

// Helper to convert float32 tensor to target type (float16 or bfloat16)
std::unique_ptr<OrtValue> ConvertPixelValues(const OrtValue& float_tensor,
                                             ONNXTensorElementDataType target_type,
                                             Ort::Allocator& allocator) {
  if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    // No conversion needed, return a copy
    auto shape = float_tensor.GetTensorTypeAndShapeInfo()->GetShape();
    auto result = OrtValue::CreateTensor<float>(allocator, shape);
    const float* src = float_tensor.GetTensorData<float>();
    float* dst = result->GetTensorMutableData<float>();
    size_t count = float_tensor.GetTensorTypeAndShapeInfo()->GetElementCount();
    std::copy(src, src + count, dst);
    return result;
  }

  auto shape = float_tensor.GetTensorTypeAndShapeInfo()->GetShape();
  size_t count = float_tensor.GetTensorTypeAndShapeInfo()->GetElementCount();

  std::unique_ptr<OrtValue> result;
  if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    result = OrtValue::CreateTensor<Ort::BFloat16_t>(allocator, shape);
  } else if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    result = OrtValue::CreateTensor<Ort::Float16_t>(allocator, shape);
  } else {
    throw std::runtime_error("Unsupported target type for pixel values conversion");
  }

  // Use CPU device Cast method for optimized conversion
  auto* cpu_device = GetDeviceInterface(DeviceType::CPU);
  void* input_data = const_cast<void*>(static_cast<const void*>(float_tensor.GetTensorData<float>()));
  void* output_data = result->GetTensorMutableRawData();
  cpu_device->Cast(input_data, output_data, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, target_type, count);

  return result;
}

std::tuple<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>>
ProcessImagePrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                   OrtxTensor* pixel_values, OrtxTensor* image_grid_thw,
                   const int64_t* computed_grid_data, int64_t computed_grid_num_images,
                   Ort::Allocator& allocator, int64_t spatial_merge_size) {
  constexpr char vision_start_token[] = "<|vision_start|>";
  constexpr char vision_end_token[] = "<|vision_end|>";
  constexpr char image_pad_token[] = "<|image_pad|>";

  int64_t num_images = 0;
  int64_t total_image_tokens = 0;
  const int64_t* image_grid_thw_data = nullptr;

  if (pixel_values) {
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
      int64_t tokens = (t * h * w) / (spatial_merge_size * spatial_merge_size);
      total_image_tokens += tokens;
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
      int64_t num_pads = (t * h * w) / (spatial_merge_size * spatial_merge_size);

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
    : pixel_values_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},  // Default to float, will be determined at runtime if vision session exists
      spatial_merge_size_{config.model.vision.spatial_merge_size} {
  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  // Try to get pixel values type from session info if vision session exists (for MultiModalLanguageModel)
  // For pipeline models like Qwen2_5_VL_PipelineModel, vision session won't be in session_info
  try {
    pixel_values_type_ = session_info.GetInputDataType(config.model.vision.inputs.pixel_values);
  } catch (...) {
    // Vision session not in session_info (e.g., for pipeline models), keep default float type
  }

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
}

std::unique_ptr<NamedTensors> QwenImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  std::string prompt = std::string(payload.prompt);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!images || images->num_images_ == 0) {
    auto [input_ids, num_img_tokens] = ProcessImagePrompt(tokenizer, prompt, nullptr, nullptr, nullptr, 0, allocator, spatial_merge_size_);
    named_tensors->emplace(std::string(Config::Defaults::InputIdsName), std::make_shared<Tensor>(std::move(input_ids)));
    named_tensors->emplace(std::string(Config::Defaults::NumImageTokens), std::make_shared<Tensor>(std::move(num_img_tokens)));
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

  // Check if pixel_values needs patching (shape should be [1, height, width, channels] in HWC format)
  if (pixel_values_num_dims == 4 && pixel_values_shape[0] == 1) {
    constexpr int64_t kPatchSize = 14;
    constexpr int64_t kTemporalPatchSize = 2;

    int64_t height = pixel_values_shape[1];  // HWC: [batch, height, width, channels]
    int64_t width = pixel_values_shape[2];
    int64_t channels = pixel_values_shape[3];

    int64_t height_patches = height / kPatchSize;
    int64_t width_patches = width / kPatchSize;
    int64_t total_patches = height_patches * width_patches;
    int64_t patch_dim = channels * kTemporalPatchSize * kPatchSize * kPatchSize;

    // Create patched pixel_values: [1, total_patches, patch_dim] for NPU pipeline compatibility
    // NPU pipeline expects rank 3, CUDA/CPU models will squeeze if needed
    patched_pixel_values = OrtValue::CreateTensor<float>(
        allocator, std::vector<int64_t>{1, total_patches, patch_dim});
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
                                                        image_grid_thw, computed_grid_data, computed_grid_num_images, allocator, spatial_merge_size_);
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName), std::make_shared<Tensor>(std::move(input_ids)));

  // Use patched pixel_values if we computed it, otherwise use processor output
  if (patched_pixel_values) {
    auto converted_tensor = ConvertPixelValues(*patched_pixel_values, pixel_values_type_, allocator);
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                           std::make_shared<Tensor>(std::move(converted_tensor)));
  } else {
    // For non-patched pixel_values, we need to handle potential batch dimension from PatchImage extension
    // Model expects [num_patches, 1176] but extension might return [batch, num_patches, 1176]
    const void* pixel_data{};
    const int64_t* pixel_shape{};
    size_t pixel_num_dims;
    CheckResult(OrtxGetTensorData(pixel_values, &pixel_data, &pixel_shape, &pixel_num_dims));

    // Squeeze out leading dimension of size 1 if present
    std::vector<int64_t> pixel_target_shape;
    size_t squeeze_offset = 0;
    if (pixel_num_dims >= 3 && pixel_shape[0] == 1) {
      // Skip the batch dimension
      squeeze_offset = 1;
    }
    for (size_t i = squeeze_offset; i < pixel_num_dims; ++i) {
      pixel_target_shape.push_back(pixel_shape[i]);
    }

    int64_t num_pixel_elements = std::accumulate(pixel_target_shape.begin(), pixel_target_shape.end(), 1LL, std::multiplies<int64_t>());

    // Create temporary float tensor from processor output
    auto float_tensor = OrtValue::CreateTensor<float>(allocator, pixel_target_shape);
    std::copy(static_cast<const float*>(pixel_data),
              static_cast<const float*>(pixel_data) + num_pixel_elements,
              float_tensor->GetTensorMutableData<float>());

    // Convert to target type
    auto converted_tensor = ConvertPixelValues(*float_tensor, pixel_values_type_, allocator);
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                           std::make_shared<Tensor>(std::move(converted_tensor)));
  }

  // Add image_grid_thw tensor (either from processor or computed)
  if (image_grid_thw) {
    // Get the tensor data and shape from OrtxTensor
    const int64_t* grid_data{};
    const int64_t* grid_shape{};
    size_t grid_num_dims;
    CheckResult(OrtxGetTensorData(image_grid_thw, reinterpret_cast<const void**>(&grid_data),
                                  &grid_shape, &grid_num_dims));

    // The vision model expects shape [num_images, 3], but PatchImage might return [batch, num_images, 3]
    // Squeeze out leading dimension of size 1
    std::vector<int64_t> grid_target_shape;
    size_t grid_squeeze_offset = 0;
    if (grid_num_dims >= 3 && grid_shape[0] == 1) {
      // Skip the batch dimension
      grid_squeeze_offset = 1;
    }
    for (size_t i = grid_squeeze_offset; i < grid_num_dims; ++i) {
      grid_target_shape.push_back(grid_shape[i]);
    }

    // Ensure we have rank 2 [num_images, 3]
    if (grid_target_shape.size() != 2 || grid_target_shape[1] != 3) {
      throw std::runtime_error("image_grid_thw must have shape [num_images, 3], got shape with " +
                               std::to_string(grid_target_shape.size()) + " dimensions");
    }

    int64_t num_grid_elements = std::accumulate(grid_target_shape.begin(), grid_target_shape.end(), 1LL, std::multiplies<int64_t>());
    auto grid_tensor = OrtValue::CreateTensor<int64_t>(allocator, grid_target_shape);
    std::copy(grid_data, grid_data + num_grid_elements, grid_tensor->GetTensorMutableData<int64_t>());

    named_tensors->emplace("image_grid_thw", std::make_shared<Tensor>(std::move(grid_tensor)));
  } else if (computed_image_grid_thw) {
    named_tensors->emplace("image_grid_thw",
                           std::make_shared<Tensor>(std::move(computed_image_grid_thw)));
  }

  named_tensors->emplace(std::string(Config::Defaults::NumImageTokens), std::make_shared<Tensor>(std::move(num_img_tokens)));

  return named_tensors;
}

}  // namespace Generators
