// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/model.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/lfm2_vl_image_processor.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace Generators {

namespace {

// Number of decoder tokens along one axis after the projector's pixel unshuffle.
// Matches `Lfm2VlProcessor._compute_tokens_for_image`, which rounds up so a patch grid that is not
// a multiple of the downsample factor still contributes a (partially padded) token.
int64_t DownsampledLength(int64_t patches, int64_t downsample_factor) {
  return (patches + downsample_factor - 1) / downsample_factor;
}

// Reads a [N, 2] int64 tensor of per-image (height, width) produced by the PixtralImageSizes step.
// The batched pixel_values tensor is zero-padded to the largest image, so these are the only
// reliable per-image dimensions.
std::vector<std::pair<int64_t, int64_t>> ReadImageSizes(OrtxTensor* image_sizes_tensor, int64_t num_images) {
  const void* raw{};
  const int64_t* shape{};
  size_t num_dims{};
  CheckResult(OrtxGetTensorData(image_sizes_tensor, &raw, &shape, &num_dims));

  if (num_dims != 2 || shape[1] != 2) {
    throw std::runtime_error("Lfm2VlImageProcessor: expected an image_sizes tensor of shape [N, 2].");
  }
  if (shape[0] != num_images) {
    throw std::runtime_error("Lfm2VlImageProcessor: image_sizes has " + std::to_string(shape[0]) +
                             " entries but pixel_values holds " + std::to_string(num_images) + " images.");
  }

  const int64_t* sizes = static_cast<const int64_t*>(raw);
  std::vector<std::pair<int64_t, int64_t>> image_sizes;
  image_sizes.reserve(static_cast<size_t>(num_images));
  for (int64_t i = 0; i < num_images; ++i) {
    image_sizes.emplace_back(sizes[i * 2], sizes[i * 2 + 1]);
  }
  return image_sizes;
}

std::unique_ptr<OrtValue> MakeInt64Tensor(const std::vector<int64_t>& values, std::vector<int64_t> shape,
                                          Ort::Allocator& allocator) {
  auto tensor = OrtValue::CreateTensor<int64_t>(allocator, shape);
  std::copy(values.begin(), values.end(), tensor->GetTensorMutableData<int64_t>());
  return tensor;
}

std::unique_ptr<OrtValue> MakeInputIds(const std::vector<int32_t>& input_ids, Ort::Allocator& allocator) {
  auto tensor = OrtValue::CreateTensor<int32_t>(allocator, std::vector<int64_t>{1, static_cast<int64_t>(input_ids.size())});
  std::copy(input_ids.begin(), input_ids.end(), tensor->GetTensorMutableData<int32_t>());
  return tensor;
}

// The vision graph's input names come from genai_config.json; a wrong name would otherwise be
// dropped silently by ExtraInputs and only surface as ORT's "Missing Input" at run time.
ONNXTensorElementDataType VisionInputType(const SessionInfo& session_info, const std::string& name,
                                          const char* config_field) {
  if (!session_info.HasInput(name)) {
    throw std::runtime_error("Lfm2VlImageProcessor: the vision model has no input named \"" + name +
                             "\". Point model.vision.inputs." + config_field +
                             " in genai_config.json at the name the vision model uses.");
  }
  return session_info.GetInputDataType(name);
}

// pixel_attention_mask and spatial_shapes are always emitted as int64, so fail at load rather than
// at the first image if the graph was exported with another integer type.
void RequireInt64VisionInput(const SessionInfo& session_info, const std::string& name, const char* config_field) {
  const auto type = VisionInputType(session_info, name, config_field);
  if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("Lfm2VlImageProcessor: vision input \"" + name + "\" must be int64, got " +
                             TypeToString(type) + ".");
  }
}

}  // namespace

void WriteLfm2VlImagePatches(const float* image, int64_t channels, int64_t padded_height, int64_t padded_width,
                             const Lfm2VlImageGeometry& geometry, int64_t encoder_patch_size,
                             float* destination) {
  const int64_t channel_stride = padded_height * padded_width;
  const int64_t patch_dim = encoder_patch_size * encoder_patch_size * channels;

  for (int64_t row = 0; row < geometry.patch_rows; ++row) {
    for (int64_t col = 0; col < geometry.patch_cols; ++col) {
      float* patch = destination + (row * geometry.patch_cols + col) * patch_dim;
      for (int64_t y = 0; y < encoder_patch_size; ++y) {
        const int64_t source_row = row * encoder_patch_size + y;
        for (int64_t x = 0; x < encoder_patch_size; ++x) {
          const int64_t source_col = col * encoder_patch_size + x;
          for (int64_t c = 0; c < channels; ++c) {
            patch[(y * encoder_patch_size + x) * channels + c] =
                image[c * channel_stride + source_row * padded_width + source_col];
          }
        }
      }
    }
  }
}

Lfm2VlImageGeometry ComputeLfm2VlImageGeometry(int64_t image_height, int64_t image_width,
                                               int64_t encoder_patch_size, int64_t downsample_factor) {
  if (encoder_patch_size <= 0 || downsample_factor <= 0) {
    throw std::runtime_error("Lfm2VlImageProcessor: patch_size and spatial_merge_size must be positive.");
  }
  if (image_height <= 0 || image_width <= 0) {
    throw std::runtime_error("Lfm2VlImageProcessor: image dimensions must be positive. Actual: " +
                             std::to_string(image_height) + "x" + std::to_string(image_width) + ".");
  }
  if (image_height % encoder_patch_size != 0 || image_width % encoder_patch_size != 0) {
    throw std::runtime_error("Lfm2VlImageProcessor: resized image (" + std::to_string(image_height) + "x" +
                             std::to_string(image_width) + ") is not a whole number of " +
                             std::to_string(encoder_patch_size) + "-pixel patches. Check that the Resize step in " +
                             "processor_config.json uses the same patch_size as model.vision in genai_config.json.");
  }

  Lfm2VlImageGeometry geometry;
  geometry.patch_rows = image_height / encoder_patch_size;
  geometry.patch_cols = image_width / encoder_patch_size;
  geometry.num_patches = geometry.patch_rows * geometry.patch_cols;
  geometry.num_tokens = DownsampledLength(geometry.patch_rows, downsample_factor) *
                        DownsampledLength(geometry.patch_cols, downsample_factor);
  return geometry;
}

std::string BuildLfm2VlImagePlaceholder(int64_t num_tokens) {
  const std::string image_token{kLfm2VlImageToken};
  std::string placeholder;
  placeholder.reserve(std::strlen(kLfm2VlImageStartToken) + std::strlen(kLfm2VlImageEndToken) +
                      image_token.size() * static_cast<size_t>(std::max<int64_t>(num_tokens, 0)));

  placeholder += kLfm2VlImageStartToken;
  for (int64_t i = 0; i < num_tokens; ++i) {
    placeholder += image_token;
  }
  placeholder += kLfm2VlImageEndToken;
  return placeholder;
}

std::string ExpandLfm2VlImageTokens(const std::string& prompt, const std::vector<int64_t>& tokens_per_image) {
  const std::string image_token{kLfm2VlImageToken};

  std::string expanded;
  expanded.reserve(prompt.size());
  size_t next_image = 0;
  size_t position = 0;
  while (true) {
    const size_t match = prompt.find(image_token, position);
    if (match == std::string::npos) {
      expanded.append(prompt, position, std::string::npos);
      break;
    }
    if (next_image == tokens_per_image.size()) {
      throw std::runtime_error("Prompt contains more " + image_token + " tokens than the " +
                               std::to_string(tokens_per_image.size()) + " images that were provided.");
    }
    expanded.append(prompt, position, match - position);
    expanded += BuildLfm2VlImagePlaceholder(tokens_per_image[next_image]);
    ++next_image;
    position = match + image_token.size();
  }

  // Images the prompt never referenced still have to be consumed by the decoder, otherwise the
  // vision features and the placeholder positions would not line up. Put them in front of the text.
  if (next_image < tokens_per_image.size()) {
    std::string leading;
    for (size_t i = next_image; i < tokens_per_image.size(); ++i) {
      leading += BuildLfm2VlImagePlaceholder(tokens_per_image[i]);
    }
    expanded.insert(0, leading);
  }

  return expanded;
}

Lfm2VlImageProcessor::Lfm2VlImageProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{VisionInputType(session_info, config.model.vision.inputs.pixel_values, "pixel_values")},
      encoder_patch_size_{config.model.vision.patch_size},
      downsample_factor_{config.model.vision.spatial_merge_size},
      max_num_patches_{config.model.vision.max_num_patches} {
  RequireInt64VisionInput(session_info, config.model.vision.inputs.attention_mask, "attention_mask");
  RequireInt64VisionInput(session_info, config.model.vision.inputs.image_sizes, "image_sizes");

  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
  config.AddMapping(std::string(Config::Defaults::ImageAttentionMaskName), config.model.vision.inputs.attention_mask);
  config.AddMapping(std::string(Config::Defaults::ImageSizesName), config.model.vision.inputs.image_sizes);
}

std::unique_ptr<NamedTensors> Lfm2VlImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const std::string prompt{payload.prompt};
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!images) {
    named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                           std::make_shared<Tensor>(MakeInputIds(tokenizer.Encode(prompt.c_str()), allocator)));
    // The pipeline reads num_image_tokens to skip the vision run.
    named_tensors->emplace(std::string(Config::Defaults::NumImageTokens),
                           std::make_shared<Tensor>(MakeInt64Tensor({0}, {1}, allocator)));
    return named_tensors;
  }

  // ort-extensions decodes, smart-resizes, rescales and normalizes each image, then stacks them
  // into one zero-padded [N, C, max_height, max_width] batch alongside the [N, 2] real sizes.
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), images->images_.get(), result.ToBeAssigned()));

  ort_extensions::OrtxObjectPtr<OrtxTensor> pixel_values_owner;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, pixel_values_owner.ToBeAssigned()));
  ort_extensions::OrtxObjectPtr<OrtxTensor> image_sizes_owner;
  CheckResult(OrtxTensorResultGetAt(result.get(), 1, image_sizes_owner.ToBeAssigned()));

  const float* pixels{};
  const int64_t* pixels_shape{};
  size_t pixels_num_dims{};
  CheckResult(OrtxGetTensorData(pixel_values_owner.get(), reinterpret_cast<const void**>(&pixels),
                                &pixels_shape, &pixels_num_dims));
  if (pixels_num_dims != 4) {
    throw std::runtime_error(
        "Lfm2VlImageProcessor: expected 4D pixel_values [N, C, H, W] from the image preprocessor, got " +
        std::to_string(pixels_num_dims) +
        "D. The pipeline in processor_config.json must end with Permute3D "
        "followed by PixtralImageSizes.");
  }

  const int64_t num_images = pixels_shape[0];
  const int64_t channels = pixels_shape[1];
  const int64_t padded_height = pixels_shape[2];
  const int64_t padded_width = pixels_shape[3];
  const auto image_sizes = ReadImageSizes(image_sizes_owner.get(), num_images);

  std::vector<Lfm2VlImageGeometry> geometries;
  geometries.reserve(image_sizes.size());
  int64_t longest_patch_sequence = 0;
  for (const auto& [height, width] : image_sizes) {
    if (height > padded_height || width > padded_width) {
      throw std::runtime_error("Lfm2VlImageProcessor: image_sizes reports " + std::to_string(height) + "x" +
                               std::to_string(width) + ", larger than the " + std::to_string(padded_height) + "x" +
                               std::to_string(padded_width) + " pixel_values batch it should be padded into.");
    }
    geometries.push_back(ComputeLfm2VlImageGeometry(height, width, encoder_patch_size_, downsample_factor_));
    longest_patch_sequence = std::max(longest_patch_sequence, geometries.back().num_patches);
  }

  // Every image is padded to the same patch count so they can share one vision run; the padded
  // positions are masked out of attention and dropped by the projector.
  const int64_t padded_patch_count = max_num_patches_ > 0 ? max_num_patches_ : longest_patch_sequence;
  for (size_t i = 0; i < geometries.size(); ++i) {
    if (geometries[i].num_patches > padded_patch_count) {
      throw std::runtime_error("Lfm2VlImageProcessor: image " + std::to_string(i) + " needs " +
                               std::to_string(geometries[i].num_patches) + " patches, more than the " +
                               std::to_string(padded_patch_count) +
                               " allowed by model.vision.max_num_patches in genai_config.json.");
    }
  }

  const int64_t patch_dim = encoder_patch_size_ * encoder_patch_size_ * channels;
  auto patched = OrtValue::CreateTensor<float>(allocator, std::vector<int64_t>{num_images, padded_patch_count, patch_dim});
  float* patched_data = patched->GetTensorMutableData<float>();
  std::fill_n(patched_data, static_cast<size_t>(num_images * padded_patch_count * patch_dim), 0.0f);

  std::vector<int64_t> attention_mask(static_cast<size_t>(num_images * padded_patch_count), 0);
  std::vector<int64_t> spatial_shapes(static_cast<size_t>(num_images * 2), 0);
  std::vector<int64_t> tokens_per_image;
  tokens_per_image.reserve(geometries.size());
  int64_t total_image_tokens = 0;

  for (int64_t i = 0; i < num_images; ++i) {
    const auto& geometry = geometries[static_cast<size_t>(i)];
    WriteLfm2VlImagePatches(pixels + i * channels * padded_height * padded_width, channels, padded_height,
                            padded_width, geometry, encoder_patch_size_, patched_data + i * padded_patch_count * patch_dim);
    std::fill_n(attention_mask.begin() + static_cast<size_t>(i * padded_patch_count),
                static_cast<size_t>(geometry.num_patches), 1);
    spatial_shapes[static_cast<size_t>(i * 2)] = geometry.patch_rows;
    spatial_shapes[static_cast<size_t>(i * 2 + 1)] = geometry.patch_cols;
    tokens_per_image.push_back(geometry.num_tokens);
    total_image_tokens += geometry.num_tokens;
  }

  const std::vector<int32_t> input_ids = tokenizer.Encode(ExpandLfm2VlImageTokens(prompt, tokens_per_image).c_str());
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(MakeInputIds(input_ids, allocator)));

  if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName), std::make_shared<Tensor>(std::move(patched)));
  } else {
    std::unique_ptr<OrtValue> converted;
    Cast(*patched, converted, *GetDeviceInterface(DeviceType::CPU), pixel_values_type_);
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName), std::make_shared<Tensor>(std::move(converted)));
  }

  named_tensors->emplace(
      std::string(Config::Defaults::ImageAttentionMaskName),
      std::make_shared<Tensor>(MakeInt64Tensor(attention_mask, {num_images, padded_patch_count}, allocator)));
  named_tensors->emplace(
      std::string(Config::Defaults::ImageSizesName),
      std::make_shared<Tensor>(MakeInt64Tensor(spatial_shapes, {num_images, 2}, allocator)));
  named_tensors->emplace(std::string(Config::Defaults::NumImageTokens),
                         std::make_shared<Tensor>(MakeInt64Tensor({total_image_tokens}, {1}, allocator)));

  return named_tensors;
}

}  // namespace Generators
