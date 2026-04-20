// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "mistral3_image_processor.h"

namespace Generators {

namespace {

// Pixtral special tokens — resolved at runtime via tokenizer lookup.
constexpr char kImgToken[] = "[IMG]";
constexpr char kImgBreakToken[] = "[IMG_BREAK]";
constexpr char kImgEndToken[] = "[IMG_END]";
constexpr char kInstToken[] = "[INST]";

// Build input_ids for the image portion of the prompt.
// Returns the token IDs including [IMG], [IMG_BREAK], and [IMG_END].
std::vector<int32_t> BuildImageTokenSequence(int patch_rows, int patch_cols,
                                             int32_t img_id, int32_t break_id, int32_t end_id) {
  std::vector<int32_t> tokens;
  tokens.reserve(patch_rows * patch_cols + patch_rows);

  for (int r = 0; r < patch_rows; ++r) {
    for (int c = 0; c < patch_cols; ++c) {
      tokens.push_back(img_id);
    }
    if (r < patch_rows - 1) {
      tokens.push_back(break_id);
    } else {
      tokens.push_back(end_id);
    }
  }
  return tokens;
}

// Per-image dimensions: each image may have a different resolution after
// smart_resize.  When image_sizes is available (from PixtralImageSizes),
// use per-image H/W.  Otherwise fall back to the (padded) pixel_values shape.
struct PerImageInfo {
  int patch_rows;
  int patch_cols;
  int64_t num_img_tokens;  // [IMG] count only (excludes [IMG_BREAK]/[IMG_END])
  std::vector<int32_t> token_sequence;
};

std::tuple<std::unique_ptr<OrtValue>, int64_t>
ProcessPixtralPrompt(const Tokenizer& tokenizer, const std::string& prompt,
                     OrtxTensor* pixel_values, OrtxTensor* image_sizes_tensor,
                     int patch_size, int spatial_merge_size,
                     Ort::Allocator& allocator) {
  const int32_t img_token_id = tokenizer.TokenToTokenId(kImgToken);
  const int32_t img_break_id = tokenizer.TokenToTokenId(kImgBreakToken);
  const int32_t img_end_id = tokenizer.TokenToTokenId(kImgEndToken);
  const int32_t inst_token_id = tokenizer.TokenToTokenId(kInstToken);

  int64_t num_images = 0;
  std::vector<PerImageInfo> image_infos;

  if (pixel_values) {
    const float* data{};
    const int64_t* shape{};
    size_t num_dims{};
    CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&data), &shape, &num_dims));
    if (num_dims != 4) {
      throw std::runtime_error(
          "Mistral3ImageProcessor: expected 4D pixel_values [N,C,H,W], "
          "got " +
          std::to_string(num_dims) + "D tensor.");
    }
    num_images = shape[0];
    int64_t padded_h = shape[2];
    int64_t padded_w = shape[3];

    // Read per-image sizes if available, otherwise use padded dimensions
    const int64_t* sizes_data = nullptr;
    if (image_sizes_tensor) {
      const void* raw{};
      const int64_t* sizes_shape{};
      size_t sizes_dims{};
      CheckResult(OrtxGetTensorData(image_sizes_tensor, &raw, &sizes_shape, &sizes_dims));

      if (sizes_dims != 2) {
        throw std::runtime_error(
            "Mistral3ImageProcessor: expected 2D image_sizes tensor [N,2], "
            "got " +
            std::to_string(sizes_dims) + "D tensor.");
      }
      if (sizes_shape[1] != 2) {
        throw std::runtime_error(
            "Mistral3ImageProcessor: expected image_sizes tensor shape [N,2], "
            "got second dimension " +
            std::to_string(sizes_shape[1]) + ".");
      }
      if (sizes_shape[0] != num_images) {
        throw std::runtime_error(
            "Mistral3ImageProcessor: image_sizes tensor first dimension (" +
            std::to_string(sizes_shape[0]) + ") must match pixel_values batch size (" +
            std::to_string(num_images) + ").");
      }
      sizes_data = static_cast<const int64_t*>(raw);
    }

    int64_t effective_patch = static_cast<int64_t>(patch_size) * spatial_merge_size;
    for (int64_t i = 0; i < num_images; ++i) {
      int64_t h = sizes_data ? sizes_data[i * 2] : padded_h;
      int64_t w = sizes_data ? sizes_data[i * 2 + 1] : padded_w;

      if (h % effective_patch != 0 || w % effective_patch != 0) {
        throw std::runtime_error(
            "Mistral3ImageProcessor: image " + std::to_string(i) + " dimensions (" +
            std::to_string(h) + "x" + std::to_string(w) +
            ") must be divisible by patch_size*merge_size (" +
            std::to_string(effective_patch) + "). Check smart_resize configuration.");
      }

      PerImageInfo info;
      info.patch_rows = static_cast<int>(h / effective_patch);
      info.patch_cols = static_cast<int>(w / effective_patch);
      info.token_sequence = BuildImageTokenSequence(info.patch_rows, info.patch_cols,
                                                    img_token_id, img_break_id, img_end_id);
      // Count only [IMG] tokens — this equals the vision model's feature output count
      // (patch_rows * patch_cols), excluding structural [IMG_BREAK]/[IMG_END] tokens.
      info.num_img_tokens = static_cast<int64_t>(
          std::count(info.token_sequence.begin(), info.token_sequence.end(), img_token_id));
      image_infos.push_back(std::move(info));
    }
  }

  int64_t total_img_tokens = 0;
  for (const auto& info : image_infos) {
    total_img_tokens += info.num_img_tokens;
  }

  // Tokenize the text prompt
  std::vector<int32_t> input_ids;
  if (!prompt.empty()) {
    input_ids = tokenizer.Encode(prompt.c_str());
  }

  // Expand [IMG] placeholders for each image.
  // Each [IMG] (or group of consecutive [IMG] tokens) in the prompt corresponds
  // to one image, expanded with its per-image token sequence.
  if (!image_infos.empty()) {
    std::vector<int32_t> expanded_ids;
    size_t total_expansion = input_ids.size();
    for (const auto& info : image_infos) {
      total_expansion += info.token_sequence.size();
    }
    expanded_ids.reserve(total_expansion);

    size_t next_image = 0;
    for (size_t i = 0; i < input_ids.size(); ++i) {
      if (input_ids[i] == img_token_id && next_image < image_infos.size()) {
        // Replace this [IMG] (and consecutive [IMG] tokens) with the image's token sequence
        expanded_ids.insert(expanded_ids.end(),
                            image_infos[next_image].token_sequence.begin(),
                            image_infos[next_image].token_sequence.end());
        ++next_image;
        // Skip consecutive [IMG] tokens from the original prompt
        while (i + 1 < input_ids.size() && input_ids[i + 1] == img_token_id) {
          ++i;
        }
      } else {
        expanded_ids.push_back(input_ids[i]);
      }
    }

    // If not all images had placeholders, insert remaining after [INST]
    if (next_image < image_infos.size()) {
      std::vector<int32_t> remaining_tokens;
      for (size_t img = next_image; img < image_infos.size(); ++img) {
        remaining_tokens.insert(remaining_tokens.end(),
                                image_infos[img].token_sequence.begin(),
                                image_infos[img].token_sequence.end());
      }

      std::vector<int32_t> final_ids;
      final_ids.reserve(expanded_ids.size() + remaining_tokens.size());
      bool inserted = false;
      for (size_t i = 0; i < expanded_ids.size(); ++i) {
        final_ids.push_back(expanded_ids[i]);
        if (expanded_ids[i] == inst_token_id && !inserted) {
          final_ids.insert(final_ids.end(), remaining_tokens.begin(), remaining_tokens.end());
          inserted = true;
        }
      }
      if (!inserted) {
        // No [INST] found — prepend remaining image tokens
        final_ids.clear();
        final_ids.insert(final_ids.end(), remaining_tokens.begin(), remaining_tokens.end());
        final_ids.insert(final_ids.end(), expanded_ids.begin(), expanded_ids.end());
      }
      expanded_ids = std::move(final_ids);
    }

    input_ids = std::move(expanded_ids);
  }

  auto input_ids_value = OrtValue::CreateTensor<int32_t>(
      allocator, std::vector<int64_t>{1, static_cast<int64_t>(input_ids.size())});
  std::copy(input_ids.begin(), input_ids.end(),
            input_ids_value->GetTensorMutableData<int32_t>());

  return {std::move(input_ids_value), total_img_tokens};
}
}  // namespace

Mistral3ImageProcessor::Mistral3ImageProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)},
      patch_size_{config.model.vision.patch_size},
      spatial_merge_size_{config.model.vision.spatial_merge_size} {
  const auto processor_config =
      (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName),
                    config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName),
                    config.model.vision.inputs.pixel_values);
}

std::unique_ptr<NamedTensors> Mistral3ImageProcessor::Process(
    const Tokenizer& tokenizer, const Payload& payload) const {
  std::string prompt = std::string(payload.prompt);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!images) {
    // Text-only: tokenize prompt without image processing
    auto [input_ids, num_img_tokens] =
        ProcessPixtralPrompt(tokenizer, prompt, nullptr, nullptr, patch_size_,
                             spatial_merge_size_, allocator);
    named_tensors->emplace(Config::Defaults::InputIdsName,
                           std::make_shared<Tensor>(std::move(input_ids)));

    // Explicitly set num_image_tokens=0 for text-only inputs so downstream
    // pipeline components know there are no vision features to process.
    auto zero_tokens = OrtValue::CreateTensor<int64_t>(allocator, std::vector<int64_t>{1});
    zero_tokens->GetTensorMutableData<int64_t>()[0] = 0;
    named_tensors->emplace(std::string(Config::Defaults::NumImageTokens),
                           std::make_shared<Tensor>(std::move(zero_tokens)));
    return named_tensors;
  }

  // Process images through the ort-extensions processor (normalization, resizing)
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), images->images_.get(),
                                  result.ToBeAssigned()));

  OrtxTensor* pixel_values = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, &pixel_values));

  // Tensor 1: image_sizes[N, 2] from PixtralImageSizes (post-resize, pre-padding).
  // Only present when processor_config.json includes the PixtralImageSizes step.
  OrtxTensor* image_sizes = nullptr;
  if (OrtxTensorResultGetAt(result.get(), 1, &image_sizes) != kOrtxOK) {
    image_sizes = nullptr;
  }

  auto [input_ids, num_img_tokens] =
      ProcessPixtralPrompt(tokenizer, prompt, pixel_values, image_sizes, patch_size_,
                           spatial_merge_size_, allocator);

  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(std::move(input_ids)));

  // Convert pixel_values to the vision model's expected dtype (NCHW layout
  // is already handled by the Permute3D step in processor_config.json).
  {
    std::unique_ptr<OrtValue> pv_ortvalue;
    if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      pv_ortvalue = ProcessTensor<float>(pixel_values, allocator);
    } else if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      pv_ortvalue = ProcessTensor<Ort::BFloat16_t>(pixel_values, allocator);
    } else {
      pv_ortvalue = ProcessTensor<Ort::Float16_t>(pixel_values, allocator);
    }
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                           std::make_shared<Tensor>(std::move(pv_ortvalue)));
  }

  // Add image_sizes[N, 2] for PixtralVisionState to slice per-image dimensions
  if (image_sizes) {
    named_tensors->emplace(std::string(Config::Defaults::ImageSizesName),
                           std::make_shared<Tensor>(ProcessTensor<int64_t>(image_sizes, allocator)));
  }

  // Add num_image_tokens (total across all images) for the embedding model
  auto num_img_tokens_value = OrtValue::CreateTensor<int64_t>(
      allocator, std::vector<int64_t>{1});
  num_img_tokens_value->GetTensorMutableData<int64_t>()[0] = num_img_tokens;
  named_tensors->emplace(std::string(Config::Defaults::NumImageTokens),
                         std::make_shared<Tensor>(std::move(num_img_tokens_value)));

  return named_tensors;
}

}  // namespace Generators
