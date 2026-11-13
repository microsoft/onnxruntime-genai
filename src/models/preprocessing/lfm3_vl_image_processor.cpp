// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/model.h"
#include "models/parallel_utils.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/lfm3_vl_image_processor.h"

#include <algorithm>

namespace Generators {

namespace {

constexpr char kProcessorName[] = "Lfm3VlImageProcessor";
// The checkpoint's processor repeats each image along time to fill one temporal patch.
constexpr int64_t kTemporalPatchSize = 2;
constexpr int64_t kChannels = 3;

}  // namespace

void CompactLfm3VlImageRows(ThreadPool* thread_pool, const float* batch, int64_t longest_image_rows, int64_t row_width,
                            const std::vector<int64_t>& image_rows, float* destination) {
  for (size_t i = 0; i < image_rows.size(); ++i) {
    const size_t count = static_cast<size_t>(image_rows[i] * row_width);
    ParallelCopy(thread_pool, batch + static_cast<int64_t>(i) * longest_image_rows * row_width, destination, count);
    destination += count;
  }
}

std::string ExpandLfm3VlImageTokens(const std::string& prompt, const std::vector<int64_t>& tokens_per_image) {
  if (prompt.find(kLfm3VlVideoToken) != std::string::npos) {
    throw std::runtime_error(std::string(kProcessorName) + ": the prompt contains " + kLfm3VlVideoToken +
                             ", but video input is not supported.");
  }
  const std::string image_token{kLfm3VlImageToken};
  size_t num_markers = 0;
  for (size_t match = prompt.find(image_token); match != std::string::npos;
       match = prompt.find(image_token, match + image_token.size())) {
    ++num_markers;
  }
  // One marker per image, in image order, as Lfm3VlProcessor requires; anything else would misalign
  // the vision features with the decoder positions.
  if (num_markers != tokens_per_image.size()) {
    throw std::runtime_error(std::string(kProcessorName) + ": the prompt contains " + std::to_string(num_markers) +
                             " " + image_token + " tokens but " + std::to_string(tokens_per_image.size()) +
                             " images were provided. Put exactly one " + image_token + " in the prompt per image.");
  }

  std::string expanded;
  size_t next_image = 0;
  size_t position = 0;
  for (size_t match = prompt.find(image_token); match != std::string::npos; match = prompt.find(image_token, position)) {
    expanded.append(prompt, position, match - position);
    for (int64_t i = 0; i < tokens_per_image[next_image]; ++i) {
      expanded += image_token;
    }
    ++next_image;
    position = match + image_token.size();
  }
  expanded.append(prompt, position, std::string::npos);
  return expanded;
}

Lfm3VlImageProcessor::Lfm3VlImageProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{VisionInputType(session_info, config.model.vision.inputs.pixel_values, "pixel_values",
                                         kProcessorName)},
      row_width_{kChannels * kTemporalPatchSize * config.model.vision.patch_size * config.model.vision.patch_size},
      merge_size_{config.model.vision.spatial_merge_size} {
  RequireInt64VisionInput(session_info, config.model.vision.inputs.image_grid_thw, "image_grid_thw", kProcessorName);
  if (merge_size_ <= 0) {
    throw std::runtime_error(std::string(kProcessorName) + ": model.vision.spatial_merge_size must be positive, got " +
                             std::to_string(merge_size_) + ".");
  }

  // A graph built for another patch size would otherwise fail inside ORT at the first image.
  const auto pixel_values_shape = session_info.GetInputShape(config.model.vision.inputs.pixel_values);
  if (pixel_values_shape.size() != 2 || pixel_values_shape[1] != row_width_) {
    throw std::runtime_error(std::string(kProcessorName) + ": the vision model's pixel_values must be [num_patches, " +
                             std::to_string(row_width_) + "] for model.vision.patch_size " +
                             std::to_string(config.model.vision.patch_size) + ".");
  }

  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
  config.AddMapping(std::string(Config::Defaults::ImageGridThwName), config.model.vision.inputs.image_grid_thw);
}

std::unique_ptr<NamedTensors> Lfm3VlImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const std::string prompt = ResolvePrompt(payload, kProcessorName);
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!payload.images) {
    const std::string text = ExpandLfm3VlImageTokens(prompt, {});
    named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                           std::make_shared<Tensor>(MakeInputIds(tokenizer.Encode(text.c_str()), allocator)));
    named_tensors->emplace(std::string(Config::Defaults::NumImageTokens),
                           std::make_shared<Tensor>(MakeInt64Tensor({0}, {1}, allocator)));
    return named_tensors;
  }

  // ort-extensions decodes, smart-resizes, rescales, normalizes and patches each image, then stacks them
  // into a [N, longest_image_rows, row_width] batch, zero-padded in rows, and a [N, 1, 3] grid.
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), payload.images->images_.get(), result.ToBeAssigned()));
  ort_extensions::OrtxObjectPtr<OrtxTensor> pixel_values_owner;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, pixel_values_owner.ToBeAssigned()));
  ort_extensions::OrtxObjectPtr<OrtxTensor> grid_owner;
  CheckResult(OrtxTensorResultGetAt(result.get(), 1, grid_owner.ToBeAssigned()));

  const float* pixels{};
  const int64_t* pixels_shape{};
  size_t pixels_num_dims{};
  CheckResult(OrtxGetTensorData(pixel_values_owner.get(), reinterpret_cast<const void**>(&pixels),
                                &pixels_shape, &pixels_num_dims));
  const int64_t* grid{};
  const int64_t* grid_shape{};
  size_t grid_num_dims{};
  CheckResult(OrtxGetTensorData(grid_owner.get(), reinterpret_cast<const void**>(&grid), &grid_shape, &grid_num_dims));
  if (pixels_num_dims != 3 || pixels_shape[2] != row_width_ || grid_num_dims != 3 || grid_shape[0] != pixels_shape[0] ||
      grid_shape[1] != 1 || grid_shape[2] != 3) {
    throw std::runtime_error(std::string(kProcessorName) + ": expected pixel_values [N, rows, " +
                             std::to_string(row_width_) + "] and image_grid_thw [N, 1, 3] from the image preprocessor. " +
                             "The pipeline in processor_config.json must end with PatchImage, using the patch_size of " +
                             "model.vision in genai_config.json.");
  }

  const int64_t num_images = pixels_shape[0];
  std::vector<int64_t> image_rows;
  std::vector<int64_t> tokens_per_image;
  int64_t total_rows = 0;
  int64_t total_tokens = 0;
  for (int64_t i = 0; i < num_images; ++i) {
    const int64_t rows = grid[i * 3 + 1] * grid[i * 3 + 2];
    image_rows.push_back(rows);
    tokens_per_image.push_back(rows / (merge_size_ * merge_size_));
    total_rows += rows;
    total_tokens += tokens_per_image.back();
  }

  auto patched = OrtValue::CreateTensor<float>(allocator, std::vector<int64_t>{total_rows, row_width_});
  CompactLfm3VlImageRows(thread_pool_, pixels, pixels_shape[1], row_width_, image_rows,
                         patched->GetTensorMutableData<float>());

  const std::vector<int32_t> input_ids = tokenizer.Encode(ExpandLfm3VlImageTokens(prompt, tokens_per_image).c_str());
  // A tokenizer that splits the marker into pieces would hand the embedding model fewer slots than features.
  const int32_t image_token_id = tokenizer.TokenToTokenId(kLfm3VlImageToken);
  const auto image_token_count = std::count(input_ids.begin(), input_ids.end(), image_token_id);
  if (image_token_count != total_tokens) {
    throw std::runtime_error(std::string(kProcessorName) + ": the prompt encodes to " +
                             std::to_string(image_token_count) + " " + kLfm3VlImageToken + " ids but the images need " +
                             std::to_string(total_tokens) + ". The tokenizer must keep " + kLfm3VlImageToken +
                             " as one special token.");
  }
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(MakeInputIds(input_ids, allocator)));

  if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName), std::make_shared<Tensor>(std::move(patched)));
  } else {
    std::unique_ptr<OrtValue> converted;
    Cast(*patched, converted, *GetDeviceInterface(DeviceType::CPU), pixel_values_type_);
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName), std::make_shared<Tensor>(std::move(converted)));
  }
  named_tensors->emplace(std::string(Config::Defaults::ImageGridThwName),
                         std::make_shared<Tensor>(MakeInt64Tensor(std::vector<int64_t>(grid, grid + num_images * 3),
                                                                  {num_images, 3}, allocator)));
  named_tensors->emplace(std::string(Config::Defaults::NumImageTokens),
                         std::make_shared<Tensor>(MakeInt64Tensor({total_tokens}, {1}, allocator)));
  return named_tensors;
}

}  // namespace Generators
