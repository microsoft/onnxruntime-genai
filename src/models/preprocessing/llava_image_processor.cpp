// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/model.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/llava_image_processor.h"

#include <algorithm>

namespace Generators {

namespace {

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

// The list overload of MultiModalProcessor::Process fills payload.prompts and leaves payload.prompt
// empty. Batching is not supported, so a single-entry list is the prompt.
std::string ResolvePrompt(const Payload& payload) {
  if (payload.prompts.empty()) {
    return payload.prompt;
  }
  if (payload.prompts.size() != 1) {
    throw std::runtime_error("LlavaImageProcessor: batched prompts are not supported; got " +
                             std::to_string(payload.prompts.size()) + " prompts. Pass a single prompt string.");
  }
  return payload.prompts[0] ? std::string(payload.prompts[0]) : std::string{};
}

}  // namespace

std::vector<int32_t> ExpandLlavaImageTokens(const std::vector<int32_t>& input_ids, int32_t image_token_id,
                                            int64_t tokens_per_image, int64_t num_images) {
  const auto num_markers = std::count(input_ids.begin(), input_ids.end(), image_token_id);
  if (num_markers != num_images) {
    throw std::runtime_error("LlavaImageProcessor: the prompt contains " + std::to_string(num_markers) +
                             " image tokens (id " + std::to_string(image_token_id) + ") but " +
                             std::to_string(num_images) + " images were provided. Put exactly one image token " +
                             "in the prompt per image.");
  }

  std::vector<int32_t> expanded;
  expanded.reserve(input_ids.size() + static_cast<size_t>(num_markers * std::max<int64_t>(tokens_per_image - 1, 0)));
  for (const int32_t id : input_ids) {
    if (id == image_token_id) {
      expanded.insert(expanded.end(), static_cast<size_t>(tokens_per_image), image_token_id);
    } else {
      expanded.push_back(id);
    }
  }
  return expanded;
}

LlavaImageProcessor::LlavaImageProcessor(Config& config, const SessionInfo& session_info)
    : image_token_id_{config.model.image_token_id},
      tokens_per_image_{config.model.vision.num_visual_tokens} {
  if (image_token_id_ <= 0) {
    throw std::runtime_error(
        "LlavaImageProcessor: set model.image_token_id in genai_config.json to the id of the "
        "image placeholder token.");
  }

  const auto& pixel_values = config.model.vision.inputs.pixel_values;
  if (!session_info.HasInput(pixel_values)) {
    throw std::runtime_error("LlavaImageProcessor: the vision model has no input named \"" + pixel_values +
                             "\". Point model.vision.inputs.pixel_values in genai_config.json at its name.");
  }
  pixel_values_type_ = session_info.GetInputDataType(pixel_values);

  const auto& image_features = config.model.vision.outputs.image_features;
  if (!session_info.HasOutput(image_features)) {
    throw std::runtime_error("LlavaImageProcessor: the vision model has no output named \"" + image_features +
                             "\". Point model.vision.outputs.image_features in genai_config.json at its name.");
  }
  const auto features_shape = session_info.GetOutputShape(image_features);
  batched_features_ = features_shape.size() == 3;
  if (tokens_per_image_ <= 0) {
    if (!batched_features_ || features_shape[1] <= 0) {
      throw std::runtime_error(
          "LlavaImageProcessor: the number of image tokens per image cannot be read from the vision model's "
          "output shape. Set model.vision.num_visual_tokens in genai_config.json.");
    }
    tokens_per_image_ = features_shape[1];
  } else if (batched_features_ && features_shape[1] > 0 && features_shape[1] != tokens_per_image_) {
    throw std::runtime_error("LlavaImageProcessor: model.vision.num_visual_tokens is " +
                             std::to_string(tokens_per_image_) + " but the vision model emits " +
                             std::to_string(features_shape[1]) + " tokens per image.");
  }

  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
}

std::unique_ptr<NamedTensors> LlavaImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const std::string prompt = ResolvePrompt(payload);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  int64_t num_images = 0;
  if (images && images->num_images_ > 0) {
    ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
    CheckResult(OrtxImagePreProcess(processor_.get(), images->images_.get(), result.ToBeAssigned()));

    ort_extensions::OrtxObjectPtr<OrtxTensor> pixel_values;
    CheckResult(OrtxTensorResultGetAt(result.get(), 0, pixel_values.ToBeAssigned()));

    const void* data{};
    const int64_t* shape{};
    size_t num_dims{};
    CheckResult(OrtxGetTensorData(pixel_values.get(), &data, &shape, &num_dims));
    if (num_dims != 4) {
      throw std::runtime_error(
          "LlavaImageProcessor: expected 4D pixel_values [N, C, H, W] from the image "
          "preprocessor, got " +
          std::to_string(num_dims) +
          "D. The pipeline in the processor config must end with a channel-first Permute3D.");
    }
    num_images = shape[0];
    EmplaceProcessedTensor(thread_pool_, *named_tensors, Config::Defaults::PixelValuesName, pixel_values.get(),
                           pixel_values_type_, allocator);
  }

  const auto input_ids = ExpandLlavaImageTokens(tokenizer.Encode(prompt.c_str()), image_token_id_, tokens_per_image_,
                                                num_images);
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(MakeInputIds(input_ids, allocator)));

  // The vision and embedding states size the image feature buffer from this: tokens per image for a
  // [num_images, num_tokens, hidden] output, all tokens for a [total_tokens, hidden] one.
  const int64_t num_image_tokens = num_images == 0 ? 0 : (batched_features_ ? tokens_per_image_ : num_images * tokens_per_image_);
  named_tensors->emplace(std::string(Config::Defaults::NumImageTokens),
                         std::make_shared<Tensor>(MakeInt64Tensor({num_image_tokens}, {1}, allocator)));

  return named_tensors;
}

}  // namespace Generators
