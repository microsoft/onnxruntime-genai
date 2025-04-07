// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <regex>

namespace Generators {

namespace {

std::unique_ptr<OrtValue> ProcessImagePrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                                             OrtxTensor* num_img_tokens, Ort::Allocator& allocator) {
  const int64_t *num_img_tokens_data{}, *num_img_tokens_shape{};
  size_t num_img_tokens_num_dims{};
  if (num_img_tokens) {
    CheckResult(OrtxGetTensorData(num_img_tokens, reinterpret_cast<const void**>(&num_img_tokens_data),
                                  &num_img_tokens_shape, &num_img_tokens_num_dims));
  }
  const int64_t num_images = num_img_tokens_data
                                 ? std::accumulate(num_img_tokens_shape,
                                                   num_img_tokens_shape + num_img_tokens_num_dims,
                                                   1LL, std::multiplies<int64_t>())
                                 : 0LL;

  // Split the prompt string based on the occurrences of the pattern "<|image_<number>|>"
  // Here the <number> represents the image id.
  const std::regex pattern("<\\|image_\\d+\\|>");
  const std::vector<std::string> prompt_chunks(
      std::sregex_token_iterator(prompt.begin(), prompt.end(), pattern, -1),
      std::sregex_token_iterator());

  // Each chunk of the prompt string obtained after splitting is then tokenized using the tokenizer.
  std::vector<std::vector<int32_t>> input_ids_chunks(prompt_chunks.size());
  for (size_t i = 0; i < prompt_chunks.size(); ++i) {
    input_ids_chunks[i] = tokenizer.Encode(prompt_chunks[i].c_str());
  }

  // Extract the image tags from the prompt string.
  // Here the image tags are of the form "<|image_<number>|>"
  const std::vector<std::string> image_tags(
      std::sregex_token_iterator(prompt.begin(), prompt.end(), pattern),
      std::sregex_token_iterator());

  // Extract the image ids from the image tags.
  std::vector<int32_t> image_ids(image_tags.size());
  constexpr size_t image_id_position_begin = 8;  // <|image_ : Character at idx 8 is the beginning of the image_id
  for (size_t i = 0; i < image_tags.size(); ++i) {
    const size_t image_id_position_end = image_tags[i].size() - 2;  // |> : Character at idx size() - 2 is '|' which marks the end of the image_id
    image_ids[i] = std::stoi(image_tags[i].substr(image_id_position_begin,
                                                  image_id_position_end - image_id_position_begin));
  }

  if (static_cast<int64_t>(std::set<int32_t>(image_ids.begin(), image_ids.end()).size()) != num_images) {
    throw std::runtime_error("Number of unique image tags does not match the number of images.");
  }

  // Construct the input_ids tensor by interleaving the input_ids_chunks and the image tokens placeholder
  // The image tokens placeholder is represented by a sequence of negative value of the image_ids.
  // For example, the placeholder for image_id 1 is represented by the value [-1, -1, -1, -1]. The
  // length of the sequence is determined by the value of num_img_tokens_data[image_id - 1].
  std::vector<int32_t> input_ids;
  for (size_t i = 0; i < input_ids_chunks.size(); ++i) {
    input_ids.insert(input_ids.end(), input_ids_chunks[i].begin(), input_ids_chunks[i].end());
    if (i < image_ids.size()) {
      if (image_ids[i] < 1 || image_ids[i] > static_cast<int32_t>(num_images)) {
        std::string error_message = "Encountered unexpected value of image_id in the prompt. Expected a value <= " +
                                    std::to_string(num_images) + ". Actual value: " + std::to_string(image_ids[i]);
        throw std::runtime_error(error_message);
      }
      for (int64_t j = 0; j < num_img_tokens_data[image_ids[i] - 1]; ++j) {
        input_ids.push_back(-image_ids[i]);
      }
    }
  }

  // input_ids is created. Pack it into an allocated OrtValue to avoid managing the memory.
  const std::vector<int64_t> shape{1, static_cast<int64_t>(input_ids.size())};
  auto input_ids_value = OrtValue::CreateTensor<int32_t>(allocator, shape);
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());
  return input_ids_value;
}

}  // namespace

PhiImageProcessor::PhiImageProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)} {
  auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
  config.AddMapping(std::string(Config::Defaults::ImageSizesName), config.model.vision.inputs.image_sizes);
}

std::unique_ptr<NamedTensors> PhiImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  std::string prompt = std::string(payload.prompt);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!images) {
    named_tensors->emplace(Config::Defaults::InputIdsName,
                           std::make_shared<Tensor>(ProcessImagePrompt(tokenizer, prompt, nullptr, allocator)));
    return named_tensors;
  }

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), images->images_.get(), result.ToBeAssigned()));

  OrtxTensor* pixel_values = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, &pixel_values));

  OrtxTensor* image_sizes = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 1, &image_sizes));

  OrtxTensor* num_img_tokens = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 2, &num_img_tokens));

  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(ProcessImagePrompt(tokenizer, prompt, num_img_tokens, allocator)));
  if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                           std::make_shared<Tensor>(ProcessTensor<float>(pixel_values, allocator)));
  } else {
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                           std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(pixel_values, allocator)));
  }
  named_tensors->emplace(std::string(Config::Defaults::ImageSizesName),
                         std::make_shared<Tensor>(ProcessTensor<int64_t>(image_sizes, allocator)));
  named_tensors->emplace(Config::Defaults::NumImageTokens,
                         std::make_shared<Tensor>(ProcessTensor<int64_t>(num_img_tokens, allocator)));

  return named_tensors;
}

}  // namespace Generators
