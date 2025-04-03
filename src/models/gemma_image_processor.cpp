// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <regex>

namespace Generators {

namespace {

std::tuple<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>>
ProcessImagePrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                   OrtxTensor* pixel_values, Ort::Allocator& allocator) {
  constexpr char boi_token[] = "<start_of_image>";
  constexpr char image_token[] = "<image_soft_token>";
  constexpr char eoi_token[] = "<end_of_image>";
  constexpr size_t image_seq_length = 256;

  int64_t num_images{};
  if (pixel_values) {
    const float* pixel_values_data{};
    const int64_t* pixel_values_shape{};
    size_t pixel_values_num_dims;
    CheckResult(OrtxGetTensorData(pixel_values, reinterpret_cast<const void**>(&pixel_values_data),
                                  &pixel_values_shape, &pixel_values_num_dims));
    num_images = pixel_values_shape[0];
  }

  // Generate input_ids and token_type_ids
  std::string text = prompt;
  if (text.empty()) {
    for (int64_t i = 0; i < num_images; ++i) {
      text += "<start_of_image> ";
    }
    text.pop_back();
  }

  // Count the number of boi tokens and make sure it matches the number of images
  const std::regex boi_regex{std::string(boi_token)};
  const auto boi_begin = std::sregex_iterator(text.begin(), text.end(), boi_regex);
  const auto boi_end = std::sregex_iterator();
  const auto boi_tokens = std::distance(boi_begin, boi_end);
  if (num_images != boi_tokens) {
    throw std::runtime_error("Prompt contained " + std::to_string(boi_tokens) + " image tokens but received " +
                             std::to_string(num_images) + " images.");
  }

  std::string image_tokens_expanded{};
  for (size_t i = 0; i < image_seq_length; ++i) {
    image_tokens_expanded += image_token;
  }
  const std::string full_image_sequence = std::string("\n\n") + boi_token + image_tokens_expanded + eoi_token + std::string("\n\n");

  text = std::regex_replace(text, boi_regex, full_image_sequence);

  const std::vector<int32_t> input_ids = tokenizer.Encode(text.c_str());

  std::unique_ptr<OrtValue> input_ids_value = OrtValue::CreateTensor<int32_t>(allocator, std::vector<int64_t>{1, static_cast<int64_t>(input_ids.size())});
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());

  std::unique_ptr<OrtValue> token_type_ids = OrtValue::CreateTensor<int32_t>(allocator, std::vector<int64_t>{1, static_cast<int64_t>(input_ids.size())});
  const auto image_token_id = tokenizer.TokenToTokenId(image_token);
  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (input_ids[i] == image_token_id) {
      token_type_ids->GetTensorMutableData<int32_t>()[i] = 1;
    } else {
      token_type_ids->GetTensorMutableData<int32_t>()[i] = 0;
    }
  }

  std::unique_ptr<OrtValue> num_img_tokens = OrtValue::CreateTensor<int32_t>(allocator, std::vector<int64_t>{1});
  num_img_tokens->GetTensorMutableData<int32_t>()[0] = static_cast<int32_t>(image_seq_length);

  return {std::move(input_ids_value), std::move(token_type_ids), std::move(num_img_tokens)};
}

}  // namespace

GemmaImageProcessor::GemmaImageProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)} {
  const auto processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(processor_.ToBeAssigned(), processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
}

std::unique_ptr<NamedTensors> GemmaImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  std::string prompt = std::string(payload.prompt);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!images) {
    [[maybe_unused]] auto [input_ids, token_type_ids, num_img_tokens] = ProcessImagePrompt(tokenizer, prompt, nullptr, allocator);
    named_tensors->emplace(Config::Defaults::InputIdsName, std::make_shared<Tensor>(std::move(input_ids)));
    return named_tensors;
  }

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  CheckResult(OrtxImagePreProcess(processor_.get(), images->images_.get(), result.ToBeAssigned()));

  OrtxTensor* pixel_values = nullptr;
  CheckResult(OrtxTensorResultGetAt(result.get(), 0, &pixel_values));

  auto [input_ids, token_type_ids, num_img_tokens] = ProcessImagePrompt(tokenizer, prompt, pixel_values, allocator);
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName), std::make_shared<Tensor>(std::move(input_ids)));
  named_tensors->emplace(std::string(Config::Defaults::TokenTypeIdsName), std::make_shared<Tensor>(std::move(token_type_ids)));

  if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                           std::make_shared<Tensor>(ProcessTensor<float>(pixel_values, allocator)));
  } else {
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                           std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(pixel_values, allocator)));
  }

  named_tensors->emplace(std::string(Config::Defaults::NumImageTokens), std::make_shared<Tensor>(std::move(num_img_tokens)));

  return named_tensors;
}

}  // namespace Generators
