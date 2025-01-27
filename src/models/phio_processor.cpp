// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <regex>

namespace Generators {

namespace {

std::unique_ptr<OrtValue> ProcessImageAudioPrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                                                  OrtxTensor* num_img_tokens, OrtxTensor* audio_embed_sizes,
                                                  Ort::Allocator& allocator) {
  constexpr size_t audio_start_input_id = -10000;

  const int64_t *num_img_tokens_data{}, *num_img_tokens_shape{};
  size_t num_img_tokens_num_dims;
  CheckResult(OrtxGetTensorData(num_img_tokens, reinterpret_cast<const void**>(&num_img_tokens_data),
                                &num_img_tokens_shape, &num_img_tokens_num_dims));
  const int64_t num_images = num_img_tokens_data
                                 ? std::accumulate(num_img_tokens_shape,
                                                   num_img_tokens_shape + num_img_tokens_num_dims,
                                                   1LL, std::multiplies<int64_t>())
                                 : 0LL;

  const int64_t *audio_embed_sizes_data{}, *audio_embed_sizes_shape{};
  size_t audio_embed_sizes_num_dims;
  CheckResult(OrtxGetTensorData(audio_embed_sizes, reinterpret_cast<const void**>(&audio_embed_sizes_data),
                                &audio_embed_sizes_shape, &audio_embed_sizes_num_dims));
  const int64_t num_audios = audio_embed_sizes_data
                                 ? std::accumulate(audio_embed_sizes_shape,
                                                   audio_embed_sizes_shape + audio_embed_sizes_num_dims,
                                                   1LL, std::multiplies<int64_t>())
                                 : 0LL;

  // Split the prompt string based on the occurrences of the pattern "<|image_<number>|>"
  // Here the <number> represents the image id.
  const std::regex pattern("(<\\|image_\\d+\\|>|<\\|audio_\\d+\\|>)");
  const std::regex image_tag_pattern("<\\|image_\\d+\\|>");
  const std::regex audio_tag_pattern("<\\|audio_\\d+\\|>");
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
  const std::vector<std::string> tags(
      std::sregex_token_iterator(prompt.begin(), prompt.end(), pattern),
      std::sregex_token_iterator());

  // Extract the image ids from the image tags.
  std::vector<int32_t> image_ids;
  std::vector<int32_t> audio_ids;
  constexpr size_t tag_id_position_begin = 8;  // <|image_ and <|audio_ : Character at idx 8 is the beginning of the id
  for (size_t i = 0; i < tags.size(); ++i) {
    const size_t tag_id_position_end = tags[i].size() - 2;  // |> : Character at idx size() - 2 is '|' which marks the end of the tag id
    if (tags[i].rfind("<|i", 0) == 0) {
      image_ids.push_back(std::stoi(tags[i].substr(tag_id_position_begin,
                                                   tag_id_position_end - tag_id_position_begin)));
    } else {
      audio_ids.push_back(std::stoi(tags[i].substr(tag_id_position_begin,
                                                   tag_id_position_end - tag_id_position_begin)));
    }
  }

  if (std::set<int32_t>(image_ids.begin(), image_ids.end()).size() != num_images) {
    throw std::runtime_error("Number of unique image tags does not match the number of images. Please fix the prompt.");
  }

  if (std::set<int32_t>(audio_ids.begin(), audio_ids.end()).size() != num_audios) {
    throw std::runtime_error("Number of unique audio tags does not match the number of audios. Please fix the prompt.");
  }

  // Construct the input_ids tensor by interleaving the input_ids_chunks and the image/audio tokens placeholder
  // The image tokens placeholder is represented by a sequence of negative value of the image_ids.
  // For example, the placeholder for image_id 1 is represented by the value [-1, -1, -1, -1]. The
  // length of the sequence is determined by the value of num_img_tokens_data[image_id - 1].
  // The audio tokens placeholder is represented by a sequence of
  std::vector<int32_t> input_ids;
  for (size_t i = 0, image_id = 0, audio_id = 0; i < input_ids_chunks.size(); ++i) {
    input_ids.insert(input_ids.end(), input_ids_chunks[i].begin(), input_ids_chunks[i].end());
    if (tags[i].rfind("<|i", 0) == 0) {
      if (image_id < image_ids.size()) {
        if (image_ids[image_id] < 1 || image_ids[image_id] > static_cast<int32_t>(num_images)) {
          const std::string error_message = "Encountered unexpected value of image_id in the prompt. Expected a value <= " +
                                            std::to_string(num_images) + ". Actual value: " + std::to_string(image_ids[image_id]);
          throw std::runtime_error(error_message);
        }
        for (int64_t j = 0; j < num_img_tokens_data[image_ids[image_id] - 1]; ++j) {
          input_ids.push_back(-image_ids[image_id]);
        }
        ++image_id;
      }
    } else {
      if (audio_id < audio_ids.size()) {
        if (audio_ids[audio_id] < 1 || audio_ids[audio_id] > static_cast<int32_t>(num_audios)) {
          const std::string error_message = "Encountered unexpected value of audio_id in the prompt. Expected a value <= " +
                                            std::to_string(num_audios) + ". Actual value: " + std::to_string(audio_ids[audio_id]);
          throw std::runtime_error(error_message);
        }
        for (int64_t j = 0; j < audio_embed_sizes_data[audio_ids[audio_id] - 1]; ++j) {
          input_ids.push_back(-audio_ids[audio_id] + audio_start_input_id + 1);
        }
        ++audio_id;
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

PhiOProcessor::PhiOProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)},
      audio_features_type_{session_info.GetInputDataType(config.model.speech.inputs.audio_embeds)} {
  const std::string image_processor_file_name = "processor_config.json";
  const auto image_processor_config = (config.config_path / fs::path(image_processor_file_name)).string();
  CheckResult(OrtxCreateProcessor(image_processor_.ToBeAssigned(), image_processor_config.c_str()));

  const std::string audio_processor_file_name = "audio_processor_config.json";
  const auto audio_processor_config = (config.config_path / fs::path(audio_processor_file_name)).string();
  CheckResult(OrtxCreateSpeechFeatureExtractor(audio_processor_.ToBeAssigned(), audio_processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);

  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
  config.AddMapping(std::string(Config::Defaults::AttentionMaskName), config.model.vision.inputs.attention_mask);
  config.AddMapping(std::string(Config::Defaults::ImageSizesName), config.model.vision.inputs.image_sizes);

  config.AddMapping(std::string(Config::Defaults::AudioEmbedsName), config.model.speech.inputs.audio_embeds);
  config.AddMapping(std::string(Config::Defaults::AudioEmbedsSizesName), config.model.speech.inputs.audio_sizes);
}

std::unique_ptr<NamedTensors> PhiOProcessor::Process([[maybe_unused]] const Tokenizer& tokenizer, [[maybe_unused]] const Payload& payload) const {
  std::string prompt = std::string(payload.prompt);
  const Images* images = payload.images;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  if (!payload.images && !payload.audios) {
    named_tensors->emplace(Config::Defaults::InputIdsName,
                           std::make_shared<Tensor>(ProcessImageAudioPrompt(tokenizer, prompt, nullptr, nullptr, allocator)));
    return named_tensors;
  }

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> image_result;
  CheckResult(OrtxImagePreProcess(image_processor_.get(), images->images_.get(), image_result.ToBeAssigned()));

  OrtxTensor *pixel_values, *image_sizes, *image_attention_mask, *num_img_tokens;
  CheckResult(OrtxTensorResultGetAt(image_result.get(), 0, &pixel_values));
  CheckResult(OrtxTensorResultGetAt(image_result.get(), 1, &image_sizes));
  CheckResult(OrtxTensorResultGetAt(image_result.get(), 2, &image_attention_mask));
  CheckResult(OrtxTensorResultGetAt(image_result.get(), 3, &num_img_tokens));

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> audio_result;
  CheckResult(OrtxSpeechLogMel(audio_processor_.get(), payload.audios->audios_.get(), audio_result.ToBeAssigned()));

  OrtxTensor *audio_embeds, *audio_embed_sizes;
  CheckResult(OrtxTensorResultGetAt(audio_result.get(), 0, &audio_embeds));
  CheckResult(OrtxTensorResultGetAt(audio_result.get(), 1, &audio_embed_sizes));

  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(ProcessImageAudioPrompt(tokenizer, prompt, num_img_tokens, audio_embed_sizes, allocator)));

  if (pixel_values_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                           std::make_shared<Tensor>(ProcessTensor<float>(pixel_values, allocator)));
  } else {
    named_tensors->emplace(std::string(Config::Defaults::PixelValuesName),
                           std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(pixel_values, allocator)));
  }

  named_tensors->emplace(std::string(Config::Defaults::ImageSizesName),
                         std::make_shared<Tensor>(ProcessTensor<int64_t>(image_sizes, allocator)));
  named_tensors->emplace(std::string(Config::Defaults::AttentionMaskName),
                         std::make_shared<Tensor>(ProcessTensor<int64_t>(image_attention_mask, allocator)));

  if (audio_features_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    named_tensors->emplace(std::string(Config::Defaults::AudioEmbedsName),
                           std::make_shared<Tensor>(ProcessTensor<float>(audio_embeds, allocator)));
  } else {
    named_tensors->emplace(std::string(Config::Defaults::AudioEmbedsName),
                           std::make_shared<Tensor>(ProcessTensor<Ort::Float16_t>(audio_embeds, allocator)));
  }

  named_tensors->emplace(std::string(Config::Defaults::AudioEmbedsSizesName),
                         std::make_shared<Tensor>(ProcessTensor<int64_t>(audio_embed_sizes, allocator)));

  return named_tensors;
}

}  // namespace Generators
