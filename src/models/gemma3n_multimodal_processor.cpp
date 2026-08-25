// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <cstring>

namespace Generators {

namespace {

// Gemma 3n special tokens. Unlike Gemma 3, whose chat template emits the *begin*
// marker <start_of_image>, Gemma 3n's template emits the soft token itself and the
// processor is what wraps it in begin/end markers (see HF Gemma3nProcessor:
// `prompt.replace(self.image_token, self.full_image_sequence)`).
constexpr char kBoiToken[] = "<start_of_image>";
constexpr char kImageToken[] = "<image_soft_token>";
constexpr char kEoiToken[] = "<end_of_image>";
constexpr char kBoaToken[] = "<start_of_audio>";
constexpr char kAudioToken[] = "<audio_soft_token>";
constexpr char kEoaToken[] = "<end_of_audio>";

// token_type_ids values consumed by the embedding model, matching HF:
// image -> 1, audio -> 3 (0 is text).
constexpr int32_t kTextTokenType = 0;
constexpr int32_t kImageTokenType = 1;
constexpr int32_t kAudioTokenType = 3;

size_t CountOccurrences(const std::string& text, const std::string& token) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(token, pos)) != std::string::npos) {
    ++count;
    pos += token.size();
  }
  return count;
}

// Replace every occurrence of `from` with `to`, skipping past what was just
// inserted. `to` contains `from` many times over, so advancing by to.size() is
// what keeps this from looping forever.
void ReplaceAll(std::string& text, const std::string& from, const std::string& to) {
  size_t pos = 0;
  while ((pos = text.find(from, pos)) != std::string::npos) {
    text.replace(pos, from.size(), to);
    pos += to.size();
  }
}

std::string BuildSoftTokenSequence(const char* begin_token, const char* soft_token,
                                   const char* end_token, size_t count) {
  const size_t soft_token_len = std::strlen(soft_token);
  std::string expanded;
  expanded.reserve(count * soft_token_len);
  for (size_t i = 0; i < count; ++i) {
    expanded += soft_token;
  }
  return "\n\n" + std::string(begin_token) + expanded + end_token + "\n\n";
}

// Resolve how many placeholders the prompt carries for one modality, normalising
// the two spellings a prompt can use. Gemma 3n's own chat template emits the soft
// token, but a Gemma 3 style template (or a hand-written prompt) emits the begin
// marker instead, and both should work. Returns the placeholder to expand.
std::string ResolvePlaceholder(std::string& text, const char* soft_token, const char* begin_token,
                               int64_t expected_count, const char* modality) {
  const auto soft_count = static_cast<int64_t>(CountOccurrences(text, soft_token));
  if (soft_count > 0) {
    if (soft_count != expected_count) {
      throw std::runtime_error("Prompt contained " + std::to_string(soft_count) + " " + modality +
                               " placeholders but received " + std::to_string(expected_count) + " " +
                               modality + " inputs.");
    }
    return soft_token;
  }

  const auto begin_count = static_cast<int64_t>(CountOccurrences(text, begin_token));
  if (begin_count > 0) {
    if (begin_count != expected_count) {
      throw std::runtime_error("Prompt contained " + std::to_string(begin_count) + " " + modality +
                               " placeholders but received " + std::to_string(expected_count) + " " +
                               modality + " inputs.");
    }
    return begin_token;
  }

  // No placeholder at all: prepend one per input so a bare prompt still works.
  std::string prefix;
  for (int64_t i = 0; i < expected_count; ++i) {
    prefix += soft_token;
  }
  text = prefix + text;
  return soft_token;
}

}  // namespace

Gemma3nMultiModalProcessor::Gemma3nMultiModalProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)} {
  const auto image_processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(image_processor_.ToBeAssigned(), image_processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);

  if (!config.model.speech.config_filename.empty()) {
    auto speech_config_path = config.config_path / fs::path(config.model.speech.config_filename);
    if (fs::exists(speech_config_path)) {
      has_speech_ = true;
      audio_features_type_ = session_info.GetInputDataType(config.model.speech.inputs.audio_embeds);
      CheckResult(OrtxCreateSpeechFeatureExtractor(audio_processor_.ToBeAssigned(), speech_config_path.string().c_str()));

      config.AddMapping(std::string(Config::Defaults::AudioEmbedsName), config.model.speech.inputs.audio_embeds);
      config.AddMapping(std::string(Config::Defaults::AudioAttentionMaskName), config.model.speech.inputs.attention_mask);
      config.AddMapping(std::string(Config::Defaults::AudioSizesName), config.model.speech.inputs.audio_sizes);
    } else if (!config.model.speech.filename.empty()) {
      throw std::runtime_error("Speech model is configured (speech.filename=" + config.model.speech.filename +
                               ") but the audio processor config file was not found at: " +
                               speech_config_path.string());
    }
  }
}

std::unique_ptr<NamedTensors> Gemma3nMultiModalProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();
  std::string text = std::string(payload.prompt);

  if (payload.audios && !has_speech_) {
    throw std::runtime_error(
        "Audio input was provided but audio support is not configured. "
        "Ensure genai_config.json has a 'speech' section with both 'filename' and 'config_filename'.");
  }

  // Vision. The Gemma 3n tower takes one fixed 768x768 image and always emits a
  // 16x16 grid, so there is no pan-and-scan, no position ids, and no padding to
  // trim -- unlike Gemma 4.
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> image_result;
  ort_extensions::OrtxObjectPtr<OrtxTensor> pixel_values_owner;
  int64_t num_images = 0;
  if (payload.images) {
    CheckResult(OrtxImagePreProcess(image_processor_.get(), payload.images->images_.get(), image_result.ToBeAssigned()));
    CheckResult(OrtxTensorResultGetAt(image_result.get(), 0, pixel_values_owner.ToBeAssigned()));

    const float* pixel_values_data{};
    const int64_t* pixel_values_shape{};
    size_t pixel_values_num_dims;
    CheckResult(OrtxGetTensorData(pixel_values_owner.get(), reinterpret_cast<const void**>(&pixel_values_data),
                                  &pixel_values_shape, &pixel_values_num_dims));
    // [batch, 3, 768, 768]; a rank-3 tensor is a single un-batched image.
    if (pixel_values_num_dims != 3 && pixel_values_num_dims != 4) {
      throw std::runtime_error("Expected the image processor to return a rank-3 or rank-4 pixel_values tensor, got rank " +
                               std::to_string(pixel_values_num_dims) + ".");
    }
    num_images = (pixel_values_num_dims == 4) ? pixel_values_shape[0] : 1;

    const std::string placeholder =
        ResolvePlaceholder(text, kImageToken, kBoiToken, num_images, "image");
    ReplaceAll(text, placeholder,
               BuildSoftTokenSequence(kBoiToken, kImageToken, kEoiToken, image_seq_length_));
  }

  // Audio. The feature extractor pads or truncates to the length the conformer
  // reduces to audio_seq_length_ frames, so the soft-token count is fixed and
  // does not need to be derived from the mel length.
  if (payload.audios && has_speech_) {
    ort_extensions::OrtxObjectPtr<OrtxTensorResult> audio_result;
    CheckResult(OrtxFeatureExtraction(audio_processor_.get(), payload.audios->audios_.get(), audio_result.ToBeAssigned()));

    ort_extensions::OrtxObjectPtr<OrtxTensor> audio_features_owner;
    CheckResult(OrtxTensorResultGetAt(audio_result.get(), 0, audio_features_owner.ToBeAssigned()));
    OrtxTensor* audio_features = audio_features_owner.get();

    const float* audio_data{};
    const int64_t* audio_shape{};
    size_t audio_dims;
    CheckResult(OrtxGetTensorData(audio_features, reinterpret_cast<const void**>(&audio_data),
                                  &audio_shape, &audio_dims));
    // [batch, time, 128]; a rank-2 tensor is a single un-batched clip.
    if (audio_dims != 2 && audio_dims != 3) {
      throw std::runtime_error("Expected the audio feature extractor to return a rank-2 or rank-3 tensor, got rank " +
                               std::to_string(audio_dims) + ".");
    }
    const int64_t batch_dim = (audio_dims == 3) ? audio_shape[0] : 1;
    const int64_t time_dim = (audio_dims == 3) ? audio_shape[1] : audio_shape[0];

    EmplaceProcessedTensor(*named_tensors, Config::Defaults::AudioEmbedsName, audio_features,
                           audio_features_type_, allocator);

    // input_features_mask marks the valid (unpadded) frames. Single-clip inference
    // has no padding, so every frame is valid.
    auto mask = OrtValue::CreateTensor<bool>(allocator, std::vector<int64_t>{batch_dim, time_dim});
    std::fill_n(mask->GetTensorMutableData<bool>(), batch_dim * time_dim, true);
    named_tensors->emplace(std::string(Config::Defaults::AudioAttentionMaskName),
                           std::make_shared<Tensor>(std::move(mask)));

    // audio_sizes is not a session input -- it is how MultiModalPipelineState
    // learns there is audio at all. GetNumAudioTokens sums it, and the sum both
    // gates SpeechState::Run and sizes the pre-allocated audio_features buffer.
    // Omitting it leaves num_audio_tokens_ at 0, so the speech encoder never
    // runs and the embedding model is handed an empty audio_features while the
    // prompt still carries audio_seq_length_ placeholders -- which surfaces as
    // an out-of-bounds Gather on the CPU EP, or as silently wrong output on EPs
    // that clamp instead.
    //
    // One row per clip, each audio_seq_length_ long: unlike Gemma 4, the count
    // does not follow the mel length, because the audio encoder pads and
    // truncates its own output to that fixed width.
    auto audio_sizes = OrtValue::CreateTensor<int64_t>(allocator, std::vector<int64_t>{batch_dim});
    std::fill_n(audio_sizes->GetTensorMutableData<int64_t>(), batch_dim,
                static_cast<int64_t>(audio_seq_length_));
    named_tensors->emplace(std::string(Config::Defaults::AudioSizesName),
                           std::make_shared<Tensor>(std::move(audio_sizes)));

    const std::string placeholder =
        ResolvePlaceholder(text, kAudioToken, kBoaToken, batch_dim, "audio");
    ReplaceAll(text, placeholder,
               BuildSoftTokenSequence(kBoaToken, kAudioToken, kEoaToken, audio_seq_length_));
  }

  const std::vector<int32_t> input_ids = tokenizer.Encode(text.c_str());
  const auto seq_len = static_cast<int64_t>(input_ids.size());

  auto input_ids_value = OrtValue::CreateTensor<int32_t>(allocator, std::vector<int64_t>{1, seq_len});
  std::copy(input_ids.begin(), input_ids.end(), input_ids_value->GetTensorMutableData<int32_t>());
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(std::move(input_ids_value)));

  // Text-only prompts carry no soft tokens, so the remaining per-modality tensors
  // would all be trivial; skip them the way the other Gemma processors do.
  if (!payload.images && !payload.audios) {
    return named_tensors;
  }

  auto token_type_ids = OrtValue::CreateTensor<int32_t>(allocator, std::vector<int64_t>{1, seq_len});
  const auto image_token_id = tokenizer.TokenToTokenId(kImageToken);
  const auto audio_token_id = tokenizer.TokenToTokenId(kAudioToken);
  auto* token_type_data = token_type_ids->GetTensorMutableData<int32_t>();
  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (input_ids[i] == image_token_id) {
      token_type_data[i] = kImageTokenType;
    } else if (input_ids[i] == audio_token_id) {
      token_type_data[i] = kAudioTokenType;
    } else {
      token_type_data[i] = kTextTokenType;
    }
  }
  named_tensors->emplace(std::string(Config::Defaults::TokenTypeIdsName),
                         std::make_shared<Tensor>(std::move(token_type_ids)));

  if (payload.images) {
    EmplaceProcessedTensor(*named_tensors, Config::Defaults::PixelValuesName, pixel_values_owner.get(),
                           pixel_values_type_, allocator);

    // One row per image, not one row total: GetNumImageTokens sums this tensor to
    // size the pre-allocated image_features buffer, so a single element would
    // under-count a multi-image prompt by (num_images - 1) * image_seq_length_.
    auto num_img_tokens = OrtValue::CreateTensor<int64_t>(allocator, std::vector<int64_t>{num_images});
    std::fill_n(num_img_tokens->GetTensorMutableData<int64_t>(), num_images,
                static_cast<int64_t>(image_seq_length_));
    named_tensors->emplace(std::string(Config::Defaults::NumImageTokens),
                           std::make_shared<Tensor>(std::move(num_img_tokens)));
  }

  return named_tensors;
}

}  // namespace Generators
