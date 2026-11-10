// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/model.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/lfm2_audio_processor.h"
#include "models/preprocessing/lfm2_vl_image_processor.h"
#include "models/preprocessing/lfm3_audio_processor.h"
#include "nemo_mel_spectrogram.h"

#include <algorithm>
#include <cmath>

namespace Generators {

namespace {

constexpr char kProcessorName[] = "Lfm3AudioProcessor";
constexpr double kLogMelFloor = -12.0;
constexpr double kNormalizationOffset = 4.0;
constexpr double kNormalizationDivisor = 4.0;

}  // namespace

int64_t Lfm3AudioNumMelFrames(int64_t num_samples, const Lfm3AudioMelConfig& config) {
  const int64_t hop = config.hop_length;
  const int64_t to_whole_hop = ((config.fft_size - num_samples) % hop + hop) % hop;
  const int64_t padded = std::max<int64_t>(config.fft_size, num_samples + to_whole_hop);
  return 1 + (padded - config.fft_size) / hop;
}

std::vector<float> ComputeLfm3AudioMel(const float* pcm, int64_t num_samples, const Lfm3AudioMelConfig& config,
                                       int64_t& num_frames) {
  constexpr double kPi = 3.14159265358979323846;
  num_frames = Lfm3AudioNumMelFrames(num_samples, config);

  std::vector<float> padded(static_cast<size_t>((num_frames - 1) * config.hop_length + config.fft_size), 0.0f);
  std::copy(pcm, pcm + num_samples, padded.begin());

  std::vector<float> window(static_cast<size_t>(config.fft_size));
  for (size_t k = 0; k < window.size(); ++k) {
    window[k] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(k) / config.fft_size));
  }

  const auto filters = nemo_mel::CreateMelFilterbank(config.num_mels, config.fft_size, config.sample_rate);
  const auto num_mels = static_cast<size_t>(config.num_mels);
  const auto frames = static_cast<size_t>(num_frames);
  std::vector<float> mel(num_mels * frames);
  std::vector<float> power;
  for (size_t t = 0; t < frames; ++t) {
    nemo_mel::ComputeSTFTFrame(padded.data() + t * static_cast<size_t>(config.hop_length), window.data(),
                               config.fft_size, config.fft_size, power);
    for (size_t m = 0; m < num_mels; ++m) {
      double band_power = 0.0;
      for (size_t k = 0; k < power.size(); ++k) {
        band_power += static_cast<double>(filters[m][k]) * power[k];
      }
      const double log_mel = std::max(std::log10(band_power), kLogMelFloor);
      mel[m * frames + t] = static_cast<float>((log_mel + kNormalizationOffset) / kNormalizationDivisor);
    }
  }
  return mel;
}

Lfm3AudioProcessor::Lfm3AudioProcessor(Config& config, const SessionInfo& session_info)
    : mel_config_{config.model.num_mels, config.model.fft_size, config.model.hop_length, config.model.sample_rate},
      subsampling_factor_{config.model.subsampling_factor},
      audio_token_id_{config.model.audio_token_id} {
  const auto fail = [](const std::string& message) { throw std::runtime_error(std::string(kProcessorName) + ": " + message); };
  const auto& model = config.model;

  if (model.num_mels <= 0 || model.fft_size <= 0 || model.win_length <= 0 || model.hop_length <= 0 ||
      model.sample_rate <= 0 || model.subsampling_factor <= 0) {
    const std::string got = std::to_string(model.num_mels) + ", " + std::to_string(model.fft_size) + ", " +
                            std::to_string(model.win_length) + ", " + std::to_string(model.hop_length) + ", " +
                            std::to_string(model.sample_rate) + ", " + std::to_string(model.subsampling_factor);
    fail("model.num_mels, fft_size, win_length, hop_length, sample_rate and subsampling_factor must all be set in genai_config.json; got " + got + ".");
  }
  if (model.win_length != model.fft_size) {
    fail("model.win_length " + std::to_string(model.win_length) + " must equal model.fft_size " +
         std::to_string(model.fft_size) + ".");
  }
  if (audio_token_id_ <= 0 || audio_token_id_ >= model.vocab_size) {
    fail("model.audio_token_id " + std::to_string(audio_token_id_) + " is outside the vocabulary of " +
         std::to_string(model.vocab_size) + " tokens.");
  }

  const std::string& mel_input = model.speech.inputs.audio_embeds;
  if (!session_info.HasInput(mel_input)) {
    fail("the speech model has no input named \"" + mel_input + "\".");
  }
  mel_type_ = session_info.GetInputDataType(mel_input);
  if (mel_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && mel_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    fail("speech input \"" + mel_input + "\" must be float or float16, got " + TypeToString(mel_type_) + ".");
  }
  const auto mel_shape = session_info.GetInputShape(mel_input);
  if (mel_shape.size() != 3 || mel_shape[1] != model.num_mels) {
    fail("speech input \"" + mel_input + "\" must be [1, " + std::to_string(model.num_mels) + ", num_frames].");
  }

  const std::string& features_output = model.speech.outputs.audio_features;
  if (!session_info.HasOutput(features_output)) {
    fail("the speech model has no output named \"" + features_output + "\".");
  }
  if (session_info.GetOutputShape(features_output).size() != 2) {
    fail("speech output \"" + features_output + "\" must be [num_audio_tokens, hidden_size].");
  }
  const std::string& features_input = model.embedding.inputs.audio_features;
  if (!session_info.HasInput(features_input)) {
    fail("the embedding model has no input named \"" + features_input + "\".");
  }
  if (session_info.GetInputDataType(features_input) != session_info.GetOutputDataType(features_output)) {
    fail("the embedding model's \"" + features_input + "\" is " + TypeToString(session_info.GetInputDataType(features_input)) +
         " but the speech model writes " + TypeToString(session_info.GetOutputDataType(features_output)) + ".");
  }

  config.AddMapping(std::string(Config::Defaults::InputIdsName), model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::AudioEmbedsName), mel_input);
  config.AddMapping(std::string(Config::Defaults::AudioLengthsName), model.speech.inputs.audio_lengths);
  config.AddMapping(std::string(Config::Defaults::AudioSizesName), model.speech.inputs.audio_sizes);
}

std::unique_ptr<NamedTensors> Lfm3AudioProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const std::string prompt = ResolvePrompt(payload, kProcessorName);
  if (payload.images) {
    throw std::runtime_error(std::string(kProcessorName) + ": lfm3_audio takes no images.");
  }
  if (prompt.find(kLfm3AudioOutMarker) != std::string::npos) {
    throw std::runtime_error(std::string(kProcessorName) + ": the prompt contains " + kLfm3AudioOutMarker +
                             ", but speech output is not supported.");
  }
  const size_t num_clips = payload.audios ? payload.audios->num_audios_ : 0;
  const std::vector<std::string> segments = SplitLfm2AudioPrompt(prompt, num_clips, kProcessorName);

  std::vector<std::vector<float>> mels(num_clips);
  std::vector<int64_t> frames_per_clip(num_clips);
  std::vector<int64_t> tokens_per_clip(num_clips);
  for (size_t i = 0; i < num_clips; ++i) {
    const std::vector<float> samples = DecodeLfm2AudioClip(*payload.audios, i, mel_config_.sample_rate, kProcessorName);
    mels[i] = ComputeLfm3AudioMel(samples.data(), static_cast<int64_t>(samples.size()), mel_config_, frames_per_clip[i]);
    tokens_per_clip[i] = Lfm2AudioNumTokens(frames_per_clip[i], subsampling_factor_, kProcessorName);
  }

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(MakeInputIds(MakeLfm2AudioPromptTokens(tokenizer, segments, tokens_per_clip, audio_token_id_), allocator)));

  if (num_clips == 0) {
    // The pipeline reads audio_sizes to skip the speech run.
    named_tensors->emplace(std::string(Config::Defaults::AudioSizesName),
                           std::make_shared<Tensor>(MakeInt64Tensor({0}, {1}, allocator)));
    return named_tensors;
  }

  const auto clips = static_cast<int64_t>(num_clips);
  const int64_t num_mels = mel_config_.num_mels;
  const int64_t longest = *std::max_element(frames_per_clip.begin(), frames_per_clip.end());
  auto batch = OrtValue::CreateTensor<float>(allocator, std::vector<int64_t>{clips, num_mels, longest});
  float* batch_data = batch->GetTensorMutableData<float>();
  std::fill_n(batch_data, static_cast<size_t>(clips * num_mels * longest), 0.0f);
  for (size_t i = 0; i < num_clips; ++i) {
    for (int64_t m = 0; m < num_mels; ++m) {
      std::copy_n(mels[i].data() + m * frames_per_clip[i], frames_per_clip[i],
                  batch_data + (static_cast<int64_t>(i) * num_mels + m) * longest);
    }
  }

  AddLfm2AudioEmbeds(std::move(batch), mel_type_, *named_tensors);
  named_tensors->emplace(std::string(Config::Defaults::AudioLengthsName),
                         std::make_shared<Tensor>(MakeInt64Tensor(frames_per_clip, {clips}, allocator)));
  named_tensors->emplace(std::string(Config::Defaults::AudioSizesName),
                         std::make_shared<Tensor>(MakeInt64Tensor(tokens_per_clip, {clips}, allocator)));
  return named_tensors;
}

}  // namespace Generators
