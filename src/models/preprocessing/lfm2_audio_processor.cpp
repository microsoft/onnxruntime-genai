// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/model.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/lfm2_audio_processor.h"
#include "nemo_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Generators {

namespace {

// The list overload of MultiModalProcessor::Process fills payload.prompts and leaves payload.prompt
// empty. Batching is not supported, so a single-entry list is the prompt and anything longer is an error
// rather than silently dropped text.
std::string ResolvePrompt(const Payload& payload) {
  if (payload.prompts.empty()) {
    return payload.prompt;
  }
  if (payload.prompts.size() != 1) {
    throw std::runtime_error("Lfm2AudioProcessor: batched prompts are not supported; got " +
                             std::to_string(payload.prompts.size()) + " prompts. Pass a single prompt string.");
  }
  return payload.prompts[0] ? std::string(payload.prompts[0]) : std::string{};
}

// The encoder graph's input names come from genai_config.json; a wrong name would otherwise be
// dropped silently by ExtraInputs and only surface as ORT's "Missing Input" at run time.
ONNXTensorElementDataType SpeechInputType(const SessionInfo& session_info, const std::string& name,
                                          const char* config_field) {
  if (!session_info.HasInput(name)) {
    throw std::runtime_error("Lfm2AudioProcessor: the speech model has no input named \"" + name +
                             "\". Point model.speech.inputs." + config_field +
                             " in genai_config.json at the name the speech model uses.");
  }
  return session_info.GetInputDataType(name);
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

// torch.hann_window(win_length, periodic=False), the window NeMo's FilterbankFeatures registers.
std::vector<float> SymmetricHannWindow(int win_length) {
  constexpr double kPi = 3.14159265358979323846;
  std::vector<float> window(static_cast<size_t>(win_length));
  for (int n = 0; n < win_length; ++n) {
    window[static_cast<size_t>(n)] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPi * n / (win_length - 1))));
  }
  return window;
}

}  // namespace

int64_t Lfm2AudioNumMelFrames(int64_t num_samples, const Lfm2AudioMelConfig& config) {
  // torch.stft(center=True): fft_size / 2 zeros on both sides, one frame per hop_length.
  const int64_t padded = num_samples + 2 * (config.fft_size / 2);
  if (padded < config.fft_size) {
    return 0;
  }
  return (padded - config.fft_size) / config.hop_length + 1;
}

int64_t Lfm2AudioNumValidMelFrames(int64_t num_samples, const Lfm2AudioMelConfig& config) {
  // FilterbankFeatures.get_seq_len: (num_samples + 2 * (fft_size // 2) - fft_size) // hop_length.
  const int64_t padded = num_samples + 2 * (config.fft_size / 2);
  if (padded < config.fft_size) {
    return 0;
  }
  return (padded - config.fft_size) / config.hop_length;
}

int64_t Lfm2AudioNumTokens(int64_t num_mel_frames, int64_t subsampling_factor) {
  if (subsampling_factor <= 0) {
    throw std::runtime_error("Lfm2AudioProcessor: subsampling_factor must be positive.");
  }
  return (num_mel_frames + subsampling_factor - 1) / subsampling_factor;
}

std::vector<float> ComputeLfm2AudioMel(const float* pcm, int64_t num_samples, const Lfm2AudioMelConfig& config,
                                       int64_t& num_frames) {
  if (config.num_mels <= 0 || config.fft_size <= 0 || config.hop_length <= 0 || config.win_length < 2 ||
      config.win_length > config.fft_size) {
    throw std::runtime_error("Lfm2AudioProcessor: invalid mel configuration (num_mels " + std::to_string(config.num_mels) +
                             ", fft_size " + std::to_string(config.fft_size) + ", win_length " +
                             std::to_string(config.win_length) + ", hop_length " + std::to_string(config.hop_length) + ").");
  }

  num_frames = Lfm2AudioNumMelFrames(num_samples, config);
  const int64_t valid_frames = Lfm2AudioNumValidMelFrames(num_samples, config);
  // The per-feature normalization divides by valid_frames - 1; NeMo raises on a single frame and
  // produces NaNs on none, so refuse both up front with the clip length that would be needed.
  if (valid_frames < 2) {
    throw std::runtime_error("Lfm2AudioProcessor: audio clip is too short (" + std::to_string(num_samples) +
                             " samples at " + std::to_string(config.sample_rate) + " Hz); at least " +
                             std::to_string(2 * config.hop_length) + " samples are needed.");
  }

  const auto mel_filters = nemo_mel::CreateMelFilterbank(config.num_mels, config.fft_size, config.sample_rate);
  const auto window = SymmetricHannWindow(config.win_length);

  std::vector<float> preemphasized(static_cast<size_t>(num_samples));
  nemo_mel::ApplyPreemphasis(pcm, static_cast<size_t>(num_samples), config.preemph, 0.0f, preemphasized.data());

  // Constant (zero) padding on both sides: NeMo's stft uses pad_mode="constant".
  const int64_t pad = config.fft_size / 2;
  std::vector<float> padded(static_cast<size_t>(pad + num_samples + pad), 0.0f);
  std::copy(preemphasized.begin(), preemphasized.end(), padded.begin() + pad);

  // The win_length window sits centered inside the fft_size frame; the power spectrum does not
  // depend on where the windowed samples land within the FFT buffer.
  const int64_t win_offset = (config.fft_size - config.win_length) / 2;
  const int num_bins = config.fft_size / 2 + 1;
  const auto num_mels = static_cast<size_t>(config.num_mels);

  std::vector<float> mel(static_cast<size_t>(num_frames) * num_mels);
  std::vector<float> magnitudes;
  for (int64_t t = 0; t < num_frames; ++t) {
    const float* frame = padded.data() + t * config.hop_length + win_offset;
    nemo_mel::ComputeSTFTFrame(frame, window.data(), config.win_length, config.fft_size, magnitudes);
    float* row = mel.data() + static_cast<size_t>(t) * num_mels;
    for (size_t m = 0; m < num_mels; ++m) {
      float value = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        value += mel_filters[m][static_cast<size_t>(k)] * magnitudes[static_cast<size_t>(k)];
      }
      row[m] = std::log(value + config.log_eps);
    }
  }

  // normalize_batch(normalize_type="per_feature"): mean and unbiased standard deviation over the
  // valid frames only, then the frames past the valid length are set to pad_value (0).
  for (size_t m = 0; m < num_mels; ++m) {
    double sum = 0.0;
    for (int64_t t = 0; t < valid_frames; ++t) {
      sum += mel[static_cast<size_t>(t) * num_mels + m];
    }
    const float mean = static_cast<float>(sum / static_cast<double>(valid_frames));
    double variance = 0.0;
    for (int64_t t = 0; t < valid_frames; ++t) {
      const double delta = mel[static_cast<size_t>(t) * num_mels + m] - mean;
      variance += delta * delta;
    }
    const float std = static_cast<float>(std::sqrt(variance / static_cast<double>(valid_frames - 1))) + config.norm_eps;
    for (int64_t t = 0; t < valid_frames; ++t) {
      float& value = mel[static_cast<size_t>(t) * num_mels + m];
      value = (value - mean) / std;
    }
  }
  std::fill(mel.begin() + static_cast<size_t>(valid_frames) * num_mels, mel.end(), 0.0f);
  return mel;
}

std::vector<std::string> SplitLfm2AudioPrompt(const std::string& prompt, size_t num_clips) {
  const std::string marker{kLfm2AudioMarker};
  std::vector<std::string> segments;
  size_t position = 0;
  for (size_t match = prompt.find(marker, position); match != std::string::npos; match = prompt.find(marker, position)) {
    segments.push_back(prompt.substr(position, match - position));
    position = match + marker.size();
  }
  segments.push_back(prompt.substr(position));

  // Same rule as the LFM2-VL processor applies to <image>: one marker per clip, in clip order.
  // Silently inventing or dropping placeholders would misalign the encoder features with the
  // decoder positions.
  const size_t num_markers = segments.size() - 1;
  if (num_markers != num_clips) {
    throw std::runtime_error("Lfm2AudioProcessor: the prompt contains " + std::to_string(num_markers) + " " + marker +
                             " markers but " + std::to_string(num_clips) + " audio clips were provided. Put exactly one " +
                             marker + " in the prompt per clip.");
  }
  return segments;
}

Lfm2AudioProcessor::Lfm2AudioProcessor(Config& config, const SessionInfo& session_info)
    : mel_type_{SpeechInputType(session_info, config.model.speech.inputs.audio_embeds, "audio_embeds")},
      subsampling_factor_{config.model.subsampling_factor > 0 ? config.model.subsampling_factor : 8},
      audio_token_id_{config.model.audio_token_id} {
  if (mel_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && mel_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    throw std::runtime_error("Lfm2AudioProcessor: speech input \"" + config.model.speech.inputs.audio_embeds +
                             "\" must be float or float16, got " + TypeToString(mel_type_) + ".");
  }
  // The clip lengths are always emitted as int64, so fail at load rather than at the first clip if
  // the graph was exported with another integer type.
  const auto lengths_type = SpeechInputType(session_info, config.model.speech.inputs.audio_lengths, "audio_lengths");
  if (lengths_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("Lfm2AudioProcessor: speech input \"" + config.model.speech.inputs.audio_lengths +
                             "\" must be int64, got " + TypeToString(lengths_type) + ".");
  }
  if (audio_token_id_ <= 0) {
    throw std::runtime_error(
        "Lfm2AudioProcessor: model.audio_token_id must be set in genai_config.json to the id "
        "the embedding model replaces with the audio features.");
  }

  // The mel front end defaults to the settings of the published checkpoints; a genai_config.json can
  // override any of them with the same top-level fields the Parakeet and Nemotron models use.
  const auto& m = config.model;
  if (m.num_mels > 0) mel_config_.num_mels = m.num_mels;
  if (m.fft_size > 0) mel_config_.fft_size = m.fft_size;
  if (m.win_length > 0) mel_config_.win_length = m.win_length;
  if (m.hop_length > 0) mel_config_.hop_length = m.hop_length;
  if (m.sample_rate > 0) mel_config_.sample_rate = m.sample_rate;
  if (m.preemph > 0.0f) mel_config_.preemph = m.preemph;
  if (m.log_eps > 0.0f) mel_config_.log_eps = m.log_eps;
  if (m.norm_eps > 0.0f) mel_config_.norm_eps = m.norm_eps;

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::AudioEmbedsName), config.model.speech.inputs.audio_embeds);
  config.AddMapping(std::string(Config::Defaults::AudioLengthsName), config.model.speech.inputs.audio_lengths);
  config.AddMapping(std::string(Config::Defaults::AudioSizesName), config.model.speech.inputs.audio_sizes);
}

std::unique_ptr<NamedTensors> Lfm2AudioProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const std::string prompt = ResolvePrompt(payload);
  const Audios* audios = payload.audios;
  const size_t num_clips = audios ? audios->num_audios_ : 0;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // With no clips the prompt must not ask for any; the split with zero clips enforces that.
  const std::vector<std::string> segments = SplitLfm2AudioPrompt(prompt, num_clips);

  // Mel spectrogram per clip, decoded to mono at the encoder's sample rate.
  std::vector<std::vector<float>> mels;
  std::vector<int64_t> mel_lengths;
  std::vector<int64_t> tokens_per_clip;
  int64_t longest = 0;
  int64_t total_tokens = 0;
  for (size_t i = 0; i < num_clips; ++i) {
    ort_extensions::OrtxObjectPtr<OrtxTensorResult> decoded;
    CheckResult(OrtxDecodeAudio(audios->audios_.get(), i, static_cast<int64_t>(mel_config_.sample_rate),
                                /*stereo_to_mono=*/1, decoded.ToBeAssigned()));
    ort_extensions::OrtxObjectPtr<OrtxTensor> pcm_tensor;
    CheckResult(OrtxTensorResultGetAt(decoded.get(), 0, pcm_tensor.ToBeAssigned()));

    const float* pcm{};
    const int64_t* pcm_shape{};
    size_t pcm_dims{};
    CheckResult(OrtxGetTensorData(pcm_tensor.get(), reinterpret_cast<const void**>(&pcm), &pcm_shape, &pcm_dims));
    int64_t num_samples = 0;
    if (pcm_dims == 1) {
      num_samples = pcm_shape[0];
    } else if (pcm_dims == 2 && pcm_shape[0] == 1) {
      num_samples = pcm_shape[1];
    } else {
      throw std::runtime_error("Lfm2AudioProcessor: expected mono PCM from the audio decoder for clip " +
                               std::to_string(i) + ", got a rank " + std::to_string(pcm_dims) + " tensor.");
    }

    int64_t num_frames = 0;
    mels.push_back(ComputeLfm2AudioMel(pcm, num_samples, mel_config_, num_frames));
    mel_lengths.push_back(num_frames);
    tokens_per_clip.push_back(Lfm2AudioNumTokens(num_frames, subsampling_factor_));
    longest = std::max(longest, num_frames);
    total_tokens += tokens_per_clip.back();
  }

  // Text segments are tokenized on their own, as the reference ChatState does, and each clip
  // contributes one placeholder per encoder frame between them.
  std::vector<int32_t> input_ids;
  for (size_t i = 0; i < segments.size(); ++i) {
    if (!segments[i].empty()) {
      const std::vector<int32_t> ids = tokenizer.Encode(segments[i].c_str());
      input_ids.insert(input_ids.end(), ids.begin(), ids.end());
    }
    if (i < tokens_per_clip.size()) {
      input_ids.insert(input_ids.end(), static_cast<size_t>(tokens_per_clip[i]), audio_token_id_);
    }
  }
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(MakeInputIds(input_ids, allocator)));

  if (num_clips == 0) {
    // The pipeline reads audio_sizes to skip the speech run.
    named_tensors->emplace(std::string(Config::Defaults::AudioSizesName),
                           std::make_shared<Tensor>(MakeInt64Tensor({0}, {1}, allocator)));
    return named_tensors;
  }

  // Clips share one encoder run as a zero-padded [num_clips, longest, num_mels] batch with their
  // real lengths alongside, the same batching the reference applies before its encoder.
  const int64_t num_mels = mel_config_.num_mels;
  auto batch = OrtValue::CreateTensor<float>(allocator, std::vector<int64_t>{static_cast<int64_t>(num_clips), longest, num_mels});
  float* batch_data = batch->GetTensorMutableData<float>();
  std::fill_n(batch_data, static_cast<size_t>(num_clips) * static_cast<size_t>(longest * num_mels), 0.0f);
  for (size_t i = 0; i < num_clips; ++i) {
    std::copy(mels[i].begin(), mels[i].end(), batch_data + i * static_cast<size_t>(longest * num_mels));
  }

  if (mel_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    named_tensors->emplace(std::string(Config::Defaults::AudioEmbedsName), std::make_shared<Tensor>(std::move(batch)));
  } else {
    std::unique_ptr<OrtValue> converted;
    Cast(*batch, converted, *GetDeviceInterface(DeviceType::CPU), mel_type_);
    named_tensors->emplace(std::string(Config::Defaults::AudioEmbedsName), std::make_shared<Tensor>(std::move(converted)));
  }
  named_tensors->emplace(std::string(Config::Defaults::AudioLengthsName),
                         std::make_shared<Tensor>(MakeInt64Tensor(mel_lengths, {static_cast<int64_t>(num_clips)}, allocator)));
  named_tensors->emplace(std::string(Config::Defaults::AudioSizesName),
                         std::make_shared<Tensor>(MakeInt64Tensor(tokens_per_clip, {static_cast<int64_t>(num_clips)}, allocator)));
  return named_tensors;
}

}  // namespace Generators
