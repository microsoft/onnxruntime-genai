// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/model.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/lfm2_audio_processor.h"
#include "models/preprocessing/qwen3_asr_processor.h"
#include "nemo_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Generators {

namespace {

// The list overload of MultiModalProcessor::Process fills payload.prompts and leaves payload.prompt
// empty. Batching is not supported, so a single-entry list is the prompt and anything longer is an error.
std::string ResolvePrompt(const Payload& payload) {
  if (payload.prompts.empty()) {
    return payload.prompt;
  }
  if (payload.prompts.size() != 1) {
    throw std::runtime_error("Qwen3AsrProcessor: batched prompts are not supported; got " +
                             std::to_string(payload.prompts.size()) + " prompts. Pass a single prompt string.");
  }
  return payload.prompts[0] ? std::string(payload.prompts[0]) : std::string{};
}

// The encoder graph's input names come from genai_config.json; a wrong name would otherwise be
// dropped silently by ExtraInputs and only surface as ORT's "Missing Input" at run time.
ONNXTensorElementDataType SpeechInputType(const SessionInfo& session_info, const std::string& name,
                                          const char* config_field) {
  if (!session_info.HasInput(name)) {
    throw std::runtime_error("Qwen3AsrProcessor: the speech model has no input named \"" + name +
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

// torch.hann_window(length), periodic, the window the feature extractor hands to torch.stft.
std::vector<float> PeriodicHannWindow(int length) {
  constexpr double kPi = 3.14159265358979323846;
  std::vector<float> window(static_cast<size_t>(length));
  for (int n = 0; n < length; ++n) {
    window[static_cast<size_t>(n)] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPi * n / length)));
  }
  return window;
}

// Output length of the encoder's three stride-2 convolutions (kernel 3, padding 1) on `frames` frames.
int64_t ConvStackLength(int64_t frames) {
  for (int i = 0; i < 3; ++i) {
    frames = (frames + 1) / 2;
  }
  return frames;
}

}  // namespace

int64_t Qwen3AsrNumValidMelFrames(int64_t num_samples, const Qwen3AsrMelConfig& config) {
  return std::max(num_samples, config.min_samples) / config.hop_length;
}

int64_t Qwen3AsrNumTokens(int64_t valid_frames, int64_t chunk_frames) {
  if (chunk_frames <= 0) {
    throw std::runtime_error("Qwen3AsrProcessor: the chunk length must be positive.");
  }
  return (valid_frames / chunk_frames) * ConvStackLength(chunk_frames) + ConvStackLength(valid_frames % chunk_frames);
}

std::vector<float> ComputeQwen3AsrMel(const float* pcm, int64_t num_samples, const Qwen3AsrMelConfig& config,
                                      int64_t& valid_frames, int64_t& num_frames) {
  if (config.num_mels <= 0 || config.fft_size < 2 || config.fft_size % 2 != 0 || config.hop_length <= 0 ||
      config.chunk_frames <= 0) {
    throw std::runtime_error("Qwen3AsrProcessor: invalid mel configuration (num_mels " + std::to_string(config.num_mels) +
                             ", fft_size " + std::to_string(config.fft_size) + ", hop_length " +
                             std::to_string(config.hop_length) + ", chunk_frames " + std::to_string(config.chunk_frames) +
                             ").");
  }

  // Clips shorter than min_samples are zero-padded first, as the reference does, and the padding is
  // treated as part of the clip.
  std::vector<float> samples(pcm, pcm + num_samples);
  if (static_cast<int64_t>(samples.size()) < config.min_samples) {
    samples.resize(static_cast<size_t>(config.min_samples), 0.0f);
  }
  const int64_t n = static_cast<int64_t>(samples.size());
  const int64_t pad = config.fft_size / 2;
  if (n <= pad) {
    throw std::runtime_error("Qwen3AsrProcessor: audio clip is too short (" + std::to_string(num_samples) +
                             " samples); at least " + std::to_string(pad + 1) + " samples are needed.");
  }

  // torch.stft(center=True) pads fft_size / 2 samples on both sides in reflect mode (the edge sample
  // is not repeated).
  std::vector<float> padded(static_cast<size_t>(n + 2 * pad));
  for (int64_t i = 0; i < pad; ++i) {
    padded[static_cast<size_t>(i)] = samples[static_cast<size_t>(pad - i)];
    padded[static_cast<size_t>(pad + n + i)] = samples[static_cast<size_t>(n - 2 - i)];
  }
  std::copy(samples.begin(), samples.end(), padded.begin() + pad);

  valid_frames = Qwen3AsrNumValidMelFrames(num_samples, config);
  num_frames = (valid_frames + config.chunk_frames - 1) / config.chunk_frames * config.chunk_frames;

  const auto mel_filters = nemo_mel::CreateMelFilterbank(config.num_mels, config.fft_size, config.sample_rate);
  const auto window = PeriodicHannWindow(config.fft_size);
  const int num_bins = config.fft_size / 2 + 1;
  const auto num_mels = static_cast<size_t>(config.num_mels);

  std::vector<float> mel(static_cast<size_t>(num_frames) * num_mels, 0.0f);
  std::vector<float> power;
  float max_log = -std::numeric_limits<float>::infinity();
  for (int64_t t = 0; t < valid_frames; ++t) {
    nemo_mel::ComputeSTFTFrame(padded.data() + t * config.hop_length, window.data(), config.fft_size, config.fft_size,
                               power);
    float* row = mel.data() + static_cast<size_t>(t) * num_mels;
    for (size_t m = 0; m < num_mels; ++m) {
      float value = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        value += mel_filters[m][static_cast<size_t>(k)] * power[static_cast<size_t>(k)];
      }
      row[m] = std::log10(std::max(value, config.log_floor));
      max_log = std::max(max_log, row[m]);
    }
  }

  // Whisper-style dynamic range and scaling over the clip's own frames; the padding frames stay 0.
  const float floor_value = max_log - config.dynamic_range;
  for (int64_t t = 0; t < valid_frames; ++t) {
    float* row = mel.data() + static_cast<size_t>(t) * num_mels;
    for (size_t m = 0; m < num_mels; ++m) {
      row[m] = (std::max(row[m], floor_value) + 4.0f) / 4.0f;
    }
  }
  return mel;
}

std::vector<std::string> SplitQwen3AsrPrompt(const std::string& prompt, size_t num_clips) {
  const std::string marker{kQwen3AsrAudioMarker};
  std::vector<std::string> segments;
  size_t position = 0;
  for (size_t match = prompt.find(marker, position); match != std::string::npos; match = prompt.find(marker, position)) {
    segments.push_back(prompt.substr(position, match - position));
    position = match + marker.size();
  }
  segments.push_back(prompt.substr(position));

  // One marker per clip, in clip order; inventing or dropping placeholders would misalign the
  // encoder features with the decoder positions.
  const size_t num_markers = segments.size() - 1;
  if (num_markers != num_clips) {
    throw std::runtime_error("Qwen3AsrProcessor: the prompt contains " + std::to_string(num_markers) + " " + marker +
                             " markers but " + std::to_string(num_clips) + " audio clips were provided. Put exactly one " +
                             marker + " (inside <|audio_start|>...<|audio_end|>) in the prompt per clip.");
  }
  return segments;
}

Qwen3AsrProcessor::Qwen3AsrProcessor(Config& config, const SessionInfo& session_info)
    : mel_type_{SpeechInputType(session_info, config.model.speech.inputs.audio_embeds, "audio_embeds")},
      audio_token_id_{config.model.audio_token_id} {
  if (mel_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && mel_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    throw std::runtime_error("Qwen3AsrProcessor: speech input \"" + config.model.speech.inputs.audio_embeds +
                             "\" must be float or float16, got " + TypeToString(mel_type_) + ".");
  }
  const auto lengths_type = SpeechInputType(session_info, config.model.speech.inputs.audio_lengths, "audio_lengths");
  if (lengths_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("Qwen3AsrProcessor: speech input \"" + config.model.speech.inputs.audio_lengths +
                             "\" must be int64, got " + TypeToString(lengths_type) + ".");
  }
  if (audio_token_id_ <= 0) {
    throw std::runtime_error(
        "Qwen3AsrProcessor: model.audio_token_id must be set in genai_config.json to the id "
        "the embedding model replaces with the audio features.");
  }
  if (audio_token_id_ >= config.model.vocab_size) {
    throw std::runtime_error("Qwen3AsrProcessor: model.audio_token_id " + std::to_string(audio_token_id_) +
                             " is outside the vocabulary of " + std::to_string(config.model.vocab_size) + " tokens.");
  }

  // The front end defaults to the checkpoint's settings; genai_config.json can override them with the
  // top-level fields the other audio models use. The window length is always fft_size.
  const auto& m = config.model;
  if (m.num_mels > 0) mel_config_.num_mels = m.num_mels;
  if (m.fft_size > 0) mel_config_.fft_size = m.fft_size;
  if (m.hop_length > 0) mel_config_.hop_length = m.hop_length;
  if (m.sample_rate > 0) mel_config_.sample_rate = m.sample_rate;

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Config::Defaults::AudioEmbedsName), config.model.speech.inputs.audio_embeds);
  config.AddMapping(std::string(Config::Defaults::AudioLengthsName), config.model.speech.inputs.audio_lengths);
  config.AddMapping(std::string(Config::Defaults::AudioSizesName), config.model.speech.inputs.audio_sizes);
}

Qwen3AsrProcessor::ClipMel Qwen3AsrProcessor::ComputeClipMel(const Audios& audios, size_t index) const {
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> decoded;
  CheckResult(OrtxDecodeAudio(audios.audios_.get(), index, /*target_sample_rate=*/0, /*stereo_to_mono=*/1,
                              decoded.ToBeAssigned()));

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
    throw std::runtime_error("Qwen3AsrProcessor: expected mono PCM from the audio decoder for clip " +
                             std::to_string(index) + ", got a rank " + std::to_string(pcm_dims) + " tensor.");
  }

  ort_extensions::OrtxObjectPtr<OrtxTensor> rate_tensor;
  CheckResult(OrtxTensorResultGetAt(decoded.get(), 1, rate_tensor.ToBeAssigned()));
  const int64_t* source_rate{};
  CheckResult(OrtxGetTensorData(rate_tensor.get(), reinterpret_cast<const void**>(&source_rate), nullptr, nullptr));

  // The reference takes 16 kHz audio. Other rates are brought to it with the resampler of the
  // LFM2-Audio processor (torchaudio's default sinc filter).
  const std::vector<float> samples = ResampleLfm2Audio(pcm, num_samples, *source_rate, mel_config_.sample_rate);

  ClipMel clip;
  clip.mel = ComputeQwen3AsrMel(samples.data(), static_cast<int64_t>(samples.size()), mel_config_, clip.valid_frames,
                                clip.num_frames);
  clip.num_tokens = Qwen3AsrNumTokens(clip.valid_frames, mel_config_.chunk_frames);
  return clip;
}

std::vector<int32_t> Qwen3AsrProcessor::MakePromptTokens(const Tokenizer& tokenizer,
                                                         const std::vector<std::string>& segments,
                                                         const std::vector<ClipMel>& clips) const {
  // <|audio_pad|> is a special token, so the reference tokenizer splits the text at it before BPE;
  // tokenizing the segments on their own gives the same ids.
  std::vector<int32_t> input_ids;
  for (size_t i = 0; i < segments.size(); ++i) {
    if (!segments[i].empty()) {
      const std::vector<int32_t> ids = tokenizer.Encode(segments[i].c_str());
      input_ids.insert(input_ids.end(), ids.begin(), ids.end());
    }
    if (i < clips.size()) {
      input_ids.insert(input_ids.end(), static_cast<size_t>(clips[i].num_tokens), audio_token_id_);
    }
  }
  return input_ids;
}

void Qwen3AsrProcessor::AddAudioTensors(const std::vector<ClipMel>& clips, Ort::Allocator& allocator,
                                        NamedTensors& named_tensors) const {
  const int64_t num_clips = static_cast<int64_t>(clips.size());
  const int64_t num_mels = mel_config_.num_mels;
  int64_t longest = 0;
  std::vector<int64_t> valid_frames;
  std::vector<int64_t> tokens_per_clip;
  for (const auto& clip : clips) {
    longest = std::max(longest, clip.num_frames);
    valid_frames.push_back(clip.valid_frames);
    tokens_per_clip.push_back(clip.num_tokens);
  }

  // A container, not a batched encoder run: with several clips the speech state slices each clip's
  // frames (rounded up to whole chunks) back out and runs the encoder once per clip.
  auto batch = OrtValue::CreateTensor<float>(allocator, std::vector<int64_t>{num_clips, longest, num_mels});
  float* batch_data = batch->GetTensorMutableData<float>();
  std::fill_n(batch_data, static_cast<size_t>(num_clips) * static_cast<size_t>(longest * num_mels), 0.0f);
  for (size_t i = 0; i < clips.size(); ++i) {
    std::copy(clips[i].mel.begin(), clips[i].mel.end(), batch_data + i * static_cast<size_t>(longest * num_mels));
  }

  if (mel_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    named_tensors.emplace(std::string(Config::Defaults::AudioEmbedsName), std::make_shared<Tensor>(std::move(batch)));
  } else {
    std::unique_ptr<OrtValue> converted;
    Cast(*batch, converted, *GetDeviceInterface(DeviceType::CPU), mel_type_);
    named_tensors.emplace(std::string(Config::Defaults::AudioEmbedsName), std::make_shared<Tensor>(std::move(converted)));
  }
  named_tensors.emplace(std::string(Config::Defaults::AudioLengthsName),
                        std::make_shared<Tensor>(MakeInt64Tensor(valid_frames, {num_clips}, allocator)));
  named_tensors.emplace(std::string(Config::Defaults::AudioSizesName),
                        std::make_shared<Tensor>(MakeInt64Tensor(tokens_per_clip, {num_clips}, allocator)));
}

std::unique_ptr<NamedTensors> Qwen3AsrProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const std::string prompt = ResolvePrompt(payload);
  const Audios* audios = payload.audios;
  const size_t num_clips = audios ? audios->num_audios_ : 0;
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  const std::vector<std::string> segments = SplitQwen3AsrPrompt(prompt, num_clips);

  std::vector<ClipMel> clips;
  clips.reserve(num_clips);
  for (size_t i = 0; i < num_clips; ++i) {
    clips.push_back(ComputeClipMel(*audios, i));
  }

  named_tensors->emplace(std::string(Config::Defaults::InputIdsName),
                         std::make_shared<Tensor>(MakeInputIds(MakePromptTokens(tokenizer, segments, clips), allocator)));

  if (clips.empty()) {
    // The pipeline reads audio_sizes to skip the speech run.
    named_tensors->emplace(std::string(Config::Defaults::AudioSizesName),
                           std::make_shared<Tensor>(MakeInt64Tensor({0}, {1}, allocator)));
    return named_tensors;
  }

  AddAudioTensors(clips, allocator, *named_tensors);
  return named_tensors;
}

}  // namespace Generators
