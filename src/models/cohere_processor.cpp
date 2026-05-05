// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "cohere_processor.h"
#include "silero_vad.h"
#include "speech_features.hpp"
#include "c_api_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace Generators {

CohereProcessor::CohereProcessor(Config& config, const SessionInfo& session_info, Model& model)
    : audio_features_type_{session_info.GetInputDataType(config.model.encoder.inputs.audio_features)},
      config_{&config} {
  mel_cfg_.num_mels    = config.model.num_mels;
  mel_cfg_.fft_size    = config.model.fft_size;
  mel_cfg_.hop_length  = config.model.hop_length;
  mel_cfg_.win_length  = config.model.win_length;
  mel_cfg_.sample_rate = config.model.sample_rate;
  mel_cfg_.preemph     = config.model.preemph;
  mel_cfg_.log_eps     = config.model.log_eps;
  norm_eps_ = config.model.norm_eps;

  config.AddMapping(std::string(Config::Defaults::AudioFeaturesName), config.model.encoder.inputs.audio_features);
  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.decoder.inputs.input_ids);

  // ComputeMelFromPCM produces float32 mel tensors. If the encoder expects a
  // different dtype (e.g. float16/bfloat16), the runtime will fail with an
  // opaque type-mismatch error; surface it here with a clear message instead.
  if (audio_features_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(
        "CohereProcessor: encoder input '" + config.model.encoder.inputs.audio_features +
        "' must be float32; got ONNX type " + std::to_string(static_cast<int>(audio_features_type_)) +
        ". Cohere mel computation only supports float32 audio features.");
  }

  if (!config.model.vad.filename.empty()) {
    vad_ = CreateSileroVad(model);
  }
}

CohereProcessor::~CohereProcessor() = default;

static std::pair<const float*, size_t> GetDecodedPCM(
    OrtxRawAudios* raw_audios, size_t index, int target_sample_rate,
    ort_extensions::OrtxObjectPtr<OrtxTensorResult>& decode_result_holder,
    int& out_sample_rate) {
  OrtxTensorResult* decode_result = nullptr;
  // Pass target_sample_rate so ortx resamples on decode. Always downmix to mono.
  CheckResult(OrtxDecodeAudio(raw_audios, index, target_sample_rate, /*stereo_to_mono=*/1, &decode_result));
  decode_result_holder.reset(decode_result);

  ort_extensions::OrtxObjectPtr<OrtxTensor> pcm_tensor;
  CheckResult(OrtxTensorResultGetAt(decode_result, 0, pcm_tensor.ToBeAssigned()));

  ort_extensions::OrtxObjectPtr<OrtxTensor> sr_tensor;
  CheckResult(OrtxTensorResultGetAt(decode_result, 1, sr_tensor.ToBeAssigned()));

  const void* pcm_data = nullptr;
  const int64_t* pcm_shape = nullptr;
  size_t pcm_dims = 0;
  CheckResult(OrtxGetTensorData(pcm_tensor.get(), &pcm_data, &pcm_shape, &pcm_dims));

  const void* sr_data = nullptr;
  const int64_t* sr_shape = nullptr;
  size_t sr_dims = 0;
  CheckResult(OrtxGetTensorData(sr_tensor.get(), &sr_data, &sr_shape, &sr_dims));

  out_sample_rate = static_cast<int>(*static_cast<const int64_t*>(sr_data));

  size_t num_samples = 1;
  for (size_t d = 0; d < pcm_dims; ++d) num_samples *= pcm_shape[d];

  return {static_cast<const float*>(pcm_data), num_samples};
}

// Waveform splitting with Silero VAD.
//   1. Score full audio in non-overlapping windows (window_size depends on SR).
//   2. Convert to binary speech mask using vad_->GetThreshold().
//   3. Merge speech regions separated by silence shorter than min_silence_ms.
//   4. Drop regions shorter than min_speech_ms.
//   5. Pad each region by speech_pad_ms on both sides (clamped to file bounds).
//   6. Force-split regions longer than max_speech_s.
//
// Returns a list of (start, end) sample ranges. Empty if no speech detected.
std::vector<std::vector<std::pair<size_t, size_t>>> CohereProcessor::SplitWaveformByVad(
    const float* samples, size_t num_samples, int sample_rate) const {
  if (!vad_) return {};

  const size_t window = static_cast<size_t>(vad_->GetWindowSize());
  const float threshold = vad_->GetThreshold();

  std::vector<uint8_t> is_speech;
  is_speech.reserve(num_samples / window);
  for (size_t off = 0; off + window <= num_samples; off += window) {
    float p = vad_->ProcessWindow(samples + off, window);
    is_speech.push_back(p >= threshold ? 1u : 0u);
  }

  // Step 1: collect raw [start_frame, end_frame) speech regions.
  std::vector<std::pair<size_t, size_t>> regions;
  for (size_t i = 0; i < is_speech.size();) {
    if (!is_speech[i]) { ++i; continue; }
    size_t s = i;
    while (i < is_speech.size() && is_speech[i]) ++i;
    regions.push_back({s, i});
  }
  if (regions.empty()) return {};

  // Step 2: merge regions whose gap is shorter than min_silence_ms. These are
  // very short pauses inside a single utterance; we keep them as one region.
  const size_t samples_per_ms = static_cast<size_t>(sample_rate) / 1000;
  const size_t min_silence_frames =
      std::max<size_t>(1, (static_cast<size_t>(config_->model.cohere_vad_min_silence_ms) * samples_per_ms) / window);
  std::vector<std::pair<size_t, size_t>> merged;
  merged.reserve(regions.size());
  merged.push_back(regions.front());
  for (size_t k = 1; k < regions.size(); ++k) {
    auto& last = merged.back();
    if (regions[k].first - last.second < min_silence_frames) {
      last.second = regions[k].second;
    } else {
      merged.push_back(regions[k]);
    }
  }

  // Step 3: group regions into chunks so each chunk has at least
  // `min_speech_ms` of *speech*
  const size_t min_speech_frames =
      std::max<size_t>(1, (static_cast<size_t>(config_->model.cohere_vad_min_speech_ms) * samples_per_ms) / window);

  auto speech_frames_in_chunk = [](const std::vector<std::pair<size_t, size_t>>& chunk) {
    size_t total = 0;
    for (auto& r : chunk) total += r.second - r.first;
    return total;
  };

  std::vector<std::vector<std::pair<size_t, size_t>>> grouped;  // chunks of frame ranges
  for (auto& r : merged) {
    if (!grouped.empty() && speech_frames_in_chunk(grouped.back()) < min_speech_frames) {
      grouped.back().push_back(r);
    } else {
      grouped.push_back({r});
    }
  }
  // Tail-fixup: if last chunk still has too little speech, fold into prior.
  if (grouped.size() >= 2 && speech_frames_in_chunk(grouped.back()) < min_speech_frames) {
    auto tail = std::move(grouped.back());
    grouped.pop_back();
    for (auto& r : tail) grouped.back().push_back(r);
  }
  if (grouped.empty()) return {};

  // Step 4: convert each sub-region from frame to sample space, apply
  // speech_pad_ms padding, clamp to file. Then enforce max_speech_s on the
  // total speech duration of a chunk (force-split inside the sub-region list
  // if needed).
  const size_t pad_samples = static_cast<size_t>(config_->model.cohere_vad_speech_pad_ms) * samples_per_ms;
  const size_t max_speech_samples =
      static_cast<size_t>(config_->model.cohere_vad_max_speech_s * static_cast<float>(sample_rate));

  std::vector<std::vector<std::pair<size_t, size_t>>> out;
  out.reserve(grouped.size());
  for (auto& chunk_frames : grouped) {
    std::vector<std::pair<size_t, size_t>> chunk_samples;
    chunk_samples.reserve(chunk_frames.size());
    for (auto& r : chunk_frames) {
      size_t s = r.first * window;
      size_t e = r.second * window;
      s = (s > pad_samples) ? s - pad_samples : 0;
      e = std::min(num_samples, e + pad_samples);
      chunk_samples.push_back({s, e});
    }
    // Greedy pack: emit a new chunk when adding the next region would exceed
    // max_speech_samples. If a single region is itself longer than the cap,
    // hard-split it into back-to-back max_speech_samples pieces (the encoder
    // has a hard 30s limit).
    std::vector<std::pair<size_t, size_t>> pending_chunk;
    size_t pending_len = 0;
    for (auto& r : chunk_samples) {
      size_t s = r.first;
      while (r.second - s > max_speech_samples) {
        if (!pending_chunk.empty()) {
          out.push_back(std::move(pending_chunk));
          pending_chunk.clear();
          pending_len = 0;
        }
        out.push_back({{s, s + max_speech_samples}});
        s += max_speech_samples;
      }
      size_t len = r.second - s;
      if (pending_len + len > max_speech_samples && !pending_chunk.empty()) {
        out.push_back(std::move(pending_chunk));
        pending_chunk.clear();
        pending_len = 0;
      }
      pending_chunk.push_back({s, r.second});
      pending_len += len;
    }
    if (!pending_chunk.empty()) out.push_back(std::move(pending_chunk));
  }
  return out;
}

std::pair<std::unique_ptr<OrtValue>, int64_t> CohereProcessor::ComputeMelFromPCM(
    const float* samples, size_t num_samples) const {
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};

  // Step 1: PCM -> log-mel spectrogram.
  int num_frames = 0;
  auto mel_data = nemo_mel::NemoComputeLogMelBatch(samples, num_samples, mel_cfg_, num_frames);
  const int64_t num_mels = mel_cfg_.num_mels;

  // Step 2: per-feature normalize with feature-first normalization.
  ort_extensions::PerFeatureNormalize norm_kernel;
  ort_extensions::AttrDict norm_attrs{
      {"eps", static_cast<double>(norm_eps_)},
      {"feature_first", int64_t{1}},
  };
  if (auto status = norm_kernel.Init(norm_attrs); !status.IsOk()) {
    throw std::runtime_error(std::string("PerFeatureNormalize::Init failed: ") + status.Message());
  }

  std::vector<int64_t> mel_shape{num_mels, static_cast<int64_t>(num_frames)};
  ortc::Tensor<float> norm_in(mel_shape, mel_data.data());
  ortc::Tensor<float> norm_out(&ort_extensions::CppAllocator::Instance());
  if (auto status = norm_kernel.Compute(norm_in, norm_out); !status.IsOk()) {
    throw std::runtime_error(std::string("PerFeatureNormalize::Compute failed: ") + status.Message());
  }

  // Step 3: copy normalized data into an OrtValue [1, num_mels, num_frames].
  auto shape = std::array<int64_t, 3>{1, num_mels, static_cast<int64_t>(num_frames)};
  auto tensor = OrtValue::CreateTensor<float>(allocator, std::span<int64_t>(shape.data(), 3));
  std::memcpy(tensor->GetTensorMutableData<float>(),
              norm_out.Data(),
              static_cast<size_t>(num_mels) * num_frames * sizeof(float));

  return {std::move(tensor), static_cast<int64_t>(num_frames)};
}

std::unique_ptr<NamedTensors> CohereProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  const auto* audios = payload.audios;
  if (!audios || !audios->audios_)
    throw std::runtime_error("No audios provided to process.");

  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<NamedTensors>();

  // Decode audio to PCM, resampled to the model's expected sample rate.
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> decode_result;
  int sample_rate = 0;
  auto [pcm_data, num_samples] = GetDecodedPCM(audios->audios_.get(), 0, mel_cfg_.sample_rate, decode_result, sample_rate);

  // Split waveform: VAD-based when configured, else pass the whole audio as a
  // single chunk. Cohere's training enforced a 30s max-utterance limit, so for
  // long-form audio the user MUST enable VAD via `model.vad.filename`. With no
  // VAD configured, we trust the caller to feed audio that fits the model.
  std::vector<std::vector<std::pair<size_t, size_t>>> chunk_ranges;
  if (vad_) {
    chunk_ranges = SplitWaveformByVad(pcm_data, num_samples, sample_rate);
  } else {
    chunk_ranges.push_back({{0, num_samples}});
  }
  if (chunk_ranges.empty()) {
    throw std::runtime_error("CohereProcessor: no audio chunks produced (empty waveform or invalid chunking config).");
  }

  // Helper: concatenate the speech-only samples for a chunk into a single
  // contiguous buffer (no silence between sub-regions).
  auto build_chunk_pcm = [pcm_data](const std::vector<std::pair<size_t, size_t>>& subs) {
    size_t total = 0;
    for (auto& r : subs) total += r.second - r.first;
    std::vector<float> buf(total);
    size_t off = 0;
    for (auto& r : subs) {
      size_t n = r.second - r.first;
      std::memcpy(buf.data() + off, pcm_data + r.first, n * sizeof(float));
      off += n;
    }
    return buf;
  };

  // Compute mel for first chunk
  {
    auto chunk_pcm = build_chunk_pcm(chunk_ranges[0]);
    auto [mel_tensor, mel_frames] = ComputeMelFromPCM(chunk_pcm.data(), chunk_pcm.size());

    named_tensors->emplace(std::string(Config::Defaults::AudioFeaturesName),
                           std::make_shared<Tensor>(std::move(mel_tensor)));

    auto ml_shape = std::array<int64_t, 1>{1};
    auto ml_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(ml_shape.data(), 1));
    ml_tensor->GetTensorMutableData<int64_t>()[0] = mel_frames;
    named_tensors->emplace("mel_length", std::make_shared<Tensor>(std::move(ml_tensor)));
  }

  // Compute mel for remaining chunks
  for (size_t i = 1; i < chunk_ranges.size(); ++i) {
    auto chunk_pcm = build_chunk_pcm(chunk_ranges[i]);
    auto [mel_tensor, mel_frames] = ComputeMelFromPCM(chunk_pcm.data(), chunk_pcm.size());

    named_tensors->emplace("cohere_chunk_" + std::to_string(i),
                           std::make_shared<Tensor>(std::move(mel_tensor)));

    auto ml_shape = std::array<int64_t, 1>{1};
    auto ml_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(ml_shape.data(), 1));
    ml_tensor->GetTensorMutableData<int64_t>()[0] = mel_frames;
    named_tensors->emplace("cohere_chunk_mel_length_" + std::to_string(i),
                           std::make_shared<Tensor>(std::move(ml_tensor)));
  }

  // Store chunk count
  {
    auto count_shape = std::array<int64_t, 1>{1};
    auto count_tensor = OrtValue::CreateTensor<int64_t>(allocator, std::span<int64_t>(count_shape.data(), 1));
    count_tensor->GetTensorMutableData<int64_t>()[0] = static_cast<int64_t>(chunk_ranges.size());
    named_tensors->emplace("cohere_chunk_count", std::make_shared<Tensor>(std::move(count_tensor)));
  }

  // Encode prompt tokens
  std::shared_ptr<Tensor> input_ids;
  if (!payload.prompt.empty()) {
    const char* prompt_cstr = payload.prompt.c_str();
    input_ids = tokenizer.EncodeBatch(std::span<const char*>(&prompt_cstr, 1));
  } else if (!payload.prompts.empty()) {
    input_ids = tokenizer.EncodeBatch(payload.prompts);
  } else {
    throw std::runtime_error(
        "CohereProcessor: a non-empty prompt is required for Cohere Transcribe "
        "(neither payload.prompt nor payload.prompts was provided).");
  }
  named_tensors->emplace(std::string(Config::Defaults::InputIdsName), input_ids);

  return named_tensors;
}

}  // namespace Generators
