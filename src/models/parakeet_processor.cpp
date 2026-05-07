// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "../generators.h"
#include "model.h"
#include "parakeet_processor.h"
#include "nemo_mel_spectrogram.h"

namespace Generators {

namespace {

// Decode one audio file from OrtxRawAudios into resampled mono PCM (float32).
std::vector<float> DecodeAudioToMonoPCM(OrtxRawAudios* raw_audios, size_t index, int target_sample_rate) {
  OrtxTensorResult* decode_result = nullptr;
  CheckResult(OrtxDecodeAudio(raw_audios, index, target_sample_rate, /*stereo_to_mono=*/1, &decode_result));
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> dr_holder(decode_result);

  ort_extensions::OrtxObjectPtr<OrtxTensor> pcm_tensor;
  CheckResult(OrtxTensorResultGetAt(decode_result, 0, pcm_tensor.ToBeAssigned()));

  const void* pcm_data = nullptr;
  const int64_t* pcm_shape = nullptr;
  size_t pcm_dims = 0;
  CheckResult(OrtxGetTensorData(pcm_tensor.get(), &pcm_data, &pcm_shape, &pcm_dims));
  size_t n = 1;
  for (size_t d = 0; d < pcm_dims; ++d) n *= static_cast<size_t>(pcm_shape[d]);
  const float* p = static_cast<const float*>(pcm_data);
  return std::vector<float>(p, p + n);
}

// In-place per-feature (per mel bin) normalization across time. mel layout: [num_mels, num_frames] row-major.
void PerFeatureNormalize(float* mel, int num_mels, int num_frames, float eps = 1e-5f) {
  if (num_frames <= 0) return;
  for (int m = 0; m < num_mels; ++m) {
    float* row = mel + static_cast<size_t>(m) * num_frames;
    double sum = 0.0;
    for (int t = 0; t < num_frames; ++t) sum += row[t];
    double mean = sum / num_frames;
    double var = 0.0;
    for (int t = 0; t < num_frames; ++t) { double d = row[t] - mean; var += d * d; }
    var /= num_frames;
    float stddev = std::sqrt(static_cast<float>(var) + eps);
    for (int t = 0; t < num_frames; ++t) row[t] = static_cast<float>((row[t] - mean) / stddev);
  }
}

}  // namespace

ParakeetProcessor::ParakeetProcessor(Config& config, const SessionInfo& /*session_info*/) {
  const auto& m = config.model;
  if (m.sample_rate > 0) sample_rate_ = m.sample_rate;
  if (m.num_mels > 0) num_mels_ = m.num_mels;
  if (m.fft_size > 0) fft_size_ = m.fft_size;
  if (m.hop_length > 0) hop_length_ = m.hop_length;
  if (m.win_length > 0) win_length_ = m.win_length;
  if (m.preemph != 0.0f) preemph_ = m.preemph;
  if (m.log_eps > 0.0f) log_eps_ = m.log_eps;
  if (m.vocab_size > 0) blank_id_ = m.vocab_size;
  if (m.blank_id > 0) blank_id_ = m.blank_id;

  // Load tokens.txt next to the model config. Format: "<token> <id>" per line.
  fs::path tokens_path = config.config_path / "tokens.txt";
  std::ifstream f(tokens_path.string());
  if (!f.is_open()) {
    throw std::runtime_error("ParakeetProcessor: cannot open tokens.txt at " + tokens_path.string());
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    // Last whitespace-separated field is the id; everything before is the token (may contain spaces — but normally single token).
    auto pos = line.find_last_of(" \t");
    if (pos == std::string::npos) continue;
    std::string tok = line.substr(0, pos);
    int id = 0;
    try {
      id = std::stoi(line.substr(pos + 1));
    } catch (...) { continue; }
    if (id < 0) continue;
    if (static_cast<size_t>(id) >= id_to_token_.size()) id_to_token_.resize(id + 1);
    id_to_token_[id] = std::move(tok);
  }

  // Register the audio_features mapping for SetInputs flow.
  config.AddMapping(std::string(Config::Defaults::AudioFeaturesName),
                    config.model.encoder.inputs.audio_features.empty()
                        ? std::string{Config::Defaults::AudioFeaturesName}
                        : config.model.encoder.inputs.audio_features);
}

std::unique_ptr<NamedTensors> ParakeetProcessor::Process(const Tokenizer& /*tokenizer*/,
                                                          const Payload& payload) const {
  if (!payload.audios || !payload.audios->audios_) {
    throw std::runtime_error("ParakeetProcessor::Process called without audios.");
  }
  if (payload.audios->num_audios_ != 1) {
    throw std::runtime_error("ParakeetProcessor: only batch_size=1 supported (got " +
                             std::to_string(payload.audios->num_audios_) + ").");
  }

  // 1. Decode + resample to mono PCM.
  std::vector<float> pcm = DecodeAudioToMonoPCM(payload.audios->audios_.get(), 0, sample_rate_);

  // 2. Compute log-mel.
  nemo_mel::NemoMelConfig cfg{
      num_mels_, fft_size_, hop_length_, win_length_, sample_rate_, preemph_, log_eps_,
  };
  int num_frames = 0;
  std::vector<float> mel = nemo_mel::NemoComputeLogMelBatch(pcm.data(), pcm.size(), cfg, num_frames);

  // 3. Per-feature normalize across the whole utterance (matches NeMo per_feature normalize_type).
  PerFeatureNormalize(mel.data(), num_mels_, num_frames);

  // 4. Wrap as [1, num_mels, num_frames] float32 OrtValue.
  Ort::Allocator& allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto shape = std::array<int64_t, 3>{1, num_mels_, num_frames};
  auto ort = OrtValue::CreateTensor<float>(allocator, shape);
  std::memcpy(ort->GetTensorMutableData<float>(), mel.data(), sizeof(float) * mel.size());

  auto named = std::make_unique<NamedTensors>();
  named->emplace(std::string(Config::Defaults::AudioFeaturesName),
                 std::make_shared<Tensor>(std::move(ort)));

  // 5. Provide a 1-token dummy input_ids so Generator.SetInputs triggers ComputeLogits.
  // The token value is irrelevant — ParakeetState ignores it.
  auto ids_shape = std::array<int64_t, 2>{1, 1};
  auto ids = OrtValue::CreateTensor<int32_t>(allocator, ids_shape);
  ids->GetTensorMutableData<int32_t>()[0] = 0;
  named->emplace(std::string(Config::Defaults::InputIdsName), std::make_shared<Tensor>(std::move(ids)));

  return named;
}

std::optional<std::string> ParakeetProcessor::Decode(std::span<const int32_t> tokens) const {
  static const std::string kSpace{"\xe2\x96\x81"};  // U+2581 SentencePiece word boundary
  std::string out;
  out.reserve(tokens.size() * 2);
  for (int32_t id : tokens) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
    if (id == blank_id_) continue;
    const std::string& t = id_to_token_[id];
    if (t.empty()) continue;
    // Skip special tokens of the form <...>
    if (t.size() >= 2 && t.front() == '<' && t.back() == '>') continue;

    // Replace leading ▁ with a space (word boundary). SentencePiece convention.
    if (t.compare(0, kSpace.size(), kSpace) == 0) {
      if (!out.empty()) out.push_back(' ');
      out.append(t, kSpace.size(), std::string::npos);
    } else {
      out.append(t);
    }
  }
  // Trim leading space if any
  size_t i = 0;
  while (i < out.size() && out[i] == ' ') ++i;
  if (i > 0) out.erase(0, i);
  return out;
}

}  // namespace Generators
