// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// NemotronAudioProcessor implementation — STFT, Hann window, Slaney mel filterbank,
// pre-emphasis, center padding, natural-log mel spectrogram.
// All parameters driven by JSON config.  Matches NeMo's
// AudioToMelSpectrogramPreprocessor output.

#include "nemotron_audio_processor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "../filesystem.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Minimal JSON helpers ────────────────────────────────────────────────────
namespace {

std::string Trim(const std::string& s) {
  auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return {};
  return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

bool JsonInt(const std::string& json, const std::string& key, int& out) {
  auto pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos) return false;
  pos = json.find(':', pos);
  if (pos == std::string::npos) return false;
  auto val_start = json.find_first_of("-0123456789", pos + 1);
  if (val_start == std::string::npos) return false;
  auto val_end = json.find_first_not_of("-0123456789", val_start);
  out = std::stoi(json.substr(val_start, val_end - val_start));
  return true;
}

bool JsonFloat(const std::string& json, const std::string& key, float& out) {
  auto pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos) return false;
  pos = json.find(':', pos);
  if (pos == std::string::npos) return false;
  auto val_start = json.find_first_of("-0123456789.eE", pos + 1);
  if (val_start == std::string::npos) return false;
  auto val_end = json.find_first_not_of("-0123456789.eE+", val_start);
  out = std::stof(json.substr(val_start, val_end - val_start));
  return true;
}

}  // anonymous namespace

namespace Generators {

// ── Construction ────────────────────────────────────────────────────────────

NemotronAudioProcessor::NemotronAudioProcessor(const fs::path& config_dir,
                                               const std::string& config_filename) {
  auto config_path = config_dir / config_filename;
  if (config_path.exists()) {
    LoadConfig(config_path);
  }
  // else: keep defaults — they match the Nemotron 0.6B model

  BuildHannWindow();
  BuildMelFilterbank();
}

// ── Config loading ──────────────────────────────────────────────────────────

void NemotronAudioProcessor::LoadConfig(const fs::path& config_path) {
  std::ifstream f(config_path.string());
  if (!f.is_open()) return;

  std::ostringstream ss;
  ss << f.rdbuf();
  std::string json = ss.str();

  int val_int{};
  float val_float{};

  if (JsonInt(json, "target_sample_rate", val_int)) sample_rate_ = val_int;
  if (JsonInt(json, "n_fft", val_int)) n_fft_ = val_int;
  if (JsonInt(json, "frame_length", val_int)) frame_length_ = val_int;
  if (JsonInt(json, "hop_length", val_int)) hop_length_ = val_int;
  if (JsonInt(json, "n_mel", val_int)) n_mel_ = val_int;
  if (JsonFloat(json, "preemph", val_float)) preemph_ = val_float;
}

// ── Pre-computed tables ─────────────────────────────────────────────────────

void NemotronAudioProcessor::BuildHannWindow() {
  // Symmetric Hann window (periodic=False), matching NeMo / torch.hann_window
  hann_window_.resize(frame_length_);
  for (int i = 0; i < frame_length_; ++i) {
    hann_window_[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (frame_length_ - 1)));
  }
}

void NemotronAudioProcessor::BuildMelFilterbank() {
  // Slaney mel scale (librosa default, htk=False) with Slaney area normalization
  // Matches NeMo's AudioToMelSpectrogramPreprocessor defaults:
  //   mel_scale='slaney', mel_norm='slaney'

  // Slaney mel scale breakpoint
  constexpr float kMinLogHz = 1000.0f;
  constexpr float kMinLogMel = 15.0f;   // hz_to_mel(1000)
  constexpr float kLogStep = 0.06875177742094912f;  // ln(6.4) / 27

  auto hz_to_mel = [&](float hz) -> float {
    if (hz < kMinLogHz) {
      return hz / (kMinLogHz / kMinLogMel);  // linear region: f / 66.667
    }
    return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
  };

  auto mel_to_hz = [&](float mel) -> float {
    if (mel < kMinLogMel) {
      return mel * (kMinLogHz / kMinLogMel);  // linear region
    }
    return kMinLogHz * std::exp((mel - kMinLogMel) * kLogStep);
  };

  int num_bins = n_fft_ / 2 + 1;
  float fmin = 0.0f;
  float fmax = static_cast<float>(sample_rate_) / 2.0f;
  float mel_low  = hz_to_mel(fmin);
  float mel_high = hz_to_mel(fmax);

  // n_mel + 2 uniformly spaced points in mel space
  std::vector<float> mel_hz(n_mel_ + 2);
  for (int i = 0; i < n_mel_ + 2; ++i) {
    mel_hz[i] = mel_to_hz(mel_low + (mel_high - mel_low) * i / (n_mel_ + 1));
  }

  // Convert to FFT bin indices (using n_fft, not n_fft+1)
  std::vector<float> bin_pts(n_mel_ + 2);
  for (int i = 0; i < n_mel_ + 2; ++i) {
    bin_pts[i] = static_cast<float>(n_fft_) * mel_hz[i] / static_cast<float>(sample_rate_);
  }

  // Build triangular filters with Slaney normalization
  mel_filterbank_.assign(n_mel_, std::vector<float>(num_bins, 0.0f));
  for (int m = 0; m < n_mel_; ++m) {
    float lower = bin_pts[m];
    float center = bin_pts[m + 1];
    float upper = bin_pts[m + 2];

    // Slaney normalization: 2 / (upper_hz - lower_hz)
    float enorm = 2.0f / (mel_hz[m + 2] - mel_hz[m]);

    for (int k = 0; k < num_bins; ++k) {
      float fk = static_cast<float>(k);
      float weight = 0.0f;
      if (fk >= lower && fk <= center && center > lower) {
        weight = (fk - lower) / (center - lower);
      } else if (fk >= center && fk <= upper && upper > center) {
        weight = (upper - fk) / (upper - center);
      }
      mel_filterbank_[m][k] = weight * enorm;
    }
  }
}

// ── DSP ─────────────────────────────────────────────────────────────────────

void NemotronAudioProcessor::ComputePowerSpectrum(const float* frame,
                                                  std::vector<float>& out) const {
  int num_bins = n_fft_ / 2 + 1;
  out.resize(num_bins);

  // frame has frame_length_ windowed samples; DFT is n_fft_ points
  // (implicit zero-padding for indices >= frame_length_)
  for (int k = 0; k < num_bins; ++k) {
    float real_sum = 0.0f, imag_sum = 0.0f;
    for (int n = 0; n < frame_length_; ++n) {
      float val = frame[n] * hann_window_[n];
      float angle = 2.0f * static_cast<float>(M_PI) * k * n / n_fft_;
      real_sum += val * std::cos(angle);
      imag_sum -= val * std::sin(angle);
    }
    out[k] = real_sum * real_sum + imag_sum * imag_sum;  // power spectrum
  }
}

LogMelResult NemotronAudioProcessor::ComputeLogMel(const float* audio,
                                                   size_t num_samples) const {
  // 1. Pre-emphasis: x'[n] = x[n] - preemph * x[n-1]
  std::vector<float> preemph_audio(num_samples);
  if (num_samples > 0) {
    preemph_audio[0] = audio[0];
    for (size_t i = 1; i < num_samples; ++i) {
      preemph_audio[i] = audio[i] - preemph_ * audio[i - 1];
    }
  }

  // 2. Center padding: pad n_fft/2 zeros on each side (matches NeMo center=True)
  int pad = n_fft_ / 2;
  std::vector<float> padded(pad + preemph_audio.size() + pad, 0.0f);
  std::copy(preemph_audio.begin(), preemph_audio.end(), padded.begin() + pad);

  // Ensure at least one full frame
  if (static_cast<int>(padded.size()) < frame_length_) {
    padded.resize(frame_length_, 0.0f);
  }

  int num_frames = static_cast<int>((padded.size() - frame_length_) / hop_length_) + 1;

  LogMelResult result;
  result.num_mels  = n_mel_;
  result.num_frames = num_frames;
  result.data.resize(n_mel_ * num_frames);

  int num_bins = n_fft_ / 2 + 1;
  std::vector<float> power(num_bins);

  // Log guard: NeMo uses log(x + 2^-24)
  constexpr float kLogGuard = 5.960464477539063e-8f;  // 2^-24

  for (int t = 0; t < num_frames; ++t) {
    const float* frame = padded.data() + t * hop_length_;
    ComputePowerSpectrum(frame, power);

    for (int m = 0; m < n_mel_; ++m) {
      float val = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        val += mel_filterbank_[m][k] * power[k];
      }
      // Natural log with additive guard (matches NeMo)
      result.data[m * num_frames + t] = std::log(val + kLogGuard);
    }
  }

  return result;
}

}  // namespace Generators
