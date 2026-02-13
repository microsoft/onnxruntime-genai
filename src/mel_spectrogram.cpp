// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Standalone log-mel spectrogram extraction (Slaney scale, matching librosa/NeMo).
// No ONNX Runtime or other framework dependencies — pure C++ with standard library only.

#include "mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mel {

// ─── Slaney mel scale constants ─────────────────────────────────────────────

static constexpr float kMinLogHz = 1000.0f;
static constexpr float kMinLogMel = 15.0f;               // 1000 / (200/3)
static constexpr float kLinScale = 200.0f / 3.0f;        // Hz per mel (linear region)
static constexpr float kLogStep = 0.06875177742094912f;   // log(6.4) / 27

// ─── Mel scale conversions ──────────────────────────────────────────────────

float HzToMel(float hz) {
  if (hz < kMinLogHz) return hz / kLinScale;
  return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
}

float MelToHz(float mel) {
  if (mel < kMinLogMel) return mel * kLinScale;
  return kMinLogHz * std::exp((mel - kMinLogMel) * kLogStep);
}

// ─── Filterbank creation ────────────────────────────────────────────────────

std::vector<std::vector<float>> CreateMelFilterbank(int num_mels, int fft_size, int sample_rate) {
  int num_bins = fft_size / 2 + 1;
  float mel_low = HzToMel(0.0f);
  float mel_high = HzToMel(static_cast<float>(sample_rate) / 2.0f);

  // Compute mel center frequencies in Hz (num_mels + 2 points)
  std::vector<float> mel_f(num_mels + 2);
  for (int i = 0; i < num_mels + 2; ++i) {
    float m = mel_low + (mel_high - mel_low) * i / (num_mels + 1);
    mel_f[i] = MelToHz(m);
  }

  // Differences between consecutive mel center frequencies (Hz)
  std::vector<float> fdiff(num_mels + 1);
  for (int i = 0; i < num_mels + 1; ++i) {
    fdiff[i] = mel_f[i + 1] - mel_f[i];
  }

  // FFT bin center frequencies in Hz
  std::vector<float> fft_freqs(num_bins);
  for (int k = 0; k < num_bins; ++k) {
    fft_freqs[k] = static_cast<float>(k) * sample_rate / fft_size;
  }

  // Build triangular filterbank with Slaney normalization (matches librosa exactly)
  std::vector<std::vector<float>> filterbank(num_mels, std::vector<float>(num_bins, 0.0f));
  for (int m = 0; m < num_mels; ++m) {
    for (int k = 0; k < num_bins; ++k) {
      float lower = (fft_freqs[k] - mel_f[m]) / (fdiff[m] + 1e-10f);
      float upper = (mel_f[m + 2] - fft_freqs[k]) / (fdiff[m + 1] + 1e-10f);
      filterbank[m][k] = std::max(0.0f, std::min(lower, upper));
    }
    // Slaney area normalization: 2 / bandwidth
    float enorm = 2.0f / (mel_f[m + 2] - mel_f[m] + 1e-10f);
    for (int k = 0; k < num_bins; ++k) {
      filterbank[m][k] *= enorm;
    }
  }
  return filterbank;
}

// ─── Window functions ───────────────────────────────────────────────────────

std::vector<float> HannWindowSymmetric(int win_length) {
  std::vector<float> window(win_length);
  for (int i = 0; i < win_length; ++i) {
    window[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (win_length - 1)));
  }
  return window;
}

std::vector<float> HannWindowPeriodic(int fft_size, int win_length) {
  std::vector<float> window(fft_size, 0.0f);
  int offset = (fft_size - win_length) / 2;
  for (int i = 0; i < win_length; ++i) {
    window[offset + i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / win_length));
  }
  return window;
}

// ─── Single-frame DFT ──────────────────────────────────────────────────────

void ComputeSTFTFrame(const float* frame, const float* window, int frame_len,
                      int fft_size, std::vector<float>& magnitudes) {
  int num_bins = fft_size / 2 + 1;
  magnitudes.resize(num_bins);

  for (int k = 0; k < num_bins; ++k) {
    float real_sum = 0.0f, imag_sum = 0.0f;
    for (int n = 0; n < frame_len; ++n) {
      float val = frame[n] * window[n];
      float angle = 2.0f * static_cast<float>(M_PI) * k * n / fft_size;
      real_sum += val * std::cos(angle);
      imag_sum -= val * std::sin(angle);
    }
    magnitudes[k] = real_sum * real_sum + imag_sum * imag_sum;
  }
}

// ─── Batch (offline) log-mel extraction ─────────────────────────────────────

std::vector<float> ComputeLogMelBatch(const float* audio, size_t num_samples,
                                      const MelConfig& cfg, int& out_num_frames) {
  // Lazily-initialized statics are fine for batch mode (same config per process).
  // If you need thread-safety with multiple configs, pass the filterbank in explicitly.
  static auto mel_filters = CreateMelFilterbank(cfg.num_mels, cfg.fft_size, cfg.sample_rate);
  static auto window = HannWindowPeriodic(cfg.fft_size, cfg.win_length);

  int n = static_cast<int>(num_samples);

  // Apply pre-emphasis: y[n] = x[n] - preemph * x[n-1]
  std::vector<float> preemphasized(n);
  if (n > 0) {
    preemphasized[0] = audio[0];  // No previous sample for first sample
    for (int i = 1; i < n; ++i) {
      preemphasized[i] = audio[i] - cfg.preemph * audio[i - 1];
    }
  }

  // Center-pad both sides: fft_size/2 zeros on each side (matching torch.stft center=True)
  int pad = cfg.fft_size / 2;
  std::vector<float> padded(pad + n + pad, 0.0f);
  if (n > 0) {
    std::memcpy(padded.data() + pad, preemphasized.data(), n * sizeof(float));
  }

  if (static_cast<int>(padded.size()) < cfg.fft_size) {
    padded.resize(cfg.fft_size, 0.0f);
  }

  // Frame count using fft_size as frame size (matching torch.stft)
  int num_frames = static_cast<int>((padded.size() - cfg.fft_size) / cfg.hop_length) + 1;
  out_num_frames = num_frames;

  int num_bins = cfg.fft_size / 2 + 1;
  std::vector<float> magnitudes;
  std::vector<float> mel_spec(cfg.num_mels * num_frames);

  for (int t = 0; t < num_frames; ++t) {
    const float* frame = padded.data() + t * cfg.hop_length;
    ComputeSTFTFrame(frame, window.data(), cfg.fft_size, cfg.fft_size, magnitudes);

    for (int m = 0; m < cfg.num_mels; ++m) {
      float val = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        val += mel_filters[m][k] * magnitudes[k];
      }
      mel_spec[m * num_frames + t] = std::log(val + cfg.log_eps);
    }
  }

  return mel_spec;
}

// ─── Streaming log-mel extraction ───────────────────────────────────────────

StreamingMelExtractor::StreamingMelExtractor(const MelConfig& cfg)
    : cfg_(cfg) {
  mel_filters_ = CreateMelFilterbank(cfg_.num_mels, cfg_.fft_size, cfg_.sample_rate);
  hann_window_ = HannWindowSymmetric(cfg_.win_length);
  audio_overlap_.assign(cfg_.fft_size / 2, 0.0f);
  preemph_last_sample_ = 0.0f;
}

void StreamingMelExtractor::Reset() {
  audio_overlap_.assign(cfg_.fft_size / 2, 0.0f);
  preemph_last_sample_ = 0.0f;
}

std::pair<std::vector<float>, int> StreamingMelExtractor::Process(
    const float* audio, size_t num_samples) {
  // Apply pre-emphasis filter: y[n] = x[n] - preemph * x[n-1]
  std::vector<float> preemphasized(num_samples);
  if (num_samples > 0) {
    preemphasized[0] = audio[0] - cfg_.preemph * preemph_last_sample_;
    for (size_t i = 1; i < num_samples; ++i) {
      preemphasized[i] = audio[i] - cfg_.preemph * audio[i - 1];
    }
    preemph_last_sample_ = audio[num_samples - 1];
  }

  // Left-only center pad for streaming: prepend overlap from previous chunk.
  // For the first chunk this is zeros (matching center=True left edge).
  int pad = cfg_.fft_size / 2;  // 256 samples
  std::vector<float> padded(pad + num_samples);
  std::memcpy(padded.data(), audio_overlap_.data(), pad * sizeof(float));
  std::memcpy(padded.data() + pad, preemphasized.data(), num_samples * sizeof(float));

  // Update overlap buffer for next chunk
  if (num_samples >= static_cast<size_t>(pad)) {
    audio_overlap_.assign(preemphasized.data() + num_samples - pad,
                          preemphasized.data() + num_samples);
  } else {
    size_t keep = pad - num_samples;
    std::vector<float> new_overlap(pad, 0.0f);
    std::memcpy(new_overlap.data(), audio_overlap_.data() + num_samples, keep * sizeof(float));
    std::memcpy(new_overlap.data() + keep, preemphasized.data(), num_samples * sizeof(float));
    audio_overlap_ = std::move(new_overlap);
  }

  // Window centering offset (symmetric window smaller than fft_size)
  int win_offset = (cfg_.fft_size - cfg_.win_length) / 2;  // e.g. 56

  // Right-pad to accommodate the window offset for the last frame
  padded.resize(padded.size() + win_offset, 0.0f);

  if (static_cast<int>(padded.size()) < win_offset + cfg_.win_length) {
    padded.resize(win_offset + cfg_.win_length, 0.0f);
  }

  // Frame count
  int num_frames = static_cast<int>((padded.size() - win_offset - cfg_.win_length) / cfg_.hop_length) + 1;

  int num_bins = cfg_.fft_size / 2 + 1;
  std::vector<float> mel_spec(cfg_.num_mels * num_frames);

  for (int t = 0; t < num_frames; ++t) {
    const float* frame = padded.data() + t * cfg_.hop_length + win_offset;

    // Inline DFT with symmetric Hann window (win_length samples)
    std::vector<float> magnitudes(num_bins);
    for (int k = 0; k < num_bins; ++k) {
      float real_sum = 0.0f, imag_sum = 0.0f;
      for (int n = 0; n < cfg_.win_length; ++n) {
        float val = frame[n] * hann_window_[n];
        float angle = 2.0f * static_cast<float>(M_PI) * k * n / cfg_.fft_size;
        real_sum += val * std::cos(angle);
        imag_sum -= val * std::sin(angle);
      }
      magnitudes[k] = real_sum * real_sum + imag_sum * imag_sum;
    }

    // Apply mel filterbank + log
    for (int m = 0; m < cfg_.num_mels; ++m) {
      float val = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        val += mel_filters_[m][k] * magnitudes[k];
      }
      mel_spec[m * num_frames + t] = std::log(val + cfg_.log_eps);
    }
  }

  return {mel_spec, num_frames};
}

}  // namespace mel
