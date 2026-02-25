// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// NeMo-compatible log-mel spectrogram — standalone C++ implementation.
// Matches NeMo's AudioToMelSpectrogramPreprocessor + librosa.filters.mel exactly.

#include "nemo_mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <complex>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace parakeet_mel {

// ─── Symmetric Hann window (periodic=False) ─────────────────────────────────
// w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))  for n = 0..N-1
// Matches torch.hann_window(N, periodic=False)

static std::vector<float> SymmetricHannWindow(int length) {
  std::vector<float> window(length);
  if (length == 1) {
    window[0] = 1.0f;
    return window;
  }
  for (int i = 0; i < length; ++i) {
    window[i] = 0.5f * (1.0f - std::cos(2.0 * M_PI * i / (length - 1)));
  }
  return window;
}

// ─── Radix-2 Cooley-Tukey FFT (in-place, decimation-in-time) ────────────────

static void FFT(std::vector<std::complex<float>>& x) {
  int N = static_cast<int>(x.size());
  if (N <= 1) return;

  // Bit-reversal permutation
  for (int i = 1, j = 0; i < N; ++i) {
    int bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(x[i], x[j]);
  }

  // Butterfly stages
  for (int len = 2; len <= N; len <<= 1) {
    float angle = -2.0f * static_cast<float>(M_PI) / len;
    std::complex<float> wlen(std::cos(angle), std::sin(angle));
    for (int i = 0; i < N; i += len) {
      std::complex<float> w(1.0f, 0.0f);
      for (int j = 0; j < len / 2; ++j) {
        auto u = x[i + j];
        auto v = x[i + j + len / 2] * w;
        x[i + j] = u + v;
        x[i + j + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
}

// ─── Librosa-compatible mel filterbank (Slaney scale) ───────────────────────
// Matches librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm=None)

static float HzToMelSlaney(float hz) {
  const float f_sp = 200.0f / 3.0f;
  float mel = hz / f_sp;
  const float min_log_hz = 1000.0f;
  const float min_log_mel = min_log_hz / f_sp;
  const float logstep = std::log(6.4f) / 27.0f;
  if (hz >= min_log_hz) {
    mel = min_log_mel + std::log(hz / min_log_hz) / logstep;
  }
  return mel;
}

static float MelToHzSlaney(float mel) {
  const float f_sp = 200.0f / 3.0f;
  float hz = mel * f_sp;
  const float min_log_hz = 1000.0f;
  const float min_log_mel = min_log_hz / f_sp;
  const float logstep = std::log(6.4f) / 27.0f;
  if (mel >= min_log_mel) {
    hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
  }
  return hz;
}

static std::vector<std::vector<float>> CreateMelFilterbank(int num_mels, int fft_size,
                                                            int sample_rate,
                                                            float fmin, float fmax) {
  int num_bins = fft_size / 2 + 1;

  float mel_min = HzToMelSlaney(fmin);
  float mel_max = HzToMelSlaney(fmax);

  int num_points = num_mels + 2;
  std::vector<float> mel_points(num_points);
  for (int i = 0; i < num_points; ++i) {
    mel_points[i] = mel_min + (mel_max - mel_min) * i / (num_points - 1);
  }

  std::vector<float> hz_points(num_points);
  for (int i = 0; i < num_points; ++i) {
    hz_points[i] = MelToHzSlaney(mel_points[i]);
  }

  std::vector<float> fft_freqs(num_bins);
  for (int i = 0; i < num_bins; ++i) {
    fft_freqs[i] = static_cast<float>(sample_rate) * i / fft_size;
  }

  std::vector<std::vector<float>> filters(num_mels, std::vector<float>(num_bins, 0.0f));
  for (int m = 0; m < num_mels; ++m) {
    float left = hz_points[m];
    float center = hz_points[m + 1];
    float right = hz_points[m + 2];
    for (int k = 0; k < num_bins; ++k) {
      float freq = fft_freqs[k];
      if (freq >= left && freq <= center && center > left) {
        filters[m][k] = (freq - left) / (center - left);
      } else if (freq >= center && freq <= right && right > center) {
        filters[m][k] = (right - freq) / (right - center);
      }
    }
  }
  return filters;
}

// ─── ComputeLogMel ──────────────────────────────────────────────────────────

std::vector<float> ComputeLogMel(const float* audio, size_t num_samples,
                                  const MelConfig& cfg, int& out_num_frames) {
  const int n_fft = cfg.fft_size;
  const int hop = cfg.hop_length;
  const int win_len = cfg.win_length;
  const int num_mels = cfg.num_mels;
  const int num_bins = n_fft / 2 + 1;
  const float log_guard = std::pow(2.0f, -24.0f);  // NeMo default: 2^-24

  // 1. Preemphasis: x[0] unchanged, x[n] = x[n] - preemph * x[n-1]
  std::vector<float> preemph_audio(num_samples);
  if (num_samples > 0) {
    preemph_audio[0] = audio[0];
    for (size_t i = 1; i < num_samples; ++i) {
      preemph_audio[i] = audio[i] - cfg.preemph * audio[i - 1];
    }
  }

  // 2. Center-pad: zero-pad n_fft/2 on each side (torch.stft center=True, pad_mode="constant")
  int pad = n_fft / 2;
  size_t padded_len = num_samples + 2 * pad;
  std::vector<float> padded(padded_len, 0.0f);
  std::memcpy(padded.data() + pad, preemph_audio.data(), num_samples * sizeof(float));

  // 3. Symmetric Hann window, center-padded to n_fft
  // torch.stft centers the window when win_length < n_fft
  auto hann = SymmetricHannWindow(win_len);
  std::vector<float> window(n_fft, 0.0f);
  int win_offset = (n_fft - win_len) / 2;
  for (int i = 0; i < win_len; ++i) {
    window[win_offset + i] = hann[i];
  }

  // 4. STFT frames
  int num_stft_frames = static_cast<int>((padded_len - n_fft) / hop) + 1;

  // Mel filterbank (computed once)
  auto mel_filters = CreateMelFilterbank(num_mels, n_fft, cfg.sample_rate, cfg.fmin, cfg.fmax);

  // Valid frames = num_samples // hop_length (NeMo truncation)
  int valid_frames = static_cast<int>(num_samples) / hop;
  if (valid_frames > num_stft_frames) valid_frames = num_stft_frames;
  out_num_frames = valid_frames;

  if (valid_frames <= 0) {
    out_num_frames = 0;
    return {};
  }

  std::vector<float> result(num_mels * valid_frames, 0.0f);
  std::vector<std::complex<float>> fft_buf(n_fft);
  std::vector<float> power_spectrum(num_bins);

  for (int frame = 0; frame < valid_frames; ++frame) {
    const float* frame_start = padded.data() + frame * hop;
    for (int i = 0; i < n_fft; ++i) {
      fft_buf[i] = std::complex<float>(frame_start[i] * window[i], 0.0f);
    }

    FFT(fft_buf);

    // Power spectrum: re^2 + im^2
    for (int k = 0; k < num_bins; ++k) {
      float re = fft_buf[k].real();
      float im = fft_buf[k].imag();
      power_spectrum[k] = re * re + im * im;
    }

    // Mel filterbank + log
    for (int m = 0; m < num_mels; ++m) {
      float mel_energy = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        mel_energy += mel_filters[m][k] * power_spectrum[k];
      }
      result[m * valid_frames + frame] = std::log(mel_energy + log_guard);
    }
  }

  return result;
}

}  // namespace parakeet_mel
