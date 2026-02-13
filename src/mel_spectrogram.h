// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Standalone log-mel spectrogram extraction (Slaney scale, matching librosa/NeMo).
// No ONNX Runtime or other framework dependencies — pure C++ with standard library only.
// Designed to be portable to other repos.

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace mel {

// ─── Configuration ──────────────────────────────────────────────────────────

struct MelConfig {
  int num_mels{128};
  int fft_size{512};
  int hop_length{160};   // 10ms @ 16kHz
  int win_length{400};   // 25ms @ 16kHz
  int sample_rate{16000};
  float preemph{0.97f};
  float log_eps{5.96046448e-08f};  // NeMo default log_zero_guard value
};

// ─── Mel scale conversions (Slaney) ────────────────────────────────────────

float HzToMel(float hz);
float MelToHz(float mel);

// ─── Filterbank creation ────────────────────────────────────────────────────

/// Build a triangular mel filterbank with Slaney normalization (matches librosa).
/// Returns shape [num_mels][num_bins] where num_bins = fft_size/2 + 1.
std::vector<std::vector<float>> CreateMelFilterbank(int num_mels, int fft_size, int sample_rate);

// ─── Window functions ───────────────────────────────────────────────────────

/// Symmetric Hann window of length win_length (periodic=False).
/// Matches torch.hann_window(win_length, periodic=False).
std::vector<float> HannWindowSymmetric(int win_length);

/// Periodic Hann window of length win_length centered in an fft_size frame.
/// Matches torch.stft behavior: window placed at offset (fft_size - win_length) / 2.
/// Returns a vector of fft_size elements (zero-padded outside the window).
std::vector<float> HannWindowPeriodic(int fft_size, int win_length);

// ─── Single-frame DFT ──────────────────────────────────────────────────────

/// Compute |DFT|^2 (power spectrum) for a single windowed frame.
/// frame: pointer to fft_size samples (or win_length samples with window applied).
/// window: pointer to window coefficients (same length as frame_len).
/// frame_len: number of samples to read from frame and window.
/// fft_size: DFT size (output has fft_size/2 + 1 bins).
/// magnitudes: output power spectrum (resized to fft_size/2 + 1).
void ComputeSTFTFrame(const float* frame, const float* window, int frame_len,
                      int fft_size, std::vector<float>& magnitudes);

// ─── Batch (offline) log-mel extraction ─────────────────────────────────────

/// Compute log-mel spectrogram for a complete audio buffer.
/// Applies pre-emphasis, center-pads both sides (fft_size/2 zeros), computes STFT
/// with a periodic Hann window, applies mel filterbank, and takes log(mel + eps).
///
/// Output layout: row-major [num_mels, num_frames].
/// out_num_frames is set to the number of time frames produced.
std::vector<float> ComputeLogMelBatch(const float* audio, size_t num_samples,
                                      const MelConfig& cfg, int& out_num_frames);

// ─── Streaming log-mel extraction ───────────────────────────────────────────

/// Stateful streaming mel extractor that maintains overlap and pre-emphasis state
/// across successive audio chunks.
///
/// Usage:
///   mel::StreamingMelExtractor extractor(cfg);
///   auto [mel, frames] = extractor.Process(chunk1, n1);
///   auto [mel2, frames2] = extractor.Process(chunk2, n2);
///   extractor.Reset();  // new utterance
///
class StreamingMelExtractor {
 public:
  explicit StreamingMelExtractor(const MelConfig& cfg = {});

  /// Process one chunk of raw PCM audio (mono, float32).
  /// Returns (mel_data, num_frames) where mel_data is row-major [num_mels, num_frames].
  std::pair<std::vector<float>, int> Process(const float* audio, size_t num_samples);

  /// Reset all streaming state for a new utterance.
  void Reset();

  const MelConfig& config() const { return cfg_; }

 private:
  MelConfig cfg_;
  std::vector<std::vector<float>> mel_filters_;
  std::vector<float> hann_window_;  // symmetric, length = win_length

  // Streaming state
  std::vector<float> audio_overlap_;   // last fft_size/2 pre-emphasized samples
  float preemph_last_sample_{0.0f};
};

}  // namespace mel
