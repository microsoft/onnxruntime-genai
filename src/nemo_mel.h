// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// NeMo-compatible log-mel spectrogram for Parakeet TDT models.
// Matches NeMo's AudioToMelSpectrogramPreprocessor + librosa.filters.mel exactly.
// Standalone C++ — no dependency on kaldi-native-fbank or onnxruntime-extensions.
#pragma once

#include <cstddef>
#include <vector>

namespace parakeet_mel {

struct MelConfig {
  int num_mels = 128;
  int fft_size = 512;       // n_fft
  int hop_length = 160;
  int win_length = 400;
  int sample_rate = 16000;
  float preemph = 0.97f;
  float fmin = 0.0f;
  float fmax = 8000.0f;
};

/// Compute NeMo-compatible log-mel spectrogram.
///
/// Pipeline (matches parakeet_onnx_streaming_continue.py):
///   1. Preemphasis: x[0] unchanged, x[n] = x[n] - 0.97*x[n-1]
///   2. STFT: center=True (zero-pad n_fft/2 both sides), symmetric Hann window
///   3. Power spectrum: |STFT|^2
///   4. Mel filterbank: librosa-compatible (Slaney scale, 128 bins)
///   5. Log: log(mel + 2^-24)
///   6. Truncate to seq_len // hop_length frames
///
/// Output: row-major [num_mels, num_frames].
std::vector<float> ComputeLogMel(const float* audio, size_t num_samples,
                                  const MelConfig& cfg, int& out_num_frames);

}  // namespace parakeet_mel
