// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet mel spectrogram — matches NeMo's AudioToMelSpectrogramPreprocessor
// and librosa.filters.mel exactly (Slaney scale, symmetric Hann, center=True).
//
// This is a standalone implementation with no dependency on onnxruntime-extensions.
#pragma once

#include <cstddef>
#include <vector>

namespace parakeet_mel {

struct ParakeetMelConfig {
  int num_mels = 128;
  int fft_size = 512;       // n_fft
  int hop_length = 160;
  int win_length = 400;
  int sample_rate = 16000;
  float preemph = 0.97f;
  float fmin = 0.0f;
  float fmax = 8000.0f;
};

/// Compute log-mel spectrogram matching NeMo/Parakeet exactly.
///
/// Pipeline (matches parakeet_onnx_streaming_continue.py):
///   1. Preemphasis: x[0] unchanged, x[n] = x[n] - 0.97*x[n-1]
///   2. STFT: center=True (zero-pad n_fft/2 both sides), symmetric Hann window,
///            n_fft=512, hop=160, win=400
///   3. Power spectrum: |STFT|^2
///   4. Mel filterbank: librosa-compatible (Slaney scale, 128 bins, fmin=0, fmax=8000)
///   5. Log: log(mel + 2^-24)  (NeMo default log_zero_guard)
///   6. Truncate to seq_len // hop_length frames
///
/// Output layout: row-major [num_mels, num_frames].
/// out_num_frames is set to the number of time frames produced.
std::vector<float> ComputeLogMel(const float* audio, size_t num_samples,
                                  const ParakeetMelConfig& cfg, int& out_num_frames);

}  // namespace parakeet_mel
