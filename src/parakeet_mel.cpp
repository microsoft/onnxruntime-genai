// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet mel spectrogram — pipeline matches NeMo's
// AudioToMelSpectrogramPreprocessor + librosa.filters.mel exactly.
//
// The heavy-lifting pieces (Slaney mel filterbank, pre-emphasis, real FFT /
// power spectrum) are delegated to onnxruntime-extensions
// (shared/api/nemo_mel_spectrogram.h, namespace nemo_mel). Only the bits that
// are parakeet/NeMo-specific are kept here:
//
//   * Symmetric Hann window (torch.hann_window(N, periodic=False)).
//     The batch helper in extensions (`NemoComputeLogMelBatch`) uses a
//     *periodic* Hann window, so we can't reuse it directly without changing
//     numerical output.
//   * Center-padded STFT framing loop with the window centered inside an
//     fft_size buffer (win_offset = (n_fft - win_length) / 2).
//   * Truncation to `num_samples / hop_length` valid frames.
//   * log_zero_guard = 2^-24 (NeMo default).

#include "parakeet_mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "nemo_mel_spectrogram.h"  // onnxruntime-extensions: shared/api

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace parakeet_mel {

// Symmetric Hann window: matches torch.hann_window(N, periodic=False).
// Kept locally because nemo_mel::NemoComputeLogMelBatch uses the *periodic*
// variant (sin(pi*n/N)^2), which produces different numerical output.
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

std::vector<float> ComputeLogMel(const float* audio, size_t num_samples,
                                  const ParakeetMelConfig& cfg, int& out_num_frames) {
  const int n_fft = cfg.fft_size;
  const int hop = cfg.hop_length;
  const int win_len = cfg.win_length;
  const int num_mels = cfg.num_mels;
  const int num_bins = n_fft / 2 + 1;
  const float log_guard = std::pow(2.0f, -24.0f);  // NeMo default: 2^-24

  // 1. Pre-emphasis: y[n] = x[n] - preemph * x[n-1]
  // Delegated to onnxruntime-extensions (nemo_mel::ApplyPreemphasis).
  std::vector<float> preemph_audio(num_samples);
  if (num_samples > 0) {
    nemo_mel::ApplyPreemphasis(audio, num_samples, cfg.preemph,
                               /*prev_sample=*/0.0f, preemph_audio.data());
  }

  // 2. Center-pad: zero-pad n_fft/2 on each side
  //    (matches torch.stft center=True, pad_mode="constant").
  const int pad = n_fft / 2;
  const size_t padded_len = num_samples + 2 * static_cast<size_t>(pad);
  std::vector<float> padded(padded_len, 0.0f);
  if (num_samples > 0) {
    std::memcpy(padded.data() + pad, preemph_audio.data(), num_samples * sizeof(float));
  }

  // 3. Symmetric Hann window, centered inside an fft_size buffer.
  //    torch.stft centers the window when win_length < n_fft:
  //      win_offset = (n_fft - win_length) / 2
  //    We advance the frame pointer by win_offset and pass the unpadded
  //    window of length win_len (matches the pattern used in extensions'
  //    own NemoComputeLogMelBatch).
  auto window = SymmetricHannWindow(win_len);
  const int win_offset = (n_fft - win_len) / 2;

  // 4. Frame layout & truncation.
  const int num_stft_frames = padded_len >= static_cast<size_t>(n_fft)
                                  ? static_cast<int>((padded_len - n_fft) / hop) + 1
                                  : 0;
  int valid_frames = static_cast<int>(num_samples) / hop;
  if (valid_frames > num_stft_frames) valid_frames = num_stft_frames;
  out_num_frames = valid_frames;

  if (valid_frames <= 0) {
    out_num_frames = 0;
    return {};
  }

  // 5. Mel filterbank (Slaney scale, librosa-compatible).
  //    Delegated to onnxruntime-extensions (nemo_mel::CreateMelFilterbank).
  //    Note: extensions builds the filterbank with fmin=0, fmax=sample_rate/2.
  //    Parakeet's default config has fmin=0 and fmax=sample_rate/2 (e.g.
  //    fmax=8000 at sr=16000), so this is identical.
  auto mel_filters =
      nemo_mel::CreateMelFilterbank(num_mels, n_fft, cfg.sample_rate);

  // 6. Per-frame STFT power spectrum + mel projection + log.
  //    The real-FFT power spectrum is computed by extensions
  //    (nemo_mel::ComputeSTFTFrame, backed by dlib::fftr).
  std::vector<float> result(static_cast<size_t>(num_mels) * valid_frames, 0.0f);
  std::vector<float> power_spectrum;
  power_spectrum.reserve(num_bins);

  for (int frame = 0; frame < valid_frames; ++frame) {
    const float* frame_start = padded.data() + frame * hop + win_offset;
    nemo_mel::ComputeSTFTFrame(frame_start, window.data(), win_len, n_fft,
                               power_spectrum);

    for (int m = 0; m < num_mels; ++m) {
      const auto& filter = mel_filters[m];
      float mel_energy = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        mel_energy += filter[k] * power_spectrum[k];
      }
      result[m * valid_frames + frame] = std::log(mel_energy + log_guard);
    }
  }

  return result;
}

}  // namespace parakeet_mel
