// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// NemotronAudioProcessor — self-contained log-mel spectrogram feature extractor.
//
// Reads all DSP parameters (n_fft, hop_length, frame_length, n_mel, sample_rate,
// mel_scale, log_type, log_floor) from a JSON config file so nothing is hardcoded.
//
// Designed so the whole class can be swapped for an ORT Extensions-based
// implementation once SpeechLibLogMel is registered there.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../filesystem.h"

namespace Generators {

/// Result of a log-mel spectrogram computation.
struct LogMelResult {
  std::vector<float> data;  // Flat buffer, layout [1, n_mel, num_frames]
  int num_mels{};
  int num_frames{};
};

/// Pure audio-processing class — no model / ORT dependency.
/// Reads its parameters from an audio_processor_config.json.
class NemotronAudioProcessor {
 public:
  /// Construct from a config directory path.  Looks for
  /// `audio_processor_config.json` (or the supplied filename) inside that dir.
  explicit NemotronAudioProcessor(const fs::path& config_dir,
                                  const std::string& config_filename = "audio_processor_config.json");

  /// Compute log-mel spectrogram from raw mono float32 PCM at the configured
  /// sample rate.  Output layout is [1, n_mel, num_frames].
  LogMelResult ComputeLogMel(const float* audio, size_t num_samples) const;

  // ── Accessors for the loaded config ──────────────────────────────────────
  int n_fft()        const { return n_fft_; }
  int hop_length()   const { return hop_length_; }
  int frame_length() const { return frame_length_; }
  int n_mel()        const { return n_mel_; }
  int sample_rate()  const { return sample_rate_; }

 private:
  // ── DSP parameters (loaded from JSON) ────────────────────────────────────
  int n_fft_{512};
  int hop_length_{160};
  int frame_length_{400};   // window length
  int n_mel_{128};
  int sample_rate_{16000};
  float preemph_{0.97f};    // pre-emphasis coefficient (NeMo default)

  // ── Pre-computed tables ──────────────────────────────────────────────────
  std::vector<float> hann_window_;                      // [frame_length]
  std::vector<std::vector<float>> mel_filterbank_;      // [n_mel][n_fft/2+1]

  // ── Helpers ──────────────────────────────────────────────────────────────
  void LoadConfig(const fs::path& config_path);
  void BuildHannWindow();
  void BuildMelFilterbank();

  /// Single-frame power-spectrum via naive DFT.
  void ComputePowerSpectrum(const float* frame, std::vector<float>& out) const;
};

}  // namespace Generators
