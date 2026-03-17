// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// SileroVad: Voice Activity Detection using the Silero VAD ONNX model.
// See https://github.com/snakers4/silero-vad for model details.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace Generators {

/// Wraps the Silero VAD ONNX model for streaming voice activity detection.
///
/// The model is stateful: it maintains hidden state (`state_`) and a context
/// buffer (`context_`) that carry information across consecutive windows.
/// Call Reset() to start a new utterance.
///
/// Silero VAD model I/O:
///   Inputs:  input  [1, window_size + context_size] float32  (PCM audio with context prepended)
///            state  [2, 1, 128]                     float32  (hidden state)
///            sr     [1]                              int64    (sample rate)
///   Outputs: output [1, 1]                           float32  (speech probability)
///            stateN [2, 1, 128]                      float32  (updated hidden state)
struct SileroVad : LeakChecked<SileroVad> {
  /// Construct a SileroVad instance.
  /// @param model_path  Path to the silero_vad.onnx file.
  /// @param sample_rate 16000 or 8000.
  /// @param threshold   Speech probability threshold (default 0.5).
  SileroVad(const char* model_path, int sample_rate = 16000, float threshold = 0.5f);
  ~SileroVad();

  /// Process a single VAD window of audio.
  /// The number of samples must equal GetWindowSize() (512 for 16kHz, 256 for 8kHz).
  /// Returns the speech probability in [0.0, 1.0].
  float ProcessWindow(const float* samples, size_t num_samples);

  /// Process arbitrary-length audio.
  /// Internally splits into windows and runs VAD on each.
  /// Returns true if any window's speech probability exceeds the threshold.
  bool ContainsSpeech(const float* samples, size_t num_samples);

  /// Reset hidden state and context for a new utterance.
  void Reset();

  float GetThreshold() const { return threshold_; }
  void SetThreshold(float threshold) { threshold_ = threshold; }

  /// Window size in samples (512 for 16kHz, 256 for 8kHz).
  int GetWindowSize() const { return window_size_; }

  /// Context size in samples (64 for 16kHz, 32 for 8kHz).
  int GetContextSize() const { return context_size_; }

  int GetSampleRate() const { return sample_rate_; }

 private:
  std::unique_ptr<OrtSession> session_;
  std::unique_ptr<OrtSessionOptions> session_options_;
  std::unique_ptr<OrtMemoryInfo> memory_info_;

  int sample_rate_;
  float threshold_;
  int window_size_;   // 512 for 16kHz, 256 for 8kHz
  int context_size_;  // 64 for 16kHz, 32 for 8kHz

  // Persistent hidden state [2 * 1 * 128 = 256 floats]
  static constexpr size_t kStateSize = 2 * 1 * 128;
  std::vector<float> state_;

  // Context buffer: last context_size_ samples from the previous window
  std::vector<float> context_;

  // Scratch space for sr tensor (must outlive the OrtValue that wraps it)
  int64_t sr_value_{};
};

std::unique_ptr<SileroVad> CreateSileroVad(const char* model_path, int sample_rate, float threshold);

}  // namespace Generators
