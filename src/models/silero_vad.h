// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// SileroVad: Voice Activity Detection using the Silero VAD ONNX model,
// integrated with GenAI's Model/State infrastructure.
// See https://github.com/snakers4/silero-vad for model details.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace Generators {

// Full definition of SileroVadState lives in silero_vad.cpp (it's an internal
// helper, no caller needs the layout).
struct SileroVadState;

/// Wraps the Silero VAD ONNX model for streaming voice activity detection.
///
/// Uses GenAI's Model infrastructure for session creation and State infrastructure
/// for inference (inheriting provider options, session options, run options, and
/// EP features like session_terminate from genai_config.json).
///
/// Silero VAD model I/O:
///   Inputs:  input  [1, window_size + context_size] float32
///            state  [2, 1, 128]                     float32
///            sr     [1]                              int64
///   Outputs: output [1, 1]                           float32 (speech probability)
///            stateN [2, 1, 128]                      float32 (updated hidden state)
struct SileroVad {
  SileroVad(Model& model);
  ~SileroVad();

  float ProcessWindow(const float* samples, size_t num_samples);
  bool ContainsSpeech(const float* samples, size_t num_samples);

  float GetThreshold() const { return threshold_; }
  void SetThreshold(float threshold) {
    threshold_ = std::max(0.0f, std::min(1.0f, threshold));
  }

  int GetWindowSize() const { return static_cast<int>(window_size_); }
  int GetContextSize() const { return static_cast<int>(context_size_); }
  int GetSampleRate() const { return static_cast<int>(sample_rate_); }

 private:
  void Initialize(int sample_rate, float threshold);
  void EnsureState();  // Lazily creates the State on first use

  Model& model_;
  std::unique_ptr<OrtSession> session_;
  std::unique_ptr<OrtSessionOptions> session_options_;

  // State is created lazily on first ProcessWindow call with internal GeneratorParams
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<SileroVadState> state_;

  float threshold_{};
  int64_t sample_rate_{};
  int64_t window_size_{};
  int64_t context_size_{};

  static constexpr size_t kStateSize = 2 * 1 * 128;  // Silero hidden state: [2, 1, 128]
  std::vector<float> state_data_;
  std::vector<float> context_;
  std::vector<float> input_data_;
};

std::unique_ptr<SileroVad> CreateSileroVad(Model& model);

}  // namespace Generators

