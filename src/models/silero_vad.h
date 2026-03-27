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
    if (threshold < 0.0f) threshold = 0.0f;
    if (threshold > 1.0f) threshold = 1.0f;
    threshold_ = threshold;
  }

  int GetWindowSize() const { return window_size_; }
  int GetContextSize() const { return context_size_; }
  int GetSampleRate() const { return sample_rate_; }

 private:
  void Initialize(int sample_rate, float threshold);
  void EnsureState();  // Lazily creates the State on first use

  Model& model_;
  std::unique_ptr<OrtSession> session_;
  std::unique_ptr<OrtSessionOptions> session_options_;

  // State is created lazily on first ProcessWindow call with internal GeneratorParams
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<SileroVadState> state_;

  float threshold_;
  int sample_rate_;
  int window_size_;
  int context_size_;

  static constexpr size_t kStateSize = 2 * 1 * 128;
  std::vector<float> state_data_;
  std::vector<float> context_;
  int64_t sr_value_{};
  std::vector<float> input_data_;
};

/// Internal State for SileroVad — uses State::Run for proper EP/run-options support.
struct SileroVadState : State {
  SileroVadState(const Model& model, const GeneratorParams& params, OrtSession& session,
                 float* input_data, int input_size,
                 float* state_data, int64_t* sr_value);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  void SetRunOptionsFromConfig(const Config::RunOptions& run_options);

 private:
  OrtSession& vad_session_;
  std::unique_ptr<OrtValue> input_tensor_;
  std::unique_ptr<OrtValue> state_tensor_;
  std::unique_ptr<OrtValue> sr_tensor_;
};

std::unique_ptr<SileroVad> CreateSileroVad(Model& model);

}  // namespace Generators
