// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// SileroVad: Voice Activity Detection using the Silero VAD ONNX model,
// integrated with GenAI's session and run-options infrastructure.
// See https://github.com/snakers4/silero-vad for model details.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace Generators {

/// Wraps the Silero VAD ONNX model for streaming voice activity detection.
///
/// Uses GenAI's Model infrastructure for session creation (inheriting provider
/// options and session options from genai_config.json) and OrtRunOptions.
///
/// Silero VAD model I/O:
///   Inputs:  input  [1, window_size + context_size] float32  (PCM audio with context prepended)
///            state  [2, 1, 128]                     float32  (hidden state)
///            sr     [1]                              int64    (sample rate)
///   Outputs: output [1, 1]                           float32  (speech probability)
///            stateN [2, 1, 128]                      float32  (updated hidden state)
struct SileroVad : LeakChecked<SileroVad> {
  /// Construct from a Model reference (uses config for session/run options).
  /// The VAD model file is resolved relative to the model's config directory.
  SileroVad(Model& model);

  /// Construct with an explicit model path and options (programmatic override).
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
  void Initialize(int sample_rate, float threshold);
  void RegisterInputsOutputs();
  void Run();

  std::unique_ptr<OrtSession> session_;
  std::unique_ptr<OrtSessionOptions> session_options_;
  std::unique_ptr<OrtRunOptions> run_options_;
  Ort::Allocator* allocator_{nullptr};  // Points to model's device allocator (or CPU for standalone)

  int sample_rate_;
  float threshold_;
  int window_size_;
  int context_size_;

  static constexpr size_t kStateSize = 2 * 1 * 128;
  std::vector<float> state_;
  std::vector<float> context_;
  int64_t sr_value_{};

  // Registered I/O following the State pattern for GenAI infra consistency.
  std::vector<const char*> input_names_;
  std::vector<OrtValue*> inputs_;
  std::vector<const char*> output_names_;
  std::vector<OrtValue*> outputs_;

  // Owned input tensors (updated each ProcessWindow call)
  std::unique_ptr<OrtValue> input_tensor_;
  std::unique_ptr<OrtValue> state_tensor_;
  std::unique_ptr<OrtValue> sr_tensor_;

  // Scratch buffer for building [context | samples] input
  std::vector<float> input_data_;
};

std::unique_ptr<SileroVad> CreateSileroVad(Model& model);
std::unique_ptr<SileroVad> CreateSileroVad(const char* model_path, int sample_rate, float threshold);

}  // namespace Generators
