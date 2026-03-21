// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../generators.h"
#include "silero_vad.h"

namespace Generators {

// Shared initialization for both constructors
void SileroVad::Initialize(int sample_rate, float threshold) {
  sample_rate_ = sample_rate;
  threshold_ = threshold;

  if (sample_rate != 16000 && sample_rate != 8000) {
    throw std::runtime_error("SileroVad only supports sample rates 16000 and 8000. Got: " +
                             std::to_string(sample_rate));
  }

  const int factor = sample_rate / 8000;
  window_size_ = 256 * factor;
  context_size_ = 32 * factor;

  memory_info_ = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  state_.assign(kStateSize, 0.0f);
  context_.assign(static_cast<size_t>(context_size_), 0.0f);
}

// Constructor from Model — uses GenAI's session creation infrastructure
SileroVad::SileroVad(Model& model) {
  auto& vad_config = model.config_->model.vad;

  // Use the model's existing session options (inherits provider options, thread settings, etc.)
  // VAD is a lightweight model that doesn't need its own provider configuration.
  OrtSessionOptions* session_opts = model.session_options_.get();

  // If VAD-specific session options are provided in config, create dedicated options
  if (vad_config.session_options.has_value()) {
    session_options_ = OrtSessionOptions::Create();
    // Silero VAD is a ~2MB model with minimal compute (<1ms per window).
    // Single-threaded execution avoids thread pool overhead which would exceed
    // the inference time itself, and prevents contention with the main ASR model.
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetInterOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_opts = session_options_.get();
  }

  // Load session through Model::CreateSession (handles model data spans, path resolution)
  std::string filename = vad_config.filename;
  if (filename.empty()) filename = "silero_vad.onnx";
  session_ = model.CreateSession(GetOrtEnv(), filename, session_opts);

  // Create run options from config if specified
  if (vad_config.run_options.has_value()) {
    run_options_ = OrtRunOptions::Create();
    for (const auto& [key, value] : vad_config.run_options.value()) {
      run_options_->AddConfigEntry(key.c_str(), value.c_str());
    }
  }

  // Use VAD-specific sample_rate if configured, otherwise fall back to model's sample_rate
  int sr = vad_config.sample_rate > 0 ? vad_config.sample_rate : model.config_->model.sample_rate;
  Initialize(sr, vad_config.threshold);
}

// Constructor with explicit path — standalone/programmatic usage
SileroVad::SileroVad(const char* model_path, int sample_rate, float threshold) {
  session_options_ = OrtSessionOptions::Create();
  // Silero VAD is a ~2MB model with minimal compute (<1ms per window).
  // Single-threaded execution avoids thread pool overhead which would exceed
  // the inference time itself, and prevents contention with the main ASR model.
  session_options_->SetIntraOpNumThreads(1);
  session_options_->SetInterOpNumThreads(1);
  session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  auto& ort_env = GetOrtEnv();
#ifdef _WIN32
  std::wstring wide_path;
  int len = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, nullptr, 0);
  wide_path.resize(len - 1);
  MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wide_path.data(), len);
  session_ = OrtSession::Create(ort_env, wide_path.c_str(), session_options_.get());
#else
  session_ = OrtSession::Create(ort_env, model_path, session_options_.get());
#endif

  Initialize(sample_rate, threshold);
}

SileroVad::~SileroVad() = default;

void SileroVad::Reset() {
  std::fill(state_.begin(), state_.end(), 0.0f);
  std::fill(context_.begin(), context_.end(), 0.0f);
}

float SileroVad::ProcessWindow(const float* samples, size_t num_samples) {
  if (static_cast<int>(num_samples) != window_size_) {
    throw std::runtime_error("SileroVad::ProcessWindow expects " + std::to_string(window_size_) +
                             " samples, got " + std::to_string(num_samples));
  }

  // Build input: [context | samples] -> [1, context_size + window_size]
  const int effective_size = context_size_ + window_size_;
  std::vector<float> input_data(static_cast<size_t>(effective_size));
  std::memcpy(input_data.data(), context_.data(), context_size_ * sizeof(float));
  std::memcpy(input_data.data() + context_size_, samples, window_size_ * sizeof(float));

  // Create input tensors using GenAI's ORT wrappers
  int64_t input_shape[] = {1, effective_size};
  auto input_tensor = OrtValue::CreateTensor<float>(
      *memory_info_, std::span<float>(input_data), std::span<const int64_t>(input_shape, 2));

  int64_t state_shape[] = {2, 1, 128};
  auto state_tensor = OrtValue::CreateTensor<float>(
      *memory_info_, std::span<float>(state_), std::span<const int64_t>(state_shape, 3));

  sr_value_ = static_cast<int64_t>(sample_rate_);
  int64_t sr_shape[] = {1};
  auto sr_tensor = OrtValue::CreateTensor<int64_t>(
      *memory_info_, std::span<int64_t>(&sr_value_, 1), std::span<const int64_t>(sr_shape, 1));

  const char* input_names[] = {"input", "state", "sr"};
  const char* output_names[] = {"output", "stateN"};
  OrtValue* inputs[] = {input_tensor.get(), state_tensor.get(), sr_tensor.get()};

  // Run inference through the session with proper run options
  auto ort_outputs = session_->Run(
      run_options_ ? run_options_.get() : nullptr,
      input_names, inputs, 3,
      output_names, 2);

  // Extract speech probability
  float speech_prob = ort_outputs[0]->GetTensorMutableData<float>()[0];

  // Update hidden state from output
  const float* new_state = ort_outputs[1]->GetTensorMutableData<float>();
  std::memcpy(state_.data(), new_state, kStateSize * sizeof(float));

  // Update context with the last context_size_ samples from this window
  std::memcpy(context_.data(), samples + window_size_ - context_size_,
              context_size_ * sizeof(float));

  return speech_prob;
}

bool SileroVad::ContainsSpeech(const float* samples, size_t num_samples) {
  const size_t window = static_cast<size_t>(window_size_);

  // Process audio in non-overlapping windows of window_size_ samples (e.g. 512 for 16kHz).
  // Returns true as soon as any window exceeds the speech probability threshold,
  // allowing early exit without processing the entire chunk.
  for (size_t offset = 0; offset + window <= num_samples; offset += window) {
    float prob = ProcessWindow(samples + offset, window);
    if (prob >= threshold_) {
      return true;
    }
  }

  // Handle leftover samples that don't fill a complete window.
  // Pad with silence (zeros) to reach window_size_ and check for speech.
  // This ensures no audio at the end of a chunk is silently ignored.
  size_t processed = (num_samples / window) * window;
  if (processed < num_samples) {
    std::vector<float> padded(window, 0.0f);
    std::memcpy(padded.data(), samples + processed, (num_samples - processed) * sizeof(float));
    float prob = ProcessWindow(padded.data(), window);
    if (prob >= threshold_) {
      return true;
    }
  }

  return false;
}

std::unique_ptr<SileroVad> CreateSileroVad(Model& model) {
  return std::make_unique<SileroVad>(model);
}

std::unique_ptr<SileroVad> CreateSileroVad(const char* model_path, int sample_rate, float threshold) {
  return std::make_unique<SileroVad>(model_path, sample_rate, threshold);
}

}  // namespace Generators
