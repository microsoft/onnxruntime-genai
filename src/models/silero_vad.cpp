// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../generators.h"
#include "silero_vad.h"

namespace Generators {

SileroVadState::SileroVadState(const Model& model, const GeneratorParams& params, OrtSession& session,
                               float* input_data, int64_t input_size,
                               float* state_data, int64_t* sample_rate)
    : State{params, model},
      vad_session_{session} {
  auto& allocator = model.p_device_inputs_->GetAllocator();
  const auto& memory_info = allocator.GetInfo();

  // Register inputs — tensors wrap external buffers (owned by SileroVad, not us)
  int64_t input_shape[] = {1, input_size};
  input_tensor_ = OrtValue::CreateTensor<float>(
      memory_info, std::span<float>(input_data, input_size), std::span<const int64_t>(input_shape, 2));
  input_names_.push_back("input");
  inputs_.push_back(input_tensor_.get());

  int64_t state_shape[] = {2, 1, 128};
  state_tensor_ = OrtValue::CreateTensor<float>(
      memory_info, std::span<float>(state_data, 256), std::span<const int64_t>(state_shape, 3));
  input_names_.push_back("state");
  inputs_.push_back(state_tensor_.get());

  int64_t sr_shape[] = {1};
  sr_tensor_ = OrtValue::CreateTensor<int64_t>(
      memory_info, std::span<int64_t>(sample_rate, 1), std::span<const int64_t>(sr_shape, 1));
  input_names_.push_back("sr");
  inputs_.push_back(sr_tensor_.get());

  // Register outputs (ORT allocates)
  output_names_.push_back("output");
  outputs_.push_back(nullptr);
  output_names_.push_back("stateN");
  outputs_.push_back(nullptr);

  // Apply run options from config if specified
  if (model.config_->model.vad.run_options.has_value()) {
    State::SetRunOptions(model.config_->model.vad.run_options.value());
  }
}

DeviceSpan<float> SileroVadState::Run(int /*total_length*/, DeviceSpan<int32_t>& /*next_tokens*/,
                                      DeviceSpan<int32_t> /*next_indices*/) {
  // Use State::Run which handles run options, session terminate, EP options, graph capture
  State::Run(vad_session_);
  return {};
}

void SileroVad::Initialize(int sample_rate, float threshold) {
  sample_rate_ = static_cast<int64_t>(sample_rate);
  threshold_ = threshold;

  if (sample_rate != 16000 && sample_rate != 8000) {
    throw std::runtime_error("SileroVad only supports a sample rate of 8,000 Hz or 16,000 Hz. Got: " +
                             std::to_string(sample_rate));
  }

  // Silero VAD expects fixed window/context sizes that scale with sample rate.
  // At 8kHz: window=256, context=32. At 16kHz: window=512, context=64.
  const int64_t factor = sample_rate / 8000;
  window_size_ = 256 * factor;
  context_size_ = 32 * factor;

  state_data_.assign(kStateSize, 0.0f);
  context_.assign(static_cast<size_t>(context_size_), 0.0f);

  const int64_t effective_size = context_size_ + window_size_;
  input_data_.resize(static_cast<size_t>(effective_size), 0.0f);
}

SileroVad::SileroVad(Model& model)
    : model_{model} {
  auto& vad_config = model.config_->model.vad;

  // Create session options via CreateSessionOptionsFromConfig (public on Model).
  // Falls back to decoder session options if VAD-specific ones aren't provided.
  session_options_ = OrtSessionOptions::Create();
  model.CreateSessionOptionsFromConfig(
      vad_config.session_options.has_value()
          ? vad_config.session_options.value()
          : model.config_->model.decoder.session_options,
      *session_options_, false, true);

  // Load session through Model::CreateSession
  std::string filename = vad_config.filename;
  if (filename.empty()) {
    throw std::runtime_error("VAD filename must be specified in genai_config.json");
  }
  session_ = model.CreateSession(GetOrtEnv(), filename, session_options_.get());

  Initialize(model.config_->model.sample_rate, vad_config.threshold);
}

SileroVad::~SileroVad() = default;

void SileroVad::EnsureState() {
  if (!state_) {
    // Create GeneratorParams internally — VAD doesn't need search/generation params
    params_ = CreateGeneratorParams(model_);
    const int64_t effective_size = context_size_ + window_size_;
    state_ = std::make_unique<SileroVadState>(
        model_, *params_, *session_,
        input_data_.data(), effective_size,
        state_data_.data(), &sample_rate_);
  }
}

float SileroVad::ProcessWindow(const float* samples, size_t num_samples) {
  if (static_cast<int64_t>(num_samples) != window_size_) {
    throw std::runtime_error("SileroVad::ProcessWindow expects " + std::to_string(window_size_) +
                             " samples, got " + std::to_string(num_samples));
  }

  // Build input in-place
  std::memcpy(input_data_.data(), context_.data(), context_size_ * sizeof(float));
  std::memcpy(input_data_.data() + context_size_, samples, window_size_ * sizeof(float));

  // Ensure State is created (deferred until first use)
  EnsureState();

  // Run through State::Run — gets EP infrastructure (terminate, run options, etc.)
  DeviceSpan<int32_t> dummy_tokens;
  state_->Run(0, dummy_tokens);

  // Extract speech probability
  auto* output = state_->GetOutput("output");
  float speech_prob = output->GetTensorMutableData<float>()[0];

  // Update hidden state
  auto* new_state = state_->GetOutput("stateN");
  const float* new_state_data = new_state->GetTensorMutableData<float>();
  std::memcpy(state_data_.data(), new_state_data, kStateSize * sizeof(float));

  // Update context
  std::memcpy(context_.data(), samples + window_size_ - context_size_,
              context_size_ * sizeof(float));

  return speech_prob;
}

bool SileroVad::ContainsSpeech(const float* samples, size_t num_samples) {
  const size_t window = static_cast<size_t>(window_size_);

  // Process audio in non-overlapping windows.
  // Returns true as soon as any window exceeds the speech probability threshold.
  for (size_t offset = 0; offset + window <= num_samples; offset += window) {
    float prob = ProcessWindow(samples + offset, window);
    if (prob >= threshold_) {
      return true;
    }
  }

  // Handle leftover samples — pad with silence to reach window_size_.
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

}  // namespace Generators
