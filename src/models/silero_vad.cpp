// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../generators.h"
#include "silero_vad.h"

namespace Generators {

SileroVad::SileroVad(const char* model_path, int sample_rate, float threshold)
    : sample_rate_{sample_rate},
      threshold_{threshold} {
  if (sample_rate != 16000 && sample_rate != 8000) {
    throw std::runtime_error("SileroVad only supports sample rates 16000 and 8000. Got: " +
                             std::to_string(sample_rate));
  }

  window_size_ = (sample_rate == 16000) ? 512 : 256;
  context_size_ = (sample_rate == 16000) ? 64 : 32;

  // Create lightweight session options for the small VAD model
  session_options_ = OrtSessionOptions::Create();
  session_options_->SetIntraOpNumThreads(1);
  session_options_->SetInterOpNumThreads(1);
  session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Load the ONNX model
  auto& ort_env = GetOrtEnv();
#ifdef _WIN32
  // Convert to wide string on Windows
  std::wstring wide_path;
  int len = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, nullptr, 0);
  wide_path.resize(len - 1);
  MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wide_path.data(), len);
  session_ = OrtSession::Create(ort_env, wide_path.c_str(), session_options_.get());
#else
  session_ = OrtSession::Create(ort_env, model_path, session_options_.get());
#endif

  // Create CPU memory info (kept for lifetime of this object)
  memory_info_ = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  // Initialize state and context to zeros
  state_.assign(kStateSize, 0.0f);
  context_.assign(static_cast<size_t>(context_size_), 0.0f);
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

  // Create input tensors using the project's ORT wrappers
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

  // Input/output names
  const char* input_names[] = {"input", "state", "sr"};
  const char* output_names[] = {"output", "stateN"};

  // Set up input pointers
  OrtValue* inputs[] = {input_tensor.get(), state_tensor.get(), sr_tensor.get()};

  // Run inference (ORT allocates outputs)
  auto ort_outputs = session_->Run(
      nullptr,  // default run options
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

  for (size_t offset = 0; offset + window <= num_samples; offset += window) {
    float prob = ProcessWindow(samples + offset, window);
    if (prob >= threshold_) {
      return true;
    }
  }

  // Handle remainder: if there are leftover samples, pad with zeros and check
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

std::unique_ptr<SileroVad> CreateSileroVad(const char* model_path, int sample_rate, float threshold) {
  return std::make_unique<SileroVad>(model_path, sample_rate, threshold);
}

}  // namespace Generators
