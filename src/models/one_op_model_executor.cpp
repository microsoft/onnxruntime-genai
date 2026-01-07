// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "one_op_model_executor.h"
#include "one_op_model_builder.h"
#include "../generators.h"
#include <functional>
#include <mutex>
#include <unordered_map>

namespace Generators {

// Global cache for 1-op model sessions
// Stored in OrtGlobals to ensure proper cleanup before OrtEnv destruction
struct OneOpSessionCache {
  std::unordered_map<uint64_t, std::unique_ptr<OrtSession>> sessions_;
  std::mutex mutex_;
};

static OneOpSessionCache& GetOneOpSessionCache() {
  static OneOpSessionCache cache;
  return cache;
}

// Generate a cache key from the model configuration and EP name
uint64_t OneOpModelExecutor::GenerateCacheKey(
    const OneOpModelConfig& config,
    const std::string& ep_name,
    const std::vector<const char*>& session_config_keys,
    const std::vector<const char*>& session_config_values) {
  // Simple hash combining op_type, input/output types, EP name, and session config
  std::hash<std::string> hasher;
  uint64_t key = hasher(config.op_type);

  // Hash EP name
  key ^= hasher(ep_name) + 0x9e3779b9 + (key << 6) + (key >> 2);

  // Hash input types
  for (const auto& input : config.inputs) {
    key ^= static_cast<uint64_t>(input.elem_type) + 0x9e3779b9 + (key << 6) + (key >> 2);
  }

  // Hash output types
  for (const auto& output : config.outputs) {
    key ^= static_cast<uint64_t>(output.elem_type) + 0x9e3779b9 + (key << 6) + (key >> 2);
  }

  // Hash attributes
  for (const auto& attr : config.attributes) {
    key ^= hasher(attr.name) + 0x9e3779b9 + (key << 6) + (key >> 2);

    switch (attr.type) {
      case AttributeType::INT:
        key ^= static_cast<uint64_t>(attr.int_value) + 0x9e3779b9 + (key << 6) + (key >> 2);
        break;
      case AttributeType::FLOAT: {
        uint32_t float_bits;
        std::memcpy(&float_bits, &attr.float_value, sizeof(float));
        key ^= static_cast<uint64_t>(float_bits) + 0x9e3779b9 + (key << 6) + (key >> 2);
        break;
      }
      case AttributeType::STRING:
        key ^= hasher(attr.string_value) + 0x9e3779b9 + (key << 6) + (key >> 2);
        break;
      case AttributeType::INTS:
        for (auto val : attr.ints_value) {
          key ^= static_cast<uint64_t>(val) + 0x9e3779b9 + (key << 6) + (key >> 2);
        }
        break;
      case AttributeType::FLOATS:
        for (auto val : attr.floats_value) {
          uint32_t float_bits;
          std::memcpy(&float_bits, &val, sizeof(float));
          key ^= static_cast<uint64_t>(float_bits) + 0x9e3779b9 + (key << 6) + (key >> 2);
        }
        break;
      case AttributeType::STRINGS:
        for (const auto& val : attr.strings_value) {
          key ^= hasher(val) + 0x9e3779b9 + (key << 6) + (key >> 2);
        }
        break;
    }
  }

  // Hash session config keys and values
  for (size_t i = 0; i < session_config_keys.size(); i++) {
    key ^= hasher(session_config_keys[i]) + 0x9e3779b9 + (key << 6) + (key >> 2);
    key ^= hasher(session_config_values[i]) + 0x9e3779b9 + (key << 6) + (key >> 2);
  }

  return key;
}

// Create a new session for the given model and EP
std::unique_ptr<OrtSession> OneOpModelExecutor::CreateSession(
    const std::vector<uint8_t>& model_bytes,
    const std::string& ep_name,
    const std::vector<const char*>& session_config_keys,
    const std::vector<const char*>& session_config_values) {
  auto session_options = OrtSessionOptions::Create();
  session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Apply session configuration entries
  for (size_t i = 0; i < session_config_keys.size(); i++) {
    session_options->AddConfigEntry(session_config_keys[i], session_config_values[i]);
  }

  // Append execution provider
  if (!ep_name.empty()) {
    session_options->AppendExecutionProvider(ep_name.c_str(), nullptr, nullptr, 0);
  }

  return OrtSession::Create(GetOrtEnv(), model_bytes.data(), model_bytes.size(), session_options.get());
}

// Get or create a cached session
OrtSession* OneOpModelExecutor::GetOrCreateSession(
    const OneOpModelConfig& config,
    const std::string& ep_name,
    const std::vector<const char*>& session_config_keys,
    const std::vector<const char*>& session_config_values) {
  auto& cache = GetOneOpSessionCache();
  uint64_t key = GenerateCacheKey(config, ep_name, session_config_keys, session_config_values);

  std::lock_guard<std::mutex> lock(cache.mutex_);

  auto it = cache.sessions_.find(key);
  if (it != cache.sessions_.end()) {
    return it->second.get();
  }

  // Create new session
  auto model_bytes = OneOpModelBuilder::Build(config);
  auto session = CreateSession(model_bytes, ep_name, session_config_keys, session_config_values);

  OrtSession* session_ptr = session.get();
  cache.sessions_[key] = std::move(session);

  return session_ptr;
}

// Execute a 1-op model
void OneOpModelExecutor::Execute(
    const OneOpModelConfig& model_config,
    const OneOpExecutionParams& exec_params) {
  // Get or create session
  OrtSession* session = GetOrCreateSession(
      model_config,
      exec_params.execution_provider_name,
      exec_params.session_config_keys,
      exec_params.session_config_values);

  // Create IOBinding for efficient execution
  auto io_binding = OrtIoBinding::Create(*session);

  // Bind inputs
  for (size_t i = 0; i < exec_params.inputs.size(); i++) {
    const auto& input_spec = exec_params.inputs[i];
    const auto& input_config = model_config.inputs[i];

    auto input_tensor = OrtValue::CreateTensor(
        *exec_params.memory_info,
        input_spec.data,
        input_spec.size_in_bytes,
        input_spec.shape,
        input_spec.elem_type);

    io_binding->BindInput(input_config.name.c_str(), *input_tensor);
  }

  // Bind outputs
  for (size_t i = 0; i < exec_params.outputs.size(); i++) {
    const auto& output_spec = exec_params.outputs[i];
    const auto& output_config = model_config.outputs[i];

    auto output_tensor = OrtValue::CreateTensor(
        *exec_params.memory_info,
        output_spec.data,
        output_spec.size_in_bytes,
        output_spec.shape,
        output_spec.elem_type);

    io_binding->BindOutput(output_config.name.c_str(), *output_tensor);
  }

  // Run inference
  session->Run(nullptr, *io_binding);
}

// Clear all cached sessions
void OneOpModelExecutor::ClearCache() {
  auto& cache = GetOneOpSessionCache();
  std::lock_guard<std::mutex> lock(cache.mutex_);
  cache.sessions_.clear();
}

// Helper function for Cast operation
void ExecuteCastOp(
    void* input_data,
    void* output_data,
    ONNXTensorElementDataType input_type,
    ONNXTensorElementDataType output_type,
    size_t element_count,
    const std::string& execution_provider_name,
    const OrtMemoryInfo* memory_info,
    const std::vector<const char*>& session_config_keys,
    const std::vector<const char*>& session_config_values) {
  // Build Cast model configuration with dynamic shape (-1) to support any element count
  OneOpModelConfig config("Cast");
  config.inputs.push_back(TensorConfig("input", input_type, {-1}));
  config.outputs.push_back(TensorConfig("output", output_type, {-1}));
  config.attributes.push_back(AttributeValue::Int("to", static_cast<int64_t>(output_type)));

  // Build execution parameters
  OneOpExecutionParams params(execution_provider_name, memory_info);
  params.inputs.push_back(OneOpTensorSpec(
      input_data,
      input_type,
      {static_cast<int64_t>(element_count)},
      element_count * Ort::SizeOf(input_type)));
  params.outputs.push_back(OneOpTensorSpec(
      output_data,
      output_type,
      {static_cast<int64_t>(element_count)},
      element_count * Ort::SizeOf(output_type)));

  // Apply session config entries if provided
  params.session_config_keys = session_config_keys;
  params.session_config_values = session_config_values;

  OneOpModelExecutor::Execute(config, params);
}

}  // namespace Generators
