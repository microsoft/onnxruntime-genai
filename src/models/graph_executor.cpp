// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "graph_executor.h"
#include "graph_builder.h"
#include "../generators.h"
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cstring>

namespace Generators {

namespace {

// Generate a cache key from the model configuration and EP name
uint64_t GenerateCacheKey(
    const ModelConfig& config,
    const std::string& ep_name) {
  // Hash combining op_type, input/output types, and EP name
  // Attributes and shapes are excluded since they're passed as uniforms and can be changed dynamically
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

  return key;
}

// Create a new session for the given model and EP
std::unique_ptr<OrtSession> CreateSession(
    OrtModel* model,
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

  // Create session from OrtModel using Model Editor API
  const auto& model_editor_api = Ort::GetModelEditorApi();
  OrtSession* session_ptr = nullptr;
  Ort::ThrowOnError(model_editor_api.CreateSessionFromModel(&GetOrtEnv(), model, session_options.get(), &session_ptr));

  return std::unique_ptr<OrtSession>(session_ptr);
}

// Get or create a cached session
OrtSession* GetOrCreateSession(
    const ModelConfig& config,
    const std::string& ep_name,
    const std::vector<const char*>& session_config_keys,
    const std::vector<const char*>& session_config_values) {
  auto& cache = GetOrtGlobals()->graph_session_cache_;
  uint64_t key = GenerateCacheKey(config, ep_name);

  std::lock_guard<std::mutex> lock(cache.mutex_);

  auto it = cache.sessions_.find(key);
  if (it != cache.sessions_.end()) {
    return it->second.get();
  }

  // Build model using Model Editor API
  OrtModel* model = GraphBuilder::Build(config);

  // Create session from model
  auto session = CreateSession(model, ep_name, session_config_keys, session_config_values);

  // Release the model - session has its own copy
  Ort::api->ReleaseModel(model);

  OrtSession* session_ptr = session.get();
  cache.sessions_[key] = std::move(session);

  return session_ptr;
}

}  // anonymous namespace

void GraphExecutor::Execute(
    const ModelConfig& model_config,
    const ExecutionParams& exec_params) {
  // Validate input/output counts match
  if (exec_params.inputs.size() != model_config.inputs.size()) {
    throw std::invalid_argument("Number of inputs in exec_params doesn't match model_config");
  }
  if (exec_params.outputs.size() != model_config.outputs.size()) {
    throw std::invalid_argument("Number of outputs in exec_params doesn't match model_config");
  }

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
  ModelConfig config("Cast");
  config.inputs.push_back(TensorConfig("input", input_type, {-1}));
  config.outputs.push_back(TensorConfig("output", output_type, {-1}));
  config.attributes.push_back(AttributeValue::Int("to", static_cast<int64_t>(output_type)));

  // Build execution parameters
  ExecutionParams params(execution_provider_name, memory_info);
  params.inputs.push_back(TensorSpec(
      input_data,
      input_type,
      {static_cast<int64_t>(element_count)},
      element_count * Ort::SizeOf(input_type)));
  params.outputs.push_back(TensorSpec(
      output_data,
      output_type,
      {static_cast<int64_t>(element_count)},
      element_count * Ort::SizeOf(output_type)));

  // Apply session config entries if provided
  params.session_config_keys = session_config_keys;
  params.session_config_values = session_config_values;

  GraphExecutor::Execute(config, params);
}

}  // namespace Generators
