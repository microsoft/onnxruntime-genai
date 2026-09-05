// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>

#include "generator/generators.h"
#include "search.h"
#include "engine/engine.h"
#include "engine/request.h"
#include "test_utils.h"

namespace Generators {
namespace test {

// Loads the tiny checked-in decoder model used by the engine component tests. The model only has to
// load as a decoder-only Generators::Model on CPU: the tests drive the scheduler and engine with the
// recording doubles in engine_test_doubles.h, so the ONNX graph is never executed. Minting a Request
// still needs a real model because Request builds a Search from the model's GeneratorParams and
// allocates its prompt tokens on the model's device.
inline std::shared_ptr<Model> LoadDummyDecoderModel() {
  return CreateModel(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
}

// Prefill chunking is model/Engine configuration rather than per-Request policy, so a test that
// needs a chunked prefill loads the model with the chunk size already set.
inline std::shared_ptr<Model> LoadDummyDecoderModelWithChunking(size_t chunk_size) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  config->search.chunk_size = chunk_size;
  return CreateModel(GetOrtEnv(), std::move(config));
}

inline std::shared_ptr<Model> LoadSyntheticPagedModel() {
  return CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged");
}

inline std::shared_ptr<Model> LoadSyntheticPagedMtpModel() {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged");
  config->model.mtp.filename = "decoder.onnx";
  config->model.mtp.num_key_value_heads = 1;
  config->model.mtp.head_size = 1;
  config->model.mtp.inputs.hidden_states = "past_key_values.1.key";
  config->model.mtp.outputs.hidden_states = "hidden_states";
  // Engine::CreateDependencies sets this on the target decoder when a head is configured. Tests
  // that drive ModelExecutor directly have to mirror it to get the hidden-states output bound.
  config->engine.hidden_states_output_required = true;
  return CreateModel(GetOrtEnv(), std::move(config));
}

#if USE_CUDA
inline std::shared_ptr<Model> LoadSyntheticPagedCudaModel() {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", {}, {});
  return CreateModel(GetOrtEnv(), std::move(config));
}
#endif

// Loads the tiny checked-in hybrid decoder used by the fixed-state-pool tests. Its config declares
// two fixed decoder state groups (convolution layers [0, 3] and recurrent layers [2, 5]) whose
// bindings resolve to real session inputs/outputs, so FixedStatePool can validate the manifest and
// derive per-request geometry. Like the other engine test models, the graph never executes: the
// pool allocates and stages its own tensors.
inline std::shared_ptr<Model> LoadSyntheticHybridModel() {
  return CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-hybrid");
}

// Loads the tiny checked-in composite decoder used by the Engine composite-transaction tests. Its
// config declares one paged_kv group (layers [1, 4]) alongside two fixed decoder state groups
// (convolution [0, 3] and recurrent [2, 5]) plus engine.dynamic_batching, so CacheManager::Create
// builds a real PagedKeyValueCache and a real FixedStatePool. The composite tests drive it with the
// recording executor doubles, while the Python integration test executes the graph. Most fixed
// outputs are Identity pass-throughs; convolution layer 0 increments its state so committed
// fixed-state propagation affects observable logits.
inline std::shared_ptr<Model> LoadSyntheticCompositeModel() {
  return CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
}

inline std::shared_ptr<Model> LoadSyntheticCompositeModelWithChunking(size_t chunk_size) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
  config->search.chunk_size = chunk_size;
  return CreateModel(GetOrtEnv(), std::move(config));
}

inline std::shared_ptr<Model> LoadSyntheticCompositeMtpModel() {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
  config->model.mtp.filename = "unused-mtp-head.onnx";
  config->engine.hidden_states_output_required = true;
  return CreateModel(GetOrtEnv(), std::move(config));
}

inline void PrepareRequestStep(const std::shared_ptr<Model>& model,
                               RequestStepPlan entry) {
  if (entry.unprocessed_token_count == 0) {
    entry.unprocessed_token_count =
        static_cast<size_t>(entry.request->CurrentSequenceLength() -
                            entry.request->ProcessedSequenceLength());
  }
  StepPlan plan;
  plan.requests.push_back(std::move(entry));
  static_cast<void>(ScheduledRequests{plan, model, nullptr, nullptr});
}

// Swaps in an alternative scoring device for the lifetime of the guard. Requests derive their
// search from the model, so this is how a test injects a failing or recording Search without any
// production seam. Every Request minted while the guard is live shares the substitute device.
class ScopedScoringDevice {
 public:
  ScopedScoringDevice(Model& model, DeviceInterface& device)
      : model_{model}, original_{model.p_device_scoring_} {
    model_.p_device_scoring_ = &device;
  }
  ScopedScoringDevice(const ScopedScoringDevice&) = delete;
  ScopedScoringDevice& operator=(const ScopedScoringDevice&) = delete;
  ~ScopedScoringDevice() { model_.p_device_scoring_ = original_; }

 private:
  Model& model_;
  DeviceInterface* original_;
};

// Creates an Engine-owned Request. The Engine derives its search from the model, so the only
// resident-session knob is the session token limit.
inline std::shared_ptr<Request> CreateEngineRequest(
    const std::shared_ptr<Engine>& engine) {
  return engine->CreateRequest();
}

inline std::shared_ptr<Request> CreateEngineRequest(
    const std::shared_ptr<Engine>& engine,
    size_t max_session_tokens) {
  RequestOptions options;
  options.max_session_tokens = max_session_tokens;
  return engine->CreateRequest(options);
}

// Creates an Engine-owned Request and begins its first turn, leaving it Assigned and queued.
inline std::shared_ptr<Request> CreateRequestWithPrompt(
    const std::shared_ptr<Engine>& engine,
    std::span<const int32_t> prompt_tokens) {
  auto request = CreateEngineRequest(engine);
  request->BeginTurn(prompt_tokens);
  return request;
}

inline std::shared_ptr<Request> CreateRequestWithPrompt(
    const std::shared_ptr<Engine>& engine,
    std::span<const int32_t> prompt_tokens,
    const TurnOptions& turn_options) {
  auto request = CreateEngineRequest(engine);
  TurnOptions options = turn_options;
  options.request = request;
  request->BeginTurn(prompt_tokens, options);
  return request;
}

// The model's end-of-stream token, used to script a request to completion.
inline int32_t EosToken(const Model& model) {
  return model.config_->model.eos_token_id.empty() ? 0 : model.config_->model.eos_token_id.front();
}

// Test-only capacity-one adapter over the canonical bulk Engine operation.
inline EngineEvent RunOne(Engine& engine) {
  std::array<EngineEvent, 1> storage;
  const size_t event_count = engine.Run(storage);
  return event_count == 0 ? EngineEvent{} : std::move(storage.front());
}

}  // namespace test
}  // namespace Generators
