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

inline std::shared_ptr<Model> LoadSyntheticPagedModel() {
  return CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged");
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

// Builds GeneratorParams for the dummy model with greedy, single-sequence search so a minted Request
// advances deterministically (SelectTop) when fed scripted logits.
inline std::shared_ptr<GeneratorParams> MakeGreedyParams(const Model& model) {
  auto params = CreateGeneratorParams(model);
  params->search.max_length = model.config_->model.context_length;
  params->search.num_beams = 1;
  params->search.batch_size = 1;
  params->search.do_sample = false;
  return params;
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

// Creates an Engine-owned Request from a frozen snapshot of `params`. Configuration changes must be
// made before this call.
inline std::shared_ptr<Request> CreateEngineRequest(
    const std::shared_ptr<Engine>& engine,
    const GeneratorParams& params) {
  return engine->CreateRequest(params);
}

inline std::shared_ptr<Request> CreateEngineRequest(
    const std::shared_ptr<Engine>& engine,
    const Model& model) {
  auto params = MakeGreedyParams(model);
  return CreateEngineRequest(engine, *params);
}

// Creates an Engine-owned Request and begins its first turn, leaving it Assigned and queued.
inline std::shared_ptr<Request> CreateRequestWithPrompt(
    const std::shared_ptr<Engine>& engine,
    const GeneratorParams& params,
    std::span<const int32_t> prompt_tokens) {
  auto request = CreateEngineRequest(engine, params);
  request->BeginTurn(prompt_tokens);
  return request;
}

inline std::shared_ptr<Request> CreateRequestWithPrompt(
    const std::shared_ptr<Engine>& engine,
    const Model& model,
    std::span<const int32_t> prompt_tokens) {
  auto params = MakeGreedyParams(model);
  auto request = CreateEngineRequest(engine, *params);
  request->BeginTurn(prompt_tokens);
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
