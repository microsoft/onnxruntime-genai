// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "generators.h"
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
// recording executor doubles, so the ONNX graph is never executed; the fixed present outputs are
// Identity pass-throughs and only the declared session I/O has to validate.
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

// Mints a Request carrying `prompt_tokens` but does not assign it, leaving it Unassigned so it can be
// handed to Engine::AddRequest (which assigns it and enqueues it on the engine's scheduler).
inline std::shared_ptr<Request> MintRequest(const Model& model, std::span<const int32_t> prompt_tokens) {
  auto request = std::make_shared<Request>(MakeGreedyParams(model));
  request->AddTokens(prompt_tokens);
  return request;
}

// Mints a Request carrying `prompt_tokens` and assigns it to `engine`, leaving it in the Assigned
// state (prompt finalized on device, ready to be scheduled). The engine is only the assignment
// target; the request may afterwards be driven through any scheduler under test.
inline std::shared_ptr<Request> MintAssignedRequest(const std::shared_ptr<Engine>& engine,
                                                    const Model& model,
                                                    std::span<const int32_t> prompt_tokens) {
  auto request = std::make_shared<Request>(MakeGreedyParams(model));
  request->AddTokens(prompt_tokens);
  request->Assign(engine);
  return request;
}

// The model's end-of-stream token, used to script a request to completion.
inline int32_t EosToken(const Model& model) {
  return model.config_->model.eos_token_id.empty() ? 0 : model.config_->model.eos_token_id.front();
}

}  // namespace test
}  // namespace Generators
